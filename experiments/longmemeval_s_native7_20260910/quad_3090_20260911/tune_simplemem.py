"""Drain SimpleMem, compare two independent long requests, and resume safely."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from urllib.request import Request, urlopen

import quad_config as cfg
from pipeline import rpc

Q = Path(__file__).resolve().parent
RECEIPT = Q / 'simplemem_tuning.json'
PORTS = {'sm0': 18081, 'sm1': 18091}
TERMINAL = {'complete', 'rolled_back', 'needs_attention'}


def record(state, phase, **details):
    state.update(phase=phase, updated_at=time.time(), **details)
    cfg.atomic(RECEIPT, state)
    print(json.dumps({'phase': phase, 'time': state['updated_at']}), flush=True)


def http(port, path, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(f'http://127.0.0.1:{port}{path}', data=data,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode()
    return json.loads(body) if path != '/metrics' else body


def metrics(port):
    body = http(port, '/metrics')
    values = {}
    for key in ('num_requests_running', 'num_requests_waiting', 'num_preemptions_total'):
        rows = re.findall(r'^vllm:' + key + r'(?:\{[^\n]*\})? ([0-9.eE+\-]+)$', body, re.M)
        if not rows:
            raise ValueError('Missing vLLM metric: ' + key)
        values[key] = sum(map(float, rows))
    return values


def memory(lane):
    uuid = cfg.deployment()['lanes'][lane]['gpu_uuid']
    result = subprocess.run(['nvidia-smi', '--id=' + uuid,
                             '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=10)
    used, total = map(int, result.stdout.strip().split(','))
    return used, total


def monitor_inference(include_simplemem=False):
    """Keep the paused pipeline's existing bounded recovery during tuning."""
    names = ['quad-lm0', 'quad-lm1', 'quad-meter', 'quad-minilm']
    if include_simplemem:
        names += ['quad-sm0', 'quad-sm1']
    path = Q / 'service_restarts.json'
    for name in names:
        if rpc().getProcessInfo(name)['statename'] != 'EXITED':
            continue
        counts = cfg.read(path) if path.exists() else {}
        if counts.get(name, 0) >= 2:
            raise RuntimeError('Existing inference restart budget exhausted: ' + name)
        counts[name] = counts.get(name, 0) + 1
        cfg.atomic(path, counts)
        rpc().startProcess(name, False)


def healthy(lane, maximum=720, monitor=True):
    deadline = time.monotonic() + maximum
    while time.monotonic() < deadline:
        if monitor:
            monitor_inference()
        info = rpc().getProcessInfo('quad-' + lane)
        if info['statename'] in {'EXITED', 'FATAL', 'STOPPED'}:
            raise RuntimeError('Inference startup failed: ' + lane)
        try:
            models = http(PORTS[lane], '/v1/models')['data']
            if any(row['id'] == cfg.MODEL and row.get('max_model_len') == 65536 for row in models):
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(3)
    raise TimeoutError('Inference readiness: ' + lane)


def restart(lane, count, state):
    # This runs only after the coordinator has drained and its lock is held.
    record(state, 'restarting_' + lane, current_lane=lane, current_count=count)
    info = rpc().getProcessInfo('quad-' + lane)
    if info['statename'] in {'RUNNING', 'STARTING'}:
        rpc().stopProcess('quad-' + lane, True)
    deadline = time.monotonic() + 90
    while rpc().getProcessInfo('quad-' + lane)['statename'] == 'STOPPING':
        if time.monotonic() > deadline:
            raise TimeoutError('Inference did not stop: ' + lane)
        time.sleep(2)
    path = Q / ('seqs_' + str(PORTS[lane]) + '.txt')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(str(count) + '\n')
    temporary.replace(path)
    rpc().startProcess('quad-' + lane, False)
    healthy(lane, monitor=count != 1)


def request_pair(lane):
    port = PORTS[lane]
    prompts = []
    for index in (0, 1):
        # Different from the first token onward, preventing shared prefix-cache blocks.
        prefix = 'Alpha amber north river. ' if index == 0 else 'Zulu violet south mountain. '
        tokens = http(port, '/tokenize', {'model': cfg.MODEL, 'prompt': prefix * 9000}, 60)['tokens']
        if len(tokens) < 28000:
            raise ValueError('Synthetic prompt too short')
        prompts.append(tokens[:28000])
    before = metrics(port)
    started = time.monotonic()
    peak_running, peak_used, total = 0, 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(http, port, '/v1/completions', {
            'model': cfg.MODEL, 'prompt': prompt, 'max_tokens': 512,
            'min_tokens': 512, 'ignore_eos': True, 'temperature': 0, 'seed': 20260909,
        }, 420) for prompt in prompts]
        while not all(future.done() for future in futures):
            monitor_inference()
            sample = metrics(port)
            peak_running = max(peak_running, sample['num_requests_running'])
            used, total = memory(lane)
            peak_used = max(peak_used, used)
            if time.monotonic() - started > 450:
                raise TimeoutError('Bounded canary exceeded')
            time.sleep(1)
        replies = [future.result() for future in futures]
    elapsed = time.monotonic() - started
    after = metrics(port)
    usages = [reply['usage'] for reply in replies]
    if any(row['prompt_tokens'] != 28000 or row['completion_tokens'] != 512 for row in usages):
        raise ValueError('Canary token workload differs')
    return {'seconds': elapsed, 'tokens_per_second': 1024 / elapsed, 'usages': usages,
            'peak_running': peak_running, 'peak_vram_mib': peak_used, 'total_vram_mib': total,
            'preemptions': after['num_preemptions_total'] - before['num_preemptions_total']}


def acceptable(baseline, candidate):
    return (candidate['peak_running'] >= 2 and candidate['preemptions'] == 0
            and candidate['peak_vram_mib'] < candidate['total_vram_mib'] - 512
            and candidate['seconds'] <= baseline['seconds'] * 0.95)


def drain(state):
    info = rpc().getProcessInfo('quad-simplemem')
    if info['statename'] == 'RUNNING':
        # SIGTERM goes to coordinator only; its handler finishes current histories.
        record(state, 'draining_current_histories', coordinator_pid=info['pid'])
        os.kill(info['pid'], signal.SIGTERM)
    deadline = state.setdefault('drain_deadline', time.time() + 5700)
    cfg.atomic(RECEIPT, state)
    while rpc().getProcessInfo('quad-simplemem')['statename'] in {'RUNNING', 'STARTING'}:
        monitor_inference(include_simplemem=True)
        if time.time() > deadline:
            raise TimeoutError('Current histories did not drain; workers left intact')
        time.sleep(5)
    status = cfg.read(Q / 'simplemem_quad_status.json')
    if status.get('inflight'):
        raise RuntimeError('SimpleMem workers still present')


def resume(state, outcome):
    record(state, 'resuming_pipeline', result=outcome)
    if rpc().getProcessInfo('quad-pipeline')['statename'] not in {'RUNNING', 'STARTING'}:
        rpc().startProcess('quad-pipeline', False)
    deadline = time.monotonic() + 1500
    while time.monotonic() < deadline:
        phase = cfg.read(Q / 'pipeline_status.json').get('phase')
        info = rpc().getProcessInfo('quad-pipeline')
        if phase == 'experiments_running' and info['statename'] == 'RUNNING':
            queue = rpc().getProcessInfo('quad-simplemem')
            if queue['statename'] == 'RUNNING':
                record(state, outcome, resumed=True)
                return
        if info['statename'] in {'EXITED', 'FATAL', 'STOPPED'}:
            raise RuntimeError('Pipeline did not resume')
        time.sleep(5)
    raise TimeoutError('Pipeline resume timed out')


def main():
    if os.environ.get('CONTAINER_ID') != cfg.INSTANCE:
        raise ValueError('Wrong instance')
    with (Q / 'simplemem_tuning.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = cfg.read(RECEIPT) if RECEIPT.exists() else {}
        identity = {'instance_id': cfg.INSTANCE,
                    'fresh_run_sha256': cfg.sha(Q / 'FRESH_RUN.json'),
                    'launcher_sha256': cfg.sha(Q / 'serve_qwen.sh'),
                    'baseline_launcher_sha256': cfg.sha(Q / 'serve_qwen.before_tuning.sh')}
        if state and any(state.get(key) != value for key, value in identity.items()):
            raise ValueError('Persisted tuning identity differs')
        if state.get('phase') in TERMINAL:
            return
        # A drain-only restart has not changed model settings and can continue its canary.
        drain_only = (state.get('phase') == 'draining_current_histories'
                      and not any((Q / ('seqs_' + str(port) + '.txt')).exists() for port in PORTS.values()))
        interrupted = bool(state) and not drain_only
        state.setdefault('started_at', time.time())
        state.setdefault('instance_id', cfg.INSTANCE)
        state.setdefault('fresh_run_sha256', cfg.sha(Q / 'FRESH_RUN.json'))
        state.setdefault('launcher_sha256', cfg.sha(Q / 'serve_qwen.sh'))
        state.setdefault('baseline_launcher_sha256', cfg.sha(Q / 'serve_qwen.before_tuning.sh'))
        state.setdefault('lanes', cfg.deployment()['lanes'])
        state.setdefault('results', {})
        state.setdefault('attempts', 0)
        state['attempts'] += 1
        if state['attempts'] > 2:
            record(state, 'needs_attention', error='Tuner restart budget exhausted')
            return
        try:
            if not interrupted:
                cfg.require_ready()
                if cfg.read(Q / 'pipeline_status.json')['phase'] != 'experiments_running':
                    raise ValueError('Pipeline is not in steady experiment phase')
            record(state, 'pausing_monitor')
            if rpc().getProcessInfo('quad-pipeline')['statename'] == 'RUNNING':
                rpc().stopProcess('quad-pipeline', True)
            drain(state)
            lock_path = cfg.ROOT / 'fast_native2_20260911/simplemem_queue.lock'
            with lock_path.open('a') as queue_lock:
                fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    for lane in PORTS:
                        if interrupted:
                            restart(lane, 1, state)
                            state['results'].setdefault(lane, {})['selected'] = 1
                            cfg.atomic(RECEIPT, state)
                            continue
                        if any(metrics(PORTS[lane])[key] for key in
                               ('num_requests_running', 'num_requests_waiting')):
                            raise RuntimeError('Unexpected requests after drain')
                        restart(lane, 1, state)
                        record(state, 'baseline_' + lane)
                        baseline = request_pair(lane)
                        state['results'][lane] = {'baseline': baseline}
                        restart(lane, 2, state)
                        record(state, 'candidate_' + lane)
                        candidate = request_pair(lane)
                        selected = 2 if acceptable(baseline, candidate) else 1
                        state['results'][lane].update(candidate=candidate, selected=selected)
                        cfg.atomic(RECEIPT, state)
                        if selected == 1:
                            restart(lane, 1, state)
                except Exception as error:
                    record(state, 'rolling_back', error_type=type(error).__name__, error=str(error))
                    for lane in PORTS:
                        restart(lane, 1, state)
                        state['results'].setdefault(lane, {})['selected'] = 1
                    interrupted = True
            # Release coordinator lock before allowing it to restart.
            resume(state, 'rolled_back' if interrupted else 'complete')
        except Exception as error:
            phase = 'retry_recovery' if state['attempts'] < 2 else 'needs_attention'
            record(state, phase, error_type=type(error).__name__, error=str(error),
                   note='LightMem and provider instance remain running; no native worker killed')
            raise


if __name__ == '__main__':
    main()