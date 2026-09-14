"""Drain only LightMem, compare max_num_seqs 1/2, and retain native workloads."""
import concurrent.futures
from contextlib import contextmanager
import hashlib
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
RECEIPT = Q / 'lightmem_tuning.json'
PORTS = {'lm0': 18101, 'lm1': 18111}
TERMINAL = {'complete', 'rolled_back', 'needs_attention'}
PROMPT_TOKENS, OUTPUT_TOKENS, REQUESTS = 2000, 1024, 5

HOLDER_CODE = r"""
import json, os, pathlib, signal, sys, time
import torch
size, destination, parent, index, uuid = sys.argv[1:]
size, parent = int(size), int(parent)
if os.getppid() != parent:
    raise SystemExit(1)
# A killed tuner must not leave its synthetic CUDA allocation alive.
import ctypes
if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
    raise RuntimeError('Cannot bind holder lifetime to parent')
if os.getppid() != parent:
    raise SystemExit(1)
allocation = torch.empty(size, dtype=torch.uint8, device='cuda:0')
allocation.fill_(0)
torch.cuda.synchronize()
path = pathlib.Path(destination)
temporary = path.with_suffix('.tmp')
with temporary.open('w') as output:
    json.dump({'pid': os.getpid(), 'parent_pid': parent, 'allocated_bytes': allocation.numel(),
               'cuda_index': int(index), 'gpu_uuid': uuid}, output)
    output.flush()
    os.fsync(output.fileno())
temporary.replace(path)
deadline = time.monotonic() + 900
while os.getppid() == parent and time.monotonic() < deadline:
    time.sleep(1)
"""


def record(state, phase, **details):
    state.update(phase=phase, updated_at=time.time(), **details)
    cfg.atomic(RECEIPT, state)
    print(json.dumps({'phase': phase, 'time': state['updated_at']}), flush=True)


def http(port, path, payload=None, timeout=5):
    request = Request(f'http://127.0.0.1:{port}{path}',
                      data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode()
    return body if path == '/metrics' else json.loads(body)


def metrics(port):
    values = {}
    body = http(port, '/metrics')
    for key in ('num_requests_running', 'num_requests_waiting', 'num_preemptions_total'):
        rows = re.findall(r'^vllm:' + key + r'(?:\{[^\n]*\})? ([0-9.eE+\-]+)$', body, re.M)
        if not rows:
            raise ValueError('Missing vLLM metric: ' + key)
        values[key] = sum(map(float, rows))
    return values


def gpu_query(lane, fields):
    if lane not in PORTS:
        raise ValueError('Only LightMem GPUs may be tuned')
    uuid = cfg.deployment()['lanes'][lane]['gpu_uuid']
    result = subprocess.run(['nvidia-smi', '--id=' + uuid, '--query-gpu=' + fields,
                             '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, check=True, timeout=10)
    return [value.strip() for value in result.stdout.strip().split(',')]


def memory(lane):
    used, total = map(int, gpu_query(lane, 'memory.used,memory.total'))
    return used, total


def monitor_inference(exclude=()):
    """Use the existing recovery budget without resetting any service counter."""
    path = Q / 'service_restarts.json'
    for name in ('quad-sm0', 'quad-sm1', 'quad-meter', 'quad-minilm', 'quad-lm0', 'quad-lm1'):
        if name in exclude:
            continue
        status = rpc().getProcessInfo(name)['statename']
        if status in {'FATAL', 'STOPPED', 'UNKNOWN'}:
            raise RuntimeError('Unhealthy inference service: ' + name)
        if status != 'EXITED':
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
            monitor_inference(exclude=('quad-' + lane,))
        if rpc().getProcessInfo('quad-' + lane)['statename'] in {'EXITED', 'FATAL', 'STOPPED'}:
            raise RuntimeError('Inference startup failed: ' + lane)
        try:
            if any(row['id'] == cfg.MODEL and row.get('max_model_len') == 65536
                   for row in http(PORTS[lane], '/v1/models')['data']):
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(3)
    raise TimeoutError('Inference readiness: ' + lane)


def write_seqs(lane, count):
    if lane not in PORTS or count not in (1, 2):
        raise ValueError('Unexpected tuning target')
    path = Q / ('seqs_' + str(PORTS[lane]) + '.txt')
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as output:
        output.write(str(count) + '\n')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def restart(lane, count, state):
    if lane not in PORTS or count not in (1, 2):
        raise ValueError('Unexpected tuning target')
    record(state, 'restarting_' + lane, current_lane=lane, current_count=count)
    info = rpc().getProcessInfo('quad-' + lane)
    if info['statename'] in {'RUNNING', 'STARTING'}:
        rpc().stopProcess('quad-' + lane, True)
    deadline = time.monotonic() + 90
    while rpc().getProcessInfo('quad-' + lane)['statename'] == 'STOPPING':
        if time.monotonic() >= deadline:
            raise TimeoutError('Inference did not stop: ' + lane)
        time.sleep(2)
    write_seqs(lane, count)
    rpc().startProcess('quad-' + lane, False)
    # Restoration must not be blocked by a different GPU's exhausted retry budget.
    healthy(lane, monitor=count != 1)


def rollback(state):
    failures = []
    for lane in PORTS:
        try:
            write_seqs(lane, 1)
        except Exception as error:
            failures.append({'lane': lane, 'stage': 'write', 'type': type(error).__name__})
    for lane in PORTS:
        try:
            restart(lane, 1, state)
            state['results'].setdefault(lane, {})['selected'] = 1
        except Exception as error:
            failures.append({'lane': lane, 'stage': 'restart', 'type': type(error).__name__})
    cfg.atomic(RECEIPT, state)
    if failures:
        raise RuntimeError('LightMem rollback incomplete: ' + json.dumps(failures))


def make_prompts(lane):
    prompts = []
    for prefix in ('Amber orchard north. ', 'Birch ocean east. ', 'Cobalt valley south. ',
                   'Dahlia forest west. ', 'Eagle summit center. '):
        tokens = http(PORTS[lane], '/tokenize',
                      {'model': cfg.MODEL, 'prompt': prefix * 1500}, 60)['tokens']
        if (len(tokens) < PROMPT_TOKENS
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError('Synthetic tokenization differs')
        prompts.append(tokens[:PROMPT_TOKENS])
    if len({prompt[0] for prompt in prompts}) != REQUESTS:
        raise ValueError('Canary prompts must differ from their first token')
    return prompts


@contextmanager
def memory_holder(lane, before_used, reserve_mib):
    used_after, total = memory(lane)
    if type(reserve_mib) is not int or reserve_mib < 0:
        raise ValueError('Invalid fixed native helper reserve')
    index, = gpu_query(lane, 'index')
    if not index.isdigit() or int(index) not in range(4):
        raise ValueError('Unexpected UUID-resolved CUDA index')
    uuid = cfg.deployment()['lanes'][lane]['gpu_uuid']
    destination = Q / ('lightmem_holder_' + lane + '_' + str(time.time_ns()) + '.json')
    proof = {'requested_mib': reserve_mib, 'pre_drain_used_mib': before_used,
             'used_after_restart_mib': used_after, 'total_vram_mib': total,
             'cuda_index': int(index), 'gpu_uuid': uuid}
    if reserve_mib == 0:
        yield {**proof, 'allocated_bytes': 0, 'holder_required': False}
        return
    environment = {key: value for key, value in os.environ.items()
                   if not any(word in key.upper() for word in ('TOKEN', 'API_KEY', 'SECRET', 'PASSWORD'))}
    environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=index,
                       PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    command = [str(cfg.ROOT / '.venv-lightmem/bin/python'), '-u', '-c', HOLDER_CODE,
               str(reserve_mib * 1024**2), str(destination), str(os.getpid()), index, uuid]
    with destination.with_suffix('.log').open('a') as output:
        process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 120
            while not destination.exists():
                if process.poll() is not None:
                    raise RuntimeError('CUDA helper footprint holder failed')
                if time.monotonic() >= deadline:
                    raise TimeoutError('CUDA helper footprint holder readiness')
                monitor_inference()
                time.sleep(1)
            ready = cfg.read(destination)
            if (ready.get('pid') != process.pid or ready.get('parent_pid') != os.getpid()
                    or ready.get('allocated_bytes') != reserve_mib * 1024**2
                    or ready.get('cuda_index') != int(index) or ready.get('gpu_uuid') != uuid
                    or process.poll() is not None):
                raise ValueError('CUDA holder readiness proof differs')
            yield {**proof, **ready, 'holder_required': True}
            if process.poll() is not None:
                raise RuntimeError('CUDA holder exited during canary')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def request_batch(lane, prompts):
    if len(prompts) != REQUESTS or any(len(prompt) != PROMPT_TOKENS for prompt in prompts):
        raise ValueError('Canary prompt workload differs')
    port = PORTS[lane]
    before = metrics(port)
    started = time.monotonic()
    peak_running = 0
    peak_used, total = memory(lane)
    with concurrent.futures.ThreadPoolExecutor(max_workers=REQUESTS) as pool:
        futures = [pool.submit(http, port, '/v1/completions', {
            'model': cfg.MODEL, 'prompt': prompt, 'max_tokens': OUTPUT_TOKENS,
            'min_tokens': OUTPUT_TOKENS, 'ignore_eos': True, 'temperature': 0, 'seed': 20260909,
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
    used, total = memory(lane)
    peak_used = max(peak_used, used)
    usages = [reply['usage'] for reply in replies]
    for reply, usage in zip(replies, usages):
        choices = reply.get('choices', [])
        if (usage.get('prompt_tokens') != PROMPT_TOKENS
                or usage.get('completion_tokens') != OUTPUT_TOKENS
                or usage.get('total_tokens') != PROMPT_TOKENS + OUTPUT_TOKENS
                or len(choices) != 1 or not isinstance(choices[0].get('text'), str)
                or not choices[0]['text'].strip()):
            raise ValueError('Canary token workload or output differs')
    return {'seconds': elapsed, 'tokens_per_second': REQUESTS * OUTPUT_TOKENS / elapsed,
            'usages': usages, 'outputs_nonempty': True, 'peak_running': peak_running,
            'peak_vram_mib': peak_used, 'total_vram_mib': total,
            'preemptions': after['num_preemptions_total'] - before['num_preemptions_total']}


def acceptable(baseline, candidate):
    return (candidate['peak_running'] >= 2 and candidate['preemptions'] == 0
            and candidate['peak_vram_mib'] < candidate['total_vram_mib'] - 512
            and candidate['seconds'] <= baseline['seconds'] * 0.90
            and candidate['outputs_nonempty'] is True)


def drain(state):
    info = rpc().getProcessInfo('quad-lightmem')
    if info['statename'] == 'RUNNING':
        record(state, 'draining_current_histories', coordinator_pid=info['pid'])
        os.kill(info['pid'], signal.SIGTERM)
    deadline = state.setdefault('drain_deadline', time.time() + 5700)
    cfg.atomic(RECEIPT, state)
    while rpc().getProcessInfo('quad-lightmem')['statename'] in {'RUNNING', 'STARTING'}:
        monitor_inference()
        if time.time() > deadline:
            raise TimeoutError('Current histories did not drain; workers left intact')
        time.sleep(5)
    if cfg.read(Q / 'lightmem_quad_status.json').get('inflight'):
        raise RuntimeError('LightMem workers still present')


def resume(state, outcome):
    record(state, 'resuming_pipeline', result=outcome)
    if rpc().getProcessInfo('quad-pipeline')['statename'] not in {'RUNNING', 'STARTING'}:
        rpc().startProcess('quad-pipeline', False)
    deadline = time.monotonic() + 1500
    while time.monotonic() < deadline:
        phase = cfg.read(Q / 'pipeline_status.json').get('phase')
        info = rpc().getProcessInfo('quad-pipeline')
        if phase == 'experiments_running' and info['statename'] == 'RUNNING':
            if rpc().getProcessInfo('quad-lightmem')['statename'] == 'RUNNING':
                record(state, outcome, resumed=True)
                return
        if info['statename'] in {'EXITED', 'FATAL', 'STOPPED'}:
            raise RuntimeError('Pipeline did not resume')
        time.sleep(5)
    raise TimeoutError('Pipeline resume timed out')


def run_locked(state, interrupted):
    drain(state)
    import fcntl
    with (cfg.ROOT / 'fast_native2_20260911/lightmem_queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if interrupted:
            rollback(state)
            return 'rolled_back'
        try:
            for lane in PORTS:
                if any(metrics(PORTS[lane])[key] for key in
                       ('num_requests_running', 'num_requests_waiting')):
                    raise RuntimeError('Unexpected requests after drain')
                prompts = make_prompts(lane)
                state['results'][lane] = {'prompt_tokens': PROMPT_TOKENS, 'requests': REQUESTS,
                    'prompt_ids_sha256': hashlib.sha256(json.dumps(prompts).encode()).hexdigest()}
                for count, phase in ((1, 'baseline'), (2, 'candidate')):
                    restart(lane, count, state)
                    if count == 1:
                        used_after, _ = memory(lane)
                        state['results'][lane]['helper_reserve_mib'] = max(
                            0, state['pre_drain_vram'][lane]['used_mib'] - used_after)
                    record(state, phase + '_' + lane)
                    with memory_holder(lane, state['pre_drain_vram'][lane]['used_mib'],
                                       state['results'][lane]['helper_reserve_mib']) as footprint:
                        result = request_batch(lane, prompts)
                        result['helper_footprint'] = footprint
                    state['results'][lane][phase] = result
                    cfg.atomic(RECEIPT, state)
                result = state['results'][lane]
                selected = 2 if acceptable(result['baseline'], result['candidate']) else 1
                result['selected'] = selected
                cfg.atomic(RECEIPT, state)
                if selected == 1:
                    restart(lane, 1, state)
            return 'complete'
        except Exception as error:
            record(state, 'rolling_back', error_type=type(error).__name__, error=str(error))
            rollback(state)
            return 'rolled_back'


def main():
    if os.environ.get('CONTAINER_ID') != cfg.INSTANCE:
        raise ValueError('Wrong instance')
    import fcntl
    with (Q / 'lightmem_tuning.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = cfg.read(RECEIPT) if RECEIPT.exists() else {}
        identity = {'instance_id': cfg.INSTANCE, **cfg.fresh_identity(),
                    'launcher_sha256': cfg.sha(Q / 'serve_qwen.sh'),
                    'baseline_launcher_sha256': cfg.sha(Q / 'serve_qwen.before_tuning.sh')}
        if state and any(state.get(key) != value for key, value in identity.items()):
            raise ValueError('Persisted tuning identity differs')
        if state.get('phase') in TERMINAL:
            return
        interrupted = bool(state.get('mutations_started'))
        state.update(identity)
        state.setdefault('started_at', time.time())
        state.setdefault('results', {})
        state['attempts'] = state.get('attempts', 0) + 1
        if state['attempts'] > 2:
            record(state, 'needs_attention', error='Tuner restart budget exhausted')
            return
        try:
            if not interrupted:
                cfg.require_ready()
                if cfg.read(Q / 'pipeline_status.json')['phase'] != 'experiments_running':
                    raise ValueError('Pipeline is not in steady experiment phase')
                other = Q / 'simplemem_tuning.json'
                if other.exists() and cfg.read(other).get('phase') not in TERMINAL:
                    raise ValueError('SimpleMem tuning must finish before LightMem tuning')
                if rpc().getProcessInfo('quad-simplemem')['statename'] != 'RUNNING':
                    raise ValueError('SimpleMem queue must remain running')
                state['pre_drain_vram'] = {}
                for lane in PORTS:
                    used, total = memory(lane)
                    state['pre_drain_vram'][lane] = {'used_mib': used, 'total_mib': total, 'at': time.time()}
            record(state, 'pausing_monitor', mutations_started=True)
            if rpc().getProcessInfo('quad-pipeline')['statename'] in {'RUNNING', 'STARTING'}:
                rpc().stopProcess('quad-pipeline', True)
            outcome = run_locked(state, interrupted)
            # Release the coordinator lock before resuming pipeline dispatch.
            resume(state, outcome)
        except Exception as error:
            phase = 'retry_recovery' if state['attempts'] < 2 else 'needs_attention'
            record(state, phase, error_type=type(error).__name__, error=str(error),
                   note='SimpleMem queue and provider instance are never stopped by this tuner')
            raise


if __name__ == '__main__':
    main()
