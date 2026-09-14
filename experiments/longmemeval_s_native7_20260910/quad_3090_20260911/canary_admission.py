"""Bounded noncanonical OpenAI-SDK canary. Run under GNU timeout 900 seconds."""
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time
from urllib.request import ProxyHandler, build_opener
import uuid

import quad_config as cfg

PINS = {
    'quad_admission.py': '1a785ac42c264b75acdf6b8c996b5e6df4c45e7710fd3ad28d7e12437a723423',
    'quad_metered_proxy.py': '700b043d38268e64a9114c3d6bf1e7b82d88b5510ad24a1b5e4cf1eaa8b0ac90',
    'deployment.json': cfg.DEPLOYMENT_SHA,
}
FROZEN_METER_SHA = '8ef745e6df482e02df494422dcbf96eaf56cb67d278656d4f14699b1e429cb93'
COUNTS = {'sm0': 16, 'sm1': 4}
TOKENS = {'sm0': 2560, 'sm1': 256}
OOM = re.compile(rb'out[ -]of[ -]memory|OutOfMemoryError|CUDA[^\n]*memory allocation', re.I)


def identity():
    if os.environ.get('CONTAINER_ID') != cfg.INSTANCE:
        raise ValueError('Wrong instance')
    if any(cfg.sha(cfg.STATE / name) != expected for name, expected in PINS.items()):
        raise ValueError('Admission code/deployment differs')
    frozen = cfg.ROOT / 'fast_native2_20260911/dual_gpu_20260911/dual_metered_proxy.py'
    if cfg.sha(frozen) != FROZEN_METER_SHA:
        raise ValueError('Frozen meter changed')
    return {**cfg.fresh_identity(), 'deployment_sha256': cfg.DEPLOYMENT_SHA,
            'code_sha256': dict(PINS), 'frozen_meter_sha256': FROZEN_METER_SHA,
            'canary_script_sha256': cfg.sha(Path(__file__))}


def http(url):
    with build_opener(ProxyHandler({})).open(url, timeout=5) as response:
        return response.read().decode()


def child_env():
    return {key: value for key, value in os.environ.items()
            if not any(word in key.upper() for word in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}


def parse_metrics(body):
    result = {}
    for name in ('num_preemptions_total', 'num_requests_running', 'num_requests_waiting'):
        values = re.findall(r'^vllm:' + name + r'(?:\{[^\n]*\})? ([0-9.eE+\-]+)$', body, re.M)
        if not values or not all(math.isfinite(float(value)) and float(value) >= 0 for value in values):
            raise ValueError('Required vLLM metric unavailable: ' + name)
        result[name] = sum(map(float, values))
    finished = {}
    for labels, value in re.findall(r'^vllm:request_success_total\{([^\n]*)\} ([0-9.eE+\-]+)$', body, re.M):
        reason = re.search(r'(?:^|,)finished_reason="([a-z_]+)"(?:,|$)', labels)
        if not reason or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError('Invalid request completion metric')
        key = reason.group(1)
        finished[key] = finished.get(key, 0) + float(value)
    if set(finished) != {'stop', 'length', 'abort', 'error', 'repetition'}:
        raise ValueError('Actual vLLM completion counter schema differs')
    return {'preemptions': result['num_preemptions_total'], 'running': result['num_requests_running'],
            'waiting': result['num_requests_waiting'], 'finished': finished}


def observe(lane):
    binding = cfg.deployment()['lanes'][lane]
    counters = parse_metrics(http(binding['api_base'].removesuffix('/v1') + '/metrics'))
    command = ['nvidia-smi', '--id=' + binding['gpu_uuid'], '--query-gpu=memory.free', '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10, env=child_env())
    free = int(result.stdout.strip())
    if free <= 512:
        raise ValueError('GPU free memory is not above 512 MiB')
    return {**counters, 'free_mib': free, 'gpu_uuid': binding['gpu_uuid']}


def verify_engine_delta(before, after, count):
    delta = {key: after['finished'][key] - before['finished'][key] for key in before['finished']}
    if delta != {'stop': 0, 'length': count, 'abort': 0, 'error': 0, 'repetition': 0}:
        raise ValueError('Engine completions differ or contain hidden/aborted/failed requests')
    if after['running'] != 0 or after['waiting'] != 0 or after['preemptions'] != before['preemptions']:
        raise ValueError('Engine has residual work or preemptions')
    return delta


def engine_pid(lane):
    result = subprocess.run(['supervisorctl', 'pid', 'quad-' + lane], capture_output=True,
                            text=True, check=True, timeout=10, env=child_env())
    pid = int(result.stdout.strip())
    if pid <= 0:
        raise ValueError('Current inference engine is not running')
    return pid


def log_start(lane):
    path = cfg.STATE / ('quad-' + lane + '.log')
    stat = path.stat()
    with path.open('rb') as stream:
        stream.seek(max(0, stat.st_size - 512))
        prefix = stream.read(stat.st_size - stream.tell())
    return {'path': path, 'inode': stat.st_ino, 'device': stat.st_dev, 'offset': stat.st_size, 'prefix': prefix}


def log_result(start):
    stat = start['path'].stat()
    if (stat.st_ino != start['inode'] or stat.st_dev != start['device'] or stat.st_size < start['offset']):
        raise ValueError('Engine log rotated or truncated during canary')
    with start['path'].open('rb') as stream:
        stream.seek(start['offset'])
        raw = stream.read(20 * 1024**2 + 1)
    if len(raw) > 20 * 1024**2:
        raise ValueError('Engine log observation exceeded bound')
    prefix = start['prefix']
    count = sum(match.end() > len(prefix) for match in OOM.finditer(prefix + raw))
    if count:
        raise ValueError('New engine OOM log entry')
    return {'path': str(start['path']), 'offset': start['offset'], 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest(), 'new_oom_entries': 0}


def sdk_request(lane, sample):
    import httpx
    import openai
    payload = {'model': cfg.MODEL, 'stream': True, 'temperature': 0,
               'messages': [{'role': 'user', 'content': 'Generate a numbered list of ordinary words until the output limit.'}],
               'max_tokens': TOKENS[lane], 'extra_body': {'min_tokens': TOKENS[lane], 'ignore_eos': True,
                                                        'chat_template_kwargs': {'enable_thinking': False}}}
    headers = {'X-Meter-Method': 'runtime_probe', 'X-Meter-Run-Id': sample,
               'X-Meter-Sample-Id': sample, 'X-Meter-Question-Id': sample, 'X-Meter-Phase': 'admission_canary'}
    digest, size = hashlib.sha256(), 0
    with openai.OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:18083/v1', max_retries=0,
            http_client=httpx.Client(timeout=httpx.Timeout(600, connect=10), trust_env=False)) as client:
        with client.chat.completions.create(**payload, extra_headers=headers) as stream:
            for item in stream:
                for choice in item.choices:
                    raw = (choice.delta.content or '').encode()
                    digest.update(raw)
                    size += len(raw)
    if not size:
        raise ValueError('Canary stream had no generated content')
    return {'sample_id': sample, 'lane': lane, 'response_sha256': digest.hexdigest(), 'response_bytes': size,
            'request_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            'openai_version': openai.__version__}


def run_batch(lane, samples, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Canary execution bound reached')
    results = queue.Queue()

    def request(sample):
        try:
            results.put((True, sdk_request(lane, sample)))
        except Exception as error:
            results.put((False, {'sample_id': sample, 'error_class': type(error).__name__}))

    for sample in samples:
        threading.Thread(target=request, args=(sample,), daemon=True).start()
    completed, observations, next_observation = [], [], 0
    while len(completed) < len(samples):
        if time.monotonic() >= deadline:
            raise TimeoutError('Canary execution bound reached')
        if time.monotonic() >= next_observation:
            observations.append(observe(lane))
            next_observation = time.monotonic() + 5
        try:
            success, result = results.get(timeout=0.25)
        except queue.Empty:
            continue
        if not success:
            raise RuntimeError('Canary SDK failure: ' + result['error_class'])
        completed.append(result)
    return completed, observations


def validate_records(rows, expected, deployment):
    fresh = cfg.fresh_identity()
    selected = [row for row in rows if row.get('run_id') in expected]
    if len(selected) != 20 or {row.get('run_id') for row in selected} != set(expected):
        raise ValueError('Expected exactly 20 unique canary journal records')
    result = {}
    for lane, count in COUNTS.items():
        lane_rows = [row for row in selected if expected[row['run_id']] == lane]
        if len(lane_rows) != count:
            raise ValueError('Canary lane coverage differs')
        for row in lane_rows:
            if (row.get('method') != 'runtime_probe' or row.get('sample_id') != row['run_id']
                    or row.get('question_id') != row['run_id'] or row.get('canonical_benchmark') is not False
                    or row.get('success') is not True or row.get('http_status') != 200
                    or row.get('stream') is not True or row.get('error_type') is not None
                    or row.get('client_disconnected') is not False
                    or row.get('deployment_sha256') != cfg.DEPLOYMENT_SHA
                    or any(row.get(key) != value for key, value in fresh.items())
                    or row.get('stream_complete') is not True or row.get('usage_status') != 'reported'
                    or row.get('finish_reasons') != ['length']
                    or row.get('inference_lane') != lane or row.get('inference_instance_id') != cfg.INSTANCE
                    or row.get('inference_gpu_uuid') != deployment['lanes'][lane]['gpu_uuid']
                    or row.get('usage', {}).get('completion_tokens') != TOKENS[lane]
                    or row.get('admission_active_at_start') not in (1, 2)):
                raise ValueError('Canary success/usage/route proof differs')
            usage = row['usage']
            if (any(type(usage.get(key)) is not int or usage[key] < 0 for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
                    or usage['prompt_tokens'] + usage['completion_tokens'] != usage['total_tokens']):
                raise ValueError('Canary token usage is invalid')
            for key in ('upstream_started_at', 'upstream_completed_at', 'admission_wait_seconds', 'upstream_ttfb_seconds'):
                if type(row.get(key)) not in (float, int) or not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError('Canary timing metadata unavailable')
            if row['upstream_completed_at'] < row['upstream_started_at']:
                raise ValueError('Invalid upstream interval')
        sequence = [row.get('admission_sequence') for row in lane_rows]
        if any(type(value) is not int or value <= 0 for value in sequence) or len(set(sequence)) != count:
            raise ValueError('Admission sequence is missing or duplicated')
        ordered = sorted(lane_rows, key=lambda row: row['admission_sequence'])
        if any(ordered[index]['upstream_started_at'] > ordered[index + 2]['upstream_started_at'] for index in range(count - 2)):
            raise ValueError('FIFO admission order differs beyond the two concurrent slots')
        events = [(row['upstream_started_at'], 1) for row in lane_rows] + [(row['upstream_completed_at'], -1) for row in lane_rows]
        active = peak = 0
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        if peak > 2:
            raise ValueError('Canary exceeded two upstream requests')
        result[lane] = {'requests': count, 'peak_upstream': peak,
                        'max_admission_wait_seconds': max(row['admission_wait_seconds'] for row in lane_rows),
                        'max_upstream_ttfb_seconds': max(row['upstream_ttfb_seconds'] for row in lane_rows)}
    if result['sm0']['max_admission_wait_seconds'] <= 600:
        raise ValueError('Canary did not exercise admission wait over 600 seconds')
    return selected, result


def wait_for_records(journal, offset, expected, deadline, skip_fragment=False):
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('Meter did not record all current canary IDs')
        health = json.loads(http('http://127.0.0.1:18083/health'))
        if health.get('status') != 'ok':
            raise ValueError('Meter journal is not healthy')
        with journal.open('rb') as stream:
            stream.seek(offset)
            raw = stream.read(20 * 1024**2 + 1)
        if len(raw) > 20 * 1024**2:
            raise ValueError('Canary journal span exceeded bound')
        content = raw.split(b'\n', 1)[-1] if skip_fragment else raw
        rows = [json.loads(line) for line in content.splitlines(keepends=True) if line.endswith(b'\n')]
        observed = {row.get('run_id') for row in rows if row.get('run_id') in expected}
        if observed == set(expected) and time.monotonic() < deadline:
            return raw, rows
        if time.monotonic() >= deadline:
            raise TimeoutError('Meter did not record all current canary IDs')
        time.sleep(0.25)


def execute(state, receipt, deadline):
    deployment, initial_identity = cfg.deployment(), identity()
    state.update(initial_identity)
    starts = {lane: log_start(lane) for lane in COUNTS}
    pids = {lane: engine_pid(lane) for lane in COUNTS}
    before = {lane: observe(lane) for lane in COUNTS}
    if any(item['running'] or item['waiting'] for item in before.values()):
        raise ValueError('Engine has existing work before canary')
    state.update(engine_before=before, engine_pids=pids)
    journal = cfg.STATE / 'request_usage.jsonl'
    offset = journal.stat().st_size
    with journal.open('rb') as stream:
        stream.seek(max(0, offset - 1))
        skip_fragment = bool(offset and stream.read(1) != b'\n')
    expected, responses, samples = {}, [], {}
    cfg.atomic(receipt, state)
    for lane, count in COUNTS.items():
        state['phase'] = 'collecting_' + lane
        cfg.atomic(receipt, state)
        ids = ['quad_probe_' + lane + '_' + state['canary_id'] + '_' + str(index) for index in range(count)]
        expected.update(dict.fromkeys(ids, lane))
        result, samples[lane] = run_batch(lane, ids, deadline)
        responses.extend(result)
    raw, rows = wait_for_records(journal, offset, expected, deadline, skip_fragment)
    selected, summary = validate_records(rows, expected, deployment)
    after = {lane: observe(lane) for lane in COUNTS}
    state['engine_after'] = after
    logs = {lane: log_result(starts[lane]) for lane in COUNTS}
    for lane in COUNTS:
        summary[lane]['engine_completion_delta'] = verify_engine_delta(before[lane], after[lane], COUNTS[lane])
        if engine_pid(lane) != pids[lane] or after[lane]['preemptions'] != before[lane]['preemptions']:
            raise ValueError('Current engine changed or preempted during canary')
        if any(item['preemptions'] != before[lane]['preemptions'] for item in samples[lane]):
            raise ValueError('Observed engine preemption during canary')
        summary[lane].update(preemptions=0, new_oom_entries=0,
            minimum_free_mib=min(item['free_mib'] for item in [before[lane], after[lane], *samples[lane]]))
    if identity() != initial_identity:
        raise ValueError('Canary code or fresh identity changed')
    state.update(status='admission_canary_verified', summary=summary, responses=responses, engine_logs=logs,
                 journal_offset=offset, journal_span_sha256=hashlib.sha256(raw).hexdigest(), journal_records=selected,
                 completed_at=time.time())
    cfg.atomic(receipt, state)


def main():
    canary_id = 'admission_' + uuid.uuid4().hex
    receipt = cfg.STATE / ('canary_' + canary_id + '.json')
    state = {'status': 'running', 'canary_id': canary_id, 'instance_id': cfg.INSTANCE, 'started_at': time.time()}

    def terminated(_signal, _frame):
        raise TimeoutError('Canary received its external timeout signal')

    signal.signal(signal.SIGTERM, terminated)
    try:
        execute(state, receipt, time.monotonic() + 870)
    except Exception as error:
        state.update(status='admission_canary_failed', error_class=type(error).__name__,
                     failure_reason=str(error)[:300], completed_at=time.time())
        cfg.atomic(receipt, state)
    print(json.dumps({'status': state['status'], 'receipt': str(receipt), 'canary_id': canary_id}), flush=True)
    return 0 if state['status'] == 'admission_canary_verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())