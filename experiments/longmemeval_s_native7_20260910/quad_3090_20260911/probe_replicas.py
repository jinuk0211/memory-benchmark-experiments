"""Prove all four inference routes and native CUDA on both LightMem GPUs."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen
import uuid

import quad_config as cfg


def snapshot():
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name', '--format=csv,noheader'],
        capture_output=True, text=True, check=True, timeout=15)
    found = []
    for row in result.stdout.splitlines():
        pid, gpu, name = [part.strip() for part in row.split(',', 2)]
        try:
            command = (Path('/proc') / pid / 'cmdline').read_bytes().lower()
        except OSError:
            command = b''
        if 'vllm' in name.lower() or b'vllm' in command:
            found.append({'pid': int(pid), 'gpu_uuid': gpu})
    return found


def stream_probe(lane, binding):
    os.environ.setdefault('QUAD_LANE', 'sm0')
    import runtime_probe as native
    probe_id = 'quad_probe_' + lane + '_' + uuid.uuid4().hex[:12]
    path = cfg.STATE / ('probe_' + lane) / probe_id
    path.mkdir(parents=True, mode=0o700)
    started = time.time()
    before = snapshot()
    with urlopen(binding['api_base'] + '/models', timeout=20) as response:
        model = next(row for row in json.load(response)['data'] if row['id'] == cfg.MODEL)
    if model.get('max_model_len') != 65536:
        raise ValueError('Replica context does not equal 65536: ' + lane)
    headers = {'X-Meter-Run-Id': probe_id, 'X-Meter-Method': 'runtime_probe',
               'X-Meter-Phase': 'synthetic_stream', 'X-Meter-Sample-Id': probe_id,
               'X-Meter-Question-Id': probe_id}
    stream_api = 'http://127.0.0.1:18083/v1' if binding['method'] == 'simplemem' else binding['api_base']
    stream = native.streamed_chat(stream_api + '/chat/completions', headers)
    with native.post('http://127.0.0.1:18083/v1/chat/completions',
                     {'model': cfg.MODEL, 'messages': [{'role': 'user', 'content': 'Reply with one short word confirming readiness.'}],
                      'temperature': 0, 'max_tokens': 32}, headers) as response:
        routed = json.load(response)
    if not routed['choices'][0]['message'].get('content', '').strip():
        raise ValueError('Empty metered replica response')
    central_route = native.route_proof(probe_id, lane, binding['gpu_uuid'])
    after = snapshot()
    pids = {p['pid'] for p in before if p['gpu_uuid'] == binding['gpu_uuid']}
    live = pids.intersection(p['pid'] for p in after if p['gpu_uuid'] == binding['gpu_uuid'])
    if not live:
        raise ValueError('Expected replica GPU process did not survive: ' + lane)
    result = {'status': 'runtime_probe_verified', 'instance_id': cfg.INSTANCE,
              'lane': lane, 'gpu_uuid': binding['gpu_uuid'], 'api_base': binding['api_base'],
              'probe_id': probe_id, 'benchmark_population_member': False,
              'script_sha256': cfg.sha(Path(__file__)), 'started_at': started,
              'completed_at': time.time(), 'streamed_chat': stream,
              'qwen_models': {'model': cfg.MODEL, 'max_model_len': 65536},
              'resident_vllm_pids': sorted(live), 'central_route': central_route}
    cfg.atomic(path / 'result.json', result)
    return path / 'result.json'


def light_probe(lane, binding):
    environment = cfg.child_environment(binding, 'lightmem')
    environment['QUAD_LANE'] = lane
    start = time.time()
    log = cfg.STATE / ('probe_' + lane + '.log')
    with log.open('a') as stream:
        subprocess.run(
            ['timeout', '--signal=TERM', '--kill-after=15s', '600',
             str(cfg.ROOT / '.venv-lightmem/bin/python'), str(cfg.STATE / 'runtime_probe.py')],
            cwd=cfg.ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
            timeout=630, check=True)
    shared = cfg.STATE / ('probe_' + lane) / 'runtime_probe_verified.json'
    proof = cfg.read(shared)
    path = shared.parent / proof['probe_id'] / 'result.json'
    if (proof != cfg.read(path) or proof['status'] != 'runtime_probe_verified'
            or proof['started_at'] < start or proof['instance_id'] != cfg.INSTANCE
            or proof['lane'] != lane or proof['gpu_uuid'] != binding['gpu_uuid']
            or proof['script_sha256'] != cfg.sha(cfg.STATE / 'runtime_probe.py')):
        raise ValueError('Fresh native runtime proof differs: ' + lane)
    return path


def main():
    if os.environ.get('CONTAINER_ID') != cfg.INSTANCE:
        raise ValueError('Wrong instance')
    config = cfg.deployment()
    artifacts = cfg.read(cfg.STATE / 'fresh_artifacts_verified.json')
    if (artifacts['status'] != 'fresh_artifacts_verified'
            or artifacts['fresh_run_sha256'] != cfg.sha(cfg.STATE / 'FRESH_RUN.json')):
        raise ValueError('Fresh artifact verification required')
    streams = {lane: stream_probe(lane, binding) for lane, binding in config['lanes'].items()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(light_probe, lane, binding): lane
                   for lane, binding in config['lanes'].items() if binding['method'] == 'lightmem'}
        for future in concurrent.futures.as_completed(futures):
            streams[futures[future]] = future.result()
    result = {'status': 'runtime_verified', 'instance_id': cfg.INSTANCE,
              'deployment_sha256': cfg.sha(cfg.STATE / 'deployment.json'),
              'fresh_run_sha256': cfg.sha(cfg.STATE / 'FRESH_RUN.json'),
              'model_proof': artifacts['model_proof'], 'protocol_proof': artifacts['protocol_proof'],
              'artifact_receipt_sha256': cfg.sha(cfg.STATE / 'fresh_artifacts_verified.json'),
              'verified_at': time.time(), 'lanes': {}, 'replicas': {}}
    for lane, path in streams.items():
        binding = config['lanes'][lane]
        result['lanes'][lane] = {'receipt': str(path.relative_to(cfg.STATE)), 'sha256': cfg.sha(path)}
        result['replicas'][lane] = {'status': 'inference_verified', 'model': cfg.MODEL,
                                    'max_model_len': 65536, 'gpu_uuid': binding['gpu_uuid'],
                                    'api_base': binding['api_base']}
    cfg.atomic(cfg.STATE / 'runtime_verified.json', result)
    cfg.atomic(cfg.STATE / 'READY.json', result)
    cfg.require_ready()
    print(json.dumps({'status': result['status'], 'lanes': list(streams)}), flush=True)


if __name__ == '__main__':
    main()