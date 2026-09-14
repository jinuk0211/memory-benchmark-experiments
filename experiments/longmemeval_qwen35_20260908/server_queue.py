"""Supervisor-owned LongMemEval transition; independent of the user's PC."""
import argparse
import configparser
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import shlex
import shutil
import socket
import subprocess
import time

import httpx

from controlled_reader import DATA_SHA256, MODEL, MODEL_REVISION, reader_request
from materialize_inputs import write_once
from native_six import EMBEDDING_MODEL, EMBEDDING_REVISION
from prepare_inputs import METHODS, REFERENCE_SHA256, digest
from run_history import read_json, verify_files

ROOT = Path('/workspace/lme_baselines_20260908')
HARNESS = ROOT / 'experiments/longmemeval_qwen35_20260908'
TRANSFER = Path('/workspace/generalization_20260908/full_transfer')
QWEN = TRANSFER / 'runs/qwen35_lme500_fullrole'
DATA = TRANSFER.parent / 'data/longmemeval_s_cleaned.json'
SERVER_PY = TRANSFER.parent / 'modern/.venv/bin/python'
CLIENT_PY = ROOT / '.venv-client/bin/python'
RUN = ROOT / 'runs/five_full500'
TOKENIZER = ROOT / 'tokenizer' / MODEL_REVISION
CHAT_URL = 'http://127.0.0.1:18083/v1'
EMBED_URL = 'http://127.0.0.1:18084/v1'
SERVICES = ('lme-five-embedding', 'lme-five-chat', 'lme-five-embedding-meter', 'lme-five-chat-meter')


def status(phase: str, **fields) -> None:
    record = dict(phase=phase, time=datetime.now(timezone.utc).isoformat(), **fields)
    path = ROOT / 'SERVER_QUEUE_STATUS.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps(record), flush=True)


def validate_gate(gate: dict) -> None:
    expected = dict(status='qwen_generation_verified_gpu_idle', gemma_held=True,
                    dataset_sha256=DATA_SHA256, reference_protocol_sha256=REFERENCE_SHA256,
                    memory_files_verified=1500, reader_replays_verified=1500)
    if any(type(gate.get(k)) is not type(v) or gate[k] != v for k, v in expected.items()):
        raise ValueError('Unverified Qwen completion gate')
    if set(gate['arms']) != {'seed', 'r40_fused_four_turn', 's_parent_single_2000'}:
        raise ValueError('Missing Qwen comparison arm')
    if any(type(arm.get('questions')) is not int or arm['questions'] != 500 for arm in gate['arms'].values()):
        raise ValueError('Qwen arm is not complete500')


def wait_for_qwen(once: bool = False) -> int:
    import os
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    command = [str(SERVER_PY), '-B', str(HARNESS / 'verify_qwen_completion.py'),
               '--run-dir', str(QWEN), '--transfer-root', str(TRANSFER), '--dataset', str(DATA),
               '--out', 'pending']
    while True:
        fresh = RUN / f'qwen_completion_{time.time_ns()}.json'
        command[-1] = str(fresh)
        with (RUN / 'completion_gate.log').open('a', encoding='utf-8') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode == 0:
            validate_gate(read_json(fresh))
            # Keep every fresh proof; this file is only the current-proof pointer copy.
            temporary = RUN / 'qwen_completion.latest.tmp'
            temporary.write_bytes(fresh.read_bytes())
            temporary.replace(RUN / 'qwen_completion.json')
            return 0
        if result.returncode != 2:
            raise RuntimeError('Qwen completion validation failed; see completion_gate.log')
        status('waiting_qwen', baseline_services_started=False)
        if once:
            return 2
        time.sleep(300)


def service_state(name: str) -> str:
    result = subprocess.run(['supervisorctl', 'status', name], capture_output=True, text=True,
                            check=False, timeout=20)
    fields = result.stdout.split()
    if result.returncode not in (0, 3) or len(fields) < 2 or fields[0] != name:
        raise RuntimeError(f'Cannot observe {name}')
    return fields[1]


def wait_models(url: str, model: str, names: tuple[str, ...]) -> None:
    deadline = time.monotonic() + 1200
    with httpx.Client(timeout=5, trust_env=False) as client:
        while time.monotonic() < deadline:
            if any(service_state(name) not in ('STARTING', 'RUNNING') for name in names):
                raise RuntimeError(f'Service stopped before readiness: {names}')
            try:
                response = client.get(url + '/models')
                response.raise_for_status()
                ids = {item['id'] for item in response.json()['data']}
                if ids != {model}:
                    raise ValueError(f'Wrong served model: {ids}')
                return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
                time.sleep(3)
    raise TimeoutError(f'Service readiness timeout: {names}')


def validate_vectors(response: dict) -> None:
    rows = sorted(response['data'], key=lambda row: row['index'])
    if [row['index'] for row in rows] != [0, 1]:
        raise ValueError('Embedding coverage mismatch')
    for row in rows:
        vector = row['embedding']
        if len(vector) != 1024 or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
            raise ValueError('Embedding is not a finite1024-dimensional vector')
        if sum(x*x for x in vector) == 0:
            raise ValueError('Zero embedding vector')


def verify_installed_specs() -> None:
    spec = read_json(ROOT / 'server_services.json')
    for name in SERVICES:
        template = ROOT / 'server_supervisor' / (name + '.conf')
        installed = Path('/etc/supervisor/conf.d') / (name + '.conf')
        if installed.read_bytes() != template.read_bytes():
            raise ValueError(f'Installed Supervisor config changed: {name}')
        parsed = configparser.ConfigParser(interpolation=None)
        parsed.read_string(template.read_text())
        expected = f'/bin/bash -o pipefail -c "{shlex.join(spec[name])} 2>&1 | tee -a {ROOT}/logs/{name}.log"'
        if parsed['program:' + name]['command'] != expected:
            raise ValueError(f'Supervisor command differs from frozen specification: {name}')


def validate_live_model(config: dict, expected_path: str, max_length: int) -> None:
    if (config.get('model') != expected_path or config.get('dtype') != 'torch.bfloat16'
            or 'quantization' not in config or config['quantization'] is not None
            or config.get('max_model_len') != max_length):
        raise ValueError('Actual vLLM model configuration does not match the experiment')


def verify_services() -> dict:
    from openai import OpenAI
    from transformers import AutoTokenizer
    gate = read_json(RUN / 'qwen_completion.json')
    validate_gate(gate)
    snapshot = read_json(HARNESS / 'source_snapshot.json')
    verify_files(ROOT / 'MemoryData', snapshot['source_files_sha256'])
    verify_files(ROOT / 'HiGMem', snapshot['higmem_files_sha256'])
    frozen = read_json(ROOT / 'server_launch_freeze.json')
    verify_files(ROOT, frozen['files_sha256'])
    verify_installed_specs()
    manifest = read_json(ROOT / 'inputs_five/manifest.json')
    if tuple(manifest['required_methods']) != tuple(METHODS) or len(METHODS) != 5:
        raise ValueError('The user-approved method scope changed')
    input_receipt = read_json(ROOT / 'FIVE_INPUT_RECEIPT.json')
    if digest(manifest) != input_receipt['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    spec = read_json(ROOT / 'server_services.json')
    live_models = {}
    with httpx.Client(timeout=120, trust_env=False) as info_client:
        for name, port, limit in (('lme-five-chat', 18081, 65536), ('lme-five-embedding', 18082, 32768)):
            response = info_client.get(f'http://127.0.0.1:{port}/server_info?config_format=json')
            response.raise_for_status()
            actual = response.json()['vllm_config']['model_config']
            command = spec[name]
            validate_live_model(actual, command[command.index('--model') + 1], limit)
            live_models[name] = actual
    protocol = read_json(QWEN / 'protocol.json')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    headers = {'X-Meter-Run-Id': 'lme-five-full500', 'X-Meter-Method': 'infrastructure',
               'X-Meter-Phase': 'service_verification', 'X-Meter-Sample-Id': '_service_probe',
               'X-Meter-Question-Id': '_service_probe'}
    with OpenAI(base_url=EMBED_URL, api_key='EMPTY', timeout=600, max_retries=0,
                default_headers=headers) as client:
        embeddings = client.embeddings.create(model=EMBEDDING_MODEL, dimensions=1024,
                         input=['A service verification document.', 'A service verification query.'])
    validate_vectors(embeddings.model_dump())
    with OpenAI(base_url=CHAT_URL, api_key='EMPTY', timeout=600, max_retries=0,
                default_headers=headers) as client:
        request, audit = reader_request(['The service check code is BLUE.'],
            'What is the service check code?', '2026-09-08',
            lambda text: len(tokenizer.encode(text, add_special_tokens=False)), protocol)
        answer = client.chat.completions.create(**request,
                    extra_headers={'X-Meter-Phase': 'final_reader'})
        choice = answer.choices[0]
        if choice.finish_reason not in ('stop', 'length') or not choice.message.content or answer.usage is None:
            raise ValueError('Common reader service probe failed')
        tool = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=256,
            messages=[{'role':'user','content':'Call record_check with value service-ready. Do not answer normally.'}],
            tools=[{'type':'function','function':{'name':'record_check',
                'description':'Record the service check value.', 'parameters':{'type':'object',
                'properties':{'value':{'type':'string'}},'required':['value']}}}],
            tool_choice='auto', extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        calls = tool.choices[0].message.tool_calls or []
        if len(calls) != 1 or calls[0].function.name != 'record_check' or tool.usage is None:
            raise ValueError('Native tool-calling service probe failed')
        if json.loads(calls[0].function.arguments) != {'value':'service-ready'}:
            raise ValueError('Native tool-calling arguments failed')
    write_once(RUN / 'service_probe.json', dict(embedding_response=embeddings.model_dump(),
               reader_request=request, reader_audit=audit, reader_response=answer.model_dump(),
               tool_response=tool.model_dump(), live_model_configs=live_models))
    # Both the actual engine configuration and the frozen startup specification are checked.
    for name in ('lme-five-chat', 'lme-five-embedding'):
        command = spec[name]
        if command[command.index('--dtype') + 1] != 'bfloat16' or '--quantization' in command:
            raise ValueError('Unexpected dtype or quantization request')
        model_path = Path(command[command.index('--model') + 1])
        revision = MODEL_REVISION if name == 'lme-five-chat' else EMBEDDING_REVISION
        if model_path.name != revision or read_json(model_path / 'config.json').get('quantization_config'):
            raise ValueError('Model revision or quantization config mismatch')
    hashes = lambda folder: {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}
    receipt = dict(status='baseline_services_verified_after_qwen_complete',
        model=MODEL, model_revision=MODEL_REVISION,
        dtype=live_models['lme-five-chat']['dtype'].removeprefix('torch.'),
        embedding_model=EMBEDDING_MODEL, embedding_revision=EMBEDDING_REVISION,
        embedding_dimensions=1024, qwen_completed_questions_per_arm=500, qwen_completed_arms=3,
        gemma_held=True, run_id='lme-five-full500', llm_url=CHAT_URL, embedding_url=EMBED_URL,
        input_manifest_sha256=digest(manifest), source_files_sha256=snapshot['source_files_sha256'],
        higmem_files_sha256=snapshot['higmem_files_sha256'], tokenizer_files_sha256=hashes(TOKENIZER),
        harness_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HARNESS.glob('*.py'))},
        server_launch_freeze_sha256=digest(frozen), service_probe_sha256=digest(read_json(RUN/'service_probe.json')))
    write_once(RUN / 'runtime_receipt.json', receipt)
    return receipt


def start_baselines() -> None:
    frozen = read_json(ROOT / 'server_launch_freeze.json')
    verify_files(ROOT, frozen['files_sha256'])
    verify_installed_specs()
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise RuntimeError('Less than8GiB free; no baseline service launched')
    for port in (18081, 18082, 18083, 18084):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    with ExitStack() as cleanup:
        for name in SERVICES:
            if service_state(name) != 'STOPPED':
                raise RuntimeError(f'{name} is not stopped; refusing a duplicate start')
            status('starting_baseline_service', service=name)
            cleanup.callback(subprocess.run, ['supervisorctl','stop',name], check=False, timeout=150)
            subprocess.run(['supervisorctl','start',name],check=True,timeout=60)
            if name == 'lme-five-embedding':
                wait_models('http://127.0.0.1:18082/v1', EMBEDDING_MODEL, (name,))
            elif name == 'lme-five-chat':
                wait_models('http://127.0.0.1:18081/v1', MODEL, (name,))
        wait_models(CHAT_URL, MODEL, ('lme-five-chat', 'lme-five-chat-meter'))
        wait_models(EMBED_URL, EMBEDDING_MODEL, ('lme-five-embedding','lme-five-embedding-meter'))
        status('verifying_baseline_services')
        verify_services()
        config = dict(client_python=str(CLIENT_PY), run_root=str(RUN / 'population'),
            inputs=str(ROOT/'inputs_five'), reference=str(QWEN/'protocol.json'),
            source_root=str(ROOT/'MemoryData'), higmem_root=str(ROOT/'HiGMem'), tokenizer=str(TOKENIZER),
            runtime_receipt=str(RUN/'runtime_receipt.json'), llm_url=CHAT_URL, embedding_url=EMBED_URL)
        write_once(RUN / 'launch_config.json', config)
        status('running_five_baselines', required_methods=list(METHODS), questions_per_method=500)
        subprocess.run([str(CLIENT_PY),'-B',str(HARNESS/'run_population.py'),
                        '--launch-config',str(RUN/'launch_config.json')],check=True)
        complete = read_json(RUN/'population/generation_complete.json')
        if complete['total_predictions'] != 2500 or tuple(complete['methods']) != tuple(METHODS):
            raise ValueError('Incomplete baseline population')
        status('baseline_generation_complete_judge_pending', official_judge_complete=False)


def main() -> int:
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-once', action='store_true', help='Only check the Qwen gate; never launch services')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'server_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def terminate(*_args):
            raise SystemExit(143)
        signal.signal(signal.SIGTERM, terminate)
        try:
            result = wait_for_qwen(args.check_once)
            if args.check_once:
                return result
            start_baselines()
        except Exception as exc:
            status('failed', error=type(exc).__name__, message=str(exc), artifacts_preserved=True)
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
