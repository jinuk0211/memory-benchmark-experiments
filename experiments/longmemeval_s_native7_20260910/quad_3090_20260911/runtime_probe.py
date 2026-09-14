"""Synthetic runtime proof only. Caller: timeout --kill-after=15s 600 <LightMem Python> runtime_probe.py."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from urllib.request import Request, urlopen
import uuid

ROOT = Path('/workspace/quad_3090_20260911') / ('probe_' + os.environ['QUAD_LANE'])
EXPERIMENT = Path('/workspace/longmemeval_s_native7_20260910')
MODEL = 'Qwen/Qwen3.5-9B'
LANE_ENDPOINT = {'sm0':18081,'sm1':18091,'lm0':18101,'lm1':18111}[os.environ['QUAD_LANE']]
DIRECT_API = f'http://127.0.0.1:{LANE_ENDPOINT}/v1'
MINILM = '/workspace/.hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
COMPRESSOR = '/workspace/.hf_home/hub/models--microsoft--llmlingua-2-bert-base-multilingual-cased-meetingbank/snapshots/5f0c82792b7ea14c6484e015b6a072009496b7f2'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def native_log(path):
    with path.open('w', encoding='utf-8', buffering=1) as output:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = (os.dup(1), os.dup(2))
        try:
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            with redirect_stdout(output), redirect_stderr(output):
                yield
        finally:
            output.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def gpu_snapshot():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    gpus = [{'index': int(row[0]), 'uuid': row[1].strip(), 'name': row[2].strip(),
             'used_mib': int(row[3]), 'total_mib': int(row[4])}
            for row in csv.reader(result.stdout.splitlines()) if row]
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    processes = []
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        pid, name = int(row[0]), row[1].strip()
        try:
            argv = (Path('/proc') / str(pid) / 'cmdline').read_bytes().lower()
        except OSError:
            argv = b''
        processes.append({'pid': pid, 'name': name,
                          'used_mib': int(row[2]) if row[2].strip().isdigit() else None, 'gpu_uuid': row[3].strip(),
                          'is_vllm': 'vllm' in name.lower() or b'vllm' in argv})
    return {'gpus': gpus, 'processes': processes}


def post(url, payload, headers):
    request = Request(url, data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json', **headers}, method='POST')
    return urlopen(request, timeout=120)


def vector_proof(vector):
    if len(vector) != 384 or any(type(item) not in (int, float) or not math.isfinite(item) for item in vector):
        raise ValueError('Expected 384 finite embedding values')
    return {'dimensions': 384, 'finite': True, 'sha256': digest(vector)}


def streamed_chat(url, headers):
    content, chunks, finished, done = [], 0, None, False
    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Reply with one short word confirming readiness.'}],
               'temperature': 0, 'max_tokens': 32, 'stream': True, 'stream_options': {'include_usage': True}}
    with post(url, payload, headers) as response:
        for raw in response:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            event = json.loads(data)
            if event.get('error'):
                raise ValueError('Streaming endpoint returned an error')
            for choice in event.get('choices', []):
                fragment = choice.get('delta', {}).get('content')
                if fragment:
                    content.append(fragment)
                    chunks += 1
                if choice.get('finish_reason') is not None:
                    finished = choice['finish_reason']
    text = ''.join(content)
    if not text.strip() or not done or finished != 'stop':
        raise ValueError('Streaming response did not finish with nonempty text')
    return {'content_characters': len(text), 'content_chunks': chunks,
            'finish_reason': finished, 'done_received': done, 'content_sha256': digest(text)}


def run(probe, probe_id, result):
    with urlopen(DIRECT_API + '/models', timeout=10) as response:
        models = json.load(response)
    model = next((row for row in models['data'] if row['id'] == MODEL), None)
    if model is None:
        raise ValueError('Qwen model not served by metered endpoint')
    max_len = model.get('max_model_len')
    if max_len is not None and (type(max_len) is not int or max_len != 65536):
        raise ValueError('Served Qwen max_model_len differs from 65536')
    result['qwen_models'] = {'model': MODEL, 'max_model_len': max_len,
                             'engine_inspection_required': max_len is None}
    headers = {'X-Meter-Run-Id': probe_id, 'X-Meter-Method': 'runtime_probe',
               'X-Meter-Phase': 'synthetic_stream', 'X-Meter-Sample-Id': probe_id,
               'X-Meter-Question-Id': probe_id}
    result['streamed_chat'] = streamed_chat(DIRECT_API + '/chat/completions', headers)
    with post('http://127.0.0.1:18084/v1/embeddings',
              {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'input': 'Synthetic runtime embedding check.'},
              {**headers, 'X-Meter-Phase': 'synthetic_cpu_embedding'}) as response:
        embeddings = json.load(response)
    result['cpu_embedding_endpoint'] = vector_proof(embeddings['data'][0]['embedding'])
    result['cpu_embedding_endpoint']['endpoint'] = 'http://127.0.0.1:18084/v1'
    # serve_minilm.py pins device=cpu; retain the exact source proof as well.
    result['cpu_embedding_endpoint']['server_source_sha256'] = hashlib.sha256(
        (EXPERIMENT / 'serve_minilm.py').read_bytes()).hexdigest()

    import torch
    import native_lightmem as native
    if not torch.cuda.is_available() or '3090' not in torch.cuda.get_device_name(0):
        raise ValueError('Expected live RTX 3090 CUDA device')
    if torch.__version__.split('+')[0] != '2.8.0' or version('transformers') != '4.57.0':
        raise ValueError('Probe must run in the pinned LightMem environment')
    result['versions'] = {name: version(name) for name in (
        'torch', 'transformers', 'llmlingua', 'sentence-transformers', 'qdrant-client', 'nltk')}
    result['python'] = sys.version
    result['torch_cuda_version'] = torch.version.cuda
    torch.cuda.reset_peak_memory_stats()
    result['gpu_before'] = gpu_snapshot()
    initial_vllm = {p['pid'] for p in result['gpu_before']['processes'] if p['is_vllm'] and p['gpu_uuid'] == os.environ['CUDA_VISIBLE_DEVICES']}
    if not initial_vllm:
        raise ValueError('No resident vLLM GPU process was observed')

    source = {
        'haystack_session_ids': [probe_id],
        'haystack_dates': ['2026/09/11 (Fri) 17:00'],
        'haystack_sessions': [[
            {'role': 'user', 'content': (
                'Please remember my storage arrangement. My spare observatory key is inside the blue box '
                'on the cedar shelf in my study. The blue box has a silver latch and the label ORION-SEVEN. '
                'I placed the spare key there this morning after returning from the observatory. '
                'The cedar shelf is above my writing desk. This is the storage location I want to remember.')},
            {'role': 'assistant', 'content': 'I understand the storage arrangement you described.'},
            {'role': 'user', 'content': (
                'To confirm, my spare observatory key remains in the blue box on the cedar shelf. '
                'I have not moved it. I will need the spare key when I visit the observatory tomorrow.')},
            {'role': 'assistant', 'content': 'I understand that the location has stayed the same.'},
        ]],
    }
    query = {'question': 'Where is my spare observatory key stored?', 'question_date': '2026/09/11 (Fri) 17:10'}
    save(probe / 'synthetic_inputs.json', {'source': source, 'query': query})
    result['synthetic_inputs_sha256'] = digest({'source': source, 'query': query})
    cache = probe / 'qdrant'
    if cache.exists() or not cache.resolve().is_relative_to(probe.resolve()):
        raise ValueError('Synthetic Qdrant directory must be new and isolated')
    memory_api = f'http://127.0.0.1:18083/meter/{probe_id}/runtime_probe/synthetic_memory/{probe_id}/{probe_id}/v1'
    reader_api = f'http://127.0.0.1:18083/meter/{probe_id}/runtime_probe/synthetic_reader/{probe_id}/{probe_id}/v1'
    scope = native.load_native(EXPERIMENT / 'source/LightMem', memory_api, MODEL, MINILM, COMPRESSOR, cache)
    result['native_runner_sha256'] = hashlib.sha256((EXPERIMENT / 'native_lightmem.py').read_bytes()).hexdigest()
    result['native_upstream_sha256'] = native.UPSTREAM_SHA256
    memory = scope['load_lightmem'](probe_id)
    reader = scope['LLMModel'](MODEL, 'local', reader_api)
    compressor_model = memory.compressor.inner_compressor.model
    embedding_model = memory.text_embedder.model[0].auto_model
    if (not next(compressor_model.parameters()).is_cuda or not next(embedding_model.parameters()).is_cuda
            or memory.text_embedder.use_api or memory.text_embedder.model.get_sentence_embedding_dimension() != 384):
        raise ValueError('Native compressor and internal MiniLM must use CUDA')
    result['native_model_devices'] = {
        'compressor': str(next(compressor_model.parameters()).device),
        'compressor_dtype': str(next(compressor_model.parameters()).dtype),
        'internal_minilm': str(next(embedding_model.parameters()).device),
        'internal_minilm_dtype': str(next(embedding_model.parameters()).dtype)}
    forwards = {'compressor': 0, 'internal_minilm': 0}

    def count(name):
        def observe(_module, _args, _output):
            forwards[name] += 1
        return observe

    handles = [compressor_model.register_forward_hook(count('compressor')),
               embedding_model.register_forward_hook(count('internal_minilm'))]
    try:
        result['gpu_loaded'] = gpu_snapshot()
        build = native.build_memory(memory, source)
        count_points = memory.embedding_retriever.client.count(collection_name=probe_id, exact=True).count
        stored, _ = memory.embedding_retriever.client.scroll(
            collection_name=probe_id, limit=1, with_vectors=True, with_payload=False)
        if count_points < 1 or not stored:
            raise ValueError('Native extraction did not store any Qdrant memory')
        result['stored_embedding'] = vector_proof(stored[0].vector)
        answer, memories = native.answer_question(memory, reader, query)
        torch.cuda.synchronize()
        if (not isinstance(answer, str) or not answer.strip() or not memories
                or any(not isinstance(item, str) or not item.strip() for item in memories)):
            raise ValueError('Native retrieval/reader returned empty output')
        if 'blue' not in answer.lower() or 'cedar' not in answer.lower():
            raise ValueError('Native reader did not recover the synthetic storage fact')
        extraction_calls = sum(item.get('api_call_nums', 0) for item in build['native_add_results'])
        if build['turns_passed_to_native'] != 4 or extraction_calls < 1:
            raise ValueError('Native extraction did not process the synthetic pairs')
        if forwards['compressor'] < 1 or forwards['internal_minilm'] < 2:
            raise ValueError('Actual CUDA compression and embedding forwards were not observed')
        result['native_lightmem'] = {
            'qdrant_path': str(cache / probe_id), 'stored_points': count_points,
            'source_turns': build['turns_passed_to_native'], 'extraction_calls': extraction_calls,
            'retrieved_memories': len(memories), 'answer_characters': len(answer),
            'answer_sha256': digest(answer), 'synthetic_fact_recovered': True,
            'cuda_forward_calls': forwards, 'construction_seconds': build['construction_seconds']}
        result['gpu_after'] = gpu_snapshot()
        remaining_vllm = {p['pid'] for p in result['gpu_after']['processes'] if p['is_vllm'] and p['gpu_uuid'] == os.environ['CUDA_VISIBLE_DEVICES']}
        if not initial_vllm.intersection(remaining_vllm):
            raise ValueError('The resident vLLM GPU process did not survive native compression')
        result['concurrent_vllm_pids'] = sorted(initial_vllm.intersection(remaining_vllm))
        result['torch_memory'] = {
            'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        result['nvidia_smi_used_peak_mib_approx'] = max(
            row['used_mib'] for key in ('gpu_before', 'gpu_loaded', 'gpu_after') for row in result[key]['gpus'])
        result['nvidia_smi_peak_scope'] = 'Maximum of three process-stage samples; torch peak counters are continuous for this process'
    finally:
        for handle in handles:
            handle.remove()
        memory.embedding_retriever.client.close()
        reader.client.close()


def route_proof(probe_id, lane, gpu_uuid):
    journal = Path('/workspace/quad_3090_20260911/request_usage.jsonl')
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        rows = []
        if journal.exists():
            for line in journal.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (row.get('run_id') == probe_id and row.get('success') is True
                        and row.get('inference_lane') == lane
                        and row.get('inference_gpu_uuid') == gpu_uuid
                        and row.get('inference_instance_id') == '50577514'
                        and row.get('canonical_benchmark') is False):
                    rows.append(row)
        if rows:
            return {'successful_requests': len(rows), 'inference_lane': lane,
                    'inference_gpu_uuid': gpu_uuid, 'records_sha256': digest(rows)}
        time.sleep(0.2)
    raise ValueError('Actual metered GPU route was not recorded')

def main():
    if os.environ.get('CONTAINER_ID') != '50577514':
        raise ValueError('Probe is fixed to recovery instance 50577514')
    for name in ('CONTAINER_API_KEY', 'OPENROUTER_API_KEY'):
        os.environ.pop(name, None)
    os.environ.update(HF_HOME='/workspace/.hf_home', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                      MKL_NUM_THREADS='4', OPENAI_API_KEY='EMPTY', NO_PROXY='localhost,127.0.0.1,::1',
                      no_proxy='localhost,127.0.0.1,::1',
                      NLTK_DATA='/root/.cache/longmemeval_s_native7_20260910/nltk_data')
    sys.path.insert(0, str(EXPERIMENT))
    probe_id = 'quad_probe_' + os.environ['QUAD_LANE'] + '_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:12]
    probe = ROOT / probe_id
    probe.mkdir(mode=0o700, parents=True)
    previous = ROOT / 'runtime_probe_verified.json'
    if previous.exists():
        previous.replace(probe / 'previous_shared_runtime_probe_verified.json')
    result = {'status': 'runtime_probe_running', 'instance_id': '50577514', 'probe_id': probe_id,
              'lane': os.environ['QUAD_LANE'], 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'benchmark_population_member': False, 'scope': 'Synthetic only; excluded from all canonical 500 results',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'started_at': time.time()}
    log = probe / 'native.log'
    try:
        with native_log(log):
            try:
                run(probe, probe_id, result)
            except Exception:
                traceback.print_exc()
                raise
        if any(marker in log.read_text(encoding='utf-8', errors='replace').lower()
               for marker in ('compress error', 'secondary compress error')):
            raise ValueError('Native compression fallback was observed')
        result['status'] = ('runtime_probe_needs_engine_inspection'
                            if result['qwen_models']['engine_inspection_required'] else 'runtime_probe_verified')
        result['central_route'] = route_proof(probe_id, os.environ['QUAD_LANE'], os.environ['CUDA_VISIBLE_DEVICES'])
        result['completed_at'] = time.time()
        save(probe / 'result.json', result)
        if result['status'] == 'runtime_probe_verified':
            save(ROOT / 'runtime_probe_verified.json', result)
        print(json.dumps({'status': result['status'], 'probe_id': probe_id,
                          'result_path': str(probe / 'result.json'), 'log_path': str(log)}), flush=True)
        return 0 if result['status'] == 'runtime_probe_verified' else 2
    except Exception as error:
        result.update(status='runtime_probe_failed', error_type=type(error).__name__, completed_at=time.time())
        save(probe / 'result.json', result)
        print(json.dumps({'status': result['status'], 'error_type': type(error).__name__,
                          'probe_id': probe_id, 'log_path': str(log)}), flush=True)
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'status': 'runtime_probe_failed', 'error_type': type(error).__name__}), flush=True)
        sys.exit(1)
