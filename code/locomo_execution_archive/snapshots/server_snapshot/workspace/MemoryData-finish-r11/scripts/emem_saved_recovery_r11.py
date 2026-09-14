"""One-shot measured recovery of the two failed r4 QA from closed saved memories."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import locomo_server_queue as queue
from scripts.prepare_saved_emem_retry import prepare_artifacts
from scripts.response_delivery_audit import audit_delivery
from scripts.score_locomo_comparison import read_jsonl, summarize_usage, write_report
from utils.artifact_paths import generate_agent_save_folder_path


def main():
    plan = json.loads(Path('/workspace/finish-emem-saved-plan-r11.json').read_text())
    queue.validate_plan(plan)
    queue.wait_vllm(plan)
    if queue.supervisor_state('locomo-finish-emem-amem-r4') != 'EXITED':
        raise RuntimeError('Original E-Mem writer is not closed')
    from scripts.retry_simplemem_after_queue import require_free_ports
    require_free_ports()
    out = Path(plan['output'])
    out.mkdir(exist_ok=False)
    item = next(item for item in plan['methods'] if item['method'] == 'e_mem')
    data = json.loads(Path(plan['dataset']).read_text())
    expected_mapping = {}
    for context, sample in enumerate(data):
        for index, qa in enumerate(sample['qa']):
            if qa['category'] in (1, 2, 3, 4):
                expected_mapping[f"{sample['sample_id']}_qa{index}"] = (context, len(expected_mapping))
    source = Path('/workspace/locomo-finish-emem-amem-r4/e_mem')
    preparation = prepare_artifacts(source, out / 'artifacts', expected_mapping)
    write_report(out / 'preparation.json', preparation)
    retry_ids = {'conv-26_qa74', 'conv-49_qa38'}
    if set(preparation['retry_ids']) != retry_ids:
        raise ValueError('Unexpected failed QA selection')
    agent = yaml.safe_load(Path(item['agent_config']).read_text())
    dataset = yaml.safe_load(Path(item['dataset_config']).read_text())
    for index in range(10):
        state = Path(generate_agent_save_folder_path(agent, dataset, index, str(out / 'artifacts')))
        if not (state / 'e_mem_ready.txt').is_file():
            raise ValueError(f'Native saved-state path mismatch for context {index}')
    original = json.loads(Path(preparation['results']).read_text())['data']
    env = os.environ.copy()
    env.update(OPENAI_API_KEY='local-comparison', OPENAI_BASE_URL=queue.LLM_ORIGIN + '/v1',
               EMBEDDING_BASE_URL=queue.EMBED_ORIGIN + '/v1',
               METER_RUN_ID=plan['run_id'], METER_METHOD='e_mem',
               METER_LLM_PROXY_ORIGIN=queue.LLM_ORIGIN, METER_EMBEDDING_PROXY_ORIGIN=queue.EMBED_ORIGIN,
               METER_TIMING_JOURNAL=str(out / 'timing.jsonl'), BASELINE_STRICT_COMPARISON='1',
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    proxy = [plan['server_python'], str(ROOT / 'scripts/repairing_openai_proxy.py'),
             '--timeout', '1200', '--run-id', plan['run_id'], '--method', 'e_mem']
    command = [plan['client_python'], str(ROOT / 'main.py'),
               '--agent_config', item['agent_config'], '--dataset_config', item['dataset_config'],
               '--artifact_root', str(out / 'artifacts'), '--retry_failed_queries', '--fail-on-query-error']
    with queue.service([plan['server_python'], str(ROOT / 'scripts/serve_minilm_embeddings.py'),
                        '--model-path', plan['embedding_path']], out / 'embedding_server.log') as encoder:
        queue.wait_health('http://127.0.0.1:18081', encoder)
        with queue.service(proxy + ['--journal', str(out / 'llm_usage.jsonl'), '--upstream-base-url',
                           'http://127.0.0.1:18080/v1', '--port', '18082', '--comparison-policy'],
                           out / 'llm_proxy.log') as llm, \
             queue.service(proxy + ['--journal', str(out / 'embedding_usage.jsonl'), '--upstream-base-url',
                           'http://127.0.0.1:18081/v1', '--port', '18083'],
                           out / 'embedding_proxy.log') as embed:
            queue.wait_health(queue.LLM_ORIGIN, llm)
            queue.wait_health(queue.EMBED_ORIGIN, embed)
            started = time.monotonic()
            samples = [queue.gpu_sample()]
            with queue.service(command, out / 'benchmark.log', env) as process:
                while process.poll() is None:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    samples.append(queue.gpu_sample())
                code = process.returncode
            queue.drain_proxies([llm, embed])
            wall = time.monotonic() - started
    usage = read_jsonl(out / 'llm_usage.jsonl') + read_jsonl(out / 'embedding_usage.jsonl')
    write_report(out / 'gpu_samples.json', samples)
    write_report(out / 'telemetry.json', queue.telemetry(samples, read_jsonl(out / 'timing.jsonl'), wall, out))
    write_report(out / 'attempt_usage.json', summarize_usage(usage, True))
    after = json.loads(Path(preparation['results']).read_text())['data']
    after_by_id = {row['qa_pair_id']: row for row in after}
    good = (code == 0 and len(after) == len(after_by_id) == 1540
            and set(after_by_id) == set(expected_mapping)
            and all(row.get('status') != 'failed' and str(row.get('output') or '').strip() for row in after))
    for row in original:
        if row['qa_pair_id'] not in retry_ids:
            good &= all(after_by_id.get(row['qa_pair_id'], {}).get(key) == row.get(key)
                        for key in ('output', 'query', 'qa_pair_id', 'context_id', 'query_id', 'eval_metadata'))
    for filename, digest in preparation['source_sha256'].items():
        good &= hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
    expected = queue.expected_questions(plan['dataset'], 1540)
    selected = {key: qa for key, qa in expected.items() if f'{key[0]}_qa{key[1]}' in retry_ids}
    queue.audit_coverage(usage, selected, plan['run_id'], 'e_mem')
    good &= not any(row.get('phase') == 'memory_add' for row in usage)
    good &= audit_delivery(usage)['complete'] and summarize_usage(usage, True)['complete']
    write_report(out / 'recovery_status.json', {'predictions_recovered': good, 'complete': False,
                 'requires_composite_audit': True, 'retry_ids': sorted(retry_ids),
                 'note': 'Recovery overhead only. Original construction/1538 QA and failed attempts remain separate.'})
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
