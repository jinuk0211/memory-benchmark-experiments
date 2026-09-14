"""One-shot, metered native retry of exactly three QA against copied r17 memories."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import locomo_server_queue as queue
from scripts.finalize_locomo_comparison import get_template, running_writers
from scripts.prepare_saved_simplemem_retry import (
    RETRIES,
    RUNS,
    digest,
    prepare_artifacts,
    validate_rows,
)
from scripts.response_delivery_audit import audit_delivery
from scripts.retry_simplemem_after_queue import require_free_ports
from scripts.score_locomo_comparison import read_jsonl, summarize_usage, write_report

PLAN_PATH = Path('/workspace/finish-simplemem-saved-plan-r21.json')
OUTPUT = Path('/workspace/locomo-simplemem-qa-repair-r21')
RUN_ID = 'locomo-simplemem-qa-repair-20260908-r21'
PREDECESSORS = ('locomo-simplemem-failed-context-recovery-r17', 'locomo-amem-sampling-diagnostic-r20')
PRESERVED_FIELDS = ('output', 'input_len', 'output_len', 'memory_construction_time', 'query_time_len',
                    'query', 'answer', 'qa_pair_id', 'context_id', 'query_id', 'eval_metadata',
                    'retrieved_source_id_groups', 'requested_recall_k')


def inspect_saved_tables(python: str, preparation: dict) -> None:
    """Open ONLY copied tables with the native client dependency; never create a table."""
    code = """import json,sys,lancedb
for item in json.load(sys.stdin):
    db=lancedb.connect(item['destination_runtime'])
    if item['table_name'] not in db.table_names():
        raise ValueError('Missing copied table')
    table=db.open_table(item['table_name'])
    ids=table.to_arrow().column('entry_id').to_pylist()
    if not table.count_rows()==len(ids)==len(set(ids))>0:
        raise ValueError('Empty or duplicate copied memories')
    if set(ids)!=set(item['entry_ids']):
        raise ValueError('Copied table/source-map identity mismatch')
print('copied_tables_verified')
"""
    result = subprocess.run([python, '-c', code], input=json.dumps(preparation['memory_lineage']),
                            capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0 or result.stdout.strip() != 'copied_tables_verified':
        raise ValueError(f'Copied memory table inspection failed: {result.stderr}')


def inspect_native_seed(python: str, preparation: dict, dataset: dict) -> dict:
    """Exercise the actual client's unchanged retry loader BEFORE any model service."""
    code = """import json,sys
from utils.initialization import load_existing_results
item=json.load(sys.stdin)
original=json.load(open(item['results'], encoding='utf-8'))['data']
pairs=[[{} for _ in range(count)] for count in item['counts']]
_,rows,completed,skipped=load_existing_results(item['results'],item['dataset'],pairs,retry_failed_queries=True)
if len(rows)!=1537 or skipped!=set(range(1540))-{681,930,993} or completed!={0,1,2,3,5,7,8,9}:
    raise ValueError('Native retry skip identities changed')
by_id={row['qa_pair_id']:row for row in rows}
if len(by_id)!=1537:
    raise ValueError('Native retry produced duplicate skipped identities')
for row in original:
    if row['query_id'] in skipped:
        if any(by_id.get(row['qa_pair_id'],{}).get(key)!=row.get(key) for key in item['preserved']):
            raise ValueError('Native skip inference fields changed: '+row['qa_pair_id'])
print('native_skip_verified')
"""
    payload = {'results': preparation['results'], 'counts': preparation['context_question_counts'],
               'dataset': dataset, 'preserved': PRESERVED_FIELDS}
    result = subprocess.run([python, '-c', code], input=json.dumps(payload), cwd=ROOT,
                            capture_output=True, text=True, timeout=120, check=False)
    if result.returncode != 0 or not result.stdout.rstrip().endswith('native_skip_verified'):
        raise ValueError(f'Native skip inspection failed: {result.stderr}\n{result.stdout}')
    return {'passed': True, 'skipped_queries': 1537, 'retry_global_ids': [681, 930, 993],
            'unchanged_inference_fields': list(PRESERVED_FIELDS)}


def _provenance(plan: dict) -> tuple[dict, dict, dict]:
    if plan.get('run_id') != RUN_ID or Path(plan['output']) != OUTPUT or plan.get('dtype') != 'float16':
        raise ValueError('Wrong r21 run, output or FP16 precision')
    saved, receipts = plan['saved_simplemem'], {}
    current = next(item for item in plan['methods'] if item['method'] == 'simplemem')
    for tag, run in RUNS.items():
        for kind in ('plan', 'receipt'):
            path = Path(saved[f'{kind}_{tag}']).resolve()
            if plan['file_sha256'].get(str(path)) != digest(path):
                raise ValueError('Original plan/closed receipt is not pinned')
        previous = json.loads(Path(saved[f'plan_{tag}']).read_text())
        if (previous.get('run_id') != run or (Path(previous['output']) / 'simplemem').resolve() != Path(saved[f'source_{tag}']).resolve()
                or any(not plan.get(key) or plan[key] != previous.get(key) for key in
                       ('model', 'model_revision', 'dtype', 'embedding_path', 'dataset', 'scorer'))):
            raise ValueError('Original model/data/embedding conditions or provenance changed')
        item = next(item for item in previous['methods'] if item['method'] == 'simplemem')
        for key in ('agent_config', 'dataset_config'):
            if previous['file_sha256'].get(str(Path(item[key]).resolve())) != digest(item[key]):
                raise ValueError('Original native configuration changed from its plan pin')
            if yaml.safe_load(Path(item[key]).read_text()) != yaml.safe_load(Path(current[key]).read_text()):
                raise ValueError('Native inference or dataset configuration changed')
        receipts[tag] = json.loads(Path(saved[f'receipt_{tag}']).read_text())
    return saved, receipts, current


def _audit_attempt(plan: dict, out: Path, preparation: dict, original: list[dict],
                   expected: dict, template: str, code: int | None, error: str | None) -> int:
    usage, journal_errors = [], {}
    for filename in ('llm_usage.jsonl', 'embedding_usage.jsonl'):
        try:
            if not (out / filename).is_file():
                raise FileNotFoundError(f'Required usage journal is absent: {filename}')
            usage.extend(read_jsonl(out / filename))
        except (OSError, ValueError) as failure:
            journal_errors[filename] = str(failure)
    with (out / 'usage.jsonl').open('x', encoding='utf-8') as stream:
        for row in usage:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = summarize_usage(usage, True)
    if journal_errors or any(row.get('usage_invalid') or row.get('usage_status') != 'reported' for row in usage):
        summary['complete'] = False
    summary['journal_errors'] = journal_errors
    summary['exact_tokens'] = summary['reported_tokens'] if summary['complete'] else None
    good, issues, after, delivery = code == 0 and not error, [], [], {'complete': False}
    try:
        after = json.loads(Path(preparation['results']).read_text())['data']
        validate_rows(after, expected, template)
        after_ids = {row['qa_pair_id']: row for row in after}
        if any(row.get('status') == 'failed' for row in after):
            raise ValueError('Some repaired QA still failed')
        if any(any(after_ids[row['qa_pair_id']].get(key) != row.get(key) for key in PRESERVED_FIELDS)
               for row in original if row['qa_pair_id'] not in RETRIES):
            raise ValueError('A skipped seed prediction/input/retrieval field changed')
        selected = {key: qa for key, qa in expected.items() if f'{key[0]}_qa{key[1]}' in RETRIES}
        for row in usage:
            if row.get('phase') not in ('initialize', 'qa') or row.get('request_kind') not in ('chat_completion', 'embedding'):
                raise ValueError('Recovery performed unapproved construction or request kind')
            if row['phase'] == 'qa':
                q = row.get('question_id')
                if type(q) is not int and not (isinstance(q, str) and q.isascii() and q.isdecimal() and str(int(q)) == q):
                    raise ValueError('Recovery usage lacks an original question index')
                if (row.get('sample_id'), int(q)) not in selected:
                    raise ValueError('Recovery made an unselected QA request')
        queue.audit_coverage(usage, selected, RUN_ID, 'simplemem')
        delivery = audit_delivery(usage)
    except (OSError, ValueError, KeyError, TypeError) as failure:
        good = False
        issues.append(str(failure))
    unchanged = all(digest(path) == value for path, value in preparation['source_sha256'].items())
    good &= unchanged and summary['complete'] and delivery['complete']
    report = {'predictions_recovered': bool(good), 'complete': False, 'requires_composite_audit': True,
        'run_id': RUN_ID, 'retry_ids': sorted(RETRIES), 'seed_is_final_selection': False,
        'source_artifacts_unchanged': unchanged, 'usage': summary, 'delivery': delivery,
        'new_memory_add_requests': sum(row.get('phase') == 'memory_add' for row in usage) if summary['complete'] else None,
        'error': error, 'issues': issues, 'exit_code': code,
        'note': 'Only r21 QA repair overhead; r13/r17 construction and all previous attempts remain separate. '
                'The 1537 seed successes are skip evidence, not final answer selection. No full1540 score claimed.'}
    write_report(out / 'recovered_queries.json', {'data': [row for row in after if row.get('qa_pair_id') in RETRIES]})
    write_report(out / 'recovery_status.json', report)
    return 0 if good else 1


def main() -> int:
    plan = json.loads(PLAN_PATH.read_text())
    queue.validate_plan(plan)
    saved, receipts, item = _provenance(plan)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    while True:
        states = [queue.supervisor_state(name) for name in PREDECESSORS]
        if any(state not in {'RUNNING', 'STARTING', 'STOPPING', 'EXITED'} for state in states):
            raise RuntimeError('Unexpected predecessor state')
        if all(state == 'EXITED' for state in states):
            break
        time.sleep(15)
    if any(running_writers(saved[f'source_{tag}']) for tag in RUNS):
        raise RuntimeError('Original SimpleMem writers are still live')
    agent = yaml.safe_load(Path(item['agent_config']).read_text())
    dataset = yaml.safe_load(Path(item['dataset_config']).read_text())
    expected = queue.expected_questions(plan['dataset'], 1540)
    out = OUTPUT / 'simplemem'
    preparation = prepare_artifacts(Path(saved['source_r17']), out / 'artifacts', expected,
        agent, dataset, receipts['r17'], Path(saved['source_r13']), receipts['r13'])
    preparation['source_sha256'].update({str(Path(saved[f'{kind}_{tag}']).resolve()): digest(saved[f'{kind}_{tag}'])
                                        for tag in RUNS for kind in ('plan', 'receipt')})
    write_report(out / 'preparation.json', preparation)
    inspect_saved_tables(plan['client_python'], preparation)
    preparation['copied_tables_verified'] = True
    preparation['native_skip_probe'] = inspect_native_seed(plan['client_python'], preparation, dataset)
    write_report(out / 'preparation.json', preparation)
    require_free_ports()
    queue.validate_plan(plan)
    queue.wait_vllm(plan)
    original = json.loads(Path(preparation['results']).read_text())['data']
    env = os.environ.copy()
    env.update(OPENAI_API_KEY='local-comparison', OPENAI_BASE_URL=queue.LLM_ORIGIN + '/v1',
        EMBEDDING_BASE_URL=queue.EMBED_ORIGIN + '/v1', METER_RUN_ID=RUN_ID, METER_METHOD='simplemem',
        METER_LLM_PROXY_ORIGIN=queue.LLM_ORIGIN, METER_EMBEDDING_PROXY_ORIGIN=queue.EMBED_ORIGIN,
        METER_TIMING_JOURNAL=str(out / 'timing.jsonl'), BASELINE_STRICT_COMPARISON='1',
        CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    proxy = [plan['server_python'], str(ROOT / 'scripts/repairing_openai_proxy.py'),
             '--timeout', '1200', '--run-id', RUN_ID, '--method', 'simplemem']
    command = [plan['client_python'], str(ROOT / 'main.py'), '--agent_config', item['agent_config'],
               '--dataset_config', item['dataset_config'], '--artifact_root', str(out / 'artifacts'),
               '--retry_failed_queries', '--fail-on-query-error']
    code, error, samples, started = None, None, [], time.monotonic()
    try:
        with queue.service([plan['server_python'], str(ROOT / 'scripts/serve_minilm_embeddings.py'),
                           '--model-path', plan['embedding_path']], out / 'embedding_server.log') as encoder:
            queue.wait_health('http://127.0.0.1:18081', encoder)
            with queue.service(proxy + ['--journal', str(out / 'llm_usage.jsonl'), '--upstream-base-url',
                'http://127.0.0.1:18080/v1', '--port', '18082', '--comparison-policy'], out / 'llm_proxy.log') as llm, \
                 queue.service(proxy + ['--journal', str(out / 'embedding_usage.jsonl'), '--upstream-base-url',
                'http://127.0.0.1:18081/v1', '--port', '18083'], out / 'embedding_proxy.log') as embed:
                try:
                    queue.wait_health(queue.LLM_ORIGIN, llm)
                    queue.wait_health(queue.EMBED_ORIGIN, embed)
                    samples.append(queue.gpu_sample())
                    with queue.service(command, out / 'benchmark.log', env) as process:
                        while process.poll() is None:
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                pass
                            samples.append(queue.gpu_sample())
                        code = process.returncode
                finally:
                    queue.drain_proxies([llm, embed])
    except Exception as failure:  # noqa: BLE001 - preserve metered failure evidence, then return failure.
        error = f'{type(failure).__name__}: {failure}'
    write_report(out / 'gpu_samples.json', samples)
    timing = read_jsonl(out / 'timing.jsonl') if (out / 'timing.jsonl').exists() else []
    write_report(out / 'telemetry.json', queue.telemetry(samples, timing, time.monotonic() - started, out))
    return _audit_attempt(plan, out, preparation, original, expected,
                          get_template(dataset['sub_dataset'], 'query', agent['agent_name']), code, error)


if __name__ == '__main__':
    raise SystemExit(main())
