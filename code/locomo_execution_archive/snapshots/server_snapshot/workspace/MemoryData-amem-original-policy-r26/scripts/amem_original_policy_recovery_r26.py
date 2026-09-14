"""Opt-in original A-MEM policy: real 80-turn/3-QA smoke, then original 1540 QA.

Preparation only until explicitly launched. Never deletes results or controls PC
power. Prior diagnostic failures need only be CLOSED, not claimed successful.
"""
import hashlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import locomo_server_queue as queue
from scripts.amem_failed_context_recovery_r15 import failed_contexts
from scripts.audit_amem_original_policy import POLICY, audit_original_policy
from scripts.finalize_locomo_comparison import (
    get_template,
    normalize_records,
    running_writers,
)
from scripts.retry_simplemem_after_queue import require_free_ports
from scripts.run_locomo_live_probes import run_probe
from scripts.score_locomo_comparison import (
    build_report,
    read_jsonl,
    validate_predictions,
    write_report,
)

RUN_ID = 'locomo-amem-original-policy-20260908-r26'
PROBE_RUN_ID = 'locomo-amem-original-policy-probe-20260908-r26'
PLAN_PATH = Path('/workspace/amem-original-policy-plan-r26.json')
FULL_OUTPUT = Path('/workspace/locomo-amem-original-policy-r26')
PROBE_OUTPUT = Path('/workspace/locomo-amem-original-policy-probe-r26')
PRIOR_OUTPUT = Path('/workspace/locomo-finish-amem-validated-r12/a_mem')
PREDECESSORS = ('locomo-amem-probe-then-full-r12', 'locomo-amem-failed-context-recovery-r15',
    'locomo-amem-native-length-diagnostic-r18', 'locomo-amem-sampling-diagnostic-r20',
    'locomo-amem-upstream-prompt-diagnostic-r22', 'locomo-amem-upstream-schema-diagnostic-r23')
OTHER_PRIOR_OUTPUTS = tuple(Path('/workspace') / name for name in (
    'locomo-amem-live-probe-r12', 'locomo-amem-r12-payloads', 'locomo-amem-recovery-probe-r15',
    'locomo-finish-amem-failed-contexts-r15', 'locomo-amem-r15-payloads',
    'locomo-amem-native-length-diagnostic-r18', 'locomo-amem-sampling-diagnostic-r20',
    'locomo-amem-upstream-prompt-diagnostic-r22', 'locomo-amem-upstream-schema-diagnostic-r23'))


@contextmanager
def policy_environment(output: Path):
    """Scope only the two task-owned instrumentation variables to child processes."""
    values = {'AMEM_ORIGINAL_FAILURE_POLICY': '1', 'AMEM_SEMANTIC_JOURNAL_DIR': str(output / 'semantic_outcomes')}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def require_policy(plan: dict) -> None:
    """Keep all six pinned configurations and explicitly skip the other five."""
    queue.validate_plan(plan)
    if (plan.get('run_id') != RUN_ID or Path(plan['output']) != FULL_OUTPUT
            or plan.get('amem_failure_policy') != POLICY or plan.get('dtype') != 'float16'
            or plan.get('model') != 'Qwen/Qwen3.5-9B' or not plan.get('model_revision')
            or not plan.get('embedding_path') or type(plan.get('context_workers')) is not int
            or not 2 <= plan['context_workers'] <= 10
            or [item['method'] for item in plan['methods']] != list(queue.METHODS)
            or set(plan.get('skip_methods', {})) != set(queue.METHODS) - {'a_mem'}):
        raise ValueError('r26 requires its explicit native policy, fixed comparison model and A-MEM-only scope')
    item = next(item for item in plan['methods'] if item['method'] == 'a_mem')
    agent = yaml.safe_load(Path(item['agent_config']).read_text())
    if (agent.get('model') != plan['model'] or agent.get('temperature') != 0
            or agent.get('qwen3_disable_thinking') is not True
            or agent.get('a_mem_embedding_model') != 'sentence-transformers/all-MiniLM-L6-v2'):
        raise ValueError('Common Qwen sampling or MiniLM configuration changed')


def audit_attempt(plan: dict, output: Path, expected: dict, supplied_rows: list | None = None) -> dict:
    """Keep incomplete journals/events and report unknowns; never manufacture calls."""
    rows, events, journal_errors, event_errors = [], [], {}, {}
    for name in ('llm_usage.jsonl', 'embedding_usage.jsonl'):
        try:
            rows.extend(read_jsonl(output / name))
        except (OSError, ValueError) as error:
            journal_errors[name] = str(error)
    combined = output / 'usage.jsonl'
    if combined.exists() or plan['run_id'] == RUN_ID:
        try:
            if read_jsonl(combined) != rows:
                journal_errors['usage.jsonl'] = 'Combined ledger differs from both original journals'
        except (OSError, ValueError) as error:
            journal_errors['usage.jsonl'] = str(error)
    if supplied_rows is not None and supplied_rows != rows:
        journal_errors['callback_rows'] = 'Probe callback differs from original journals'
    directory = output / 'semantic_outcomes'
    paths = sorted(directory.glob('*'))
    if not paths:
        event_errors[str(directory)] = 'Required native outcome journal is absent or empty'
    for path in paths:
        try:
            if not path.is_file() or path.is_symlink() or not re.fullmatch(r'[0-9]+\.jsonl', path.name):
                raise ValueError('Unexpected native outcome journal path')
            events.extend(read_jsonl(path))
        except (OSError, ValueError) as error:
            event_errors[str(path)] = str(error)
    report = audit_original_policy(rows, events, plan['run_id'], expected)
    report.update(journal_errors=journal_errors, event_errors=event_errors)
    if journal_errors or event_errors:
        report['complete'] = report['original_policy_delivery_complete'] = False
        report['outcome_counts_complete'] = False
        report['issues'].append('required_journal_or_event_evidence_incomplete')
    if journal_errors:
        report['usage']['complete'] = False
        report['usage']['exact_tokens'] = None
    write_report(output / 'native_policy_audit.json', report)
    return report


def finalize_attempt(plan: dict, output: Path, expected: dict) -> dict:
    """Score only an exact successful original full set, keeping true native lengths."""
    audit = audit_attempt(plan, output, expected)
    status = json.loads((output / 'parallel_status.json').read_text())
    if (failed_contexts(status) or status.get('selected_contexts_complete') is not True
            or status.get('context_indices') != list(range(10)) or len(expected) != 1540):
        raise ValueError('Full native attempt lacks the exact ten completed original contexts')
    raw = json.loads((output / 'selected_results.json').read_text())['data']
    samples = list(dict.fromkeys(sample for sample, _ in expected))
    identities = {str(qa.get('question_id') or f'{sample}_qa{index}'): (samples.index(sample), offset, sample)
                  for offset, ((sample, index), qa) in enumerate(expected.items())}
    merged = []
    for context, sample in enumerate(samples):
        prefix = output / f'context_{context:02d}'
        paths = list((prefix / 'artifacts').rglob('*_results.json'))
        if len(paths) != 1:
            raise ValueError('Missing or ambiguous original full result shard')
        shard = json.loads(paths[0].read_text())['data']
        marker = json.loads((prefix / 'completion.json').read_text())
        wanted = [offset for c, offset, sid in identities.values() if c == context and sid == sample]
        if (marker.get('context_index') != context or type(marker.get('context_index')) is not int
                or marker.get('sample_id') != sample or marker.get('questions') != len(wanted)
                or marker.get('result_path') != str(paths[0]) or [row.get('query_id') for row in shard] != wanted):
            raise ValueError('Full shard completion identity/count changed')
        for row in shard:
            qid, metadata = row.get('qa_pair_id'), row.get('eval_metadata') or {}
            if (type(row.get('context_id')) is not int or type(row.get('query_id')) is not int
                    or identities.get(qid) != (row['context_id'], row['query_id'], row.get('sample_id'))
                    or row.get('status') is not None or row.get('question_id') != qid
                    or metadata.get('qa_pair_id') != qid or metadata.get('question_id') != qid
                    or metadata.get('sample_id') != sample):
                raise ValueError('Original/global/context result metadata changed')
        merged.extend(shard)
    if raw != merged:
        raise ValueError('Selected results differ from untouched original shards')
    item = next(item for item in plan['methods'] if item['method'] == 'a_mem')
    agent = yaml.safe_load(Path(item['agent_config']).read_text())
    dataset = yaml.safe_load(Path(item['dataset_config']).read_text())
    normalized = output / 'normalized_results.json'
    write_report(normalized, {'data': normalize_records(raw, expected,
        get_template(dataset['sub_dataset'], 'query', agent['agent_name']))})
    predictions = queue.canonical_predictions([normalized], expected)
    valid, issues, missing = validate_predictions(predictions, expected)
    if issues or missing or len(valid) != 1540 or any(not row['prediction'].strip() for row in valid):
        raise ValueError('The original 1540 QA are missing, invalid or empty')
    path = output / 'predictions.jsonl'
    with path.open('x', encoding='utf-8') as stream:
        for row in valid:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    coverage = True
    try:
        queue.audit_coverage(read_jsonl(output / 'llm_usage.jsonl') + read_jsonl(output / 'embedding_usage.jsonl'),
                             expected, plan['run_id'], 'a_mem')
    except (OSError, ValueError):
        coverage = False
    report = build_report(plan['dataset'], path, plan['scorer'], output / 'usage.jsonl',
                          output / 'telemetry.json', plan.get('hourly_rate_usd'))
    report.update(method='a_mem', run_id=plan['run_id'], policy=POLICY, native_policy_audit=audit,
        inference_complete=True, usage_complete=audit['usage']['complete'], coverage_complete=coverage,
        original_policy_delivery_complete=audit['original_policy_delivery_complete'],
        strict_all_generation_delivery=audit['strict_all_generation_delivery'],
        benchmark_complete=bool(audit['complete'] and coverage), complete=False,
        source_sha256=plan['file_sha256'], exclusive_gpu_energy_wh=None,
        note='Original native memory fallback policy, not all-generations-strict success. '
             'QA terminals still require stop and all1540 answers are nonempty. All token rows including '
             'native lengths/fallbacks remain in gross usage. Prior attempts/probe are separate, not duplicated. '
             'Device/rental observations are not exclusive allocations. Closed artifact backup remains external.')
    write_report(output / 'original_policy_report.json', report)
    return report


def main() -> int:
    plan = json.loads(PLAN_PATH.read_text())
    require_policy(plan)
    if FULL_OUTPUT.exists():
        raise FileExistsError(FULL_OUTPUT)
    PROBE_OUTPUT.mkdir(exist_ok=False)
    state_path = PROBE_OUTPUT / 'recovery_status.json'
    write_report(state_path, {'state': 'waiting_for_predecessors', 'complete': False, 'policy': POLICY})
    active_plan, active_output, expected = None, None, None
    try:
        while True:
            states = {name: queue.supervisor_state(name) for name in PREDECESSORS}
            if any(value not in {'RUNNING', 'STARTING', 'STOPPING', 'EXITED'} for value in states.values()):
                raise RuntimeError(f'Unexpected predecessor state: {states}')
            if all(value == 'EXITED' for value in states.values()):
                break
            time.sleep(15)
        roots = (PRIOR_OUTPUT, *OTHER_PRIOR_OUTPUTS)
        if any(running_writers(path) for path in roots):
            raise RuntimeError('Prior artifact writers are still live')
        indices = failed_contexts(json.loads((PRIOR_OUTPUT / 'parallel_status.json').read_text()))
        if indices != list(range(10)) or any(json.loads(path.read_text()).get('data')
                for path in PRIOR_OUTPUT.rglob('*_results.json')):
            raise ValueError('Original r12 must retain all ten failed contexts and zero stored QA')
        preserved = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for root in roots for path in root.rglob('*') if path.is_file()}
        write_report(PROBE_OUTPUT / 'prior_artifact_sha256.json', preserved)
        expected = queue.expected_questions(plan['dataset'], 1540)
        probe_plan = dict(plan, output=str(PROBE_OUTPUT), run_id=PROBE_RUN_ID)
        probe_expected = dict(list(expected.items())[:3])
        encoder = [plan['server_python'], str(ROOT / 'scripts/serve_minilm_embeddings.py'), '--model-path', plan['embedding_path']]
        require_free_ports()
        queue.wait_vllm(plan)
        active_plan, active_output = probe_plan, PROBE_OUTPUT / 'a_mem'
        write_report(state_path, {'state': 'testing', 'complete': False, 'policy': POLICY})
        with policy_environment(active_output), queue.service(encoder, PROBE_OUTPUT / 'embedding_server.log') as process:
            queue.wait_health('http://127.0.0.1:18081', process)
            probe = run_probe(probe_plan, 'a_mem', 80,
                delivery_auditor=lambda rows: audit_attempt(probe_plan, active_output, probe_expected, rows))
        if (probe.get('passed') is not True or probe.get('qa_count') != 3 or probe.get('memory_turns') != 80
                or probe.get('response_delivery', {}).get('original_policy_delivery_complete') is not True):
            raise RuntimeError('Original-policy real smoke did not pass')
        require_policy(plan)
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != value for path, value in preserved.items()):
            raise RuntimeError('Prior artifacts changed during the smoke')
        require_free_ports()
        FULL_OUTPUT.mkdir(exist_ok=False)
        active_plan, active_output = plan, FULL_OUTPUT / 'a_mem'
        write_report(state_path, {'state': 'running_full', 'complete': False, 'smoke_passed': True, 'context_indices': indices})
        with policy_environment(active_output), queue.service(encoder, FULL_OUTPUT / 'embedding_server.log') as process:
            queue.wait_health('http://127.0.0.1:18081', process)
            queue.run_method(plan, next(item for item in plan['methods'] if item['method'] == 'a_mem'), expected, context_indices=indices)
        report = finalize_attempt(plan, active_output, expected)
        require_policy(plan)
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != value for path, value in preserved.items()):
            raise RuntimeError('Prior artifacts changed during the full run')
        if report.get('benchmark_complete') is not True:
            raise RuntimeError('Native-policy full inference/accounting audit is incomplete')
        write_report(state_path, {'state': 'finished_native_policy_inference', 'complete': False,
            'context_indices': indices, 'original_artifact_hashes_preserved': True, 'report': report})
        return 0
    except Exception as error:
        audit = None
        if active_output is not None and active_output.exists():
            scope = dict(list(expected.items())[:3]) if active_plan['run_id'] == PROBE_RUN_ID else expected
            try:
                audit = audit_attempt(active_plan, active_output, scope)
            except (OSError, ValueError, TypeError, KeyError) as audit_error:
                audit = {'complete': False, 'usage': {'complete': False, 'exact_tokens': None}, 'error': str(audit_error)}
        write_report(state_path, {'state': 'failed', 'complete': False, 'error_type': type(error).__name__,
                                 'error': str(error), 'native_policy_audit': audit})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
