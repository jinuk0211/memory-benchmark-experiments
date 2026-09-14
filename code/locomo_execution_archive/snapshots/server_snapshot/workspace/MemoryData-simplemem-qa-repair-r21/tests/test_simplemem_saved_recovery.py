"""Native saved-memory recovery must touch only the three proven failed QA."""
import hashlib
import importlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts.finalize_locomo_comparison import get_template
from scripts.score_locomo_comparison import expected_questions
from utils.artifact_paths import (
    build_hashed_runtime_dir,
    generate_agent_save_folder_path,
)

COUNTS = [152, 81, 152, 199, 178, 123, 150, 191, 156, 158]
SAMPLES = ['conv-26', 'conv-30', 'conv-41', 'conv-42', 'conv-43',
           'conv-44', 'conv-47', 'conv-48', 'conv-49', 'conv-50']
RETRIES = {'conv-43_qa97': (4, 681), 'conv-47_qa45': (6, 930), 'conv-47_qa108': (6, 993)}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(source, run, contexts, failed):
    status = {'active_contexts': [], 'failed': bool(failed), 'complete': False, 'events': [
        *[{'event': 'start', 'context': c, 'pid': 100 + c} for c in contexts],
        *[{'event': 'end', 'context': c, 'exit_code': int(c in failed)} for c in contexts]]}
    dump(source / 'parallel_status.json', status)
    return {'method': 'simplemem', 'run_id': run, 'original_attempt_closed': True,
            'parallel_status': status, 'raw_source_sha256': {
                str(p.resolve()): sha(p) for p in source.rglob('*') if p.is_file()}}


@pytest.fixture
def original(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('s')  # Keep native hashed paths below Windows MAX_PATH.
    source, old, destination = [tmp_path / name for name in ('r17/simplemem', 'r13/simplemem', 'retry/artifacts')]
    agent = {'agent_name': 'Agentic_memory_simplemem', 'model': 'Qwen/Qwen3.5-9B', 'agent_chunk_size': 1}
    config = {'dataset': 'locomo', 'sub_dataset': 'locomo_qa', 'chunk_size': 1, 'debug': False}
    data = [{'sample_id': sample, 'qa': [{'question': f'Question {c}/{q}?', 'answer': f'Answer {c}/{q}',
        'evidence': [f'D{q}:1'], 'category': q % 4 + 1} for q in range(COUNTS[c])]} for c, sample in enumerate(SAMPLES)]
    dataset_path = tmp_path / 'dataset.json'
    dump(dataset_path, data)
    template = get_template(config['sub_dataset'], 'query', agent['agent_name'])
    for context, sample in enumerate(data):
        artifacts = (old if context == 5 else source) / f'context_{context:02d}' / 'artifacts'
        rows = []
        for index, qa in enumerate(sample['qa']):
            qid = f"{sample['sample_id']}_qa{index}"
            row = {'query_id': sum(COUNTS[:context]) + index, 'context_id': context,
                   'qa_pair_id': qid, 'sample_id': sample['sample_id'],
                   'query': template.format(question=qa['question']), 'answer': qa['answer'],
                   'output': ' preserved output ', 'input_len': 10, 'output_len': 2,
                   'query_time_len': 1.0, 'memory_construction_time': 5.0,
                   'eval_metadata': {'dataset': 'locomo_qa',
                   'sample_id': sample['sample_id'], 'question_id': qid, 'qa_pair_id': qid,
                   'category': str(qa['category']), 'evidence': qa['evidence']}}
            if qid in RETRIES:
                row.update(status='failed', output='', error='Original native parser error')
            else:
                row['question_id'] = qid
            rows.append(row)
        dump(artifacts / 'outputs/simplemem/native_results.json', {'data': rows})
        if context in (4, 6):
            state = Path(generate_agent_save_folder_path(agent, config, context, str(artifacts)))
            state.mkdir(parents=True)
            (state / 'simplemem_ready.txt').write_text('ready')
            dump(state / 'simplemem_source_map.json', {'entry-1': ['D1:1']})
            runtime = Path(build_hashed_runtime_dir(str(state), '_simplemem_runtime'))
            for rel in ('locomo_qa_memory.lance/_versions/1.manifest',
                        'locomo_qa_memory.lance/data/1.lance',
                        'locomo_qa_memory.lance/_indices/fts/meta.json',
                        'locomo_qa_memory.lance/_indices/fts/segment.idx'):
                path = runtime / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('preserved opaque state')
    r17 = receipt(source, 'locomo-simplemem-failed-contexts-20260908-r17', [c for c in range(10) if c != 5], [4, 6])
    r13 = receipt(old, 'locomo-full-simplemem-validated-20260908-r13', list(range(10)), [c for c in range(10) if c != 5])
    return source, destination, expected_questions(dataset_path, 1540), agent, config, r17, old, r13


def prepare(original):
    return importlib.import_module('scripts.prepare_saved_simplemem_retry').prepare_artifacts(*original)


def test_exact_native_seed_rekeys_only_two_memories_and_preserves_sources(original):
    source, destination, expected, agent, config, _, old, _ = original
    before = {p: p.read_bytes() for root in (source, old) for p in root.rglob('*') if p.is_file()}
    result = prepare(original)
    assert result['rows'] == 1540 and result['retry_ids'] == sorted(RETRIES)
    assert result['skipped_seed_rows'] == 1537 and result['seed_is_final_selection'] is False
    rows = json.loads(Path(result['results']).read_text())['data']
    assert [r['query_id'] for r in rows] == list(range(1540))
    assert [r['qa_pair_id'] for r in rows if r.get('status') == 'failed'] == list(RETRIES)
    assert len(list(destination.rglob('simplemem_ready.txt'))) == 2
    for context in (4, 6):
        new_state = Path(generate_agent_save_folder_path(agent, config, context, str(destination)))
        old_state = Path(generate_agent_save_folder_path(agent, config, context, str(source / f'context_{context:02d}/artifacts')))
        old_runtime = Path(build_hashed_runtime_dir(str(old_state), '_simplemem_runtime'))
        new_runtime = Path(build_hashed_runtime_dir(str(new_state), '_simplemem_runtime'))
        assert old_runtime.name != new_runtime.name
        assert (new_runtime / 'locomo_qa_memory.lance/data/1.lance').read_bytes() == (old_runtime / 'locomo_qa_memory.lance/data/1.lance').read_bytes()
    assert all(p.read_bytes() == value for p, value in before.items())
    assert len(result['memory_lineage']) == 2
    assert set(expected) == {(r['sample_id'], int(r['qa_pair_id'].split('_qa')[-1])) for r in rows}


@pytest.mark.parametrize('fault', ['missing_ready', 'missing_map', 'missing_lance', 'missing_fts',
    'changed_hash', 'missing_pin', 'bad_global', 'bad_prompt', 'failed_metadata', 'extra_failure',
    'missing_shard', 'old_failure', 'live_receipt', 'live_status', 'existing_output', 'overlap'])
def test_preparation_refuses_unproven_state_before_creating_output(original, fault):
    source, destination, _, _, _, r17, old, r13 = original
    if fault in ('missing_ready', 'missing_map', 'missing_lance', 'missing_fts'):
        pattern = {'missing_ready': 'simplemem_ready.txt', 'missing_map': 'simplemem_source_map.json',
                   'missing_lance': '*.lance', 'missing_fts': 'meta.json'}[fault]
        target = next(p for p in (source / 'context_04').rglob(pattern) if p.is_file())
        target.unlink()
    elif fault == 'changed_hash':
        next(source.rglob('simplemem_ready.txt')).write_text('tampered')
    elif fault == 'missing_pin':
        r17['raw_source_sha256'].pop(str(next(source.rglob('simplemem_ready.txt')).resolve()))
    elif fault in ('bad_global', 'bad_prompt', 'failed_metadata', 'extra_failure'):
        path = source / 'context_04/artifacts/outputs/simplemem/native_results.json'
        raw = json.loads(path.read_text())
        row = raw['data'][97]
        if fault == 'bad_global':
            row['query_id'] = True
        elif fault == 'bad_prompt':
            row['query'] += ' changed'
        elif fault == 'failed_metadata':
            row['answer'] = 'changed'
        else:
            raw['data'][0].update(status='failed', output='')
        dump(path, raw)
        r17['raw_source_sha256'][str(path.resolve())] = sha(path)
    elif fault == 'missing_shard':
        next((source / 'context_03').rglob('*_results.json')).unlink()
    elif fault == 'old_failure':
        path = next(old.rglob('*_results.json'))
        raw = json.loads(path.read_text())
        raw['data'][0].update(status='failed', output='')
        dump(path, raw)
        r13['raw_source_sha256'][str(path.resolve())] = sha(path)
    elif fault == 'live_receipt':
        r17['original_attempt_closed'] = False
    elif fault == 'live_status':
        r17['parallel_status']['active_contexts'] = [4]
    elif fault == 'existing_output':
        destination.mkdir(parents=True)
    else:
        original = (source, source / 'unsafe-new-output', *original[2:])
    with pytest.raises((ValueError, FileNotFoundError, FileExistsError, KeyError)):
        prepare(original)
    if fault != 'existing_output':
        assert not destination.exists()


def test_native_retry_tracking_skips_exactly_1537_seed_rows(original):
    from utils.initialization import load_existing_results
    result = prepare(original)
    pairs = [[{} for _ in range(count)] for count in COUNTS]
    _, rows, completed, skipped = load_existing_results(result['results'], original[4], pairs, retry_failed_queries=True)
    assert len(rows) == len(skipped) == 1537
    assert completed == {0, 1, 2, 3, 5, 7, 8, 9}
    assert set(range(1540)) - skipped == {681, 930, 993}


@pytest.fixture
def driver(original, tmp_path, monkeypatch):
    target = importlib.import_module('scripts.simplemem_saved_recovery_r21')
    source, _, _, agent, config, r17, old, r13 = original
    tmp_path = source.parent.parent
    out, plan_path = tmp_path / 'run', tmp_path / 'plan.json'
    agent_path, config_path = tmp_path / 'agent.json', tmp_path / 'config.json'
    dump(agent_path, agent)
    dump(config_path, config)
    base = {'model': 'Qwen/Qwen3.5-9B', 'dtype': 'float16', 'model_revision': 'revision',
            'embedding_path': 'same-minilm', 'dataset': str(tmp_path / 'dataset.json'), 'scorer': 'same-scorer',
            'methods': [{'method': 'simplemem', 'agent_config': str(agent_path), 'dataset_config': str(config_path)}],
            'server_python': 'server-python', 'client_python': 'client-python'}
    saved, pins = {}, {}
    for tag, root, content, run in (('r17', source, r17, 'locomo-simplemem-failed-contexts-20260908-r17'),
                                   ('r13', old, r13, 'locomo-full-simplemem-validated-20260908-r13')):
        previous = tmp_path / f'{tag}-plan.json'
        record = tmp_path / f'{tag}-receipt.json'
        dump(previous, {**base, 'output': str(root.parent), 'run_id': run,
                        'file_sha256': {str(agent_path): sha(agent_path), str(config_path): sha(config_path)}})
        dump(record, content)
        saved.update({f'source_{tag}': str(root), f'plan_{tag}': str(previous), f'receipt_{tag}': str(record)})
        pins.update({str(previous): sha(previous), str(record): sha(record)})
    plan = {**base, 'output': str(out), 'run_id': 'locomo-simplemem-qa-repair-20260908-r21',
            'saved_simplemem': saved, 'file_sha256': pins}
    monkeypatch.setattr(target, 'PLAN_PATH', plan_path)
    monkeypatch.setattr(target, 'OUTPUT', out)
    dump(plan_path, plan)
    events, calls = [], []
    monkeypatch.setattr(target.queue, 'validate_plan', lambda _: events.append('pins'))
    monkeypatch.setattr(target.queue, 'supervisor_state', lambda _: 'EXITED')
    monkeypatch.setattr(target, 'running_writers', lambda _: [])
    monkeypatch.setattr(target, 'require_free_ports', lambda: events.append('ports'))
    monkeypatch.setattr(target.queue, 'wait_vllm', lambda _: events.append('vllm'))
    monkeypatch.setattr(target.queue, 'wait_health', lambda *_: None)
    monkeypatch.setattr(target.queue, 'gpu_sample', dict)
    monkeypatch.setattr(target.queue, 'telemetry', lambda *_: {'wall_seconds': 1})
    monkeypatch.setattr(target.queue, 'drain_proxies', lambda _: events.append('drained'))
    monkeypatch.setattr(target, 'inspect_saved_tables', lambda *_: events.append('saved-tables'))
    monkeypatch.setattr(target, 'inspect_native_seed', lambda *_: events.append('native-skip'))

    @contextmanager
    def service(command, log, env=None):
        calls.append((command, log, env))
        if '--retry_failed_queries' in command:
            artifacts = Path(command[command.index('--artifact_root') + 1])
            results = next(artifacts.rglob('*_results.json'))
            raw = json.loads(results.read_text())
            for row in raw['data']:
                if row.get('status') == 'failed':
                    row.pop('status')
                    row.pop('error')
                    row['output'] = 'repaired answer'
                    row['question_id'] = row['qa_pair_id']
            dump(results, raw)
            usage = []
            for qid, (context, _) in RETRIES.items():
                for kind in ('chat_completion', 'embedding'):
                    usage.append({'schema_version': 1, 'request_id': qid + kind, 'response_id': 'response-' + qid + kind,
                        'run_id': plan['run_id'], 'method': 'simplemem', 'sample_id': SAMPLES[context], 'phase': 'qa',
                        'question_id': qid.split('_qa')[-1], 'request_kind': kind, 'success': True,
                        'finish_reasons': ['stop'], 'usage_status': 'reported',
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 2 if kind == 'chat_completion' else 0,
                                  'total_tokens': 12 if kind == 'chat_completion' else 10}})
            folder = out / 'simplemem'
            for kind, filename in (('chat_completion', 'llm_usage.jsonl'), ('embedding', 'embedding_usage.jsonl')):
                (folder / filename).write_text('\n'.join(json.dumps(r) for r in usage if r['request_kind'] == kind) + '\n')
            (folder / 'timing.jsonl').write_text('')
        process = type('Process', (), {'returncode': 0, 'poll': lambda self: 0})()
        yield process

    monkeypatch.setattr(target.queue, 'service', service)
    return target, plan, events, calls, out


def test_driver_uses_native_three_query_retry_with_both_meters_and_no_final_claim(driver):
    target, plan, events, calls, out = driver
    assert target.main() == 0
    command, _, env = next(item for item in calls if '--retry_failed_queries' in item[0])
    assert '--force' not in command and '--fail-on-query-error' in command
    assert env['METER_RUN_ID'] == plan['run_id'] and env['METER_METHOD'] == 'simplemem'
    assert 'drained' in events and len(calls) == 4
    report = json.loads((out / 'simplemem/recovery_status.json').read_text())
    assert report['predictions_recovered'] is True and report['complete'] is False
    assert report['retry_ids'] == sorted(RETRIES) and report['new_memory_add_requests'] == 0
    assert report['usage']['reported_tokens']['total_tokens'] == 66
    assert len(json.loads((out / 'simplemem/recovered_queries.json').read_text())['data']) == 3


@pytest.mark.parametrize('fault', ['live_service', 'live_writer', 'unpin_receipt', 'changed_model', 'existing_output', 'prior_config_unpinned'])
def test_driver_gates_before_any_model_service(driver, monkeypatch, fault):
    target, plan, _, calls, out = driver
    if fault == 'live_service':
        monkeypatch.setattr(target.queue, 'supervisor_state', lambda _: 'RUNNING')
        monkeypatch.setattr(target.time, 'sleep', lambda _: (_ for _ in ()).throw(RuntimeError('wait only')))
    elif fault == 'live_writer':
        monkeypatch.setattr(target, 'running_writers', lambda _: [123])
    elif fault == 'unpin_receipt':
        del plan['file_sha256'][plan['saved_simplemem']['receipt_r17']]
        dump(target.PLAN_PATH, plan)
    elif fault == 'changed_model':
        plan['dtype'] = 'int8'
        dump(target.PLAN_PATH, plan)
    elif fault == 'prior_config_unpinned':
        path = Path(plan['saved_simplemem']['plan_r17'])
        previous = json.loads(path.read_text())
        previous['file_sha256'] = {}
        dump(path, previous)
        plan['file_sha256'][str(path)] = sha(path)
        dump(target.PLAN_PATH, plan)
    else:
        out.mkdir()
    with pytest.raises((RuntimeError, ValueError, FileExistsError, KeyError)):
        target.main()
    assert not calls


def test_r21_supervisor_identifiers_are_manual_start_and_never_power_actions():
    target = importlib.import_module('scripts.simplemem_saved_recovery_r21')
    assert target.PREDECESSORS == ('locomo-simplemem-failed-context-recovery-r17', 'locomo-amem-sampling-diagnostic-r20')
    text = (target.ROOT / 'scripts/simplemem_saved_recovery_r21.conf').read_text()
    assert '/workspace/MemoryData-simplemem-qa-repair-r21/scripts/simplemem_saved_recovery_r21.py' in text
    assert 'autostart=false' in text and 'autorestart=false' in text


@pytest.mark.parametrize('fault', ['failed_query', 'skipped_answer', 'unselected_qa', 'memory_add',
                                  'unknown_usage', 'missing_output', 'service_failure', 'invalid_usage_flag',
                                  'missing_embedding_journal', 'corrupt_embedding_journal'])
def test_attempt_failure_preserves_metered_artifacts_and_incomplete_receipt(driver, monkeypatch, fault):
    target, _, _, _, out = driver
    native_service = target.queue.service

    @contextmanager
    def service(command, log, env=None):
        with native_service(command, log, env) as process:
            if '--retry_failed_queries' in command:
                folder = out / 'simplemem'
                result = next((folder / 'artifacts').rglob('*_results.json'))
                if fault in ('failed_query', 'skipped_answer'):
                    raw = json.loads(result.read_text())
                    if fault == 'failed_query':
                        raw['data'][681].update(status='failed', output='')
                    else:
                        raw['data'][0]['output'] = 'changed skipped answer'
                    dump(result, raw)
                elif fault == 'missing_output':
                    result.unlink()
                elif fault == 'service_failure':
                    raise RuntimeError('mock native process failed after metering')
                elif fault == 'missing_embedding_journal':
                    (folder / 'embedding_usage.jsonl').unlink()
                elif fault == 'corrupt_embedding_journal':
                    (folder / 'embedding_usage.jsonl').write_text('{broken JSON\n')
                else:
                    journal = folder / 'llm_usage.jsonl'
                    rows = [json.loads(line) for line in journal.read_text().splitlines()]
                    if fault == 'memory_add':
                        rows[0]['phase'] = 'memory_add'
                    elif fault == 'unselected_qa':
                        rows[0]['question_id'] = '0'
                    elif fault == 'invalid_usage_flag':
                        rows[0]['usage_invalid'] = True
                    else:
                        rows[0]['usage'] = None
                    journal.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            yield process

    monkeypatch.setattr(target.queue, 'service', service)
    assert target.main() == 1
    report = json.loads((out / 'simplemem/recovery_status.json').read_text())
    assert report['predictions_recovered'] is False and report['complete'] is False
    assert (out / 'simplemem/usage.jsonl').is_file()
    if fault in ('unknown_usage', 'invalid_usage_flag', 'missing_embedding_journal', 'corrupt_embedding_journal'):
        assert report['usage']['exact_tokens'] is None


def test_copied_table_probe_uses_only_client_local_table_api(monkeypatch):
    target = importlib.import_module('scripts.simplemem_saved_recovery_r21')
    captured = []

    def execute(command, **kwargs):
        captured.append((command, kwargs))
        return type('Result', (), {'returncode': 0, 'stdout': 'copied_tables_verified\n', 'stderr': ''})()

    monkeypatch.setattr(target.subprocess, 'run', execute)
    preparation = {'memory_lineage': [{'destination_runtime': '/new/copied', 'table_name': 'locomo_qa_memory', 'entry_ids': ['one']}]}
    target.inspect_saved_tables('client-python', preparation)
    assert captured[0][0][:2] == ['client-python', '-c']
    assert 'create_table' not in captured[0][0][2]
    assert json.loads(captured[0][1]['input']) == preparation['memory_lineage']
    monkeypatch.setattr(target.subprocess, 'run', lambda *_args, **_kwargs:
                        type('Result', (), {'returncode': 1, 'stdout': '', 'stderr': 'missing table'})())
    with pytest.raises(ValueError, match='inspection failed'):
        target.inspect_saved_tables('client-python', preparation)


def test_table_validation_survives_python_optimize(monkeypatch):
    import contextlib
    import io
    import sys
    from types import SimpleNamespace

    target = importlib.import_module('scripts.simplemem_saved_recovery_r21')
    column = SimpleNamespace(to_pylist=list)
    arrow = SimpleNamespace(column=lambda _: column)
    table = SimpleNamespace(to_arrow=lambda: arrow, count_rows=lambda: 0)
    database = SimpleNamespace(table_names=lambda: ['locomo_qa_memory'], open_table=lambda _: table)
    monkeypatch.setitem(sys.modules, 'lancedb', SimpleNamespace(connect=lambda _: database))

    def execute(command, **kwargs):
        output = io.StringIO()
        monkeypatch.setattr(sys, 'stdin', io.StringIO(kwargs['input']))
        try:
            with contextlib.redirect_stdout(output):
                exec(compile(command[2], 'local_saved_probe', 'exec', optimize=1), {})  # noqa: S102 - trusted owned probe regression.
        except ValueError as error:
            return SimpleNamespace(returncode=1, stdout=output.getvalue(), stderr=str(error))
        return SimpleNamespace(returncode=0, stdout=output.getvalue(), stderr='')

    monkeypatch.setattr(target.subprocess, 'run', execute)
    with pytest.raises(ValueError, match='inspection failed'):
        target.inspect_saved_tables('client-python', {'memory_lineage': [
            {'destination_runtime': '/copy', 'table_name': 'locomo_qa_memory', 'entry_ids': ['one']}]})


def test_native_seed_preflight_uses_unchanged_client_loader_and_checks_fields(monkeypatch):
    from types import SimpleNamespace

    target = importlib.import_module('scripts.simplemem_saved_recovery_r21')
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='native_skip_verified\n', stderr='')

    monkeypatch.setattr(target.subprocess, 'run', execute)
    preparation = {'results': '/copied/seed.json', 'context_question_counts': COUNTS}
    target.inspect_native_seed('native-client', preparation, {'dataset': 'locomo'})
    code = calls[0][0][2]
    assert 'load_existing_results' in code and 'retry_failed_queries=True' in code
    assert 'assert ' not in code and 'preserved' in code
    assert json.loads(calls[0][1]['input'])['counts'] == COUNTS
    monkeypatch.setattr(target.subprocess, 'run', lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=1, stdout='', stderr='Native skip fields changed'))
    with pytest.raises(ValueError, match='skip inspection failed'):
        target.inspect_native_seed('native-client', preparation, {})
