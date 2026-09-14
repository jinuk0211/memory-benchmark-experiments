"""Offline SimpleMem finalization: exact native identities, closed inputs, no inference."""
import copy
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath

import pytest

from scripts.finalize_locomo_comparison import get_template
from scripts.score_locomo_comparison import read_jsonl

OLD = 'locomo-full-simplemem-validated-20260908-r13'
NEW = 'locomo-simplemem-failed-contexts-20260908-r17'
COUNTS = [152, 81, 152, 154, 154, 123, 154, 191, 221, 158]
SAMPLES = [f'conv-{c}' for c in range(10)]
REPAIR = 'locomo-simplemem-qa-repair-20260908-r21'
SELECTED = [0, 1, 2, 3, 4, 6, 7, 8, 9]


def put(manifest, key, value, *, lines=False, text=False):
    path = Path(manifest['_directory']) / key.lstrip('/')
    if len(str(path)) > 230:
        path = Path(manifest['_directory']) / '_long' / hashlib.sha256(key.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = value if text else ('\n'.join(json.dumps(r) for r in value) + '\n'
                                 if lines else json.dumps(value, ensure_ascii=False))
    path.write_text(content, encoding='utf-8')
    manifest['files'][key] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return key


def get(manifest, key, *, lines=False):
    path = Path(manifest['files'][key]['path'])
    return read_jsonl(path) if lines else json.loads(path.read_text(encoding='utf-8'))


def repin_receipts(manifest):
    names = ('old', 'new', 'repair') if 'repair' in manifest else ('old', 'new')
    for name in names:
        source = manifest[name]
        receipt = get(manifest, source['receipt'])
        receipt['raw_source_sha256'] = {key: value['sha256'] for key, value in manifest['files'].items()
                                        if key.startswith(source['root'] + '/')}
        if name == 'repair':
            receipt['recovery_status'] = get(manifest, source['root'] + '/recovery_status.json')
            receipt['preparation_sha256'] = manifest['files'][source['root'] + '/preparation.json']['sha256']
            receipt['plan_sha256'] = manifest['files'][source['plan']]['sha256']
        else:
            receipt['parallel_status'] = get(manifest, source['root'] + '/parallel_status.json')
        put(manifest, source['receipt'], receipt)
    inventories = {key: value for name in names
                   for key, value in get(manifest, manifest[name]['receipt'])['raw_source_sha256'].items()}
    runs = [OLD, NEW, REPAIR] if 'repair' in manifest else [OLD, NEW]
    put(manifest, '/closure.json', {'method': 'simplemem', 'run_ids': runs,
        'supervisor_states': dict.fromkeys(runs, 'EXITED'),
        'writers': {manifest[name]['root']: [] for name in names},
        'files_sha256_before': inventories, 'files_sha256_after': inventories})


@pytest.fixture
def bundle(tmp_path):
    manifest = {'schema_version': 1, '_directory': str(tmp_path / 'inputs'), 'files': {},
                'dataset': '/dataset.json', 'scorer': '/scorer.py', 'closure': '/closure.json'}
    dataset = [{'sample_id': SAMPLES[c], 'qa': [
        {'question': f'Question {c}/{q}?', 'answer': f'Answer {c}/{q}',
         'category': q % 4 + 1, 'evidence': [f'D{q}:1']} for q in range(count)]}
        for c, count in enumerate(COUNTS)]
    put(manifest, '/dataset.json', dataset)
    put(manifest, '/scorer.py', 'def eval_question_answering(rows, eval_key, metric):\n'
        '    return [float(row[eval_key] == row["answer"]) for row in rows], [], []\n', text=True)
    template = get_template('locomo_qa', 'query', 'Agentic_memory_simplemem')
    all_new = []
    for name, run, contexts in (('old', OLD, list(range(10))), ('new', NEW, SELECTED)):
        root = f'/workspace/{name}/simplemem'
        source = manifest[name] = {'root': root, 'plan': f'/{name}-plan.json', 'receipt': f'/{name}-receipt.json'}
        agent = put(manifest, f'/{name}-source/comparison_config/simplemem.yaml',
                    'agent_name: Agentic_memory_simplemem\nmodel: Qwen/Qwen3.5-9B\ntemperature: 0.0\n', text=True)
        config = put(manifest, f'/{name}-source/comparison_config/locomo.yaml', 'sub_dataset: locomo_qa\n', text=True)
        template_key = put(manifest, f'/{name}-source/benchmark/memoryagentbench/prompts/benchmark_templates.py',
                           Path(importlib.import_module(get_template.__module__).__file__).read_text(encoding='utf-8'), text=True)
        plan = {'run_id': run, 'model': 'Qwen/Qwen3.5-9B', 'model_revision': 'revision',
                'dtype': 'float16', 'embedding_path': '/models/all-MiniLM-L6-v2/revision',
                'dataset': '/dataset.json', 'scorer': '/scorer.py', 'output': f'/workspace/{name}',
                'hourly_rate_usd': 0.4, 'methods': [{'method': 'simplemem',
                'agent_config': agent, 'dataset_config': config}],
                'file_sha256': {key: manifest['files'][key]['sha256']
                               for key in (agent, config, template_key, '/dataset.json', '/scorer.py')}}
        put(manifest, source['plan'], plan)
        events = [{'context': c, 'event': 'start', 'pid': c + 100, 'time': c} for c in contexts]
        events += [{'context': c, 'event': 'end', 'exit_code': int(name == 'old' and c != 5),
                    'time': c + 100} for c in contexts]
        status = {'events': events, 'active_contexts': [], 'failed': name == 'old', 'complete': False}
        if name == 'new':
            status.update(selected_contexts_complete=True, context_indices=contexts)
            put(manifest, root + '/selected_attempt.json', {'run_id': run, 'method': 'simplemem',
                'requires_composite': True, 'complete': False, 'context_indices': contexts})
        put(manifest, root + '/parallel_status.json', status)
        llm, embeddings = [], []
        for c in contexts:
            put(manifest, root + f'/context_{c:02d}/worker.log', '', text=True)
            for phase in ('initialize', 'memory_add', 'memory_finalize'):
                llm.append(usage(f'{run}-{c}-{phase}', run, c, phase=phase))
            if name == 'old' and c not in (0, 1, 2, 5, 7, 9):
                continue
            rows = []
            for q, qa in enumerate(dataset[c]['qa']):
                failed_count = {0: 1, 1: 1, 2: 4, 7: 2, 9: 1}.get(c, 0)
                failed = name == 'old' and q >= COUNTS[c] - failed_count
                qid = f'{SAMPLES[c]}_qa{q}'
                row = {'query': template.format(question=qa['question']), 'answer': qa['answer'],
                       'output': '' if failed else (' old wrong \n' if name == 'old' else qa['answer']),
                       'sample_id': SAMPLES[c], 'qa_pair_id': qid,
                       'query_id': sum(COUNTS[:c]) + q, 'context_id': c,
                       'eval_metadata': {'dataset': 'locomo_qa', 'sample_id': SAMPLES[c],
                        'question_id': qid, 'qa_pair_id': qid, 'category': str(qa['category']),
                        'evidence': qa['evidence']}}
                if failed:
                    row.update(status='failed', error='Saved native failure')
                else:
                    row['question_id'] = qid
                rows.append(row)
                llm.append(usage(f'{run}-{c}-{q}', run, c, q))
                embeddings.append(usage(f'{run}-{c}-{q}-embed', run, c, q, kind='embedding'))
            result = put(manifest, root + f'/context_{c:02d}/artifacts/native_results.json', {'data': rows})
            if name == 'new':
                all_new.extend(rows)
            if name == 'new' or c == 5:
                put(manifest, root + f'/context_{c:02d}/completion.json',
                    {'context_index': c, 'sample_id': SAMPLES[c], 'questions': COUNTS[c], 'result_path': result})
        if name == 'new':
            put(manifest, root + '/selected_results.json', {'data': all_new})
        for filename, rows in (('llm_usage.jsonl', llm), ('embedding_usage.jsonl', embeddings),
                               ('usage.jsonl', llm + embeddings)):
            put(manifest, root + '/' + filename, rows, lines=True)
        put(manifest, root + '/telemetry.json', {'wall_seconds': 120, 'gpu_energy_wh': 50})
        put(manifest, source['receipt'], {'method': 'simplemem', 'run_id': run,
            'original_attempt_closed': True, 'parallel_status': status, 'raw_source_sha256': {}})
    repin_receipts(manifest)
    return manifest, tmp_path / 'output'


def usage(request_id, run, context, question=None, phase='qa', kind='chat_completion', **extra):
    return {'schema_version': 1, 'request_id': request_id, 'response_id': 'response-' + request_id,
            'run_id': run, 'method': 'simplemem', 'sample_id': SAMPLES[context], 'phase': phase,
            'question_id': question, 'request_kind': kind, 'model': 'Qwen/Qwen3.5-9B',
            'usage_status': 'reported', 'usage': {'prompt_tokens': 10, 'completion_tokens': 0 if kind == 'embedding' else 2,
                'total_tokens': 10 if kind == 'embedding' else 12}, 'success': True,
            'finish_reasons': [] if kind == 'embedding' else ['stop'], **extra}


def run(bundle):
    return importlib.import_module('scripts.finalize_simplemem_recovery').finalize(*bundle)


def test_exact_old_first_score_and_ledger_are_separate_and_inputs_unchanged(bundle):
    manifest, output = bundle
    original = copy.deepcopy(manifest)
    report = run(bundle)
    assert report['benchmark_complete'] is True
    assert report['complete'] is False and report['metrics_complete'] is False
    assert report['selected_questions'] == {'old': 848, 'new': 692}
    assert report['excluded_new_duplicate_questions'] == 725
    assert report['evaluated'] == 1540 and report['official_f1'] == pytest.approx(692 / 1540)
    assert report['gross_attempt_usage']['requests'] == 4605
    assert len(read_jsonl(output / 'gross_usage.jsonl')) == 4605
    selected, excluded = read_jsonl(output / 'selected_usage.jsonl'), read_jsonl(output / 'excluded_usage.jsonl')
    assert len(selected) + len(excluded) == 4605
    assert {r['run_id'] for r in selected if r['sample_id'] == 'conv-0' and r['phase'] == 'memory_add'} == {OLD, NEW}
    assert report['semantic_discards']['exact_tokens'] is None
    assert report['exclusive_gpu_energy_wh'] is None
    assert manifest == original
    for item in manifest['files'].values():
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']


@pytest.mark.parametrize('change', ['failed_answer', 'failed_global', 'failed_metadata', 'foreign',
                                  'duplicate', 'empty_new', 'missing_new', 'reordered', 'marker_escape'])
def test_raw_identity_validation_includes_failed_and_excluded_rows(bundle, change):
    manifest, _ = bundle
    root = manifest['old' if change.startswith('failed') else 'new']['root']
    key = root + '/context_00/artifacts/native_results.json'
    raw = get(manifest, key)
    if change == 'failed_answer':
        raw['data'][-1]['answer'] = 'changed'
    elif change == 'failed_global':
        raw['data'][-1]['query_id'] = True
    elif change == 'failed_metadata':
        raw['data'][-1]['eval_metadata']['qa_pair_id'] = 'foreign'
    elif change == 'foreign':
        raw['data'][0]['sample_id'] = 'foreign'
    elif change == 'duplicate':
        raw['data'][1] = copy.deepcopy(raw['data'][0])
    elif change == 'empty_new':
        raw['data'][0]['output'] = '  '
    elif change == 'missing_new':
        raw['data'].pop()
    elif change == 'reordered':
        raw['data'].reverse()
    else:
        marker_key = root + '/context_00/completion.json'
        marker = get(manifest, marker_key)
        marker['result_path'] = '/foreign/native_results.json'
        put(manifest, marker_key, marker)
    put(manifest, key, raw)
    repin_receipts(manifest)
    with pytest.raises(ValueError):
        run(bundle)


@pytest.mark.parametrize('change', ['foreign_excluded', 'duplicate_request', 'invalid_excluded',
                                  'missing_usage', 'wrong_combined', 'changed_plan', 'hash', 'missing_pin'])
def test_all_usage_and_source_pins_fail_closed_before_slicing(bundle, change):
    manifest, output = bundle
    root = manifest['new']['root']
    key = root + '/llm_usage.jsonl'
    rows = get(manifest, key, lines=True)
    if change in ('foreign_excluded', 'duplicate_request', 'invalid_excluded', 'missing_usage'):
        row = next(r for r in rows if r['phase'] == 'qa' and r['question_id'] == 0)
        if change == 'foreign_excluded':
            row['run_id'] = 'diagnostic-run'
        elif change == 'duplicate_request':
            row['request_id'] = rows[0]['request_id']
        elif change == 'invalid_excluded':
            row['usage']['prompt_tokens'] = True
        else:
            row['usage'] = None
        put(manifest, key, rows, lines=True)
        put(manifest, root + '/usage.jsonl', rows + get(manifest, root + '/embedding_usage.jsonl', lines=True), lines=True)
    elif change == 'wrong_combined':
        put(manifest, root + '/usage.jsonl', rows, lines=True)
    elif change == 'changed_plan':
        plan = get(manifest, manifest['new']['plan'])
        plan['dtype'] = 'int8'
        put(manifest, manifest['new']['plan'], plan)
    elif change == 'hash':
        manifest['files'][key]['sha256'] = '0' * 64
    else:
        del manifest['files'][key]
    if change not in ('hash', 'missing_pin'):
        repin_receipts(manifest)
    if change in ('invalid_excluded', 'missing_usage'):
        report = run(bundle)
        assert report['usage_complete'] is False and report['benchmark_complete'] is False
        assert report['gross_attempt_usage']['exact_tokens'] is None
        assert report['official_f1'] == pytest.approx(692 / 1540)
    else:
        with pytest.raises((ValueError, KeyError)):
            run(bundle)
        assert not (output / 'report.json').exists()


@pytest.mark.parametrize('change', ['missing', 'writers', 'changed_inventory', 'running', 'receipt_open'])
def test_missing_or_open_closure_proof_never_claims_completion(bundle, change):
    manifest, _ = bundle
    proof = get(manifest, manifest['closure'])
    if change == 'missing':
        manifest.pop('closure')
    elif change == 'writers':
        proof['writers'][manifest['new']['root']] = [123]
    elif change == 'changed_inventory':
        proof['files_sha256_after'] = {}
    elif change == 'running':
        proof['supervisor_states'][NEW] = 'RUNNING'
    else:
        key = manifest['new']['receipt']
        receipt = get(manifest, key)
        receipt['original_attempt_closed'] = False
        put(manifest, key, receipt)
    put(manifest, '/closure.json', proof)
    report = run(bundle)
    assert report['closed_sources_complete'] is False
    assert report['benchmark_complete'] is False and report['complete'] is False


def test_semantic_rejections_are_joined_but_never_added_again(bundle):
    manifest, output = bundle
    root = manifest['new']['root']
    row = get(manifest, root + '/llm_usage.jsonl', lines=True)[0]
    event = {'response_id': row['response_id'], 'attempt': 0, 'retry_scheduled': True,
             'error': 'not an object', 'rejected_content': '["part A\u2028part B"]'}
    put(manifest, root + '/context_00/worker.log', 'SIMPLEMEM_JSON_REPAIR ' + json.dumps(event, ensure_ascii=False) + '\n', text=True)
    repin_receipts(manifest)
    report = run(bundle)
    audit = report['semantic_discards']
    assert audit['event_count'] == 1 and audit['recorded_tokens']['total_tokens'] == 12
    assert audit['events'][0]['event'] == event and audit['exact_tokens'] is None
    assert sum(r['usage']['total_tokens'] for r in read_jsonl(output / 'gross_usage.jsonl')) == report['gross_attempt_usage']['reported_tokens']['total_tokens']


def test_fresh_destination_and_cli_manifest_pin(bundle, capsys):
    target = importlib.import_module('scripts.finalize_simplemem_recovery')
    manifest, output = bundle
    path = Path(manifest['_directory']).parent / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='manifest'):
        target.main(['--manifest', str(path), '--expected-manifest-sha256', '0' * 64, '--output', str(output)])
    code = target.main(['--manifest', str(path), '--expected-manifest-sha256', hashlib.sha256(path.read_bytes()).hexdigest(), '--output', str(output)])
    assert code == 1 and 'benchmark_complete' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        run(bundle)


def test_native_semantic_event_cannot_be_attributed_to_another_context(bundle):
    manifest, _ = bundle
    root = manifest['new']['root']
    row = get(manifest, root + '/llm_usage.jsonl', lines=True)[0]
    event = {'response_id': row['response_id'], 'attempt': 0, 'retry_scheduled': True,
             'error': 'not an object', 'rejected_content': '[]'}
    put(manifest, root + '/context_01/worker.log', 'SIMPLEMEM_JSON_REPAIR ' + json.dumps(event), text=True)
    repin_receipts(manifest)
    report = run(bundle)
    assert report['semantic_discards']['recorded_events_complete'] is False
    assert report['benchmark_complete'] is False


def test_output_cannot_be_nested_under_input_attempt(bundle):
    manifest, _ = bundle
    root = Path(manifest['files'][manifest['old']['root'] + '/usage.jsonl']['path']).parent
    with pytest.raises(ValueError, match='overlap'):
        run((manifest, root / 'new-evaluation'))


def test_unknown_raw_status_is_not_a_success(bundle):
    manifest, _ = bundle
    key = manifest['old']['root'] + '/context_00/artifacts/native_results.json'
    raw = get(manifest, key)
    raw['data'][0]['status'] = 'cancelled'
    put(manifest, key, raw)
    repin_receipts(manifest)
    with pytest.raises(ValueError, match='status'):
        run(bundle)


def test_repair_attempts_and_embeddings_remain_in_chosen_ledger(bundle):
    manifest, output = bundle
    root = manifest['new']['root']
    llm = get(manifest, root + '/llm_usage.jsonl', lines=True)
    terminal = next(r for r in llm if r['sample_id'] == 'conv-0' and r['question_id'] == 151)
    terminal.update(repair_group_id='repair-group', repair_attempt=2, delivered_to_client=True)
    for index in range(2):
        llm.insert(0, usage(f'repair-{index}', NEW, 0, 151, repair_group_id='repair-group',
                           repair_attempt=index, finish_reasons=['length'], success=False,
                           delivered_to_client=False, retry_scheduled=True))
    put(manifest, root + '/llm_usage.jsonl', llm, lines=True)
    put(manifest, root + '/usage.jsonl', llm + get(manifest, root + '/embedding_usage.jsonl', lines=True), lines=True)
    repin_receipts(manifest)
    report = run(bundle)
    chosen = read_jsonl(output / 'selected_usage.jsonl')
    assert len([r for r in chosen if r.get('repair_group_id') == 'repair-group']) == 3
    assert report['response_delivery']['repaired_requests'] == 1
    assert report['response_delivery']['discarded_attempt_total_tokens'] == 24
    assert report['benchmark_complete'] is True
    assert report['gross_attempt_usage']['requests'] == 4607


@pytest.mark.parametrize('change', ['selected_failed', 'terminal_missing', 'receipt_foreign', 'missing_generation'])
def test_selected_attempt_and_generation_coverage_gates(bundle, change):
    manifest, _ = bundle
    root = manifest['new']['root']
    if change in ('selected_failed', 'terminal_missing'):
        key = root + '/parallel_status.json'
        status = get(manifest, key)
        if change == 'selected_failed':
            status['failed'] = True
        else:
            status['events'].pop()
        put(manifest, key, status)
        repin_receipts(manifest)
    elif change == 'receipt_foreign':
        key = manifest['new']['receipt']
        receipt = get(manifest, key)
        receipt['run_id'] = 'other'
        put(manifest, key, receipt)
    else:
        rows = get(manifest, root + '/llm_usage.jsonl', lines=True)
        rows = [r for r in rows if not (r['sample_id'] == 'conv-0' and r['question_id'] == 151)]
        put(manifest, root + '/llm_usage.jsonl', rows, lines=True)
        put(manifest, root + '/usage.jsonl', rows + get(manifest, root + '/embedding_usage.jsonl', lines=True), lines=True)
        repin_receipts(manifest)
    if change == 'missing_generation':
        report = run(bundle)
        assert report['coverage_complete'] is False and report['benchmark_complete'] is False
    else:
        with pytest.raises(ValueError):
            run(bundle)


@pytest.fixture
def bundle_three(tmp_path_factory, monkeypatch):
    from scripts.prepare_saved_simplemem_retry import RETRIES
    from scripts.simplemem_saved_recovery_r21 import PRESERVED_FIELDS
    from utils.artifact_paths import generate_agent_save_folder_path

    monkeypatch.setattr(__import__(__name__), 'COUNTS', [152, 81, 152, 199, 178, 123, 150, 191, 156, 158])
    monkeypatch.setattr(__import__(__name__), 'SAMPLES',
                        ['conv-26', 'conv-30', 'conv-41', 'conv-42', 'conv-43',
                         'conv-44', 'conv-47', 'conv-48', 'conv-49', 'conv-50'])
    manifest, output = bundle.__wrapped__(tmp_path_factory.mktemp('s3'))
    new = manifest['new']['root']
    repair = '/workspace/repair/simplemem'
    seed, original_keys, lineage, copies = [], [], [], {}
    for c in range(10):
        root = manifest['old']['root'] if c == 5 else new
        key = root + f'/context_{c:02d}/artifacts/native_results.json'
        raw = get(manifest, key)
        for row in raw['data']:
            if row['qa_pair_id'] in RETRIES:
                row.update(status='failed', output='', error='saved SQL/FTS failure')
                row.pop('question_id')
        put(manifest, key, raw)
        original_keys.append(key)
        seed.extend(raw['data'])
    status = get(manifest, new + '/parallel_status.json')
    status.update(failed=True, selected_contexts_complete=False)
    # Actual failed r17 status has terminal events but no success-only scope keys.
    status.pop('context_indices')
    status.pop('selected_contexts_complete')
    for row in status['events']:
        if row['event'] == 'end' and row['context'] in (4, 6):
            row['exit_code'] = 1
    put(manifest, new + '/parallel_status.json', status)
    for suffix in ('selected_results.json', 'selected_attempt.json', 'context_04/completion.json', 'context_06/completion.json'):
        del manifest['files'][new + '/' + suffix]
    agent = {'agent_name': 'Agentic_memory_simplemem', 'model': 'Qwen/Qwen3.5-9B', 'temperature': 0.0}
    for c in (4, 6):
        paths = [PurePosixPath(generate_agent_save_folder_path(agent, {'sub_dataset': 'locomo_qa'}, c, root).replace('\\', '/'))
                 for root in (new + f'/context_{c:02d}/artifacts', repair + '/artifacts')]
        runtimes = [p.parent.parent / '_simplemem_runtime' / hashlib.sha256(str(p).encode()).hexdigest()[:16] for p in paths]
        for source_path, destination_path, files in (
            (paths[0], paths[1], {'simplemem_ready.txt': 'ready', 'simplemem_source_map.json': '{"entry": ["D1:1"]}'}),
            (runtimes[0], runtimes[1], {'locomo_qa_memory.lance/_versions/1.manifest': 'version',
             'locomo_qa_memory.lance/data/1.lance': 'data', 'locomo_qa_memory.lance/_indices/fts/meta.json': '{}',
             'locomo_qa_memory.lance/_indices/fts/1.idx': 'index'})):
            for relative, content in files.items():
                original = put(manifest, str(source_path / relative), content, text=True)
                target = put(manifest, str(destination_path / relative), content, text=True)
                original_keys.append(original)
                copies[target] = manifest['files'][original]['sha256']
        lineage.append({'context_id': c, 'sample_id': SAMPLES[c], 'run_id': NEW,
            'source_agent': str(paths[0]), 'destination_agent': str(paths[1]),
            'source_runtime': str(runtimes[0]), 'destination_runtime': str(runtimes[1]),
            'table_name': 'locomo_qa_memory', 'entry_ids': ['entry']})
    repin_receipts(manifest)
    manifest.update(mode='r13_r17_r21', repair={'root': repair, 'plan': '/repair-plan.json', 'receipt': '/repair-receipt.json'})
    plan = get(manifest, manifest['new']['plan'])
    plan.update(run_id=REPAIR, output='/workspace/repair', saved_simplemem={})
    for name, tag in (('old', 'r13'), ('new', 'r17')):
        source = manifest[name]
        original_keys.append(source['root'] + '/parallel_status.json')
        plan['saved_simplemem']['source_' + tag] = source['root']
        for kind in ('plan', 'receipt'):
            plan['saved_simplemem'][kind + '_' + tag] = source[kind]
            plan['file_sha256'][source[kind]] = manifest['files'][source[kind]]['sha256']
            original_keys.append(source[kind])
    put(manifest, '/repair-plan.json', plan)
    result_path = repair + '/artifacts/native_results.json'
    preparation = {'rows': 1540, 'retry_ids': sorted(RETRIES), 'skipped_seed_rows': 1537,
        'context_question_counts': COUNTS, 'results': result_path, 'memory_lineage': lineage,
        'source_sha256': {key: manifest['files'][key]['sha256'] for key in original_keys},
        'copy_sha256': copies, 'seed_is_final_selection': False, 'copied_tables_verified': True,
        'native_skip_probe': {'passed': True, 'skipped_queries': 1537, 'retry_global_ids': [681, 930, 993],
                             'unchanged_inference_fields': list(PRESERVED_FIELDS)}}
    put(manifest, repair + '/preparation.json', preparation)
    recovered = []
    for row in seed:
        if row['qa_pair_id'] in RETRIES:
            row.pop('status')
            row.pop('error')
            row.update(output=row['answer'], question_id=row['qa_pair_id'])
            recovered.append(row)
    put(manifest, result_path, {'data': seed})
    put(manifest, repair + '/recovered_queries.json', {'data': recovered})
    llm = [usage('r21-' + qid, REPAIR, c, int(qid.split('_qa')[1])) for qid, (c, _) in RETRIES.items()]
    embeddings = [usage('r21-embed-' + qid, REPAIR, c, int(qid.split('_qa')[1]), kind='embedding') for qid, (c, _) in RETRIES.items()]
    for filename, rows in (('llm_usage.jsonl', llm), ('embedding_usage.jsonl', embeddings), ('usage.jsonl', llm + embeddings)):
        put(manifest, repair + '/' + filename, rows, lines=True)
    recovery = {'run_id': REPAIR, 'predictions_recovered': True, 'complete': False,
        'requires_composite_audit': True, 'retry_ids': sorted(RETRIES), 'seed_is_final_selection': False,
        'source_artifacts_unchanged': True, 'usage': {'complete': True}, 'delivery': {'complete': True},
        'new_memory_add_requests': 0, 'error': None, 'issues': [], 'exit_code': 0}
    put(manifest, repair + '/recovery_status.json', recovery)
    put(manifest, repair + '/benchmark.log', '', text=True)
    put(manifest, repair + '/telemetry.json', {'wall_seconds': 45, 'gpu_energy_wh': 1})
    put(manifest, '/repair-receipt.json', {'method': 'simplemem', 'run_id': REPAIR,
        'original_attempt_closed': True, 'plan_path': '/repair-plan.json'})
    repin_receipts(manifest)
    return manifest, output


def test_three_closed_runs_preserve_old848_new689_repair3_and_gross(bundle_three):
    manifest, output = bundle_three
    before = copy.deepcopy(manifest)
    report = run(bundle_three)
    assert report['benchmark_complete'] is True and report['complete'] is False
    assert report['selected_questions'] == {'old': 848, 'new': 689, 'repair': 3}
    assert report['run_ids'] == [OLD, NEW, REPAIR]
    assert report['official_f1'] == pytest.approx(692 / 1540)
    assert report['excluded_new_duplicate_questions'] == 725
    assert report['gross_attempt_usage']['requests'] == 4611
    chosen = read_jsonl(output / 'selected_usage.jsonl')
    assert len([row for row in chosen if row['run_id'] == REPAIR]) == 6
    assert len([row for row in chosen if row['run_id'] == NEW and row['sample_id'] == 'conv-43' and row['phase'] == 'memory_add']) == 1
    assert not any(row['run_id'] == REPAIR and row['phase'] == 'memory_add' for row in chosen)
    assert report['memory_reuse']['construction_run_id'] == NEW
    assert report['semantic_discards']['exact_tokens'] is None
    assert manifest == before


@pytest.mark.parametrize('change', ['implicit_mode', 'wrong_mode', 'wrong_run', 'fake_selected',
    'wrong_failed_id', 'changed_skip', 'missing_repair', 'changed_recovered', 'open_repair',
    'missing_source_pin', 'wrong_copy_hash', 'wrong_runtime', 'empty_sidecar', 'missing_probe',
    'missing_embedding', 'foreign_qa', 'new_construction', 'receipt_status', 'plan_lineage'])
def test_three_run_provenance_and_native_skip_gates(bundle_three, change):
    manifest, _ = bundle_three
    root = manifest['repair']['root']
    key = root + '/preparation.json'
    value = get(manifest, key)
    if change == 'implicit_mode':
        manifest.pop('mode')
    elif change == 'wrong_mode':
        manifest['mode'] = 'auto'
    elif change == 'wrong_run':
        key = manifest['repair']['plan']
        value = get(manifest, key)
        value['run_id'] = NEW
    elif change == 'fake_selected':
        key = manifest['new']['root'] + '/selected_results.json'
        value = {'data': []}
    elif change == 'wrong_failed_id':
        key = manifest['new']['root'] + '/context_04/artifacts/native_results.json'
        value = get(manifest, key)
        value['data'][97]['status'] = None
    elif change in ('changed_skip', 'missing_repair'):
        key = value['results']
        value = get(manifest, key)
        if change == 'changed_skip':
            value['data'][0]['output'] = 'better answer must not replace skipped seed'
        else:
            value['data'].pop()
    elif change == 'changed_recovered':
        key = root + '/recovered_queries.json'
        value = get(manifest, key)
        value['data'][0]['output'] = 'does not match native post-seed'
    elif change == 'open_repair':
        key = root + '/recovery_status.json'
        value = get(manifest, key)
        value['predictions_recovered'] = False
    elif change == 'missing_source_pin':
        value['source_sha256'].pop(next(iter(value['source_sha256'])))
    elif change == 'wrong_copy_hash':
        value['copy_sha256'][next(iter(value['copy_sha256']))] = '0' * 64
    elif change == 'wrong_runtime':
        value['memory_lineage'][0]['destination_runtime'] += 'wrong'
    elif change == 'empty_sidecar':
        value['memory_lineage'][0]['entry_ids'] = []
    elif change == 'missing_probe':
        value['native_skip_probe']['passed'] = False
    elif change == 'missing_embedding':
        del manifest['files'][root + '/embedding_usage.jsonl']
    elif change in ('foreign_qa', 'new_construction'):
        key = root + '/llm_usage.jsonl'
        value = get(manifest, key, lines=True)
        value[0]['question_id' if change == 'foreign_qa' else 'phase'] = 0 if change == 'foreign_qa' else 'memory_add'
        put(manifest, key, value, lines=True)
        put(manifest, root + '/usage.jsonl', value + get(manifest, root + '/embedding_usage.jsonl', lines=True), lines=True)
    elif change == 'receipt_status':
        key = manifest['repair']['receipt']
        value = get(manifest, key)
        value['recovery_status']['exit_code'] = 9
    else:
        key = manifest['repair']['plan']
        value = get(manifest, key)
        value['saved_simplemem']['source_r17'] = manifest['old']['root']
    if change not in ('foreign_qa', 'new_construction', 'missing_embedding', 'implicit_mode', 'wrong_mode'):
        put(manifest, key, value)
    if change != 'receipt_status':
        repin_receipts(manifest)
    with pytest.raises((ValueError, KeyError)):
        run(bundle_three)


def test_three_run_missing_closure_preserves_score_but_not_completion(bundle_three):
    manifest, _ = bundle_three
    manifest.pop('closure')
    report = run(bundle_three)
    assert report['predictions_complete'] is True and report['official_f1'] == pytest.approx(692 / 1540)
    assert report['benchmark_complete'] is False and report['closed_sources_complete'] is False
