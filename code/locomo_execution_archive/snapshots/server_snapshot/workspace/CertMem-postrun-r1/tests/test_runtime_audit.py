"""Pure offline contract fixtures; no model, service or network invocation."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/audit_certmem_runtime.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8')
    return sha(path)


@pytest.fixture
def target():
    spec = importlib.util.spec_from_file_location('postrun_audit', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(record):
    record.receipt['request_sha256'] = record.receipt['request_hash'] = put(record.paths['request'], record.request)
    if record.response is not None:
        record.receipt['response_sha256'] = put(record.paths['response'], record.response)
    put(record.paths['receipt'], record.receipt)


def generation(b, context, count, cap):
    sequence = len(b.gen) + 1
    request = {'model': b.t.MODEL, 'model_revision': b.t.REVISION, 'context': context,
        'sampling': {'temperature': 0.0, 'max_tokens': cap, 'n': 1, 'truncate_prompt_tokens': None},
        'prompts': [f'Original {sequence}-{i} \u2028 원문' for i in range(count)]}
    response = {'responses': [{'request_id': f'{sequence}-{i}', 'prompt': prompt,
        'prompt_token_ids': [10, 11, 12], 'num_cached_tokens': 2, 'finished': True,
        'outputs': [{'index': 0, 'text': ' answer ', 'token_ids': [20, 21],
                     'finish_reason': 'length' if i == 0 else 'stop'}]} for i, prompt in enumerate(request['prompts'])]}
    receipt = {'type': 'generation_batch', 'batch_sequence': sequence, 'context': context.copy(),
        'status': 'success', 'usage_complete': True, 'request_count': count, 'response_count': count,
        'generation_count': count, 'prompt_tokens': count * 3, 'completion_tokens': count * 2,
        'total_tokens': count * 5, 'preflight_prompt_tokens': [3] * count,
        'finish_reasons': [r['outputs'][0]['finish_reason'] for r in response['responses']],
        'duration_s': 1.0, 'inference_duration_s': 0.9}
    paths = {suffix: b.out / f'runtime/generation/{sequence:08d}.{suffix}.json' for suffix in ('request', 'response', 'receipt')}
    record = SimpleNamespace(request=request, response=response, receipt=receipt, paths=paths)
    save(record)
    b.gen.append(record)


def embedding(b, context):
    sequence = len(b.emb) + 1
    counts = {'input_tokens': 7, 'padded_tokens': 8, 'tokens_per_input': [3, 4], 'shape': [2, 4]}
    request = {'model': b.t.EMBED_MODEL, 'model_revision': b.t.EMBED_REVISION,
               'native_max_seq_length': 256, 'context': context, **counts}
    receipt = {'type': 'embedding_forward', 'batch_sequence': sequence, 'context': context.copy(),
               'status': 'success', 'usage_complete': True, 'duration_s': 0.5, **counts}
    paths = {suffix: b.out / f'runtime/embedding/{sequence:08d}.{suffix}.json' for suffix in ('request', 'receipt')}
    record = SimpleNamespace(request=request, response=None, receipt=receipt, paths=paths)
    save(record)
    b.emb.append(record)


def close(b):
    b.closure['raw_source_sha256'] = {p.relative_to(b.out).as_posix(): sha(p) for p in b.out.rglob('*') if p.is_file()}
    put(b.cp, b.closure)


def build(tmp_path, monkeypatch, target, counts):
    b = SimpleNamespace(t=target, out=tmp_path / 'output', source=tmp_path / 'source',
                        mp=tmp_path / 'manifest.json', cp=tmp_path / 'closure.json', gen=[], emb=[])
    for name in target.REQUIRED_SOURCE:
        path = b.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# pinned source\n', encoding='utf-8')
    dataset = b.source / 'data/locomo10.json'
    data = [{'sample_id': cid, 'qa': [{'category': 1, 'question': f'Q{i}', 'answer': [i, 'answer']} for i in range(n)]} for cid, n in counts.items()]
    put(dataset, data)
    monkeypatch.setattr(target, 'DATASET_SHA256', sha(dataset))
    monkeypatch.setattr(target, 'QA_TOTAL', sum(counts.values()))
    monkeypatch.setattr(target, 'CONV_TOTAL', len(counts))
    manifest = {'schema_version': 1, 'run_id': target.RUN_ID, 'output': str(b.out), 'source_root': str(b.source),
        'dataset': str(dataset), 'files_sha256': {str(p): sha(p) for p in b.source.rglob('*') if p.is_file()}}
    put(b.mp, manifest)
    rows = []
    for cid, count in counts.items():
        generation(b, {'phase': 'fact_extraction', 'conv_id': cid}, 1, 700)
        for config, policies in target.POLICIES.items():
            generation(b, {'phase': 'memory_build', 'conv_id': cid, 'config': config}, 1, 512)
            generation(b, {'phase': 'query_reform', 'conv_id': cid, 'config': config}, count, 60)
            embedding(b, {'phase': 'retrieval', 'conv_id': cid, 'config': config})
            items = [{'policy': policy, 'question_ordinal': i} for i in range(count) for policy in policies]
            generation(b, {'phase': 'qa', 'conv_id': cid, 'config': config, 'items': items}, len(items), 32)
            rows.extend({'conv_id': cid, 'config': config, **item, 'category': 1, 'question': f'Q{item["question_ordinal"]}',
                         'gold': str([item['question_ordinal'], 'answer']), 'pred': 'answer'} for item in items)
    with (b.out / 'items.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    put(b.out / 'receipt.json', {'run_id': target.RUN_ID, 'manifest_sha256': sha(b.mp), 'source_unchanged': True,
        'inference_complete': True, 'child_exit_code': 0, 'runtime': {'sampled_device_energy_wh': 123, 'estimated_rental_usd': 1.25}})
    b.closure = {'schema_version': 1, 'run_id': target.RUN_ID, 'output': str(b.out), 'supervisor_state': 'EXITED',
                 'writer_pids': [], 'child_exit_code': 0}
    close(b)
    return b


@pytest.fixture
def bundle(tmp_path, monkeypatch, target):
    return build(tmp_path, monkeypatch, target, {'conv-26': 2})


def audit(b):
    return b.t.audit_runtime(b.out, b.mp, sha(b.mp), b.cp, sha(b.cp))


def test_gross_policy_and_native_embedding_no_shared_construction_multiplication(bundle):
    b = bundle
    before = {str(p): p.read_bytes() for p in b.out.rglob('*') if p.is_file()}
    result = audit(b)
    assert result['complete'] is True
    assert result['generation']['exact_tokens'] == {'prompt_tokens': 252, 'completion_tokens': 168, 'total_tokens': 420}
    assert result['generation']['length_responses'] == 10
    assert result['generation']['native_generate_batch_count'] == 10
    assert result['generation']['recorded_batch_attempt_count'] == 10
    assert result['generation']['cached_prompt_tokens']['exact'] == 168
    assert result['embedding']['exact_input_tokens'] == 21 and result['embedding']['recorded_padded_tokens'] == 24
    assert result['by_phase_config']['fact_extraction/shared']['generation']['reported_tokens']['total_tokens'] == 5
    assert sum(v['generation']['reported_tokens']['total_tokens'] for v in result['by_phase_config'].values()) == 420
    assert len(result['qa_by_policy']) == 37
    naive = result['qa_by_policy']['full/full_raw']
    assert naive['generation']['exact_tokens']['total_tokens'] == 10
    assert naive['causal_embedding_tokens'] == naive['causal_construction_tokens'] == 0
    assert naive['exclusive_gpu_energy_wh'] is None
    assert len(result['generation_ledger']) == 84 and len(result['embedding_ledger']) == 3
    assert before == {str(p): p.read_bytes() for p in b.out.rglob('*') if p.is_file()}


def test_actual1540_scope_requires56980_distinct_qa(tmp_path, monkeypatch, target):
    b = build(tmp_path, monkeypatch, target, {f'conv-{i}': 154 for i in range(10)})
    result = audit(b)
    assert result['complete'] is True and result['qa_coverage']['count'] == 56980
    assert all(v['response_count'] == 1540 for v in result['qa_by_policy'].values())


def test_large_batch_artifact_hash_is_not_repeated_per_response(bundle, monkeypatch):
    from collections import Counter
    calls, original = Counter(), bundle.t.digest
    def counted(path):
        calls[str(path)] += 1
        return original(path)
    monkeypatch.setattr(bundle.t, 'digest', counted)
    assert audit(bundle)['complete'] is True
    assert max(calls.values()) <= 8


@pytest.mark.parametrize('bad', ['pending', 'no_response', 'unknown_tokens', 'failed_known', 'wrong_total',
    'response_hash', 'request_hash', 'duplicate_id', 'prompt_mismatch', 'missing_candidate', 'candidate_index',
    'context_mismatch', 'foreign_phase', 'foreign_conv', 'foreign_policy', 'duplicate_qa', 'wrong_model',
    'wrong_revision', 'wrong_cap', 'truncation', 'sequence_gap', 'partial_file', 'duration', 'missing_items', 'csv_partial'])
def test_bad_generation_fails_closed_preserving_known_costs(bundle, bad):
    b, r = bundle, bundle.gen[-1]
    direct = bad in ('pending', 'no_response', 'sequence_gap', 'partial_file', 'csv_partial')
    if bad == 'pending': r.paths['receipt'].unlink()
    elif bad == 'no_response': r.paths['response'].unlink()
    elif bad == 'unknown_tokens': r.response['responses'][0]['outputs'][0]['token_ids'] = None
    elif bad == 'failed_known': r.receipt.update(status='error', usage_complete=False)
    elif bad == 'wrong_total': r.receipt['total_tokens'] = 999
    elif bad == 'duplicate_id': r.response['responses'][0]['request_id'] = '1-0'
    elif bad == 'prompt_mismatch': r.response['responses'][0]['prompt'] = 'other'
    elif bad == 'missing_candidate': r.response['responses'][0]['outputs'] = []
    elif bad == 'candidate_index': r.response['responses'][0]['outputs'][0]['index'] = 1
    elif bad == 'context_mismatch': r.receipt['context'] = {}
    elif bad == 'foreign_phase': r.request['context']['phase'] = 'other'
    elif bad == 'foreign_conv': r.request['context']['conv_id'] = 'other'
    elif bad == 'foreign_policy': r.request['context']['items'][0]['policy'] = 'other'
    elif bad == 'duplicate_qa': r.request['context']['items'][-1] = r.request['context']['items'][0]
    elif bad == 'wrong_model': r.request['model'] = 'other'
    elif bad == 'wrong_revision': r.request['model_revision'] = 'other'
    elif bad == 'wrong_cap': r.request['sampling']['max_tokens'] = 31
    elif bad == 'truncation': r.request['sampling']['truncate_prompt_tokens'] = 10
    elif bad == 'duration': r.receipt['duration_s'] = -1
    elif bad == 'missing_items': r.request['context'].pop('items')
    elif bad == 'csv_partial': (b.out / 'items.csv').write_text('conv_id,question_ordinal\nconv-26\n')
    elif bad == 'partial_file': (b.out / 'runtime/generation/unclosed.json.partial').write_text('{}')
    elif bad == 'sequence_gap':
        for p in b.gen[0].paths.values(): p.rename(p.with_name(p.name.replace('00000001', '00000099')))
    if not direct:
        save(r)
        if bad in ('response_hash', 'request_hash'):
            r.receipt['response_sha256' if bad == 'response_hash' else 'request_hash'] = '0' * 64
            put(r.paths['receipt'], r.receipt)
    close(b)
    result = audit(b)
    assert result['complete'] is False and result['issues']
    assert result['generation']['reported_tokens']['total_tokens'] > 0
    if bad == 'unknown_tokens':
        assert result['generation']['exact_tokens'] is None
        assert result['generation']['reported_tokens']['completion_tokens'] == 166
    if bad == 'failed_known': assert result['generation']['reported_tokens']['total_tokens'] == 420


@pytest.mark.parametrize('bad', ['pending', 'wrong_model', 'width', 'mask_sum', 'padded', 'counts_mismatch', 'bool_count', 'failed'])
def test_embedding_masks_are_actual_unpadded256_and_unknown_not_zero(bundle, bad):
    b, r = bundle, bundle.emb[-1]
    if bad == 'pending': r.paths['receipt'].unlink()
    else:
        if bad == 'wrong_model': r.request['model'] = 'other'
        elif bad == 'width': r.request['shape'] = [2, 257]
        elif bad == 'mask_sum': r.request['input_tokens'] = 8
        elif bad == 'padded': r.request['padded_tokens'] = 7
        elif bad == 'counts_mismatch': r.receipt['tokens_per_input'] = [4, 3]
        elif bad == 'bool_count': r.request['input_tokens'] = True
        elif bad == 'failed': r.receipt.update(status='error', usage_complete=False)
        save(r)
    close(b)
    result = audit(b)
    assert result['complete'] is False and result['embedding']['exact_input_tokens'] is None
    if bad in ('failed', 'wrong_model'): assert result['embedding']['recorded_input_tokens'] == 21


@pytest.mark.parametrize('bad', ['no_closure', 'writers', 'running', 'failed_exit', 'manifest_hash', 'source_hash', 'snapshot_hash', 'missing_pin', 'launcher_hash'])
def test_proof_required_without_erasing_recorded_lower_bound(bundle, bad):
    b = bundle
    if bad == 'no_closure': b.cp.unlink()
    elif bad == 'writers': b.closure['writer_pids'] = [1]
    elif bad == 'running': b.closure['supervisor_state'] = 'RUNNING'
    elif bad == 'failed_exit': b.closure['child_exit_code'] = 1
    elif bad == 'source_hash': (b.source / 'certmem/runtime.py').write_text('changed')
    elif bad == 'snapshot_hash': b.closure['raw_source_sha256']['receipt.json'] = '0' * 64
    elif bad == 'missing_pin': b.closure['raw_source_sha256'].pop('receipt.json')
    elif bad == 'launcher_hash':
        p = b.out / 'receipt.json'
        data = json.loads(p.read_text())
        data['manifest_sha256'] = '0' * 64
        put(p, data)
        close(b)
    if b.cp.exists(): put(b.cp, b.closure)
    result = b.t.audit_runtime(b.out, b.mp, '0' * 64 if bad == 'manifest_hash' else sha(b.mp),
                              b.cp if b.cp.exists() else None, sha(b.cp) if b.cp.exists() else None)
    assert result['complete'] is False and result['closed_sources_complete'] is False
    assert result['generation']['reported_tokens']['total_tokens'] == 420
    assert result['generation']['exact_tokens'] is None


def test_cli_fresh_output_outside_original_snapshot(bundle):
    b = bundle
    dest = b.mp.parent / 'audit.json'
    dest.write_text('preserve')
    args = ['--runtime-output', str(b.out), '--manifest', str(b.mp), '--expected-manifest-sha256', sha(b.mp), '--output', str(dest)]
    with pytest.raises(FileExistsError): b.t.main(args)
    assert dest.read_text() == 'preserve'
    args[-1] = str(b.out / 'audit.json')
    with pytest.raises(ValueError): b.t.main(args)


@pytest.mark.parametrize('bad', ['embedding_context', 'qa_conv_list', 'qa_config_list', 'qa_policy_list',
                                'n_bool', 'candidate_index_bool', 'inference_duration', 'dataset_row'])
def test_malformed_metadata_never_escapes_incomplete_audit(bundle, bad):
    b, r = bundle, bundle.gen[-1]
    if bad == 'embedding_context':
        r = b.emb[-1]
        r.request['context'] = None
    elif bad == 'qa_conv_list': r.request['context']['conv_id'] = []
    elif bad == 'qa_config_list': r.request['context']['config'] = []
    elif bad == 'qa_policy_list': r.request['context']['items'][0]['policy'] = []
    elif bad == 'n_bool': r.request['sampling']['n'] = True
    elif bad == 'candidate_index_bool': r.response['responses'][0]['outputs'][0]['index'] = False
    elif bad == 'inference_duration': r.receipt['inference_duration_s'] = float('inf')
    elif bad == 'dataset_row':
        manifest = json.loads(b.mp.read_text())
        put(Path(manifest['dataset']), [None])
    save(r)
    close(b)
    result = audit(b)
    assert result['complete'] is False and result['issues']
    if bad == 'embedding_context': assert result['embedding']['recorded_input_tokens'] == 21


def test_cli_never_writes_pinned_source_and_preserves_summary(bundle):
    b = bundle
    args = ['--runtime-output', str(b.out), '--manifest', str(b.mp), '--expected-manifest-sha256', sha(b.mp),
            '--closure', str(b.cp), '--expected-closure-sha256', sha(b.cp), '--output', str(b.source / 'new.json')]
    with pytest.raises(ValueError): b.t.main(args)
    args[-1] = str(b.mp.parent / 'final-audit.json')
    assert b.t.main(args) == 0
    result = json.loads(Path(args[-1]).read_text(encoding='utf-8'))
    assert result['complete'] is True
    assert all('native_text' not in row for row in result['generation_ledger'])


@pytest.mark.parametrize('bad_qa', [None, 1, []])
def test_invalid_dataset_qa_object_preserves_incomplete_audit(bundle, bad_qa):
    b = bundle
    manifest = json.loads(b.mp.read_text())
    put(Path(manifest['dataset']), [{'sample_id': 'conv-26', 'qa': [bad_qa]}])
    result = audit(b)
    assert result['complete'] is False and result['issues']
    assert result['generation']['exact_tokens'] is None
    assert result['generation']['reported_tokens']['total_tokens'] == 420


@pytest.mark.parametrize('suffix', ['request', 'response'])
def test_directory_instead_of_runtime_artifact_fails_closed(bundle, suffix):
    b = bundle
    path = b.gen[-1].paths[suffix]
    path.unlink()
    path.mkdir()
    close(b)
    result = audit(b)
    assert result['complete'] is False and result['generation']['exact_tokens'] is None
    assert result['generation']['reported_tokens']['total_tokens'] > 0
