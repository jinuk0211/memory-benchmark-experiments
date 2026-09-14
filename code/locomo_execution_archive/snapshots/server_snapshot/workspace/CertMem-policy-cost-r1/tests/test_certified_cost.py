"""Synthetic CPU-only selector contracts; the frozen base gate has its own suite."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/audit_certified_cost.py'
BASE = Path('D:/MemoryData/certmem_postrun_r1/scripts/audit_certmem_runtime.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + '\n', encoding='utf-8')
    return sha(path)


@pytest.fixture
def target():
    assert SCRIPT.is_file(), 'The required offline certified cost selector is not implemented'
    spec = importlib.util.spec_from_file_location('certified_cost', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(record):
    record.rec['request_hash'] = record.rec['request_sha256'] = put(record.rq, record.req)
    put(record.rp, record.rec)


def close(b):
    b.closure['raw_source_sha256'] = b.base.inventory(b.out)
    put(b.cp, b.closure)


def stats(b):
    with (b.out / 'write_stats.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['conv_id', 'config', 'session', 'mem_tok', 'raw_tok'])
        writer.writeheader()
        writer.writerows(b.stats)


@pytest.fixture
def bundle(tmp_path, monkeypatch, target):
    base = target.load_base(BASE)
    b = SimpleNamespace(t=target, base=base, out=tmp_path / 'output', source=tmp_path / 'source',
                        mp=tmp_path / 'manifest.json', cp=tmp_path / 'closure.json', records=[], stats=[], calls=[])
    b.data = [{'sample_id': cid, 'conversation': {'session_1': [{'speaker': 'A', 'text': 'Original turn'}]},
               'qa': [{'category': 1, 'question': f'Q{i}'} for i in range(n)]}
              for cid, n in [('conv-26', 2), ('conv-30', 1)]]
    dp = b.source / 'data/locomo10.json'
    put(dp, b.data)
    b.manifest = {'run_id': base.RUN_ID, 'source_root': str(b.source), 'output': str(b.out),
                  'dataset': str(dp), 'python': '/venv/main/bin/python', 'files_sha256': {str(dp): sha(dp)}}
    put(b.mp, b.manifest)
    monkeypatch.setattr(target, 'MANIFEST_SHA256', sha(b.mp))
    monkeypatch.setattr(target, 'QA_TOTAL', 3)
    monkeypatch.setattr(target, 'CONV_TOTAL', 2)
    for sample in b.data:
        cid, n = sample['sample_id'], len(sample['qa'])
        b.stats.extend({'conv_id': cid, 'config': cfg, 'session': 1, 'mem_tok': 20, 'raw_tok': 30}
                       for cfg in ('full', 'no_adaptive', 'no_residual'))
        for cfg in ('full', 'no_adaptive', 'no_residual'):
            # Unequal observed prefix batch sizes: no assumed ST default batch size.
            counts = [[2, 3, 4], [5], [6, 7]] if cfg == 'full' else [[8]]
            counts += [[9 + i] for i in range(n) for _ in range(12)] if cfg == 'full' else []
            for per in counts:
                seq = len(b.records) + 1
                ctx = {'phase': 'retrieval', 'config': cfg, 'conv_id': cid}
                fields = {'input_tokens': sum(per), 'padded_tokens': len(per) * max(per),
                          'tokens_per_input': per, 'shape': [len(per), max(per)]}
                req = {'model': base.EMBED_MODEL, 'model_revision': base.EMBED_REVISION,
                       'native_max_seq_length': 256, 'context': ctx, **fields}
                rec = {'type': 'embedding_forward', 'batch_sequence': seq, 'context': ctx,
                       'status': 'success', 'usage_complete': True, 'duration_s': .01, **fields}
                record = SimpleNamespace(req=req, rec=rec, rq=b.out / f'runtime/embedding/{seq:08d}.request.json',
                                         rp=b.out / f'runtime/embedding/{seq:08d}.receipt.json')
                save(record)
                b.records.append(record)
    stats(b)
    command = [b.manifest['python'], '-u', str(b.source / 'scripts/run_system_v15.py'), '--comparison',
               '--dataset', 'locomo', '--configs', 'full,no_adaptive,no_residual', '--budgets', '400,1600,4000', '--out', str(b.out)]
    put(b.out / 'receipt.json', {'command': command})
    def gen(n):
        return {'exact_tokens': {'prompt_tokens': n, 'completion_tokens': 1, 'total_tokens': n + 1}, 'response_count': 3}
    full_records = [r for r in b.records if r.req['context']['config'] == 'full']
    em = {'forward_count': len(full_records), 'exact_input_tokens': sum(r.req['input_tokens'] for r in full_records),
          'recorded_padded_tokens': sum(r.req['padded_tokens'] for r in full_records)}
    b.report = {'complete': True, 'closed_sources_complete': True, 'usage_complete': True,
        'qa_coverage': {'complete': True, 'count': 111, 'expected': 111},
        'by_phase_config': {key: {'generation': gen(n)} for key, n in
                           [('fact_extraction/shared', 10), ('memory_build/full', 20), ('query_reform/full', 30)]},
        'qa_by_policy': {key: {'generation': gen(n), 'response_count': 3} for key, n in
                         [('full/certified', 40), ('full/full_raw', 50)]}, 'embedding_ledger': []}
    b.report['by_phase_config']['retrieval/full'] = {'embedding': em}
    b.report['embedding_ledger'] = [{'batch_sequence': r.rec['batch_sequence'], 'context': r.req['context'],
         'bucket': f"retrieval/{r.req['context']['config']}", 'input_tokens': r.req['input_tokens'],
         'padded_tokens': r.req['padded_tokens'], 'request_sha256': sha(r.rq), 'status': 'success'} for r in b.records]
    b.closure = {'schema_version': 1, 'run_id': base.RUN_ID, 'output': str(b.out),
                 'supervisor_state': 'EXITED', 'writer_pids': [], 'child_exit_code': 0}
    close(b)
    def gate(*args):
        b.calls.append(args)
        return copy.deepcopy(b.report)
    monkeypatch.setattr(base, 'audit_runtime', gate)
    monkeypatch.setattr(base, 'source_proof', lambda *args: {})
    monkeypatch.setattr(target, 'load_base', lambda path: base)
    return b


def audit(b):
    return b.t.audit_certified_cost(b.out, b.mp, sha(b.mp), b.cp, sha(b.cp), BASE)


def test_dependency_sum_prefix_plus_first_three_and_naive(bundle):
    b = bundle
    before = b.base.inventory(b.out)
    result = audit(b)
    assert result['complete'] is True
    assert result['certified']['generation']['exact_tokens'] == {'prompt_tokens': 100, 'completion_tokens': 4, 'total_tokens': 104}
    selected = result['certified']['embedding']['selected']
    excluded = result['certified']['embedding']['excluded']
    assert [r['batch_sequence'] for r in selected] == [1, 2, 3, 4, 5, 6, 16, 17, 18, 30, 31, 32, 33, 34, 35]
    assert len(selected) == 15 and len(excluded) == 27
    assert result['certified']['embedding']['exact_input_tokens'] == sum(r['input_tokens'] for r in selected)
    assert result['naive']['generation']['exact_tokens']['total_tokens'] == 51
    assert result['naive']['causal_embedding_tokens'] == result['naive']['causal_construction_tokens'] == 0
    assert result['exclusive_gpu_energy_wh'] is None and result['exclusive_rental_usd'] is None
    assert result['additive_across_policies'] is False and result['standalone_actual_run'] is False
    assert b.calls == [(b.out, b.mp, sha(b.mp), b.cp, sha(b.cp))]
    assert b.base.inventory(b.out) == before


@pytest.mark.parametrize('field', ['complete', 'usage_complete', 'closed_sources_complete'])
def test_base_must_actually_pass(bundle, field):
    bundle.report[field] = False
    result = audit(bundle)
    assert not result['complete'] and result['certified']['generation']['exact_tokens'] is None


@pytest.mark.parametrize('mutation', ['empty_raw', 'empty_memory', 'missing_session', 'duplicate_session', 'foreign_stats', 'bad_cli', 'qa_gap'])
def test_native_recipe_evidence_failclosed(bundle, mutation):
    b = bundle
    if mutation == 'empty_raw':
        b.data[0]['conversation']['session_1'] = []
        put(Path(b.manifest['dataset']), b.data)
    elif mutation == 'empty_memory':
        b.stats[0]['mem_tok'] = 0
    elif mutation == 'missing_session':
        b.stats.pop(0)
    elif mutation == 'duplicate_session':
        b.stats.append(b.stats[0].copy())
    elif mutation == 'foreign_stats':
        b.stats[0]['conv_id'] = 'foreign'
    elif mutation == 'bad_cli':
        put(b.out / 'receipt.json', {'command': ['--no_reform']})
    else:
        b.report['qa_coverage']['count'] = 110
    stats(b)
    close(b)
    result = audit(b)
    assert not result['complete'] and result['certified']['embedding']['exact_input_tokens'] is None


@pytest.mark.parametrize('mutation', ['missing', 'hash', 'zero', 'shape', 'counts', 'singleton', 'different', 'context', 'sequence', 'status', 'unknown_usage'])
def test_raw_embedding_contract_failclosed(bundle, mutation):
    b, r = bundle, bundle.records[3]
    if mutation == 'missing':
        r.rp.unlink()
    elif mutation == 'hash':
        r.rec['request_sha256'] = '0' * 64
        put(r.rp, r.rec)
    else:
        if mutation == 'zero': r.req.update(input_tokens=0, tokens_per_input=[0])
        if mutation == 'shape': r.req['shape'] = [1, 257]
        if mutation == 'counts': r.req['input_tokens'] = 99
        if mutation == 'singleton': r.req.update(input_tokens=18, padded_tokens=18, tokens_per_input=[9, 9], shape=[2, 9])
        if mutation == 'different': r.req.update(input_tokens=8, tokens_per_input=[8])
        if mutation == 'context': r.rec['context'] = {'phase': 'retrieval', 'config': 'full', 'conv_id': 'foreign'}
        if mutation == 'sequence': r.rec['batch_sequence'] = 99
        if mutation == 'status': r.rec['status'] = 'error'
        if mutation == 'unknown_usage': r.rec['usage_complete'] = False
        save(r)
    close(b)
    result = audit(b)
    assert not result['complete'] and result['certified']['embedding']['exact_input_tokens'] is None


def test_missing_closure_and_changed_snapshot_failclosed(bundle):
    b = bundle
    put(b.out / 'foreign.json', {})
    assert not audit(b)['complete']
    b.cp.unlink()
    result = b.t.audit_certified_cost(b.out, b.mp, sha(b.mp), None, None, BASE)
    assert not result['complete']


def test_frozen_loader_rejects_other_code_and_manifest(target, tmp_path):
    assert callable(target.load_base(BASE).audit_runtime)
    fake = tmp_path / 'fake.py'
    fake.write_text('raise RuntimeError("must never execute")')
    with pytest.raises(ValueError, match='auditor'):
        target.load_base(fake)
    result = target.audit_certified_cost(tmp_path, fake, '0' * 64, None, None, BASE)
    assert not result['complete']


@pytest.mark.parametrize('mutation', ['suffix_counts', 'prefix_missing', 'noncontiguous', 'reconcile', 'bad_policy', 'bad_tokens', 'bad_qa', 'partial_stats'])
def test_remaining_selection_gates(bundle, mutation):
    b = bundle
    if mutation == 'suffix_counts':
        r = b.records[3]
        r.req.update(input_tokens=8, padded_tokens=8, tokens_per_input=[8], shape=[1, 8])
        r.rec.update({k: r.req[k] for k in ('input_tokens', 'padded_tokens', 'tokens_per_input', 'shape')})
        save(r)
        b.report['embedding_ledger'][3].update(input_tokens=8, padded_tokens=8, request_sha256=sha(r.rq))
    elif mutation == 'prefix_missing':
        b.report['embedding_ledger'] = b.report['embedding_ledger'][2:]
    elif mutation == 'noncontiguous':
        b.report['embedding_ledger'].pop(1)
    elif mutation == 'reconcile':
        b.report['by_phase_config']['retrieval/full']['embedding']['exact_input_tokens'] += 1
    elif mutation == 'bad_policy':
        b.report['qa_by_policy']['full/full_raw']['response_count'] = 2
    elif mutation == 'bad_tokens':
        b.report['by_phase_config']['memory_build/full']['generation']['exact_tokens']['prompt_tokens'] = None
    elif mutation == 'bad_qa':
        b.data[0]['qa'] = [None]
        put(Path(b.manifest['dataset']), b.data)
    else:
        (b.out / 'write_stats.csv').write_text('conv_id,config,session,mem_tok,raw_tok\nconv-26\n')
    close(b)
    result = audit(b)
    assert not result['complete'] and result['certified']['generation']['exact_tokens'] is None


def cli_args(b, destination):
    return ['--runtime-output', str(b.out), '--manifest', str(b.mp), '--base-auditor', str(BASE),
            '--expected-manifest-sha256', sha(b.mp), '--closure', str(b.cp),
            '--expected-closure-sha256', sha(b.cp), '--output', str(destination)]


def test_cli_fresh_destination_success_and_incomplete(bundle, tmp_path):
    b = bundle
    destination = tmp_path / 'audit.json'
    assert b.t.main(cli_args(b, destination)) == 0
    assert json.loads(destination.read_text())['complete']
    with pytest.raises(ValueError, match='already exists'):
        b.t.main(cli_args(b, destination))
    b.report['complete'] = False
    assert b.t.main(cli_args(b, tmp_path / 'incomplete.json')) == 1


@pytest.mark.parametrize('where', ['snapshot', 'source'])
def test_cli_cannot_mutate_protected_inputs(bundle, where):
    destination = (bundle.out if where == 'snapshot' else bundle.source) / 'audit.json'
    with pytest.raises(ValueError, match='outside'):
        bundle.t.main(cli_args(bundle, destination))
    assert not destination.exists()


def test_inventory_change_during_selection(bundle, monkeypatch):
    b, original, calls = bundle, bundle.base.inventory, []
    def changed(path):
        value = original(path)
        calls.append(path)
        if len(calls) == 2:
            value['race'] = 'changed'
        return value
    monkeypatch.setattr(b.base, 'inventory', changed)
    assert not audit(b)['complete']


def test_real_frozen_base_rejects_unclosed_synthetic_source(bundle, monkeypatch):
    b = bundle
    spec = importlib.util.spec_from_file_location('clean_certified', SCRIPT)
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    monkeypatch.setattr(real, 'MANIFEST_SHA256', sha(b.mp))
    result = real.audit_certified_cost(b.out, b.mp, sha(b.mp), b.cp, sha(b.cp), BASE)
    assert not result['complete'] and result['base_complete'] is False


@pytest.mark.parametrize('field', ['input_tokens', 'tokens_per_input', 'shape'])
def test_receipt_boolean_counts_are_not_numeric_evidence(bundle, field):
    b = bundle
    # A consistent singleton group of ones isolates bool == int coercion.
    for i in range(3, 15):
        r = b.records[i]
        values = {'input_tokens': 1, 'padded_tokens': 1, 'tokens_per_input': [1], 'shape': [1, 1]}
        r.req.update(values)
        r.rec.update(values)
        save(r)
        b.report['embedding_ledger'][i].update(input_tokens=1, padded_tokens=1, request_sha256=sha(r.rq))
    r = b.records[3]
    r.rec[field] = True if field == 'input_tokens' else ([True] if field == 'tokens_per_input' else [True, 1])
    put(r.rp, r.rec)
    em = b.report['by_phase_config']['retrieval/full']['embedding']
    em['exact_input_tokens'] -= 96
    em['recorded_padded_tokens'] -= 96
    close(b)
    assert not audit(b)['complete']
