"""Synthetic CPU-only partial snapshots, with real frozen helper validation."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/audit_partial_runtime.py'
BASE = Path(os.environ.get('CERTMEM_BASE_AUDITOR', 'D:/MemoryData/certmem_postrun_r1/scripts/audit_certmem_runtime.py'))
SELECTOR = Path(os.environ.get('CERTMEM_COST_SELECTOR', 'D:/MemoryData/certmem_policy_cost_r1/scripts/audit_certified_cost.py'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False) + '\n', encoding='utf-8')
    return sha(path)


@pytest.fixture
def target():
    assert SCRIPT.is_file(), 'Intentional-partial auditor is not implemented'
    spec = importlib.util.spec_from_file_location('partial_runtime', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(record):
    record.rec['request_hash'] = record.rec['request_sha256'] = put(record.paths['request'], record.req)
    if record.res is not None:
        record.rec['response_sha256'] = put(record.paths['response'], record.res)
    put(record.paths['receipt'], record.rec)


def generation(b, context, count, cap):
    seq = len(b.gen) + 1
    req = {'model': b.base.MODEL, 'model_revision': b.base.REVISION, 'context': context,
           'sampling': {'temperature': 0.0, 'max_tokens': cap, 'n': 1, 'truncate_prompt_tokens': None},
           'prompts': [f'Original {seq}:{i} 원문' for i in range(count)]}
    res = {'responses': [{'request_id': f'{seq}:{i}', 'prompt': p, 'prompt_token_ids': [1, 2, 3],
        'num_cached_tokens': 2, 'finished': True, 'outputs': [{'index': 0, 'text': ' answer ',
        'token_ids': [4, 5], 'finish_reason': 'length' if i == 0 else 'stop'}]}
        for i, p in enumerate(req['prompts'])]}
    rec = {'type': 'generation_batch', 'batch_sequence': seq, 'context': copy.deepcopy(context),
        'status': 'success', 'usage_complete': True, 'request_count': count, 'response_count': count,
        'generation_count': count, 'prompt_tokens': count * 3, 'completion_tokens': count * 2,
        'total_tokens': count * 5, 'preflight_prompt_tokens': [3] * count, 'duration_s': 1.0,
        'inference_duration_s': .9, 'finish_reasons': [r['outputs'][0]['finish_reason'] for r in res['responses']]}
    r = SimpleNamespace(req=req, res=res, rec=rec,
        paths={k: b.out / f'runtime/generation/{seq:08d}.{k}.json' for k in ('request', 'response', 'receipt')})
    save(r)
    b.gen.append(r)


def embedding(b, context, per):
    seq = len(b.emb) + 1
    fields = {'input_tokens': sum(per), 'padded_tokens': len(per) * max(per), 'tokens_per_input': per,
              'shape': [len(per), max(per)]}
    req = {'model': b.base.EMBED_MODEL, 'model_revision': b.base.EMBED_REVISION,
           'native_max_seq_length': 256, 'context': context, **fields}
    rec = {'type': 'embedding_forward', 'batch_sequence': seq, 'context': copy.deepcopy(context),
           'status': 'success', 'usage_complete': True, 'duration_s': .1, **fields}
    r = SimpleNamespace(req=req, res=None, rec=rec,
        paths={k: b.out / f'runtime/embedding/{seq:08d}.{k}.json' for k in ('request', 'receipt')})
    save(r)
    b.emb.append(r)


def close(b):
    b.closure['raw_source_sha256'] = b.base.inventory(b.out)
    put(b.cp, b.closure)
    with tarfile.open(b.archive, 'w:gz') as stream:
        stream.add(b.out, arcname=b.out.name)
        stream.add(b.cp, arcname=f'provenance/{b.cp.name}')
        stream.add(b.mp, arcname=f'provenance/{b.mp.name}')


def stats(b):
    with (b.out / 'write_stats.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['conv_id', 'config', 'session', 'mem_tok', 'raw_tok'])
        writer.writeheader()
        writer.writerows(b.stats)


@pytest.fixture
def bundle(tmp_path, monkeypatch, target):
    base, selector = target.load_dependencies(BASE, SELECTOR)
    b = SimpleNamespace(t=target, base=base, out=tmp_path / 'output', source=tmp_path / 'source',
        mp=tmp_path / 'manifest.json', cp=tmp_path / 'closure.json', archive=tmp_path / 'snapshot.tar.gz', gen=[], emb=[], stats=[])
    monkeypatch.setattr(target, 'load_dependencies', lambda *_: (base, selector))
    monkeypatch.setattr(base, 'audit_runtime', lambda *_: pytest.fail('Do not suppress full auditor failures'))
    monkeypatch.setattr(selector, 'audit_certified_cost', lambda *_: pytest.fail('Do not bypass full selector gate'))
    monkeypatch.setattr(base, 'CONV_TOTAL', 3)
    monkeypatch.setattr(base, 'QA_TOTAL', 4)
    monkeypatch.setattr(target, 'SELECTED_CIDS', ('conv-26', 'conv-30'))
    monkeypatch.setattr(target, 'SELECTED_QA', 3)
    for name in base.REQUIRED_SOURCE:
        path = b.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture source\n', encoding='utf-8')
    data = [{'sample_id': cid, 'conversation': {'session_1': [{'speaker': 'A', 'text': 'Original turn'}]},
             'qa': [{'category': 1, 'question': f'Q{i}', 'answer': ['gold', i]} for i in range(n)]}
            for cid, n in [('conv-26', 2), ('conv-30', 1), ('conv-47', 1)]]
    dataset = b.source / 'data/locomo10.json'
    monkeypatch.setattr(base, 'DATASET_SHA256', put(dataset, data))
    manifest = {'schema_version': 1, 'run_id': base.RUN_ID, 'output': str(b.out), 'source_root': str(b.source),
        'dataset': str(dataset), 'python': '/venv/main/bin/python',
        'files_sha256': {str(p): sha(p) for p in b.source.rglob('*') if p.is_file()}}
    monkeypatch.setattr(target, 'MANIFEST_SHA256', put(b.mp, manifest))
    rows = []
    for sample in data[:2]:
        cid, count = sample['sample_id'], len(sample['qa'])
        generation(b, {'phase': 'fact_extraction', 'conv_id': cid}, 1, 700)
        for config, policies in base.POLICIES.items():
            generation(b, {'phase': 'memory_build', 'conv_id': cid, 'config': config}, 1, 512)
            generation(b, {'phase': 'query_reform', 'conv_id': cid, 'config': config}, count, 60)
            b.stats.append({'conv_id': cid, 'config': config, 'session': 1, 'mem_tok': 20, 'raw_tok': 30})
            batches = [[2, 3], [4]] + [[5 + i] for i in range(count) for _ in range(12)] if config == 'full' else [[3]]
            for per in batches:
                embedding(b, {'phase': 'retrieval', 'config': config, 'conv_id': cid}, per)
            items = [{'policy': policy, 'question_ordinal': i} for i in range(count) for policy in policies]
            generation(b, {'phase': 'qa', 'config': config, 'conv_id': cid, 'items': items}, len(items), 32)
            rows.extend({'conv_id': cid, 'config': config, **item, 'category': 1,
                'question': f'Q{item["question_ordinal"]}', 'gold': str(['gold', item['question_ordinal']]), 'pred': 'answer'} for item in items)
    with (b.out / 'items.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stats(b)
    command = [manifest['python'], '-u', str(b.source / 'scripts/run_system_v15.py'), '--comparison', '--dataset',
               'locomo', '--configs', 'full,no_adaptive,no_residual', '--budgets', '400,1600,4000', '--out', str(b.out)]
    b.launcher = {'schema_version': 1, 'run_id': base.RUN_ID, 'manifest_sha256': sha(b.mp), 'command': command,
        'child_exit_code': -15, 'source_unchanged': True, 'signals_received': [15], 'errors': [],
        'inference_complete': False, 'complete': False, 'official_scoring_complete': False,
        'native_outputs': {'rows': 0, 'complete': False, 'issues': ['summary.csv missing']},
        'runtime': {'sampled_device_energy_wh': 100, 'estimated_rental_usd': 1}}
    put(b.out / 'receipt.json', b.launcher)
    b.closure = {'schema_version': 1, 'run_id': base.RUN_ID, 'output': str(b.out),
        'closure_kind': 'intentional-partial', 'global_writer_absence_proven': False, 'writer_pids': None,
        'benchmark_complete': False, 'covered_conversations': list(target.SELECTED_CIDS),
        'captured_qa_per_policy': 3, 'remaining_qa_per_policy': 1, 'reader_outputs': 111,
        'generation_groups': len(b.gen), 'embedding_groups': len(b.emb),
        'manifest_sha256': sha(b.mp),
        'supervisor_state': 'EXITED', 'child_exit_code': -15, 'known_benchmark_writer_pids': [],
        'uninspected_processes': [{'pid': p, 'uid': 1001, 'reason': 'PermissionError /proc cwd/fd'} for p in (925, 952, 1038, 1055)]}
    monkeypatch.setattr(target, 'EXPECTED_GROUPS', {'generation': len(b.gen), 'embedding': len(b.emb)})
    monkeypatch.setattr(target, 'ITEMS_SHA256', sha(b.out / 'items.csv'))
    close(b)
    return b


def audit(b):
    return b.t.audit_partial(b.out, b.mp, sha(b.mp), b.cp, sha(b.cp), b.archive, sha(b.archive), BASE, SELECTOR)


def test_recorded_exact_tokens_do_not_claim_global_closure_or_full_completion(bundle):
    b = bundle
    before = b.base.inventory(b.out)
    r = audit(b)
    assert r['recorded_snapshot_usage_complete'] is True, r['issues']
    assert r['closed_sources_complete'] is r['benchmark_complete'] is r['complete'] is False
    assert r['child_exit_code'] == -15 and r['uninspected_processes'] == b.closure['uninspected_processes']
    assert r['qa_coverage'] == {'partial_matrix_complete': True, 'recorded': 111, 'partial_expected': 111,
                               'original_expected': 148, 'qa_per_policy': 3, 'original_qa_per_policy': 4}
    assert r['generation']['exact_tokens']['total_tokens'] == 640
    assert r['generation']['length_responses'] == 20
    assert r['embedding']['exact_input_tokens'] == 222
    assert r['limitations'] and not r['issues']
    assert b.base.inventory(b.out) == before
    assert b.base.read_object(b.out / 'receipt.json')['native_outputs']['rows'] == 0


def test_partial_primary_dependency_usage_not_whole_baseline_or_grid_gross(bundle):
    r = audit(bundle)
    s = r['selected_primary_usage']
    assert s['qa_per_policy'] == 3
    assert s['certified']['generation']['exact_tokens']['total_tokens'] == 50
    assert s['certified']['embedding']['exact_input_tokens'] == 66
    assert s['naive']['generation']['exact_tokens']['total_tokens'] == 15
    assert s['naive']['causal_embedding_tokens'] == s['naive']['causal_construction_tokens'] == 0
    assert s['additive_across_policies'] is s['standalone_actual_run'] is False
    assert r['exclusive_gpu_energy_wh'] is r['exclusive_rental_usd'] is None


@pytest.mark.parametrize('bad', ['exit0', 'inference_true', 'source_changed', 'writer', 'unknown_scan_missing',
                                'closure_inventory', 'wrong_recipe', 'launcher_error'])
def test_bad_termination_or_pins_never_promote_exact(bundle, bad):
    b = bundle
    if bad == 'exit0': b.launcher['child_exit_code'] = 0
    elif bad == 'inference_true': b.launcher['inference_complete'] = True
    elif bad == 'source_changed': b.launcher['source_unchanged'] = False
    elif bad == 'writer': b.closure['known_benchmark_writer_pids'] = [88]
    elif bad == 'unknown_scan_missing': b.closure.pop('uninspected_processes')
    elif bad == 'wrong_recipe': b.launcher['command'][-3] = '400'
    elif bad == 'launcher_error': b.launcher['errors'] = ['unexpected failure']
    put(b.out / 'receipt.json', b.launcher)
    close(b)
    if bad == 'closure_inventory':
        b.closure['raw_source_sha256'].pop('items.csv')
        put(b.cp, b.closure)
    r = audit(b)
    assert not r['recorded_snapshot_usage_complete'] and r['issues']
    assert r['generation']['exact_tokens'] is r['selected_primary_usage'] is None


@pytest.mark.parametrize('bad', ['missing_receipt', 'partial', 'bad_usage', 'wrong_model', 'wrong_tokens',
                                'foreign_conv', 'duplicate_response', 'bad_mask', 'bad_duration', 'csv_gold',
                                'stats_gap', 'stats_empty', 'archive_stale', 'group_count'])
def test_raw_corruption_or_incomplete_boundary_fails_closed(bundle, monkeypatch, bad):
    b, x = bundle, bundle.gen[-1]
    if bad == 'missing_receipt': x.paths['receipt'].unlink()
    elif bad == 'partial': (b.out / 'runtime/generation/inflight.partial').write_text('{}')
    elif bad == 'bad_usage': x.rec['usage_complete'] = False; save(x)
    elif bad == 'wrong_model': x.req['model'] = 'other'; save(x)
    elif bad == 'wrong_tokens': x.res['responses'][0]['outputs'][0]['token_ids'] = None; save(x)
    elif bad == 'foreign_conv': x.req['context']['conv_id'] = 'conv-47'; save(x)
    elif bad == 'duplicate_response': x.res['responses'][0]['request_id'] = '1:0'; save(x)
    elif bad == 'bad_mask': b.emb[0].rec['tokens_per_input'] = [99]; save(b.emb[0])
    elif bad == 'bad_duration': b.emb[0].rec['duration_s'] = -1; save(b.emb[0])
    elif bad == 'csv_gold':
        p = b.out / 'items.csv'
        p.write_text(p.read_text(encoding='utf-8').replace('gold', 'changed'), encoding='utf-8')
        monkeypatch.setattr(b.t, 'ITEMS_SHA256', sha(p))
    elif bad == 'stats_gap': b.stats.pop(); stats(b)
    elif bad == 'stats_empty':
        for row in b.stats: row['mem_tok'] = 0
        stats(b)
    elif bad == 'group_count': monkeypatch.setattr(b.t, 'EXPECTED_GROUPS', {'generation': 999, 'embedding': len(b.emb)})
    close(b)
    if bad == 'archive_stale':
        (b.out / 'benchmark.log').write_text('changed\n')
        b.closure['raw_source_sha256'] = b.base.inventory(b.out)
        put(b.cp, b.closure)
    r = audit(b)
    assert not r['recorded_snapshot_usage_complete'] and r['issues']
    assert r['generation']['exact_tokens'] is None and r['benchmark_complete'] is False


def test_exact_default_production_scope(target):
    assert target.SELECTED_CIDS == ('conv-26', 'conv-30', 'conv-41', 'conv-42', 'conv-43', 'conv-44')
    assert target.SELECTED_QA == 885 and target.EXPECTED_GROUPS == {'generation': 3123, 'embedding': 36516}
    assert target.ITEMS_SHA256 == '4bd01e08b31324ef6916bc450436cba032f2267727a063d8339bb906b6c7bfac'
    assert target.MANIFEST_SHA256 == 'fd0aac2763441be3efcac3f88800d7c5d6251df8543346502eb7fcffebd9e944'


def test_dependency_hash_rejected_before_execution(target, tmp_path):
    fake = tmp_path / 'auditor.py'
    fake.write_text('raise AssertionError("must not execute")\n')
    with pytest.raises(ValueError, match='hash'):
        target.load_dependencies(fake, SELECTOR)


def cli_args(b, report):
    return ['--output', str(b.out), '--manifest', str(b.mp), '--expected-manifest-sha256', sha(b.mp),
            '--closure', str(b.cp), '--expected-closure-sha256', sha(b.cp), '--archive', str(b.archive),
            '--expected-archive-sha256', sha(b.archive), '--base-auditor', str(BASE), '--selector', str(SELECTOR), '--report', str(report)]


def test_cli_refuses_snapshot_source_and_existing_report(bundle):
    for path in (bundle.out / 'audit.json', bundle.source / 'audit.json', bundle.cp):
        with pytest.raises(ValueError):
            bundle.t.main(cli_args(bundle, path))


def test_cli_success_is_explicitly_snapshot_only(bundle, tmp_path):
    report = tmp_path / 'derived.json'
    assert bundle.t.main(cli_args(bundle, report)) == 0
    assert json.loads(report.read_text())['closed_sources_complete'] is False


@pytest.mark.parametrize('failure', [PermissionError, FileNotFoundError])
def test_unreadable_runtime_directory_retains_other_known_rows_and_failed_report(bundle, monkeypatch, failure):
    original = Path.iterdir
    def unavailable(path):
        if path == bundle.out / 'runtime/generation':
            raise failure('synthetic unavailable runtime directory')
        return original(path)
    monkeypatch.setattr(Path, 'iterdir', unavailable)
    result = audit(bundle)
    assert result['recorded_snapshot_usage_complete'] is False
    assert result['complete'] is result['closed_sources_complete'] is result['benchmark_complete'] is False
    assert result['generation']['exact_tokens'] is result['embedding']['exact_input_tokens'] is None
    assert result['embedding']['recorded_input_tokens'] == 222
    assert any('unavailable runtime directory' in issue for issue in result['issues'])
