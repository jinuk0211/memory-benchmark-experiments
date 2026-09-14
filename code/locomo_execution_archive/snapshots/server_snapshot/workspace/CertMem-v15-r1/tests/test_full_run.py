"""CPU-only contract tests; never load a model or invoke a GPU."""
import csv
import hashlib
import importlib.util
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts' / 'run_certmem_full.py'
COUNTS = dict(zip(['conv-26', 'conv-30', 'conv-41', 'conv-42', 'conv-43',
                  'conv-44', 'conv-47', 'conv-48', 'conv-49', 'conv-50'],
                 [152, 81, 152, 199, 178, 123, 150, 191, 156, 158]))
OMITTED = {'conv-41', 'conv-43', 'conv-44'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def target():
    spec = importlib.util.spec_from_file_location('certmem_full_launcher', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path, monkeypatch, target):
    root, out = tmp_path / 'source', tmp_path / 'output'
    root.mkdir()
    for name in ('scripts/run_system_v15.py', 'certmem/locomo.py', 'README.md', 'pyproject.toml'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture\n', encoding='utf-8')
    data = [{'sample_id': cid, 'qa': [{'question': f'Q{i}', 'answer': [i, 'answer'],
             'category': i % 4 + 1} for i in range(n)] + [{'category': 5}]}
            for cid, n in COUNTS.items()]
    dataset = root / 'data/locomo10.json'
    write_json(dataset, data)
    preflight = {'schema_version': 1, 'dataset_sha256': sha(dataset),
        'model': 'Qwen/Qwen3.5-9B', 'revision': target.REVISION,
        'max_model_len': 49152, 'reader_max_tokens': 32, 'qa_total': 1540,
        'all_fit': True, 'original_full_raw_omitted_qa': 453,
        'rows': [{'conv_id': cid, 'qa_count': n, 'raw_tokens': 40000 if cid in OMITTED else 1000,
                  'max_chat_prompt_tokens': 40311, 'max_total_tokens': 40343,
                  'original_full_raw_included': cid not in OMITTED} for cid, n in COUNTS.items()]}
    pre = tmp_path / 'token_preflight.json'
    write_json(pre, preflight)
    manifest = {'schema_version': 1, 'run_id': target.RUN_ID, 'source_root': str(root),
        'dataset': str(dataset), 'output': str(out), 'python': target.PYTHON,
        'files_sha256': {str(p): sha(p) for p in root.rglob('*') if p.is_file()},
        'token_preflight': {'path': str(pre), 'sha256': sha(pre)}}
    mp = tmp_path / 'manifest.json'
    write_json(mp, manifest)
    monkeypatch.setattr(target, 'ROOT', root)
    monkeypatch.setattr(target, 'OUTPUT', out)
    monkeypatch.setattr(target, 'DATASET_SHA256', sha(dataset))
    return SimpleNamespace(root=root, out=out, dataset=dataset, data=data, manifest=manifest,
                           mp=mp, pre=pre, preflight=preflight, target=target)


def repin(bundle):
    write_json(bundle.pre, bundle.preflight)
    bundle.manifest['token_preflight']['sha256'] = sha(bundle.pre)
    write_json(bundle.mp, bundle.manifest)


def test_preflight_pins_source_and_all_raw_1540_before_any_execution(bundle):
    before = {str(p): p.read_bytes() for p in bundle.root.rglob('*') if p.is_file()}
    prepared = bundle.target.preflight(bundle.mp, sha(bundle.mp))
    assert prepared['command'] == [bundle.target.PYTHON, '-u', str(bundle.root / 'scripts/run_system_v15.py'),
        '--comparison', '--dataset', 'locomo', '--configs', 'full,no_adaptive,no_residual',
        '--budgets', '400,1600,4000', '--out', str(bundle.out)]
    assert len(prepared['questions']) == 1540
    assert not bundle.out.exists()
    assert before == {str(p): p.read_bytes() for p in bundle.root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['manifest_hash', 'file_hash', 'missing_pin', 'new_source',
    'output_exists', 'output_foreign', 'dataset_path', 'dataset_hash', 'token_hash',
    'token_overflow', 'token_sum', 'qa_count', 'row_duplicate', 'all_fit', 'model',
    'revision', 'max_len', 'cap', 'raw_guard', 'omit_count', 'unknown_category'])
def test_invalid_inputs_fail_before_output_or_child(bundle, monkeypatch, change):
    b, t = bundle, bundle.target
    if change == 'file_hash': b.manifest['files_sha256'][str(b.dataset)] = '0' * 64
    elif change == 'missing_pin': b.manifest['files_sha256'].pop(str(b.root / 'certmem/locomo.py'))
    elif change == 'new_source': (b.root / 'untracked.py').write_text('pass')
    elif change == 'output_exists': b.out.mkdir()
    elif change == 'output_foreign': b.manifest['output'] = str(b.root / 'other')
    elif change == 'dataset_path': b.manifest['dataset'] = str(b.root / 'other.json')
    elif change == 'dataset_hash': monkeypatch.setattr(t, 'DATASET_SHA256', '0' * 64)
    elif change == 'token_overflow': b.preflight['rows'][0].update(max_chat_prompt_tokens=50000, max_total_tokens=50032)
    elif change == 'token_sum': b.preflight['rows'][0]['max_total_tokens'] += 1
    elif change == 'qa_count': b.preflight['rows'][0]['qa_count'] -= 1
    elif change == 'row_duplicate': b.preflight['rows'][-1] = b.preflight['rows'][0].copy()
    elif change == 'all_fit': b.preflight['all_fit'] = False
    elif change == 'model': b.preflight['model'] = 'other'
    elif change == 'revision': b.preflight['revision'] = 'other'
    elif change == 'max_len': b.preflight['max_model_len'] = 65536
    elif change == 'cap': b.preflight['reader_max_tokens'] = 31
    elif change == 'raw_guard': b.preflight['rows'][0]['original_full_raw_included'] = False
    elif change == 'omit_count': b.preflight['original_full_raw_omitted_qa'] = 0
    elif change == 'unknown_category':
        b.data[0]['qa'][0]['category'] = 0
        write_json(b.dataset, b.data)
        b.manifest['files_sha256'][str(b.dataset)] = sha(b.dataset)
        monkeypatch.setattr(t, 'DATASET_SHA256', sha(b.dataset))
        b.preflight['dataset_sha256'] = sha(b.dataset)
    repin(b)
    if change == 'token_hash':
        b.pre.write_text('{}')
    monkeypatch.setattr(t.subprocess, 'Popen', lambda *a, **kw: pytest.fail('child started'))
    with pytest.raises((ValueError, FileExistsError)):
        t.run(b.mp, '0' * 64 if change == 'manifest_hash' else sha(b.mp))
    assert b.out.exists() is (change == 'output_exists')


def items(bundle, change=None):
    prepared = bundle.target.preflight(bundle.mp, sha(bundle.mp))
    rows = []
    for (cid, ordinal), qa in prepared['questions'].items():
        for config, policies in bundle.target.POLICIES.items():
            for policy in policies:
                rows.append({'conv_id': cid, 'question_ordinal': ordinal, 'config': config,
                    'policy': policy, 'category': qa['category'], 'question': qa['question'],
                    'gold': str(qa.get('answer', '')), 'pred': 'answer'})
    if change == 'missing_fullraw': rows = [r for r in rows if not (r['policy'] == 'full_raw' and r['conv_id'] in OMITTED)]
    elif change == 'duplicate': rows[-1] = rows[0].copy()
    elif change == 'empty': rows[0]['pred'] = ' '
    elif change == 'metadata': rows[0]['gold'] = 'changed'
    return prepared, rows


def write_items(out, rows):
    with (out / 'items.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ('summary.csv', 'write_stats.csv'):
        (out / name).write_text('native\nresult\n')


@pytest.mark.parametrize('change', [None, 'missing_fullraw', 'duplicate', 'empty', 'metadata'])
def test_native_output_matrix_requires_all_56980_without_rewriting(bundle, change):
    prepared, rows = items(bundle, change)
    bundle.out.mkdir()
    write_items(bundle.out, rows)
    before = (bundle.out / 'items.csv').read_bytes()
    result = bundle.target.audit_outputs(bundle.out, prepared['questions'])
    assert result['complete'] is (change in (None, 'empty'))
    assert result['expected_rows'] == 56980
    assert result['empty_predictions'] == (1 if change == 'empty' else 0)
    if change is None: assert result['full_raw_rows'] == 1540
    assert (bundle.out / 'items.csv').read_bytes() == before


@pytest.mark.parametrize('exit_code,interrupt,start_error', [(0, False, False), (7, False, False),
                                                            (-15, True, False), (None, False, True)])
def test_foreground_wait_failure_receipt_and_own_child_signal(bundle, monkeypatch, exit_code, interrupt, start_error):
    t, b = bundle.target, bundle
    prepared, rows = items(b)
    handlers, calls, forwarded = {}, [], []
    monkeypatch.setattr(t.signal, 'signal', lambda sig, fn: handlers.setdefault(sig, fn))
    monkeypatch.setattr(t.signal, 'getsignal', lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(t, 'gpu_sample', lambda: {'power_w': 200.0, 'utilization_percent': 50.0, 'memory_mib': 9000.0})
    monkeypatch.setattr(t, 'forward_signal', lambda child, sig: forwarded.append((child, sig)))
    class Child:
        pid = 123
        attempts = 0
        def wait(self, timeout=None):
            self.attempts += 1
            if self.attempts == 1:
                if interrupt: handlers[signal.SIGTERM](signal.SIGTERM, None)
                raise subprocess.TimeoutExpired('fixture', 5)
            if exit_code == 0: write_items(b.out, rows)
            return exit_code
    child = Child()
    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        if start_error: raise OSError('fixture spawn failed')
        return child
    monkeypatch.setattr(t.subprocess, 'Popen', spawn)
    code = t.run(b.mp, sha(b.mp))
    receipt = json.loads((b.out / 'receipt.json').read_text(encoding='utf-8'))
    assert code == (0 if exit_code == 0 else 1)
    assert receipt['child_exit_code'] == exit_code
    assert receipt['inference_complete'] is (exit_code == 0)
    assert receipt['complete'] is False and receipt['official_scoring_complete'] is False
    assert receipt['usage_complete'] is None and receipt['exact_tokens'] is None
    assert receipt['runtime']['estimated_rental_usd'] == pytest.approx(receipt['runtime']['wall_seconds'] / 3600 * t.HOURLY_RATE)
    assert receipt['runtime']['exclusive_gpu_energy_wh'] is None
    assert calls[0][0] == prepared['command']
    assert calls[0][1]['cwd'] == str(b.root)
    assert calls[0][1]['env']['HF_HUB_OFFLINE'] == '1'
    assert calls[0][1]['env']['OMP_NUM_THREADS'] == '4'
    assert len(calls) == 1
    assert bool(forwarded) is interrupt
    assert (b.out / 'benchmark.log').exists()


@pytest.mark.parametrize('output,valid', [('200, 80, 12000\n', True), ('N/A, 0, 0', False),
                                        ('nan, 20, 200', False), ('-1, 0, 0', False),
                                        ('10, 0, 0\n20, 0, 0', False)])
def test_gpu_samples_unknown_not_zero(target, monkeypatch, output, valid):
    monkeypatch.setattr(target.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=output))
    result = target.gpu_sample()
    assert (result is not None) is valid


def test_gpu_unavailable_and_energy_gaps_remain_unknown(target, monkeypatch):
    def unavailable(*args, **kwargs): raise OSError('not installed')
    monkeypatch.setattr(target.subprocess, 'run', unavailable)
    assert target.gpu_sample() is None
    assert target.energy_wh([{'elapsed_s': 0, 'gpu': None}, {'elapsed_s': 5, 'gpu': {'power_w': 200}}]) is None
    assert target.energy_wh([]) is None
    assert target.energy_wh([{'elapsed_s': 0, 'gpu': {'power_w': 100}},
                             {'elapsed_s': 3600, 'gpu': {'power_w': 200}}]) == 150


def test_cli_passes_only_manifest_and_hash(target, monkeypatch):
    seen = []
    monkeypatch.setattr(target, 'run', lambda *args: seen.append(args) or 1)
    assert target.main(['--manifest', 'input.json', '--expected-manifest-sha256', 'a' * 64]) == 1
    assert seen == [(Path('input.json'), 'a' * 64)]


@pytest.mark.parametrize('platform', ['posix', 'nt', 'exited'])
def test_signal_only_targets_owned_child(target, monkeypatch, platform):
    sent = []
    child = SimpleNamespace(pid=654, send_signal=lambda sig: sent.append(('child', sig)))
    def killpg(pid, sig):
        if platform == 'exited': raise ProcessLookupError()
        sent.append((pid, sig))
    monkeypatch.setattr(target, 'os', SimpleNamespace(name='nt' if platform == 'nt' else 'posix', killpg=killpg))
    target.forward_signal(child, signal.SIGTERM)
    assert sent == ([] if platform == 'exited' else [('child' if platform == 'nt' else 654, signal.SIGTERM)])


def test_source_mutation_and_monitor_error_preserve_failed_receipt(bundle, monkeypatch):
    t, b, forwarded = bundle.target, bundle, []
    class Child:
        calls = 0
        def wait(self, timeout=None):
            self.calls += 1
            if timeout is not None: raise subprocess.TimeoutExpired('fixture', 5)
            return -15
    child = Child()
    monkeypatch.setattr(t.subprocess, 'Popen', lambda *a, **k: child)
    monkeypatch.setattr(t.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(t, 'forward_signal', lambda proc, sig: forwarded.append((proc, sig)))
    samples = []
    def sample():
        samples.append(1)
        if len(samples) == 2:
            (b.root / 'certmem/locomo.py').write_text('changed')
            raise OSError('fixture monitor failed')
    monkeypatch.setattr(t, 'gpu_sample', sample)
    assert t.run(b.mp, sha(b.mp)) == 1
    receipt = json.loads((b.out / 'receipt.json').read_text(encoding='utf-8'))
    assert receipt['child_exit_code'] == -15
    assert receipt['source_unchanged'] is False and receipt['inference_complete'] is False
    assert receipt['runtime']['sampled_device_energy_wh'] is None
    assert forwarded == [(child, signal.SIGTERM)]
    assert child.calls == 2


@pytest.mark.parametrize('bad', ['duplicate_conversation', 'missing_conversation', 'empty_pins', 'missing_preflight_row'])
def test_structural_preflight_gates(bundle, monkeypatch, bad):
    b, t = bundle, bundle.target
    if bad == 'duplicate_conversation': b.data[-1] = b.data[0]
    elif bad == 'missing_conversation': b.data.pop()
    elif bad == 'empty_pins': b.manifest['files_sha256'] = {}
    elif bad == 'missing_preflight_row': b.preflight['rows'].pop()
    if bad.endswith('conversation'):
        write_json(b.dataset, b.data)
        b.manifest['files_sha256'][str(b.dataset)] = sha(b.dataset)
        monkeypatch.setattr(t, 'DATASET_SHA256', sha(b.dataset))
        b.preflight['dataset_sha256'] = sha(b.dataset)
    repin(b)
    with pytest.raises(ValueError): t.preflight(b.mp, sha(b.mp))


def test_empty_native_output_fails_closed(bundle):
    prepared = bundle.target.preflight(bundle.mp, sha(bundle.mp))
    bundle.out.mkdir()
    (bundle.out / 'items.csv').touch()
    assert bundle.target.audit_outputs(bundle.out, prepared['questions'])['complete'] is False


def test_interrupted_partial_csv_preserves_failed_receipt(bundle, monkeypatch):
    t, b = bundle.target, bundle
    partial = 'conv_id,question_ordinal,config,policy,category,question,gold,pred\nconv-26\n'
    class Child:
        def wait(self, timeout=None):
            (b.out / 'items.csv').write_text(partial, encoding='utf-8')
            for name in ('summary.csv', 'write_stats.csv'):
                (b.out / name).write_text('native\npartial\n', encoding='utf-8')
            return -15
    monkeypatch.setattr(t.subprocess, 'Popen', lambda *args, **kwargs: Child())
    monkeypatch.setattr(t, 'gpu_sample', lambda: None)
    monkeypatch.setattr(t.signal, 'signal', lambda *args: None)
    assert t.run(b.mp, sha(b.mp)) == 1
    receipt = json.loads((b.out / 'receipt.json').read_text(encoding='utf-8'))
    assert receipt['child_exit_code'] == -15 and receipt['inference_complete'] is False
    assert receipt['native_outputs']['complete'] is False
    assert receipt['native_outputs']['issues']
    assert (b.out / 'items.csv').read_text(encoding='utf-8') == partial
