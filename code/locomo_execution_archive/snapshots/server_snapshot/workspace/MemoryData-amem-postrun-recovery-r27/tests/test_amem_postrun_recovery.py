"""Offline recovery tests; model/service calls are mocked, old artifacts stay unchanged."""
import copy
import hashlib
import importlib
import json
import os
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('AMEM_R26_TEST_SOURCE', 'D:/MemoryData/amem_r26_deployment/source'))
ARCHIVE = Path(os.environ.get('AMEM_R26_TEST_ARCHIVE',
    'D:/MemoryData/amem-r26-postrun-failure-artifacts-50156979.tar.gz'))
DATASET = Path(os.environ.get('AMEM_R26_TEST_DATASET', 'D:/MemoryData/HiGMem/data/locomo10.json'))
sys.path[:0] = [str(ROOT), str(SOURCE)]


def module():
    return importlib.import_module('run_amem_postrun_recovery')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def probe(tmp_path):
    m = module()
    driver = importlib.import_module('scripts.amem_original_policy_recovery_r26')
    old = tmp_path / 'closed-probe'
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        for number, member in enumerate(archive):
            if member.isfile() and member.name.startswith('locomo-amem-original-policy-probe-r26/'):
                target = old / member.name.split('/', 1)[1]
                # Test-only relocation avoids Windows MAX_PATH; all native bytes survive.
                if '/a_mem/artifacts/' in member.name:
                    name = 'saved_results.json' if member.name.endswith('_results.json') else f'{number}.bin'
                    target = old / 'a_mem/artifacts' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    out = old / 'a_mem'
    config = driver.yaml.safe_load((out / 'dataset.yaml').read_text())
    config['test_files'] = str(out / 'test_dataset.json')
    write(out / 'dataset.yaml', config)
    result = next((out / 'artifacts').rglob('*_results.json'))
    closure = {'schema_version': 1, 'closure_kind': 'probe-postrun-import-failure',
        'probe_root': str(old), 'output': str(out), 'run_id': driver.PROBE_RUN_ID,
        'native_exit_code': 1, 'supervisor_state': 'EXITED', 'known_benchmark_writer_pids': [],
        'global_writer_absence_proven': False, 'uninspected_processes': [925],
        'benchmark_complete': False, 'full_output_exists': False,
        'original_failure_preserved': True, 'memory_turns': 80, 'qa_count': 3,
        'result_path': str(result), 'result_sha256': sha(result),
        'benchmark_log_path': str(out / 'benchmark.log'), 'benchmark_log_sha256': sha(out / 'benchmark.log'),
        'recovery_status_path': str(old / 'recovery_status.json'),
        'recovery_status_sha256': sha(old / 'recovery_status.json')}
    closure['raw_source_sha256'] = m.inventory(old)
    plan = {'dataset': str(DATASET), 'run_id': driver.RUN_ID,
        'methods': [{'method': 'a_mem', 'agent_config': str(SOURCE / 'comparison_config/a_mem.yaml'),
                    'dataset_config': str(SOURCE / 'comparison_config/locomo.yaml')}]}
    return SimpleNamespace(m=m, driver=driver, old=old, out=out, closure=closure, plan=plan,
                           destination=tmp_path / 'recovery', result=result)


def test_recover_existing_three_answers_without_writing_old_probe(probe):
    probe.destination.mkdir()
    before = probe.m.inventory(probe.old)
    report = probe.m.validate_probe(probe.plan, probe.closure, probe.driver, probe.destination)
    assert report['passed'] is True and report['qa_count'] == 3 and report['memory_turns'] == 80
    assert report['usage']['exact_tokens']['total_tokens'] == 297045
    assert report['original_native_exit_code'] == 1
    assert report['global_writer_absence_proven'] is False
    assert report['new_inference_calls'] == 0
    assert probe.m.inventory(probe.old) == before


@pytest.mark.parametrize('change', ['empty', 'query', 'question_id', 'query_id', 'context_id',
                                  'failed', 'metadata', 'duplicate', 'missing'])
def test_bad_saved_qa_never_restores_probe_gate(probe, change):
    data = json.loads(probe.result.read_text())
    row = data['data'][0]
    if change == 'empty':
        row['output'] = ' '
    elif change == 'query':
        row['query'] += ' altered'
    elif change == 'question_id':
        row['question_id'] = 'conv-26_qa99'
    elif change == 'query_id':
        row['query_id'] = 7
    elif change == 'context_id':
        row['context_id'] = 1
    elif change == 'failed':
        row['status'] = 'failed'
    elif change == 'metadata':
        row['eval_metadata']['sample_id'] = 'conv-30'
    elif change == 'duplicate':
        data['data'][1] = copy.deepcopy(row)
    else:
        data['data'].pop()
    write(probe.result, data)
    probe.closure['result_sha256'] = sha(probe.result)
    probe.closure['raw_source_sha256'] = probe.m.inventory(probe.old)
    probe.destination.mkdir()
    with pytest.raises(ValueError):
        probe.m.validate_probe(probe.plan, probe.closure, probe.driver, probe.destination)


@pytest.mark.parametrize('change', ['prefix', 'config', 'usage', 'event', 'coverage', 'traceback', 'exit'])
def test_native_probe_inputs_delivery_and_original_failure_are_required(probe, change):
    if change == 'prefix':
        path = probe.out / 'test_dataset.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        data[0]['conversation']['session_1'].pop()
        write(path, data)
    elif change == 'config':
        path = probe.out / 'dataset.yaml'
        data = json.loads(path.read_text())
        data['generation_max_length'] = 1
        write(path, data)
    elif change in {'usage', 'coverage'}:
        path = probe.out / 'llm_usage.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == 'usage':
            rows[-1]['usage'] = None
        else:
            rows[-1]['question_id'] = 99
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    elif change == 'event':
        next((probe.out / 'semantic_outcomes').glob('*.jsonl')).write_text('')
    elif change == 'traceback':
        path = Path(probe.closure['benchmark_log_path'])
        path.write_text('unrelated error')
        probe.closure['benchmark_log_sha256'] = sha(path)
    else:
        probe.closure['native_exit_code'] = 0
    probe.closure['raw_source_sha256'] = probe.m.inventory(probe.old)
    probe.destination.mkdir()
    with pytest.raises((ValueError, TypeError)):
        probe.m.validate_probe(probe.plan, probe.closure, probe.driver, probe.destination)


def test_inventory_rejects_changed_or_added_old_artifacts(probe):
    (probe.old / 'new.txt').write_text('writer')
    with pytest.raises(ValueError, match='inventory'):
        probe.m.verify_probe_inventory(probe.closure)


def test_alias_is_exclusive_byte_copy_and_never_rewrites_native_output(tmp_path):
    m = module()
    raw = tmp_path / 'parallel_results.json'
    raw.write_bytes(b'{ "data": [] }\n')
    report = m.create_result_alias(tmp_path)
    assert (tmp_path / 'selected_results.json').read_bytes() == raw.read_bytes()
    assert report['sha256'] == sha(raw)
    with pytest.raises(FileExistsError):
        m.create_result_alias(tmp_path)


def test_cpu_postrun_uses_only_scoped_native_import_and_records_result(tmp_path, monkeypatch):
    m = module()
    seen = []
    def run(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='AMEM_NATIVE_POSTRUN_NOOP\n', stderr='')
    monkeypatch.setattr(m.subprocess, 'run', run)
    monkeypatch.setenv('PYTHONPATH', 'parent-unmodified')
    closure = {'helper_path': '/helper/evaluation/longmemeval/memoryagentbench_longmemeval_recall.py',
               'output': '/probe/a_mem', 'result_path': '/probe/results.json'}
    report = m.recover_postrun({'source_root': '/frozen', 'client_python': '/python'}, closure, tmp_path)
    command, kwargs = seen[0]
    assert command[0:3] == ['/python', '-B', '-c']
    assert 'import main' in command[3] and '_maybe_generate_post_run_reports' in command[3]
    assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert kwargs['env']['PYTHONPATH'] == os.pathsep.join(['/frozen', str(Path('/helper'))])
    assert os.environ['PYTHONPATH'] == 'parent-unmodified'
    assert report['exit_code'] == 0 and report['new_inference_calls'] == 0


@pytest.mark.parametrize('returncode,stdout', [(1, ''), (0, 'wrong marker')])
def test_cpu_failure_never_restores_gate(tmp_path, monkeypatch, returncode, stdout):
    m = module()
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **k:
        SimpleNamespace(returncode=returncode, stdout=stdout, stderr='failure'))
    with pytest.raises(RuntimeError):
        m.recover_postrun({'source_root': '/frozen', 'client_python': '/python'},
            {'helper_path': '/helper/evaluation/longmemeval/h.py', 'output': '/probe', 'result_path': '/r'}, tmp_path)
    assert (tmp_path / 'postrun_stderr.log').read_text() == 'failure'


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    m = module()
    calls = []
    out = tmp_path / 'full'
    recovery = tmp_path / 'recovery'
    manifest = {'recovery_output': str(recovery), 'recovery_run_id': m.RECOVERY_RUN_ID}
    plan = {'output': str(out), 'run_id': 'native-r26', 'source_root': '/frozen',
            'server_python': '/python', 'embedding_path': '/minilm', 'dataset': '/dataset',
            'methods': [{'method': 'a_mem'}]}
    closure = {'raw_source_sha256': {}, 'probe_root': str(tmp_path / 'probe'),
               'global_writer_absence_proven': False, 'uninspected_processes': [925]}
    @contextmanager
    def service(*args):
        calls.append('encoder')
        yield object()
    @contextmanager
    def policy(path):
        calls.append(('policy', path))
        yield
    def run_method(plan, item, expected, *, context_indices):
        assert context_indices == list(range(10)) and isinstance(context_indices, list)
        assert item == {'method': 'a_mem'}
        calls.append('full')
        (out / 'a_mem').mkdir()
        (out / 'a_mem/parallel_results.json').write_bytes(b'{"data":[]}')
    queue = SimpleNamespace(expected_questions=lambda *a: {'original': 1540},
        supervisor_state=lambda name: 'EXITED', service=service, wait_vllm=lambda p: calls.append('vllm'),
        wait_health=lambda *a: None, run_method=run_method)
    driver = SimpleNamespace(queue=queue, require_policy=lambda p: calls.append('policy_check'),
        require_free_ports=lambda: None, policy_environment=policy, PREDECESSORS=(),
        finalize_attempt=lambda *a: {'benchmark_complete': True})
    monkeypatch.setattr(m, 'load_inputs', lambda *a: (manifest, plan, closure, driver))
    monkeypatch.setattr(m, 'verify_probe_inventory', lambda *a: calls.append('inventory'))
    monkeypatch.setattr(m, 'verify_prior_artifacts', lambda *a: calls.append('prior'))
    monkeypatch.setattr(m, 'validate_probe', lambda *a: {'passed': True})
    monkeypatch.setattr(m, 'recover_postrun', lambda *a: calls.append('cpu') or {'exit_code': 0})
    return SimpleNamespace(m=m, plan=plan, closure=closure, driver=driver, calls=calls,
                           out=out, recovery=recovery)


def test_full_starts_after_saved_gate_cpu_only_no_probe_rerun(orchestration):
    f = orchestration
    assert f.m.run_recovery(Path('manifest'), 'a' * 64) == 0
    assert f.calls.index('cpu') < f.calls.index('vllm') < f.calls.index('full')
    assert f.calls.count('full') == 1
    receipt = json.loads((f.recovery / 'recovery_receipt.json').read_text())
    assert receipt['native_run_id'] == 'native-r26'
    assert receipt['benchmark_complete'] is True and receipt['complete'] is False
    assert receipt['probe_inference_repeated'] is False


def test_failed_gate_cannot_start_full(orchestration, monkeypatch):
    f = orchestration
    monkeypatch.setattr(f.m, 'validate_probe', Mock(side_effect=ValueError('bad probe')))
    with pytest.raises(ValueError, match='bad probe'):
        f.m.run_recovery(Path('manifest'), 'a' * 64)
    assert 'cpu' not in f.calls and 'full' not in f.calls and not f.out.exists()
    assert json.loads((f.recovery / 'recovery_receipt.json').read_text())['state'] == 'failed'


@pytest.mark.parametrize('existing', ['recovery', 'out'])
def test_existing_output_never_overwritten(orchestration, existing):
    f = orchestration
    getattr(f, existing).mkdir()
    with pytest.raises(FileExistsError):
        f.m.run_recovery(Path('manifest'), 'a' * 64)
    assert 'cpu' not in f.calls and 'full' not in f.calls


def test_incomplete_full_preserves_failure_receipt(orchestration):
    f = orchestration
    f.driver.finalize_attempt = lambda *a: {'benchmark_complete': False}
    with pytest.raises(RuntimeError, match='incomplete'):
        f.m.run_recovery(Path('manifest'), 'a' * 64)
    assert json.loads((f.recovery / 'recovery_receipt.json').read_text())['benchmark_complete'] is False


@pytest.fixture
def input_bundle(tmp_path, monkeypatch):
    m = module()
    source = tmp_path / 'frozen'
    source.mkdir()
    sourcefile = source / 'scripts/amem_original_policy_recovery_r26.py'
    sourcefile.parent.mkdir()
    sourcefile.write_text('frozen fixture')
    helper = tmp_path / 'helper/evaluation/longmemeval/helper.py'
    helper.parent.mkdir(parents=True)
    helper.write_text('original helper fixture')
    old = tmp_path / 'probe'
    old.mkdir()
    (old / 'raw').write_text('untouched')
    plan = {'source_root': str(source), 'file_sha256': {str(sourcefile): sha(sourcefile)}}
    planpath, closurepath = tmp_path / 'plan.json', tmp_path / 'closure.json'
    write(planpath, plan)
    closure = {'source_plan_path': str(planpath), 'source_plan_sha256': sha(planpath),
        'helper_sha256': sha(helper), 'helper_path': str(helper),
        'supplementary_package_files': {str(helper): sha(helper), str(helper.parent / '__init__.py'): None},
        'probe_root': str(old), 'raw_source_sha256': m.inventory(old)}
    write(closurepath, closure)
    manifest = {'schema_version': 1, 'recovery_run_id': m.RECOVERY_RUN_ID,
        'source_root': str(source), 'recovery_output': m.RECOVERY_OUTPUT,
        'source_plan': {'path': str(planpath), 'sha256': sha(planpath)},
        'probe_closure': {'path': str(closurepath), 'sha256': sha(closurepath)},
        'file_sha256': {str(Path(m.__file__).resolve()): sha(Path(m.__file__))}}
    path = tmp_path / 'manifest.json'
    write(path, manifest)
    for name, value in [('SOURCE_ROOT', str(source)), ('PLAN_SHA', sha(planpath)),
                        ('CLOSURE_SHA', sha(closurepath)), ('HELPER_SHA', sha(helper))]:
        monkeypatch.setattr(m, name, value)
    driver = SimpleNamespace(__file__=str(sourcefile), require_policy=Mock())
    monkeypatch.setattr(m.importlib, 'import_module', lambda name: driver)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    return SimpleNamespace(m=m, path=path, manifest=manifest, driver=driver,
                           sourcefile=sourcefile, helper=helper, old=old)


def test_additive_manifest_validates_all_inputs_before_import(input_bundle):
    f = input_bundle
    manifest, plan, closure, driver = f.m.load_inputs(f.path, sha(f.path))
    assert manifest == f.manifest and driver is f.driver
    driver.require_policy.assert_called_once_with(plan)
    assert closure['raw_source_sha256'] == f.m.inventory(f.old)


@pytest.mark.parametrize('change', ['manifest_hash', 'plan_pin', 'closure_pin', 'runner_pin',
                                  'source_bytes', 'package_initializer', 'scope', 'new_probe_file'])
def test_changed_pins_or_scope_cannot_recover(input_bundle, change):
    f = input_bundle
    expected = sha(f.path)
    if change == 'manifest_hash':
        expected = '0' * 64
    elif change in {'plan_pin', 'closure_pin'}:
        f.manifest['source_plan' if change == 'plan_pin' else 'probe_closure']['sha256'] = '0' * 64
    elif change == 'runner_pin':
        f.manifest['file_sha256'] = {str(f.sourcefile): sha(f.sourcefile)}
    elif change == 'source_bytes':
        f.sourcefile.write_text('changed')
    elif change == 'package_initializer':
        (f.helper.parent / '__init__.py').write_text('side effect')
    elif change == 'scope':
        f.manifest['recovery_output'] = '/workspace/old-output'
    else:
        (f.old / 'new').write_text('writer')
    write(f.path, f.manifest)
    if change != 'manifest_hash':
        expected = sha(f.path)
    with pytest.raises(ValueError):
        f.m.load_inputs(f.path, expected)
    f.driver.require_policy.assert_not_called()


def test_prior_artifacts_are_hash_checked(tmp_path):
    m = module()
    artifact = tmp_path / 'old'
    artifact.write_text('original')
    write(tmp_path / 'prior_artifact_sha256.json', {str(artifact): sha(artifact)})
    m.verify_prior_artifacts({'probe_root': str(tmp_path)})
    artifact.write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        m.verify_prior_artifacts({'probe_root': str(tmp_path)})


def test_cli_requires_manifest_pin_and_dispatches_only_explicit_run(monkeypatch):
    m = module()
    run = Mock(return_value=0)
    monkeypatch.setattr(m, 'run_recovery', run)
    assert m.main(['--manifest', 'additive.json', '--manifest-sha256', 'a' * 64]) == 0
    run.assert_called_once_with(Path('additive.json'), 'a' * 64)


@pytest.mark.parametrize('optimization', [0, 1, 2])
def test_native_postrun_call_cannot_be_removed_by_python_optimization(tmp_path, monkeypatch, optimization):
    m = module()
    config = tmp_path / 'dataset.yaml'
    config.write_text('dataset: LoCoMo')
    native = Mock(return_value=None)
    supports = Mock(return_value=False)
    monkeypatch.setitem(sys.modules, 'main', SimpleNamespace(_maybe_generate_post_run_reports=native))
    monkeypatch.setitem(sys.modules, 'yaml', SimpleNamespace(safe_load=lambda text: {'dataset': 'LoCoMo'}))
    monkeypatch.setitem(sys.modules, 'evaluation.longmemeval.memoryagentbench_longmemeval_recall',
                        SimpleNamespace(supports_memoryagentbench_longmemeval_recall=supports))
    monkeypatch.setattr(sys, 'argv', ['child', '/frozen', '/helper', str(config), '/saved/result.json'])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setenv('PYTHONOPTIMIZE', str(optimization))
    exec(compile(m.POSTRUN_CODE, '<native-postrun-child>', 'exec', optimize=optimization), {})  # noqa: S102 - fixed owned child code, mocked imports
    supports.assert_called_once_with({'dataset': 'LoCoMo'})
    native.assert_called_once_with('/saved/result.json', {'dataset': 'LoCoMo'})
