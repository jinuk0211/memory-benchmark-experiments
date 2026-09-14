"""Real-model smoke tests must not modify the original benchmark or accept partial QA."""

import copy
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_locomo_live_probes.py'
    assert path.is_file(), 'Missing isolated, audited live-probe runner'
    spec = importlib.util.spec_from_file_location('live_probes', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_prefix_preserves_qa_ids_timestamps_and_original():
    original = [{'sample_id': '26', 'qa': [{'category': 5}, {'category': 1}],
                 'conversation': {'speaker_a': 'A', 'session_1_date_time': 'date1',
                                  'session_1': [1, 2], 'session_2_date_time': 'date2',
                                  'session_2': [3, 4], 'session_3_date_time': 'date3',
                                  'session_3': [5]}}]
    before = copy.deepcopy(original)
    result = module().prefix_dataset(original, 3)
    assert original == before
    assert result[0]['qa'] == original[0]['qa']
    assert result[0]['conversation'] == {'speaker_a': 'A', 'session_1_date_time': 'date1',
                                       'session_1': [1, 2], 'session_2_date_time': 'date2',
                                       'session_2': [3]}


def test_full_first_conversation_is_unchanged():
    original = [{'sample_id': '26', 'qa': [], 'conversation': {'session_1': [1, 2]}}]
    assert module().prefix_dataset(original, None) == original


def test_probe_can_target_known_failed_conversation_without_reordering_source():
    source = [{'sample_id': str(i), 'qa': [{'question_id': str(i)}],
               'conversation': {'session_1': [1, 2], 'session_1_date_time': 'date'}} for i in range(7)]
    before = copy.deepcopy(source)
    result = module().prefix_dataset(source, 1, sample_index=6)
    assert result[0]['sample_id'] == '6'
    assert result[0]['qa'] == source[6]['qa']
    assert result[0]['conversation']['session_1'] == [1]
    assert source == before


@pytest.mark.parametrize('index', [-1, True, 1, '0'])
def test_probe_rejects_invalid_sample_index(index):
    with pytest.raises(ValueError):
        module().prefix_dataset([{'conversation': {}}], 1, sample_index=index)


@pytest.mark.parametrize('count', [0, -1, True])
def test_invalid_probe_turn_count_is_rejected(count):
    with pytest.raises(ValueError):
        module().prefix_dataset([{'conversation': {}}], count)


def test_probe_gate_rejects_empty_incomplete_or_failed_usage():
    check = module().check_gate
    valid = {'complete': True, 'failed_requests': 0, 'truncated_requests': 0}
    check([{'prediction': 'answer'}] * 3, [], [], valid)
    for rows, issues, missing, usage in [
        ([{'prediction': ''}] * 3, [], [], valid),
        ([{'prediction': 'answer'}] * 2, [], [], valid),
        ([{'prediction': 'answer'}] * 3, ['bad'], [], valid),
        ([{'prediction': 'answer'}] * 3, [], ['missing'], valid),
        ([{'prediction': 'answer'}] * 3, [], [], dict(valid, complete=False)),
        ([{'prediction': 'answer'}] * 3, [], [], dict(valid, failed_requests=1)),
        ([{'prediction': 'answer'}] * 3, [], [], dict(valid, truncated_requests=1)),
    ]:
        with pytest.raises(ValueError):
            check(rows, issues, missing, usage)


def test_full_run_repair_gate_accepts_only_audited_terminal_responses():
    check = module().check_gate
    rows = [{'prediction': 'complete answer'}] * 3
    usage = {'complete': True, 'failed_requests': 1, 'truncated_requests': 1}
    check(rows, [], [], usage, {'complete': True})
    with pytest.raises(ValueError):
        check(rows, [], [], usage, {'complete': False})
    with pytest.raises(ValueError):
        check(rows, [], [], dict(usage, complete=False), {'complete': True})


@pytest.mark.parametrize('sample_index', [0, 1])
@pytest.mark.parametrize('failure', [None, 'exit', 'missing_result', 'bad_usage', 'bad_embedding'])
def test_live_probe_orchestration_is_separate_and_fail_closed(tmp_path, monkeypatch, failure, sample_index):
    target = module()
    source = tmp_path / 'original.json'
    source.write_text(json.dumps([{'sample_id': sample, 'qa': [], 'conversation': {'session_1': [1, 2]}}
                                  for sample in ['26', '47']]))
    agent = tmp_path / 'agent.yaml'
    agent.write_text('agent_name: test')
    config = tmp_path / 'dataset.yaml'
    config.write_text('sub_dataset: locomo_qa')
    output = tmp_path / 'diagnostics'
    output.mkdir()
    plan = {'output': str(output), 'dataset': str(source), 'run_id': 'test-only', 'hf_home': 'hf',
                'client_python': 'client', 'server_python': 'server', 'file_sha256': {},
                'methods': [{'method': 'a_mem', 'agent_config': str(agent), 'dataset_config': str(config)}]}
    expected = {(sample, n): {} for sample in ['26', '47'] for n in [1, 3, 4]}
    monkeypatch.setattr(target.queue, 'expected_questions', lambda *args: expected)
    calls = []

    @contextmanager
    def service(command, log, env=None):
        calls.append((command, env))
        if '--artifact_root' in command and failure != 'missing_result':
            artifact = Path(command[command.index('--artifact_root') + 1])
            artifact.mkdir()
            (artifact / 'probe_results.json').write_text('{"data": []}')
        polls = iter([None, 0])
        yield SimpleNamespace(poll=lambda: next(polls), wait=lambda **kwargs: None,
                              returncode=1 if failure == 'exit' else 0)

    monkeypatch.setattr(target.queue, 'service', service)
    monkeypatch.setattr(target.queue, 'wait_health', lambda *args: None)
    monkeypatch.setattr(target.queue, 'drain_proxies', lambda processes: calls.append(('drained', None)))
    monkeypatch.setattr(target.queue, 'gpu_sample', lambda: {'unavailable': True})
    monkeypatch.setattr(target, 'read_jsonl', lambda path:
        [{'request_kind': 'embedding', 'success': False}]
        if failure == 'bad_embedding' and Path(path).name == 'embedding_usage.jsonl' else [])
    def normalize(records, wanted, template):
        assert {key[0] for key in wanted} == {['26', '47'][sample_index]}
        return records
    monkeypatch.setattr(target, 'normalize_records', normalize)
    monkeypatch.setattr(target, 'get_template', lambda *args: '{question}')
    predictions = [{'prediction': 'answer'}] * 3
    monkeypatch.setattr(target.queue, 'canonical_predictions', lambda *args: predictions)
    monkeypatch.setattr(target, 'validate_predictions', lambda *args: (predictions, [], []))
    monkeypatch.setattr(target.queue, 'audit_coverage', lambda *args: None)
    monkeypatch.setattr(target, 'audit_delivery', lambda *args: {'complete': failure != 'bad_usage'})
    monkeypatch.setattr(target, 'summarize_usage', lambda *args: {'complete': True, 'truncated_requests': 0,
                                                                 'failed_requests': int(failure in ('bad_usage', 'bad_embedding'))})
    before = source.read_bytes()
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            target.run_probe(plan, 'a_mem', 1, sample_index=sample_index)
        assert not (output / 'a_mem/report.json').exists()
    else:
        result = target.run_probe(plan, 'a_mem', 1, sample_index=sample_index)
        assert result['passed'] and result['diagnostic_only']
        assert result['memory_turns'] == 1
        assert result['sample_index'] == sample_index
        assert result['sample_id'] == ['26', '47'][sample_index]
        command, env = next(item for item in calls if '--artifact_root' in item[0])
        assert command[command.index('--max_test_queries_ablation') + 1] == '3'
        assert env['METER_RUN_ID'] == 'test-only'
        assert env['BASELINE_STRICT_COMPARISON'] == '1'
        meter_commands = [item[0] for item in calls if '--journal' in item[0]]
        assert all(any(str(arg).endswith('repairing_openai_proxy.py') for arg in command)
                   for command in meter_commands)
        assert all(command[command.index('--timeout') + 1] == '1200' for command in meter_commands)
    assert source.read_bytes() == before
    assert ('drained', None) in calls


@pytest.mark.parametrize('failed_method', [None, 'mem0'])
def test_all_three_are_tested_without_launching_full_run(tmp_path, monkeypatch, failed_method):
    target = module()
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps({'output': str(tmp_path), 'server_python': 'server', 'embedding_path': 'embedding'}))
    monkeypatch.setattr(target.sys, 'argv', ['probe', '--plan', str(plan_path)])
    monkeypatch.setattr(target.queue, 'validate_plan', lambda plan: None)
    monkeypatch.setattr(target.queue, 'wait_vllm', lambda plan: None)
    monkeypatch.setattr(target.queue, 'wait_health', lambda *args: None)

    @contextmanager
    def service(*args):
        yield object()

    monkeypatch.setattr(target.queue, 'service', service)
    tested = []

    def probe(plan, method, turns):
        tested.append((method, turns))
        if method == failed_method:
            raise ValueError('real failure')
        return {'method': method, 'passed': True}

    monkeypatch.setattr(target, 'run_probe', probe)
    assert target.main() == (1 if failed_method else 0)
    assert tested == list(target.PROBES)
    status = json.loads((tmp_path / 'probe_status.json').read_text())
    assert status['passed'] == (failed_method is None)
