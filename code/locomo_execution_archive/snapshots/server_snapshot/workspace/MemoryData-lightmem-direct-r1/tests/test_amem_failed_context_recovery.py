"""Do not rerun live or successful A-MEM shards; require a fresh metered smoke."""
from contextlib import contextmanager
import copy
import importlib
import json

import pytest


def terminal_status():
    return {'active_contexts': [], 'failed': True, 'events': [
        *[{'event': 'start', 'context': i, 'pid': 100 + i} for i in range(10)],
        *[{'event': 'end', 'context': i, 'exit_code': 1 if i in (0, 4, 6) else 0}
          for i in range(10)]]}


def test_only_failed_terminal_contexts_are_selected():
    target = importlib.import_module('scripts.amem_failed_context_recovery_r15')
    status = terminal_status()
    original = copy.deepcopy(status)
    assert target.failed_contexts(status) == [0, 4, 6]
    assert status == original


@pytest.mark.parametrize('case', ['active', 'missing_end', 'duplicate_end', 'missing_start',
                                 'invalid_code', 'invalid_index', 'inconsistent_failure'])
def test_ambiguous_or_live_context_status_is_rejected(case):
    target = importlib.import_module('scripts.amem_failed_context_recovery_r15')
    status = terminal_status()
    if case == 'active':
        status['active_contexts'] = [5]
    elif case == 'missing_end':
        status['events'].pop()
    elif case == 'duplicate_end':
        status['events'].append(status['events'][-1])
    elif case == 'missing_start':
        status['events'].pop(0)
    elif case == 'invalid_code':
        status['events'][-1]['exit_code'] = True
    elif case == 'invalid_index':
        status['events'][-1]['context'] = 10
    else:
        status['failed'] = False
    with pytest.raises(ValueError):
        target.failed_contexts(status)


@pytest.mark.parametrize('failure', [None, 'smoke', 'occupied', 'orphan', 'simplemem_orphan', 'existing', 'run', 'unknown_service', 'all_success'])
def test_recovery_waits_for_both_jobs_and_preserves_old_artifacts(tmp_path, monkeypatch, failure):
    target = importlib.import_module('scripts.amem_failed_context_recovery_r15')
    plan_path, full, probe, prior = [tmp_path / x for x in ('plan.json', 'full', 'probe', 'prior')]
    prior.mkdir()
    status = terminal_status()
    if failure == 'all_success':
        status['failed'] = False
        for row in status['events']:
            if row['event'] == 'end':
                row['exit_code'] = 0
    old_file = prior / 'parallel_status.json'
    old_file.write_text(json.dumps(status))
    original = old_file.read_bytes()
    plan_path.write_text(json.dumps({'output': str(full), 'server_python': 'server',
        'embedding_path': 'minilm', 'dataset': 'original1540.json',
        'methods': [{'method': 'a_mem'}]}))
    if failure == 'existing':
        full.mkdir()
    monkeypatch.setattr(target, 'PLAN_PATH', plan_path)
    monkeypatch.setattr(target, 'PRIOR_OUTPUT', prior)
    monkeypatch.setattr(target, 'PROBE_OUTPUT', probe)
    events = []
    monkeypatch.setattr(target.queue, 'validate_plan', lambda plan: events.append('pins'))
    counts = {}
    def state(name):
        counts[name] = counts.get(name, 0) + 1
        if failure == 'unknown_service':
            return 'UNKNOWN'
        return 'RUNNING' if counts[name] == 1 else 'EXITED'
    monkeypatch.setattr(target.queue, 'supervisor_state', state)
    monkeypatch.setattr(target.time, 'sleep', lambda value: events.append('wait'))
    monkeypatch.setattr(target, 'running_writers', lambda path: [456] if (
        failure == 'orphan' or failure == 'simplemem_orphan' and 'simplemem' in str(path)) else [])
    def ports():
        events.append('ports')
        if failure == 'occupied':
            raise RuntimeError('occupied')
    monkeypatch.setattr(target, 'require_free_ports', ports)
    monkeypatch.setattr(target.queue, 'wait_vllm', lambda plan: events.append('fp16'))
    monkeypatch.setattr(target.queue, 'wait_health', lambda *args: None)
    monkeypatch.setattr(target.queue, 'expected_questions', lambda *args: {'original1540': True})
    @contextmanager
    def service(*args):
        events.append('encoder')
        yield object()
        events.append('encoder_closed')
    monkeypatch.setattr(target.queue, 'service', service)
    def smoke(plan, method, turns):
        assert method == 'a_mem' and turns == 80
        assert all(value >= 2 for value in counts.values())
        events.append('smoke')
        if failure == 'smoke':
            raise ValueError('smoke failed')
        return {'passed': True, 'qa_count': 3, 'memory_turns': 80,
                'response_delivery': {'complete': True}}
    monkeypatch.setattr(target, 'run_probe', smoke)
    def run(plan, item, expected, context_indices):
        assert context_indices == [0, 4, 6]
        assert old_file.read_bytes() == original
        assert 'smoke' in events and full.is_dir()
        events.append('run')
        if failure == 'run':
            raise RuntimeError('selected run failed')
        return {'complete': False, 'requires_composite': True, 'context_indices': context_indices}
    monkeypatch.setattr(target.queue, 'run_method', run)
    if failure not in (None, 'all_success'):
        with pytest.raises((ValueError, RuntimeError, FileExistsError)):
            target.main()
        if failure != 'run':
            assert 'run' not in events
    else:
        assert target.main() == 0
        result = json.loads((probe / 'recovery_status.json').read_text())
        assert result['complete'] is False
        assert result['state'] == ('no_failed_contexts' if failure == 'all_success' else 'awaiting_composite_audit')
        assert ('run' in events) is (failure != 'all_success')
    assert old_file.read_bytes() == original
