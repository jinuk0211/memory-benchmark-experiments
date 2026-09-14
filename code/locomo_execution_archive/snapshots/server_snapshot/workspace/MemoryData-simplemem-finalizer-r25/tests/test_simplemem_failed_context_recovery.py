"""Recover only failed SimpleMem shards after both original runs release the ports."""
from contextlib import contextmanager
import importlib
import json

import pytest


@pytest.mark.parametrize('failure', [None, 'smoke', 'ports', 'writer', 'existing', 'unknown', 'run', 'all_success'])
@pytest.mark.parametrize('version', ['r16', 'r17'])
def test_simplemem_recovery_preserves_successes_and_tests_known_failed_window(tmp_path, monkeypatch, failure, version):
    target = importlib.import_module('scripts.simplemem_failed_context_recovery_' + version)
    prior, probe, full, plan_path = [tmp_path / name for name in ['prior', 'probe', 'full', 'plan.json']]
    prior.mkdir()
    status = {'active_contexts': [], 'failed': failure != 'all_success', 'events': [
        *[{'event': 'start', 'context': i, 'pid': i + 100} for i in range(10)],
        *[{'event': 'end', 'context': i, 'exit_code': int(i == 6 and failure != 'all_success')} for i in range(10)]]}
    old_file = prior / 'parallel_status.json'
    old_file.write_text(json.dumps(status))
    original = old_file.read_bytes()
    plan_path.write_text(json.dumps({'output': str(full), 'server_python': 'server',
        'embedding_path': 'minilm', 'dataset': 'original1540.json', 'methods': [{'method': 'simplemem'}]}))
    monkeypatch.setattr(target, 'PLAN_PATH', plan_path)
    monkeypatch.setattr(target, 'PRIOR_OUTPUT', prior)
    monkeypatch.setattr(target, 'PROBE_OUTPUT', probe)
    if failure == 'existing':
        full.mkdir()
    events, counts = [], {}
    monkeypatch.setattr(target.queue, 'validate_plan', lambda _: events.append('pins'))
    def state(name):
        counts[name] = counts.get(name, 0) + 1
        return 'UNKNOWN' if failure == 'unknown' else ('RUNNING' if counts[name] == 1 else 'EXITED')
    monkeypatch.setattr(target.queue, 'supervisor_state', state)
    monkeypatch.setattr(target.time, 'sleep', lambda _: events.append('wait'))
    monkeypatch.setattr(target, 'running_writers', lambda _: [123] if failure == 'writer' else [])
    def ports():
        if failure == 'ports':
            raise RuntimeError('ports busy')
    monkeypatch.setattr(target, 'require_free_ports', ports)
    monkeypatch.setattr(target.queue, 'wait_vllm', lambda _: None)
    monkeypatch.setattr(target.queue, 'wait_health', lambda *args: None)
    monkeypatch.setattr(target.queue, 'expected_questions', lambda *args: {'original1540': True})
    @contextmanager
    def service(*args):
        yield object()
    monkeypatch.setattr(target.queue, 'service', service)
    def smoke(plan, method, turns, *, sample_index):
        assert plan['run_id'] == 'locomo-simplemem-recovery-probe-20260908-' + version
        assert method == 'simplemem' and turns == 160 and sample_index == 6
        assert set(counts) == {'locomo-simplemem-probe-then-full-r13', 'locomo-amem-failed-context-recovery-r15'}
        assert min(counts.values()) >= 2
        events.append('smoke')
        if failure == 'smoke':
            raise ValueError('smoke failed')
        return {'passed': True}
    monkeypatch.setattr(target, 'run_probe', smoke)
    def run(plan, item, expected, context_indices):
        assert item['method'] == 'simplemem' and context_indices == [6]
        assert old_file.read_bytes() == original and 'smoke' in events
        events.append('run')
        if failure == 'run':
            raise RuntimeError('run failed')
        return {'complete': False, 'requires_composite': True}
    monkeypatch.setattr(target.queue, 'run_method', run)
    if failure not in (None, 'all_success'):
        with pytest.raises((ValueError, RuntimeError, FileExistsError)):
            target.main()
        assert 'run' not in events or failure == 'run'
    else:
        assert target.main() == 0
        state = json.loads((probe / 'recovery_status.json').read_text())
        assert state['complete'] is False
        assert state['state'] == ('no_failed_contexts' if failure == 'all_success' else 'awaiting_composite_audit')
    assert old_file.read_bytes() == original
