import errno
import importlib.util
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import dual_metered_proxy as proxy
from test_proxy_existing import running_proxy


def routes(tmp_path):
    value = {'protocol_sha256': 'protocol', 'lanes': {}, 'histories': {}}
    for lane, instance, port, qid in [('local', '50468468', 18081, 'a'), ('v100', '50558359', 18091, 'b')]:
        value['lanes'][lane] = {'instance_id': instance, 'api_base': f'http://127.0.0.1:{port}/v1'}
        value['histories'][qid] = {'lane': lane, 'instance_id': instance, 'protocol_sha256': 'protocol'}
    path = tmp_path / 'routes.json'
    path.write_text(json.dumps(value))
    return path, value


def test_memory_and_qa_choose_same_history_lane(tmp_path):
    path, _ = routes(tmp_path)
    for question in (None, 'b'):
        url, record = proxy.select_upstream({'method': 'simplemem', 'run_id': 'protocol', 'sample_id': 'b', 'question_id': question}, 'old', path)
        assert url == 'http://127.0.0.1:18091/v1'
        assert record['inference_instance_id'] == '50558359'


@pytest.mark.parametrize('change', [{'question_id': 'other'}, {'run_id': 'wrong'}, {'sample_id': 'unknown'}])
def test_conflicting_or_unassigned_routing_rejected(tmp_path, change):
    path, _ = routes(tmp_path)
    request = {'method': 'simplemem', 'run_id': 'protocol', 'sample_id': 'b'} | change
    with pytest.raises(ValueError):
        proxy.select_upstream(request, 'old', path)


def test_missing_manifest_cannot_move_remote_history_local(tmp_path):
    with pytest.raises(ValueError):
        proxy.select_upstream({'method': 'simplemem'}, 'old', tmp_path/'missing.json')


def test_unapproved_upstream_or_instance_rejected(tmp_path):
    path, value = routes(tmp_path)
    for field, bad in [('api_base', 'https://unapproved.example/v1'), ('instance_id', 'another')]:
        original = value['lanes']['v100'][field]
        value['lanes']['v100'][field] = bad
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            proxy.select_upstream({'method': 'simplemem', 'run_id': 'protocol', 'sample_id': 'b'}, 'old', path)
        value['lanes']['v100'][field] = original


def test_parallel_routes_do_not_mutate_shared_upstream(tmp_path):
    path, _ = routes(tmp_path)
    def select(qid):
        return proxy.select_upstream({'method': 'simplemem', 'run_id': 'protocol', 'sample_id': qid}, 'old', path)[0]
    ids = ['a', 'b'] * 30
    with ThreadPoolExecutor(8) as pool:
        actual = list(pool.map(select, ids))
    assert actual == ['http://127.0.0.1:18081/v1', 'http://127.0.0.1:18091/v1'] * 30
    assert proxy.select_upstream({'method': 'lightmem'}, 'old', path)[0] == 'old'


def test_only_completed_success_updates_idle_guard(tmp_path, monkeypatch):
    activity = tmp_path/'inference_activity.json'
    monkeypatch.setenv('METER_ACTIVITY_PATH', str(activity))
    monkeypatch.setenv('METER_INSTANCE_ID', '50558359')
    with running_proxy(tmp_path) as (_, client, _):
        assert client.post('/v1/chat/completions', json={'model':'test-model','messages':[]}).status_code == 200
    value = json.loads(activity.read_text())
    assert value['last_success'] > 0
    assert value['instance_id'] == '50558359'


def guard_module(monkeypatch):
    if sys.platform == 'win32':
        monkeypatch.setitem(sys.modules, 'fcntl', types.SimpleNamespace(flock=lambda *a:None, LOCK_EX=1, LOCK_NB=2))
    spec=importlib.util.spec_from_file_location('guard_test',Path(__file__).with_name('v100_guard.py'))
    guard=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard


@pytest.mark.parametrize('low_disk', [True, False])
def test_state_write_failure_still_sends_stop(tmp_path, monkeypatch, low_disk):
    guard = guard_module(monkeypatch)
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    monkeypatch.setattr(guard, 'decision', lambda *args: 'low_disk_preserve_results' if low_disk else None)
    def fail(*args):
        raise OSError(errno.ENOSPC, 'full')
    monkeypatch.setattr(guard, 'save', fail)
    calls=[]
    def provider(stop=False):
        calls.append(stop)
        return {'accepted':True} if stop else {'actual_status':'running','intended_status':'running','cur_state':'running'}
    monkeypatch.setattr(guard, 'provider', provider)
    ticks=[]
    class EndTest(Exception):
        pass
    def sleep(_):
        ticks.append(1)
        if len(ticks) >= (1 if low_disk else 2):
            raise EndTest
    monkeypatch.setattr(guard.time, 'sleep', sleep)
    with pytest.raises(EndTest):
        guard.main()
    assert calls[:2] == [False, True]