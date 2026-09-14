"""Original A-MEM failure semantics stay opt-in and every response stays metered."""
import __future__

import ast
import copy
import hashlib
import importlib
import json
import logging
import os
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from scripts.metered_openai_proxy import MeteredProxy
from scripts.repairing_openai_proxy import RepairHandler
from utils import request_metering

ROOT = Path(__file__).resolve().parents[1]
LAYER = ROOT / 'methods/a_mem/source/a_mem/memory_layer.py'
INDICES = [1, 0, 2]
REMOVED_LINES = (
    '                                Include each distinct tag only once. Do not repeat tags or contexts.\n',
    '                                Emit each action at most once and only reference input neighbor indices.\n',
)


def native_method(class_name, method_name):
    tree = ast.parse(LAYER.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    node.decorator_list = []
    namespace = {'json': json, 'os': os, 're': __import__('re'), 'logger': logging.getLogger(__name__),
                 '_should_disable_qwen3_thinking': lambda _model: True}
    helper = importlib.util.spec_from_file_location('test_native_repair', LAYER.with_name('evolution_repair.py'))
    module = importlib.util.module_from_spec(helper)
    helper.loader.exec_module(module)
    namespace['evolution_completion'] = module.evolution_completion
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(LAYER), 'exec',  # noqa: S102 - trusted native AST test target
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[method_name]


def native_template():
    tree = ast.parse(LAYER.read_text())
    return next(ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == 'evolution_system_prompt' for t in node.targets))


def completion(content, reason='stop', number=1):
    return SimpleNamespace(id=f'native-response-{number}', model='Qwen/Qwen3.5-9B', choices=[
        SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content=content))])


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv('AMEM_ORIGINAL_FAILURE_POLICY', '1')
    monkeypatch.setenv('AMEM_SEMANTIC_JOURNAL_DIR', str(tmp_path / 'semantic'))
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', '1')
    monkeypatch.setenv('METER_RUN_ID', 'original-test')
    monkeypatch.setenv('METER_METHOD', 'a_mem')
    request_metering.install_request_metering({'METER_RUN_ID': 'original-test', 'METER_METHOD': 'a_mem',
        'METER_LLM_PROXY_ORIGIN': 'http://127.0.0.1:1', 'METER_EMBEDDING_PROXY_ORIGIN': 'http://127.0.0.1:2'})
    sdk = Mock()
    controller = SimpleNamespace(model='Qwen/Qwen3.5-9B', client=sdk)
    controller.get_completion = MethodType(native_method('OpenAIController', 'get_completion'), controller)
    memories = {str(i): SimpleNamespace(context=f'old context {i}', tags=['old', 'old']) for i in range(3)}
    memory = SimpleNamespace(memories=memories, evolution_system_prompt=native_template(),
        find_related_memories=Mock(return_value=('\n'.join(f'memory index:{i}\t all original facts' for i in INDICES), INDICES)),
        llm_controller=SimpleNamespace(llm=controller))
    note = SimpleNamespace(content='all original content', context='context', keywords=['key'], tags=['old', 'old'], links=[])
    yield memory, note, sdk, controller, tmp_path
    request_metering._reset_for_tests()


def events(tmp_path):
    return [json.loads(line) for path in (tmp_path / 'semantic').glob('*.jsonl')
            for line in path.read_text().splitlines()]


def process(state, content, reason='stop'):
    memory, note, sdk, _, _ = state
    sdk.chat.completions.create.return_value = completion(content, reason)
    with request_metering.meter_operation('memory_add', sample_id='conv-26'):
        return native_method('AgenticMemorySystem', 'process_memory')(memory, note)


@pytest.mark.parametrize('reason', ['stop', 'length'])
def test_original_json_error_skips_only_evolution_without_retry_and_journals(state, reason):
    memory, note, sdk, _, tmp_path = state
    before = copy.deepcopy((memory.memories, note))
    assert process(state, '{"unfinished":', reason) == (False, note)
    assert sdk.chat.completions.create.call_count == 1 and (memory.memories, note) == before
    row, = events(tmp_path)
    assert row['stage'] == 'evolution' and row['outcome'] == 'evolution_json_skip'
    assert row['response_id'] == 'native-response-1' and row['finish_reason'] == reason
    assert row['policy'] == 'amem_original_failure_v1' and row['operation_id']
    assert row['run_id'] == 'original-test' and row['sample_id'] == 'conv-26'
    assert row['content_sha256'] == hashlib.sha256(b'{"unfinished":').hexdigest()
    assert 'usage' not in row and row['applied_neighbor_count'] == 0


def evolution(**updates):
    data = {'should_evolve': True, 'actions': ['strengthen', 'update_neighbor'],
            'suggested_connections': [999, 999], 'tags_to_update': ['same', 'same'],
            'new_context_neighborhood': ['new context'],
            'new_tags_neighborhood': [['new', 'new'], ['second', 'second']]}
    return {**data, **updates}


def test_original_partial_neighbor_loop_and_context_fallback_preserve_duplicates(state):
    memory, note, sdk, _, tmp_path = state
    assert process(state, json.dumps(evolution())) == (True, note)
    assert note.links == [999, 999] and note.tags == ['same', 'same']
    assert memory.memories['1'].context == 'new context'
    assert memory.memories['0'].context == 'old context 0'
    assert memory.memories['0'].tags == ['second', 'second']
    assert memory.memories['2'].tags == ['old', 'old']
    row, = events(tmp_path)
    assert row['outcome'] == 'evolution_applied' and row['neighbor_count'] == 3
    assert row['applied_neighbor_count'] == 2 and row['context_fallback_count'] == 1
    assert sdk.chat.completions.create.call_count == 1


@pytest.mark.parametrize('data', [{'should_evolve': False}, {'should_evolve': True, 'actions': []}])
def test_original_unused_fields_are_not_required_or_repaired(state, data):
    memory, note, sdk, _, tmp_path = state
    before = copy.deepcopy((memory.memories, note))
    assert process(state, json.dumps(data)) == (data['should_evolve'], note)
    assert (memory.memories, note) == before and sdk.chat.completions.create.call_count == 1
    assert events(tmp_path)[0]['outcome'] == 'evolution_noop'


def test_original_ignored_unknown_action_is_truthfully_recorded_as_noop(state):
    memory, note, _, _, tmp_path = state
    before = copy.deepcopy((memory.memories, note))
    assert process(state, json.dumps({'should_evolve': True, 'actions': ['unknown']})) == (True, note)
    assert (memory.memories, note) == before
    assert events(tmp_path)[0]['outcome'] == 'evolution_noop'


def test_original_duplicate_actions_are_retained_and_actual_updates_counted(state):
    payload = evolution(actions=['update_neighbor', 'update_neighbor'])
    process(state, json.dumps(payload))
    row, = events(state[-1])
    assert row['applied_neighbor_count'] == 4 and row['context_fallback_count'] == 2


@pytest.mark.parametrize('broken', ['missing_line', 'duplicate_line', 'wrong_constraint', 'wrong_schema'])
def test_exact_original_transform_rejects_changed_capture_before_call(broken):
    from scripts.amem_original_policy import original_completion
    from scripts.diagnose_amem_native_length_repair import _format
    prompt = ''.join(REMOVED_LINES)
    schema = _format(3)
    if broken == 'missing_line':
        prompt = REMOVED_LINES[0]
    elif broken == 'duplicate_line':
        prompt += REMOVED_LINES[0]
    elif broken == 'wrong_constraint':
        schema['json_schema']['schema']['properties']['actions']['maxItems'] = 9
    else:
        schema['json_schema']['name'] = 'invented'
    callback = Mock()
    with pytest.raises(ValueError):
        original_completion(callback, INDICES, prompt, response_format=schema)
    callback.assert_not_called()


@pytest.mark.parametrize('field,value', [('id', ''), ('model', 'wrong-model'), ('choices', [])])
def test_unidentifiable_response_cannot_create_semantic_success(state, field, value):
    from scripts.amem_original_policy import capture_response
    response = completion('{}')
    setattr(response, field, value)
    with request_metering.meter_operation('memory_add', sample_id='conv-26'), pytest.raises(ValueError):
        capture_response(state[3], response)
    assert not events(state[-1])


def test_missing_operation_or_response_cannot_create_or_duplicate_event(state):
    from scripts.amem_original_policy import capture_response, record_outcome
    with pytest.raises(RuntimeError, match='attributed'):
        capture_response(state[3], completion('{}'))
    with pytest.raises(RuntimeError, match='unconsumed'):
        record_outcome(state[3], 'metadata', 'metadata_accepted')
    process(state, json.dumps({'should_evolve': False}))
    with pytest.raises(RuntimeError, match='unconsumed'):
        record_outcome(state[3], 'evolution', 'evolution_noop')
    assert len(events(state[-1])) == 1


@pytest.mark.parametrize('strict', ['0', '1'])
def test_metadata_journal_write_failure_is_not_swallowed_by_native_fallback(state, monkeypatch, strict):
    from scripts import amem_original_policy
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', strict)
    monkeypatch.setattr(amem_original_policy, 'record_outcome', Mock(side_effect=OSError('journal failed')))
    state[2].chat.completions.create.return_value = completion('invalid JSON')
    with request_metering.meter_operation('memory_add', sample_id='conv-26'), pytest.raises(OSError):
        native_method('MemoryNote', 'analyze_content')('all original facts', state[0].llm_controller)
    assert not events(state[-1])


@pytest.mark.parametrize('content,error', [('{}', KeyError), ('null', TypeError), ('[]', TypeError), (None, ValueError)])
def test_original_does_not_turn_non_jsondecode_errors_into_noop(state, content, error):
    with pytest.raises(error):
        process(state, content)
    assert not events(state[-1])


def test_original_transport_failure_propagates_without_semantic_fallback(state):
    memory, note, sdk, _, tmp_path = state
    sdk.chat.completions.create.side_effect = RuntimeError('transport failure')
    with request_metering.meter_operation('memory_add', sample_id='conv-26'), pytest.raises(RuntimeError):
        native_method('AgenticMemorySystem', 'process_memory')(memory, note)
    assert sdk.chat.completions.create.call_count == 1 and not events(tmp_path)


@pytest.mark.parametrize('content,outcome', [('not json', 'metadata_json_fallback'),
    ('{"keywords":["same","same"],"context":"original","tags":["x","x"]}', 'metadata_accepted')])
def test_original_metadata_cap_and_json_policy(state, content, outcome):
    memory, _, sdk, _, tmp_path = state
    sdk.chat.completions.create.return_value = completion(content, 'length')
    with request_metering.meter_operation('memory_add', sample_id='conv-26'):
        result = native_method('MemoryNote', 'analyze_content')('all original facts', memory.llm_controller)
    assert sdk.chat.completions.create.call_count == 1
    assert sdk.chat.completions.create.call_args.kwargs['max_tokens'] == 1000
    assert result == ({'keywords': [], 'context': 'General', 'tags': []} if outcome.endswith('fallback') else json.loads(content))
    row, = events(tmp_path)
    assert row['stage'] == 'metadata' and row['outcome'] == outcome


def test_optin_effective_prompt_and_schema_exactly_upstream_and_default_untouched(state, monkeypatch):
    _, _, sdk, _, _ = state
    process(state, json.dumps({'should_evolve': False}))
    sent = sdk.chat.completions.create.call_args.kwargs
    expected = native_template()
    for line in REMOVED_LINES:
        assert expected.count(line) == 1
        expected = expected.replace(line, '')
    assert hashlib.sha256(expected.encode()).hexdigest() == 'f49ef5bfd021f876325315009363e124dd83b6870dc289097422c9e70a0b5a60'
    assert not any(line in sent['messages'][1]['content'] for line in REMOVED_LINES)
    assert 'all original content' in sent['messages'][1]['content']
    assert hashlib.sha256(json.dumps(sent['response_format'], sort_keys=True, separators=(',', ':')).encode()).hexdigest() == '5223745074b0dab699e769148f74449565ecebf332167c905605b7bfda861520'
    assert sent['max_tokens'] == 1000
    monkeypatch.delenv('AMEM_ORIGINAL_FAILURE_POLICY')
    sdk.reset_mock()
    with pytest.raises(ValueError):
        process(state, '{"unfinished":')
    assert sdk.chat.completions.create.call_count == 3
    default = sdk.chat.completions.create.call_args_list[0].kwargs
    assert default['max_tokens'] == 640 and all(line in default['messages'][1]['content'] for line in REMOVED_LINES)
    assert default['response_format']['json_schema']['schema']['properties']['new_tags_neighborhood']['maxItems'] == 3


def test_journal_failure_and_missing_context_cannot_silently_pass(state, monkeypatch):
    _, _, _, _, tmp_path = state
    monkeypatch.setenv('AMEM_SEMANTIC_JOURNAL_DIR', str(tmp_path / 'blocked'))
    (tmp_path / 'blocked').write_text('not a directory')
    with pytest.raises(OSError):
        process(state, json.dumps({'should_evolve': False}))


def proxy_request(stage):
    from scripts.diagnose_amem_native_length_repair import _format
    schema = _format(3)
    props = schema['json_schema']['schema']['properties']
    if stage == 'evolution':
        props['actions'].pop('maxItems')
        props['actions']['items'].pop('enum')
        props['suggested_connections'].pop('maxItems')
        for name in ('new_context_neighborhood', 'new_tags_neighborhood'):
            props[name].pop('minItems')
            props[name].pop('maxItems')
        prompt = 'You are an AI memory evolution agent responsible for managing and evolving a knowledge base. Original facts.'
    else:
        schema['json_schema']['schema']['properties'] = {'keywords': {'type': 'array', 'items': {'type': 'string'}},
            'context': {'type': 'string'}, 'tags': {'type': 'array', 'items': {'type': 'string'}}}
        schema['json_schema']['schema']['required'] = ['keywords', 'context', 'tags']
        prompt = 'Generate a structured analysis of the following content by: Original facts.'
    return {'model': 'Qwen/Qwen3.5-9B', 'max_tokens': 1000, 'temperature': 0.7, 'response_format': schema,
            'messages': [{'role': 'system', 'content': 'You must respond with a JSON object.'},
                         {'role': 'user', 'content': prompt}]}


@pytest.mark.parametrize('stage', ['metadata', 'evolution'])
@pytest.mark.parametrize('reason', ['stop', 'length'])
@pytest.mark.parametrize('scope', ['native', 'unset', 'qa', 'other_method'])
def test_real_loopback_proxy_native_single_call_and_default_strict_unchanged(tmp_path, monkeypatch, stage, reason, scope):
    payloads = tmp_path / 'failed_payloads'
    monkeypatch.setenv('METER_FAILED_PAYLOAD_DIR', str(payloads))
    if scope == 'unset':
        monkeypatch.delenv('AMEM_ORIGINAL_FAILURE_POLICY', raising=False)
    else:
        monkeypatch.setenv('AMEM_ORIGINAL_FAILURE_POLICY', '1')
    method = 'simplemem' if scope == 'other_method' else 'a_mem'
    phase = 'qa' if scope == 'qa' else 'memory_add'
    body = {'id': 'native-proxy-response', 'model': 'Qwen/Qwen3.5-9B', 'choices': [
        {'finish_reason': reason, 'message': {'content': '{"unfinished":'}}],
        'usage': {'prompt_tokens': 40, 'completion_tokens': 1000, 'total_tokens': 1040}}
    if scope == 'native':
        body['choices'][0]['message']['content'] += '"원문\u2028preserved'
    calls = []

    def upstream(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    journal = tmp_path / 'usage.jsonl'
    with MeteredProxy(('127.0.0.1', 0), 'http://unused.invalid/v1', journal, 'native-test',
                      method, comparison_policy=True) as server:
        server.client.close()
        server.client = httpx.Client(transport=httpx.MockTransport(upstream))
        server.RequestHandlerClass = RepairHandler
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        thread.start()
        try:
            with httpx.Client(trust_env=False) as client:
                response = client.post(f'http://127.0.0.1:{server.server_port}/meter/native-test/{method}/{phase}/conv-26/-/v1/chat/completions', json=proxy_request(stage))
                assert response.status_code == (200 if scope == 'native' or reason == 'stop' else 502)
                if response.status_code == 200:
                    assert response.json() == body
        finally:
            server.shutdown()
            thread.join(timeout=5)
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(calls) == (1 if scope == 'native' or reason == 'stop' else 3)
    assert sum(row['usage']['total_tokens'] for row in rows) == len(calls) * 1040
    assert calls[0]['temperature'] == 0
    saved = list(payloads.glob('*.json'))
    assert len(saved) == (len(rows) if scope == 'native' or reason == 'length' else 0)
    for number, row in enumerate(rows if saved else []):
        payload = json.loads((payloads / f"{row['request_id']}.json").read_text(encoding='utf-8'))
        assert payload == {'request_id': row['request_id'], 'request': calls[number], 'response': body}
    if scope != 'native':
        assert 'max_tokens' not in calls[0]
        assert all('native_failure_policy' not in row for row in rows)
        assert rows[-1]['finish_reasons'] == [reason]
        assert rows[-1]['delivered_to_client'] is (reason == 'stop')
        return
    assert calls[0]['max_tokens'] == 1000
    assert calls[0]['messages'] == proxy_request(stage)['messages']
    assert calls[0]['response_format'] == proxy_request(stage)['response_format']
    row, = rows
    assert row['native_failure_policy'] == 'amem_original_failure_v1' and row['native_stage'] == stage
    assert row['delivered_to_client'] is True and row['finish_reasons'] == [reason]
    assert row['comparison_policy']['effective']['max_tokens'] == 1000
    assert row['native_content_sha256'] == hashlib.sha256(body['choices'][0]['message']['content'].encode()).hexdigest()
    assert row['usage'] == body['usage'] and row['repair_attempt'] == 0 and not row.get('retry_scheduled')
    assert row['success'] is (reason == 'stop')
    assert row['error_type'] == ('native_length_output' if reason == 'length' else None)


def test_native_payload_write_failure_retains_usage_and_does_not_deliver(tmp_path, monkeypatch):
    monkeypatch.setenv('AMEM_ORIGINAL_FAILURE_POLICY', '1')
    blocked = tmp_path / 'blocked_payload_directory'
    blocked.write_text('not a directory')
    monkeypatch.setenv('METER_FAILED_PAYLOAD_DIR', str(blocked))
    body = {'id': 'native-write-failure', 'model': 'Qwen/Qwen3.5-9B', 'choices': [
        {'finish_reason': 'length', 'message': {'content': '{"unfinished":'}}],
        'usage': {'prompt_tokens': 40, 'completion_tokens': 1000, 'total_tokens': 1040}}
    journal, calls = tmp_path / 'usage.jsonl', []

    def upstream(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    with MeteredProxy(('127.0.0.1', 0), 'http://unused.invalid/v1', journal, 'native-test',
                      'a_mem', comparison_policy=True) as server:
        server.client.close()
        server.client = httpx.Client(transport=httpx.MockTransport(upstream))
        server.RequestHandlerClass = RepairHandler
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        thread.start()
        try:
            with httpx.Client(trust_env=False) as client:
                response = client.post(f'http://127.0.0.1:{server.server_port}/meter/native-test/a_mem/memory_add/conv-26/-/v1/chat/completions',
                                       json=proxy_request('evolution'))
                assert response.status_code == 502
        finally:
            server.shutdown()
            thread.join(timeout=5)
    row, = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(calls) == 1 and row['usage'] == body['usage']
    assert row['finish_reasons'] == ['length'] and row['success'] is False
    assert row['delivered_to_client'] is False and row['error_type'] == 'repair_proxy_error'
    assert not row.get('retry_scheduled') and blocked.read_text() == 'not a directory'


@pytest.mark.parametrize('disabled,method,phase', [(True, 'a_mem', 'memory_add'),
    (False, 'a_mem', 'qa'), (False, 'simplemem', 'memory_add')])
def test_native_optin_recognizer_does_not_change_other_requests(monkeypatch, disabled, method, phase):
    proxy = importlib.import_module('scripts.repairing_openai_proxy')
    monkeypatch.setenv('AMEM_ORIGINAL_FAILURE_POLICY', '0' if disabled else '1')
    assert proxy.amem_original_stage(proxy_request('evolution'), {'method': method, 'phase': phase}) is None
