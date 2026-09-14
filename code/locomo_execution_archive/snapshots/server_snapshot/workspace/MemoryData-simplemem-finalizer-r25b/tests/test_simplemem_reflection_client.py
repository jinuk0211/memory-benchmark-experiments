"""Native extraction failures retry without losing the original reflection evidence."""
import importlib
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest


@pytest.fixture
def client(monkeypatch):
    package = '_simplemem_reflection_client_test'
    source = Path(__file__).resolve().parents[1] / 'methods/simplemem/source/SimpleMem'
    for suffix, attrs in {'': {'__path__': [str(source)]},
                          '.utils': {'__path__': [str(source / 'utils')]}, '.config': {}}.items():
        module = types.ModuleType(package + suffix)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, package + suffix, module)
    target = importlib.import_module(package + '.utils.llm_client')
    obj = target.LLMClient.__new__(target.LLMClient)
    obj.model, obj.base_url, obj.use_streaming = 'Qwen/Qwen3.5-9B', 'http://meter/v1', False
    obj.client = Mock()
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', '1')
    monkeypatch.setattr('time.sleep', lambda _: None)
    yield obj
    for name in list(sys.modules):
        if name.startswith(package + '.'):
            sys.modules.pop(name, None)


def messages():
    return [
        {'role': 'system', 'content': 'You are an information completeness evaluator. You must output valid JSON format.'},
        {'role': 'user', 'content': 'Analyze whether the provided information is sufficient to completely answer the original question, based on the identified information requirements.\nQuestion and all retrieved original facts.'},
    ]


def response(content, response_id):
    return types.SimpleNamespace(id=response_id, choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content=content))])


@pytest.mark.parametrize('bad', ['{"assessment":"incomplete","reasoning":"unfinished', '[]'])
def test_native_rejected_reflection_retries_and_records_response_identity(client, capsys, bad):
    original = messages()
    good = '{"assessment":"complete","reasoning":"all evidence present"}'
    client.client.chat.completions.create.side_effect = [response(bad, 'bad-response'), response(good, 'good-response')]
    assert client.chat_completion(original, temperature=0.1) == good
    calls = client.client.chat.completions.create.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs['messages'] == original
    assert calls[1].kwargs['messages'][:-1] == original
    assert original == messages()
    assert calls[0].kwargs['temperature'] == calls[1].kwargs['temperature'] == 0.1
    event = next(line for line in capsys.readouterr().out.splitlines() if line.startswith('SIMPLEMEM_JSON_REPAIR '))
    diagnostic = json.loads(event.removeprefix('SIMPLEMEM_JSON_REPAIR '))
    assert diagnostic['response_id'] == 'bad-response' and diagnostic['retry_scheduled'] is True


@pytest.mark.parametrize('good', ['Result: {"assessment":"complete"}', '{"assessment":"complete",}',
                                'Text ```json\n{"assessment":"incomplete"}\n```'])
def test_native_accepted_objects_are_not_changed(client, good):
    client.client.chat.completions.create.return_value = response(good, 'native-response')
    assert client.chat_completion(messages()) == good
    client.client.chat.completions.create.assert_called_once()


def test_persistent_native_json_failure_is_not_fake_success(client, capsys):
    client.client.chat.completions.create.return_value = response('{"assessment":', 'bad-response')
    with pytest.raises(ValueError, match='JSON'):
        client.chat_completion(messages())
    assert client.client.chat.completions.create.call_count == 3
    events = [json.loads(line.removeprefix('SIMPLEMEM_JSON_REPAIR ')) for line in capsys.readouterr().out.splitlines()
              if line.startswith('SIMPLEMEM_JSON_REPAIR ')]
    assert [event['retry_scheduled'] for event in events] == [True, True, False]


def test_other_tasks_and_non_strict_behavior_unchanged(client, monkeypatch):
    client.client.chat.completions.create.return_value = response('[]', 'native-response')
    assert client.chat_completion([{'role': 'user', 'content': 'Return a memory list'}]) == '[]'
    monkeypatch.delenv('BASELINE_STRICT_COMPARISON')
    assert client.chat_completion(messages()) == '[]'
    assert client.client.chat.completions.create.call_count == 2
