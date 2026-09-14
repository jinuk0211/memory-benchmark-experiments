"""Exercise the vendored Mem0 method without optional model dependencies."""

import ast
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def memory_method():
    source = Path(__file__).resolve().parents[1] / "methods/mem0/source/mem0/memory/main.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    memory = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Memory")
    method = next(node for node in memory.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_add_to_vector_store")
    update_prompt = Mock(return_value="update prompt")
    namespace = {
        "json": json, "logging": logging, "os": os,
        "parse_messages": lambda messages: "conversation",
        "get_fact_retrieval_messages": lambda text: ("system", text),
        "get_update_memory_messages": update_prompt,
        "remove_code_blocks": lambda text: text,
        "capture_event": Mock(),
    }
    # Compile the real production method, not a reimplementation of its logic.
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    instance = SimpleNamespace(
        custom_fact_extraction_prompt=None, custom_update_memory_prompt=None,
        api_version="v1.1", llm=Mock(), embedding_model=Mock(), vector_store=Mock(),
        _create_memory=Mock(return_value="new-id"), _update_memory=Mock(), _delete_memory=Mock(),
    )
    instance.vector_store.search.return_value = []
    instance.embedding_model.embed.return_value = [0.1, 0.2]
    run = lambda: namespace["_add_to_vector_store"](
        instance, [{"role": "user", "content": "conversation"}], {}, {"user_id": "test"}, True
    )
    return instance, run, update_prompt


@pytest.mark.parametrize("strict, count", [(True, 6), (False, 2)])
def test_all_facts_reach_embeddings_and_update_prompt_in_strict_mode(memory_method, monkeypatch, strict, count):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1" if strict else "0")
    memory, run, update_prompt = memory_method
    facts = [f"fact {i}" for i in range(6)]
    memory.llm.generate_response.side_effect = [json.dumps({"facts": facts}), '{"memory": []}']
    assert run() == []
    assert [call.args[0] for call in memory.embedding_model.embed.call_args_list] == facts[:count]
    assert update_prompt.call_args.args[1] == facts[:count]


@pytest.mark.parametrize("response", ['not json', '{}', '{"facts": "fact"}', '{"facts": [null]}',
                                      '{"facts": [" "]}'])
def test_invalid_facts_fail_strict_run(memory_method, monkeypatch, response):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1")
    memory, run, _ = memory_method
    memory.llm.generate_response.return_value = response
    with pytest.raises((ValueError, KeyError)):
        run()
    memory.embedding_model.embed.assert_not_called()


def test_valid_empty_facts_skip_action_call(memory_method, monkeypatch):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1")
    memory, run, _ = memory_method
    memory.llm.generate_response.return_value = '{"facts": []}'
    assert run() == []
    assert memory.llm.generate_response.call_count == 1


@pytest.mark.parametrize("response", [RuntimeError("request failed"), 'not json', '{}', 'null',
                                      '{"memory": {}}', '{"memory": [{"event": "UNKNOWN"}]}',
                                      '{"memory": [{"event": "ADD"}]}',
                                      '{"memory": [{"event": "ADD", "text": "   "}]}',
                                      '{"memory": [{"event": "UPDATE", "text": 42}]}'])
def test_action_failures_are_not_silently_dropped(memory_method, monkeypatch, response):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1")
    memory, run, _ = memory_method
    memory.llm.generate_response.side_effect = ['{"facts": ["fact"]}', response]
    with pytest.raises((RuntimeError, ValueError)):
        run()


def test_storage_failures_propagate(memory_method, monkeypatch):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1")
    memory, run, _ = memory_method
    memory.llm.generate_response.side_effect = [
        '{"facts": ["fact"]}', '{"memory": [{"event": "ADD", "text": "fact"}]}',
    ]
    memory._create_memory.side_effect = OSError("database unavailable")
    with pytest.raises(OSError, match="database unavailable"):
        run()


@pytest.mark.parametrize("actions", [
    '{"memory": [{"event": "NONE"}]}',
    '[{"event": "NONE"}]',
    '{"memories": [{"event": "NONE"}]}',
])
def test_native_noop_and_supported_response_shapes(memory_method, monkeypatch, actions):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1")
    memory, run, _ = memory_method
    memory.llm.generate_response.side_effect = ['{"facts": ["fact"]}', actions]
    assert run() == []
    memory._create_memory.assert_not_called()
