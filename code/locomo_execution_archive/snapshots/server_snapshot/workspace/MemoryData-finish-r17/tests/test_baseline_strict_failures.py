"""Real vendored methods must not silently degrade a strict comparison."""

import __future__
import ast
import concurrent.futures
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SIMPLE = "methods/simplemem/source/SimpleMem/"


def load_method(relative, name):
    source = Path(__file__).resolve().parents[1] / relative
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(item for cls in tree.body if isinstance(cls, ast.ClassDef)
                for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {
        "os": os, "config": SimpleNamespace(USE_JSON_FORMAT=True),
        "logger": logging.getLogger(__name__), "concurrent": concurrent,
        "AGGREGATOR_PROMPT": "Query: {query}; evidence: {results}",
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


@pytest.mark.parametrize("method, args", [
    ("_analyze_query", ("question",)),
    ("_generate_search_queries", ("question",)),
    ("_check_answer_adequacy", ("question", ["memory"])),
    ("_generate_additional_queries", ("question", ["memory"])),
    ("_analyze_information_requirements", ("question",)),
    ("_generate_targeted_queries", ("question", {})),
    ("_analyze_information_completeness", ("question", ["memory"], {})),
    ("_generate_missing_info_queries", ("question", ["memory"], {})),
])
def test_simplemem_planning_failures_propagate_after_native_retries(monkeypatch, method, args):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1")
    instance = Mock()
    instance.llm_client.chat_completion.side_effect = ValueError("bad model JSON")
    function = load_method(SIMPLE + "core/hybrid_retriever.py", method)
    with pytest.raises(ValueError, match="bad model JSON"):
        function(instance, *args)
    assert instance.llm_client.chat_completion.call_count == (3 if method == "_analyze_query" else 1)


@pytest.mark.parametrize("method, args", [
    ("_init_fts_index", ()), ("semantic_search", ("question",)),
    ("keyword_search", (["keyword"],)), ("structured_search", ()),
])
def test_simplemem_database_failures_never_become_empty_results(monkeypatch, method, args):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1")
    instance = Mock(_fts_initialized=False, _is_cloud_storage=False, _fts_enabled=True)
    instance.table.count_rows.side_effect = OSError("database error")
    instance.table.create_fts_index.side_effect = OSError("database error")
    with pytest.raises(OSError, match="database error"):
        load_method(SIMPLE + "database/vector_store.py", method)(instance, *args)


def test_simplemem_missing_fts_is_not_a_successful_lexical_search(monkeypatch):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1")
    instance = Mock(_fts_enabled=False)
    instance.table.count_rows.return_value = 1
    with pytest.raises(RuntimeError, match="FTS index"):
        load_method(SIMPLE + "database/vector_store.py", "keyword_search")(instance, ["word"])


@pytest.mark.parametrize("method, args", [
    ("_generate_memory_entries", ([SimpleNamespace(dialogue_id=1)],)),
    ("_generate_memory_entries_worker", ([SimpleNamespace(dialogue_id=1)], [1], 1)),
])
def test_simplemem_exhausted_extraction_retries_fail(monkeypatch, method, args):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1")
    instance = Mock(previous_entries=[], context_window=32768, output_reserve=4096, window_size=20)
    instance._count_prompt_tokens.return_value = 10
    instance.llm_client.chat_completion.side_effect = ValueError("invalid entries")
    with pytest.raises((ValueError, RuntimeError)):
        load_method(SIMPLE + "core/memory_builder.py", method)(instance, *args)
    assert instance.llm_client.chat_completion.call_count == 3


@pytest.mark.parametrize("method, args, worker", [
    ("_execute_parallel_searches", (["question"],), "_semantic_search_worker"),
    ("_execute_parallel_additional_searches", (["question"], 1), "_additional_search_worker"),
])
def test_simplemem_failed_parallel_search_is_not_discarded(monkeypatch, method, args, worker):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1")
    instance = Mock(max_retrieval_workers=2)
    getattr(instance, worker).side_effect = OSError("search failed")
    with pytest.raises(OSError, match="search failed"):
        load_method(SIMPLE + "core/hybrid_retriever.py", method)(instance, *args)


@pytest.mark.parametrize("strict", [False, True])
def test_emem_routing_failure_propagates_only_in_comparison(monkeypatch, strict):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1" if strict else "0")
    instance = Mock(max_blocks=1, agent=["first"], enable_router=True, use_llm_fallback=False)
    instance._score_summary_similarity.side_effect = OSError("scoring failed")
    function = load_method("methods/e-mem/src/memory/router/hybrid_router.py", "_map_blocks")
    if strict:
        with pytest.raises(OSError, match="scoring failed"):
            function(instance, "question")
    else:
        assert function(instance, "question") == ["first"]


@pytest.mark.parametrize("strict", [False, True])
def test_emem_aggregation_failure_is_not_reported_as_success(monkeypatch, strict):
    monkeypatch.setenv("BASELINE_STRICT_COMPARISON", "1" if strict else "0")
    instance = Mock()
    instance.aggregator_llm.chat.completions.create.side_effect = OSError("aggregation failed")
    function = load_method("methods/e-mem/src/conversation_manager/base_chat_manager.py", "_aggregate_memory_results")
    if strict:
        with pytest.raises(OSError, match="aggregation failed"):
            function(instance, "question", "raw evidence")
    else:
        assert function(instance, "question", "raw evidence") == "raw evidence"
