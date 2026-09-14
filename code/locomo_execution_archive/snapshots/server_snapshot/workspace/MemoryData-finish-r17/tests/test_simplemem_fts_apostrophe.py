"""Retry only malformed apostrophe FTS queries; keep valid native search unchanged."""
import importlib
from pathlib import Path
import sys
import types
from unittest.mock import Mock, call

import pytest
import tantivy


@pytest.fixture
def store(monkeypatch):
    package = '_simplemem_fts_test'
    source = Path(__file__).resolve().parents[1] / 'methods/simplemem/source/SimpleMem'
    for suffix, attrs in {'': {'__path__': [str(source)]},
                          '.utils.embedding': {'EmbeddingModel': object}, '.config': {}}.items():
        module = types.ModuleType(package + suffix)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, package + suffix, module)
    target = importlib.import_module(package + '.database.vector_store')
    obj = target.VectorStore.__new__(target.VectorStore)
    obj.table = Mock()
    obj.table.count_rows.return_value = 2
    obj._fts_enabled = True
    obj._results_to_entries = lambda rows: rows
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', '1')
    yield obj
    for name in list(sys.modules):
        if name.startswith(package + '.'):
            sys.modules.pop(name, None)


def test_exact_live_apostrophe_failure_retries_escaped_query(store, capsys):
    keywords = ["Dave's", 'shop', 'employ', 'people']
    chain = Mock()
    chain.limit.return_value.to_list.return_value = ['record']
    store.table.search.side_effect = [ValueError("Syntax Error: Dave's shop employ people"), chain]
    assert store.keyword_search(keywords, top_k=5) == ['record']
    assert store.table.search.call_args_list == [call("Dave's shop employ people"), call("Dave\\'s shop employ people")]
    chain.limit.assert_called_once_with(5)
    assert keywords == ["Dave's", 'shop', 'employ', 'people']
    assert 'apostrophe' in capsys.readouterr().out


def test_valid_query_with_apostrophe_is_not_rewritten(store):
    store.table.search.return_value.limit.return_value.to_list.return_value = ['native']
    assert store.keyword_search(['"Dave\'s"', 'shop']) == ['native']
    store.table.search.assert_called_once_with('"Dave\'s" shop')


@pytest.mark.parametrize('error,keywords', [
    (RuntimeError('database unavailable'), ["Dave's"]),
    (ValueError('other failure'), ["Dave's"]),
    (ValueError('Syntax Error: bad('), ['bad(']),
    (ValueError("Syntax Error: 'unfinished"), ["'unfinished"]),
])
def test_unrelated_failures_are_not_hidden_or_retried(store, error, keywords):
    store.table.search.side_effect = error
    with pytest.raises(type(error), match=str(error).replace('(', r'\(')):
        store.keyword_search(keywords)
    store.table.search.assert_called_once()


def test_repaired_search_failure_still_raises(store):
    store.table.search.side_effect = [ValueError("Syntax Error: Dave's"), RuntimeError('disk failure')]
    with pytest.raises(RuntimeError, match='disk failure'):
        store.keyword_search(["Dave's"])
    assert store.table.search.call_count == 2


def test_non_strict_behavior_and_missing_index_are_unchanged(store, monkeypatch):
    monkeypatch.delenv('BASELINE_STRICT_COMPARISON')
    store.table.search.side_effect = ValueError("Syntax Error: Dave's")
    assert store.keyword_search(["Dave's"]) == []
    store.table.search.assert_called_once()
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', '1')
    store._fts_enabled = False
    with pytest.raises(RuntimeError, match='requires its FTS index'):
        store.keyword_search(['shop'])
    monkeypatch.delenv('BASELINE_STRICT_COMPARISON')
    assert store.keyword_search(['shop']) == []


@pytest.mark.parametrize('empty_keywords,empty_table', [(True, False), (False, True)])
def test_empty_inputs_do_not_search(store, empty_keywords, empty_table):
    store.table.count_rows.return_value = 0 if empty_table else 2
    assert store.keyword_search([] if empty_keywords else ['shop']) == []
    store.table.search.assert_not_called()


@pytest.mark.parametrize('raw,expected', [
    ("Dave's shop employ people", '"Dave\'s" shop employ people'),
    ("'lake park' Dave's", "'lake park' Dave\\'s"),
    ("Dave\\'s John's", "Dave\\'s John\\'s"),
    ("John veterans' rights support event", "John veterans\\' rights support event"),
    ("John marching event veterans' rights inspired", "John marching event veterans\\' rights inspired"),
    ("'lake park' veterans' rights", "'lake park' veterans\\' rights"),
    ('"Dave\'s shop" veterans\' rights', '"Dave\'s shop" veterans\\\' rights'),
    ("veterans'", "veterans\\'"),
])
def test_actual_tantivy_parser_keeps_keyword_boolean_structure(store, raw, expected):
    schema = tantivy.SchemaBuilder()
    schema.add_text_field('text', stored=True)
    index = tantivy.Index(schema.build())
    parsed = []
    def search(query):
        parsed.append(index.parse_query(query, ['text']))
        chain = Mock()
        chain.limit.return_value.to_list.return_value = ['hit']
        return chain
    store.table.search.side_effect = search
    assert store.keyword_search([raw]) == ['hit']
    assert str(parsed[0]) == str(index.parse_query(expected, ['text']))
