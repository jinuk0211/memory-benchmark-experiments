"""Retry only malformed apostrophe FTS queries; keep valid native search unchanged."""
import importlib
import sys
import types
from pathlib import Path
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


@pytest.mark.parametrize('raw,expected', [
    ('CS:GO tournaments', r'CS\:GO tournaments'),
    ('CS:GO OR football', r'CS\:GO OR football'),
    ('"CS:GO" OR CS:GO', r'"CS:GO" OR CS\:GO'),
])
def test_actual_tantivy_unknown_field_retries_literal_colon(store, raw, expected):
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
    assert store.table.search.call_args_list == [call(raw), call(expected)]


def test_mixed_fts_syntax_repairs_are_finite_and_keep_valid_field_scope(store):
    schema = tantivy.SchemaBuilder()
    schema.add_text_field('text', stored=True)
    index = tantivy.Index(schema.build())
    seen = []

    def search(query):
        seen.append(query)
        parsed = index.parse_query(query, ['text'])
        chain = Mock()
        chain.limit.return_value.to_list.return_value = [str(parsed)]
        return chain

    store.table.search.side_effect = search
    actual = store.keyword_search(["text:football OR CS:GO Dave's AC:DC"])
    expected = index.parse_query(r"text:football OR CS\:GO Dave\'s AC\:DC", ['text'])
    assert actual == [str(expected)]
    assert len(seen) == len(set(seen)) == 4


def test_unknown_field_error_without_repairable_field_fails_closed(store):
    store.table.search.side_effect = ValueError("Field does not exist: 'CS'")
    with pytest.raises(ValueError, match='Field does not exist'):
        store.keyword_search(['ordinary words'])
    store.table.search.assert_called_once()


@pytest.mark.parametrize('field,value', [
    ('entities', "Harry Potter and the Philosopher's Stone"),
    ('entities', "McGee's bar"),
    ('persons', "O'Brien"),
    ('persons', "x') OR true --"),
    ('entities', 'ordinary unchanged'),
])
def test_actual_lancedb_structured_literal_is_preserved(store, tmp_path, field, value):
    import lancedb
    rows = [{'entry_id': 'wanted', 'persons': [value] if field == 'persons' else ['Tim'],
             'entities': [value] if field == 'entities' else ['book'],
             'location': 'room', 'timestamp': '2023-01-01'},
            {'entry_id': 'other', 'persons': ['Other'], 'entities': ['other'],
             'location': 'room', 'timestamp': '2023-01-01'}]
    store.table = lancedb.connect(str(tmp_path / 'db')).create_table('memories', data=rows)
    terms = [value]
    assert [row['entry_id'] for row in store.structured_search(**{field: terms})] == ['wanted']
    assert terms == [value]
