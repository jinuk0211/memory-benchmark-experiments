"""Keep a singleton location value intact when adapting to the native scalar field."""
import copy
import importlib
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest
from pydantic import ValidationError


@pytest.fixture
def native(monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'methods/simplemem/source/SimpleMem'
    package = '_simplemem_location_test'
    for suffix, attrs in {
        '': {'__path__': [str(source)]},
        '.utils.llm_client': {'LLMClient': object},
        '.database.vector_store': {'VectorStore': object},
        '.config': {},
    }.items():
        module = types.ModuleType(package + suffix)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, package + suffix, module)
    provider = types.ModuleType('utils.provider_utils')
    provider.ApproximateTokenizer = object
    provider.load_local_hf_tokenizer = lambda _: None
    monkeypatch.setitem(sys.modules, 'utils.provider_utils', provider)
    target = importlib.import_module(package + '.core.memory_builder')
    yield target
    for name in list(sys.modules):
        if name.startswith(package + '.'):
            sys.modules.pop(name, None)


def parse(native, payload):
    builder = native.MemoryBuilder.__new__(native.MemoryBuilder)
    builder.llm_client = Mock()
    builder.llm_client.extract_json.return_value = payload
    return builder._parse_llm_response(json.dumps(payload), [11, 12])


def test_singleton_location_retains_exact_text_and_all_other_fields(native, monkeypatch, capsys):
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', '1')
    item = {'lossless_restatement': 'Anna visited Italy.', 'location': ['Italy'],
            'keywords': ['Italy', 'Italy'], 'persons': ['Anna'], 'entities': [],
            'timestamp': None, 'topic': 'Travel'}
    original = copy.deepcopy(item)
    entries = parse(native, [item])
    assert entries[0].model_dump(exclude={'entry_id'}) == dict(item, location='Italy')
    assert item == original
    assert 'normalized location array' in capsys.readouterr().out


@pytest.mark.parametrize('location', ['Italy', None, '', ' Italy, France '])
def test_native_scalar_values_are_unchanged(native, monkeypatch, location):
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', '1')
    assert parse(native, [{'lossless_restatement': 'Fact', 'location': location}])[0].location == location


def test_native_single_object_and_invalid_top_level(native, monkeypatch):
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', '1')
    assert parse(native, {'lossless_restatement': 'Fact', 'location': ['Italy']})[0].location == 'Italy'
    with pytest.raises(ValueError, match='Expected JSON array'):
        parse(native, 123)


@pytest.mark.parametrize('location', [[], [7], [None], [['Italy']], {'name': 'Italy'}])
def test_ambiguous_or_invalid_locations_still_fail(native, monkeypatch, location):
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', '1')
    with pytest.raises(ValidationError):
        parse(native, [{'lossless_restatement': 'Fact', 'location': location}])


def test_multiple_location_names_are_all_retained_in_native_description(native, monkeypatch):
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', '1')
    item = {'lossless_restatement': 'Anna visited the Eiffel Tower in Paris.',
            'location': ['Paris', 'Eiffel Tower', 'Paris']}
    before = copy.deepcopy(item)
    entry = parse(native, [item])[0]
    assert entry.location == 'Paris, Eiffel Tower, Paris'
    assert entry.lossless_restatement == item['lossless_restatement']
    assert item == before


@pytest.mark.parametrize('strict', [None, '0'])
def test_non_comparison_parser_behavior_is_unchanged(native, monkeypatch, strict):
    if strict is None:
        monkeypatch.delenv('BASELINE_STRICT_COMPARISON', raising=False)
    else:
        monkeypatch.setenv('BASELINE_STRICT_COMPARISON', strict)
    with pytest.raises(ValidationError):
        parse(native, [{'lossless_restatement': 'Fact', 'location': ['Italy']}])


def test_does_not_coerce_other_fields_or_drop_entries(native, monkeypatch):
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', '1')
    with pytest.raises(ValidationError):
        parse(native, [{'lossless_restatement': 'Fact', 'location': ['Italy'], 'topic': ['Travel']}])
    payload = {'entries': [{'lossless_restatement': 'Same', 'location': ['Italy']}] * 2}
    assert [entry.location for entry in parse(native, payload)] == ['Italy', 'Italy']


def test_process_window_stores_complete_entries_without_new_model_retry(native, monkeypatch):
    monkeypatch.setenv('BASELINE_STRICT_COMPARISON', '1')
    builder = native.MemoryBuilder.__new__(native.MemoryBuilder)
    builder.dialogue_buffer = [native.Dialogue(dialogue_id=11, speaker='Anna', content='I visited Italy.')]
    builder.window_size = builder.step_size = 1
    builder.processed_count = 0
    builder.previous_entries = []
    builder.context_window = 32768
    builder.output_reserve = 1024
    builder._count_prompt_tokens = lambda _: 100
    builder.llm_client = Mock()
    builder.llm_client.chat_completion.return_value = '[{"lossless_restatement":"Anna visited Italy.","location":["Italy"]}]'
    builder.llm_client.extract_json.side_effect = json.loads
    builder.vector_store = Mock()
    builder.process_window()
    builder.llm_client.chat_completion.assert_called_once()
    builder.vector_store.add_entries.assert_called_once()
    entries = builder.vector_store.add_entries.call_args.args[0]
    assert entries[0].location == 'Italy'
    assert entries[0].lossless_restatement == 'Anna visited Italy.'
    assert builder.processed_count == 1
