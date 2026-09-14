"""CPU regressions for common metadata parsing with untouched frozen source."""
import copy
from datetime import date
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'source')]
import date_metadata_adapter as adapter  # noqa: E402

pytest_plugins = ('test_refined_runtime',)


@pytest.fixture
def calendar_modules(monkeypatch):
    modules = [importlib.import_module(name) for name in ('continuous', 'continuous_v2')]
    for module in modules:
        original = module.parse_date
        if getattr(original, '_date_metadata_adapter_version', None) == adapter.VERSION:
            original = original.__wrapped__
        monkeypatch.setattr(module, 'parse_date', original)
    return modules


@pytest.mark.parametrize('text', ['8 May 2023', '10:00 am on 8 May, 2023', 'unknown', 'May 2023'])
def test_original_locomo_success_and_unknown_behavior_unchanged(calendar_modules, text):
    expected = [module.parse_date(text) for module in calendar_modules]
    adapter.install_date_adapter()
    assert [module.parse_date(text) for module in calendar_modules] == expected


@pytest.mark.parametrize('text,expected', [
    ('2023/05/30 (Tue) 23:40', date(2023, 5, 30)),
    ('2024/02/29 (Thu) 00:00', date(2024, 2, 29)),
    ('2023-05-30', date(2023, 5, 30)),
    ('2023-05-30 23:40', date(2023, 5, 30)),
])
def test_complete_generic_numeric_formats(calendar_modules, text, expected):
    assert all(module.parse_date(text) is None for module in calendar_modules)
    adapter.install_date_adapter()
    assert all(module.parse_date(text) == expected for module in calendar_modules)
    assert all(type(module.parse_date(text)) is date for module in calendar_modules)


@pytest.mark.parametrize('text', [
    '2023/05/30 (Mon) 23:40', '2023/02/30 (Thu) 01:00', '2023/05/30 (Tue) 24:00',
    '2023/05/30 (Tue) 23:60', ' 2023-05-30', '2023-05-30 trailing text',
    '2023-5-30', '2023/05/30', '2023-02-29', '2023-05-30T23:40Z',
])
def test_invalid_or_unsupported_fallback_keeps_original_none(calendar_modules, text):
    assert all(module.parse_date(text) is None for module in calendar_modules)
    adapter.install_date_adapter()
    assert all(module.parse_date(text) is None for module in calendar_modules)


def test_original_result_and_exception_identity_are_preserved():
    value = date(2023, 5, 30)
    assert adapter.wrap_parser(lambda text: value)('unrecognized by fallback') is value
    error = ValueError('original parser failed')
    def original(text):
        raise error
    with pytest.raises(ValueError) as caught:
        adapter.wrap_parser(original)('2023/05/30 (Tue) 23:40')
    assert caught.value is error


def test_original_invalid_named_month_error_is_not_swallowed(calendar_modules):
    adapter.install_date_adapter()
    for module in calendar_modules:
        with pytest.raises(ValueError):
            module.parse_date('31 February 2023')


def source_fixture():
    stamp = '2023/05/30 (Tue) 23:40'
    sample = {'sample_id': 'synthetic-history', 'conversation': {
        'session_1_date_time': stamp,
        'session_1': [{'speaker': 'Alice', 'text': 'I moved yesterday.', 'dia_id': 'D1:1'}],
    }, 'qa': [{'question_date': stamp, 'question': 'unused question', 'answer': 'unused gold'}]}
    return sample


def test_both_real_temporal_call_paths_use_common_parser_without_mutating_input(calendar_modules):
    import refine
    import memory_ops
    import portable_parent
    sample = source_fixture()
    sample_before = copy.deepcopy(sample)
    sessions = refine.session_data(sample)
    original_sessions = copy.deepcopy(sessions)
    seed = [{'text': '(Session date: ' + sessions[0]['date'] + ') Alice moved yesterday.',
             'sources': ['D1:1'], 'session': 1, 'kind': 'fact'}]
    original_seed = copy.deepcopy(seed)
    assert all(module.calendar_anchor('Alice moved yesterday.', sessions[0]['date']) == 'Alice moved yesterday.'
               for module in calendar_modules)
    adapter.install_date_adapter()
    for module in calendar_modules:
        assert module.calendar_anchor('Alice moved yesterday.', sessions[0]['date']) == (
            'Alice moved 29 May 2023 [original relative expression: yesterday].')
    rt = SimpleNamespace(ntok=lambda text: len(text.split()))
    r06 = calendar_modules[1].construct(rt, sessions, seed, {
        'operations': [{'op': 'anchor_time', 'style': 'month'}],
    })
    r40 = memory_ops.fuse_evidence(rt, sessions, r06, {'anchor': True, 'size': 4, 'overlap': 0})
    refined, _, _ = portable_parent.augment(rt, sessions, r40, [])
    assert any('29 May 2023 [original relative expression: yesterday]' in unit['text'] for unit in r06)
    assert any('29 May 2023 [original relative expression: yesterday]' in unit['text'] for unit in r40)
    assert refined == r40  # No source utility option: Refined preserves the same anchored parent.
    assert sample == sample_before and sessions == original_sessions and seed == original_seed
    assert sessions[0]['date'] == sample['qa'][0]['question_date'] == '2023/05/30 (Tue) 23:40'
    assert 'yesterday' in seed[0]['text']  # No temporal stage is injected into Seed.


def test_install_is_idempotent_and_binds_both_actual_callsite_globals(calendar_modules):
    profile = adapter.install_date_adapter()
    installed = [module.parse_date for module in calendar_modules]
    assert adapter.install_date_adapter() == profile
    assert [module.parse_date for module in calendar_modules] == installed
    for module in calendar_modules:
        assert module.calendar_anchor.__globals__['parse_date'] is module.parse_date
        assert module.anchor_units.__globals__['parse_date'] is module.parse_date
    assert profile['scope'].startswith('Common parser compatibility')
    assert not any('dataset' in key or 'method_id' in key or 'question_id' in key for key in profile)


def test_foreign_module_is_refused_without_partially_installing(monkeypatch, calendar_modules):
    first, second = calendar_modules
    original = first.parse_date
    monkeypatch.setattr(second, '__file__', '/different/project/continuous_v2.py')
    with pytest.raises(ValueError, match='foreign calendar module'):
        adapter.install_date_adapter()
    assert first.parse_date is original


def test_changed_parser_is_refused_without_partially_installing(monkeypatch, calendar_modules):
    first, second = calendar_modules
    original = first.parse_date
    monkeypatch.setattr(second, 'parse_date', lambda text: None)
    with pytest.raises(ValueError, match='not the frozen source'):
        adapter.install_date_adapter()
    assert first.parse_date is original


def test_protocol_records_adapter_and_installs_after_source_check_before_data(setup, monkeypatch):
    b = setup
    monkeypatch.setitem(sys.modules, 'transfer_runtime', b.t)
    spec = importlib.util.spec_from_file_location('date_paper_runner', ROOT / 'run_transfer.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner.importlib.metadata, 'version', lambda name: 'cpu-test-stub')
    events = []
    original_plans, original_install, original_samples = runner.source_plans, runner.install_date_adapter, runner.load_samples
    def plans():
        events.append('source_validation')
        return original_plans()
    def install():
        events.append('date_installation')
        return original_install()
    def samples(args):
        events.append('data_load')
        return original_samples(args)
    monkeypatch.setattr(runner, 'source_plans', plans)
    monkeypatch.setattr(runner, 'install_date_adapter', install)
    monkeypatch.setattr(runner, 'load_samples', samples)
    data, env, output = b.out / 'data.json', b.out / 'env.json', b.out / 'new-date-run'
    data.write_text(json.dumps([source_fixture()]), encoding='utf-8')
    env.write_text(json.dumps(b.env), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', [str(ROOT / 'run_transfer.py'), 'plan', '--dataset', 'locomo',
        '--data', str(data), '--environment', str(env), '--out', str(output)])
    runner.main()
    assert events == ['source_validation', 'date_installation', 'data_load']
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    assert protocol['date_metadata_compatibility'] == adapter.PROFILE
    assert protocol['source_hashes']['date_metadata_adapter.py'] == hashlib.sha256((ROOT / 'date_metadata_adapter.py').read_bytes()).hexdigest()
    assert protocol['methods'] == ['seed', 'r40_fused_four_turn', 's_parent_single_2000']
    assert protocol['read_budget'] == 2048 and protocol['max_answer_tokens'] == 96
    assert protocol['extra_storage_budget'] == 2000
    assert not b.engines and not b.encoders
    sessions = json.loads((output / 'source_sessions.json').read_text(encoding='utf-8'))
    assert sessions['synthetic-history'][0]['date'] == source_fixture()['qa'][0]['question_date']
