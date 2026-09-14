"""CPU-only r2 launcher scope, native tables, and process receipt contracts."""

import csv
import json
import signal
import sys
from pathlib import Path

import pytest
import test_full_run
import test_primary_continuation
from test_full_run import repin, sha, write_json

bundle = test_full_run.bundle
target = test_full_run.target
native = test_primary_continuation.native

SELECTED = ('conv-47', 'conv-48', 'conv-49', 'conv-50')


@pytest.fixture
def primary(bundle, monkeypatch):
    b = bundle
    for sample in b.data:
        sample['conversation'] = {'session_1': [{}], 'session_2': [{}]}
    write_json(b.dataset, b.data)
    b.manifest['files_sha256'][str(b.dataset)] = sha(b.dataset)
    monkeypatch.setattr(b.target, 'DATASET_SHA256', sha(b.dataset))
    b.preflight['dataset_sha256'] = sha(b.dataset)
    repin(b)
    return b


def rows_for(prepared):
    return [{'conv_id': cid, 'question_ordinal': ordinal, 'config': 'full', 'policy': policy,
             'category': qa['category'], 'question': qa['question'], 'gold': str(qa['answer']),
             'pred': 'answer'} for (cid, ordinal), qa in prepared['questions'].items()
            for policy in ('certified', 'full_raw')]


def tables(prepared):
    return {
        'items.csv': rows_for(prepared),
        'summary.csv': [{'config': 'full', 'policy': policy, 'n': 655, 'f1': 0.5}
                        for policy in ('certified', 'full_raw')],
        'write_stats.csv': [{'conv_id': cid, 'config': 'full', 'session': session,
                             'mem_tok': 10, 'raw_tok': 20, 'deferred': 0}
                            for cid in SELECTED for session in (1, 2)],
    }


def write_tables(out, data):
    for name, rows in data.items():
        with (out / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def test_fixed_r2_identity_and_primary_matrix(target):
    assert target.ROOT == Path('/workspace/CertMem-primary-r2')
    assert target.OUTPUT == Path('/workspace/locomo-certmem-primary-r2')
    assert target.RUN_ID == 'locomo-certmem-primary-r2'
    assert target.POLICIES == {'full': ('certified', 'full_raw')}


def test_command_selects_only_remaining_655_but_validates_canonical_1540(primary):
    b, t = primary, primary.target
    prepared = t.preflight(b.mp, sha(b.mp))
    assert prepared['command'] == [t.PYTHON, '-u', str(b.root / 'scripts/run_system_v15.py'),
        '--comparison', '--primary-only', '--dataset', 'locomo', '--configs', 'full',
        '--conversation-ids', ','.join(SELECTED), '--budgets', '400,1600,4000', '--out', str(b.out)]
    assert len(prepared['questions']) == 655
    assert {cid for cid, ordinal in prepared['questions']} == set(SELECTED)
    assert prepared['write_sessions'] == {(cid, session) for cid in SELECTED for session in (1, 2)}
    assert len(b.data) == 10
    assert sum(len(sample['qa']) - 1 for sample in b.data) == 1540
    assert b.preflight['qa_total'] == 1540 and len(b.preflight['rows']) == 10
    assert not b.out.exists()


def test_generated_command_is_accepted_by_native_parser_before_model(primary, native, monkeypatch):
    command = primary.target.preflight(primary.mp, sha(primary.mp))['command']

    class BeforeModel(Exception):
        pass

    def no_model(cfg):
        raise BeforeModel

    native.namespace['Engine'] = no_model
    monkeypatch.setattr(sys, 'argv', command[2:])
    with pytest.raises(BeforeModel):
        native.namespace['main']()


@pytest.mark.parametrize('change', ['completed_context_overflow', 'missing_completed_context',
                                   'r1_run_id', 'wrong_selected_counts'])
def test_full_preflight_and_new_identity_remain_mandatory(primary, monkeypatch, change):
    b, t = primary, primary.target
    if change == 'completed_context_overflow':
        b.preflight['rows'][0].update(max_chat_prompt_tokens=50000, max_total_tokens=50032)
    elif change == 'missing_completed_context':
        b.preflight['rows'].pop(0)
    elif change == 'r1_run_id':
        b.manifest['run_id'] = 'locomo-certmem-v15-20260908-r1'
    else:
        b.data[6]['qa'].insert(0, b.data[7]['qa'].pop(0))
        write_json(b.dataset, b.data)
        b.manifest['files_sha256'][str(b.dataset)] = sha(b.dataset)
        monkeypatch.setattr(t, 'DATASET_SHA256', sha(b.dataset))
        b.preflight['dataset_sha256'] = sha(b.dataset)
        b.preflight['rows'][6]['qa_count'] += 1
        b.preflight['rows'][7]['qa_count'] -= 1
    repin(b)
    with pytest.raises(ValueError):
        t.preflight(b.mp, sha(b.mp))
    assert not b.out.exists()


@pytest.mark.parametrize('change', [None, 'empty_prediction', 'missing_qa', 'duplicate_qa',
    'completed_context', 'extra_policy', 'summary_count', 'summary_missing', 'summary_duplicate',
    'write_missing_session', 'write_duplicate', 'write_other_context', 'write_other_config'])
def test_native_two_policy_matrix_and_summary_write_tables_are_exact(primary, change):
    b, t = primary, primary.target
    prepared = t.preflight(b.mp, sha(b.mp))
    data = tables(prepared)
    if change == 'empty_prediction': data['items.csv'][0]['pred'] = ' '
    elif change == 'missing_qa': data['items.csv'].pop()
    elif change == 'duplicate_qa': data['items.csv'][-1] = data['items.csv'][0].copy()
    elif change == 'completed_context': data['items.csv'][0]['conv_id'] = 'conv-26'
    elif change == 'extra_policy': data['items.csv'][0]['policy'] = 'units_only'
    elif change == 'summary_count': data['summary.csv'][0]['n'] = 654
    elif change == 'summary_missing': data['summary.csv'].pop()
    elif change == 'summary_duplicate': data['summary.csv'][-1] = data['summary.csv'][0].copy()
    elif change == 'write_missing_session': data['write_stats.csv'].pop()
    elif change == 'write_duplicate': data['write_stats.csv'][-1] = data['write_stats.csv'][0].copy()
    elif change == 'write_other_context': data['write_stats.csv'][0]['conv_id'] = 'conv-26'
    elif change == 'write_other_config': data['write_stats.csv'][0]['config'] = 'no_residual'
    b.out.mkdir()
    write_tables(b.out, data)
    before = {name: (b.out / name).read_bytes() for name in data}
    report = t.audit_outputs(b.out, prepared['questions'], prepared['write_sessions'])
    assert report['complete'] is (change in (None, 'empty_prediction'))
    assert report['expected_rows'] == 1310
    assert report['empty_predictions'] == (change == 'empty_prediction')
    if change is None:
        assert report['rows'] == 1310 and report['full_raw_rows'] == 655
        assert report['summary_rows'] == 2 and report['write_stats_rows'] == 8
    assert before == {name: (b.out / name).read_bytes() for name in data}


@pytest.mark.parametrize('exit_code', [0, 7])
def test_foreground_receipt_distinguishes_selected_qa_from_full_token_preflight(primary, monkeypatch, exit_code):
    b, t = primary, primary.target
    prepared = t.preflight(b.mp, sha(b.mp))
    calls = []

    class Child:
        def wait(self, timeout=None):
            write_tables(b.out, tables(prepared))
            return exit_code

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return Child()

    monkeypatch.setattr(t.subprocess, 'Popen', spawn)
    monkeypatch.setattr(t, 'gpu_sample', lambda: None)
    monkeypatch.setattr(t.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(t.signal, 'getsignal', lambda sig: signal.SIG_DFL)
    assert t.run(b.mp, sha(b.mp)) == (0 if exit_code == 0 else 1)
    receipt = json.loads((b.out / 'receipt.json').read_text())
    assert receipt['run_id'] == 'locomo-certmem-primary-r2'
    assert receipt['native_outputs']['expected_rows'] == 1310
    assert receipt['full_raw_policy']['qa_total'] == 655
    assert receipt['full_raw_policy']['canonical_dataset_qa_total'] == 1540
    assert receipt['inference_complete'] is (exit_code == 0)
    assert receipt['complete'] is False and receipt['usage_complete'] is None
    assert receipt['official_scoring_complete'] is False and receipt['exact_tokens'] is None
    assert len(calls) == 1 and calls[0][0] == prepared['command']
    assert calls[0][1]['env']['HF_HUB_OFFLINE'] == '1'
