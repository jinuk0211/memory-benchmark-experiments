"""Native JSON rejection is overhead within, not additional to, metered usage."""
import copy
import importlib
import json

import pytest


def audit(rows, logs):
    target = importlib.import_module('scripts.audit_simplemem_semantic_discards')
    return target.audit_semantic_discards(rows, logs)


def row(response_id='rejected', **changes):
    return {
        'response_id': response_id, 'request_id': 'request-' + response_id,
        'run_id': 'run-17', 'method': 'simplemem', 'phase': 'qa',
        'sample_id': 'conv-30', 'question_id': 'conv-30_qa68',
        'request_kind': 'chat_completion', 'success': True,
        'delivered_to_client': True, 'usage_status': 'reported',
        'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
        **changes,
    }


def event(response_id='rejected', **changes):
    return {'response_id': response_id, 'attempt': 0, 'retry_scheduled': True,
            'error': 'Reflection JSON must be an object',
            'rejected_content': '[]', **changes}


def line(payload=None):
    return 'SIMPLEMEM_JSON_REPAIR ' + json.dumps(event() if payload is None else payload)


def test_transport_success_can_be_semantically_discarded_without_double_counting():
    rows = [row(), row('accepted')]
    original = copy.deepcopy(rows)
    diagnostic = event(rejected_content='[]\nSIMPLEMEM_JSON_REPAIR is content')
    result = audit(rows, ['startup\n' + line(diagnostic), 'ordinary second log'])
    assert result['complete'] is True
    assert result['matched_discard_count'] == result['metered_discard_count'] == 1
    assert result['event_count'] == 1
    assert result['recorded_tokens'] == result['exact_tokens'] == original[0]['usage']
    assert rows == original and sum(r['usage']['total_tokens'] for r in rows) == 240
    saved = result['events'][0]
    assert saved['event'] == diagnostic
    assert saved['usage_row'] == original[0]
    assert saved['log_index'] == 0 and saved['line_number'] == 2
    assert saved['raw_line'] == line(diagnostic)
    saved['usage_row']['usage']['total_tokens'] = 999
    assert rows == original


@pytest.mark.parametrize('payload', ['{', '[]', 'null'])
def test_malformed_or_nonobject_event_fails_closed_and_retains_raw_line(payload):
    raw = 'SIMPLEMEM_JSON_REPAIR ' + payload
    result = audit([row()], [raw])
    assert result['complete'] is False and result['exact_tokens'] is None
    assert result['events'][0]['raw_line'] == raw
    assert result['issues']


@pytest.mark.parametrize('response_id', [None, '', ' ', 42, ['rejected']])
def test_invalid_response_id_fails_closed(response_id):
    result = audit([row()], [line(event(response_id))])
    assert result['complete'] is False and result['matched_discard_count'] == 0
    assert result['exact_tokens'] is None


def test_unmatched_response_is_unknown_not_zero():
    result = audit([row('accepted')], [line()])
    assert result['complete'] is False and result['exact_tokens'] is None
    assert result['recorded_tokens']['total_tokens'] == 0
    assert result['matched_discard_count'] == 0
    assert result['issues'][0]['error'] == 'unmatched_response_id'


def test_duplicate_events_fail_closed_but_never_double_count():
    result = audit([row()], [line(), line(event(attempt=1))])
    assert result['complete'] is False and result['event_count'] == 2
    assert result['matched_discard_count'] == result['metered_discard_count'] == 1
    assert result['recorded_tokens']['total_tokens'] == 120
    assert result['exact_tokens'] is None
    assert result['issues'][0]['error'] == 'duplicate_event_response_id'


def test_ambiguous_ledger_response_id_is_not_silently_deduplicated():
    result = audit([row(), row()], [line()])
    assert result['complete'] is False and result['matched_discard_count'] == 0
    assert result['exact_tokens'] is None
    assert result['issues'][0]['error'] == 'ambiguous_response_id'


@pytest.mark.parametrize('changes', [
    {'method': 'a_mem'}, {'request_kind': 'embedding'}, {'method': None},
])
def test_wrong_method_or_request_kind_is_not_simplemem_overhead(changes):
    result = audit([row(**changes)], [line()])
    assert result['complete'] is False and result['metered_discard_count'] == 0
    assert result['exact_tokens'] is None


@pytest.mark.parametrize('usage', [None, {}, [],
    {'prompt_tokens': True, 'completion_tokens': 20, 'total_tokens': 21},
    {'prompt_tokens': -1, 'completion_tokens': 20, 'total_tokens': 19},
    {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 121},
    {'prompt_tokens': 100, 'completion_tokens': 20.0, 'total_tokens': 120},
])
def test_missing_or_invalid_usage_retains_matched_identity_but_unknown_exact_cost(usage):
    result = audit([row(usage=usage)], [line()])
    assert result['complete'] is False and result['matched_discard_count'] == 1
    assert result['metered_discard_count'] == 0 and result['exact_tokens'] is None
    assert result['events'][0]['usage_row']['usage'] == usage


@pytest.mark.parametrize('changes', [{'usage_status': 'missing'}, {'usage_invalid': True}])
def test_untrusted_usage_status_is_not_overridden_by_numeric_shape(changes):
    result = audit([row(**changes)], [line()])
    assert result['complete'] is False and result['metered_discard_count'] == 0


def test_partial_known_tokens_are_lower_bound_not_exact_total():
    result = audit([row(), row('unknown', usage=None)],
                   [line() + '\n' + line(event('unknown', retry_scheduled=False))])
    assert result['complete'] is False and result['matched_discard_count'] == 2
    assert result['recorded_tokens']['total_tokens'] == 120
    assert result['exact_tokens'] is None


@pytest.mark.parametrize('changes', [{'attempt': True}, {'attempt': -1},
                                   {'retry_scheduled': 1}, {'error': None},
                                   {'rejected_content': None}])
def test_missing_or_malformed_required_event_provenance_fails_closed(changes):
    result = audit([row()], [line(event(**changes))])
    assert result['complete'] is False and result['exact_tokens'] is None


def test_no_events_only_proves_no_observed_discards_in_supplied_logs():
    result = audit([row()], ['normal output'])
    assert result['complete'] is True and result['event_count'] == 0
    assert result['exact_tokens'] == {'prompt_tokens': 0, 'completion_tokens': 0,
                                     'total_tokens': 0}
    assert 'supplied logs' in result['note']


@pytest.mark.parametrize('separator', ['\u2028', '\u2029', '\u0085'])
@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_unicode_content_is_not_a_physical_log_record_boundary(separator, newline):
    diagnostic = event(rejected_content='part A' + separator + 'part B')
    raw = 'SIMPLEMEM_JSON_REPAIR ' + json.dumps(diagnostic, ensure_ascii=False)
    result = audit([row()], ['startup' + newline + raw + newline + 'done'])
    assert result['complete'] is True and result['event_count'] == 1
    assert result['events'][0]['event'] == diagnostic
    assert result['events'][0]['raw_line'] == raw + newline.removesuffix('\n')
    assert result['events'][0]['line_number'] == 2
    assert result['exact_tokens'] == row()['usage']
