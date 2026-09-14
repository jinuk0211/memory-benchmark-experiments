"""Original native fallback is an explicit semantic outcome, never a fake stop."""
import copy
import importlib

import pytest

RUN = 'locomo-amem-original-policy-20260908-r26'
POLICY = 'amem_original_failure_v1'
EXPECTED = {('conv-26', 0): {}}


def row(request, phase='qa', kind='chat_completion'):
    return {'schema_version': 1, 'request_id': request, 'response_id': 'response-' + request,
        'run_id': RUN, 'method': 'a_mem', 'phase': phase, 'sample_id': 'conv-26',
        'question_id': 0 if phase == 'qa' else None, 'request_kind': kind,
        'model': 'Qwen/Qwen3.5-9B', 'response_model': 'Qwen/Qwen3.5-9B',
        'usage_status': 'reported', 'usage_invalid': False, 'http_status': 200,
        'finish_reasons': ['stop'] if kind == 'chat_completion' else [],
        'success': True, 'delivered_to_client': True, 'error_type': None,
        'usage': {'prompt_tokens': 10, 'completion_tokens': 2 if kind == 'chat_completion' else 0,
                  'total_tokens': 12 if kind == 'chat_completion' else 10}}


def native(stage='metadata', outcome='metadata_json_fallback'):
    call = row('native-' + stage, 'memory_add')
    call.update(native_failure_policy=POLICY, native_stage=stage, native_content_sha256='a' * 64,
        repair_attempt=0, retry_scheduled=False, finish_reasons=['length'],
        success=False, error_type='native_length_output',
        comparison_policy={'effective': {'max_tokens': 1000}},
        usage={'prompt_tokens': 10, 'completion_tokens': 1000, 'total_tokens': 1010})
    event = {key: call[key] for key in ('run_id', 'method', 'phase', 'sample_id', 'question_id', 'response_id', 'response_model')}
    event.update(schema_version=1, policy=POLICY, operation_id='one-memory-operation',
                 stage=stage, outcome=outcome, finish_reason='length', content_sha256='a' * 64)
    if stage == 'evolution':
        event.update(neighbor_count=5, applied_neighbor_count=2 if outcome == 'evolution_applied' else 0,
                     context_fallback_count=1 if outcome == 'evolution_applied' else 0)
    return call, event


def audit(rows, events):
    return importlib.import_module('scripts.audit_amem_original_policy').audit_original_policy(rows, events, RUN, EXPECTED)


def test_native_length_is_counted_without_changing_ledger_or_qa_delivery():
    call, event = native()
    rows = [call, row('qa'), row('embed', kind='embedding')]
    before = copy.deepcopy((rows, [event]))
    result = audit(rows, [event])
    assert result['complete'] is True and result['original_policy_delivery_complete'] is True
    assert result['strict_all_generation_delivery']['complete'] is False
    assert result['native_length_responses'] == 1
    assert result['outcome_counts']['metadata_json_fallback'] == 1
    assert result['usage']['exact_tokens']['total_tokens'] == 1032
    assert result['events'][0]['usage_row'] == call
    assert (rows, [event]) == before


@pytest.mark.parametrize('stage,outcome', [('metadata', 'metadata_accepted'), ('metadata', 'metadata_json_fallback'),
    ('evolution', 'evolution_json_skip'), ('evolution', 'evolution_noop'), ('evolution', 'evolution_applied')])
def test_original_native_outcomes_are_explicit_and_operation_may_span_two_responses(stage, outcome):
    call, event = native(stage, outcome)
    if stage == 'evolution':
        meta, meta_event = native()
        result = audit([meta, call, row('qa')], [meta_event, event])
        assert result['native_response_count'] == 2
    else:
        call.update(finish_reasons=['stop'], success=True, error_type=None)
        event['finish_reason'] = 'stop'
        result = audit([call, row('qa')], [event])
    assert result['complete'] is True and result['outcome_counts'][outcome] >= 1


def test_duplicate_original_neighbor_actions_are_counted_without_deduplication():
    call, event = native('evolution', 'evolution_applied')
    event.update(neighbor_count=5, applied_neighbor_count=10, context_fallback_count=7)
    result = audit([call, row('qa')], [event])
    assert result['complete'] is True
    assert result['events'][0]['event']['applied_neighbor_count'] == 10


def test_qa_terminal_requires_stop_not_a_memory_policy_or_tool_call():
    call, event = native()
    qa = row('qa')
    qa['finish_reasons'] = ['tool_calls']
    assert audit([call, qa], [event])['complete'] is False


def test_no_recorded_calls_cannot_prove_zero_upstream_token_work():
    result = audit([], [])
    assert result['usage']['complete'] is False
    assert result['usage']['exact_tokens'] is None


@pytest.mark.parametrize('event', [None, [], 1])
def test_non_object_event_is_a_reportable_validation_failure(event):
    call, _ = native()
    with pytest.raises(ValueError, match='event must be an object'):
        audit([call, row('qa')], [event])


@pytest.mark.parametrize('bad_row', [None, [], 1])
def test_non_object_usage_is_a_reportable_validation_failure(bad_row):
    call, event = native()
    with pytest.raises(ValueError, match='usage row must be an object'):
        audit([bad_row, call, row('qa')], [event])


@pytest.mark.parametrize('policy', [None, [], 1, {'effective': None}])
def test_non_object_policy_metadata_is_a_reportable_validation_failure(policy):
    call, event = native()
    call['comparison_policy'] = policy
    with pytest.raises(ValueError, match='policy metadata must contain objects'):
        audit([call, row('qa')], [event])


@pytest.mark.parametrize('change', ['http', 'model'])
def test_qa_transport_and_model_metadata_must_be_consistent(change):
    call, event = native()
    qa = row('qa')
    qa['http_status' if change == 'http' else 'response_model'] = 502 if change == 'http' else 'other-model'
    assert audit([call, qa], [event])['complete'] is False


@pytest.mark.parametrize('failure', ['missing_event', 'duplicate_event', 'orphan_event', 'foreign_run',
    'wrong_model', 'wrong_stage', 'wrong_hash', 'wrong_cap', 'transport', 'qa_length', 'qa_native',
    'embedding_error', 'missing_usage', 'invalid_usage', 'duplicate_request', 'bad_count',
    'bad_outcome', 'wrong_finish', 'missing_operation', 'unknown_policy', 'foreign_qa'])
def test_unaccounted_or_non_native_failure_is_never_accepted(failure):
    call, event = native('evolution', 'evolution_applied')
    rows, events = [call, row('qa'), row('embed', kind='embedding')], [event]
    if failure == 'missing_event':
        events.clear()
    elif failure == 'duplicate_event':
        events.append(copy.deepcopy(event))
    elif failure == 'orphan_event':
        event['response_id'] = 'unmetered'
    elif failure == 'foreign_run':
        call['run_id'] = 'diagnostic'
    elif failure == 'wrong_model':
        event['response_model'] = 'other-model'
    elif failure == 'wrong_stage':
        event['stage'] = 'metadata'
    elif failure == 'wrong_hash':
        event['content_sha256'] = 'b' * 64
    elif failure == 'wrong_cap':
        call['comparison_policy']['effective']['max_tokens'] = 999
    elif failure == 'transport':
        call.update(http_status=502, error_type='http_error')
    elif failure == 'qa_length':
        rows[1].update(finish_reasons=['length'], success=False)
    elif failure == 'qa_native':
        call['phase'] = event['phase'] = 'qa'
    elif failure == 'embedding_error':
        rows[2]['success'] = False
    elif failure == 'missing_usage':
        rows[1]['usage'] = None
    elif failure == 'invalid_usage':
        rows[2]['usage']['prompt_tokens'] = True
    elif failure == 'duplicate_request':
        rows[1]['request_id'] = call['request_id']
    elif failure == 'bad_count':
        event['context_fallback_count'] = 6
    elif failure == 'bad_outcome':
        event['outcome'] = 'silently_accepted'
    elif failure == 'wrong_finish':
        event['finish_reason'] = 'stop'
    elif failure == 'missing_operation':
        event['operation_id'] = ''
    elif failure == 'unknown_policy':
        call['native_failure_policy'] = 'other'
    else:
        rows[1]['question_id'] = 999
    result = audit(rows, events)
    assert result['complete'] is False and result['issues']
    if failure in ('missing_usage', 'invalid_usage'):
        assert result['usage']['exact_tokens'] is None
