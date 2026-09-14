"""Audit explicit native memory outcomes without rewriting a metered response.

Only metadata/evolution memory_add calls may use the original native failure
policy. QA and transport failures remain strict. Events never create token rows.
"""
import re
from collections import Counter, defaultdict
from copy import deepcopy

from scripts.response_delivery_audit import audit_delivery
from scripts.score_locomo_comparison import summarize_usage

POLICY = 'amem_original_failure_v1'
OUTCOMES = {'metadata': {'metadata_accepted', 'metadata_json_fallback'},
            'evolution': {'evolution_json_skip', 'evolution_noop', 'evolution_applied'}}


def audit_original_policy(rows: list[dict], events: list[dict], run_id: str, expected: dict) -> dict:
    """Join every original-policy response once; missing numeric usage stays unknown."""
    issues, joined, native, strict_rows = [], [], defaultdict(list), []
    samples = {sample for sample, _ in expected}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('native policy usage row must be an object')
        phase, sample = row.get('phase'), row.get('sample_id')
        if (row.get('run_id') != run_id or row.get('method') != 'a_mem' or sample not in samples
                or phase not in ('initialize', 'memory_add', 'memory_finalize', 'qa')
                or row.get('request_kind') not in ('chat_completion', 'embedding')):
            issues.append('foreign_usage_attribution')
        if (row.get('http_status') != 200 or row.get('request_kind') == 'chat_completion'
                and any(row.get(key) != 'Qwen/Qwen3.5-9B' for key in ('model', 'response_model'))):
            issues.append('transport_or_model_metadata_mismatch')
        if phase == 'qa':
            question = row.get('question_id')
            valid = type(question) is int or (isinstance(question, str) and question.isascii()
                and question.isdecimal() and str(int(question)) == question)
            if not valid or (sample, int(question)) not in expected:
                issues.append('foreign_qa_identity')
        if row.get('native_failure_policy') is not None:
            policy_metadata = row.get('comparison_policy')
            if (not isinstance(policy_metadata, dict)
                    or not isinstance(policy_metadata.get('effective'), dict)):
                raise ValueError('native policy metadata must contain objects')
            native[row.get('response_id')].append(row)
        else:
            strict_rows.append(row)
            if phase == 'memory_add' and row.get('request_kind') == 'chat_completion':
                issues.append('missing_native_memory_policy')
        if row.get('request_kind') == 'embedding' and (row.get('success') is not True or row.get('http_status') != 200):
            issues.append('embedding_failure')
    usage = summarize_usage(rows, True)
    if any(row.get('usage_invalid') or row.get('usage_status') != 'reported' for row in rows):
        usage['complete'] = False
    usage['exact_tokens'] = usage['reported_tokens'].copy() if usage['complete'] else None
    if not usage['complete']:
        issues.append('incomplete_usage')
    seen, counts = set(), Counter()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('native semantic event must be an object')
        response = event.get('response_id')
        matches = native.get(response, [])
        joined.append({'event': deepcopy(event), 'usage_row': deepcopy(matches[0]) if len(matches) == 1 else None})
        if not isinstance(response, str) or not response or response in seen or len(matches) != 1:
            issues.append('missing_duplicate_or_unmatched_native_event')
            continue
        seen.add(response)
        row, stage = matches[0], event.get('stage')
        reason = event.get('finish_reason')
        if (event.get('schema_version') != 1 or event.get('policy') != POLICY
                or row.get('native_failure_policy') != POLICY or stage not in OUTCOMES
                or event.get('outcome') not in OUTCOMES.get(stage, set()) or row.get('native_stage') != stage
                or not isinstance(event.get('operation_id'), str) or not event['operation_id']
                or any(event.get(key) != row.get(key) for key in
                       ('run_id', 'method', 'phase', 'sample_id', 'question_id', 'response_model'))
                or row.get('phase') != 'memory_add' or row.get('request_kind') != 'chat_completion'
                or row.get('model') != 'Qwen/Qwen3.5-9B' or row.get('response_model') != 'Qwen/Qwen3.5-9B'
                or reason not in ('stop', 'length') or row.get('finish_reasons') != [reason]
                or not isinstance(event.get('content_sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', event['content_sha256']) is None
                or event['content_sha256'] != row.get('native_content_sha256')
                or row.get('http_status') != 200 or row.get('delivered_to_client') is not True
                or row.get('success') is not (reason == 'stop')
                or row.get('error_type') != ('native_length_output' if reason == 'length' else None)
                or type(row.get('repair_attempt')) is not int or row['repair_attempt'] != 0
                or row.get('retry_scheduled', False) is not False
                or row.get('comparison_policy', {}).get('effective', {}).get('max_tokens') != 1000):
            issues.append('invalid_native_outcome_provenance')
            continue
        if stage == 'evolution':
            values = [event.get(key) for key in ('neighbor_count', 'applied_neighbor_count', 'context_fallback_count')]
            if (any(type(value) is not int or value < 0 for value in values)
                    or values[2] > values[1]
                    or event['outcome'] != 'evolution_applied' and any(values[1:])):
                issues.append('invalid_native_neighbor_counts')
                continue
        counts[event['outcome']] += 1
    if not native or seen != set(native) or any(len(values) != 1 for values in native.values()):
        issues.append('native_event_coverage_incomplete')
    strict = audit_delivery(rows)
    remaining = audit_delivery(strict_rows)
    if not remaining['complete']:
        issues.append('strict_non_native_delivery_failed')
    qa_groups = defaultdict(list)
    for row in strict_rows:
        if row.get('phase') == 'qa' and row.get('request_kind') == 'chat_completion':
            qa_groups[row.get('repair_group_id') or row['request_id']].append(row)
    if any(max(group, key=lambda row: row.get('repair_attempt', 0)).get('finish_reasons') != ['stop']
           for group in qa_groups.values()):
        issues.append('qa_terminal_must_stop')
    complete = not issues
    return {'complete': complete, 'original_policy_delivery_complete': complete,
        'policy': POLICY, 'usage': usage, 'strict_all_generation_delivery': strict,
        'strict_non_native_delivery': remaining, 'events': joined, 'issues': issues,
        'native_response_count': sum(map(len, native.values())),
        'native_length_responses': sum(row.get('finish_reasons') == ['length'] for group in native.values() for row in group),
        'outcome_counts': dict(counts), 'outcome_counts_complete': complete,
        'note': 'Length remains length and native fallback remains explicit. Every original row stays in gross usage; '
                'semantic events add no token rows. Counts are only recorded lower bounds when this audit is incomplete.'}
