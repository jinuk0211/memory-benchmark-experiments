"""Prove that each consumed model response completed; retain retry overhead."""
from collections import defaultdict


def audit_delivery(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get('request_kind') == 'chat_completion':
            groups[row.get('repair_group_id') or row['request_id']].append(row)
    issues, repaired, discarded_tokens = [], 0, 0
    for group, attempts in groups.items():
        attempts.sort(key=lambda x: x.get('repair_attempt', 0))
        terminal = attempts[-1]
        if any(row.get('repair_group_id') for row in attempts):
            if [row.get('repair_attempt') for row in attempts] != list(range(len(attempts))):
                issues.append({'group': group, 'error': 'missing_or_duplicate_repair_attempt'})
            keys = ('run_id', 'method', 'phase', 'sample_id', 'question_id', 'model')
            if any(any(row.get(key) != terminal.get(key) for key in keys) for row in attempts):
                issues.append({'group': group, 'error': 'repair_attribution_changed'})
        delivered = terminal.get('delivered_to_client', terminal.get('success'))
        if not delivered or not terminal.get('success') or any(
                reason in {'length', 'content_filter'} for reason in terminal.get('finish_reasons', [])) or not any(
                reason in {'stop', 'tool_calls', 'function_call'} for reason in terminal.get('finish_reasons', [])):
            issues.append({'group': group, 'error': 'no_complete_terminal_response'})
        for row in attempts[:-1]:
            if (row.get('delivered_to_client') is not False or not row.get('retry_scheduled')
                    or row.get('success') is not False):
                issues.append({'group': group, 'error': 'unaudited_discarded_attempt'})
            discarded_tokens += (row.get('usage') or {}).get('total_tokens', 0)
        repaired += len(attempts) > 1
    return {'complete': bool(groups) and not issues, 'logical_requests': len(groups),
            'repaired_requests': repaired, 'discarded_attempt_total_tokens': discarded_tokens,
            'issues': issues, 'note': 'All attempts remain in usage totals. Only complete terminal responses may reach a memory implementation.'}
