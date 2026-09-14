"""Attribute native SimpleMem JSON rejections within the existing usage ledger."""
import json
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from scripts.metered_openai_proxy import usage_status


def audit_semantic_discards(
    usage_rows: Iterable[dict[str, Any]], log_texts: Iterable[str],
) -> dict[str, Any]:
    """Audit only supplied logs; never create ledger rows or add overhead twice.

    `recorded_tokens` is a lower bound when `exact_tokens` is None. Completion
    says nothing about whether all historical client logs were supplied.
    """
    by_response = defaultdict(list)
    for row in usage_rows:
        response_id = row.get('response_id')
        if isinstance(response_id, str) and response_id.strip():
            by_response[response_id].append(row)
    prefix = 'SIMPLEMEM_JSON_REPAIR '
    events, issues, seen = [], [], set()
    tokens = dict.fromkeys(('prompt_tokens', 'completion_tokens', 'total_tokens'), 0)
    matched = metered = 0
    for log_index, text in enumerate(log_texts):
        # Native ensure_ascii=False JSON can contain Unicode line separators.
        for line_number, raw in enumerate(text.split('\n'), start=1):
            if not raw.startswith(prefix):
                continue
            entry = {'log_index': log_index, 'line_number': line_number,
                     'raw_line': raw, 'event': None, 'usage_row': None}
            events.append(entry)
            try:
                payload = json.loads(raw[len(prefix):])
                entry['event'] = payload
                if not isinstance(payload, dict):
                    raise ValueError('event_not_object')  # noqa: TRY004 - invalid log value
                response_id = payload.get('response_id')
                if not isinstance(response_id, str) or not response_id.strip():
                    raise ValueError('invalid_response_id')
                if (type(payload.get('attempt')) is not int or payload['attempt'] < 0
                        or type(payload.get('retry_scheduled')) is not bool
                        or not isinstance(payload.get('error'), str)
                        or not isinstance(payload.get('rejected_content'), str)):
                    raise ValueError('invalid_event_provenance')
                if response_id in seen:
                    raise ValueError('duplicate_event_response_id')
                seen.add(response_id)
                candidates = by_response.get(response_id, [])
                if not candidates:
                    raise ValueError('unmatched_response_id')
                if len(candidates) != 1:
                    raise ValueError('ambiguous_response_id')
                row = candidates[0]
                entry['usage_row'] = deepcopy(row)
                if row.get('method') != 'simplemem':
                    raise ValueError('method_mismatch')
                if row.get('request_kind') != 'chat_completion':
                    raise ValueError('request_kind_mismatch')
                matched += 1
                usage = row.get('usage')
                status = usage_status(usage, 'chat_completion')
                if (status != 'reported' or row.get('usage_status') != 'reported'
                        or row.get('usage_invalid')):
                    raise ValueError('missing_or_invalid_usage')
                for key in tokens:
                    tokens[key] += usage[key]
                metered += 1
            except json.JSONDecodeError as error:
                issues.append({'event_index': len(events) - 1,
                               'error': 'malformed_event_json', 'detail': str(error)})
            except ValueError as error:
                issues.append({'event_index': len(events) - 1, 'error': str(error)})
    return {
        'complete': not issues, 'event_count': len(events),
        'matched_discard_count': matched, 'metered_discard_count': metered,
        'recorded_tokens': tokens, 'exact_tokens': dict(tokens) if not issues else None,
        'events': events, 'issues': issues,
        'note': ('Only supplied logs are audited. These tokens are already in the '
                 'usage ledger; do not add them again. Recorded tokens are a lower '
                 'bound when exact_tokens is null. This is not a delivery audit.'),
    }
