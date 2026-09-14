"""Meter every attempt; retry incomplete generations without consuming partial text.

Successful first requests retain their original prompts and sampling settings.
Repairs add explicit format feedback only; model, context and output limits,
temperature, facts, retrieval and tool-round settings are not reduced.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.metered_openai_proxy import (
    MeteredProxy, ProxyHandler, ResponseMeter, comparison_payload,
)

REPAIR_FEEDBACK = (
    'The previous generation was incomplete or empty and was discarded without applying it. '
    'Regenerate a complete, finite response to the original task. Do not repeat items, tags, '
    'sentences, or sections. Include each required fact once, preserve all relevant facts, '
    'and close every JSON object and array. Match the exact number of neighbors when '
    'neighbor arrays are requested. Do not mention this formatting correction in the answer.'
)


def repair_feedback(original, previous_response):
    """Use the requested XML contract when a native E-MEM aggregator loops."""
    aggregator = any(isinstance(message.get('content'), str)
        and message['content'].startswith('# ROLE: Memory Fact Aggregator & Logic Solver')
        and '<aggregator_output>' in message['content']
        for message in original.get('messages', []))
    if not aggregator:
        return 'generic-format-v1', REPAIR_FEEDBACK
    text = '\n'.join((choice.get('message') or {}).get('content') or ''
                     for choice in (previous_response or {}).get('choices', []))
    quotes = Counter(re.findall(r'<quote\b[^>]*>.*?</quote>', text, flags=re.DOTALL))
    repeated = [{'element_excerpt': quote[:400], 'occurrences': count}
                for quote, count in quotes.most_common(2) if count > 1]
    feedback = (
        'The previous aggregator XML was incomplete and was discarded. '
        'Regenerate one complete <aggregator_output> for the SAME original query and memories. '
        'The output must contain, in order, one <evidence_quotes>, one <logic_trace>, and '
        'one <answer_core>, followed by </aggregator_output>. '
        'Inside evidence_quotes, emit each distinct timestamp-and-quote pair only once. '
        'Do not cycle through the same supporting phrases or enumerate repeated versions of one fact. '
        'Finish and close evidence_quotes, then complete logic_trace and answer_core; do not restart the evidence list. '
        'Preserve every distinct query-relevant fact, exact terms, and the original conflict-resolution rules. '
        'Do not invent facts. Do not change the query or copy irrelevant input blocks. '
        'Use the requested XML format, and do not discuss this correction in the response.'
    )
    if repeated:
        # A bounded diagnostic excerpt is not a cap or truncation of task input/output.
        feedback += ' Repetition diagnostics from the discarded output (not new evidence): ' + json.dumps(repeated)
    return 'xml-aggregator-v1', feedback


class RepairHandler(ProxyHandler):
    def do_POST(self):
        path, attribution = self._route()
        if not self.server.comparison_policy or path != '/v1/chat/completions':
            return super().do_POST()
        if not self.server.begin_attempt():
            self._body()
            self._json(503, {'error': {'message': 'Usage journal unavailable'}})
            return
        group = str(uuid.uuid4())
        record = None
        pending = False
        try:
            original = json.loads(self._body())
            previous_response = None
            for attempt in range(3):
                started = time.monotonic()
                record = {
                    'schema_version': 1, 'request_id': str(uuid.uuid4()),
                    'timestamp': time.time(), **attribution, 'endpoint': path,
                    'request_kind': 'chat_completion', 'model': original.get('model'),
                    'response_model': None, 'response_id': None, 'http_status': None,
                    'usage': None, 'usage_status': 'missing', 'usage_invalid': False,
                    'finish_reasons': [], 'stream': False, 'stream_complete': None,
                    'client_disconnected': False, 'success': False, 'error_type': None,
                    'repair_group_id': group, 'repair_attempt': attempt,
                    'delivered_to_client': False, 'repair_policy': 'format-feedback-v2',
                }
                pending = True
                payload = comparison_payload(original, record)
                if attempt:
                    kind, feedback = repair_feedback(original, previous_response)
                    record['repair_feedback_kind'] = kind
                    payload['messages'] = list(payload['messages']) + [
                        {'role': 'user', 'content': feedback + f' Correction attempt {attempt}.'}]
                response = self.server.client.post(self.server.upstream + '/chat/completions',
                                                   json=payload, headers=self._request_headers())
                record['http_status'] = response.status_code
                data = response.json()
                ResponseMeter(record).accept(data)
                if response.status_code >= 300:
                    record['error_type'] = 'http_error'
                choices = data.get('choices') or []
                empty = bool(response.status_code == 200 and (not choices or any(
                    not ((c.get('message') or {}).get('content') or '').strip()
                    and not (c.get('message') or {}).get('tool_calls') for c in choices)))
                truncated = 'length' in record['finish_reasons']
                filtered = 'content_filter' in record['finish_reasons']
                incomplete = truncated or empty or filtered
                if incomplete:
                    record['error_type'] = 'comparison_incomplete_output'
                    record['incomplete_kind'] = 'length' if truncated else 'content_filter' if filtered else 'empty'
                    folder = os.getenv('METER_FAILED_PAYLOAD_DIR')
                    if folder:
                        target = Path(folder) / (record['request_id'] + '.json')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(json.dumps({'request_id': record['request_id'],
                            'request': payload, 'response': data}, ensure_ascii=False))
                    if attempt < 2 and not filtered:
                        previous_response = data
                        record['retry_scheduled'] = True
                        self._record_attempt(record, started)
                        pending = False
                        continue
                    record['client_http_status'] = 502
                    self._json(502, {'error': {'type': 'comparison_incomplete_output',
                        'message': 'No complete response after explicit format repair'}})
                else:
                    record['delivered_to_client'] = True
                    self._response_headers(response, len(response.content))
                    self.wfile.write(response.content)
                self._record_attempt(record, started)
                pending = False
                return
        except Exception as exc:
            if record is not None and pending:
                record.update(error_type='repair_proxy_error', error_class=type(exc).__name__)
                self._record_attempt(record, started)
                pending = False
            self.close_connection = True
            try:
                self._json(502, {'error': {'type': 'repair_proxy_error', 'message': type(exc).__name__}})
            except OSError:
                pass
        finally:
            self.server.end_attempt()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream-base-url', required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--journal', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--comparison-policy', action='store_true')
    a = p.parse_args()
    with MeteredProxy((a.host, a.port), a.upstream_base_url, a.journal, a.run_id,
                     a.method, a.timeout, comparison_policy=a.comparison_policy) as server:
        server.RequestHandlerClass = RepairHandler
        print(f'Repairing meter listening on {a.host}:{server.server_port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
