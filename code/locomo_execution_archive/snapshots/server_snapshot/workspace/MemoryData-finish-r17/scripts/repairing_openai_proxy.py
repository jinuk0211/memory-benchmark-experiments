"""Meter every attempt; retry incomplete generations without consuming partial text.

Successful first requests retain their original prompts and sampling settings.
Repairs add explicit format feedback. Length-truncated E-Mem aggregation or A-MEM evolution also
uses a disclosed retry-only repetition penalty of 1.1 when none was supplied.
Model, context/output limits, temperature, facts and retrieval are not reduced.
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


def simplemem_reflection_request(original: dict, method: str | None) -> bool:
    """Recognize only the native information-completeness evaluator contract."""
    messages = original.get('messages', [])
    return method == 'simplemem' and any(
        message.get('role') == 'system' and message.get('content') ==
        'You are an information completeness evaluator. You must output valid JSON format.'
        for message in messages) and any(
        message.get('role') == 'user' and isinstance(message.get('content'), str)
        and message['content'].lstrip().startswith(
            'Analyze whether the provided information is sufficient to completely answer the original question, based on the identified information requirements.')
        for message in messages)


def simplemem_reflection_object_error(original: dict, data: dict, method: str | None) -> bool:
    """Reject proven non-objects; leave other text to the native JSON extractor."""
    if not simplemem_reflection_request(original, method):
        return False
    for choice in data.get('choices') or []:
        content = ((choice.get('message') or {}).get('content') or '').strip()
        fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
        if fenced:
            content = fenced.group(1)
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            # Native extraction accepts prefixes, embedded blocks and trailing commas.
            # Do not replace a potentially valid native assessment with new inference.
            continue
        if not isinstance(value, dict):
            return True
    return False


def repair_feedback(original, previous_response, method=None):
    """Use the original task contract for targeted incomplete-output repairs."""
    if simplemem_reflection_request(original, method):
        return 'simplemem-reflection-object-v1', (
            'The previous information completeness evaluation was not a complete JSON object '
            'and was discarded without applying an assessment. Repeat the SAME original '
            'evaluation using ALL original question, information requirements and retrieved evidence. '
            'Return one JSON OBJECT, not an array, with assessment (complete or incomplete), '
            'reasoning, missing_info_types and coverage_percentage as requested originally. '
            'Do not assume completeness, omit evidence, invent facts, or discuss this correction.'
        )
    schema = (((original.get('response_format') or {}).get('json_schema') or {}).get('schema') or {}) if method == 'a_mem' else {}
    evolution_fields = {'should_evolve', 'actions', 'suggested_connections', 'tags_to_update',
                        'new_context_neighborhood', 'new_tags_neighborhood'}
    evolution = method == 'a_mem' and set(schema.get('properties', {})) == evolution_fields and any(
        isinstance(message.get('content'), str) and message['content'].lstrip().startswith(
            'You are an AI memory evolution agent responsible for managing and evolving a knowledge base.')
        for message in original.get('messages', []))
    if evolution and any(choice.get('finish_reason') == 'length'
                         for choice in (previous_response or {}).get('choices', [])):
        return 'amem-evolution-loop-v1', (
            'The previous evolution JSON did not finish and was discarded without applying it. '
            'Perform the SAME original memory evolution task with ALL original facts and neighbors. '
            'Return one complete object with should_evolve, actions, suggested_connections, '
            'tags_to_update, new_context_neighborhood and new_tags_neighborhood. '
            'Finish every tag array; do not cycle indefinitely through the same tags. '
            'Keep all relevant information and the original schema, neighbor order and global indices. '
            'Close both neighborhood arrays and the object, then stop. Do not invent facts or discuss this correction.'
        )
    text = '\n'.join((choice.get('message') or {}).get('content') or ''
                     for choice in (previous_response or {}).get('choices', []))
    retrieval = any(isinstance(message.get('content'), str)
        and message['content'].lstrip().startswith('# ROLE: Memory Retrieval & Analysis Agent')
        for message in original.get('messages', []))
    if retrieval and text.lstrip().startswith('<response_type>retrieval</response_type>'):
        return 'xml-retrieval-v1', (
            'The previous retrieval XML repeated memory segments and did not finish; it was discarded. '
            'Repeat the SAME original retrieval task on the SAME original memory and question. '
            'Start with <response_type>retrieval</response_type>, then one <relevant_memories> block. '
            'Copy every distinct query-relevant source segment VERBATIM, including its original timestamp, '
            'inside <memory_segment> tags. Emit each exact source segment only once; do not cycle or '
            'produce repeated variants. Do not paraphrase, merge, alter or omit distinct relevant facts. '
            'After the finite evidence list, close </relevant_memories> and provide one '
            '<model_reasoning> containing the grounded preliminary answer and brief reasoning, '
            'then close </model_reasoning> and stop. If evidence is absent, explicitly say so. '
            'Do not invent evidence. Do not discuss this correction in the output.'
        )
    aggregator = any(isinstance(message.get('content'), str)
        and message['content'].startswith('# ROLE: Memory Fact Aggregator & Logic Solver')
        and '<aggregator_output>' in message['content']
        for message in original.get('messages', []))
    if not aggregator:
        return 'generic-format-v1', REPAIR_FEEDBACK
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


def amem_retry_messages(messages: list[dict], feedback: str) -> list[dict]:
    """Retain native system text without adding a second system role to Qwen."""
    if messages and messages[0].get('role') == 'system' and isinstance(messages[0].get('content'), str):
        first = {**messages[0], 'content': messages[0]['content'] + '\n\n' + feedback}
        return [first, *messages[1:]]
    return [{'role': 'system', 'content': feedback}, *messages]


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
                    kind, feedback = repair_feedback(original, previous_response, record['method'])
                    record['repair_feedback_kind'] = kind
                    if kind in ('xml-aggregator-v1', 'amem-evolution-loop-v1') and any(
                            choice.get('finish_reason') == 'length'
                            for choice in (previous_response or {}).get('choices', [])):
                        # Retry-only decoding variant; the first native request stays intact.
                        if kind == 'xml-aggregator-v1':
                            _, feedback = repair_feedback(original, {})
                        if kind == 'amem-evolution-loop-v1':
                            payload['messages'] = amem_retry_messages(payload['messages'], feedback)
                        else:
                            payload['messages'] = [{'role': 'system', 'content': feedback}] + list(payload['messages'])
                        if 'repetition_penalty' not in payload:
                            payload['repetition_penalty'] = 1.1
                            record['repair_sampling_change'] = {'repetition_penalty': 1.1}
                        record['comparison_policy']['effective']['repetition_penalty'] = payload['repetition_penalty']
                        if kind == 'xml-aggregator-v1':
                            record['repair_feedback_kind'] = 'xml-aggregator-loop-v2'
                    else:
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
                reflection_object_error = response.status_code == 200 and simplemem_reflection_object_error(
                    original, data, record['method'])
                incomplete = truncated or empty or filtered or reflection_object_error
                if incomplete:
                    record['error_type'] = 'comparison_incomplete_output'
                    record['incomplete_kind'] = ('length' if truncated else 'content_filter' if filtered
                                                else 'empty' if empty else 'simplemem_reflection_object')
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
