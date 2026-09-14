"""Verify retries never deliver partial text or erase failed-attempt accounting."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
import sys
from unittest.mock import MagicMock, patch
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.metered_openai_proxy import MeteredProxy
from scripts.repairing_openai_proxy import RepairHandler, main, repair_feedback
from scripts.response_delivery_audit import audit_delivery


class RepairTests(unittest.TestCase):
    def invoke(self, reasons, prompt='Original task', *, extra=None, method='a_mem'):
        with tempfile.TemporaryDirectory() as directory:
            requests = []
            def upstream(request):
                requests.append(json.loads(request.content))
                reason, content = reasons[min(len(requests)-1, len(reasons)-1)]
                return httpx.Response(200, json={'id': f'resp-{len(requests)}',
                    'model': 'Qwen/Qwen3.5-9B', 'choices': [{'finish_reason': reason,
                    'message': {'content': content}}], 'usage': {'prompt_tokens': 10,
                    'completion_tokens': 5, 'total_tokens': 15}})
            journal = Path(directory) / 'usage.jsonl'
            server = MeteredProxy(('127.0.0.1', 0), 'http://upstream/v1', journal,
                                  'test', method, comparison_policy=True)
            server.RequestHandlerClass = RepairHandler
            server.client.close()
            server.client = httpx.Client(transport=httpx.MockTransport(upstream))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                response = httpx.post(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                    json={'model': 'Qwen/Qwen3.5-9B', 'messages': [{'role': 'user', 'content': prompt}], **(extra or {})},
                    trust_env=False)
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            return response, requests, rows

    def test_success_keeps_original_prompt(self):
        response, requests, rows = self.invoke([('stop', 'complete')])
        self.assertEqual(requests[0]['messages'], [{'role': 'user', 'content': 'Original task'}])
        self.assertEqual(response.json()['choices'][0]['message']['content'], 'complete')
        self.assertTrue(audit_delivery(rows)['complete'])

    def test_partial_is_replaced_and_metered(self):
        response, requests, rows = self.invoke([('length', 'partial'), ('stop', 'complete')])
        self.assertEqual(response.json()['choices'][0]['message']['content'], 'complete')
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]['delivered_to_client'])
        self.assertFalse(rows[0]['success'])
        self.assertNotIn('partial', str(requests[1]))
        self.assertEqual(audit_delivery(rows)['discarded_attempt_total_tokens'], 15)
        self.assertTrue(audit_delivery(rows)['complete'])

    def test_persistent_truncation_stays_failed(self):
        response, requests, rows = self.invoke([('length', 'partial')])
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(rows), 3)
        self.assertEqual(response.headers['x-should-retry'], 'false')
        self.assertFalse(audit_delivery(rows)['complete'])

    def test_filter_is_not_retried(self):
        response, requests, rows = self.invoke([('content_filter', '')])
        self.assertEqual(len(requests), 1)
        self.assertEqual(response.status_code, 502)
        self.assertFalse(audit_delivery(rows)['complete'])

    def test_empty_can_be_repaired(self):
        response, requests, rows = self.invoke([('stop', ''), ('stop', 'complete')])
        self.assertEqual(len(requests), 2)
        self.assertTrue(audit_delivery(rows)['complete'])

    def test_missing_finish_reason_cannot_pass_audit(self):
        response, requests, rows = self.invoke([(None, 'complete')])
        self.assertFalse(audit_delivery(rows)['complete'])

    def test_repair_cannot_borrow_another_questions_success(self):
        response, requests, rows = self.invoke([('length', 'partial'), ('stop', 'complete')])
        rows[0]['question_id'] = 'different-question'
        self.assertFalse(audit_delivery(rows)['complete'])

    def test_missing_attempt_cannot_pass_audit(self):
        response, requests, rows = self.invoke([('length', 'partial'), ('stop', 'complete')])
        self.assertFalse(audit_delivery(rows[1:])['complete'])

    def test_xml_aggregator_success_keeps_native_prompt(self):
        prompt = '# ROLE: Memory Fact Aggregator & Logic Solver\n<aggregator_output>\nAll original memories'
        response, requests, rows = self.invoke([('stop', '<aggregator_output>complete</aggregator_output>')], prompt)
        self.assertEqual(requests[0]['messages'], [{'role': 'user', 'content': prompt}])
        self.assertEqual(len(requests), 1)
        self.assertTrue(audit_delivery(rows)['complete'])

    def test_xml_loop_repair_closes_native_fields_and_keeps_input(self):
        prompt = '# ROLE: Memory Fact Aggregator & Logic Solver\n<aggregator_output>\nAll original memories'
        quote = '<quote timestamp="2023-10-17">bond with nature</quote>'
        partial = '<aggregator_output><evidence_quotes>' + quote * 100
        complete = '<aggregator_output><evidence_quotes>' + quote + '</evidence_quotes><logic_trace>Grounded synthesis</logic_trace><answer_core>A complete factual sentence.</answer_core></aggregator_output>'
        response, requests, rows = self.invoke([('length', partial), ('stop', complete)], prompt)
        self.assertEqual(requests[0]['messages'], [{'role': 'user', 'content': prompt}])
        self.assertEqual(requests[1]['messages'][1:], requests[0]['messages'])
        self.assertEqual(requests[1]['messages'][0]['role'], 'system')
        feedback = requests[1]['messages'][0]['content']
        self.assertIn('<logic_trace>', feedback)
        self.assertIn('<answer_core>', feedback)
        self.assertIn('timestamp-and-quote pair only once', feedback)
        self.assertNotIn('Repetition diagnostics', feedback)
        self.assertNotIn('repetition_penalty', requests[0])
        self.assertEqual(requests[1]['repetition_penalty'], 1.1)
        self.assertEqual(rows[1]['comparison_policy']['effective']['repetition_penalty'], 1.1)
        self.assertNotIn('JSON', feedback)
        self.assertNotIn(partial, str(requests[1]))
        self.assertEqual(rows[1]['repair_feedback_kind'], 'xml-aggregator-loop-v2')
        self.assertEqual(response.json()['choices'][0]['message']['content'], complete)
        self.assertTrue(audit_delivery(rows)['complete'])
        self.assertEqual(audit_delivery(rows)['discarded_attempt_total_tokens'], 15)
        for key in ('model', 'temperature', 'enable_thinking', 'max_tokens', 'max_completion_tokens'):
            self.assertEqual(requests[0].get(key), requests[1].get(key))

    def test_other_xml_text_does_not_trigger_aggregator_repair(self):
        kind, feedback = repair_feedback({'messages': [{'content': 'Some text <aggregator_output>'}]}, {})
        self.assertEqual(kind, 'generic-format-v1')

    def test_native_retrieval_loop_preserves_verbatim_extraction_contract(self):
        prompt = '\n# ROLE: Memory Retrieval & Analysis Agent\nOriginal memory and question stay intact.'
        segment = '<memory_segment>[2023-06-27] Original quoted source.</memory_segment>'
        partial = '<response_type>retrieval</response_type><relevant_memories>' + segment * 100
        complete = '<response_type>retrieval</response_type><relevant_memories>' + segment + '</relevant_memories><model_reasoning>Grounded answer.</model_reasoning>'
        response, requests, rows = self.invoke([('length', partial), ('stop', complete)], prompt)
        self.assertEqual(rows[1]['repair_feedback_kind'], 'xml-retrieval-v1')
        self.assertEqual(requests[0]['messages'], requests[1]['messages'][:-1])
        feedback = requests[1]['messages'][-1]['content']
        self.assertIn('VERBATIM', feedback)
        self.assertIn('</relevant_memories>', feedback)
        self.assertIn('<model_reasoning>', feedback)
        self.assertNotIn('JSON', feedback)
        self.assertNotIn(partial, str(requests[1]))
        self.assertEqual(response.json()['choices'][0]['message']['content'], complete)
        self.assertTrue(audit_delivery(rows)['complete'])
        self.assertEqual(audit_delivery(rows)['discarded_attempt_total_tokens'], 15)

    def test_native_summary_mode_does_not_get_retrieval_correction(self):
        original = {'messages': [{'role': 'system', 'content': '\n# ROLE: Memory Retrieval & Analysis Agent'}]}
        previous = {'choices': [{'message': {'content': '<response_type>summary</response_type><summary_content>partial'}}]}
        kind, feedback = repair_feedback(original, previous)
        self.assertEqual(kind, 'generic-format-v1')

    def test_cli_sets_repair_handler_on_dedicated_meter(self):
        instance = MagicMock()
        instance.server_port = 18084
        arguments = ['repair', '--upstream-base-url', 'http://127.0.0.1:18080/v1',
                     '--port', '18084', '--journal', 'diagnostic.jsonl', '--run-id', 'probe',
                     '--method', 'e_mem', '--timeout', '1200', '--comparison-policy']
        with patch('sys.argv', arguments), patch('scripts.repairing_openai_proxy.MeteredProxy') as server:
            server.return_value.__enter__.return_value = instance
            main()
            self.assertIs(instance.RequestHandlerClass, RepairHandler)
            instance.serve_forever.assert_called_once_with()
            server.assert_called_once_with(('127.0.0.1', 18084), 'http://127.0.0.1:18080/v1',
                                           'diagnostic.jsonl', 'probe', 'e_mem', 1200,
                                           comparison_policy=True)

    def evolution_request(self):
        fields = ('should_evolve', 'actions', 'suggested_connections', 'tags_to_update',
                  'new_context_neighborhood', 'new_tags_neighborhood')
        return {'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'response', 'strict': True, 'schema': {
                'type': 'object', 'properties': {field: {} for field in fields},
                'required': list(fields), 'additionalProperties': False}}}}

    def test_amem_evolution_length_retry_preserves_schema_facts_and_accounting(self):
        prompt = '\n You are an AI memory evolution agent responsible for managing and evolving a knowledge base.\nAll original facts and five neighbors.'
        extra = self.evolution_request()
        complete = '{"tags_to_update": ["native tag", "native tag"]}'
        response, requests, rows = self.invoke([('length', 'loop ' * 100), ('stop', complete)], prompt, extra=extra)
        self.assertNotIn('repetition_penalty', requests[0])
        self.assertEqual(requests[1].get('repetition_penalty'), 1.1)
        self.assertEqual(requests[1]['messages'][1:], requests[0]['messages'])
        self.assertEqual(requests[1]['messages'][0]['role'], 'system')
        self.assertIn('new_tags_neighborhood', requests[1]['messages'][0]['content'])
        self.assertNotIn('loop loop', str(requests[1]))
        for key in ('model', 'response_format', 'temperature', 'max_tokens', 'max_completion_tokens', 'chat_template_kwargs'):
            self.assertEqual(requests[0].get(key), requests[1].get(key))
        self.assertEqual(response.json()['choices'][0]['message']['content'], complete)
        self.assertEqual(rows[1]['repair_feedback_kind'], 'amem-evolution-loop-v1')
        self.assertEqual(rows[1]['repair_sampling_change'], {'repetition_penalty': 1.1})
        self.assertEqual(rows[1]['comparison_policy']['effective']['repetition_penalty'], 1.1)
        self.assertTrue(audit_delivery(rows)['complete'])
        self.assertEqual(audit_delivery(rows)['discarded_attempt_total_tokens'], 15)

    def test_amem_evolution_retry_preserves_explicit_penalty(self):
        prompt = 'You are an AI memory evolution agent responsible for managing and evolving a knowledge base.'
        extra = {**self.evolution_request(), 'repetition_penalty': 1.05}
        _, requests, rows = self.invoke([('length', 'partial'), ('stop', 'complete')], prompt, extra=extra)
        self.assertEqual(requests[1]['repetition_penalty'], 1.05)
        self.assertNotIn('repair_sampling_change', rows[1])
        self.assertEqual(rows[1]['repair_feedback_kind'], 'amem-evolution-loop-v1')

    def test_amem_evolution_success_keeps_duplicate_tags_unchanged(self):
        prompt = 'You are an AI memory evolution agent responsible for managing and evolving a knowledge base.'
        complete = '{"tags_to_update": ["tag", "tag"]}'
        response, requests, _ = self.invoke([('stop', complete)], prompt, extra=self.evolution_request())
        self.assertEqual(len(requests), 1)
        self.assertNotIn('repetition_penalty', requests[0])
        self.assertEqual(response.json()['choices'][0]['message']['content'], complete)

    def test_amem_loop_retry_does_not_match_other_tasks_or_non_length_failure(self):
        prompt = 'You are an AI memory evolution agent responsible for managing and evolving a knowledge base.'
        cases = [('e_mem', prompt, self.evolution_request(), 'length'),
                 ('a_mem', 'An unrelated task', self.evolution_request(), 'length'),
                 ('a_mem', prompt, {}, 'length'),
                 ('a_mem', prompt, self.evolution_request(), 'stop')]
        for method, text, extra, finish in cases:
            with self.subTest(method=method, prompt=text, finish=finish, schema=bool(extra)):
                _, requests, rows = self.invoke([(finish, ''), ('stop', 'complete')], text, extra=extra, method=method)
                self.assertNotIn('repetition_penalty', requests[1])
                self.assertEqual(rows[1]['repair_feedback_kind'], 'generic-format-v1')

    def test_amem_persistent_length_never_delivers_partial(self):
        prompt = 'You are an AI memory evolution agent responsible for managing and evolving a knowledge base.'
        response, requests, rows = self.invoke([('length', 'partial')], prompt, extra=self.evolution_request())
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(requests), 3)
        self.assertFalse(any(row['delivered_to_client'] for row in rows))
        self.assertEqual(requests[2].get('repetition_penalty'), 1.1)
        self.assertFalse(audit_delivery(rows)['complete'])

    def test_other_methods_ignore_null_json_schema_during_xml_repair(self):
        prompt = '# ROLE: Memory Fact Aggregator & Logic Solver\n<aggregator_output>'
        original = {'messages': [{'content': prompt}],
                    'response_format': {'type': 'json_object', 'json_schema': None}}
        kind, _ = repair_feedback(original, {'choices': [{'finish_reason': 'length'}]}, 'e_mem')
        self.assertEqual(kind, 'xml-aggregator-v1')

    def test_amem_native_system_message_is_preserved_without_second_system_role(self):
        messages = [{'role': 'system', 'content': 'You must respond with a JSON object.'},
                    {'role': 'user', 'content': 'You are an AI memory evolution agent responsible for managing and evolving a knowledge base.\nOriginal input.'}]
        extra = {**self.evolution_request(), 'messages': messages}
        _, requests, rows = self.invoke([('length', 'partial'), ('stop', 'complete')], extra=extra)
        self.assertEqual(requests[0]['messages'], messages)
        self.assertEqual(len(requests[1]['messages']), len(messages))
        self.assertEqual(requests[1]['messages'][1:], messages[1:])
        self.assertTrue(requests[1]['messages'][0]['content'].startswith(messages[0]['content'] + '\n\n'))
        self.assertEqual(sum(m['role'] == 'system' for m in requests[1]['messages']), 1)
        self.assertEqual(rows[1]['repair_feedback_kind'], 'amem-evolution-loop-v1')


if __name__ == '__main__':
    unittest.main()
