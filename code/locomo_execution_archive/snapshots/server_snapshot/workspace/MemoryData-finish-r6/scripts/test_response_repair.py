"""Verify retries never deliver partial text or erase failed-attempt accounting."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
import sys
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.metered_openai_proxy import MeteredProxy
from scripts.repairing_openai_proxy import RepairHandler, repair_feedback
from scripts.response_delivery_audit import audit_delivery


class RepairTests(unittest.TestCase):
    def invoke(self, reasons, prompt='Original task'):
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
                                  'test', 'a_mem', comparison_policy=True)
            server.RequestHandlerClass = RepairHandler
            server.client.close()
            server.client = httpx.Client(transport=httpx.MockTransport(upstream))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                response = httpx.post(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                    json={'model': 'Qwen/Qwen3.5-9B', 'messages': [{'role': 'user', 'content': prompt}]},
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
        self.assertEqual(requests[1]['messages'][0], requests[0]['messages'][0])
        feedback = requests[1]['messages'][-1]['content']
        self.assertIn('<logic_trace>', feedback)
        self.assertIn('<answer_core>', feedback)
        self.assertIn('timestamp-and-quote pair only once', feedback)
        self.assertNotIn('JSON', feedback)
        self.assertNotIn(partial, str(requests[1]))
        self.assertEqual(rows[1]['repair_feedback_kind'], 'xml-aggregator-v1')
        self.assertEqual(response.json()['choices'][0]['message']['content'], complete)
        self.assertTrue(audit_delivery(rows)['complete'])
        self.assertEqual(audit_delivery(rows)['discarded_attempt_total_tokens'], 15)
        for key in ('model', 'temperature', 'enable_thinking', 'max_tokens', 'max_completion_tokens'):
            self.assertEqual(requests[0].get(key), requests[1].get(key))

    def test_other_xml_text_does_not_trigger_aggregator_repair(self):
        kind, feedback = repair_feedback({'messages': [{'content': 'Some text <aggregator_output>'}]}, {})
        self.assertEqual(kind, 'generic-format-v1')


if __name__ == '__main__':
    unittest.main()
