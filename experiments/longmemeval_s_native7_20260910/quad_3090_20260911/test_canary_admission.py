"""Canary validation tests; no real inference, GPU, Supervisor, or provider calls."""
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import canary_admission as app
import quad_config as cfg
from test_quad_queues import Fixture


class CanaryChecks(Fixture):
    def rows(self, canary_id='fixture'):
        expected, rows = {}, []
        for lane, count in app.COUNTS.items():
            for index in range(count):
                sample = f'quad_probe_{lane}_{canary_id}_{index}'
                expected[sample] = lane
                started = 1000 + (index // 2) * 100
                tokens = app.TOKENS[lane]
                rows.append({'run_id': sample, 'sample_id': sample, 'question_id': sample,
                    'method': 'runtime_probe', 'canonical_benchmark': False, 'success': True,
                    'http_status': 200, 'stream': True, 'error_type': None, 'client_disconnected': False,
                    'stream_complete': True, 'usage_status': 'reported', 'finish_reasons': ['length'],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': tokens, 'total_tokens': 10 + tokens},
                    'inference_lane': lane, 'inference_instance_id': cfg.INSTANCE,
                    'inference_gpu_uuid': self.deployment['lanes'][lane]['gpu_uuid'],
                    'deployment_sha256': cfg.DEPLOYMENT_SHA, **cfg.fresh_identity(),
                    'admission_active_at_start': 2, 'admission_sequence': index + 10,
                    'upstream_started_at': started, 'upstream_completed_at': started + 99,
                    'upstream_ttfb_seconds': 1, 'admission_wait_seconds': (index // 2) * 100})
        return expected, rows

    def test_full_twenty_unique_noncanonical_records_pass_exact_usage_fifo_and_cap(self):
        expected, rows = self.rows()
        selected, summary = app.validate_records(rows, expected, self.deployment)
        self.assertEqual(len(selected), 20)
        self.assertEqual(summary['sm0']['peak_upstream'], 2)
        self.assertEqual(summary['sm0']['max_admission_wait_seconds'], 700)
        self.assertEqual(summary['sm1']['requests'], 4)

    def test_missing_duplicate_false_success_bad_usage_done_or_old_fresh_rejected(self):
        expected, original = self.rows()
        mutations = [lambda rows: rows.pop(), lambda rows: rows.append(rows[0]),
            lambda rows: rows[0].update(success=False), lambda rows: rows[0].update(finish_reasons=['stop']), lambda rows: rows[0].update(stream_complete=False),
            lambda rows: rows[0].update(canonical_benchmark=True), lambda rows: rows[0].update(fresh_run_sha256='old'),
            lambda rows: rows[0].update(inference_gpu_uuid='wrong'),
            lambda rows: rows[0]['usage'].update(completion_tokens=2559),
            lambda rows: rows[0]['usage'].update(total_tokens=0)]
        for mutate in mutations:
            rows = copy.deepcopy(original)
            mutate(rows)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                app.validate_records(rows, expected, self.deployment)

    def test_wait_must_exceed_600_and_three_way_overlap_is_rejected(self):
        expected, rows = self.rows()
        for row in rows:
            row['admission_wait_seconds'] = min(row['admission_wait_seconds'], 600)
        with self.assertRaisesRegex(ValueError, 'over 600'):
            app.validate_records(rows, expected, self.deployment)
        expected, rows = self.rows()
        rows[2]['upstream_started_at'] = 1001
        with self.assertRaisesRegex(ValueError, 'two upstream'):
            app.validate_records(rows, expected, self.deployment)

    def test_duplicate_admission_sequence_and_nonfinite_timing_fail(self):
        expected, rows = self.rows()
        rows[1]['admission_sequence'] = rows[0]['admission_sequence']
        with self.assertRaises(ValueError):
            app.validate_records(rows, expected, self.deployment)
        expected, rows = self.rows()
        rows[0]['upstream_ttfb_seconds'] = float('nan')
        with self.assertRaises(ValueError):
            app.validate_records(rows, expected, self.deployment)

    def test_log_checks_only_new_bytes_and_catches_oom_across_boundary(self):
        path = self.directory / 'quad-sm0.log'
        path.write_bytes(b'Old CUDA out of memory\nReady\n')
        start = app.log_start('sm0')
        with path.open('ab') as stream:
            stream.write(b'Healthy request completed\n')
        self.assertEqual(app.log_result(start)['new_oom_entries'], 0)
        with path.open('ab') as stream:
            stream.write(b'CUDA OutOf')
        start = app.log_start('sm0')
        with path.open('ab') as stream:
            stream.write(b'MemoryError\n')
        with self.assertRaisesRegex(ValueError, 'OOM'):
            app.log_result(start)

    def metrics_text(self):
        return ('vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen3.5-9B"} 0\n'
                'vllm:num_requests_running{engine="0",model_name="Qwen/Qwen3.5-9B"} 0\n'
                'vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen3.5-9B"} 0\n'
                + ''.join('vllm:request_success_total{engine="0",finished_reason="' + reason
                          + '",model_name="Qwen/Qwen3.5-9B"} 0\n'
                          for reason in ('stop', 'length', 'abort', 'error', 'repetition')))

    def test_gpu_headroom_and_unavailable_preemption_metric_fail(self):
        with patch.object(app, 'http', return_value=self.metrics_text()), patch.object(app.subprocess, 'run', return_value=SimpleNamespace(stdout='513\n')):
            self.assertEqual(app.observe('sm0')['free_mib'], 513)
        with patch.object(app, 'http', return_value=self.metrics_text()), patch.object(app.subprocess, 'run', return_value=SimpleNamespace(stdout='512\n')):
            with self.assertRaises(ValueError):
                app.observe('sm0')
        with patch.object(app, 'http', return_value='unavailable'):
            with self.assertRaises(ValueError):
                app.observe('sm0')

    def test_actual_metric_schema_and_exact_finished_counter_deltas(self):
        before = app.parse_metrics(self.metrics_text())
        after = copy.deepcopy(before)
        after['finished']['length'] = 16
        self.assertEqual(app.verify_engine_delta(before, after, 16)['length'], 16)
        for reason in ('stop', 'abort', 'error', 'repetition', 'length'):
            changed = copy.deepcopy(after)
            changed['finished'][reason] += 1
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                app.verify_engine_delta(before, changed, 16)
        for key in ('running', 'waiting', 'preemptions'):
            changed = copy.deepcopy(after)
            changed[key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                app.verify_engine_delta(before, changed, 16)
        with self.assertRaises(ValueError):
            app.parse_metrics('vllm:num_preemptions_total 0\n')

    def test_sdk_payload_forces_requested_tokens_and_only_content_hash_is_returned(self):
        stream = MagicMock()
        stream.__enter__.return_value = iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='private synthetic response'))])])
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.completions.create.return_value = stream
        constructor = Mock(return_value=client)
        with patch.dict('sys.modules', {'openai': SimpleNamespace(OpenAI=constructor, __version__='fixture')}):
            result = app.sdk_request('sm0', 'quad_probe_sm0_fixture_0')
        payload = client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload['max_tokens'], 2560)
        self.assertEqual(payload['extra_body']['min_tokens'], 2560)
        self.assertTrue(payload['extra_body']['ignore_eos'])
        self.assertEqual(payload['extra_headers']['X-Meter-Method'], 'runtime_probe')
        self.assertEqual(constructor.call_args.kwargs['max_retries'], 0)
        self.assertEqual(constructor.call_args.kwargs['http_client'].timeout.read, 600)
        constructor.call_args.kwargs['http_client'].close()
        self.assertEqual(result['response_sha256'], hashlib.sha256(b'private synthetic response').hexdigest())
        self.assertNotIn('private synthetic response', json.dumps(result))
        self.assertNotIn('messages', result)

    def test_expired_batch_does_not_start_sdk_threads(self):
        with patch.object(app, 'sdk_request') as request:
            with self.assertRaises(TimeoutError):
                app.run_batch('sm0', ['id'], -1)
        request.assert_not_called()

    def test_execute_writes_verified_receipt_only_after_all_resource_and_journal_checks(self):
        expected, rows = self.rows()
        journal = self.directory / 'request_usage.jsonl'
        journal.write_bytes(b'')
        receipt = self.directory / 'canary_receipt.json'
        state = {'canary_id': 'fixture', 'status': 'running'}
        observations = {**app.parse_metrics(self.metrics_text()), 'free_mib': 1024}
        after0, after1 = copy.deepcopy(observations), copy.deepcopy(observations)
        after0['finished']['length'] = 16
        after1['finished']['length'] = 4

        def batch(lane, samples, deadline):
            self.assertEqual(set(samples), {sample for sample, target in expected.items() if target == lane})
            with journal.open('a') as stream:
                for row in rows:
                    if row['inference_lane'] == lane:
                        stream.write(json.dumps(row) + '\n')
            return [{'response_sha256': 'fixture', 'sample_id': sample} for sample in samples], [observations]

        with patch.object(app, 'identity', return_value={'fresh_run_sha256': 'fixture', 'code_sha256': app.PINS}), patch.object(app, 'log_start', return_value={}), patch.object(app, 'log_result', return_value={'new_oom_entries': 0}), patch.object(app, 'engine_pid', return_value=123), patch.object(app, 'observe', side_effect=[observations, observations, after0, after1]), patch.object(app, 'run_batch', side_effect=batch), patch.object(app, 'http', return_value='{"status":"ok","in_flight":99}'):
            app.execute(state, receipt, 10**20)
        proof = cfg.read(receipt)
        self.assertEqual(proof['status'], 'admission_canary_verified')
        self.assertEqual(len(proof['journal_records']), 20)
        self.assertEqual(proof['summary']['sm0']['minimum_free_mib'], 1024)
        self.assertEqual(proof['code_sha256'], app.PINS)

    def test_expected_terminal_rows_ignore_unrelated_meter_traffic_and_partial_tail(self):
        journal = self.directory / 'partial_journal.jsonl'
        journal.write_bytes(b'foreign-fragment\n{"run_id":"owned"}\n{"run_id":"unrelated"')
        with patch.object(app, 'http', return_value='{"status":"ok","in_flight":99}'):
            raw, rows = app.wait_for_records(journal, 0, {'owned'}, 10**20, skip_fragment=True)
        self.assertEqual(rows, [{'run_id': 'owned'}])
        self.assertIn(b'foreign-fragment', raw)
        with patch.object(app, 'http', return_value='{"status":"ok","in_flight":99}'), self.assertRaises(TimeoutError):
            app.wait_for_records(journal, 0, {'missing'}, -1, skip_fragment=True)
        with patch.object(app, 'http', return_value='{"status":"journal_error","in_flight":0}'), self.assertRaises(ValueError):
            app.wait_for_records(journal, 0, {'owned'}, 10**20, skip_fragment=True)

    def test_main_records_failure_without_claiming_canary_success(self):
        with patch.object(app, 'execute', side_effect=ValueError('Queue wait did not exceed 600 seconds')), patch.object(app.signal, 'signal'), patch('builtins.print'):
            self.assertEqual(app.main(), 1)
        receipts = list(self.directory.glob('canary_admission_*.json'))
        self.assertEqual(len(receipts), 1)
        proof = cfg.read(receipts[0])
        self.assertEqual(proof['status'], 'admission_canary_failed')
        self.assertIn('600', proof['failure_reason'])

    def test_wrong_instance_refuses_identity_before_inference(self):
        with patch.dict('os.environ', {'CONTAINER_ID': 'other'}), self.assertRaises(ValueError):
            app.identity()


if __name__ == '__main__':
    unittest.main()