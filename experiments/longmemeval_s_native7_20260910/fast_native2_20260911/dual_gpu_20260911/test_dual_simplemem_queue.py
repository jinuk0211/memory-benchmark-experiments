"""Behavioral tests for the coordinator; no model or network calls."""
import copy
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dual_simplemem_queue', HERE / 'dual_simplemem_queue.py')
dual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dual)
queue = dual.load_module('simplemem_queue_for_tests', HERE.parent / 'simplemem_queue.py')
native = dual.load_module('simplemem_native_for_tests',
    HERE.parents[1] / 'official_recovery/simplemem_native_dialogues_v4/runner.py')


class Process:
    counter = 10000

    def __init__(self):
        Process.counter += 1
        self.pid, self.returncode, self.waited = Process.counter, None, False

    def poll(self):
        return self.returncode

    def wait(self):
        self.waited = True
        return self.returncode


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'runs/simplemem_native_dialogues_v4').mkdir(parents=True)
        (self.root / 'fast_native2_20260911').mkdir()
        self.receipt = self.root / 'runtime.json'
        self.receipt.write_text('{"status":"runtime_verified"}', encoding='utf-8')
        self.remote = self.root / 'inference_verified.json'
        self.protocol = {'runtime': {'api_base': 'http://127.0.0.1:18083/v1'},
                         'source_files_sha256': {'runner.py': queue.RUNNER_SHA}}
        self.data = [{'question_id': f'q{i:03}', 'question': f'Question {i}', 'question_date': '2026/09/11',
                      'answer': 'Do not leak this label', 'haystack_dates': ['2026/09/10'],
                      'haystack_session_ids': [f's{i}'],
                      'haystack_sessions': [[{'role': 'user', 'content': f'Source {i}', 'label': 'discard'}]]}
                     for i in range(500)]
        self.coordinators = []
        self.c = self.new_coordinator()
        self.popen = patch.object(dual.subprocess, 'Popen', side_effect=self.spawn)
        self.mock_popen = self.popen.start()

    def tearDown(self):
        self.popen.stop()
        for coordinator in self.coordinators:
            for job in coordinator.inflight.values():
                job['output'].close()
        self.temp.cleanup()

    def new_coordinator(self):
        coordinator = dual.Coordinator(queue, native, self.root, self.root / 'runner.py',
                                       self.data, self.protocol, self.receipt, self.remote)
        self.coordinators.append(coordinator)
        return coordinator

    def spawn(self, argv, **kwargs):
        self.assertEqual(argv[2], '--worker-dir')
        self.assertTrue(kwargs['start_new_session'])
        attempt = Path(argv[3])
        qid = native.read_json(attempt / 'query.json')['question_id']
        persisted = native.read_json(self.c.route_path)
        self.assertIn(qid, persisted['histories'])
        self.assertTrue((self.c.dual_dir / 'initialized.json').exists())
        self.assertEqual(native.read_json(attempt / 'dual_route.json'),
                         {'question_id': qid, **persisted['histories'][qid]})
        worker = native.read_json(attempt / 'worker.json')
        self.assertEqual(worker['runtime'], self.protocol['runtime'])
        self.assertEqual(worker['source_files_sha256'], self.protocol['source_files_sha256'])
        self.assertNotIn('answer', native.read_json(attempt / 'query.json'))
        self.assertNotIn('label', native.read_json(attempt / 'source.json')[0]['turns'][0])
        return Process()

    def complete(self, job):
        qid, attempt = job['qid'], job['attempt']
        for name in ('memory.json', 'dialogues.json', 'build_complete.json', 'usage.json', 'native_runtime.json'):
            native.save_json(attempt / name, {})
        for name in ('llm_calls.jsonl', 'location_compat_audit.jsonl', 'json_syntax_compat_audit.jsonl'):
            (attempt / name).write_text('', encoding='utf-8')
        native.save_json(attempt / 'prediction.json', {
            'question_id': qid, 'identity': self.c.identities[qid],
            'status': 'generated', 'hypothesis': f'Answer {qid}'})
        native.seal(attempt, self.c.identities[qid])
        job['process'].returncode = 0

    def test_disjoint_allocation_and_out_of_order_completion(self):
        self.c.launch('local', 'q000')
        self.c.launch('v100', 'q001')
        local, remote = self.c.inflight['local'], self.c.inflight['v100']
        self.assertNotEqual(local['attempt'], remote['attempt'])
        self.complete(remote)
        self.c.reap(time.monotonic())
        self.assertEqual(set(self.c.predictions), {'q001'})
        self.assertEqual(set(self.c.inflight), {'local'})
        self.assertTrue(remote['process'].waited)
        self.assertFalse(self.c.publish())
        status = native.read_json(self.c.state_dir / 'simplemem_status.json')
        self.assertEqual((status['planned'], status['generated'], status['active_question_id']), (500, 1, 'q000'))
        self.complete(local)
        self.c.reap(time.monotonic())
        self.c.publish()
        self.assertEqual([p['question_id'] for p in native.read_json(self.c.run_dir / 'predictions.json')],
                         ['q000', 'q001'])

    def test_duplicate_dispatch_is_rejected_before_attempt_allocation(self):
        self.c.launch('local', 'q000')
        with self.assertRaises(ValueError):
            self.c.launch('v100', 'q000')
        with self.assertRaises(ValueError):
            self.c.launch('local', 'q001')
        self.assertEqual(len(self.c.attempts('q000')), 1)
        self.assertEqual(self.c.attempts('q001'), [])
        self.assertEqual(self.mock_popen.call_count, 1)

    def test_retry_keeps_route_and_process_returncodes_across_restart(self):
        self.c.launch('v100', 'q000')
        original = copy.deepcopy(self.c.routes['histories']['q000'])
        self.c.inflight['v100']['process'].returncode = 17
        self.c.reap(time.monotonic())
        self.c = self.new_coordinator()
        self.assertEqual(self.c.failures['q000']['exit_code'], 17)
        self.assertEqual(self.c.candidate('v100'), 'q000')
        self.assertEqual(self.c.candidate('local'), 'q001')
        with self.assertRaises(ValueError):
            self.c.assign('q000', 'local')
        self.c.launch('v100', 'q000')
        self.assertEqual(self.c.inflight['v100']['attempt'].name, 'attempt_0002')
        self.assertEqual(self.c.routes['histories']['q000'], original)
        self.c.inflight['v100']['process'].returncode = -15
        self.c.reap(time.monotonic())
        self.assertEqual(self.c.failures['q000']['exit_code'], -15)
        self.assertEqual(self.c.failures['q000']['process_returncode'], -15)
        self.assertNotEqual(self.c.candidate('v100'), 'q000')
        with self.assertRaises(ValueError):
            self.c.launch('v100', 'q000')
        self.assertEqual(len(self.c.attempts('q000')), 2)

    def test_verified_completion_reused_and_artifact_tampering_rejected(self):
        self.c.launch('local', 'q000')
        job = self.c.inflight['local']
        self.complete(job)
        self.c.reap(time.monotonic())
        self.c = self.new_coordinator()
        self.assertIn('q000', self.c.predictions)
        self.assertEqual(self.c.candidate('local'), 'q001')
        self.assertEqual(len(self.c.attempts('q000')), 1)
        (job['attempt'] / 'memory.json').write_text('tampered', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.new_coordinator()

    def test_unsealed_legacy_attempt_is_local_and_not_reused(self):
        # Simulate the first migration, before the dual coordinator was activated.
        (self.c.dual_dir / 'initialized.json').unlink()
        self.c.route_path.unlink()
        attempt = native.next_attempt(self.c.histories['q000'])
        native.save_json(attempt / 'prediction.json', {'status': 'generated', 'hypothesis': 'Unsealed'})
        self.c = self.new_coordinator()
        self.assertNotIn('q000', self.c.predictions)
        self.assertEqual(self.c.routes['histories']['q000']['lane'], 'local')
        self.assertEqual(self.c.candidate('v100'), 'q001')
        self.assertEqual(self.c.candidate('local'), 'q000')

    def test_lost_manifest_after_v100_attempt_fails_closed(self):
        self.c.launch('v100', 'q000')
        attempt = self.c.inflight['v100']['attempt']
        self.c.inflight['v100']['process'].returncode = 17
        self.c.reap(time.monotonic())
        self.c.route_path.unlink()
        with self.assertRaisesRegex(ValueError, 'lost its routes manifest'):
            self.new_coordinator()
        self.assertFalse(self.c.route_path.exists())
        self.assertEqual(native.read_json(attempt / 'dual_route.json')['lane'], 'v100')

    def test_lost_marker_and_manifest_detected_by_attempt_receipt(self):
        self.c.launch('v100', 'q000')
        self.c.inflight['v100']['process'].returncode = 17
        self.c.reap(time.monotonic())
        self.c.route_path.unlink()
        (self.c.dual_dir / 'initialized.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Dual attempt route is missing'):
            self.new_coordinator()
        self.assertFalse(self.c.route_path.exists())

    def test_crashed_dual_attempt_without_result_cannot_lose_its_route(self):
        self.c.launch('v100', 'q000')
        attempt = self.c.inflight['v100']['attempt']
        self.assertFalse((attempt / 'coordinator_result.json').exists())
        self.c.routes['histories'].pop('q000')
        self.c.save_routes()
        with self.assertRaisesRegex(ValueError, 'Dual attempt route is missing'):
            self.new_coordinator()
        self.assertEqual(native.read_json(attempt / 'dual_route.json')['lane'], 'v100')

    def test_missing_history_entry_or_changed_lane_rejected_by_attempt_receipt(self):
        self.c.launch('v100', 'q000')
        self.c.inflight['v100']['process'].returncode = 17
        self.c.reap(time.monotonic())
        original = copy.deepcopy(self.c.routes)
        self.c.routes['histories'].pop('q000')
        self.c.save_routes()
        with self.assertRaisesRegex(ValueError, 'Dual attempt route is missing'):
            self.new_coordinator()
        original['histories']['q000'].update(lane='local', instance_id='50468468')
        queue.atomic(self.c.route_path, original)
        with self.assertRaisesRegex(ValueError, 'Dual attempt route is missing or differs'):
            self.new_coordinator()

    def test_route_protocol_or_instance_change_rejected(self):
        self.c.assign('q000', 'v100')
        changed = copy.deepcopy(self.c.routes)
        changed['histories']['q000']['instance_id'] = '50468468'
        queue.atomic(self.c.route_path, changed)
        with self.assertRaisesRegex(ValueError, 'Invalid persisted route'):
            self.new_coordinator()

    def test_remote_gate_and_independent_thirty_minute_limit(self):
        self.c.assign('q000', 'v100')
        with patch.object(dual, 'model_ready', return_value=True) as ready:
            self.c.check_health(1)
            self.assertFalse(self.c.health['v100'])
            ready.assert_called_once_with(dual.LANES['local']['api_base'])
            native.save_json(self.remote, {'status': 'inference_verified', 'instance_id': '50558359'})
            self.c.check_health(31)
            self.assertTrue(self.c.health['v100'])
        with patch.object(dual, 'model_ready', side_effect=lambda endpoint: '18081' in endpoint):
            self.c.check_health(61)
            self.c.check_health(61 + dual.REMOTE_WAIT)
        self.assertTrue(self.c.remote_disabled)
        self.assertTrue(self.c.health['local'])
        self.assertFalse(self.c.health['v100'])
        self.assertEqual(self.c.candidate('local'), 'q001')
        self.assertEqual(self.c.routes['histories']['q000']['lane'], 'v100')

    def test_sigterm_drains_live_work_without_new_dispatch(self):
        self.c.launch('local', 'q000')
        job = self.c.inflight['local']
        self.c.stop(signal.SIGTERM, None)
        with patch.object(dual.time, 'sleep', side_effect=lambda _: self.complete(job)), \
                patch.object(self.c, 'signal_group') as kill:
            self.assertEqual(self.c.execute(), 130)
        kill.assert_not_called()
        self.assertTrue(job['process'].waited)
        self.assertEqual(self.mock_popen.call_count, 1)
        self.assertIn('q000', self.c.predictions)
        self.assertEqual(native.read_json(self.c.state_dir / 'simplemem_status.json')['status'], 'paused')

    def test_timeout_terminates_group_and_reaps_actual_returncode(self):
        self.c.launch('local', 'q000')
        job = self.c.inflight['local']
        deadline = job['started'] + dual.HISTORY_TIMEOUT
        with patch.object(dual.signal, 'SIGKILL', 9, create=True), \
                patch.object(self.c, 'signal_group') as kill:
            self.c.reap(deadline)
            kill.assert_called_once_with(job['process'], signal.SIGTERM)
            self.c.reap(deadline + 10)
            self.assertEqual(kill.call_args.args, (job['process'], signal.SIGKILL))
            job['process'].returncode = -9
            self.c.reap(deadline + 11)
        self.assertTrue(job['process'].waited)
        result = self.c.failures['q000']
        self.assertEqual((result['exit_code'], result['process_returncode']), (124, -9))
        self.assertEqual(result['reason'], 'history_timeout_5400_seconds')

    def test_only_exact500_valid_zero_failure_is_complete(self):
        for qid in self.c.ids:
            self.c.predictions[qid] = {'question_id': qid, 'status': 'generated', 'hypothesis': 'Answer'}
        self.assertTrue(self.c.publish('finished'))
        self.c.failures['q000'] = {'question_id': 'q000', 'exit_code': 1}
        self.assertFalse(self.c.publish('finished'))
        self.c.failures.clear()
        self.c.predictions.pop('q000')
        self.assertFalse(self.c.publish('finished'))
        self.assertEqual(native.read_json(self.c.run_dir / 'status.json')['status'], 'generation_incomplete')

    def test_v100_unavailable_leaves_its_missing_history_pending(self):
        self.c.assign('q000', 'v100')
        self.c.remote_disabled = True
        for qid in self.c.ids[1:]:
            self.c.predictions[qid] = {'question_id': qid, 'status': 'generated', 'hypothesis': 'Answer'}
        with patch.object(dual, 'model_ready', return_value=True):
            self.assertEqual(self.c.execute(), 1)
        self.mock_popen.assert_not_called()
        status = native.read_json(self.c.dual_dir / 'status.json')
        self.assertEqual(status['pending_ids'], ['q000'])
        self.assertEqual((status['generated'], status['failed']), (499, 0))

    def test_launch_failure_consumes_one_attempt_and_preserves_failure(self):
        self.mock_popen.side_effect = OSError('Test executable unavailable')
        self.c.launch('local', 'q000')
        self.assertFalse(self.c.inflight)
        self.assertEqual(self.c.failures['q000']['exit_code'], 127)
        self.assertEqual(len(self.c.attempts('q000')), 1)
        self.assertTrue((self.c.attempts('q000')[0] / 'failure.json').exists())
        self.c.launch('local', 'q000')
        self.assertEqual(len(self.c.attempts('q000')), 2)
        self.assertEqual(self.c.candidate('local'), 'q001')

    def test_metadata_error_during_shutdown_still_reaps_other_worker(self):
        self.c.launch('local', 'q000')
        self.c.launch('v100', 'q001')
        jobs = list(self.c.inflight.values())
        self.c.stop()

        def finish_workers(_):
            for job in jobs:
                self.complete(job)

        with patch.object(self.c, 'publish', side_effect=OSError('Test disk full')), \
                patch.object(dual.time, 'sleep', side_effect=finish_workers), \
                patch('builtins.print', side_effect=OSError('Test broken log stream')):
            with self.assertRaisesRegex(OSError, 'Test disk full'):
                self.c.execute()
        self.assertFalse(self.c.inflight)
        self.assertTrue(all(job['process'].waited for job in jobs))

    def test_model_readiness_uses_only_bounded_models_request(self):
        with patch.object(dual, 'urlopen') as opening:
            opening.return_value.__enter__.return_value.read.return_value = json.dumps(
                {'data': [{'id': 'Qwen/Qwen3.5-9B'}]}).encode()
            self.assertTrue(dual.model_ready('http://127.0.0.1:18091/v1'))
        opening.assert_called_once_with('http://127.0.0.1:18091/v1/models', timeout=10)


if __name__ == '__main__':
    unittest.main()
