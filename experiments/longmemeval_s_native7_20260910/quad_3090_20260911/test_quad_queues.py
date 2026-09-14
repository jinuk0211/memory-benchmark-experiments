"""Local mocked checks: no GPU, subprocess inference, or network is used."""
import io
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

import quad_config as config
from quad_scheduler import Scheduler


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(config, 'STATE', self.directory).start()
        patch.object(signal, 'SIGKILL', 9, create=True).start()
        patch('quad_scheduler.model_ready', return_value=True).start()
        patch.dict(os.environ, {'CONTAINER_ID': config.INSTANCE}).start()
        self.deployment = config.deployment()
        config.atomic(self.directory / 'FRESH_RUN.json', {
            'status': 'fresh_run_authorized', 'instance_id': config.INSTANCE,
            'dataset_sha256': config.DATASET_SHA, 'methods': {'simplemem': 500, 'lightmem': 500},
            'past_results_reused': False, 'fresh_generation': True, 'canonical_population': 500,
            'run_id': 'fresh_test'})
        config.atomic(self.directory / 'fresh_initialized.json', {
            'status': 'fresh_initialized', 'instance_id': config.INSTANCE,
            'fresh_run_sha256': config.sha(self.directory / 'FRESH_RUN.json')})
        self.gate = {'status': 'runtime_verified', 'instance_id': config.INSTANCE,
            'deployment_sha256': config.DEPLOYMENT_SHA, 'model_proof': {'verified': True},
            'protocol_proof': self.deployment['protocols'],
            'fresh_run_sha256': config.sha(self.directory / 'FRESH_RUN.json'),
            'replicas': {lane: {**binding, 'status': 'inference_verified', 'model': config.MODEL,
                'max_model_len': 65536} for lane, binding in self.deployment['lanes'].items()}}
        config.atomic(self.directory / 'READY.json', self.gate)

    def routes(self, ids=('a', 'b'), method='simplemem', evidence=()):
        return config.Routes(method, ids, evidence)


class GateAndRoutes(Fixture):
    def test_gate_accepts_fresh_run_without_original_receipt(self):
        self.assertEqual(config.require_ready(), self.gate)
        self.assertFalse((self.directory / 'original_recovery_verified.json').exists())

    def test_gate_rejects_wrong_gpu_context_and_fresh_hash(self):
        for field, value in [('max_model_len', 32768), ('gpu_uuid', 'wrong')]:
            with self.subTest(field=field):
                gate = config.read(self.directory / 'READY.json')
                gate['replicas']['lm1'][field] = value
                config.atomic(self.directory / 'READY.json', gate)
                with self.assertRaises(ValueError):
                    config.require_ready()
                config.atomic(self.directory / 'READY.json', self.gate)
        self.gate['fresh_run_sha256'] = 'old'
        config.atomic(self.directory / 'READY.json', self.gate)
        with self.assertRaises(ValueError):
            config.require_ready()

    def test_old_results_authorization_and_old_attempt_rejected(self):
        with self.assertRaises(ValueError):
            config.require_dispatch({'deployment_sha256': config.DEPLOYMENT_SHA}, 'simplemem', 'a')
        marker = config.read(self.directory / 'FRESH_RUN.json')
        marker['past_results_reused'] = True
        config.atomic(self.directory / 'FRESH_RUN.json', marker)
        with self.assertRaises(ValueError):
            config.require_ready()

    def test_routes_are_disjoint_and_retries_cannot_move_or_exceed_two(self):
        routes = self.routes()
        routes.begin('a', 'sm0', 'first', 1)
        routes.begin('b', 'sm1', 'other', 1)
        with self.assertRaises(ValueError):
            routes.assign('a', 'sm1')
        with self.assertRaises(ValueError):
            routes.begin('a', 'sm0', 'duplicate', 1)
        routes.begin('a', 'sm0', 'second', 2)
        with self.assertRaises(ValueError):
            routes.begin('a', 'sm0', 'third', 3)
        with self.assertRaises(ValueError):
            routes.assign('unknown', 'sm0')
        with self.assertRaises(ValueError):
            routes.assign('a', 'lm0')
        self.assertEqual(self.routes().lane('b'), 'sm1')

    def test_missing_manifest_and_individual_route_fail_closed(self):
        routes = self.routes()
        receipt = routes.begin('a', 'sm1', 'first', 1)
        proof = self.directory / 'quad_route.json'
        config.atomic(proof, receipt)
        self.assertEqual(self.routes(evidence=[('a', proof)]).lane('a'), 'sm1')
        original = config.read(routes.path)
        original['histories'].pop('a')
        config.atomic(routes.path, original)
        with self.assertRaises(ValueError):
            self.routes(evidence=[('a', proof)])
        routes.path.unlink()
        with self.assertRaises(ValueError):
            self.routes(evidence=[('a', proof)])
        (self.directory / 'routes_simplemem.initialized.json').unlink()
        with self.assertRaises(ValueError):
            self.routes(evidence=[('a', proof)])

    def test_fresh_marker_change_cannot_relabel_existing_routes(self):
        self.routes().begin('a', 'sm0', 'first', 1)
        marker = config.read(self.directory / 'FRESH_RUN.json')
        marker['run_id'] = 'different_run'
        config.atomic(self.directory / 'FRESH_RUN.json', marker)
        initialized = config.read(self.directory / 'fresh_initialized.json')
        initialized['fresh_run_sha256'] = config.sha(self.directory / 'FRESH_RUN.json')
        config.atomic(self.directory / 'fresh_initialized.json', initialized)
        with self.assertRaises(ValueError):
            self.routes()

    def test_child_gpu_is_bound_and_credentials_are_removed(self):
        with patch.dict(os.environ, {'CONTAINER_API_KEY': 'secret', 'HF_TOKEN': 'secret', 'UNRELATED_PASSWORD': 'secret'}):
            env = config.child_environment(self.deployment['lanes']['lm1'], 'lightmem')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], self.deployment['lanes']['lm1']['gpu_uuid'])
        self.assertNotIn('CONTAINER_API_KEY', env)
        self.assertNotIn('HF_TOKEN', env)
        self.assertNotIn('UNRELATED_PASSWORD', env)
        self.assertEqual(config.child_environment(self.deployment['lanes']['sm0'], 'simplemem')['CUDA_VISIBLE_DEVICES'], '')


class Adapter:
    method = 'simplemem'

    def __init__(self, directory, ids=('a', 'b', 'c')):
        self.directory, self.ids = directory, list(ids)
        self.completed, self.failures, self.counts, self.finished = {}, {}, {}, []

    def attempts(self, qid):
        return self.counts.get(qid, 0)

    def prepare(self, qid, lane, route, routes):
        ordinal = self.attempts(qid) + 1
        routes.begin(qid, lane, qid + str(ordinal), ordinal)
        self.counts[qid] = ordinal
        return {'ordinal': ordinal, 'log': self.directory / (qid + '.log'), 'argv': ['unchanged_worker', qid]}

    def finish(self, job, code, process_code):
        self.finished.append((job['qid'], code, process_code))
        if code == 0:
            self.completed[job['qid']] = True
        else:
            self.failures[job['qid']] = code

    def publish(self, active, phase):
        pass


class SchedulerChecks(Fixture):
    def setUp(self):
        super().setUp()
        self.adapter = Adapter(self.directory)
        self.scheduler = Scheduler(self.adapter, self.routes(self.adapter.ids), 77)

    def job(self, qid, process, ordinal=1, started=0):
        return {'qid': qid, 'process': process, 'ordinal': ordinal, 'started': started,
                'timeout_at': None, 'output': io.StringIO()}

    def test_launches_disjoint_ids_and_rejects_duplicate_allocation(self):
        process = Mock(pid=1)
        with patch('quad_scheduler.subprocess.Popen', return_value=process) as popen:
            self.scheduler.launch('sm0', 'a')
            self.assertEqual(self.scheduler.candidate('sm1'), 'b')
            self.scheduler.launch('sm1', 'b')
            with self.assertRaises(ValueError):
                self.scheduler.launch('sm1', 'a')
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(popen.call_args.kwargs['pass_fds'], (77,))
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        for job in self.scheduler.active.values():
            job['output'].close()

    def test_out_of_order_completion_and_resume_skip_valid_complete(self):
        first, second = Mock(), Mock()
        first.poll.return_value, second.poll.return_value = None, 0
        self.scheduler.active = {'sm0': self.job('a', first), 'sm1': self.job('b', second)}
        self.scheduler.reap(1)
        self.assertEqual(self.adapter.finished, [('b', 0, 0)])
        self.assertEqual(self.scheduler.candidate('sm1'), 'c')
        first.poll.return_value = 0
        self.scheduler.reap(2)
        resumed = Scheduler(self.adapter, self.scheduler.routes, 77)
        self.assertEqual(resumed.candidate('sm0'), 'c')

    def test_retry_waits_for_first_pass_and_stays_on_original_lane(self):
        self.scheduler.routes.begin('a', 'sm1', 'first', 1)
        self.adapter.counts['a'] = 1
        self.assertEqual(self.scheduler.candidate('sm1'), 'b')
        self.adapter.completed.update(b=True, c=True)
        self.assertIsNone(self.scheduler.candidate('sm0'))
        self.assertEqual(self.scheduler.candidate('sm1'), 'a')
        self.adapter.counts['a'] = 2
        self.assertIsNone(self.scheduler.candidate('sm1'))

    def test_timeout_terminates_group_then_reaps_actual_returncode(self):
        process = Mock(pid=12)
        process.poll.return_value = None
        self.scheduler.active['sm0'] = self.job('a', process)
        with patch.object(self.scheduler, 'signal_group') as send:
            self.scheduler.reap(5400)
            send.assert_called_with(process, signal.SIGTERM)
            self.scheduler.reap(5410)
            send.assert_called_with(process, signal.SIGKILL)
            process.poll.return_value = -9
            self.scheduler.reap(5411)
        self.assertEqual(self.adapter.finished, [('a', 124, -9)])
        process.wait.assert_called_once()
        self.assertFalse(self.scheduler.active)

    def test_metadata_failure_still_reaps_both_children(self):
        first, second = Mock(), Mock()
        first.poll.return_value = second.poll.return_value = 0
        self.scheduler.active = {'sm0': self.job('a', first), 'sm1': self.job('b', second)}
        self.adapter.finish = Mock(side_effect=OSError('disk full'))
        with self.assertRaises(OSError):
            self.scheduler.reap(1)
        self.assertEqual(self.adapter.finish.call_count, 2)
        first.wait.assert_called_once()
        second.wait.assert_called_once()
        self.assertFalse(self.scheduler.active)

    def test_publish_failure_drains_live_children_without_killing_them(self):
        first, second = Mock(pid=1), Mock(pid=2)
        first.poll.side_effect = second.poll.side_effect = [None, 0]
        self.scheduler.active = {'sm0': self.job('a', first, started=10**20),
                                 'sm1': self.job('b', second, started=10**20)}
        self.adapter.publish = Mock(side_effect=OSError('disk full'))
        with patch('quad_scheduler.model_ready', return_value=True), patch('quad_scheduler.shutil.disk_usage', return_value=Mock(free=10**12)), patch.object(self.scheduler, 'signal_group') as send:
            with self.assertRaises(OSError):
                self.scheduler.execute()
        self.assertFalse(self.scheduler.active)
        self.assertEqual(len(self.adapter.finished), 2)
        send.assert_not_called()

    def test_term_stops_dispatch_but_drains_current_work(self):
        process = Mock(pid=1)
        process.poll.side_effect = [None, 0]
        self.scheduler.active = {'sm0': self.job('a', process, started=10**20)}
        self.scheduler.stop()
        with patch.object(self.scheduler, 'launch') as launch:
            self.assertFalse(self.scheduler.execute())
        launch.assert_not_called()
        self.assertEqual(self.adapter.finished, [('a', 0, 0)])


class NativeReplayChecks(Fixture):
    def simple_adapter(self):
        from quad_simplemem_queue import SimpleMem
        adapter = SimpleMem.__new__(SimpleMem)
        adapter.ids = ['a']
        adapter.histories = {'a': self.directory / 'histories/a'}
        adapter.routes = self.routes(ids=['a'])
        adapter.completed, adapter.failures = {}, {}
        adapter.rows = {'a': {'question_id': 'a', 'question': 'synthetic', 'question_date': '2026/09/11'}}
        adapter.identities = {'a': {'protocol_sha256': 'synthetic'}}
        adapter.protocol = {'runtime': {}, 'source_files_sha256': {}}
        adapter.runner = self.directory / 'unchanged_runner.py'
        adapter.native = Mock()
        adapter.native.source_only.side_effect = lambda row: {'source': 'synthetic'}
        adapter.native.save_json.side_effect = config.atomic

        def next_attempt(history):
            path = history / f'attempt_{len(list(history.glob("attempt_*"))) + 1:04d}'
            path.mkdir(parents=True)
            return path
        adapter.native.next_attempt.side_effect = next_attempt
        return adapter

    def test_simplemem_manifest_before_mkdir_survives_crash_and_preserves_budget(self):
        adapter = self.simple_adapter()
        expected = adapter.histories['a'] / 'attempt_0001'
        adapter.routes.begin('a', 'sm0', str(expected), 1)
        adapter.reconcile('a')
        self.assertEqual(adapter.attempts('a'), 1)
        self.assertEqual(config.read(expected / 'quad_route.json'), adapter.routes.receipts('a')[0])
        job = adapter.prepare('a', 'sm0', None, adapter.routes)
        self.assertEqual(job['ordinal'], 2)
        self.assertEqual(job['attempt'].name, 'attempt_0002')
        with self.assertRaises(ValueError):
            adapter.prepare('a', 'sm0', None, adapter.routes)
        self.assertEqual(adapter.attempts('a'), 2)

    def test_simplemem_crash_after_mkdir_before_route_is_reconciled(self):
        adapter = self.simple_adapter()
        expected = adapter.histories['a'] / 'attempt_0001'
        adapter.routes.begin('a', 'sm1', str(expected), 1)
        expected.mkdir(parents=True)
        adapter.reconcile('a')
        self.assertEqual(config.read(expected / 'quad_route.json')['route']['lane'], 'sm1')

    def test_simplemem_valid_seal_survives_late_nonzero_exit_without_failure_marker(self):
        adapter = self.simple_adapter()
        job = adapter.prepare('a', 'sm0', None, adapter.routes)
        job['qid'] = 'a'
        prediction = {'question_id': 'a', 'hypothesis': 'verified synthetic'}
        adapter.verified = Mock(return_value=prediction)
        adapter.finish(job, -15, -15)
        self.assertEqual(adapter.completed['a'], prediction)
        self.assertFalse((job['attempt'] / 'failure.json').exists())
        self.assertEqual(config.read(job['attempt'] / 'quad_coordinator_result.json')['process_returncode'], -15)

    def light_adapter(self):
        from quad_lightmem_queue import LightMem
        adapter = LightMem.__new__(LightMem)
        adapter.ids = ['e47becba']
        adapter.lanes = self.directory / 'lightmem_lanes'
        adapter.lanes.mkdir()
        adapter.routes = self.routes(ids=adapter.ids, method='lightmem')
        adapter.journal, adapter.completed, adapter.failures = {}, {}, {}
        adapter.journal_path = self.directory / 'lightmem_attempts.json'
        with patch('quad_config.time.time', return_value=100):
            adapter.routes.begin('e47becba', 'lm1', 'lightmem_e47becba_1', 1)
        return adapter

    def test_lightmem_route_commit_before_journal_preserves_one_consumed_attempt(self):
        adapter = self.light_adapter()
        adapter.reconcile('e47becba')
        self.assertEqual(adapter.attempts('e47becba'), 1)
        self.assertEqual(adapter.journal['e47becba']['quad_route']['route']['lane'], 'lm1')
        self.assertTrue((adapter.lanes / 'lane_1/e47becba/quad_dispatch_1.json').exists())

    def test_lightmem_prediction_before_post_exit_receipt_recovers_current_binding(self):
        adapter = self.light_adapter()
        target = adapter.lanes / 'lane_1/e47becba'
        attempt = target / 'attempt_150000000000'
        attempt.mkdir(parents=True)
        config.atomic(target / 'prediction.json', {'attempt': attempt.name, 'question_id': 'e47becba', 'hypothesis': 'synthetic'})
        adapter.reconcile('e47becba')
        self.assertEqual(config.read(attempt / 'quad_route.json'), adapter.routes.receipts('e47becba')[0])
        self.assertEqual(adapter.attempts('e47becba'), 1)
        adapter.reconcile('e47becba')
        self.assertEqual(len(adapter.routes.receipts('e47becba')), 1)

    def test_lightmem_valid_prediction_survives_late_nonzero_and_retains_first_dispatch(self):
        adapter = self.light_adapter()
        target = adapter.lanes / 'lane_1/e47becba'
        attempt = target / 'attempt_150000000000'
        attempt.mkdir(parents=True)
        config.atomic(target / 'prediction.json', {'attempt': attempt.name})
        adapter.reconcile('e47becba')
        original = config.read(attempt / 'quad_route.json')
        adapter.validate = Mock(return_value=b'validated')
        adapter.finish({'qid': 'e47becba', 'target': target, 'route_receipt': {'unrelated': 'never overwrite'}}, -15, -15)
        self.assertEqual(adapter.completed['e47becba'], target)
        self.assertEqual(adapter.journal['e47becba']['process_returncode'], -15)
        self.assertEqual(config.read(attempt / 'quad_route.json'), original)

    def test_lightmem_old_attempt_and_duplicate_worker_are_rejected(self):
        adapter = self.light_adapter()
        target = adapter.lanes / 'lane_1/e47becba'
        attempt = target / 'attempt_90000000000'
        attempt.mkdir(parents=True)
        with self.assertRaises(ValueError):
            adapter.reconcile('e47becba')
        attempt.rmdir()
        (target / 'attempt_150000000000').mkdir()
        (target / 'attempt_160000000000').mkdir()
        with self.assertRaises(ValueError):
            adapter.reconcile('e47becba')

    def test_both_main_entrypoints_share_the_current_route_object(self):
        import importlib
        from types import SimpleNamespace
        state = self.directory / 'fast_native2_20260911'
        state.mkdir()
        for module_name, adapter_name, method in [('quad_simplemem_queue', 'SimpleMem', 'simplemem'),
                                                   ('quad_lightmem_queue', 'LightMem', 'lightmem')]:
            module = importlib.import_module(module_name)
            old = self.routes(ids=['a'], method=method)
            current = self.routes(ids=['a'], method=method)
            adapter = SimpleNamespace(method=method, ids=['a'], evidence=[], routes=old)
            fcntl = SimpleNamespace(flock=Mock(), LOCK_EX=1, LOCK_NB=2)
            with self.subTest(method=method), patch.dict('sys.modules', {'fcntl': fcntl}), patch('sys.argv', [module_name]), patch.object(config, 'ROOT', self.directory), patch.object(module, adapter_name, return_value=adapter), patch.object(config, 'Routes', return_value=current), patch.object(module, 'Scheduler') as scheduler, patch.object(module.signal, 'signal'):
                scheduler.return_value.execute.return_value = False
                self.assertEqual(module.main(), 1)
                self.assertIs(adapter.routes, current)
                self.assertIs(scheduler.call_args.args[1], adapter.routes)

    def test_unavailable_backend_does_not_allocate_or_launch(self):
        adapter = Adapter(self.directory)
        scheduler = Scheduler(adapter, self.routes(adapter.ids), 77)
        with patch('quad_scheduler.model_ready', return_value=False), patch('quad_scheduler.subprocess.Popen') as popen:
            scheduler.launch('sm0', 'a')
        self.assertEqual(adapter.attempts('a'), 0)
        self.assertFalse(scheduler.routes.data['histories'])
        popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()