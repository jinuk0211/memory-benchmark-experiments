"""Focused checks for LightMem-only tuning and rollback/holder safety."""
import concurrent.futures
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import tune_lightmem as tune


class TuningTests(unittest.TestCase):
    def test_acceptance_requires_ten_percent_overlap_zero_preemption_and_headroom(self):
        baseline = {'seconds': 100}
        candidate = {'seconds': 90, 'peak_running': 2, 'preemptions': 0,
                     'peak_vram_mib': 21000, 'total_vram_mib': 24576, 'outputs_nonempty': True}
        self.assertTrue(tune.acceptable(baseline, candidate))
        for key, value in [('seconds', 90.01), ('peak_running', 1), ('preemptions', 1),
                           ('peak_vram_mib', 24064), ('outputs_nonempty', False)]:
            with self.subTest(key=key):
                self.assertFalse(tune.acceptable(baseline, {**candidate, key: value}))

    @patch.object(tune.cfg, 'read', return_value={'inflight': {}})
    @patch.object(tune.cfg, 'atomic')
    @patch.object(tune, 'record')
    @patch.object(tune.os, 'kill')
    @patch.object(tune, 'rpc')
    def test_drain_signals_only_lightmem_coordinator(self, rpc, kill, *_):
        rpc.return_value.getProcessInfo.side_effect = [
            {'statename': 'RUNNING', 'pid': 123}, {'statename': 'EXITED'}]
        tune.drain({})
        kill.assert_called_once_with(123, signal.SIGTERM)
        self.assertEqual([call.args[0] for call in rpc.return_value.getProcessInfo.call_args_list],
                         ['quad-lightmem', 'quad-lightmem'])
        rpc.return_value.stopProcess.assert_not_called()

    @patch.object(tune.cfg, 'read', return_value={'inflight': {'lm0': {'pid': 55}}})
    @patch.object(tune.cfg, 'atomic')
    @patch.object(tune, 'rpc')
    def test_drain_rejects_live_workers(self, rpc, *_):
        rpc.return_value.getProcessInfo.return_value = {'statename': 'EXITED'}
        with self.assertRaisesRegex(RuntimeError, 'workers still present'):
            tune.drain({})
        rpc.return_value.stopProcess.assert_not_called()

    @patch.object(tune, 'http')
    def test_five_distinct_prefixes_are_tokenized_once_exactly_2000_ids(self, http):
        http.side_effect = [{'tokens': [index] + [99] * 2999} for index in range(5)]
        prompts = tune.make_prompts('lm0')
        self.assertEqual(len(prompts), 5)
        self.assertEqual({len(prompt) for prompt in prompts}, {2000})
        self.assertEqual({prompt[0] for prompt in prompts}, set(range(5)))
        prefixes = [call.args[2]['prompt'].split()[0] for call in http.call_args_list]
        self.assertEqual(len(set(prefixes)), 5)

    def test_identical_batches_send_five_fixed_workloads_and_validate_outputs(self):
        calls = []
        prompts = [[index] * 2000 for index in range(5)]
        def http(port, path, payload, timeout):
            calls.append(payload)
            return {'usage': {'prompt_tokens': 2000, 'completion_tokens': 1024, 'total_tokens': 3024},
                    'choices': [{'text': 'nonempty synthetic output'}]}
        # Complete requests immediately but through real futures, independent of GPU/network.
        class Pool:
            def __init__(self, max_workers):
                self.max_workers = max_workers
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def submit(self, function, *args):
                future = concurrent.futures.Future()
                future.set_result(function(*args))
                return future
        with patch.object(tune, 'http', side_effect=http), patch.object(tune, 'metrics', return_value={
                'num_requests_running': 2, 'num_requests_waiting': 0, 'num_preemptions_total': 0}), \
                patch.object(tune, 'memory', return_value=(22000, 24576)), \
                patch.object(tune.concurrent.futures, 'ThreadPoolExecutor', Pool):
            first = tune.request_batch('lm0', prompts)
            second = tune.request_batch('lm0', prompts)
        self.assertEqual(calls[:5], calls[5:])
        self.assertEqual([row['prompt'] for row in calls[:5]], prompts)
        self.assertTrue(first['outputs_nonempty'] and second['outputs_nonempty'])
        self.assertTrue(all(row['max_tokens'] == row['min_tokens'] == 1024
                            and row['ignore_eos'] is True for row in calls))

    def test_restart_budget_is_preserved_and_current_lane_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(tune, 'Q', Path(temporary)):
            path = Path(temporary) / 'service_restarts.json'
            tune.cfg.atomic(path, {'quad-sm0': 1, 'quad-lm0': 2})
            server = Mock()
            server.getProcessInfo.side_effect = lambda name: {
                'statename': 'EXITED' if name in {'quad-sm0', 'quad-lm0'} else 'RUNNING'}
            with patch.object(tune, 'rpc', return_value=server):
                tune.monitor_inference(exclude=('quad-lm0',))
                self.assertEqual(tune.cfg.read(path), {'quad-sm0': 2, 'quad-lm0': 2})
                server.startProcess.assert_called_once_with('quad-sm0', False)
                with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                    tune.monitor_inference(exclude=('quad-lm0',))
                self.assertEqual(server.startProcess.call_count, 1)

    @patch.object(tune, 'record')
    @patch.object(tune, 'healthy')
    @patch.object(tune, 'rpc')
    def test_rollback_restart_does_not_invoke_ancillary_monitor(self, rpc, healthy, _):
        rpc.return_value.getProcessInfo.return_value = {'statename': 'STOPPED'}
        with tempfile.TemporaryDirectory() as temporary, patch.object(tune, 'Q', Path(temporary)):
            tune.restart('lm0', 1, {})
            self.assertEqual((Path(temporary) / 'seqs_18101.txt').read_text(), '1\n')
        healthy.assert_called_once_with('lm0', monitor=False)
        rpc.return_value.startProcess.assert_called_once_with('quad-lm0', False)
        rpc.return_value.stopProcess.assert_not_called()
        with self.assertRaises(ValueError):
            tune.restart('sm0', 1, {})

    @patch.object(tune.cfg, 'atomic')
    @patch.object(tune, 'write_seqs')
    @patch.object(tune, 'restart', side_effect=[RuntimeError('first lane down'), None])
    def test_rollback_attempts_both_lanes_even_when_first_fails(self, restart, write, _):
        state = {'results': {}}
        with self.assertRaisesRegex(RuntimeError, 'rollback incomplete'):
            tune.rollback(state)
        self.assertEqual([call.args[:2] for call in write.call_args_list], [('lm0', 1), ('lm1', 1)])
        self.assertEqual([call.args[:2] for call in restart.call_args_list], [('lm0', 1), ('lm1', 1)])
        self.assertEqual(state['results']['lm1']['selected'], 1)

    def holder_fixture(self, process, mismatch=False):
        def spawn(command, **kwargs):
            destination = Path(command[5])
            tune.cfg.atomic(destination, {'pid': -1 if mismatch else process.pid,
                'parent_pid': os.getpid(), 'allocated_bytes': int(command[4]),
                'cuda_index': 3, 'gpu_uuid': 'GPU-test'})
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '3')
            self.assertEqual(kwargs['env']['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
            self.assertNotIn('CONTAINER_API_KEY', kwargs['env'])
            return process
        return spawn

    def test_cuda_holder_reserves_delta_and_is_terminated_on_canary_exception(self):
        process = Mock(pid=456)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary, patch.object(tune, 'Q', Path(temporary)), \
                patch.object(tune, 'memory', return_value=(21000, 24576)), \
                patch.object(tune, 'gpu_query', return_value=['3']), \
                patch.object(tune.cfg, 'deployment', return_value={'lanes': {'lm1': {'gpu_uuid': 'GPU-test'}}}), \
                patch.dict(os.environ, {'CONTAINER_API_KEY': 'must_not_inherit'}), \
                patch.object(tune.subprocess, 'Popen', side_effect=self.holder_fixture(process)):
            with self.assertRaisesRegex(RuntimeError, 'synthetic canary failure'):
                with tune.memory_holder('lm1', 22200, 1200) as proof:
                    self.assertEqual(proof['allocated_bytes'], 1200 * 1024**2)
                    self.assertEqual(proof['requested_mib'], 1200)
                    raise RuntimeError('synthetic canary failure')
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=10)

    def test_candidate_higher_idle_vram_does_not_shrink_helper_allocation(self):
        process = Mock(pid=456)
        process.poll.return_value = None
        proofs = []
        with tempfile.TemporaryDirectory() as temporary, patch.object(tune, 'Q', Path(temporary)), \
                patch.object(tune, 'memory', side_effect=[(21000, 24576), (21600, 24576)]), \
                patch.object(tune, 'gpu_query', return_value=['3']), \
                patch.object(tune.cfg, 'deployment', return_value={'lanes': {'lm1': {'gpu_uuid': 'GPU-test'}}}), \
                patch.object(tune.subprocess, 'Popen', side_effect=self.holder_fixture(process)):
            baseline_native_reserve = 22200 - 21000
            for _ in range(2):
                with tune.memory_holder('lm1', 22200, baseline_native_reserve) as proof:
                    proofs.append(proof)
        self.assertEqual(proofs[0]['allocated_bytes'], 1200 * 1024**2)
        self.assertEqual(proofs[1]['allocated_bytes'], proofs[0]['allocated_bytes'])
        self.assertEqual(proofs[1]['used_after_restart_mib'], 21600)
        self.assertEqual(process.terminate.call_count, 2)

    def test_wrong_holder_readiness_is_rejected_and_holder_killed_if_term_times_out(self):
        process = Mock(pid=456)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('holder', 10), 0]
        with tempfile.TemporaryDirectory() as temporary, patch.object(tune, 'Q', Path(temporary)), \
                patch.object(tune, 'memory', return_value=(21000, 24576)), \
                patch.object(tune, 'gpu_query', return_value=['3']), \
                patch.object(tune.cfg, 'deployment', return_value={'lanes': {'lm1': {'gpu_uuid': 'GPU-test'}}}), \
                patch.object(tune.subprocess, 'Popen', side_effect=self.holder_fixture(process, True)):
            with self.assertRaisesRegex(ValueError, 'readiness proof'):
                with tune.memory_holder('lm1', 22200, 1200):
                    self.fail('Incorrect holder accepted')
        process.terminate.assert_called_once()
        process.kill.assert_called_once()

    @patch.object(tune, 'record')
    @patch.object(tune.cfg, 'read', return_value={'phase': 'experiments_running'})
    @patch.object(tune, 'rpc')
    def test_resume_requires_live_lightmem_and_does_not_start_simplemem(self, rpc, _, record):
        rpc.return_value.getProcessInfo.side_effect = [
            {'statename': 'STOPPED'}, {'statename': 'RUNNING'}, {'statename': 'RUNNING'}]
        tune.resume({}, 'rolled_back')
        rpc.return_value.startProcess.assert_called_once_with('quad-pipeline', False)
        self.assertEqual(rpc.return_value.getProcessInfo.call_args.args, ('quad-lightmem',))
        self.assertTrue(record.call_args.kwargs['resumed'])




class EntryPointTests(unittest.TestCase):
    def setup_files(self, directory, state):
        for name in ('serve_qwen.sh', 'serve_qwen.before_tuning.sh'):
            (directory / name).write_text(name)
        identity = {'instance_id': tune.cfg.INSTANCE, 'fresh_run_sha256': 'fresh',
                    'execution_run_id': 'run', 'fresh_initialized_sha256': 'initialized',
                    'launcher_sha256': tune.cfg.sha(directory / 'serve_qwen.sh'),
                    'baseline_launcher_sha256': tune.cfg.sha(directory / 'serve_qwen.before_tuning.sh')}
        tune.cfg.atomic(directory / 'lightmem_tuning.json', {**identity, **state})
        return {key: identity[key] for key in
                ('fresh_run_sha256', 'execution_run_id', 'fresh_initialized_sha256')}

    def test_interrupted_tuner_restores_instead_of_rerunning_canary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fresh = self.setup_files(directory, {'phase': 'candidate_lm0', 'attempts': 1,
                                                'mutations_started': True})
            server = Mock()
            server.getProcessInfo.return_value = {'statename': 'RUNNING'}
            with patch.object(tune, 'Q', directory), patch.object(tune, 'RECEIPT', directory / 'lightmem_tuning.json'), \
                    patch.dict(os.environ, {'CONTAINER_ID': tune.cfg.INSTANCE}), \
                    patch.dict('sys.modules', {'fcntl': Mock(LOCK_EX=2, LOCK_NB=4)}), \
                    patch.object(tune.cfg, 'fresh_identity', return_value=fresh), \
                    patch.object(tune, 'rpc', return_value=server), \
                    patch.object(tune, 'run_locked', return_value='rolled_back') as run, \
                    patch.object(tune, 'resume') as resume:
                tune.main()
            self.assertTrue(run.call_args.args[1])
            self.assertEqual(run.call_args.args[0]['attempts'], 2)
            server.stopProcess.assert_called_once_with('quad-pipeline', True)
            self.assertEqual(resume.call_args.args[1], 'rolled_back')

    def test_failed_preflight_retry_cannot_interrupt_an_active_simplemem_tuner(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fresh = self.setup_files(directory, {'phase': 'retry_recovery', 'attempts': 1,
                                                'mutations_started': False})
            tune.cfg.atomic(directory / 'pipeline_status.json', {'phase': 'experiments_running'})
            tune.cfg.atomic(directory / 'simplemem_tuning.json', {'phase': 'candidate_sm0'})
            server = Mock()
            with patch.object(tune, 'Q', directory), patch.object(tune, 'RECEIPT', directory / 'lightmem_tuning.json'), \
                    patch.dict(os.environ, {'CONTAINER_ID': tune.cfg.INSTANCE}), \
                    patch.dict('sys.modules', {'fcntl': Mock(LOCK_EX=2, LOCK_NB=4)}), \
                    patch.object(tune.cfg, 'fresh_identity', return_value=fresh), \
                    patch.object(tune.cfg, 'require_ready'), \
                    patch.object(tune, 'rpc', return_value=server), \
                    patch.object(tune, 'run_locked') as run:
                with self.assertRaisesRegex(ValueError, 'SimpleMem tuning'):
                    tune.main()
            server.stopProcess.assert_not_called()
            run.assert_not_called()
            self.assertEqual(tune.cfg.read(directory / 'lightmem_tuning.json')['phase'], 'needs_attention')

    def test_third_attempt_cannot_restart_any_service(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fresh = self.setup_files(directory, {'phase': 'retry_recovery', 'attempts': 2,
                                                'mutations_started': True})
            with patch.object(tune, 'Q', directory), patch.object(tune, 'RECEIPT', directory / 'lightmem_tuning.json'), \
                    patch.dict(os.environ, {'CONTAINER_ID': tune.cfg.INSTANCE}), \
                    patch.dict('sys.modules', {'fcntl': Mock(LOCK_EX=2, LOCK_NB=4)}), \
                    patch.object(tune.cfg, 'fresh_identity', return_value=fresh), \
                    patch.object(tune, 'rpc') as rpc, patch.object(tune, 'run_locked') as run:
                tune.main()
            rpc.assert_not_called()
            run.assert_not_called()

    def test_both_rollback_lanes_run_without_new_canary_on_interruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'fast_native2_20260911').mkdir()
            with patch.object(tune.cfg, 'ROOT', directory), patch.dict('sys.modules', {'fcntl': Mock(LOCK_EX=2, LOCK_NB=4)}), \
                    patch.object(tune, 'drain'), patch.object(tune, 'rollback') as rollback, \
                    patch.object(tune, 'request_batch') as request:
                state = {'results': {}}
                self.assertEqual(tune.run_locked(state, True), 'rolled_back')
            rollback.assert_called_once_with(state)
            request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
