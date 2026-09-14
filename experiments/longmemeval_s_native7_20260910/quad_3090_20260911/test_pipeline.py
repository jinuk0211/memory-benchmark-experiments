"""Local pipeline gates: mocked Supervisor, subprocesses and inference."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pipeline as app
import probe_replicas as probes

REAL_START = app.start


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state = Path(self.temporary.name)
        self.stack = []
        fake_fcntl = SimpleNamespace(flock=Mock(), LOCK_EX=2, LOCK_NB=4)
        changes = [patch.dict(sys.modules, {'fcntl': fake_fcntl}),
                   patch.dict(os.environ, {'CONTAINER_ID': app.cfg.INSTANCE}, clear=True),
                   patch.object(app.cfg, 'STATE', self.state), patch.object(app.cfg, 'ROOT', self.state),
                   patch.object(app, 'report'), patch.object(app, 'start'),
                   patch.object(app, 'run'), patch.object(app, 'wait_inference'),
                   patch.object(app, 'wait_service_finished'), patch.object(app.cfg, 'require_ready'),
                   patch.object(app.subprocess, 'run')]
        for item in changes:
            item.start()
            self.addCleanup(item.stop)

    def assert_no_queues(self):
        started = [call.args[0] for call in app.start.call_args_list]
        self.assertNotIn('quad-simplemem', started)
        self.assertNotIn('quad-lightmem', started)

    def test_failed_installer_prevents_any_follow_on_work(self):
        app.wait_service_finished.side_effect = RuntimeError('installer failed')
        with self.assertRaises(RuntimeError):
            app.main()
        app.run.assert_not_called()
        app.start.assert_not_called()

    def test_failed_model_download_prevents_fresh_preparation_and_queues(self):
        app.wait_service_finished.side_effect = [None, RuntimeError('model download failed')]
        with self.assertRaises(RuntimeError):
            app.main()
        app.run.assert_not_called()
        self.assert_no_queues()

    def test_failed_byte_verification_prevents_inference_and_queues(self):
        def run(script, *args):
            if script == 'verify_fresh.py':
                raise subprocess.CalledProcessError(1, script)
        app.run.side_effect = run
        with self.assertRaises(subprocess.CalledProcessError):
            app.main()
        app.start.assert_not_called()

    def test_failed_runtime_probe_cannot_start_canonical_queues(self):
        def run(script, *args):
            if script == 'probe_replicas.py':
                raise subprocess.CalledProcessError(1, script)
        app.run.side_effect = run
        with self.assertRaises(subprocess.CalledProcessError):
            app.main()
        self.assert_no_queues()

    def test_mismatched_ready_refused_before_canonical_dispatch(self):
        (self.state / 'READY.json').write_text('{}')
        app.cfg.require_ready.side_effect = ValueError('fresh hash mismatch')
        with self.assertRaises(ValueError):
            app.main()
        self.assert_no_queues()

    def test_pipeline_restart_cannot_bypass_exhausted_inference_retry_budget(self):
        app.cfg.atomic(self.state / 'service_restarts.json', {'quad-sm0': 2})
        rpc = Mock()
        rpc.getProcessInfo.side_effect = lambda name: {
            'statename': 'EXITED' if name == 'quad-sm0' else 'RUNNING', 'exitstatus': 1}
        app.wait_inference.side_effect = app.recover_inference
        app.start.side_effect = REAL_START
        with patch.object(app, 'rpc', return_value=rpc):
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                app.main()
        rpc.startProcess.assert_not_called()
        self.assert_no_queues()

    def test_archive_existence_alone_cannot_report_complete(self):
        (self.state / 'archive_verified.json').write_text('{}')
        with self.assertRaises((ValueError, KeyError)):
            app.main()
        self.assertFalse(any(call.args[0] == 'complete' for call in app.report.call_args_list))


class RecoveryTests(unittest.TestCase):
    def test_retry_persists_before_start_and_remains_bounded_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            rpc = Mock()
            rpc.getProcessInfo.side_effect = lambda name: {
                'statename': 'EXITED' if name == 'quad-sm0' else 'RUNNING'}
            def start(name, retry=False):
                self.assertTrue(retry)
                self.assertEqual(name, 'quad-sm0')
                self.assertIn(app.cfg.read(state / 'service_restarts.json')[name], (1, 2))
                raise RuntimeError('start failed')
            with patch.object(app.cfg, 'STATE', state), patch.object(app, 'rpc', return_value=rpc), \
                    patch.object(app, 'start', side_effect=start) as starter:
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, 'start failed'):
                        app.recover_inference()
                with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                    app.recover_inference()
                self.assertEqual(starter.call_count, 2)
                self.assertEqual(app.cfg.read(state / 'service_restarts.json'), {'quad-sm0': 2})


class SupervisorTests(unittest.TestCase):
    def test_nonzero_or_terminal_preparation_fails_immediately(self):
        for state, code in [('EXITED', 1), ('STOPPED', 0), ('FATAL', 0), ('UNKNOWN', 0)]:
            rpc = Mock()
            rpc.getProcessInfo.return_value = {'statename': state, 'exitstatus': code}
            with self.subTest(state=state), patch.object(app, 'rpc', return_value=rpc), patch.object(app.time, 'sleep') as sleep:
                with self.assertRaises(RuntimeError):
                    app.wait_service_finished('quad-models', 9000)
                sleep.assert_not_called()

    def test_zero_exit_is_preparation_success(self):
        rpc = Mock()
        rpc.getProcessInfo.return_value = {'statename': 'EXITED', 'exitstatus': 0}
        with patch.object(app, 'rpc', return_value=rpc):
            app.wait_service_finished('quad-models', 9000)


class ProbeGateTests(unittest.TestCase):
    def test_fresh_mismatch_prevents_probe_or_ready(self):
        with patch.dict(os.environ, {'CONTAINER_ID': probes.cfg.INSTANCE}), \
                patch.object(probes.cfg, 'deployment', return_value={}), \
                patch.object(probes.cfg, 'read', return_value={'status': 'fresh_artifacts_verified', 'fresh_run_sha256': 'old'}), \
                patch.object(probes.cfg, 'sha', return_value='current'), \
                patch.object(probes, 'stream_probe') as stream, patch.object(probes.cfg, 'atomic') as atomic:
            with self.assertRaises(ValueError):
                probes.main()
            stream.assert_not_called()
            atomic.assert_not_called()

    def test_failed_native_probe_cannot_publish_ready(self):
        config = probes.cfg.deployment()
        artifact = {'status': 'fresh_artifacts_verified', 'fresh_run_sha256': 'current'}
        with patch.dict(os.environ, {'CONTAINER_ID': probes.cfg.INSTANCE}), \
                patch.object(probes.cfg, 'read', return_value=artifact), \
                patch.object(probes.cfg, 'sha', return_value='current'), \
                patch.object(probes.cfg, 'deployment', return_value=config), \
                patch.object(probes, 'stream_probe', return_value=Path('synthetic-result.json')), \
                patch.object(probes, 'light_probe', side_effect=subprocess.CalledProcessError(1, 'native probe')), \
                patch.object(probes.cfg, 'atomic') as atomic:
            with self.assertRaises(subprocess.CalledProcessError):
                probes.main()
            atomic.assert_not_called()


if __name__ == '__main__':
    unittest.main()
