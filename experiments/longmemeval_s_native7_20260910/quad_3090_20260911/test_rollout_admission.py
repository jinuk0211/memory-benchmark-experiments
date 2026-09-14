"""Mocked rollout tests: no processes, services, networks or GPUs are changed."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import signal
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import quad_config as cfg
import rollout_admission as app
from test_quad_queues import Fixture


class RolloutChecks(Fixture):
    def setUp(self):
        super().setUp()
        patch.object(app, 'Q', self.directory).start()
        for key, name in {'RECEIPT': 'admission_rollout_attempt_2.json', 'WRAPPER': 'quad_metered_proxy.py',
                          'PENDING': 'quad_metered_proxy.admission_pending.py',
                          'BACKUP': 'admission_wrapper_backup.py'}.items():
            patch.object(app, key, self.directory / name).start()
        self.old, self.new = b'original-wrapper', b'admission-wrapper'
        patch.object(app, 'ORIGINAL_SHA', hashlib.sha256(self.old).hexdigest()).start()
        patch.object(app, 'WRAPPER_SHA', hashlib.sha256(self.new).hexdigest()).start()
        pins = {}
        for name in app.PINS:
            (self.directory / name).write_text(name)
            pins[name] = cfg.sha(self.directory / name)
        patch.object(app, 'PINS', pins).start()
        (self.directory / 'serve_qwen.sh').write_text('unchanged launcher')
        app.WRAPPER.write_bytes(self.old)
        app.PENDING.write_bytes(self.new)
        cfg.atomic(self.directory / 'pipeline_status.json', {'phase': 'experiments_running'})
        patch.object(cfg, 'ROOT', self.directory).start()
        (self.directory / 'fast_native2_20260911').mkdir()
        self.events, self.args = [], {}
        self.processes = {}
        for index, name in enumerate(('quad-pipeline', 'quad-meter', *app.QUEUES, 'quad-sm0', 'quad-sm1', 'quad-lm0', 'quad-lm1')):
            pid = 100 + index
            self.processes[name] = {'statename': 'RUNNING', 'pid': pid}
            self.args[pid] = [str(self.directory / app.QUEUES[name])] if name in app.QUEUES else ['python', '--max-num-seqs', '2' if '-sm' in name else '1']
        self.supervisor = SimpleNamespace(getProcessInfo=lambda name: dict(self.processes[name]),
            stopProcess=Mock(side_effect=self.stop), startProcess=Mock(side_effect=self.start))
        patch.object(app, 'rpc', return_value=self.supervisor).start()
        patch.object(app, 'argv', side_effect=lambda pid: self.args[pid]).start()
        patch.object(app.os, 'kill', side_effect=self.terminate).start()
        patch.object(app, 'health', return_value={'status': 'ok', 'in_flight': 0}).start()
        patch.dict('sys.modules', {'fcntl': SimpleNamespace(LOCK_EX=1, LOCK_NB=2,
            flock=lambda handle, flags: self.events.append(('lock', handle.name)))}).start()
        patch.object(app.time, 'sleep', side_effect=AssertionError('Unexpected waiting in mocked test')).start()
        self.rollout = app.Rollout()
        self.canary = patch.object(app.Rollout, 'run_canary', autospec=True, side_effect=self.verified).start()

    def stop(self, name, wait):
        self.assertIn(name, ('quad-pipeline', 'quad-meter'))
        self.events.append(('stop', name))
        self.processes[name]['statename'] = 'STOPPED'

    def start(self, name, wait):
        self.assertIn(name, ('quad-pipeline', 'quad-meter'))
        self.events.append(('start', name))
        self.processes[name]['statename'] = 'RUNNING'
        if name == 'quad-pipeline':
            for queue, script in app.QUEUES.items():
                pid = self.processes[queue]['pid'] + 1000
                self.processes[queue] = {'statename': 'RUNNING', 'pid': pid}
                self.args[pid] = [str(self.directory / script)]

    def terminate(self, pid, sig):
        self.assertEqual(sig, signal.SIGTERM)
        name = next(name for name in app.QUEUES if self.processes[name]['pid'] == pid)
        self.events.append(('term', name))
        self.processes[name]['statename'] = 'EXITED'

    def verified(self, rollout):
        self.assertTrue(rollout.locked)
        self.assertEqual(app.WRAPPER.read_bytes(), self.new)
        self.events.append(('canary', 'verified'))

    def test_success_drains_directly_holds_both_locks_and_only_restarts_meter(self):
        self.assertEqual(self.rollout.execute(), 0)
        self.assertEqual(self.rollout.state['phase'], 'complete')
        self.assertEqual(app.BACKUP.read_bytes(), self.old)
        operations = [item for item in self.events if item[0] != 'lock']
        self.assertEqual(operations, [('stop', 'quad-pipeline'), ('term', 'quad-simplemem'),
            ('term', 'quad-lightmem'), ('stop', 'quad-meter'), ('start', 'quad-meter'),
            ('canary', 'verified'), ('start', 'quad-pipeline')])
        self.assertEqual(len([item for item in self.events if item[0] == 'lock']), 2)
        self.assertFalse(self.rollout.locked)
        self.assertEqual(self.rollout.state['attempts'], 2)
        self.assertTrue(self.rollout.state['installed'])

    def test_canary_failure_restores_original_and_resumes_queues(self):
        self.canary.side_effect = RuntimeError('canary failed')
        self.assertEqual(self.rollout.execute(), 1)
        self.assertEqual(app.WRAPPER.read_bytes(), self.old)
        self.assertEqual(self.rollout.state['phase'], 'rolled_back')
        self.assertTrue(self.rollout.state['queues_resumed'])
        self.assertEqual(self.events.count(('start', 'quad-meter')), 2)

    def test_failed_restore_never_resumes_rejected_wrapper_or_canonical_queues(self):
        self.canary.side_effect = RuntimeError('canary failed')
        with patch.object(self.rollout, 'restore', side_effect=OSError('restore failed')):
            self.assertEqual(self.rollout.execute(), 1)
        self.assertEqual(app.WRAPPER.read_bytes(), self.new)
        self.assertEqual(self.rollout.state['phase'], 'rollback_failed')
        self.assertFalse(self.rollout.state['queues_resumed'])
        self.assertTrue(self.rollout.state['needs_attention'])
        self.assertNotIn(('start', 'quad-pipeline'), self.events)
        self.assertEqual(self.processes['quad-pipeline']['statename'], 'STOPPED')
        self.assertTrue(all(self.processes[name]['statename'] == 'EXITED' for name in app.QUEUES))
        self.assertFalse(self.rollout.locked)

    def test_signal_during_canary_rolls_back_and_cleanup_ignores_second_signal(self):
        self.canary.side_effect = lambda rollout: rollout.interrupted(signal.SIGTERM, None)
        self.assertEqual(self.rollout.execute(), 1)
        self.rollout.interrupted(signal.SIGTERM, None)
        self.assertEqual(app.WRAPPER.read_bytes(), self.old)
        self.assertEqual(self.rollout.state['phase'], 'rolled_back')

    def test_failure_before_install_keeps_queues_paused_until_meter_is_healthy(self):
        with patch.object(app, 'health', side_effect=ValueError('journal unhealthy')):
            self.assertEqual(self.rollout.execute(), 1)
        self.assertEqual(app.WRAPPER.read_bytes(), self.old)
        self.assertFalse(app.BACKUP.exists())
        self.assertNotIn(('stop', 'quad-meter'), self.events)
        self.assertNotIn(('start', 'quad-pipeline'), self.events)
        self.assertEqual(self.rollout.state['phase'], 'rollback_failed')
        self.assertFalse(self.rollout.state['queues_resumed'])
        self.assertTrue(self.rollout.state['needs_attention'])
        self.assertTrue(self.rollout.locked)
        self.rollout.locks.close()

    def test_resume_boundary_refuses_stopped_meter_even_if_http_claims_healthy(self):
        self.rollout.preflight()
        self.rollout.pause()
        self.rollout.drain()
        self.processes['quad-meter']['statename'] = 'STOPPED'
        with self.assertRaisesRegex(ValueError, 'running healthy meter'):
            self.rollout.resume()
        self.assertTrue(self.rollout.locked)
        self.assertNotIn(('start', 'quad-pipeline'), self.events)
        self.rollout.locks.close()

    def test_lost_pipeline_restart_reply_redrains_new_children_before_rollback(self):
        starts = 0
        def start(name, wait):
            nonlocal starts
            self.start(name, wait)
            if name == 'quad-pipeline':
                starts += 1
                if starts == 1:
                    raise OSError('lost response after pipeline started')
        self.supervisor.startProcess.side_effect = start
        self.assertEqual(self.rollout.execute(), 1)
        self.assertEqual(self.events.count(('term', 'quad-simplemem')), 2)
        self.assertEqual(self.events.count(('term', 'quad-lightmem')), 2)
        self.assertEqual(len([item for item in self.events if item[0] == 'lock']), 4)
        self.assertEqual(app.WRAPPER.read_bytes(), self.old)
        self.assertEqual(self.rollout.state['phase'], 'rolled_back')

    def test_persistent_enospc_receipt_during_cleanup_does_not_prevent_restore(self):
        def fail(rollout):
            patch.object(cfg, 'atomic', side_effect=OSError('ENOSPC')).start()
            raise OSError('ENOSPC')
        self.canary.side_effect = fail
        self.assertEqual(self.rollout.execute(), 1)
        self.assertEqual(app.WRAPPER.read_bytes(), self.old)
        self.assertIn(('start', 'quad-pipeline'), self.events)

    def test_attempt_two_applies_once_and_preserves_failed_attempt_one_verbatim(self):
        old = self.directory / 'admission_rollout.json'
        old.write_bytes(b'{"attempts":1,"phase":"rollback_failed","identity":{"code_sha256":"old"},"rollback_error":"OSError"}')
        unchanged = old.read_bytes()
        self.canary.side_effect = ValueError('canary failed')
        self.assertEqual(self.rollout.execute(), 1)
        events = list(self.events)
        with self.assertRaisesRegex(ValueError, 'Attempt 2 apply budget'):
            app.Rollout().execute()
        self.assertEqual(events, self.events)
        self.assertEqual(cfg.read(app.RECEIPT)['attempts'], 2)
        self.assertEqual(old.read_bytes(), unchanged)

    def test_interrupted_install_is_rolled_back_without_new_attempt_or_canary(self):
        self.rollout.preflight()
        self.rollout.report('installing_wrapper')
        app.BACKUP.write_bytes(self.old)
        app.WRAPPER.write_bytes(self.new)
        recovered = app.Rollout()
        self.assertEqual(recovered.execute(), 1)
        self.assertEqual(recovered.state['attempts'], 2)
        self.assertEqual(recovered.state['phase'], 'rolled_back')
        self.canary.assert_not_called()
        self.assertEqual(app.WRAPPER.read_bytes(), self.old)

    def test_wrong_instance_pins_and_sequence_limit_fail_before_service_mutation(self):
        for change in ('instance', 'pending', 'sequence'):
            with self.subTest(change=change):
                environment = cfg.INSTANCE if change != 'instance' else 'other'
                with patch.dict(os.environ, {'CONTAINER_ID': environment}):
                    if change == 'pending':
                        app.PENDING.write_bytes(b'changed')
                    if change == 'sequence':
                        app.PENDING.write_bytes(self.new)
                        self.args[self.processes['quad-sm0']['pid']][-1] = '1'
                    with self.assertRaises(ValueError):
                        app.Rollout().execute()
                self.assertEqual(self.events, [])

    def test_wrong_coordinator_pid_is_not_signaled_or_stopped(self):
        self.args[self.processes['quad-simplemem']['pid']] = ['/unrelated/process.py']
        self.assertEqual(self.rollout.execute(), 1)
        self.assertFalse(any(item[0] == 'term' for item in self.events))
        self.assertFalse(any(item == ('stop', name) for name in app.QUEUES for item in self.events))

    def test_changed_active_sha_restores_even_when_installed_flag_is_false(self):
        def ambiguous_swap():
            app.BACKUP.write_bytes(self.old)
            app.WRAPPER.write_bytes(self.new)
            self.assertFalse(self.rollout.state['installed'])
            raise OSError('failure before installed flag was persisted')
        with patch.object(self.rollout, 'idle_meter', side_effect=ambiguous_swap):
            self.assertEqual(self.rollout.execute(), 1)
        self.assertEqual(app.WRAPPER.read_bytes(), self.old)
        self.assertTrue(self.rollout.state['original_meter_healthy'])
        self.assertTrue(self.rollout.state['queues_resumed'])

    def test_bad_wrapper_real_http_health_rolls_back_before_any_queue_resume(self):
        fixture = self
        loaded, old_health = [self.old], []
        class Health(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                if loaded[0] != fixture.old:
                    self.close_connection = True
                    return
                old_health.append(True)
                body = json.dumps({'status': 'ok', 'in_flight': 0}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Health)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        from urllib.request import urlopen
        def health():
            with urlopen('http://127.0.0.1:' + str(server.server_port) + '/health', timeout=1) as response:
                return json.load(response)
        def start(name, wait):
            if name == 'quad-meter':
                loaded[0] = app.WRAPPER.read_bytes()
                if loaded[0] == self.new:
                    self.assertTrue(cfg.read(app.RECEIPT)['installed'])
                    old_health.clear()
            if name == 'quad-pipeline':
                self.assertEqual(app.WRAPPER.read_bytes(), self.old)
                self.assertTrue(old_health, 'Queues resumed before restored meter passed real HTTP health')
            self.start(name, wait)
        def bounded_wait(predicate, deadline, message):
            for _ in range(3):
                if predicate():
                    return
            raise TimeoutError(message)
        self.supervisor.startProcess.side_effect = start
        try:
            with patch.object(app, 'health', side_effect=health), patch.object(app, 'wait_for', side_effect=bounded_wait):
                self.assertEqual(self.rollout.execute(), 1)
            self.assertEqual(self.rollout.state['phase'], 'rolled_back')
            self.assertTrue(self.rollout.state['original_meter_healthy'])
            self.canary.assert_not_called()
        finally:
            server.shutdown()
            server.server_close()

    def test_pending_change_during_drain_is_rejected_before_install(self):
        original = self.rollout.idle_meter
        def changed():
            original()
            app.PENDING.write_bytes(b'foreign edit')
        with patch.object(self.rollout, 'idle_meter', side_effect=changed):
            self.assertEqual(self.rollout.execute(), 1)
        self.assertEqual(app.WRAPPER.read_bytes(), self.old)
        self.canary.assert_not_called()

    def test_expired_wait_never_accepts_predicate_or_resets_deadline(self):
        predicate = Mock(return_value=True)
        with self.assertRaises(TimeoutError):
            app.wait_for(predicate, 0, 'expired')
        predicate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
