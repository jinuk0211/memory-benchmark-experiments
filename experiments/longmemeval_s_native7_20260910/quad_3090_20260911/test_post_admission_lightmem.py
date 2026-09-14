"""Bounded watcher tests; no real processes, services or waiting."""
import configparser
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import post_admission_lightmem as app


class WatcherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.quad = Path(temporary.name)
        self.receipt = self.quad / 'post_admission_lightmem.json'
        self.source = self.quad / 'admission_rollout.json'
        self.elapsed = 0
        self.sleeps = []
        for name, callback in [('monotonic', lambda: self.elapsed),
                               ('time', lambda: 1000 + self.elapsed), ('sleep', self.sleep)]:
            patcher = patch.object(app.time, name, side_effect=callback)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(app.subprocess, 'run', return_value=SimpleNamespace(returncode=0))
        self.start = patcher.start()
        self.addCleanup(patcher.stop)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.elapsed += seconds

    def publish(self, **state):
        self.source.write_text(json.dumps(state), encoding='utf-8')

    def read(self):
        return json.loads(self.receipt.read_text(encoding='utf-8'))

    def test_success_waits_for_both_conditions_and_starts_once_after_durable_claim(self):
        self.source.write_text('{incomplete', encoding='utf-8')
        def advance(seconds):
            self.sleep(seconds)
            self.publish(phase='complete', queues_resumed=1 if self.elapsed == 15 else True,
                         identity={'run_id': 'synthetic-current-run'})
        app.time.sleep.side_effect = advance
        def started(*args, **kwargs):
            self.assertEqual(self.read()['status'], 'start_requested')
            self.assertTrue(self.read()['start_attempted'])
            return SimpleNamespace(returncode=0)
        self.start.side_effect = started
        self.assertEqual(app.watch(self.quad), 0)
        self.assertEqual(self.sleeps, [15, 15])
        self.start.assert_called_once_with(['supervisorctl', 'start', 'quad-tune-lightmem'],
                                          capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(self.read()['status'], 'tuner_started')
        self.assertEqual(self.read()['admission_rollout_sha256'],
                         hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(app.watch(self.quad), 0)
        self.start.assert_called_once()

    def test_failure_phases_and_failed_or_ambiguous_start_never_retry(self):
        for phase in app.FAILED_PHASES:
            with self.subTest(phase=phase):
                self.receipt.unlink(missing_ok=True)
                self.publish(phase=phase, queues_resumed=True)
                self.assertEqual(app.watch(self.quad), 1)
                self.assertEqual(self.read()['status'], 'admission_failed')
                self.start.assert_not_called()
        self.publish(phase='complete', queues_resumed=True)
        for outcome in (SimpleNamespace(returncode=1), subprocess.TimeoutExpired('supervisorctl', 30)):
            with self.subTest(outcome=outcome):
                self.receipt.unlink(missing_ok=True)
                self.start.reset_mock()
                self.start.side_effect = outcome if isinstance(outcome, Exception) else None
                self.start.return_value = outcome
                self.assertEqual(app.watch(self.quad), 1)
                self.assertEqual(self.read()['status'], 'start_failed')
                self.assertEqual(app.watch(self.quad), 1)
                self.start.assert_called_once()
        self.receipt.write_text(json.dumps({'status': 'start_requested', 'start_attempted': True}))
        self.start.reset_mock()
        self.assertEqual(app.watch(self.quad), 1)
        self.start.assert_not_called()

    def test_timeout_has_four_hour_bound_and_missing_rollout_never_starts(self):
        self.assertEqual(app.MAX_WAIT_SECONDS, 14400)
        self.assertEqual(app.POLL_SECONDS, 15)
        self.assertEqual(app.watch(self.quad), 1)
        self.assertEqual(self.elapsed, 14400)
        self.assertEqual(len(self.sleeps), 960)
        self.assertEqual(set(self.sleeps), {15})
        self.assertEqual(self.read()['status'], 'timed_out')
        self.assertFalse(self.read()['start_attempted'])
        self.start.assert_not_called()
        self.assertEqual(app.watch(self.quad), 1)
        self.assertEqual(self.elapsed, 14400)

    def test_supervisor_starts_only_when_explicitly_requested(self):
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(Path(app.__file__).with_name('quad-post-admission.conf'), encoding='utf-8')
        self.assertEqual(parser.sections(), ['program:quad-post-admission'])
        config = parser['program:quad-post-admission']
        for flag in ('autostart', 'autorestart', 'stopasgroup', 'killasgroup'):
            self.assertFalse(config.getboolean(flag))
        self.assertEqual(config.getint('startretries'), 0)
        self.assertEqual(config.getint('startsecs'), 0)
        self.assertTrue(config['command'].endswith(
            '/usr/bin/python3 /workspace/quad_3090_20260911/post_admission_lightmem.py'))


if __name__ == '__main__':
    unittest.main()
