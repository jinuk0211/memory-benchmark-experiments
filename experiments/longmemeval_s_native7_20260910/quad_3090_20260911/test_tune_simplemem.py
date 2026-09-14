import signal
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import tune_simplemem as tune


class TuningTests(unittest.TestCase):
    def test_requires_speed_memory_and_overlap(self):
        baseline = {'seconds': 100}
        candidate = {'seconds': 90, 'peak_running': 2, 'preemptions': 0,
                     'peak_vram_mib': 21000, 'total_vram_mib': 24576}
        self.assertTrue(tune.acceptable(baseline, candidate))
        for field, value in [('seconds', 99), ('peak_running', 1), ('preemptions', 1),
                             ('peak_vram_mib', 24400)]:
            with self.subTest(field=field):
                self.assertFalse(tune.acceptable(baseline, {**candidate, field: value}))

    @patch.object(tune.cfg, 'read', return_value={'inflight': {}})
    @patch.object(tune.cfg, 'atomic')
    @patch.object(tune, 'record')
    @patch.object(tune.os, 'kill')
    @patch.object(tune, 'rpc')
    def test_drains_coordinator_only(self, rpc, kill, *_):
        server = rpc.return_value
        server.getProcessInfo.side_effect = [
            {'statename': 'RUNNING', 'pid': 123}, {'statename': 'EXITED'}]
        tune.drain({})
        kill.assert_called_once_with(123, signal.SIGTERM)
        server.stopProcess.assert_not_called()

    @patch.object(tune.cfg, 'read', return_value={'inflight': {'sm0': {'pid': 999}}})
    @patch.object(tune.cfg, 'atomic')
    @patch.object(tune, 'rpc')
    def test_rejects_leftover_workers(self, rpc, *_):
        rpc.return_value.getProcessInfo.return_value = {'statename': 'EXITED'}
        with self.assertRaisesRegex(RuntimeError, 'workers still present'):
            tune.drain({})
        rpc.return_value.stopProcess.assert_not_called()

    @patch.object(tune, 'monitor_inference')
    @patch.object(tune.cfg, 'atomic')
    @patch.object(tune, 'record')
    @patch.object(tune.os, 'kill')
    @patch.object(tune, 'rpc')
    def test_drain_timeout_preserves_workers(self, rpc, kill, *_):
        rpc.return_value.getProcessInfo.return_value = {'statename': 'RUNNING', 'pid': 123}
        with self.assertRaises(TimeoutError):
            tune.drain({'drain_deadline': 0})
        kill.assert_called_once_with(123, signal.SIGTERM)
        rpc.return_value.stopProcess.assert_not_called()

    @patch.object(tune, 'http')
    def test_metrics_are_required(self, http):
        http.return_value = 'vllm:num_requests_running{model_name="Qwen"} 2.0\n'
        with self.assertRaisesRegex(ValueError, 'Missing vLLM metric'):
            tune.metrics(18081)
        http.return_value += ('vllm:num_requests_waiting{model_name="Qwen"} 0.0\n'
                              'vllm:num_preemptions_total{model_name="Qwen"} 3.0\n')
        self.assertEqual(tune.metrics(18081)['num_requests_running'], 2)
        self.assertEqual(tune.metrics(18081)['num_preemptions_total'], 3)

    @patch.object(tune, 'record')
    @patch.object(tune.cfg, 'read', return_value={'phase': 'experiments_running'})
    @patch.object(tune, 'rpc')
    def test_resume_requires_live_queue(self, rpc, _, record):
        rpc.return_value.getProcessInfo.side_effect = [
            {'statename': 'STOPPED'}, {'statename': 'RUNNING'}, {'statename': 'RUNNING'}]
        tune.resume({}, 'complete')
        rpc.return_value.startProcess.assert_called_once_with('quad-pipeline', False)
        self.assertEqual(record.call_args.args[1], 'complete')
        self.assertTrue(record.call_args.kwargs['resumed'])

    @patch.object(tune, 'record')
    @patch.object(tune, 'healthy')
    @patch.object(tune.time, 'sleep')
    @patch.object(tune, 'rpc')
    def test_restart_waits_for_stopping(self, rpc, sleep, healthy, _):
        rpc.return_value.getProcessInfo.side_effect = [
            {'statename': 'STOPPING'}, {'statename': 'STOPPING'}, {'statename': 'STOPPED'}]
        with tempfile.TemporaryDirectory() as temporary, patch.object(tune, 'Q', Path(temporary)):
            tune.restart('sm0', 2, {})
            self.assertEqual((Path(temporary) / 'seqs_18081.txt').read_text(), '2\n')
        rpc.return_value.stopProcess.assert_not_called()
        rpc.return_value.startProcess.assert_called_once_with('quad-sm0', False)
        sleep.assert_called_once_with(2)
        healthy.assert_called_once_with('sm0', monitor=True)

    @patch.object(tune, 'record')
    @patch.object(tune.cfg, 'read', return_value={'phase': 'experiments_running'})
    @patch.object(tune, 'rpc')
    def test_resume_is_idempotent(self, rpc, _, record):
        rpc.return_value.getProcessInfo.return_value = {'statename': 'RUNNING'}
        tune.resume({}, 'rolled_back')
        rpc.return_value.startProcess.assert_not_called()
        self.assertEqual(record.call_args.args[1], 'rolled_back')

if __name__ == '__main__':
    unittest.main()