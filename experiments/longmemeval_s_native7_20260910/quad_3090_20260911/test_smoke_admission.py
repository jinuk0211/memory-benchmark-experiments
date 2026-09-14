"""Real unused-loopback-port smoke with temporary fixture identity."""
import os
from pathlib import Path
from unittest.mock import patch

import smoke_admission as app
from test_quad_queues import Fixture


class IsolatedSmoke(Fixture):
    def test_exact_five_request_sequence_same_connection_and_temp_journal(self):
        source = Path(__file__).with_name('quad_metered_proxy.admission_pending.py')
        root = app.cfg.ROOT
        if os.name == 'nt':
            root = Path(__file__).parents[1]
            source = Path(__file__).with_name('quad_metered_proxy.py')
        self.assertEqual(app.cfg.sha(source), app.PINS['quad_metered_proxy.py'])
        pending = self.directory / 'quad_metered_proxy.admission_pending.py'
        pending.write_bytes(source.read_bytes())
        loaded = []
        load = app.cfg.load_module
        def track(name, path):
            loaded.append(Path(path))
            return load(name, path)
        with patch.object(app, 'PENDING', pending), patch.object(app.cfg, 'ROOT', root), patch.object(app.cfg, 'load_module', side_effect=track), patch.dict(os.environ, {}, clear=False):
            result = app.run_smoke()
        self.assertEqual(loaded, [pending, root / 'fast_native2_20260911/dual_gpu_20260911/dual_metered_proxy.py'])
        self.assertEqual([row['path'] for row in result['checks']], ['/health', '/v1/models', '/unknown', 'stream', '/health'])
        self.assertEqual([row['http_status'] for row in result['checks']], [200, 200, 404, 200, 200])
        self.assertTrue(result['same_keepalive_connection'])
        self.assertTrue(result['temporary_journal_and_activity'])
        self.assertFalse(result['production_ports_used'])
        self.assertFalse((self.directory / 'request_usage.jsonl').exists())

    def test_old_active_wrapper_cannot_be_used_as_pending_candidate(self):
        wrong = self.directory / 'quad_metered_proxy.admission_pending.py'
        wrong.write_bytes(b'old-active-wrapper')
        with patch.object(app, 'PENDING', wrong), patch.object(app.cfg, 'load_module') as loader:
            with self.assertRaisesRegex(ValueError, 'Pending candidate wrapper SHA'):
                app.run_smoke()
        loader.assert_not_called()
