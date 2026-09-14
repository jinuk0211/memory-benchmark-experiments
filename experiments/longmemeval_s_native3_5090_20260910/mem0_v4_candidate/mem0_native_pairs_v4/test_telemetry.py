"""CPU-only tests for the scoped Mem0 telemetry thread opt-out."""
import contextlib
import gzip
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mem0_v3_telemetry_tested', HERE / 'runner.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def evidence():
    return {'posthog_version': '3.25.0', 'send': False, 'sync_mode': True,
            'disabled': True, 'created_clients': 2}


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = Mock(side_effect=lambda *a, **k: types.SimpleNamespace(
            send=k['send'], sync_mode=k['sync_mode'], disabled=k['disabled'], consumers=None))
        self.telemetry = types.SimpleNamespace(
            __file__=str(HERE / 'vendor/mem0/memory/telemetry.py'),
            MEM0_TELEMETRY=False, Posthog=self.original,
            client_telemetry=types.SimpleNamespace(close=Mock()))
        self.posthog = types.SimpleNamespace(Posthog=self.original)
        self.modules = {'mem0.memory.telemetry': self.telemetry, 'posthog': self.posthog}
        self.importer = patch.object(r.importlib, 'import_module', side_effect=self.modules.__getitem__)
        self.version = patch.object(r.importlib.metadata, 'version', return_value='3.25.0')
        self.importer.start()
        self.version_mock = self.version.start()
        self.addCleanup(self.importer.stop)
        self.addCleanup(self.version.stop)

    def test_gate_rejects_wrong_version_optout_path_or_factory_without_closing(self):
        cases = [('version', '3.24.0'), ('MEM0_TELEMETRY', True),
                 ('MEM0_TELEMETRY', 'false'), ('__file__', str(HERE / 'other.py')),
                 ('Posthog', Mock())]
        for key, value in cases:
            with self.subTest(key=key, value=str(value)):
                if key == 'version':
                    self.version_mock.return_value = value
                    try:
                        with self.assertRaises(ValueError):
                            r.disable_telemetry_threads()
                    finally:
                        self.version_mock.return_value = '3.25.0'
                else:
                    with patch.object(self.telemetry, key, value):
                        with self.assertRaises(ValueError):
                            r.disable_telemetry_threads()
        self.telemetry.client_telemetry.close.assert_not_called()
        self.original.assert_not_called()

    def test_initial_close_and_only_native_factory_gets_forced_flags(self):
        state = r.disable_telemetry_threads()
        self.telemetry.client_telemetry.close.assert_called_once_with()
        self.assertIs(self.posthog.Posthog, self.original)
        for _ in range(500):
            client = self.telemetry.Posthog('synthetic-key', host='http://unused.invalid',
                                            send=True, sync_mode=False, disabled=False)
            self.assertIsNone(client.consumers)
            self.assertTrue(client.disabled)
        self.assertEqual(state['created_clients'], 500)
        self.original.assert_called_with('synthetic-key', host='http://unused.invalid',
                                         send=False, sync_mode=True, disabled=True)

    def test_wrong_client_flags_or_consumers_are_fatal_and_not_counted(self):
        state = r.disable_telemetry_threads()
        for key, bad in [('send', True), ('sync_mode', False), ('disabled', False),
                         ('consumers', []), ('consumers', [object()])]:
            with self.subTest(key=key, bad=str(bad)):
                client = types.SimpleNamespace(send=False, sync_mode=True,
                                                disabled=True, consumers=None)
                setattr(client, key, bad)
                self.original.side_effect = None
                self.original.return_value = client
                with self.assertRaises(RuntimeError):
                    self.telemetry.Posthog('synthetic-key')
        self.assertEqual(state['created_clients'], 0)

    def test_initial_close_failure_does_not_partially_install_factory(self):
        self.telemetry.client_telemetry.close.side_effect = RuntimeError('synthetic close failure')
        with self.assertRaisesRegex(RuntimeError, 'close failure'):
            r.disable_telemetry_threads()
        self.assertIs(self.telemetry.Posthog, self.original)
        self.original.assert_not_called()

    def make_attempt(self):
        attempt = self.root / 'run/histories/qid/attempt_0001'
        attempt.mkdir(parents=True)
        source, query = [], {'question_id': 'qid'}
        identity = {'source_sha256': r.harness.digest(source),
                    'query_sha256': r.harness.digest(query)}
        files = {'source.json': source, 'query.json': query, 'worker.json': {},
                 'native_config.json': {}, 'native_runtime.json': {},
                 'build_complete.json': {'identity': identity},
                 'native_answer_detail.json': {}, 'native_degradation.json': {},
                 'native_telemetry.json': evidence(),
                 'prediction.json': {'identity': identity, 'question_id': 'qid',
                                     'status': 'generated', 'hypothesis': 'synthetic answer'}}
        for name, data in files.items():
            r.harness.save_json(attempt / name, data)
        for name in ['memory/history.db', 'memory/qdrant/closed.sqlite',
                     'memory_responses.jsonl.gz', 'console.log']:
            path = attempt / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic closed artifact')
        r.NativeObservations(attempt).save()
        with gzip.open(attempt / 'memory_responses.jsonl.gz', 'wt', encoding='utf-8'):
            pass
        return attempt, identity

    def test_required_telemetry_artifact_cannot_be_omitted(self):
        attempt, identity = self.make_attempt()
        files = r.attempt_files(attempt)
        files.pop('native_telemetry.json')
        with patch.object(r, 'attempt_files', return_value=files):
            with self.assertRaisesRegex(ValueError, 'required native'):
                r.finalize_attempt(attempt, identity)
        self.assertFalse((attempt / 'completion.json').exists())

    def test_invalid_telemetry_evidence_cannot_be_sealed(self):
        attempt, identity = self.make_attempt()
        for key, bad in [('posthog_version', 'new'), ('send', True),
                         ('sync_mode', False), ('disabled', False),
                         ('created_clients', 0), ('created_clients', -1),
                         ('created_clients', True), ('created_clients', 1.5)]:
            with self.subTest(key=key, bad=bad):
                data = {**evidence(), key: bad}
                r.harness.save_json(attempt / 'native_telemetry.json', data)
                with self.assertRaisesRegex(ValueError, 'telemetry opt-out'):
                    r.finalize_attempt(attempt, identity)
                self.assertFalse((attempt / 'completion.json').exists())

    def test_valid_telemetry_is_hash_sealed_and_mutation_invalidates_cache(self):
        attempt, identity = self.make_attempt()
        r.finalize_attempt(attempt, identity)
        receipt = r.harness.read_json(attempt / 'completion.json')
        self.assertIn('native_telemetry.json', receipt['files_sha256'])
        self.assertEqual(r.verified_prediction(attempt.parent, identity)['question_id'], 'qid')
        r.harness.save_json(attempt / 'native_telemetry.json', {**evidence(), 'created_clients': 3})
        with self.assertRaisesRegex(ValueError, 'artifacts changed'):
            r.verified_prediction(attempt.parent, identity)

    def test_agent_partial_initialization_fails_with_evidence_and_removes_handler(self):
        attempt = self.root / 'run/histories/qid/attempt_0001'
        attempt.mkdir(parents=True)
        source, query = [], {'question_id': 'qid'}
        runtime = {'source_root': str(self.root), 'api_base': 'http://127.0.0.1:1/v1',
                   'embedding_api_base': 'http://127.0.0.1:2/v1'}
        protocol = {'runtime': runtime, 'source_files_sha256': {}}
        identity = {'source_sha256': r.harness.digest(source),
                    'query_sha256': r.harness.digest(query),
                    'protocol_sha256': r.harness.digest(protocol)}
        for path, data in [(attempt.parents[2] / 'protocol.json', protocol),
                           (attempt / 'worker.json', {'identity': identity, **protocol}),
                           (attempt / 'source.json', source), (attempt / 'query.json', query)]:
            r.harness.save_json(path, data)
        metering = types.SimpleNamespace(install_request_metering=lambda: None,
                                        meter_operation=lambda *a, **k: contextlib.nullcontext())
        handlers = list(logging.getLogger().handlers)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'utils.request_metering': metering}))
            stack.enter_context(patch.dict(r.os.environ, {}))
            stack.enter_context(patch.object(r.sys, 'path', list(sys.path)))
            stack.enter_context(patch.object(r.os, 'chdir'))
            for name, result in [('validate_runtime', None), ('source_hashes', {}),
                                 ('dependency_versions', {}), ('memory_config', {})]:
                stack.enter_context(patch.object(r, name, return_value=result))
            stack.enter_context(patch.object(r.harness, 'build_config', return_value=({}, {})))
            stack.enter_context(patch.object(r, 'create_agent', side_effect=RuntimeError('partial init')))
            self.assertEqual(r.worker(attempt), 1)
        self.assertIn('partial init', json.loads((attempt / 'failure.json').read_text())['traceback'])
        self.assertTrue((attempt / 'native_degradation.json').exists())
        self.assertFalse((attempt / 'native_telemetry.json').exists())
        self.assertFalse((attempt / 'completion.json').exists())
        self.assertEqual(logging.getLogger().handlers, handlers)


if __name__ == '__main__':
    unittest.main(verbosity=2)