"""No paid calls or live Vast operations; realistic full-500 export fixtures."""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finish_and_stop as app

REAL_PROCESSES = app.owned_processes


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = [f'q{index}' for index in range(500)]
        self.native = app.load_simplemem(Path(__file__).resolve().parents[1])
        self.native.source_hashes = lambda: {}
        self.save('longmemeval_s_cleaned.json', [
            {'question_id': qid, 'question_type': 'single-session-user', 'answer': 'answer',
             'question': 'Question?', 'question_date': '2026/01/01',
             'haystack_dates': ['2025/01/01'], 'haystack_session_ids': ['session'],
             'haystack_sessions': [[{'role': 'user', 'content': 'source'},
                                   {'role': 'assistant', 'content': 'response'}]]}
            for qid in self.ids])
        self.data_hash = app.sha(self.root / 'longmemeval_s_cleaned.json')
        upstream = {}
        for name in app.export_official.UPSTREAM_HASHES:
            path = self.root / 'official_longmemeval' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# fixture', encoding='utf-8')
            upstream[name] = app.sha(path)
        for target, attribute, value in (
            (app.export_official, 'DATA_SHA256', self.data_hash),
            (app.score_diagnostic_f1, 'DATA_SHA256', self.data_hash),
            (app.export_official, 'UPSTREAM_HASHES', upstream),
            (app, 'owned_processes', Mock(return_value=[])),
            (app, 'idle_queues', Mock(side_effect=lambda root: nullcontext())),
            (app, 'load_simplemem', Mock(return_value=self.native)),
        ):
            active = patch.object(target, attribute, value)
            active.start()
            self.addCleanup(active.stop)
        self.api = Mock()

    def save(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def make_runs(self, sources=False):
        simple = app.RUNS['simplemem']
        protocol = {'runtime': {'method': 'simplemem', 'model': 'Qwen/Qwen3.5-9B'},
                    'dataset_sha256': self.data_hash, 'population_ids': self.ids, 'source_files_sha256': {}}
        self.save(simple + '/protocol.json', protocol)
        self.save(simple + '/status.json', {'method': 'simplemem', 'planned': 500,
            'generated': 500, 'failed': 0, 'status': 'generation_complete'})
        self.save(simple + '/failures.json', [])
        predictions = []
        for item in app.read(self.root / 'longmemeval_s_cleaned.json'):
            qid = item['question_id']
            source = self.native.source_only(item)
            query = {key: item[key] for key in ('question_id', 'question', 'question_date')}
            identity = {'protocol_sha256': self.native.digest(protocol),
                        'source_sha256': self.native.digest(source),
                        'query_sha256': self.native.digest(query)}
            prediction = {'question_id': qid, 'hypothesis': 'answer', 'status': 'generated',
                          'identity': identity}
            predictions.append(prediction)
            if sources:
                attempt = simple + '/histories/' + hashlib.sha256(qid.encode()).hexdigest()[:24] + '/attempt_0001'
                artifacts = {'source.json': source, 'query.json': query, 'prediction.json': prediction,
                    'worker.json': {}, 'memory.json': {'memory': 'fixture'}, 'dialogues.json': [],
                    'build_complete.json': {}, 'usage.json': {}, 'native_runtime.json': {},
                    'llm_calls.jsonl': {}, 'location_compat_audit.jsonl': {}, 'json_syntax_compat_audit.jsonl': {}}
                hashes = {name: app.sha(self.save(attempt + '/' + name, value))
                          for name, value in artifacts.items()}
                self.save(attempt + '/completion.json', {'identity': identity, 'files_sha256': hashes})
        self.save(simple + '/predictions.json', predictions)
        light = app.RUNS['lightmem']
        protocol = {'dataset_sha256': self.data_hash, 'model': 'Qwen/Qwen3.5-9B',
                    'embedding_model': 'MiniLM', 'upstream_sha256': 'c' * 64}
        self.save(light + '/protocol.json', protocol)
        self.save(light + '/status.json', {'method': 'lightmem', 'selected': 500,
            'completed': 500, 'failed': [], 'population': 500, 'generation_complete': True})
        evidence = {}
        for qid in self.ids:
            prediction = {'question_id': qid, 'hypothesis': 'answer', 'attempt': 'attempt_1', **protocol}
            target = self.save(light + '/' + qid + '/prediction.json', prediction)
            if sources:
                relative = 'fast_native2_20260911/lightmem_lanes/lane_0/' + qid
                source = self.save(relative + '/prediction.json', prediction)
                self.save(str(Path(relative).parent / 'protocol.json'), protocol)
                self.save(relative + '/attempt_1/source.json', {
                    'haystack_dates': ['2025/01/01'], 'haystack_session_ids': ['session'],
                    'haystack_sessions': [[{'role': 'user', 'content': 'source'},
                                          {'role': 'assistant', 'content': 'response'}]]})
                self.save(relative + '/attempt_1/construction.json', {'source_turns_supplied': 2})
                database = self.root / relative / 'attempt_1/qdrant/storage.db'
                database.parent.mkdir()
                database.write_bytes(b'fixture-memory-database')
                evidence[qid] = {'source_dir': str(source.parent),
                                 'prediction_sha256': app.sha(target)}
        if sources:
            self.save(light + '/aggregation_receipt.json', {'sources': evidence})
            for relative in ('native_five.py', 'native_lightmem.py', 'run_all.py',
                'export_official.py', 'score_diagnostic_f1.py',
                'source/MemoryData/utils/request_metering.py',
                'official_recovery/simplemem_native_dialogues_v4/runner.py',
                'source/LightMem/lightmem.py'):
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# fixture', encoding='utf-8')

    def test_incomplete_never_stops(self):
        result = app.tick(self.root, self.api, True)
        self.assertEqual(result['status'], 'waiting')
        self.api.request.assert_not_called()

    def test_active_owned_worker_blocks_even_forged_status(self):
        app.owned_processes.return_value = [1234]
        result = app.tick(self.root, self.api, True)
        self.assertEqual(result['active_owned_pids'], [1234])
        self.api.request.assert_not_called()

    def test_forged_500_status_with_missing_prediction_never_stops(self):
        self.make_runs()
        (self.root / app.RUNS['lightmem'] / self.ids[0] / 'prediction.json').unlink()
        with self.assertRaises(ValueError):
            app.tick(self.root, self.api, True)
        self.api.request.assert_not_called()

    def test_protocol_mismatch_never_stops(self):
        self.make_runs()
        path = self.root / app.RUNS['simplemem'] / 'predictions.json'
        rows = app.read(path)
        rows[0]['identity']['protocol_sha256'] = 'wrong'
        path.write_text(json.dumps(rows), encoding='utf-8')
        with self.assertRaises(ValueError):
            app.tick(self.root, self.api, True)
        self.api.request.assert_not_called()

    def test_failed_count_prevents_stop(self):
        self.make_runs()
        path = self.root / app.RUNS['simplemem'] / 'status.json'
        status = app.read(path)
        status['failed'] = 1
        path.write_text(json.dumps(status), encoding='utf-8')
        self.assertEqual(app.tick(self.root, self.api, True)['status'], 'waiting')
        self.api.request.assert_not_called()

    def test_archive_tamper_prevents_stop(self):
        self.make_runs(sources=True)
        self.assertEqual(app.tick(self.root, self.api, False)['status'], 'verified_ready')
        target = self.root / app.ARCHIVE
        target.write_bytes(b'tampered')
        with self.assertRaises(Exception):
            app.tick(self.root, self.api, True)
        self.api.request.assert_not_called()

    def test_full_verified_stop_and_restart_are_idempotent(self):
        self.make_runs(sources=True)
        result = app.tick(self.root, self.api, True)
        self.assertEqual(result['status'], 'stop_accepted')
        self.assertEqual(self.api.request.call_count, 2)
        self.api.request.assert_called_with(stop=True)
        archive = self.root / app.ARCHIVE
        receipt = app.read(archive.with_name(archive.name + '.receipt.json'))
        self.assertEqual(receipt['archive_sha256'], app.sha(archive))
        app.verify_archive(archive, receipt['files'])
        self.assertTrue(any(name.endswith('qdrant/storage.db') for name in receipt['files']))
        app.tick(self.root, self.api, True)
        self.assertEqual(self.api.request.call_count, 2)

    def test_existing_exports_are_rechecked_against_current_predictions(self):
        self.make_runs(sources=True)
        app.export_and_score(self.root)
        app.export_and_score(self.root)
        path = self.root / app.RUNS['simplemem'] / 'predictions.json'
        rows = app.read(path)
        rows[0]['hypothesis'] = 'changed after export'
        path.write_text(json.dumps(rows), encoding='utf-8')
        with self.assertRaises(ValueError):
            app.tick(self.root, self.api, True)
        self.api.request.assert_not_called()

    def test_missing_lightmem_memory_evidence_prevents_stop(self):
        self.make_runs(sources=True)
        path = self.root / 'fast_native2_20260911/lightmem_lanes/lane_0/q0/attempt_1/construction.json'
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            app.tick(self.root, self.api, True)
        self.api.request.assert_not_called()

    def test_low_disk_never_stops(self):
        self.make_runs(sources=True)
        with patch.object(app.shutil, 'disk_usage', return_value=Mock(free=0)):
            with self.assertRaises(OSError):
                app.tick(self.root, self.api, True)
        self.api.request.assert_not_called()

    def test_uncertain_persisted_stop_is_not_automatically_repeated(self):
        self.save(app.STATE + '/stop_state.json', {'accepted': False, 'attempt': 1})
        self.assertEqual(app.tick(self.root, self.api, True)['status'], 'needs_attention')
        self.api.request.assert_not_called()

    def test_retry_limit_is_three_and_survives_restart(self):
        def request(stop=False):
            if stop:
                raise TimeoutError()
        self.api.request.side_effect = request
        with patch.object(app.time, 'sleep'):
            self.assertEqual(app.stop_once(self.root, self.api), 'needs_attention')
        self.assertEqual(self.api.request.call_count, 6)
        self.assertEqual(app.stop_once(self.root, self.api), 'needs_attention')
        self.assertEqual(self.api.request.call_count, 6)

    def test_relative_queue_argv_is_detected_using_cwd(self):
        proc = self.root / 'proc'
        entry = proc / '991234'
        entry.mkdir(parents=True)
        (entry / 'cmdline').write_bytes(b'python\0fast_native2_20260911/simplemem_queue.py\0')
        (entry / 'cwd').mkdir()
        self.assertEqual(REAL_PROCESSES(self.root, proc), [991234])

    def test_missing_simplemem_sealed_history_prevents_stop(self):
        self.make_runs(sources=True)
        history = hashlib.sha256(self.ids[0].encode()).hexdigest()[:24]
        path = self.root / app.RUNS['simplemem'] / 'histories' / history / 'attempt_0001/memory.json'
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            app.tick(self.root, self.api, True)
        self.api.request.assert_not_called()

    def test_secret_scan_and_lock_exclusion(self):
        self.assertTrue(app.excluded(Path('memory/.lock')))
        self.assertTrue(app.excluded(Path('.env')))
        self.assertFalse(app.excluded(Path('memory/storage.db-wal')))
        with self.assertRaises(ValueError):
            app.digest_stream(io.BytesIO(b'abc test-secret-key xyz'), (b'test-secret-key',))


class VastTests(unittest.TestCase):
    def test_wrong_instance_rejected_before_network(self):
        with self.assertRaises(ValueError):
            app.Vast(app.INSTANCE_ID, environ={'CONTAINER_ID': 'another', 'CONTAINER_API_KEY': 'secret'})

    def test_stop_requires_explicit_success_and_is_never_destroy(self):
        response = Mock(status=200)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read = Mock(return_value=b'{"success":false}')
        opener = Mock()
        opener.open.return_value = response
        api = app.Vast(app.INSTANCE_ID,
            environ={'CONTAINER_ID': app.INSTANCE_ID, 'CONTAINER_API_KEY': 'test-secret'}, opener=opener)
        with self.assertRaises(RuntimeError):
            api.request(stop=True)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.method, 'PUT')
        self.assertEqual(json.loads(request.data), {'state': 'stopped'})
        self.assertEqual(opener.open.call_args.kwargs['timeout'], 20)


if __name__ == '__main__':
    unittest.main()



