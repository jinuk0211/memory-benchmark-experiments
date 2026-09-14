"""Local fixtures only: no inference, provider calls, or real credentials."""
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'fast_native2_20260911'))
sys.path.insert(0, str(HERE))
import archive_results as app  # noqa: E402


class FinalResultTests(unittest.TestCase):
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

    def test_full500_export_seals_and_database_are_archived_and_rechecked(self):
        self.make_runs(sources=True)
        app.complete_statuses(self.root)
        app.export_and_score(self.root)
        extra = self.save('public_evidence.json', {'instance_id': app.INSTANCE_ID})
        receipt = app.create_archive(self.root, {'quad_3090_20260911/public_evidence.json': extra})
        app.verify_archive(Path(receipt['archive']), receipt['files'])
        self.assertTrue(any('qdrant/storage.db' in name for name in receipt['files']))
        self.assertTrue(any(name.endswith('/completion.json') for name in receipt['files']))
        self.assertIn('quad_3090_20260911/public_evidence.json', receipt['files'])
        self.assertEqual(receipt, app.create_archive(self.root, {'quad_3090_20260911/public_evidence.json': extra}))
        Path(receipt['archive']).write_bytes(b'corrupt archive')
        with self.assertRaises(tarfile.ReadError):
            app.create_archive(self.root, {'quad_3090_20260911/public_evidence.json': extra})

    def test_missing_database_or_changed_sealed_answer_blocks_export(self):
        self.make_runs(sources=True)
        database = self.root / 'fast_native2_20260911/lightmem_lanes/lane_0/q0/attempt_1/qdrant/storage.db'
        database.unlink()
        with self.assertRaisesRegex(ValueError, 'LightmemSourceEvidenceMissing'):
            app.export_and_score(self.root)
        prediction = self.root / app.RUNS['simplemem'] / 'predictions.json'
        rows = app.read(prediction)
        rows[0]['hypothesis'] = 'changed'
        prediction.write_text(json.dumps(rows))
        with self.assertRaises(ValueError):
            app.validate_simplemem_sources(self.root)

    def test_active_worker_failed_status_and_low_disk_do_not_publish_success(self):
        app.owned_processes.return_value = [123]
        with self.assertRaisesRegex(RuntimeError, 'OwnedWorkersStillActive'):
            app.finish(self.root, self.root / 'quad')
        app.owned_processes.return_value = []
        self.make_runs(sources=True)
        status = app.read(self.root / app.RUNS['simplemem'] / 'status.json')
        status['failed'] = 1
        self.save(app.RUNS['simplemem'] + '/status.json', status)
        with self.assertRaises(RuntimeError):
            app.finish(self.root, self.root / 'quad')
        status['failed'] = 0
        self.save(app.RUNS['simplemem'] + '/status.json', status)
        app.export_and_score(self.root)
        with patch.object(app.shutil, 'disk_usage', return_value=Mock(free=0)):
            with self.assertRaises(OSError):
                app.create_archive(self.root, {})
        self.assertFalse((self.root / 'quad/archive_verified.json').exists())


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = app.read(HERE / 'deployment.json')
        (self.root / 'deployment.json').write_bytes((HERE / 'deployment.json').read_bytes())
        self.save('FRESH_RUN.json', {
            'status': 'fresh_run_authorized', 'instance_id': app.INSTANCE_ID,
            'dataset_sha256': 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442',
            'methods': {'simplemem': 500, 'lightmem': 500}, 'past_results_reused': False,
            'fresh_generation': True, 'canonical_population': 500, 'run_id': 'fixture-fresh'})
        self.save('fresh_initialized.json', {'status': 'fresh_initialized', 'instance_id': app.INSTANCE_ID,
            'fresh_run_sha256': app.sha(self.root / 'FRESH_RUN.json'),
            'initial_counts': {'simplemem': 0, 'lightmem': 0}, 'past_results_reused': False})
        (self.root / 'runtime_probe.py').write_text('# public synthetic probe')
        self.gate = {'status': 'runtime_verified', 'instance_id': app.INSTANCE_ID,
                     'deployment_sha256': app.DEPLOYMENT_SHA, 'protocol_proof': self.config['protocols'],
                     'model_proof': {'test': 'hashed'}, 'fresh_run_sha256': app.sha(self.root / 'FRESH_RUN.json'),
                     'lanes': {}, 'replicas': {}}
        for lane, binding in self.config['lanes'].items():
            proof = {'status': 'runtime_probe_verified', 'instance_id': app.INSTANCE_ID,
                     'lane': lane, 'gpu_uuid': binding['gpu_uuid'], 'benchmark_population_member': False,
                     'script_sha256': app.sha(self.root / 'runtime_probe.py'),
                     'streamed_chat': {'done_received': True, 'finish_reason': 'stop', 'content_characters': 2},
                     'qwen_models': {'max_model_len': 65536},
                     'native_lightmem': {'synthetic_fact_recovered': True, 'stored_points': 1,
                                        'cuda_forward_calls': {'compressor': 1, 'internal_minilm': 2}},
                     'native_model_devices': {'compressor': 'cuda:0', 'internal_minilm': 'cuda:0'},
                     'concurrent_vllm_pids': [400]}
            path = self.save('probe_' + lane + '/unique/result.json', proof)
            self.gate['lanes'][lane] = {'receipt': path.relative_to(self.root).as_posix(), 'sha256': app.sha(path)}
            self.gate['replicas'][lane] = {'status': 'inference_verified', 'model': self.config['model'],
                                         'max_model_len': 65536, 'gpu_uuid': binding['gpu_uuid'],
                                         'api_base': binding['api_base']}
        self.publish_gate()
        for method in app.RUNS:
            self.save('routes_' + method + '.json', {'instance_id': app.INSTANCE_ID,
                'method': method, 'deployment_sha256': app.DEPLOYMENT_SHA})
        self.save('request_usage.jsonl', {'physical_call': 'synthetic fixture'})

    def save(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def publish_gate(self):
        self.save('runtime_verified.json', self.gate)
        self.save('READY.json', self.gate)

    def test_public_allowlist_preserves_all_four_proofs_and_omits_secrets_and_cache(self):
        for name in ('hf-download.dpapi', '.hf-token', 'secret.json', 'cache/model.bin'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'not archive input')
        files = app.quad_evidence(self.root)
        self.assertEqual(sum(name.endswith('/result.json') for name in files), 4)
        self.assertIn('quad_3090_20260911/request_usage.jsonl', files)
        self.assertFalse(any('token' in name or 'dpapi' in name or 'secret' in name or 'cache' in name for name in files))
        self.assertTrue(app.excluded(Path('nested/hf-download.dpapi')))

    def test_lightmem_tuning_evidence_is_included_without_widening_the_allowlist(self):
        expected = {'tune_lightmem.py', 'test_tune_lightmem.py', 'test_lightmem_integration.py',
                    'quad-tune-lightmem.conf', 'lightmem_tuning.json',
                    'serve_qwen.after_simplemem_tuning.sh',
                    'seqs_18101.txt', 'seqs_18111.txt'}
        omitted = {'seqs_18112.txt', 'lightmem_tuning.private.json', 'tune_lightmem_private.py',
                   'serve_qwen.after_simplemem_tuning.private.sh'}
        for name in expected | omitted:
            (self.root / name).write_text('public synthetic fixture')
        files = app.quad_evidence(self.root)
        names = {Path(name).name for name in files}
        self.assertTrue(expected.issubset(names))
        self.assertFalse(omitted.intersection(names))

    def test_missing_fourth_replica_rejected(self):
        self.gate['lanes'].pop('lm1')
        self.publish_gate()
        with self.assertRaisesRegex(ValueError, 'FourLaneRuntimeProofRequired'):
            app.quad_evidence(self.root)

    def test_changed_probe_receipt_or_mismatched_gpu_rejected(self):
        entry = self.gate['lanes']['lm0']
        path = self.root / entry['receipt']
        proof = app.read(path)
        proof['gpu_uuid'] = self.config['lanes']['lm1']['gpu_uuid']
        self.save(entry['receipt'], proof)
        with self.assertRaisesRegex(ValueError, 'LaneRuntimeIdentityMismatch'):
            app.quad_evidence(self.root)
        entry['sha256'] = app.sha(path)
        self.publish_gate()
        with self.assertRaisesRegex(ValueError, 'LaneRuntimeIdentityMismatch'):
            app.quad_evidence(self.root)

    def test_fake_cuda_proof_and_past_result_reuse_rejected(self):
        entry = self.gate['lanes']['lm0']
        proof = app.read(self.root / entry['receipt'])
        proof['native_lightmem']['cuda_forward_calls']['compressor'] = 0
        self.save(entry['receipt'], proof)
        entry['sha256'] = app.sha(self.root / entry['receipt'])
        self.publish_gate()
        with self.assertRaisesRegex(ValueError, 'NativeLightMemCudaProofMissing'):
            app.quad_evidence(self.root)
        fresh = app.read(self.root / 'FRESH_RUN.json')
        fresh['past_results_reused'] = True
        self.save('FRESH_RUN.json', fresh)
        with self.assertRaisesRegex(ValueError, 'FreshRunAuthorizationRequired'):
            app.quad_evidence(self.root)

    def test_empty_usage_and_escaping_evidence_rejected(self):
        (self.root / 'request_usage.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'ActualUsageJournalMissing'):
            app.quad_evidence(self.root)
        self.gate['lanes']['sm0']['receipt'] = '../outside/result.json'
        self.publish_gate()
        with self.assertRaisesRegex(ValueError, 'EvidenceFileOutsideOwnedRootOrMissing'):
            app.validate_runtime(self.root)

    def fresh_dispatch_fixture(self):
        experiment = self.root / 'experiment'
        experiment.mkdir()
        self.save('experiment/longmemeval_s_cleaned.json', [{'question_id': 'q0'}, {'question_id': 'q1'}])
        identity = {'fresh_run_sha256': app.sha(self.root / 'FRESH_RUN.json'),
                    'execution_run_id': 'fixture-fresh',
                    'fresh_initialized_sha256': app.sha(self.root / 'fresh_initialized.json')}
        journal = []
        for method, lanes in [('simplemem', ['sm0', 'sm1']), ('lightmem', ['lm0', 'lm1'])]:
            routes = {'instance_id': app.INSTANCE_ID, 'method': method,
                      'deployment_sha256': app.DEPLOYMENT_SHA,
                      'protocol_sha256': self.config['protocols'][method],
                      **identity, 'histories': {}, 'dispatches': {}}
            sources = {}
            self.save('experiment/' + app.RUNS[method] + '/protocol.json', {})
            for index, lane in enumerate(lanes):
                qid = 'q' + str(index)
                binding = self.config['lanes'][lane]
                route = {'lane': lane, 'instance_id': app.INSTANCE_ID,
                         **{key: binding[key] for key in ('gpu_uuid', 'gpu_index', 'api_base')}}
                dispatch = {'id': 'dispatch-' + qid, 'ordinal': 1, 'started_at': 1}
                routes['histories'][qid] = route
                routes['dispatches'][qid] = [dispatch]
                receipt = {'deployment_sha256': app.DEPLOYMENT_SHA, 'method': method,
                           'question_id': qid, 'route': route, 'dispatch': dispatch, **identity}
                if method == 'simplemem':
                    relative = ('experiment/' + app.RUNS[method] + '/histories/'
                                + hashlib.sha256(qid.encode()).hexdigest()[:24] + '/attempt_0001')
                    self.save(relative + '/completion.json', {})
                else:
                    directory = 'experiment/fast_native2_20260911/lightmem_lanes/lane_' + str(index) + '/' + qid
                    relative = directory + '/attempt_1'
                    self.save(directory + '/prediction.json', {'attempt': 'attempt_1'})
                    sources[qid] = {'source_dir': str(self.root / directory)}
                self.save(relative + '/quad_route.json', receipt)
                journal.append({'success': True, 'method': method, 'sample_id': qid, 'question_id': qid,
                                'run_id': self.config['protocols'][method], 'canonical_benchmark': True,
                                'inference_lane': lane, 'inference_instance_id': app.INSTANCE_ID,
                                'inference_gpu_uuid': binding['gpu_uuid'],
                                'deployment_sha256': app.DEPLOYMENT_SHA, **identity})
            self.save('routes_' + method + '.json', routes)
            if method == 'lightmem':
                self.save('experiment/' + app.RUNS[method] + '/aggregation_receipt.json', {'sources': sources})
        for lane, entry in self.gate['lanes'].items():
            proof = app.read(self.root / entry['receipt'])
            proof['probe_id'] = 'quad_probe_' + lane + '_fixture'
            row = {'success': True, 'method': 'runtime_probe', 'sample_id': proof['probe_id'],
                   'question_id': proof['probe_id'], 'run_id': proof['probe_id'], 'canonical_benchmark': False,
                   'inference_lane': lane, 'inference_instance_id': app.INSTANCE_ID,
                   'inference_gpu_uuid': self.config['lanes'][lane]['gpu_uuid'],
                   'deployment_sha256': app.DEPLOYMENT_SHA, **identity}
            journal.append(row)
            proof['central_route'] = {'successful_requests': 1, 'inference_lane': lane,
                'inference_gpu_uuid': row['inference_gpu_uuid'],
                'records_sha256': hashlib.sha256(json.dumps([row], sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
            self.save(entry['receipt'], proof)
            entry['sha256'] = app.sha(self.root / entry['receipt'])
        self.publish_gate()
        (self.root / 'request_usage.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in journal))
        return experiment, journal

    def test_physical_journal_requires_every_history_and_all_four_gpu_routes(self):
        experiment, journal = self.fresh_dispatch_fixture()
        result = app.validate_usage(experiment, self.root)
        self.assertEqual(result['canonical_histories_with_success'], {'simplemem': 2, 'lightmem': 2})
        journal = [row for row in journal if not (row['method'] == 'simplemem' and row['sample_id'] == 'q0')]
        (self.root / 'request_usage.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in journal))
        with self.assertRaisesRegex(ValueError, 'FourGpuFullPopulationUsageEvidenceMissing'):
            app.validate_usage(experiment, self.root)

    def test_dispatch_from_previous_fresh_epoch_is_rejected(self):
        experiment, _ = self.fresh_dispatch_fixture()
        native = Mock()
        native.digest.return_value = self.config['protocols']['simplemem']
        real_sha = app.sha
        light_protocol = experiment / app.RUNS['lightmem'] / 'protocol.json'
        def sha(path, *args):
            return self.config['protocols']['lightmem'] if path == light_protocol else real_sha(path, *args)
        with patch.object(app, 'load_simplemem', return_value=native), patch.object(app, 'sha', side_effect=sha):
            app.validate_fresh_results(experiment, self.root)
            path = next((experiment / app.RUNS['simplemem'] / 'histories').glob('*/attempt_0001/quad_route.json'))
            receipt = app.read(path)
            receipt['execution_run_id'] = 'past-run'
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'FreshAttemptRouteMismatch'):
                app.validate_fresh_results(experiment, self.root)

    def test_probe_journal_hash_mismatch_is_rejected(self):
        experiment, journal = self.fresh_dispatch_fixture()
        journal[-1]['extra_changed_field'] = True
        (self.root / 'request_usage.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in journal))
        with self.assertRaisesRegex(ValueError, 'SyntheticRouteProofDoesNotMatchUsageJournal'):
            app.validate_usage(experiment, self.root)

    def test_source_has_no_provider_transport_or_power_control(self):
        tree = ast.parse((HERE / 'archive_results.py').read_text())
        forbidden = {'Vast', 'NoRedirect', 'stop_once', 'urlopen', 'Request', 'urlretrieve', 'system', 'Popen', 'run'}
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                self.assertNotIn(node.name, forbidden)
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
                self.assertNotIn(name, forbidden)
        self.assertNotIn('CONTAINER_API_KEY', (HERE / 'archive_results.py').read_text())


if __name__ == '__main__':
    unittest.main()
