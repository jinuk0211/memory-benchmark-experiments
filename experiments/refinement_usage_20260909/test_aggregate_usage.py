"""Behavioral accounting tests; all native data are local synthetic receipts."""
import json
from pathlib import Path
import tempfile
import unittest

import aggregate_usage as usage


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.sequences = {}
        self.baseline = {'config': {'model': usage.MODEL, 'embed_model': usage.EMBED_MODEL, 'seed': 20260907, 'dataset': 'locomo'},
                         'model': {'path': '/model', 'revision': usage.MODEL_REVISION},
                         'embedding': {'path': '/embedding', 'revision': usage.EMBED_REVISION},
                         'dataset_sha256': usage.DATA_HASH}
        self.write('runs/qwen35_baseline/protocol.json', self.baseline)

    def write(self, name, obj):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        return usage.sha(path)

    def receipt(self, kind='generation', *, stage='evaluate', run='qwen35_baseline',
                pid='101', instance='abc', operation=None, status='success', incomplete=False,
                seconds=2.0, utility=False, request_prompts=None, context=None, users=None):
        identity = (run, stage, pid, instance)
        self.sequences[identity] = self.sequences.get(identity, 0) + 1
        seq = self.sequences[identity]
        prefix = f'runs/{run}/' + ('utility/' if utility else '') + f'runtime/{stage}/{pid}/{instance}'
        stem = f'{prefix}/{kind}/{seq:08d}'
        initial = {'schema_version': 1, 'kind': kind, 'sequence': seq, 'profile': {'role': 'scorer' if stage == 'score' else 'runtime', 'model': self.baseline['model'],
                   'embedding': self.baseline['embedding'], 'dtype': 'float16', 'quantization': None,
                   'seed': 20260908 if stage == 'score' else 20260907, 'embedding_device': 'cpu',
                   'embedding_dtype': 'float32', 'native_max_seq_length': 256},
                   'context': context or {'phase': 'evaluating', 'method': 'seed'},
                   'operation_id': operation, 'operation_kind': 'generation' if operation else None,
                   'encode_attempt_id': None, 'status': 'error'}
        receipt = dict(initial)
        payload = {}
        response = None
        if kind == 'generation':
            payload = {'args': [request_prompts or ['prompt']], 'kwargs': {}}
            initial.update(prompt_tokens=None, completion_tokens=None, total_tokens=None, usage_complete=False)
            receipt.update(initial)
            if status == 'success':
                response = [{'prompt_token_ids': [1, 2, 3], 'outputs': [{'token_ids': [4, 5], 'finish_reason': 'stop'}],
                             'finished': not incomplete, 'metrics': None}]
                receipt.update(usage.generation_usage(response, len(payload['args'][0])))
        elif kind == 'embedding':
            features = {'input_ids': [[1, 2, 0], [3, 4, 5]], 'attention_mask': [[1, 1, 0], [1, 1, 1]]}
            payload = {'features': features, 'args': [], 'kwargs': {}}
            initial.update(usage.embedding_usage(features))
            receipt.update(initial)
        elif kind == 'operation':
            payload = {'inputs': {'users': users if users is not None else ['prompt']}, 'counts_before': {'generation': 0, 'embedding': 0}}
            receipt.update(actual_invocations={'generation': 0, 'embedding': 0},
                           zero_invocations_observed=True, cache_hit_proven=False)
        receipt['request_sha256'] = self.write(stem + '.request.json', {**initial, 'payload': payload})
        if response is not None:
            receipt['response_sha256'] = self.write(stem + '.response.json', response)
        receipt.update(status=status, call_duration_s=seconds, duration_s=seconds + 1000)
        if kind in ('generation', 'embedding'):
            receipt['inference_duration_s'] = seconds
        self.write(stem + '.receipt.json', receipt)
        return stem

    def update(self, name, fn):
        obj = usage.read(self.root / name)
        fn(obj)
        self.write(name, obj)

    def manifest(self):
        rows = [{'path': p.relative_to(self.root).as_posix(), 'bytes': p.stat().st_size, 'sha256': usage.sha(p)}
                for p in sorted(self.root.rglob('*')) if p.is_file() and p.name != 'evidence_manifest.json']
        self.write('evidence_manifest.json', {'files': rows})
        return usage.Evidence(self.root)

    def predictions(self):
        comparison = {'methods': {}, 'prediction_hashes': {}, 'dataset_sha256': usage.DATA_HASH}
        for method in usage.METHODS:
            run = usage.RUNS[1] if method == usage.METHODS[-1] else usage.RUNS[0]
            name = f'runs/{run}/{method}.jsonl'
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [{'question_id': str(i), 'conversation_id': 'conv', 'method': method, 'prediction': 'yes', 'hypothesis': 'yes',
                     'input_tokens': 10, 'output_tokens': 1, 'generation_batch_seconds': 99999}
                    for i in range(usage.N)]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            comparison['methods'][method] = {'n': usage.N, 'official_f1': 0.5}
            comparison['prediction_hashes'][method] = usage.sha(path)
        self.write('runs/recursive_v1/OFFICIAL_COMPARISON.json', comparison)
        self.write('runs/qwen35_baseline/BASELINE_COMPARISON.json', {'methods': comparison['methods'], 'dataset_sha256': usage.DATA_HASH})
        child = {'baseline_protocol_sha256': usage.digest(self.baseline), 'dataset_sha256': usage.DATA_HASH,
                 'environment': {'models': {usage.MODEL: self.baseline['model'], usage.EMBED_MODEL: self.baseline['embedding']}}}
        for run, protocol in zip(usage.RUNS, (self.baseline, child)):
            self.write(f'runs/{run}/protocol.json', protocol)
            self.write(f'runs/{run}/memory_lock.json', {'protocol_sha256': usage.digest(protocol)})


class UsageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)

    def totals(self):
        evidence = self.f.manifest()
        totals, groups, operations = usage.aggregate_runtime(evidence)
        return totals, groups, operations, evidence.gaps

    def test_native_only_excludes_nested_duration_and_counts_failed_embedding(self):
        self.f.receipt(seconds=2)
        self.f.receipt(kind='embedding', seconds=3, status='error')
        self.f.receipt(kind='operation', stage='prepare', seconds=999, operation='zero')
        self.f.receipt(kind='encode_attempt', seconds=500)
        total, _, ops, gaps = self.totals()
        self.assertEqual(gaps, [])
        self.assertEqual(total['llm_prompt_tokens_known'], 3)
        self.assertEqual(total['llm_completion_tokens_known'], 2)
        self.assertEqual(total['embedding_tokens_known'], 5)
        self.assertEqual(total['embedding_padded_tokens_known'], 6)
        self.assertEqual(total['failed_native_calls'], 1)
        self.assertEqual(total['llm_inference_seconds_known'] + total['embedding_inference_seconds_known'], 5)
        self.assertEqual(ops['explicit_cache_hit_proven'], 0)
        self.assertEqual(ops['zero_invocation_operations_not_proven_cache_hits'], 1)

    def test_retry_is_real_cost_even_identical_prompt(self):
        self.f.receipt(status='error', seconds=4)
        self.f.receipt(seconds=2)
        total, _, _, gaps = self.totals()
        self.assertFalse(gaps)
        self.assertEqual(total['native_generation_calls'], 2)
        self.assertEqual(total['llm_inference_seconds_known'], 6)
        result = usage.finish_totals(total)
        self.assertIsNone(result['llm_tokens_M'])
        self.assertEqual(result['llm_tokens_known_lower_bound'], 5)

    def test_incomplete_response_keeps_recorded_lower_bound(self):
        self.f.receipt(incomplete=True)
        total, _, _, gaps = self.totals()
        self.assertFalse(gaps)
        self.assertEqual(total['incomplete_usage_calls'], 1)
        self.assertEqual(usage.finish_totals(total)['llm_tokens_known_lower_bound'], 5)
        self.assertIsNone(usage.finish_totals(total)['llm_tokens_M'])

    def test_corrupt_raw_token_count_rejected_even_with_refreshed_manifest(self):
        stem = self.f.receipt()
        self.f.update(stem + '.receipt.json', lambda row: row.update(total_tokens=600))
        total, _, _, gaps = self.totals()
        self.assertEqual(total['native_generation_calls'], 0)
        self.assertTrue(any('arithmetic' in gap['reason'] for gap in gaps))

    def test_changed_response_digest_rejected(self):
        stem = self.f.receipt()
        self.f.update(stem + '.response.json', lambda rows: rows[0].update(prompt_token_ids=[1]))
        total, _, _, gaps = self.totals()
        self.assertEqual(total['native_generation_calls'], 0)
        self.assertTrue(any('Response digest' in gap['reason'] for gap in gaps))

    def test_missing_response_and_orphan_request_detected(self):
        stem = self.f.receipt()
        (self.f.root / (stem + '.response.json')).unlink()
        self.f.write('runs/qwen35_baseline/runtime/evaluate/101/abc/generation/00000002.request.json', {})
        total, _, _, gaps = self.totals()
        self.assertEqual(total['native_generation_calls'], 0)
        self.assertTrue(any('Orphan' in gap['reason'] for gap in gaps))
        self.assertTrue(any('Missing' in gap['reason'] for gap in gaps))

    def test_manifest_detects_missing_and_tampered_files(self):
        stem = self.f.receipt()
        self.f.manifest()
        (self.f.root / (stem + '.response.json')).unlink()
        evidence = usage.Evidence(self.f.root)
        self.assertIn(stem + '.response.json', evidence.bad)

    def test_mask_arithmetic_corruption_detected(self):
        stem = self.f.receipt(kind='embedding')
        self.f.update(stem + '.receipt.json', lambda row: row.update(input_tokens=6))
        total, _, _, gaps = self.totals()
        self.assertEqual(total['native_embedding_calls'], 0)
        self.assertTrue(any('embedding token' in gap['reason'] for gap in gaps))

    def test_utility_and_multiple_pids_are_counted_once_with_shared_ownership(self):
        self.f.receipt(stage='prepare', pid='100')
        self.f.receipt(stage='score', utility=True, pid='101')
        total, groups, _, gaps = self.totals()
        self.assertFalse(gaps)
        self.assertEqual(total['native_generation_calls'], 2)
        self.assertEqual({key[3] for key in groups}, {'shared_ancestor'})
        self.assertEqual({key[1] for key in groups}, {'prepare', 'score'})

    def test_duplicate_invocation_identity_not_double_counted(self):
        stem = self.f.receipt()
        for suffix in ('.request.json', '.receipt.json', '.response.json'):
            src = self.f.root / (stem + suffix)
            dest = self.f.root / ((stem + suffix).replace('/runtime/', '/utility/runtime/'))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
        total, _, _, gaps = self.totals()
        self.assertEqual(total['native_generation_calls'], 1)
        self.assertTrue(any('Duplicate meter' in gap['reason'] for gap in gaps))

    def test_operation_child_count_mismatch_is_detected(self):
        self.f.receipt(kind='operation', operation='op', context={'phase': 'evaluating', 'method': 'seed', 'conversation': 'conv'})
        self.f.receipt(operation='op')
        total, _, _, gaps = self.totals()
        self.assertEqual(total['native_generation_calls'], 1)
        self.assertTrue(any('invocation count' in gap['reason'] for gap in gaps))

    def test_f1_exact_prediction_binding_and1540_population(self):
        self.f.predictions()
        result = usage.prediction_summary(self.f.manifest())
        self.assertEqual(result['seed']['official_f1_100'], 50)
        self.assertEqual(result['seed']['logical_reader_input_tokens'], 15400)
        path = self.f.root / 'runs/qwen35_baseline/seed.jsonl'
        rows = path.read_text().splitlines()
        path.write_text('\n'.join(rows[:-1] + [rows[0]]) + '\n')
        with self.assertRaisesRegex(ValueError, 'prediction hash'):
            usage.prediction_summary(self.f.manifest())
        comparison_name = 'runs/recursive_v1/OFFICIAL_COMPARISON.json'
        self.f.update(comparison_name, lambda row: row['prediction_hashes'].update(seed=usage.sha(path)))
        with self.assertRaisesRegex(ValueError, '1540 unique'):
            usage.prediction_summary(self.f.manifest())

    def test_method_seconds_are_native_amortized_and_not_batch_row_duration(self):
        self.f.predictions()
        self.f.receipt(seconds=2)
        self.f.receipt(kind='embedding', seconds=3)
        self.f.receipt(stage='prepare', instance='prepare')
        self.f.receipt(stage='score', instance='score', utility=True)
        self.f.receipt(stage='construct', run='recursive_v1', instance='construct')
        for method in usage.METHODS[1:]:
            run = 'recursive_v1' if method == usage.METHODS[-1] else 'qwen35_baseline'
            self.f.receipt(run=run, instance=method, context={'phase': 'evaluating', 'method': method})
        for method in usage.METHODS:
            run = 'recursive_v1' if method == usage.METHODS[-1] else 'qwen35_baseline'
            self.f.receipt(kind='operation', run=run, instance='coverage_' + method, operation='coverage',
                           context={'phase': 'evaluating', 'method': method, 'conversation': 'conv'}, users=['p'] * usage.N)
        self.f.manifest()
        report = usage.aggregate(self.f.root)
        self.assertEqual(report['methods']['seed']['mean_native_inference_seconds_per_question'], 5 / 1540)
        self.assertEqual(report['methods']['seed']['evaluation_native']['llm_tokens_M'], 5 / 1e6)
        self.assertEqual(report['methods']['seed']['logical_reader_tokens_M'], 16940 / 1e6)

    def test_missing_global_sequence_is_a_measurement_gap(self):
        self.f.receipt()
        self.f.sequences[('qwen35_baseline', 'evaluate', '101', 'abc')] += 1
        self.f.receipt(kind='embedding')
        _, _, _, gaps = self.totals()
        self.assertTrue(any('Missing meter sequence' in gap['reason'] for gap in gaps))

    def complete_cached_fixture(self):
        self.f.predictions()
        self.f.receipt(stage='prepare', instance='prepare')
        self.f.receipt(stage='score', instance='score', utility=True)
        self.f.receipt(stage='construct', run='recursive_v1', instance='construct')
        for method in usage.METHODS:
            run = 'recursive_v1' if method == usage.METHODS[-1] else 'qwen35_baseline'
            self.f.receipt(kind='operation', run=run, instance=method, operation=method,
                           context={'phase': 'evaluating', 'method': method, 'conversation': 'conv'},
                           users=['cached'] * usage.N)
        self.f.manifest()

    def test_fully_observed_cached_evaluation_is_exact_zero_not_missing(self):
        self.complete_cached_fixture()
        report = usage.aggregate(self.f.root)
        self.assertEqual(report['gaps'], [])
        for row in report['methods'].values():
            self.assertEqual(row['evaluation_native']['llm_tokens_M'], 0)
            self.assertEqual(row['mean_native_inference_seconds_per_question'], 0)
        self.assertEqual(report['operations']['explicit_cache_hit_proven'], 0)

    def test_missing_complete_stage_cannot_become_exact_zero(self):
        self.complete_cached_fixture()
        for path in (self.f.root / 'runs/qwen35_baseline/utility').rglob('*'):
            if path.is_file():
                path.unlink()
        self.f.manifest()
        report = usage.aggregate(self.f.root)
        self.assertTrue(any('Required runtime stage' in gap['reason'] for gap in report['gaps']))
        self.assertIsNone(report['totals']['llm_tokens_M'])

    def test_missing_evaluation_inputs_cannot_become_exact_zero(self):
        self.complete_cached_fixture()
        receipt_path = next((self.f.root / 'runs/qwen35_baseline/runtime/evaluate/101/seed').rglob('*.receipt.json'))
        request_path = receipt_path.with_name(receipt_path.name.replace('.receipt.', '.request.'))
        request = usage.read(request_path)
        request['payload']['inputs']['users'].pop()
        digest = self.f.write(request_path.relative_to(self.f.root).as_posix(), request)
        self.f.update(receipt_path.relative_to(self.f.root).as_posix(), lambda row: row.update(request_sha256=digest))
        self.f.manifest()
        report = usage.aggregate(self.f.root)
        self.assertTrue(any('cover1540' in gap['reason'] for gap in report['gaps']))
        self.assertIsNone(report['methods']['seed']['evaluation_native']['llm_tokens_M'])

    def test_profile_precision_mutation_rejected(self):
        stem = self.f.receipt()
        for suffix in ('.request.json', '.receipt.json'):
            self.f.update(stem + suffix, lambda row: row['profile'].update(dtype='bfloat16'))
        self.f.update(stem + '.receipt.json', lambda row: row.update(request_sha256=usage.sha(self.f.root / (stem + '.request.json'))))
        total, _, _, gaps = self.totals()
        self.assertEqual(total['native_generation_calls'], 0)
        self.assertTrue(any('profile differs' in gap['reason'] for gap in gaps))

    def test_recursive_parent_lineage_mutation_rejected(self):
        self.f.predictions()
        self.f.update('runs/recursive_v1/protocol.json', lambda row: row.update(baseline_protocol_sha256='wrong'))
        with self.assertRaisesRegex(ValueError, 'lineage mismatch'):
            usage.prediction_summary(self.f.manifest())

    def test_protocol_lock_binding_uses_original_digest_serializer(self):
        self.f.predictions()
        self.f.update('runs/recursive_v1/protocol.json', lambda row: row.update(extra='changed'))
        with self.assertRaisesRegex(ValueError, 'protocol digest'):
            usage.prediction_summary(self.f.manifest())


if __name__ == '__main__':
    unittest.main()
