"""CPU tests of final-answer attribution using actual local TimingRecorder receipts."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from evaluation_timing import TimingRecorder
import aggregate_evaluation_timing as report


class TimingReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sidecars = self.root / 'sidecars'
        self.predictions = self.root / 'answers.jsonl'
        self.method = 'recursive_fixed_views_v2'
        self.rows = [self.row(cid, index) for cid in ('a', 'b') for index in range(2)]
        self.metadata = {'runner': 'relative/source/run_recursive_v2.py',
                         'evaluator_source': 'relative/source/run_transfer.py',
                         'runner_sha256': '1' * 64, 'evaluator_source_sha256': '2' * 64,
                         'evaluator_function_sha256': '3' * 64, 'launcher_sha256': '4' * 64,
                         'recorder_sha256': '5' * 64, 'cache_lifecycle': 'retained_cache'}
        self.recorder = TimingRecorder(self.sidecars, self.metadata)
        self.rt = SimpleNamespace(args=SimpleNamespace(out=Path('runs/relative_v2'), seed=20260907,
                                  model='Qwen/Qwen3.5-9B', embed_model='MiniLM'),
                                  model_meta={'revision': 'model'}, embed_meta={'revision': 'embed'})
        self.save_final()

    def row(self, cid, index):
        return {'question_id': f'{cid}:{index}', 'conversation_id': cid, 'method': self.method,
                'prediction': '서울', 'hypothesis': '서울', 'official_f1': 0.5,
                'generation_batch_seconds': 111.123, 'input_tokens': 10, 'output_tokens': 2}

    def save_final(self):
        self.predictions.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in self.rows), encoding='utf-8')

    def call(self, cid, seconds, *, error=False, incomplete=False, rows=None, recorder=None):
        recorder = recorder or self.recorder
        returned = rows if rows is not None else [row for row in self.rows if row['conversation_id'] == cid]

        def original(rt, units, sample, method, dataset):
            if error:
                raise RuntimeError('failure details not logged')
            return returned

        before = set(self.sidecars.glob('*/*'))
        original_write = recorder._write

        def write(path, record):
            if incomplete and path.name == 'completed.json':
                raise OSError('completion receipt unavailable')
            return original_write(path, record)

        with patch('evaluation_timing.time.perf_counter_ns', side_effect=[1_000_000_000, 1_000_000_000 + int(seconds * 1e9)]), \
                patch.object(recorder, '_write', side_effect=write):
            try:
                recorder.wrap(original)(self.rt, [], {'sample_id': cid}, self.method, 'locomo')
            except (RuntimeError, OSError):
                if not (error or incomplete):
                    raise
        return (set(self.sidecars.glob('*/*')) - before).pop()

    def aggregate(self):
        return report.aggregate_timing(self.predictions, self.sidecars, self.method)

    def modify(self, path, change):
        value = json.loads(path.read_text(encoding='utf-8'))
        change(value)
        path.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')

    def complete(self):
        self.call('a', 2)
        self.call('b', 4)

    def test_valid_full_population_and_receipt_order_not_final_file_order(self):
        self.rows = [self.row(cid, index) for cid in ('a', 'b') for index in range(770)]
        self.rows.reverse()
        self.save_final()
        self.call('a', 2, rows=list(reversed([row for row in self.rows if row['conversation_id'] == 'a'])))
        self.call('b', 4)
        value = self.aggregate()
        self.assertEqual(value['questions_in_final_file'], 1540)
        self.assertTrue(value['coverage_complete_and_unambiguous'])
        self.assertEqual(value['mean_final_evaluation_seconds_per_question'], 6 / 1540)
        self.assertEqual(value['observed_all_terminal_attempt_seconds_lower_bound'], 6)
        self.assertEqual(value['final_answers_sha256'], hashlib.sha256(self.predictions.read_bytes()).hexdigest())

    def test_missing_receipts_mean_is_unknown_not_zero(self):
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertEqual(value['observed_all_terminal_attempt_seconds_lower_bound'], 0)
        self.assertFalse(value['coverage_complete_and_unambiguous'])

    def test_missing_one_conversation_coverage(self):
        self.call('a', 2)
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertEqual(value['observed_successful_attempt_seconds'], 2)
        self.assertTrue(any('Missing completed' in issue['reason'] for issue in value['issues']))

    def test_duplicate_successes_are_not_selected_by_mtime(self):
        first = self.call('a', 2)
        self.call('a', 200)
        self.call('b', 4)
        os.utime(first / 'completed.json', (9999999999, 9999999999))
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertEqual(value['observed_successful_attempt_seconds'], 206)
        self.assertEqual(len(value['attempts']), 3)
        self.assertTrue(any('Overlapping' in issue['reason'] for issue in value['issues']))

    def test_failed_retry_is_separate_observed_cost(self):
        self.call('a', 3, error=True)
        self.complete()
        value = self.aggregate()
        self.assertEqual(value['mean_final_evaluation_seconds_per_question'], 1.5)
        self.assertEqual(value['observed_failed_attempt_seconds'], 3)
        self.assertEqual(value['observed_successful_attempt_seconds'], 6)
        self.assertEqual(value['observed_all_terminal_attempt_seconds_lower_bound'], 9)
        self.assertEqual([a['status'] for a in value['attempts']].count('error'), 1)

    def test_started_only_retry_keeps_mean_unknown_even_after_complete_coverage(self):
        self.call('a', 100, incomplete=True)
        self.complete()
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertEqual(value['attempts_with_unknown_elapsed'], 1)
        self.assertEqual(value['observed_all_terminal_attempt_seconds_lower_bound'], 6)

    def test_unpersisted_success_has_observed_cost_but_cannot_match_final_rows(self):
        changed = [dict(row, official_f1=0.9) for row in self.rows if row['conversation_id'] == 'a']
        self.call('a', 7, rows=changed)
        self.call('b', 4)
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertEqual(value['observed_successful_attempt_seconds'], 11)
        self.assertTrue(any('Returned-row digest' in issue['reason'] for issue in value['issues']))

    def test_terminal_link_tamper_rejected(self):
        directory = self.call('a', 2)
        self.call('b', 4)
        self.modify(directory / 'completed.json', lambda row: row.update(started_sha256='tampered'))
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertTrue(any('linkage' in issue['reason'] for issue in value['issues']))

    def test_changed_original_batch_timing_field_breaks_row_digest(self):
        self.complete()
        self.rows[0]['generation_batch_seconds'] += 1
        self.save_final()
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertTrue(any('Returned-row digest' in issue['reason'] for issue in value['issues']))

    def test_negative_and_nonfinite_elapsed_are_not_counted(self):
        directory = self.call('a', 2)
        self.call('b', 4)
        for invalid in (-1, float('inf'), True):
            with self.subTest(elapsed=invalid):
                self.modify(directory / 'completed.json', lambda row: row.update(elapsed_seconds=invalid))
                value = self.aggregate()
                self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
                self.assertEqual(value['observed_successful_attempt_seconds'], 4)

    def test_identity_must_match_session_and_invocation_directory(self):
        directory = self.call('a', 2)
        self.call('b', 4)
        self.modify(directory / 'started.json', lambda row: row.update(session_id='0' * 32))
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertTrue(any('identity' in issue['reason'] for issue in value['issues']))

    def test_duplicate_returned_ids_rejected(self):
        directory = self.call('a', 2)
        self.call('b', 4)
        self.modify(directory / 'completed.json', lambda row: row.update(question_ids=['a:0', 'a:0']))
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertTrue(any('duplicate returned' in issue['reason'] for issue in value['issues']))

    def test_final_duplicate_ids_fail_closed(self):
        self.rows.append(dict(self.rows[0]))
        self.save_final()
        with self.assertRaisesRegex(ValueError, 'duplicate/invalid'):
            self.aggregate()

    def test_cross_session_metadata_change_is_not_silently_combined(self):
        self.call('a', 2)
        other = TimingRecorder(self.sidecars, {**self.metadata, 'launcher_sha256': '6' * 64})
        self.call('b', 4, recorder=other)
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertTrue(any('Inconsistent' in issue['reason'] for issue in value['issues']))
        self.assertEqual(value['observed_successful_attempt_seconds'], 6)

    def test_orphan_terminal_outside_invocation_directory_prevents_complete_mean(self):
        self.complete()
        (self.sidecars / 'completed.json').write_text('{}', encoding='utf-8')
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertEqual(value['observed_successful_attempt_seconds'], 6)
        self.assertTrue(any('Orphan' in issue['reason'] for issue in value['issues']))

    def test_conflicting_terminal_files_never_choose_one(self):
        directory = self.call('a', 2)
        self.call('b', 4)
        (directory / 'error.json').write_bytes((directory / 'completed.json').read_bytes())
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertEqual(value['observed_successful_attempt_seconds'], 4)
        self.assertTrue(any('Conflicting' in issue['reason'] for issue in value['issues']))

    def test_other_method_start_cannot_hide_selected_method_terminal(self):
        self.complete()
        directory = self.call('a', 7)
        self.modify(directory / 'started.json', lambda row: row.update(method='different_method'))
        value = self.aggregate()
        self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
        self.assertEqual(value['ignored_other_method_attempts'], 0)
        self.assertTrue(any('method mismatch' in issue['reason'] for issue in value['issues']))

    def test_empty_metadata_or_invalid_runtime_fields_cannot_certify_timing(self):
        directory = self.call('a', 2)
        self.call('b', 4)
        started_path, terminal_path = directory / 'started.json', directory / 'completed.json'
        original_started = started_path.read_bytes()
        original_terminal = terminal_path.read_bytes()
        mutations = [lambda row: row.update(metadata={}), lambda row: row.update(runtime={}),
                     lambda row: row['runtime'].update(seed=True),
                     lambda row: row['runtime'].update(model=''),
                     lambda row: row['metadata'].update(evaluator_source=''),
                     lambda row: row['metadata'].update(recorder_sha256='invalid')]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                started_path.write_bytes(original_started)
                terminal_path.write_bytes(original_terminal)
                self.modify(started_path, mutate)
                self.modify(terminal_path, mutate)
                self.modify(terminal_path, lambda row: row.update(started_sha256=hashlib.sha256(started_path.read_bytes()).hexdigest()))
                value = self.aggregate()
                self.assertIsNone(value['mean_final_evaluation_seconds_per_question'])
                self.assertTrue(any('metadata' in issue['reason'] for issue in value['issues']))

    def test_relative_paths_do_not_reinterpret_runtime_out_as_local_evidence_path(self):
        self.complete()
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            value = report.aggregate_timing(Path('answers.jsonl'), Path('sidecars'), self.method)
        finally:
            os.chdir(previous)
        self.assertEqual(value['mean_final_evaluation_seconds_per_question'], 1.5)
        self.assertEqual(value['metadata']['runtime']['out'], str(self.rt.args.out))


if __name__ == '__main__':
    unittest.main()
