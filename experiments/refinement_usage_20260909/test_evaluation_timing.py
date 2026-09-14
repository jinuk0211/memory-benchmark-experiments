"""Local timing-boundary and failure tests; no experiment imports or GPU calls."""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import evaluation_timing as timing


class Clock:
    def __init__(self):
        self.now = 0

    def read(self):
        return self.now

    def advance(self, nanoseconds):
        self.now += nanoseconds


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = {'source_function_sha256': 'source-pin', 'launcher_sha256': 'launcher-pin',
                         'cache_lifecycle': 'reuse_existing_cache_without_reset'}
        self.recorder = timing.TimingRecorder(self.root, self.metadata)
        self.rt = SimpleNamespace(args=SimpleNamespace(out=Path('relative/existing-run'), model='Qwen/Qwen3.5-9B',
                                  embed_model='sentence-transformers/all-MiniLM-L6-v2', seed=20260907),
                                  model_meta={'revision': 'pinned-model'}, embed_meta={'revision': 'pinned-embedding'})
        self.units = [{'text': 'memory'}]
        self.sample = {'sample_id': 'conv-test'}
        self.rows = [{'question_id': 'q-1', 'prediction': '서울', 'generation_batch_seconds': 77.0},
                     {'question_id': 'q-2', 'prediction': 'answer', 'generation_batch_seconds': 77.0}]

    def invoke(self, original):
        return self.recorder.wrap(original)(self.rt, self.units, self.sample, 'refined', 'locomo')

    def receipts(self, filename):
        return [json.loads(path.read_text(encoding='utf-8')) for path in self.root.rglob(filename)]

    def test_timer_excludes_metadata_digest_and_sidecar_io(self):
        clock = Clock()
        original_write = self.recorder._write
        original_serialize = timing.canonical_bytes

        def slow_write(path, record):
            clock.advance(11_000_000_000)
            return original_write(path, record)

        def slow_serialize(value):
            clock.advance(13_000_000_000)
            return original_serialize(value)

        def original(rt, units, sample, method, dataset):
            self.assertEqual(len(self.receipts('started.json')), 1)
            self.assertEqual(self.receipts('completed.json'), [])
            clock.advance(2_500_000_000)
            return self.rows

        with patch.object(self.recorder, '_write', side_effect=slow_write), \
                patch.object(timing, 'canonical_bytes', side_effect=slow_serialize), \
                patch.object(timing.time, 'perf_counter_ns', side_effect=clock.read), \
                patch.object(timing.time, 'time', side_effect=AssertionError('Wall clock must not be used')):
            result = self.invoke(original)
        self.assertIs(result, self.rows)
        self.assertGreater(clock.now, 30_000_000_000)
        self.assertEqual(self.receipts('completed.json')[0]['elapsed_seconds'], 2.5)
        self.assertEqual(self.rows[0]['generation_batch_seconds'], 77.0)

    def test_exact_args_keywords_result_identity_and_signature_preserved(self):
        captured = []

        def contract(rt, units, sample, method, dataset):
            pass

        def original(*args, **kwargs):
            captured.append((args, kwargs))
            return self.rows

        original.__signature__ = inspect.signature(contract)
        wrapped = self.recorder.wrap(original)
        kwargs = {'sample': self.sample, 'method': 'refined', 'dataset': 'locomo'}
        returned = wrapped(self.rt, self.units, **kwargs)
        self.assertIs(returned, self.rows)
        self.assertIs(captured[0][0][0], self.rt)
        self.assertIs(captured[0][0][1], self.units)
        self.assertEqual(captured[0][1], kwargs)
        self.assertIs(captured[0][1]['sample'], self.sample)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(original))

    def test_completed_receipt_links_exact_started_bytes_and_returned_rows(self):
        def original(rt, units, sample, method, dataset):
            return self.rows

        self.invoke(original)
        started_path = next(self.root.rglob('started.json'))
        completed = self.receipts('completed.json')[0]
        self.assertEqual(completed['status'], 'complete')
        self.assertEqual(completed['started_sha256'], hashlib.sha256(started_path.read_bytes()).hexdigest())
        canonical = json.dumps(self.rows, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.assertEqual(completed['returned_rows_sha256'], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(completed['question_count'], 2)
        self.assertEqual(completed['question_ids'], ['q-1', 'q-2'])
        self.assertEqual(completed['conversation_id'], 'conv-test')
        self.assertEqual(completed['runtime']['out'], str(self.rt.args.out))
        self.assertEqual(completed['runtime']['seed'], 20260907)
        self.assertEqual(completed['metadata'], self.metadata)
        self.assertEqual(completed['cache_lifecycle'], self.metadata['cache_lifecycle'])
        self.assertNotIn('elapsed_seconds', self.receipts('started.json')[0])

    def test_error_receipt_never_claims_completion_and_does_not_copy_message(self):
        failure = ValueError('private failure details must not be persisted')

        def original(rt, units, sample, method, dataset):
            raise failure

        with patch.object(timing.time, 'perf_counter_ns', side_effect=[100_000_000, 600_000_000]):
            with self.assertRaises(ValueError) as caught:
                self.invoke(original)
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.receipts('completed.json'), [])
        error = self.receipts('error.json')[0]
        self.assertEqual(error['status'], 'error')
        self.assertEqual(error['exception_type'], 'ValueError')
        self.assertEqual(error['elapsed_seconds'], 0.5)
        self.assertNotIn('question_count', error)
        self.assertNotIn('returned_rows_sha256', error)
        self.assertNotIn(str(failure), json.dumps(error))

    def test_error_receipt_write_failure_preserves_original_exception(self):
        failure = RuntimeError('original')
        original_write = self.recorder._write

        def fail_error_write(path, record):
            if path.name == 'error.json':
                raise OSError('sidecar disk failure')
            return original_write(path, record)

        def original(rt, units, sample, method, dataset):
            raise failure

        with patch.object(self.recorder, '_write', side_effect=fail_error_write):
            with self.assertRaises(RuntimeError) as caught:
                self.invoke(original)
        self.assertIs(caught.exception, failure)
        self.assertEqual(caught.exception.__notes__, [
            'Timing error receipt write failed (OSError); started receipt remains incomplete'])
        self.assertNotIn('sidecar disk failure', str(caught.exception.__notes__))
        self.assertEqual(len(self.receipts('started.json')), 1)
        self.assertEqual(self.receipts('error.json'), [])
        self.assertEqual(self.receipts('completed.json'), [])

    def test_interrupt_remains_original_and_is_not_complete(self):
        interruption = KeyboardInterrupt()

        def original(rt, units, sample, method, dataset):
            raise interruption

        with self.assertRaises(KeyboardInterrupt) as caught:
            self.invoke(original)
        self.assertIs(caught.exception, interruption)
        self.assertEqual(self.receipts('error.json')[0]['exception_type'], 'KeyboardInterrupt')
        self.assertEqual(self.receipts('completed.json'), [])

    def test_completion_write_failure_retains_incomplete_started_receipt(self):
        original_write = self.recorder._write
        called = []

        def fail_completion(path, record):
            if path.name == 'completed.json':
                raise OSError('disk unavailable')
            return original_write(path, record)

        def original(rt, units, sample, method, dataset):
            called.append(True)
            return self.rows

        with patch.object(self.recorder, '_write', side_effect=fail_completion):
            with self.assertRaises(OSError):
                self.invoke(original)
        self.assertEqual(called, [True])
        self.assertEqual(len(self.receipts('started.json')), 1)
        self.assertEqual(self.receipts('completed.json'), [])
        self.assertEqual(self.receipts('error.json'), [])

    def test_partial_atomic_write_is_not_published_as_completed(self):
        original_write = self.recorder._write

        def corrupt_partial(path, record):
            if path.name == 'completed.json':
                path.with_suffix('.tmp').write_bytes(b'{partial')
                raise OSError('write interrupted')
            return original_write(path, record)

        def original(rt, units, sample, method, dataset):
            return self.rows

        with patch.object(self.recorder, '_write', side_effect=corrupt_partial):
            with self.assertRaises(OSError):
                self.invoke(original)
        self.assertEqual(len(list(self.root.rglob('completed.tmp'))), 1)
        self.assertEqual(self.receipts('completed.json'), [])
        self.assertEqual(len(self.receipts('started.json')), 1)

    def test_start_write_failure_does_not_run_unrecorded_evaluation(self):
        called = []

        def original(rt, units, sample, method, dataset):
            called.append(True)
            return self.rows

        with patch.object(self.recorder, '_write', side_effect=OSError('cannot record start')):
            with self.assertRaises(OSError):
                self.invoke(original)
        self.assertEqual(called, [])
        self.assertEqual(self.receipts('completed.json'), [])

    def test_retries_keep_first_error_and_new_completion_in_distinct_directories(self):
        attempts = []

        def original(rt, units, sample, method, dataset):
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError('first attempt')
            return self.rows

        wrapped = self.recorder.wrap(original)
        with self.assertRaises(RuntimeError):
            wrapped(self.rt, self.units, self.sample, 'refined', 'locomo')
        first_path = next(self.root.rglob('error.json'))
        first_bytes = first_path.read_bytes()
        self.assertIs(wrapped(self.rt, self.units, self.sample, 'refined', 'locomo'), self.rows)
        started = self.receipts('started.json')
        self.assertEqual(len(started), 2)
        self.assertEqual(len({row['invocation_id'] for row in started}), 2)
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertNotEqual(first_path.parent, next(self.root.rglob('completed.json')).parent)

    def test_new_recorder_has_fresh_session_without_erasing_old_results(self):
        other = timing.TimingRecorder(self.root, self.metadata)
        self.assertNotEqual(other.root, self.recorder.root)
        self.assertTrue(other.root.is_dir())
        self.assertTrue(self.recorder.root.is_dir())

    def test_uncanonical_return_value_cannot_receive_complete_receipt(self):
        def original(rt, units, sample, method, dataset):
            return [{'question_id': 'q-1', 'score': float('nan')}]

        with self.assertRaises(ValueError):
            self.invoke(original)
        self.assertEqual(len(self.receipts('started.json')), 1)
        self.assertEqual(self.receipts('completed.json'), [])


if __name__ == '__main__':
    unittest.main()
