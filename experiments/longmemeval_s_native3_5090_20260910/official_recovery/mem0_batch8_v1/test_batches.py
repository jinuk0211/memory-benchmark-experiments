"""Input preservation and semantic batch receipt regressions; no model calls."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from test_support import completed_fixture, source_fixture

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("batch8_tested", HERE / "runner.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class BatchTests(unittest.TestCase):
    def test_eight_pairs_with_final_singleton_and_session_boundaries(self):
        source = source_fixture((1, 15, 16, 17, 32, 33))
        before = copy.deepcopy(source)
        batches = list(r.session_batches(source))
        self.assertEqual([len(messages) for _, messages in batches], [1, 15, 16, 16, 1, 16, 16, 16, 16, 1])
        self.assertEqual([m["batch_index"] for m, _ in batches], list(range(10)))
        for index, session in enumerate(source):
            chosen = [(m, turns) for m, turns in batches if m["session_index"] == index]
            self.assertEqual([turn for _, turns in chosen for turn in turns], session["turns"])
            for metadata, turns in chosen:
                self.assertEqual(turns, session["turns"][metadata["turn_start"]:metadata["turn_stop"]])
                self.assertEqual(metadata["turn_start"] % 16, 0)
                self.assertLessEqual(len(turns), 16)
                self.assertEqual(metadata["session_id"], session["session_id"])
                self.assertEqual(metadata["timestamp"], 1789120800)
        self.assertEqual(source, before)

    def test_invalid_empty_session_nontext_and_role_input_rejected(self):
        for turns in ([], [{"role": "user", "content": None}], [{"role": "tool", "content": "text"}]):
            with self.subTest(turns=turns), self.assertRaises(ValueError):
                list(r.session_batches([{**source_fixture((1,))[0], "turns": turns}]))

    def test_empty_and_whitespace_strings_are_preserved_at_original_positions(self):
        source = source_fixture((17,))
        source[0]["turns"][0]["content"] = ""
        source[0]["turns"][15]["content"] = " \n\t"
        source[0]["turns"][16]["content"] = ""
        before = copy.deepcopy(source)
        batches = list(r.session_batches(source))
        self.assertEqual([len(messages) for _, messages in batches], [16, 1])
        self.assertEqual([turn for _, messages in batches for turn in messages], before[0]["turns"])
        self.assertEqual(source, before)

    def test_native_metadata_mutation_cannot_change_input_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            source = source_fixture((17, 1))
            before = copy.deepcopy(source)
            calls = []
            observer = r.NativeObservations(attempt)
            def add(messages, **kwargs):
                calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
                kwargs["metadata"].update(user_id=kwargs["user_id"], native_mutation=True)
                observer.call += 2
                return []
            count = r.ingest_batches(SimpleNamespace(add=add), source, "same-question-user", attempt / "construction_progress.json", observer)
            self.assertEqual(count, 3)
            self.assertEqual([len(messages) for messages, _ in calls], [16, 1, 1])
            self.assertTrue(all(kw["infer"] is True and kw["user_id"] == "same-question-user" for _, kw in calls))
            journal = [json.loads(line) for line in (attempt / "batch_calls.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["chat_calls"] == 2 and "native_mutation" not in row["metadata"] for row in journal))
            self.assertEqual(source, before)
            self.assertEqual(r.harness.read_json(attempt / "construction_progress.json")["native_result"], [])

    def test_failed_add_never_records_success_or_skips_to_next_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            observer = r.NativeObservations(attempt)
            calls = []
            def add(messages, **kwargs):
                calls.append(messages)
                if len(calls) == 2:
                    raise TimeoutError("synthetic transport")
                observer.call += 2
                return []
            with self.assertRaises(TimeoutError):
                r.ingest_batches(SimpleNamespace(add=add), source_fixture((33,)), "user", attempt / "construction_progress.json", observer)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len((attempt / "batch_calls.jsonl").read_text().splitlines()), 1)
            progress = r.harness.read_json(attempt / "construction_progress.json")
            self.assertEqual((progress["status"], progress["completed_batches"]), ("adding", 1))


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name="run"):
        return completed_fixture(r, self.root / name / "histories/qid/attempt_0001", source=source_fixture())

    def test_full_semantic_receipt_and_same_version_cache(self):
        attempt, identity = self.fixture()
        r.finalize_attempt(attempt, identity)
        self.assertEqual(r.verified_prediction(attempt.parent, identity)["implementation"], r.VERSION)
        self.assertTrue(r.harness.read_json(attempt / "completion.json")["created_after_worker_exit"])

    def test_corrupted_journals_fail_finalize_and_even_resealed_cache(self):
        changes = {
            "drop": lambda rows: rows.pop(),
            "duplicate": lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
            "reorder": lambda rows: rows.reverse(),
            "range": lambda rows: rows[0]["metadata"].update(turn_stop=15),
            "session": lambda rows: rows[0]["metadata"].update(session_id="different"),
            "input_hash": lambda rows: rows[0].update(messages_sha256="0" * 64),
            "source_hash": lambda rows: rows[0].update(source_sha256="0" * 64),
            "boolean_index": lambda rows: rows[0].update(batch_index=False),
            "boolean_meta": lambda rows: rows[0]["metadata"].update(session_index=False),
            "boolean_calls": lambda rows: rows[0].update(chat_calls=True),
            "wrong_calls": lambda rows: rows[0].update(chat_calls=3),
            "wrong_pairs": lambda rows: rows[0].update(source_pairs=7),
            "nonfinite": lambda rows: rows[0].update(seconds=float("nan")),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                attempt, identity = self.fixture(name)
                r.finalize_attempt(attempt, identity)
                receipt = r.harness.read_json(attempt / "completion.json")
                rows = [json.loads(line) for line in (attempt / "batch_calls.jsonl").read_text().splitlines()]
                change(rows)
                (attempt / "batch_calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError):
                    r.finalize_attempt(attempt, identity)
                r.harness.save_json(attempt / "completion.json", {**receipt, "files_sha256": r.attempt_files(attempt)})
                with self.assertRaises(ValueError):
                    r.verified_prediction(attempt.parent, identity)

    def test_partial_or_missing_journal_and_wrong_final_totals_rejected(self):
        cases = ["partial", "malformed", "missing", "batches", "sessions", "turns", "pairs", "progress", "progress_meta"]
        for name in cases:
            with self.subTest(name=name):
                attempt, identity = self.fixture(name)
                if name == "partial":
                    path = attempt / "batch_calls.jsonl"
                    path.write_text(path.read_text().rstrip("\n"))
                elif name == "malformed":
                    (attempt / "batch_calls.jsonl").write_text("{invalid}\n")
                elif name == "missing":
                    (attempt / "batch_calls.jsonl").unlink()
                elif name.startswith("progress"):
                    path = attempt / "construction_progress.json"
                    value = r.harness.read_json(path)
                    if name == "progress":
                        value["completed_batches"] = True
                    else:
                        value["metadata"]["session_index"] = True
                    r.harness.save_json(path, value)
                else:
                    path = attempt / "build_complete.json"
                    r.harness.save_json(path, {**r.harness.read_json(path), name: True})
                with self.assertRaises(ValueError):
                    r.finalize_attempt(attempt, identity)

    def test_response_attribution_and_duplicate_sequence_rejected(self):
        for name in ("wrong_batch", "wrong_phase", "duplicate"):
            attempt, identity = self.fixture(name)
            with gzip.open(attempt / "memory_responses.jsonl.gz", "rt") as stream:
                events = [json.loads(line) for line in stream]
            if name == "wrong_batch":
                events[0]["batch_index"] = 1
            elif name == "wrong_phase":
                events[0]["phase"] = "update"
            else:
                events[1] = copy.deepcopy(events[0])
            with gzip.open(attempt / "memory_responses.jsonl.gz", "wt") as stream:
                stream.write("".join(json.dumps(event) + "\n" for event in events))
            with self.assertRaises(ValueError):
                r.finalize_attempt(attempt, identity)

    def test_v4_identity_and_mutated_source_or_prediction_never_reused(self):
        for name in ("v4", "source", "query", "prediction"):
            attempt, identity = self.fixture(name)
            r.finalize_attempt(attempt, identity)
            receipt = r.harness.read_json(attempt / "completion.json")
            if name == "v4":
                path = attempt / "worker.json"
                worker = r.harness.read_json(path)
                worker["identity"]["protocol_sha256"] = "old-pairs-v4"
                r.harness.save_json(path, worker)
            elif name == "source":
                path = attempt / "source.json"
                value = r.harness.read_json(path)
                value[0]["turns"][0]["content"] += "changed"
                r.harness.save_json(path, value)
            elif name == "query":
                path = attempt / "query.json"
                r.harness.save_json(path, {**r.harness.read_json(path), "question": "changed"})
            else:
                path = attempt / "prediction.json"
                r.harness.save_json(path, {**r.harness.read_json(path), "implementation": "mem0_native_pairs_v4"})
            r.harness.save_json(attempt / "completion.json", {**receipt, "files_sha256": r.attempt_files(attempt)})
            with self.assertRaises(ValueError):
                r.verified_prediction(attempt.parent, identity)

    def test_failed_attempt_preserved_and_next_attempt_distinct(self):
        attempt, identity = self.fixture()
        r.harness.save_json(attempt / "failure.json", {"reason": "synthetic transport"})
        before = {p: p.read_bytes() for p in attempt.rglob("*") if p.is_file()}
        self.assertIsNone(r.verified_prediction(attempt.parent, identity))
        following = r.harness.next_attempt(attempt.parent)
        self.assertEqual(following.name, "attempt_0002")
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with self.assertRaises(ValueError):
            r.finalize_attempt(attempt, identity)


if __name__ == "__main__":
    unittest.main()
