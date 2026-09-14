"""Focused CPU checks for exact reader replay and source manifest integrity."""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_reader_transfer as replay
from test_run_transfer import FakeRuntime


class ReaderReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def source_fixture(self) -> None:
        records = [
            {"id": f"conv:{index}", "conv_id": "conv", "qa_index": index,
             "category": 2, "split": "question_audit"}
            for index in range(100)
        ]
        protocol = {"methods": list(replay.METHODS), "audit_manifest": {"records": records}}
        replay.core.save(self.root / "protocol.json", protocol)
        replay.core.save(self.root / "memory_selection_locked.json", {
            "protocol_sha256": replay.core.digest(protocol),
            "construction_used_benchmark_questions_or_answers": False,
            "methods": list(replay.METHODS),
        })
        for method in replay.METHODS:
            replay.core.save(self.root / f"{method}_question_audit_items.json", [
                {**record, "candidate": method, "context": f" {method}\nuser: my degree is history.\n",
                 "question": "What is my degree?", "gold": "history",
                 "prediction": "source prediction", "official_f1": 0.0,
                 "read_tokens": 7, "memory_tokens": 100}
                for record in records
            ])

    def test_source_conditions_require_exact_manifest_coverage_and_same_questions(self) -> None:
        self.source_fixture()
        sources, _, _, paths = replay.load_source_artifacts(self.root)
        self.assertTrue(all(len(rows) == 100 for rows in sources.values()))
        path = paths[replay.METHODS[1]]
        original = replay.read(path)
        for bad in (original[:-1], original[:-1] + [original[0]]):
            replay.core.save(path, bad)
            with self.assertRaisesRegex(ValueError, "coverage"):
                replay.load_source_artifacts(self.root)
        altered = copy.deepcopy(original)
        altered[0]["gold"] = "DIFFERENT"
        replay.core.save(path, altered)
        with self.assertRaisesRegex(ValueError, "QA differs"):
            replay.load_source_artifacts(self.root)

    def test_source_protocol_memory_lock_and_replay_lock_reject_drift(self) -> None:
        self.source_fixture()
        path = self.root / "protocol.json"
        protocol = replay.read(path)
        replay.core.save(path, {**protocol, "changed": True})
        with self.assertRaisesRegex(ValueError, "memory lock"):
            replay.load_source_artifacts(self.root)
        output = self.root / "new_lock.json"
        replay.locked(output, {"source": "frozen"})
        replay.locked(output, {"source": "frozen"})
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            replay.locked(output, {"source": "mutated"})
        self.assertEqual(replay.read(output), {"source": "frozen"})

    def test_output_cannot_be_inside_immutable_source_directory(self) -> None:
        for output in (self.root, self.root / "nested-output"):
            with self.subTest(output=output), patch("sys.argv", [
                "run_reader_transfer.py", "--source-run", str(self.root),
                "--out", str(output), "--environment", str(self.root / "unused.json"),
            ]):
                with self.assertRaisesRegex(ValueError, "outside the immutable source"):
                    replay.main()

    def test_prompt_preserves_exact_context_and_excludes_gold_metadata(self) -> None:
        row = {
            "context": "  dated memory\nuser: history\n ",
            "question": "What is my degree?",
            "gold": "GOLD_SECRET", "category": "TYPE_SECRET", "id": "ID_SECRET",
        }
        expected = "Conversation memory:\n  dated memory\nuser: history\n \n\nQuestion: What is my degree?\nAnswer:"
        self.assertEqual(replay.user_prompt(row), expected)
        changed = {**row, "gold": "different", "category": 4, "id": "changed"}
        self.assertEqual(replay.user_prompt(row), replay.user_prompt(changed))
        self.assertNotIn("SECRET", replay.user_prompt(row))

    def test_missing_reader_generations_fail_without_dropping_ids(self) -> None:
        self.source_fixture()
        sources, _, _, _ = replay.load_source_artifacts(self.root)
        with self.assertRaisesRegex(ValueError, "Missing generation outputs"):
            replay.evaluate_rows(
                FakeRuntime(self.root / "cache", output_mode="missing"),
                sources[replay.METHODS[0]], replay.METHODS[0],
            )

    @unittest.skipUnless(importlib.util.find_spec("nltk"), "Frozen LoCoMo metric requires nltk")
    def test_generated_rows_keep_source_contexts_ids_and_cache_token_provenance(self) -> None:
        self.source_fixture()
        sources, _, _, _ = replay.load_source_artifacts(self.root)
        method = replay.METHODS[0]
        runtime = FakeRuntime(self.root / "runtime")
        rows = replay.evaluate_rows(runtime, sources[method], method)
        self.assertEqual(len(rows), 100)
        self.assertEqual([r["question_id"] for r in rows], [r["id"] for r in sources[method]])
        self.assertEqual(runtime.calls[0][1], [replay.user_prompt(r) for r in sources[method]])
        for source, row in zip(sources[method], rows):
            self.assertEqual(row["context"], source["context"])
            self.assertEqual(row["official_f1"], 1)
            self.assertEqual(row["source_reader_official_f1"], 0)
            self.assertEqual(row["source_reader_prediction"], "source prediction")
            self.assertEqual(row["source_context_sha256"], row["replayed_context_sha256"])
            self.assertEqual(row["user_prompt_sha256"], replay.text_sha256(replay.user_prompt(source)))
            self.assertEqual(row["source_context_tokens"], 7)
            self.assertEqual(row["source_memory_tokens"], 100)
            self.assertEqual(row["target_context_tokens"], runtime.ntok(source["context"]))
            self.assertGreater(row["input_tokens"], 0)
            self.assertEqual(row["output_tokens"], 1)
            self.assertEqual(len(row["generation_cache_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

