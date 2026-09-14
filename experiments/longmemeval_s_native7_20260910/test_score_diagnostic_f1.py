"""Offline correctness and completeness tests for diagnostic token F1."""
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import score_diagnostic_f1 as scorer


class DiagnosticF1Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset.json"
        self.hypotheses = self.root / "predictions.jsonl"
        self.receipt = self.root / "predictions.jsonl.receipt.json"
        self.output = self.root / "scores.json"
        self.ids = [f"q{i}" + ("_abs" if i < 30 else "") for i in range(500)]
        self.records = [{"question_id": qid, "answer": "The red car",
                         "question_type": "small" if i < 10 else "large"}
                        for i, qid in enumerate(self.ids)]
        self.dataset.write_text(json.dumps(self.records), encoding="utf-8")
        self.rows = [{"question_id": qid, "hypothesis": "green car" if i < 100 else "red car"}
                     for i, qid in enumerate(self.ids)]
        self.data_hash = hashlib.sha256(self.dataset.read_bytes()).hexdigest()
        self.write_predictions()
        active = patch.object(scorer, "DATA_SHA256", self.data_hash)
        active.start()
        self.addCleanup(active.stop)

    def write_predictions(self):
        self.hypotheses.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        self.receipt.write_text(json.dumps({
            "schema": "native-seven-official-hypotheses-v1", "scope": "full_canonical_500",
            "expected_count": 500, "expected_question_ids": self.ids, "dataset_sha256": self.data_hash,
            "hypotheses_sha256": hashlib.sha256(self.hypotheses.read_bytes()).hexdigest(),
            "answer_postprocessing": "none", "method": "higmem"}), encoding="utf-8")

    def run_score(self):
        return scorer.score(self.dataset, self.hypotheses, self.output)

    def test_metric_preserves_historical_ascii_multiset_definition(self):
        self.assertEqual(scorer.generic_f1("THE red, car!", "red car"), 1)
        self.assertAlmostEqual(scorer.generic_f1("cat cat cat", "cat cat dog"), 2 / 3)
        self.assertEqual(scorer.generic_f1("running", "run"), 0)
        self.assertEqual(scorer.generic_f1("", ""), 0)
        self.assertEqual(scorer.generic_f1("cafe", "caf\u00e9"), 0)

    def test_question_macro_types_and_abstention_without_network_or_input_mutation(self):
        before = {path: path.read_bytes() for path in (self.dataset, self.hypotheses, self.receipt)}
        with patch.object(socket, "socket", side_effect=AssertionError("Network forbidden")):
            result = self.run_score()
        summary = result["summary"]
        self.assertEqual(summary["overall"]["count"], 500)
        self.assertAlmostEqual(summary["overall"]["question_macro_f1"], 0.9)
        self.assertEqual(summary["by_question_type"]["small"]["count"], 10)
        self.assertEqual(summary["by_question_type"]["small"]["question_macro_f1"], 0.5)
        self.assertEqual(summary["by_answerability"]["abstention"]["count"], 30)
        self.assertEqual(summary["by_answerability"]["abstention"]["question_macro_f1"], 0.5)
        self.assertEqual(summary["by_answerability"]["answerable"]["count"], 470)
        self.assertTrue(summary["abstention_included_in_overall"])
        self.assertEqual([row["question_id"] for row in result["questions"]], self.ids)
        self.assertTrue(all("answer" not in row and "hypothesis" not in row for row in result["questions"]))
        self.assertIsNone(result["official_accuracy"])
        self.assertEqual(result["provenance"]["external_api_calls"], 0)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_rejects_missing_duplicate_unknown_blank_and_annotated_predictions(self):
        for change in (
            lambda rows: rows.pop(),
            lambda rows: rows[0].update(question_id=rows[1]["question_id"]),
            lambda rows: rows[0].update(question_id="outside-population"),
            lambda rows: rows[0].update(hypothesis=" \n"),
            lambda rows: rows[0].update(answer="gold annotation"),
        ):
            original = [dict(row) for row in self.rows]
            change(self.rows)
            self.write_predictions()
            with self.assertRaises(ValueError):
                self.run_score()
            self.assertFalse(self.output.exists())
            self.rows = original

    def test_rejects_wrong_reference_and_changed_or_unreceipted_export(self):
        self.hypotheses.write_text(self.hypotheses.read_text() + " ", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_score()
        self.write_predictions()
        self.dataset.write_text(self.dataset.read_text() + " ", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_score()
        self.dataset.write_text(json.dumps(self.records), encoding="utf-8")
        self.receipt.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_score()
        self.assertFalse(self.output.exists())

    def test_existing_scores_are_preserved(self):
        self.run_score()
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_score()
        self.assertEqual(self.output.read_bytes(), original)

    def test_receipt_method_must_be_one_of_the_seven_native_methods(self):
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        for method in ("", "   ", "other-method", None):
            with self.subTest(method=method):
                self.receipt.write_text(json.dumps({**receipt, "method": method}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.run_score()
                self.assertFalse(self.output.exists())

    def test_receipt_population_and_processing_identity_must_match(self):
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        invalid = {"schema": "other-export", "scope": "pilot12", "expected_count": 499,
                   "expected_question_ids": list(reversed(self.ids)), "dataset_sha256": "0" * 64,
                   "hypotheses_sha256": "0" * 64, "answer_postprocessing": "stripped"}
        for field, value in invalid.items():
            with self.subTest(field=field):
                self.receipt.write_text(json.dumps({**receipt, field: value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.run_score()
                self.assertFalse(self.output.exists())

    def test_recorded_provenance_matches_actual_input_and_scorer_bytes(self):
        expected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (self.dataset, self.hypotheses, self.receipt)}
        result = self.run_score()
        self.assertEqual(result["provenance"]["inputs_sha256"], expected)
        self.assertEqual(result["provenance"]["scorer_sha256"],
                         hashlib.sha256(Path(scorer.__file__).read_bytes()).hexdigest())
        self.assertEqual(result["provenance"]["judge_calls"], 0)
        self.assertFalse(result["provenance"]["generation_artifacts_modified"])
        counts = result["summary"]
        self.assertEqual(sum(group["count"] for group in counts["by_question_type"].values()), 500)
        self.assertEqual(sum(group["count"] for group in counts["by_answerability"].values()), 500)

    def test_input_change_during_scoring_prevents_publishing_scores(self):
        original_f1 = scorer.generic_f1
        calls = 0
        def mutate_once(prediction, gold):
            nonlocal calls
            calls += 1
            if calls == 1:
                self.hypotheses.write_bytes(self.hypotheses.read_bytes() + b" ")
            return original_f1(prediction, gold)
        with patch.object(scorer, "generic_f1", side_effect=mutate_once):
            with self.assertRaisesRegex(ValueError, "Input changed during scoring"):
                self.run_score()
        self.assertFalse(self.output.exists())
    def test_empty_groups_are_explicitly_unavailable(self):
        self.assertEqual(scorer.describe([]), {"count": 0, "question_macro_f1": None,
                                             "question_macro_f1_100": None})



if __name__ == "__main__":
    unittest.main()

