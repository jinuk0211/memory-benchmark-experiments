"""Synthetic contract checks only; no real answers or judge API requests."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import official_eval_bridge as bridge

VENDOR = Path(__file__).resolve().parents[1] / "official_longmemeval"


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.references = [{"question_id": f"q{i}", "question_type": kind}
                           for i, kind in enumerate(bridge.QUESTION_TYPES)]
        self.references.append({"question_id": "q0_abs", "question_type": bridge.QUESTION_TYPES[0]})
        self.ids = [row["question_id"] for row in self.references]
        self.args = SimpleNamespace(
            vendor=VENDOR, reference=self.root / "reference.json",
            expected_ids=self.root / "expected.json", partition=None,
            predictions=self.root / "predictions.jsonl", protocol=self.root / "protocol.json",
            hypotheses=self.root / "hypotheses.jsonl", method="s_parent_single_2000",
            results=self.root / "hypotheses.jsonl.eval-results-gpt-4o",
            report=self.root / "report.json",
        )
        self.write_json(self.args.reference, self.references)
        canonical_patch = patch.object(bridge, "CANONICAL_SHA256", bridge.sha256(self.args.reference))
        canonical_patch.start()
        self.addCleanup(canonical_patch.stop)
        self.write_json(self.args.expected_ids, self.ids)
        self.protocol = {
            "dataset_sha256": bridge.CANONICAL_SHA256,
            "selection": {"selected_ids": self.ids},
            "methods": [self.args.method], "evaluation_methods": [self.args.method],
            "config": {"model": "Qwen/Qwen3.5-9B", "embed_model": "sentence-transformers/all-MiniLM-L6-v2"},
            "model": {"revision": "model-revision"}, "embedding": {"revision": "embedding-revision"},
            "read_budget": 2048, "extra_storage_budget": 2000, "max_answer_tokens": 96,
        }
        self.write_json(self.args.protocol, self.protocol)
        self.predictions = [{"question_id": qid, "method": self.args.method,
                             "hypothesis": f"  preserved {qid}\n한국어\t ",
                             "prediction": f"  preserved {qid}\n한국어\t ", "status": "ok"}
                            for qid in reversed(self.ids)]
        self.write_jsonl(self.args.predictions, self.predictions)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def results(self, labels=None):
        bridge.export(self.args)
        rows = bridge.read_jsonl(self.args.hypotheses)
        labels = labels if labels is not None else [True, False, True, False, True, False, False]
        for row, label in zip(rows, labels):
            row["autoeval_label"] = {"model": bridge.JUDGE_MODEL, "label": label}
        self.write_jsonl(self.args.results, rows)
        return rows

    def test_exact_text_canonical_order_two_fields_and_model_provenance(self):
        receipt = bridge.export(self.args)
        rows = bridge.read_jsonl(self.args.hypotheses)
        self.assertEqual([row["question_id"] for row in rows], self.ids)
        self.assertEqual(rows[0]["hypothesis"], self.predictions[-1]["hypothesis"])
        self.assertTrue(all(set(row) == {"question_id", "hypothesis"} for row in rows))
        self.assertEqual(receipt["embedding_name"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(receipt["model_name"], "Qwen/Qwen3.5-9B")
        self.assertEqual(receipt["hypotheses_sha256"], bridge.sha256(self.args.hypotheses))

    def test_genuine_empty_generation_preserved_and_flagged_even_if_positive_judge(self):
        self.predictions[0].update(hypothesis=" \n", prediction=" \n", status="generation_empty")
        self.write_jsonl(self.args.predictions, self.predictions)
        self.results([True] * len(self.ids))
        report = bridge.verify(self.args)
        self.assertEqual(report["provenance"]["completed_empty_question_ids"], ["q0_abs"])
        self.assertEqual(report["overall"]["correct"], 7)

    def test_prediction_coverage_rejects_missing_duplicate_and_extra(self):
        bad_populations = [self.predictions[:-1], self.predictions + [self.predictions[0]],
                           self.predictions + [{**self.predictions[0], "question_id": "unknown"}]]
        for rows in bad_populations:
            with self.subTest(rows=len(rows)):
                self.write_jsonl(self.args.predictions, rows)
                with self.assertRaises(ValueError):
                    bridge.export(self.args)
                self.assertFalse(self.args.hypotheses.exists())

    def test_invalid_status_method_prediction_and_hypothesis_rejected(self):
        mutations = [{"status": "error"}, {"method": "seed"}, {"prediction": "changed"},
                     {"hypothesis": None}, {"status": "generation_empty"}]
        for mutation in mutations:
            rows = copy.deepcopy(self.predictions)
            rows[0].update(mutation)
            self.write_jsonl(self.args.predictions, rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bridge.export(self.args)

    def test_expected_population_cannot_be_inferred_from_fewer_predictions(self):
        self.write_jsonl(self.args.predictions, self.predictions[:2])
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            bridge.export(self.args)

    def test_duplicate_unknown_and_missing_expected_ids_rejected(self):
        for value in (["q0", "q0"], ["unknown"], []):
            self.write_json(self.args.expected_ids, value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                bridge.export(self.args)

    def test_frozen_split_requires_partition_and_checks_parent_hash(self):
        manifest = {"dataset": {"sha256": bridge.CANONICAL_SHA256},
                    "partitions": {"dev": {"question_ids": list(reversed(self.ids))}}}
        self.write_json(self.args.expected_ids, manifest)
        with self.assertRaisesRegex(ValueError, "partition"):
            bridge.prepare(self.args)
        self.args.partition = "dev"
        self.assertEqual(bridge.prepare(self.args)[1]["expected_question_ids"], self.ids)
        manifest["dataset"]["sha256"] = "0" * 64
        self.write_json(self.args.expected_ids, manifest)
        with self.assertRaisesRegex(ValueError, "dataset hash"):
            bridge.prepare(self.args)

    def test_explicit_selected_ids_manifest_supported(self):
        self.write_json(self.args.expected_ids, {"selected_ids": self.ids})
        self.assertEqual(bridge.prepare(self.args)[1]["expected_count"], 7)

    def test_protocol_selection_and_arm_must_match(self):
        self.protocol["selection"]["selected_ids"] = self.ids[:-1]
        self.write_json(self.args.protocol, self.protocol)
        with self.assertRaisesRegex(ValueError, "selection differs"):
            bridge.prepare(self.args)
        self.protocol["selection"]["selected_ids"] = self.ids
        self.protocol["evaluation_methods"] = ["seed"]
        self.write_json(self.args.protocol, self.protocol)
        with self.assertRaisesRegex(ValueError, "arm"):
            bridge.prepare(self.args)

    def test_derived_generation_data_needs_canonical_parent(self):
        self.protocol["dataset_sha256"] = "1" * 64
        self.write_json(self.args.protocol, self.protocol)
        with self.assertRaisesRegex(ValueError, "canonical_parent"):
            bridge.prepare(self.args)
        self.protocol["canonical_parent_sha256"] = bridge.CANONICAL_SHA256
        self.write_json(self.args.protocol, self.protocol)
        self.assertEqual(bridge.prepare(self.args)[1]["generation_data_sha256"], "1" * 64)

    def test_reference_hash_mismatch_fails_before_export(self):
        self.args.reference.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reference SHA256"):
            bridge.export(self.args)

    def test_reference_swap_after_initial_hash_cannot_claim_full_population(self):
        original_reader = bridge.read_json
        def swap_before_read(path):
            if path == self.args.reference:
                self.write_json(path, self.references[:2])
            return original_reader(path)
        with patch.object(bridge, "read_json", side_effect=swap_before_read):
            with self.assertRaisesRegex(ValueError, "reference changed"):
                bridge.export(self.args)
        self.assertFalse(self.args.hypotheses.exists())

    def test_upstream_byte_change_rejected(self):
        vendor = self.root / "vendor"
        for relative in bridge.UPSTREAM_HASHES:
            destination = vendor / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(VENDOR / relative, destination)
        bridge.verify_upstream(vendor)
        with (vendor / "src/evaluation/evaluate_qa.py").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "official file changed"):
            bridge.verify_upstream(vendor)

    def test_export_refuses_overwrite_and_preserves_bytes(self):
        bridge.export(self.args)
        original = self.args.hypotheses.read_bytes()
        with self.assertRaises(FileExistsError):
            bridge.export(self.args)
        self.assertEqual(self.args.hypotheses.read_bytes(), original)

    def test_counts_macro_and_abstention_match_official_metrics_script(self):
        self.results()
        report = bridge.verify(self.args)
        self.assertEqual(report["overall"], {"correct": 3, "count": 7, "accuracy": 3 / 7})
        self.assertAlmostEqual(report["six_type_macro_accuracy"], 2.5 / 6)
        self.assertEqual(report["abstention"], {"correct": 0, "count": 1, "accuracy": 0.0})
        completed = subprocess.run([sys.executable, "-X", "utf8", str(VENDOR / "src/evaluation/print_qa_metrics.py"),
                                    str(self.args.results), str(self.args.reference)],
                                   capture_output=True, text=True, check=True)
        self.assertIn("Overall Accuracy: 0.4286", completed.stdout)
        self.assertIn("Task-averaged Accuracy: 0.4167", completed.stdout)
        self.assertIn("Abstention Accuracy: 0.0 (1)", completed.stdout)

    def test_partial_population_has_null_missing_groups_and_diagnostic_scope(self):
        keep = self.ids[:2]
        self.write_json(self.args.expected_ids, keep)
        self.protocol["selection"]["selected_ids"] = keep
        self.write_json(self.args.protocol, self.protocol)
        self.write_jsonl(self.args.predictions, [row for row in self.predictions if row["question_id"] in keep])
        self.results([True, False])
        report = bridge.verify(self.args)
        self.assertEqual(report["provenance"]["scope"], "diagnostic_subset")
        self.assertIsNone(report["six_type_macro_accuracy"])
        self.assertIsNone(report["abstention"]["accuracy"])
        self.assertIsNone(report["per_question_type"]["knowledge-update"]["accuracy"])

    def test_official_results_missing_duplicate_extra_rejected(self):
        rows = self.results()
        for bad in (rows[:-1], rows + [rows[0]], rows + [{**rows[0], "question_id": "unknown"}]):
            self.write_jsonl(self.args.results, bad)
            with self.subTest(count=len(bad)), self.assertRaises(ValueError):
                bridge.verify(self.args)
        self.assertFalse(self.args.report.exists())

    def test_official_model_label_type_and_hypothesis_must_match(self):
        original = self.results()
        mutations = [{"autoeval_label": {"model": "gpt-4o", "label": True}},
                     {"autoeval_label": {"model": bridge.JUDGE_MODEL, "label": 1}},
                     {"autoeval_label": {"model": bridge.JUDGE_MODEL, "label": "yes"}},
                     {"autoeval_label": {"model": bridge.JUDGE_MODEL, "label": None}},
                     {"autoeval_label": None}, {"hypothesis": original[0]["hypothesis"].strip()},
                     {"verdict": "yes"}]
        for mutation in mutations:
            rows = copy.deepcopy(original)
            rows[0].update(mutation)
            self.write_jsonl(self.args.results, rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bridge.verify(self.args)

    def test_changed_predictions_protocol_manifest_hypotheses_receipt_rejected(self):
        self.results()
        for path in (self.args.predictions, self.args.protocol, self.args.expected_ids,
                     self.args.hypotheses):
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                bridge.verify(self.args)
            path.write_bytes(original)
        receipt = bridge.read_json(bridge.receipt_path(self.args.hypotheses))
        receipt["expected_count"] = 999
        self.write_json(bridge.receipt_path(self.args.hypotheses), receipt)
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            bridge.verify(self.args)

    def test_input_and_result_mutation_during_validation_rejected(self):
        original_reader = bridge.read_jsonl
        def mutate_after_read(path):
            rows = original_reader(path)
            if path == self.args.predictions:
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(" ")
            return rows
        with patch.object(bridge, "read_jsonl", side_effect=mutate_after_read):
            with self.assertRaisesRegex(ValueError, "Input changed"):
                bridge.export(self.args)
        self.write_jsonl(self.args.predictions, self.predictions)
        self.results()
        def mutate_results_after_read(path):
            rows = original_reader(path)
            if path == self.args.results:
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(" ")
            return rows
        with patch.object(bridge, "read_jsonl", side_effect=mutate_results_after_read):
            with self.assertRaisesRegex(ValueError, "results changed"):
                bridge.verify(self.args)
        self.assertFalse(self.args.report.exists())

    def test_report_refuses_overwrite(self):
        self.results()
        bridge.verify(self.args)
        original = self.args.report.read_bytes()
        with self.assertRaises(FileExistsError):
            bridge.verify(self.args)
        self.assertEqual(self.args.report.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
