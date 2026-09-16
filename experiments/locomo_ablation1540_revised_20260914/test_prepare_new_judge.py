"""Synthetic, offline tests for canonical population and native judge-input provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_new_judge as prep


class JudgeInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-locomo-judge-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results, self.run = self.root / "results", self.root / "run"
        self.results.mkdir()
        self.run.mkdir()
        self.dataset = self.root / "synthetic_dataset.json"
        categories = [category for category, count in prep.CATEGORY_COUNTS.items() for _ in range(count)]
        dataset = [{"sample_id": f"test-{i:02d}", "qa": [
            {"question": "Synthetic question?", "answer": "synthetic gold; keep suffix",
             "category": categories[i * 154 + j]} for j in range(154)]} for i in range(10)]
        self.write(self.dataset, dataset)
        self.hash_patch = patch.object(prep, "DATASET_SHA256", prep.report.sha(self.dataset))
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        self.protocol = {"population": "full1540", "question_count": 1540, "arms": ["ours"],
                         "max_output_tokens": 96, "shards": 1, "reader_system": "Synthetic reader system",
                         "environment": {"models": {prep.READER_MODEL: {
                             "path": "synthetic-test-only", "revision": prep.READER_REVISION}}}}
        self.write(self.run / "protocol.json", self.protocol)
        self.rows = []
        hashes = {}
        for cid in sorted(sample["sample_id"] for sample in dataset):
            originals = []
            for reference in prep.canonical_rows(self.dataset).values():
                if reference["conv_id"] != cid:
                    continue
                row = {key: value for key, value in reference.items() if key != "gold"}
                row.update(arm="ours", context="Synthetic context", prediction=" synthetic answer ",
                           finish_reason="stop", input_tokens=100, output_tokens=4)
                row["generation_cache_key"] = prep.native_request_key(
                    self.protocol["environment"]["models"][prep.READER_MODEL],
                    self.protocol["reader_system"], row)
                originals.append(row)
                self.rows.append({**row, "gold": reference["gold"], "official_f1": 0.5})
            path = self.run / "predictions/ours" / f"{cid}.json"
            self.write(path, originals)
            hashes[f"ours/{cid}"] = prep.report.sha(path)
        key = self.rows[0]["generation_cache_key"]
        self.receipt_path = self.run / "evaluate_0/cache/generations" / f"{key}.json"
        self.write(self.receipt_path, {"text": " synthetic answer ", "finish_reason": "stop",
                                       "input_tokens": 100, "output_tokens": 4})
        self.validation = {"status": "PASS", "arms": 1, "questions_per_arm": 1540,
                           "histories": 10, "prediction_count": 1540,
                           "protocol_sha256": prep.report.sha(self.run / "protocol.json"),
                           "prediction_file_sha256": hashes,
                           "native_receipt_sha256": {f"evaluate_0/{key}": prep.report.sha(self.receipt_path)}}
        self.flush()

    @staticmethod
    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def flush(self) -> None:
        self.write(self.results / "scored_predictions.json", {"ours": self.rows})
        self.write(self.results / "VALIDATION.json", self.validation)
        self.write(self.results / "RESULTS.json", {
            "validation": self.validation, "protocol": self.protocol,
            "results": {"ours": {"n": 1540, "histories": 10, "f1": 50.0}}})

    def validate(self) -> tuple[list[dict], dict]:
        return prep.validated_rows(self.results, self.run, "ours", self.dataset)

    def test_prepares_exact_raw_strings_and_old_runner_manifest(self) -> None:
        output = self.root / "judge_input"
        manifest = prep.prepare(self.results, self.run, "ours", output, self.dataset)
        rows = [json.loads(line) for line in (output / "input.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1540)
        self.assertEqual(rows[0]["generated_answer"], " synthetic answer ")
        self.assertEqual(rows[0]["gold_answer"], "synthetic gold; keep suffix")
        self.assertEqual(manifest["input_sha256"], prep.report.sha(output / "input.jsonl"))
        self.assertEqual(manifest["prompt_sha256"], prep.report.sha(output / "official_accuracy_prompt.txt"))
        self.assertEqual(manifest["required_returned_model"], "gpt-4o-mini-2024-07-18")
        self.assertEqual(manifest["status"], "PREPARED_NO_API_CALLS")
        with self.assertRaisesRegex(ValueError, "new empty output"):
            prep.prepare(self.results, self.run, "ours", output, self.dataset)

    def test_accepts_round2_native_receipt_schema(self) -> None:
        old = self.validation.pop("native_receipt_sha256")
        self.validation["native_reader_sha256"] = {key.split("/")[1]: value for key, value in old.items()}
        self.flush()
        rows, _ = self.validate()
        self.assertEqual(len(rows), 1540)

    def test_rejects_299_and_1539_rows(self) -> None:
        complete = self.rows
        for count in (299, 1539):
            with self.subTest(count=count):
                self.rows = complete[:count]
                self.flush()
                with self.assertRaisesRegex(ValueError, "1,540 unique"):
                    self.validate()

    def test_rejects_duplicate_or_wrong_id(self) -> None:
        for bad_id in (self.rows[1]["id"], "test-00:999"):
            with self.subTest(bad_id=bad_id):
                self.rows[0]["id"] = bad_id
                self.flush()
                with self.assertRaisesRegex(ValueError, "1,540 unique"):
                    self.validate()

    def test_rejects_question_gold_category_or_answer_changes(self) -> None:
        original = copy.deepcopy(self.rows[0])
        for field, value in (("question", "changed"), ("gold", "changed"), ("category", 4),
                             ("prediction", "changed"), ("qa_index", 999)):
            with self.subTest(field=field):
                self.rows[0] = {**original, field: value}
                self.flush()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_rejects_unvalidated_result(self) -> None:
        self.validation["status"] = "INCOMPLETE"
        self.flush()
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.validate()

    def test_rejects_changed_protocol_seal(self) -> None:
        self.validation["protocol_sha256"] = "0" * 64
        self.flush()
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.validate()

    def test_rejects_wrong_reader_model(self) -> None:
        self.protocol["environment"]["models"][prep.READER_MODEL]["revision"] = "wrong-model"
        self.write(self.run / "protocol.json", self.protocol)
        self.validation["protocol_sha256"] = prep.report.sha(self.run / "protocol.json")
        self.flush()
        with self.assertRaisesRegex(ValueError, "reader model"):
            self.validate()

    def test_rejects_native_receipt_corruption(self) -> None:
        native = prep.report.read(self.receipt_path)
        native["text"] = "corrupt native text"
        self.write(self.receipt_path, native)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.validate()
        key = self.rows[0]["generation_cache_key"]
        self.validation["native_receipt_sha256"][f"evaluate_0/{key}"] = prep.report.sha(self.receipt_path)
        self.flush()
        with self.assertRaisesRegex(ValueError, "Native reader receipt mismatch"):
            self.validate()

    def test_rejects_original_prediction_corruption(self) -> None:
        original = self.run / "predictions/ours/test-00.json"
        part = prep.report.read(original)
        part[0]["prediction"] = "corrupt source prediction"
        self.write(original, part)
        with self.assertRaisesRegex(ValueError, "prediction file"):
            self.validate()

    def test_rejects_wrong_judge_model(self) -> None:
        runner = self.root / "wrong_judge.py"
        runner.write_text("MODEL = 'gpt-4o'\n", encoding="utf-8")
        with patch.object(prep, "JUDGE_RUNNER", runner):
            with self.assertRaisesRegex(ValueError, "runner fingerprint"):
                prep.load_judge()
        with (patch.object(prep, "JUDGE_RUNNER", runner),
              patch.object(prep, "JUDGE_RUNNER_SHA256", prep.report.sha(runner))):
            with self.assertRaisesRegex(ValueError, "judge model"):
                prep.load_judge()


if __name__ == "__main__":
    unittest.main()
