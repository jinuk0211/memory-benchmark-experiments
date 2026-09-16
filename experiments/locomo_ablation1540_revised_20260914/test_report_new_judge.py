"""Offline full-population fixtures for final mini export and evidence tampering."""
from __future__ import annotations

import copy
import json
import statistics
from types import SimpleNamespace
import unittest

import prepare_new_judge as prep
import report_new_judge as report
import test_prepare_new_judge as input_tests


class JudgeExportTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = input_tests.JudgeInputTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.fixture = fixture
        first = fixture.rows[0]
        first["prediction"] = "Synthetic alternate answer"
        first["context"] = "Synthetic alternate source context"
        first["generation_cache_key"] = prep.native_request_key(
            fixture.protocol["environment"]["models"][prep.READER_MODEL],
            fixture.protocol["reader_system"], first)
        path = fixture.run / "predictions/ours/test-00.json"
        original = prep.report.read(path)
        original[0] = {key: value for key, value in first.items() if key not in {"gold", "official_f1"}}
        fixture.write(path, original)
        fixture.validation["prediction_file_sha256"]["ours/test-00"] = prep.report.sha(path)
        receipt = fixture.run / "evaluate_0/cache/generations" / f"{first['generation_cache_key']}.json"
        fixture.write(receipt, {"text": first["prediction"], "finish_reason": "stop",
                                "input_tokens": 100, "output_tokens": 4})
        fixture.validation["native_receipt_sha256"][f"evaluate_0/{first['generation_cache_key']}"] = prep.report.sha(receipt)
        for row in fixture.rows:
            row["official_f1"] = row["category"] / 10
        fixture.flush()
        results = prep.report.read(fixture.results / "RESULTS.json")
        results["results"]["ours"].update(
            f1=statistics.mean(row["official_f1"] for row in fixture.rows) * 100,
            category_f1={str(category): category * 10 for category in report.CATEGORIES},
            stored_tokens=1234.5, read_tokens=678.9)
        fixture.write(fixture.results / "RESULTS.json", results)
        self.directory = fixture.root / "judge"
        self.output = fixture.root / "export"
        prep.prepare(fixture.results, fixture.run, "ours", self.directory, fixture.dataset)
        self.judge = prep.load_judge()
        self.rows = report.read_jsonl(self.directory / "input.jsonl")
        prompt = self.judge.load_prompt()
        self.keys = [self.judge.payload_hash(self.judge.make_payload(row, prompt)) for row in self.rows]
        self.events = []
        for index, key in enumerate(dict.fromkeys(self.keys)):
            label = "WRONG" if index == 0 else "CORRECT"
            usage = {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16,
                     "prompt_tokens_details": {"cached_tokens": 2}}
            self.events.append({"kind": "response", "payload_hash": key, "attempt": 1,
                                "label": label, "response": json.dumps({"label": label}),
                                "returned_model": prep.REQUIRED_RETURNED_MODEL,
                                "response_id": f"synthetic-response-{index}", "request_id": None,
                                "finish_reason": "stop", "usage": usage,
                                "prompt_sha256": prep.PROMPT_SHA256,
                                "prices_per_million_usd": report.PRICES,
                                "cost_usd": self.judge.usage_cost(usage, report.PRICES),
                                "unknown_cost_upper_bound_usd": 0})
        self.cache = copy.deepcopy(self.events)
        manifest = prep.report.read(self.directory / "input_manifest.json")
        fixture.write(self.directory / "run_config.json", {
            "input_sha256": manifest["input_sha256"], "prompt_sha256": prep.PROMPT_SHA256,
            "model": prep.JUDGE_MODEL, "temperature": 0.0, "response_format": {"type": "json_object"},
            "prompt_source": str(self.judge.PROMPT_SOURCE),
            "source_sha256": prep.report.sha(self.judge.PROMPT_SOURCE),
            "deduplication": "Identical full API payloads share a judgment across methods"})
        self.flush()

    def write_jsonl(self, name: str, rows: list[dict]) -> None:
        (self.directory / name).write_text("".join(self.judge.canonical_json(row) + "\n" for row in rows), encoding="utf-8")

    def flush(self) -> None:
        self.write_jsonl("api_events.jsonl", self.events)
        self.write_jsonl("judge_cache.jsonl", self.cache)
        journal = SimpleNamespace(cache={row["payload_hash"]: row for row in self.cache},
                                  output_dir=self.directory, events=self.events,
                                  spent=sum(row.get("cost_usd", 0) for row in self.events),
                                  unknown_cost_upper_bound=sum(row["unknown_cost_upper_bound_usd"] for row in self.events))
        self.judge.summarize(self.rows, self.keys, journal, "synthetic_test_only")

    def export(self) -> dict:
        return report.export(self.directory, self.output)

    def test_complete_deduplicated_export_preserves_f1_and_category_accuracy(self) -> None:
        before = {name: prep.report.sha(self.directory / name) for name in report.FILES}
        result = self.export()
        self.assertEqual(result["gpt4omini_accuracy"]["correct"], 1539)
        self.assertAlmostEqual(result["gpt4omini_accuracy"]["accuracy_pct"], 1539 / 1540 * 100)
        self.assertEqual(result["gpt4omini_accuracy"]["categories"]["1"]["correct"], 281)
        self.assertEqual(result["validation"]["unique_payloads"], 2)
        self.assertEqual(result["source_metrics"]["category_f1"], {"1": 10, "2": 20, "3": 30, "4": 40})
        self.assertEqual(result["source_metrics"]["stored_tokens"], 1234.5)
        mapped = report.read_jsonl(self.output / "mapped_judgments.jsonl")
        self.assertEqual(len(mapped), 1540)
        self.assertEqual(mapped[1]["generated_answer"], " synthetic answer ")
        self.assertEqual(len(report.read_jsonl(self.output / "native_judgments.jsonl")), 2)
        latex = (self.output / "table_main_ours.tex").read_text(encoding="utf-8")
        self.assertEqual(latex.count("&"), 7)
        self.assertIn(r"& \textbf{Ours} & 10.00 & 20.00 & 30.00 & 40.00", latex)
        self.assertTrue(latex.endswith("99.94 \\\\\n"))
        self.assertEqual(before, {name: prep.report.sha(self.directory / name) for name in report.FILES})
        with self.assertRaisesRegex(ValueError, "empty report"):
            self.export()

    def test_rejects_missing_judgment_and_partial_scores(self) -> None:
        original = copy.deepcopy(self.cache)
        self.cache = self.cache[:1]
        self.flush()
        with self.assertRaisesRegex(ValueError, "all 1,540"):
            self.export()
        self.assertFalse(self.output.exists())
        self.cache = original
        self.flush()
        scores = report.read_jsonl(self.directory / "scores.jsonl")[:1539]
        self.write_jsonl("scores.jsonl", scores)
        with self.assertRaisesRegex(ValueError, "Saved scores"):
            self.export()

    def test_rejects_wrong_actual_model_on_selected_or_unselected_receipt(self) -> None:
        for model in ("gpt-4o-2024-08-06", "gpt-4o-mini-future"):
            with self.subTest(model=model):
                self.events[0]["returned_model"] = model
                self.cache[0]["returned_model"] = model
                self.flush()
                with self.assertRaisesRegex(ValueError, "identity, model or prompt"):
                    self.export()
        self.events[0]["returned_model"] = prep.REQUIRED_RETURNED_MODEL
        self.cache[0]["returned_model"] = prep.REQUIRED_RETURNED_MODEL
        extra = {**self.events[0], "response_id": "synthetic-paid-invalid",
                 "label": None, "response": "malformed", "returned_model": "gpt-4o-2024-08-06"}
        self.events.append(extra)
        self.flush()
        with self.assertRaisesRegex(ValueError, "identity, model or prompt"):
            self.export()

    def test_rejects_raw_label_finish_or_usage_tampering(self) -> None:
        original = copy.deepcopy(self.events[0])
        for field, value in (("response", '{"label":"CORRECT"}'), ("finish_reason", "length"),
                             ("response_id", ""), ("usage", None),
                             ("usage", {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 99}),
                             ("cost_usd", 999)):
            with self.subTest(field=field):
                self.events[0] = {**original, field: value}
                self.cache[0] = copy.deepcopy(self.events[0])
                self.flush()
                with self.assertRaises(ValueError):
                    self.export()

    def test_rejects_native_event_and_cache_disagreement(self) -> None:
        self.cache[0]["request_id"] = "changed-request-id"
        self.flush()
        with self.assertRaisesRegex(ValueError, "native response event"):
            self.export()

    def test_rejects_duplicate_response_id_and_unrelated_payload(self) -> None:
        self.events[1]["response_id"] = self.events[0]["response_id"]
        self.flush()
        with self.assertRaisesRegex(ValueError, "Duplicate native response"):
            self.export()
        self.events[1]["response_id"] = "synthetic-response-1"
        self.events[1]["payload_hash"] = "0" * 64
        self.flush()
        with self.assertRaisesRegex(ValueError, "unrelated"):
            self.export()

    def test_rejects_summary_and_f1_or_input_forgery(self) -> None:
        summary = prep.report.read(self.directory / "summary.json")
        summary["methods"]["ours"]["accuracy_pct"] = 100
        self.fixture.write(self.directory / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "summary differs"):
            self.export()
        self.flush()
        self.rows[0]["official_f1"] = 1.0
        self.write_jsonl("input.jsonl", self.rows)
        manifest = prep.report.read(self.directory / "input_manifest.json")
        manifest["input_sha256"] = prep.report.sha(self.directory / "input.jsonl")
        self.fixture.write(self.directory / "input_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "unchanged selected predictions or F1"):
            self.export()

    def test_rejects_protocol_or_source_change(self) -> None:
        config = prep.report.read(self.directory / "run_config.json")
        config["temperature"] = 1
        self.fixture.write(self.directory / "run_config.json", config)
        with self.assertRaisesRegex(ValueError, "run configuration"):
            self.export()
        config["temperature"] = 0
        self.fixture.write(self.directory / "run_config.json", config)
        result = prep.report.read(self.fixture.results / "RESULTS.json")
        result["results"]["ours"]["stored_tokens"] = 9999
        self.fixture.write(self.fixture.results / "RESULTS.json", result)
        with self.assertRaisesRegex(ValueError, "source/native fingerprints"):
            self.export()

    def test_retains_paid_invalid_response_without_scoring_it(self) -> None:
        extra = {**self.events[0], "response_id": "synthetic-paid-invalid", "label": None,
                 "response": "malformed JSON", "finish_reason": "length"}
        self.events.insert(0, extra)
        self.flush()
        result = self.export()
        self.assertEqual(result["validation"]["native_responses"], 3)
        self.assertEqual(result["gpt4omini_accuracy"]["correct"], 1539)
        self.assertEqual(len(report.read_jsonl(self.output / "native_judgments.jsonl")), 3)

    def test_rejects_incomplete_journal_without_repair(self) -> None:
        path = self.directory / "api_events.jsonl"
        original = path.read_bytes() + b'{"partial":'
        path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "Unfinished journal"):
            self.export()
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
