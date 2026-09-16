"""Behavioral tests for full-set ablation summary and paired inference."""
import statistics
import unittest

import report_results as report


def row(qid, cid, score, stored=20, read=5):
    return {"id": qid, "conv_id": cid, "category": 1, "official_f1": score,
            "stored_tokens": stored, "read_tokens": read, "prediction": "answer", "finish_reason": "stop"}


class StatisticsTests(unittest.TestCase):
    def test_question_weighted_cluster_bootstrap(self):
        left = [row("a", "small", 1)] + [row(str(i), "large", 0) for i in range(9)]
        right = [dict(r, official_f1=0) for r in left]
        result = report.paired_statistics(left, right)
        self.assertAlmostEqual(result["delta_pp"], 10)
        self.assertEqual(result["paired_95ci_pp"], [0, 100])
        self.assertEqual(result["permutations"], 4)
        self.assertEqual(report.paired_statistics(left, right, draws=1, seed=1)["paired_95ci_pp"], [10, 10])
        self.assertEqual(result, report.paired_statistics(left, list(reversed(right))))

    def test_exact_ten_cluster_sign_flip(self):
        left = [row(str(i), str(i), 1) for i in range(10)]
        right = [dict(r, official_f1=0) for r in left]
        result = report.paired_statistics(left, right)
        self.assertEqual(result["permutations"], 1024)
        self.assertEqual(result["sign_flip_p"], 2 / 1024)
        self.assertEqual(result["paired_95ci_pp"], [100, 100])
        self.assertEqual(report.paired_statistics(left, left)["sign_flip_p"], 1)

    def test_paired_population_validation(self):
        rows = [row("a", "one", .3), row("b", "two", .4)]
        for invalid in (rows[:1], [rows[0], rows[0]], [rows[0], dict(rows[1], conv_id="other")]):
            with self.assertRaises(ValueError):
                report.paired_statistics(rows, invalid)

    def test_holm_monotonic_adjustment(self):
        adjusted = report.holm_adjust({"a": .03, "b": .01, "c": .04})
        self.assertEqual(adjusted, {"b": .03, "a": .06, "c": .06})

    def test_storage_uses_histories_reading_uses_questions(self):
        rows = [row("a", "small", 1, 100, 10)] + [row(str(i), "large", 0, 200, 20) for i in range(9)]
        accounts = {cid: {"parent_payload_tokens": total - 3, "cue_payload_tokens": 2,
                          "distinct_key_tokens": 1, "total_stored_tokens": total}
                    for cid, total in (("small", 100), ("large", 200))}
        result = report.summarize(rows, accounts)
        self.assertEqual(result["f1"], 10)
        self.assertEqual(result["stored_tokens"], 150)
        self.assertEqual(result["read_tokens"], 19)
        self.assertEqual(result["storage_breakdown"]["parent_payload_tokens"], 147)
        with self.assertRaises(ValueError):
            report.summarize(rows + [dict(rows[0], stored_tokens=101)], accounts)
        accounts["small"]["cue_payload_tokens"] = 3
        with self.assertRaises(ValueError):
            report.summarize(rows, accounts)

    def test_random_uses_every_seed_before_paired_comparison(self):
        scores = {arm: [dict(row("q", "h", i / 10), qa_index=0, question="Q")]
                  for i, arm in enumerate(report.RANDOM_ARMS)}
        actual = report.random_mean_rows(scores)
        self.assertAlmostEqual(actual[0]["official_f1"], statistics.mean(i / 10 for i in range(10)))
        scores["random_09"][0]["id"] = "different"
        with self.assertRaises(ValueError):
            report.random_mean_rows(scores)


    def test_native_receipt_rejects_tampering_and_invalid_metadata(self):
        valid = {"prediction": "", "generation_cache_key": "key", "finish_reason": "length",
                 "input_tokens": 200, "output_tokens": 96}
        native = {"text": "", "finish_reason": "length", "input_tokens": 200, "output_tokens": 96}
        report.validate_receipt(valid, native, "key")
        for changed in (dict(valid, generation_cache_key="other"), dict(valid, prediction="different")):
            with self.assertRaises(ValueError):
                report.validate_receipt(changed, native, "key")
        for field, value in (("input_tokens", True), ("input_tokens", 0), ("input_tokens", 8097),
                             ("output_tokens", False), ("output_tokens", 97), ("finish_reason", "error")):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                report.validate_receipt(dict(valid, **{field: value}), dict(native, **{field: value}), "key")
        with self.assertRaises(ValueError):
            report.validate_receipt(dict(valid, output_tokens=1), dict(native, output_tokens=True), "key")

    def test_context_must_match_locked_payload_and_token_count(self):
        units = [{"text": "hello"}, {"text": "world"}]
        valid = {"memory_indices": [0], "context": "hello", "context_sha256": "hello", "read_tokens": 5}
        self.assertEqual(report.validate_context(valid, units, len, str), "hello")

        multiple = {"memory_indices": [0, 1], "context": "hello\n\nworld",
                    "context_sha256": "hello\n\nworld", "read_tokens": 12}
        self.assertEqual(report.validate_context(multiple, units, len, str), "hello\n\nworld")
        self.assertEqual(report.reader_request("hello\n\nworld", "Who?"),
                         "Conversation memory:\nhello\n\nworld\n\nQuestion: Who?\nAnswer:")

        for change in ({"context": "other"}, {"context_sha256": "tampered"}, {"read_tokens": 4},
                       {"read_tokens": True}, {"memory_indices": [0, 0]}, {"memory_indices": [-1]},
                       {"memory_indices": [True]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                report.validate_context(dict(valid, **change), units, len, str)

    def test_finite_reader_selection_uses_overall_score_ties_legacy(self):
        record = {"candidate_configurations": ["legacy", "grounded"], "population": "dev300",
                  "arms": ["ours"], "criterion": "highest_overall_f1_tie_legacy", "frozen_before_full1540": True,
                  "results": {"legacy": {"n": 300, "f1": 50}, "grounded": {"n": 300, "f1": 50}},
                  "chosen_reader": "legacy"}
        report.validate_selection(record, "legacy")
        with self.assertRaises(ValueError):
            report.validate_selection(dict(record, chosen_reader="grounded"), "grounded")
        record["results"]["grounded"]["f1"] = 51
        with self.assertRaises(ValueError):
            report.validate_selection(record, "legacy")
        report.validate_selection(dict(record, chosen_reader="grounded"), "grounded")


if __name__ == "__main__":
    unittest.main()
