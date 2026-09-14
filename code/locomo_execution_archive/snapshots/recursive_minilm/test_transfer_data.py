"""Behavioral checks for data isolation, preservation and paired statistics."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from transfer_data import (
    adapt_longmemeval,
    deterministic_subset,
    load_longmemeval,
    paired_summary,
    split_sample,
)


def example(qid="example"):
    return {
        "question_id": qid,
        "question_type": "single-session-user",
        "question": "What did I study?",
        "question_date": "2023/05/30 (Tue) 23:40",
        "answer": "GOLD_SENTINEL",
        "answer_session_ids": ["answer_SECRET_A"],
        "haystack_session_ids": ["answer_SECRET_A", "noise_SECRET_B", "noise_SECRET_B"],
        "haystack_dates": ["2023/05/01 10:00", "2023/05/02 11:00", "2023/05/03 12:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "I studied history.", "has_answer": True},
             {"role": "assistant", "content": "Interesting!", "has_answer": False}],
            [{"role": "user", "content": "Second session.\nUnchanged Unicode: 한글."}],
            [{"role": "assistant", "content": "Third session, duplicate raw ID."}],
        ],
    }


class AdapterTests(unittest.TestCase):
    def test_gold_labels_and_identifiers_cannot_change_model_input(self):
        original = example()
        altered = copy.deepcopy(original)
        altered.update(question_id="DIFFERENT_ID_abs", answer=90210,
                       question_type="temporal-reasoning", answer_session_ids=["new_gold"])
        altered["haystack_session_ids"] = ["new_noise", "new_gold", "new_gold"]
        for session in altered["haystack_sessions"]:
            for turn in session:
                turn["has_answer"] = not turn.get("has_answer", False)
        first_input, first_meta = split_sample(adapt_longmemeval([original])[0])
        second_input, second_meta = split_sample(adapt_longmemeval([altered])[0])
        self.assertEqual(first_input, second_input)
        self.assertNotEqual(first_meta, second_meta)
        serialized = json.dumps(first_input)
        for sentinel in ("GOLD_SENTINEL", "SECRET", "has_answer", "question_type", "example"):
            self.assertNotIn(sentinel, serialized)
        self.assertEqual(first_meta["evidence"], ["D1:1"])

    def test_preserves_all_dates_roles_content_and_duplicate_session_occurrences(self):
        row = example("example_abs")
        sample = adapt_longmemeval([row])[0]
        model_input, metadata = split_sample(sample)
        self.assertTrue(metadata["is_abstention"])
        self.assertEqual(metadata["question_type"], "single-session-user")
        self.assertEqual(metadata["cluster_id"], "example")
        self.assertEqual(model_input["question_date"], row["question_date"])
        for number, turns in enumerate(row["haystack_sessions"], 1):
            key = f"session_{number}"
            self.assertEqual(
                model_input["conversation"][f"{key}_date_time"],
                row["haystack_dates"][number - 1],
            )
            for index, turn in enumerate(turns, 1):
                self.assertEqual(model_input["conversation"][key][index - 1], {
                    "dia_id": f"D{number}:{index}",
                    "speaker": turn["role"], "text": turn["content"],
                })
        self.assertEqual(len(metadata["session_map"]), 3)
        model_input["conversation"]["session_1"][0]["text"] = "changed"
        self.assertEqual(sample["conversation"]["session_1"][0]["text"], "I studied history.")

    def test_alignment_and_unique_question_ids_are_required(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            adapt_longmemeval([example(), example()])
        row = example()
        row["haystack_dates"].pop()
        with self.assertRaisesRegex(ValueError, "Unaligned"):
            adapt_longmemeval([row])

    def test_load_and_hash_subset_keep_complete_history_and_abstentions(self):
        raw = [example(f"id{index}") for index in range(12)] + [example("id3_abs")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            samples = load_longmemeval(path)
            self.assertEqual(len(samples), 13)
            self.assertTrue(any(sample["qa"][0]["is_abstention"] for sample in samples))
            subset, manifest = deterministic_subset(samples, 5)
            reversed_subset, reversed_manifest = deterministic_subset(list(reversed(samples)), 5)
            self.assertEqual(subset, reversed_subset)
            self.assertEqual(manifest, reversed_manifest)
            self.assertEqual(load_longmemeval(path, limit=5), subset)
            self.assertEqual(len(subset[0]["conversation"]), 6)
            self.assertNotEqual(
                manifest["selected_ids"], [s["sample_id"] for s in samples[:5]]
            )


class PairedSummaryTests(unittest.TestCase):
    def rows(self):
        baseline = [
            {"question_id": qid, "score": score, "question_type": kind}
            for qid, score, kind in [
                ("a", 0, "single-session-user"),
                ("a_abs", 1, "single-session-user"),
                ("b", 0, "temporal-reasoning"),
            ]
        ]
        refined = copy.deepcopy(baseline)
        refined[0]["score"] = 1
        refined[2]["score"] = 1
        return baseline, refined

    def summary(self, baseline, refined, **kwargs):
        return paired_summary(
            baseline, refined, manifest=["a", "a_abs", "b"], n_bootstrap=300, **kwargs
        )

    def test_pairs_by_id_cluster_bootstrap_preserves_abs_and_base_types(self):
        baseline, refined = self.rows()
        result = self.summary(baseline, list(reversed(refined)))
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["cluster_count"], 2)
        self.assertAlmostEqual(result["effect_pp"], 200 / 3)
        self.assertEqual((result["wins"], result["losses"], result["ties"]), (2, 0, 1))
        self.assertEqual(result["ci95_pp"], [50.0, 100.0])
        self.assertEqual(result["abstention"]["n"], 1)
        self.assertEqual(result["by_question_type"]["single-session-user"]["n"], 2)
        self.assertEqual(result, self.summary(list(reversed(baseline)), refined))

    def test_rejects_missing_extra_duplicate_and_unjudged_pairs(self):
        baseline, refined = self.rows()
        for bad in (
            refined[:-1], refined + [refined[0]],
            refined + [{"question_id": "extra", "score": 1}],
        ):
            with self.assertRaises(ValueError):
                self.summary(baseline, bad)
        refined[0]["score"] = None
        with self.assertRaisesRegex(ValueError, "unjudged"):
            self.summary(baseline, refined)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.summary(baseline[:-1], refined[:-1])

    def test_empty_and_failed_generations_count_zero_in_full_denominator(self):
        baseline, refined = self.rows()
        refined[0].update(prediction="  ", score=1)
        refined[1].update(status="error", score=None)
        result = self.summary(baseline, refined)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["effect_pp"], 0)
        self.assertEqual(result["generation_failures_counted_zero"]["refined"], 2)

    def test_locomo_bootstraps_conversations_and_requires_group_alignment(self):
        baseline, refined = self.rows()
        for rows in (baseline, refined):
            for row in rows:
                row["conversation_id"] = "conversation1"
        result = self.summary(baseline, refined, dataset="locomo")
        self.assertEqual(result["cluster_count"], 1)
        self.assertAlmostEqual(result["ci95_pp"][0], 200 / 3)
        refined[0]["conversation_id"] = "other"
        with self.assertRaisesRegex(ValueError, "mismatched"):
            self.summary(baseline, refined, dataset="locomo")

    def test_metadata_conflicts_and_nonfinite_scores_rejected(self):
        baseline, refined = self.rows()
        refined[0]["question_type"] = "wrong"
        with self.assertRaisesRegex(ValueError, "question type"):
            self.summary(baseline, refined)
        baseline, refined = self.rows()
        refined[0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            self.summary(baseline, refined)


if __name__ == "__main__":
    unittest.main()

