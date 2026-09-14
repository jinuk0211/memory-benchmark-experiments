"""Behavioral checks for semantic grading and incomplete coverage."""
import tempfile
import unittest
from pathlib import Path

from prepare_inputs import MODEL, canonical, prompt_function
from run_semantic_judge import load_cache, parse_label, summarize


class GradingTests(unittest.TestCase):
    def test_only_complete_yes_no_is_a_verdict(self):
        self.assertTrue(parse_label(" Yes\n"))
        self.assertFalse(parse_label("no"))
        for value in ("not yes", "", "yes or no", None):
            with self.assertRaises(ValueError):
                parse_label(value)

    def test_missing_or_failed_judgments_do_not_become_wrong(self):
        rows = [{"method": "baseline", "question_id": "one", "hypothesis": "45",
                 "payload_hash": "a"},
                {"method": "own", "question_id": "one", "hypothesis": "unknown",
                 "payload_hash": "b"}]
        result = summarize(rows, {"a": {"label": True}})
        baseline = result["methods"]["baseline"]
        self.assertEqual(baseline["accuracy_on_available"], 1.0)
        self.assertIsNone(baseline["accuracy_on_same12"])
        self.assertEqual(baseline["missing_predictions"], 11)
        self.assertIsNone(result["methods"]["own"]["accuracy_on_available"])
        self.assertIsNone(result["scored_rows"][1]["correct"])

    def test_official_temporal_and_abstention_routing(self):
        prompt = prompt_function()
        temporal = prompt("temporal-reasoning", "q", "2 months", "3 months")
        self.assertIn("off-by-one", temporal)
        self.assertIn("intermediate steps", temporal)
        ordinary = prompt("single-session-user", "q", "45", "45 minutes")
        self.assertNotIn("off-by-one", ordinary)
        abstention = prompt("single-session-user", "q", "unknown", "cannot know", True)
        self.assertIn("unanswerable", abstention)
        self.assertNotIn("Correct Answer:", abstention)

    def test_cache_requires_verified_model_and_valid_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            row = {"payload_hash": "a", "status": "ok", "returned_model": MODEL,
                   "content": "yes", "label": True, "finish_reason": "stop"}
            path.write_text(canonical(row) + "\n", encoding="utf-8")
            self.assertEqual(len(load_cache(path, {"a": {}})), 1)
            row["returned_model"] = "different"
            path.write_text(canonical(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cache(path, {"a": {}})


if __name__ == "__main__":
    unittest.main()
