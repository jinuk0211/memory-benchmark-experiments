"""Source isolation tests with adversarial evaluation metadata."""
import copy
import unittest
from prepare_inputs import source_only


class SourceIsolationTests(unittest.TestCase):
    def test_evaluation_fields_cannot_change_source(self):
        original = {
            "haystack_session_ids": ["s1"], "haystack_dates": ["2025-03-01"],
            "haystack_sessions": [[{"role": "user", "content": "Use Teams.", "has_answer": True}]],
            "question": "Which app?", "answer": "Teams", "answer_session_ids": ["s1"],
            "question_type": "knowledge-update", "question_date": "2025-04-01",
        }
        changed = copy.deepcopy(original)
        for key in ("question", "answer", "answer_session_ids", "question_type", "question_date"):
            changed[key] = "POISON TARGET METADATA"
        changed["haystack_sessions"][0][0]["has_answer"] = "POISON LABEL"
        self.assertEqual(source_only(original), source_only(changed))
        self.assertEqual(source_only(original), [{
            "session_id": "s1", "date": "2025-03-01",
            "turns": [{"role": "user", "content": "Use Teams."}],
        }])

    def test_preserves_long_turns_both_roles_and_dates(self):
        text = "entire long original " * 10000
        record = {"haystack_session_ids": ["s1", "s2"],
                  "haystack_dates": ["2020-01-01", "2026-01-01"],
                  "haystack_sessions": [[{"role": "user", "content": text}],
                                        [{"role": "assistant", "content": "correction"}]]}
        source = source_only(record)
        self.assertEqual(source[0]["turns"][0]["content"], text)
        self.assertEqual(source[1]["turns"][0]["role"], "assistant")
        self.assertEqual(source[1]["date"], "2026-01-01")

    def test_rejects_misaligned_sessions(self):
        with self.assertRaises(ValueError):
            source_only({"haystack_sessions": [[]], "haystack_dates": [], "haystack_session_ids": ["s"]})


if __name__ == "__main__":
    unittest.main()
