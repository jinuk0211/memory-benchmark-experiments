import unittest
from run_semantic_judge import parse_label, summarize


class Common50Tests(unittest.TestCase):
    def test_official_period_verdicts(self):
        for value in ("Yes", "Yes.", " yes.\n"):
            self.assertTrue(parse_label(value))
        for value in ("No", "No.", " no.\n"):
            self.assertFalse(parse_label(value))
        for value in ("not yes", "yes or no", "", None):
            with self.assertRaises(ValueError):
                parse_label(value)

    def test_same50_requires_all50(self):
        rows = [{"method": "m", "question_id": str(i), "payload_hash": str(i),
                 "hypothesis": "a"} for i in range(50)]
        cache = {str(i): {"label": i < 40} for i in range(50)}
        score = summarize(rows, cache)["methods"]["m"]
        self.assertEqual(score["expected"], 50)
        self.assertEqual(score["correct"], 40)
        self.assertEqual(score["accuracy_on_same50"], 0.8)
        del cache["49"]
        partial = summarize(rows, cache)["methods"]["m"]
        self.assertIsNone(partial["accuracy_on_same50"])
        self.assertIsNone(partial["accuracy_on_available"])
        self.assertEqual(partial["pending_judgments"], 1)
        self.assertEqual(partial["missing_predictions"], 0)


if __name__ == "__main__":
    unittest.main()
