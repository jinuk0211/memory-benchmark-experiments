"""Compare the final-reader boundary to the actual frozen source, without APIs."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from controlled_reader import reader_request, validate_reference

WORKSPACE = Path(__file__).resolve().parents[2]
REFERENCE = WORKSPACE / "generalization_20260908/full_transfer/verified_qwen35_lme500_start/protocol.json"


class ControlledReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(REFERENCE.read_text(encoding="utf-8"))
        path = WORKSPACE / "generalization_20260908/source/refine.py"
        spec = importlib.util.spec_from_file_location("frozen_reader_reference", path)
        cls.core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.core)

    def test_matches_frozen_packer_including_separators_and_skips(self):
        cases = [[], ["", "a"], ["  ", "a"], ["x" * 2049, "evidence"], ["x" * 2047, "y", "z"], ["a", "b", "c"]]
        for contexts in cases:
            with self.subTest(lengths=list(map(len, contexts))):
                expected = self.core.pack(
                    [{"text": text} for text in contexts], range(len(contexts)), len, 2048
                )
                _, evidence = reader_request(contexts, "Which team?", "2026/01/01", len, self.protocol)
                actual = (evidence["context"], evidence["selected_indices"], evidence["read_tokens"])
                self.assertEqual(expected, actual)

    def test_counts_joined_text_not_sum_of_individual_costs(self):
        def count(text):
            return len(text) + (2048 if "\n\n" in text else 0)
        _, evidence = reader_request(["a", "b"], "Q", "D", count, self.protocol)
        self.assertEqual(evidence["selected_indices"], [0])

    def test_same_prompt_and_generation_limits_as_reference(self):
        request, evidence = reader_request(["Teams since March."], "Which app?", "2026/04/01", len, self.protocol)
        self.assertEqual(request["messages"][0]["content"], self.core.READER)
        self.assertEqual(request["messages"][1]["content"],
                         "Conversation memory:\nTeams since March.\n\nQuestion date: 2026/04/01\nQuestion: Which app?\nAnswer:")
        self.assertEqual(request["max_tokens"], 96)
        self.assertEqual(request["temperature"], 0)
        self.assertFalse(request["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(evidence["retrieved_candidates"], ["Teams since March."])

    def test_empty_retrieval_is_preserved_for_abstention(self):
        request, evidence = reader_request([], "Unsupported?", "D", len, self.protocol)
        self.assertEqual(evidence["read_tokens"], 0)
        self.assertIn("Conversation memory:\n\n\nQuestion date:", request["messages"][1]["content"])

    def test_rejects_changed_experimental_condition(self):
        for key, value in [("read_budget", 4096), ("max_answer_tokens", 256),
                           ("target_selection", True), ("history_truncation", 0), ("reader_prompt", "different instructions")]:
            with self.subTest(key=key):
                protocol = copy.deepcopy(self.protocol)
                protocol[key] = value
                with self.assertRaises(ValueError):
                    validate_reference(protocol)

    def test_rejects_concatenated_string(self):
        with self.assertRaises(TypeError):
            reader_request("one string", "Q", "D", len, self.protocol)


if __name__ == "__main__":
    unittest.main()
