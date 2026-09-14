"""CPU contract tests for the frozen runner; no model downloads or GPU use."""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_transfer as runner
from transfer_data import adapt_longmemeval

core = runner.core


class FakeRuntime:
    """Deterministic embeddings/answers with the actual generation cache contract."""

    def __init__(self, cache: Path, *, output_mode: str = "normal") -> None:
        self.cache = cache
        self.args = SimpleNamespace(seed=20260908)
        self.model_meta = {"model": "cpu-test-only"}
        self.output_mode = output_mode
        self.calls: list[tuple[str, list[str], int]] = []
        self.encodes: list[tuple[list[str], bool]] = []

    @staticmethod
    def ntok(text: str) -> int:
        return len(text.split())

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        self.encodes.append((list(texts), query))
        keywords = ("degree", "history", "travel", "paris", "hobbies", "hiking", "music")
        vectors = np.asarray([
            [1.0] + [float(core.lexical(text).count(word)) for word in keywords]
            for text in texts
        ], dtype=np.float64)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def generate(self, system: str, users: list[str], max_tokens: int = 96) -> list[str]:
        self.calls.append((system, list(users), max_tokens))
        if self.output_mode == "missing":
            return []
        answers = {
            "What is my degree?": "history",
            "Where did I travel?": "Paris",
            "What are my hobbies?": "hiking, music",
        }
        predictions = []
        for user in users:
            question = user.rsplit("Question: ", 1)[1].split("\nAnswer:", 1)[0]
            prediction = "" if self.output_mode == "empty" else answers.get(question, "unknown")
            key = core.digest([
                self.model_meta, self.args.seed, system, user, max_tokens, False
            ])
            core.save(self.cache / "generations" / (key + ".json"), {
                "text": prediction,
                "input_tokens": self.ntok(system + " " + user),
                "output_tokens": self.ntok(prediction),
                "finish_reason": "stop",
            })
            predictions.append(prediction)
        return predictions


def longmemeval_sample() -> dict:
    return adapt_longmemeval([{
        "question_id": "qid_SECRET",
        "question_type": "TYPE_SECRET",
        "question": "What is my degree?",
        "question_date": "2023/05/30 (Tue) 23:40",
        "answer": "GOLD_SECRET",
        "answer_session_ids": ["answer_SESSION_SECRET"],
        "haystack_session_ids": ["answer_SESSION_SECRET", "other_SESSION_SECRET"],
        "haystack_dates": ["2023/05/01 10:00", "2023/05/02 11:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "My degree is history.", "has_answer": True},
             {"role": "assistant", "content": "You studied history.", "has_answer": False}],
            [{"role": "user", "content": "I travelled to Paris and enjoy hiking and music."}],
        ],
    }])[0]


def memory_units(sample: dict, *, indexed: bool = False) -> list[dict]:
    units = core.raw_units(core.session_data(sample))
    units.append({
        "text": "oversized " * 2050,
        "sources": ["D9:1"],
        "session": 9,
        "kind": "raw",
    })
    if indexed:
        for index, unit in enumerate(units):
            unit["index_text"] = f"INDEX_ONLY_SECRET {index} " + unit["text"]
    return units


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_gold_evidence_ids_and_type_cannot_change_prompts_or_retrieval(self) -> None:
        sample = longmemeval_sample()
        changed = copy.deepcopy(sample)
        changed["sample_id"] = "DIFFERENT_SECRET_abs"
        changed["qa"][0].update(
            answer=90210, category="OTHER_TYPE_SECRET",
            question_type="OTHER_TYPE_SECRET", evidence=["D2:1"], is_abstention=True,
        )
        changed["evaluation_metadata"]["question_id"] = changed["sample_id"]
        for entry in changed["evaluation_metadata"]["session_map"]:
            entry["original_id"] = "OTHER_SESSION_SECRET"
            entry["is_answer_session"] = not entry["is_answer_session"]
        units = memory_units(sample, indexed=True)
        baseline = FakeRuntime(self.root / "baseline")
        altered = FakeRuntime(self.root / "altered")
        before = runner.evaluate_sample(baseline, units, sample, "seed", "longmemeval")
        after = runner.evaluate_sample(altered, units, changed, "seed", "longmemeval")
        self.assertEqual(baseline.calls, altered.calls)
        self.assertEqual(baseline.encodes, altered.encodes)
        self.assertEqual(before[0]["context"], after[0]["context"])
        self.assertEqual(before[0]["source_ids"], after[0]["source_ids"])
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0]["question_id"], "DIFFERENT_SECRET_abs")
        self.assertTrue(after[0]["is_abstention"])
        prompt = baseline.calls[0][1][0]
        self.assertIn("Question date: 2023/05/30 (Tue) 23:40", prompt)
        self.assertIn("Session date: 2023/05/01 10:00", prompt)
        self.assertNotIn("SECRET", prompt)
        self.assertLessEqual(before[0]["read_tokens"], 2048)
        self.assertNotIn("D9:1", before[0]["source_ids"])

    def test_question_date_changes_reader_prompt_but_not_retrieval(self) -> None:
        sample = longmemeval_sample()
        later = copy.deepcopy(sample)
        later["qa"][0]["question_date"] = "2024/08/01 01:00"
        first = FakeRuntime(self.root / "first")
        second = FakeRuntime(self.root / "second")
        units = memory_units(sample)
        before = runner.evaluate_sample(first, units, sample, "seed", "longmemeval")
        after = runner.evaluate_sample(second, units, later, "seed", "longmemeval")
        self.assertEqual(before[0]["context"], after[0]["context"])
        self.assertEqual(first.encodes, second.encodes)
        self.assertNotEqual(first.calls, second.calls)
        self.assertIn("Question date: 2024/08/01 01:00", second.calls[0][1][0])

    def test_missing_generations_fail_and_empty_answers_retain_the_row(self) -> None:
        sample = longmemeval_sample()
        units = memory_units(sample)
        with self.assertRaisesRegex(ValueError, "Missing generation outputs"):
            runner.evaluate_sample(
                FakeRuntime(self.root / "missing", output_mode="missing"),
                units, sample, "seed", "longmemeval",
            )
        rows = runner.evaluate_sample(
            FakeRuntime(self.root / "empty", output_mode="empty"),
            units, sample, "seed", "longmemeval",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question_id"], sample["sample_id"])
        self.assertEqual(rows[0]["prediction"], "")
        self.assertEqual(rows[0]["diagnostic_token_f1"], 0)

    def test_empty_memory_fails_instead_of_losing_a_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "Empty memory"):
            runner.evaluate_sample(
                FakeRuntime(self.root), [], longmemeval_sample(), "seed", "longmemeval"
            )

    @unittest.skipUnless(importlib.util.find_spec("nltk"), "Frozen LoCoMo metric requires nltk")
    def test_locomo_replays_original_and_indexed_evaluators_with_unique_qa_ids(self) -> None:
        import evaluate_indexed

        sample = longmemeval_sample()
        sample["sample_id"] = "conversation"
        sample["qa"] = [
            {"question": "What is my degree?", "answer": "history; arts",
             "category": 3, "evidence": ["D1:1"]},
            {"question": "Where did I travel?", "answer": "Paris",
             "category": 2, "evidence": ["D2:1"]},
            {"question": "What are my hobbies?", "answer": "hiking, music",
             "category": 1, "evidence": ["D2:1"]},
            {"question": "What is my radio brand?", "answer": "radio",
             "category": 4, "evidence": []},
            {"question": "Excluded adversarial?", "answer": "unused",
             "category": 5, "evidence": []},
        ]
        # Repeated equal QA dictionaries must retain distinct original question IDs.
        sample["qa"].append(copy.deepcopy(sample["qa"][0]))
        records = [
            {"id": f"conversation:{index}", "conv_id": "conversation",
             "qa_index": index, "category": qa["category"], "split": "holdout"}
            for index, qa in enumerate(sample["qa"]) if qa["category"] in (1, 2, 3, 4)
        ]
        for indexed in (False, True):
            with self.subTest(indexed=indexed):
                units = memory_units(sample, indexed=indexed)
                current = FakeRuntime(self.root / f"current-{indexed}")
                reference = FakeRuntime(self.root / f"reference-{indexed}")
                actual = runner.evaluate_sample(current, units, sample, "seed", "locomo")
                expected = evaluate_indexed.evaluate(
                    reference, {"conversation": units}, records,
                    {"conversation": sample}, 2048, "seed", self.root / f"items-{indexed}",
                )
                self.assertEqual(current.calls, reference.calls)
                self.assertEqual(current.encodes, reference.encodes)
                self.assertEqual(
                    [row["question_id"] for row in actual],
                    ["conversation:0", "conversation:1", "conversation:2",
                     "conversation:3", "conversation:5"],
                )
                self.assertEqual(len(actual), len(expected))
                for row, original in zip(actual, expected):
                    for key in ("context", "source_ids", "read_tokens", "prediction",
                                "official_f1", "memory_tokens", "memory_units"):
                        self.assertEqual(row[key], original[key], key)
                self.assertEqual([row["official_f1"] for row in actual], [1, 1, 1, 0, 1])

    def test_lock_rejects_drift_and_keeps_original_artifact(self) -> None:
        path = self.root / "lock.json"
        original = {"method": "frozen", "budget": 2048}
        runner.locked(path, original)
        runner.locked(path, copy.deepcopy(original))
        with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
            runner.locked(path, {**original, "budget": 1024})
        self.assertEqual(runner.read(path), original)


if __name__ == "__main__":
    unittest.main()

