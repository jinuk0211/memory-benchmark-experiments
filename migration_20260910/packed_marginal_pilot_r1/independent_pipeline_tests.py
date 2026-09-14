"""Independent synthetic orchestration checks, without model or network use."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest.mock import patch

PILOT_PATH = Path(__file__).with_name("pilot.py")
SPEC = importlib.util.spec_from_file_location("independent_pilot_under_test", PILOT_PATH)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


class PipelineContract(unittest.TestCase):
    def _run(self, *, other_fit_harmed: bool, qb_missing: bool = False) -> dict:
        with tempfile.TemporaryDirectory(prefix="independent_", dir=PILOT_PATH.parent) as raw:
            out = Path(raw)
            original_rows = [
                {"id": "p1", "question": "Q0-one", "answer": "A-one", "split": "probe_fit"},
                {"id": "p2", "question": "Q0-two", "answer": "A-two", "split": "probe_fit"},
            ]
            fit = [{**row, "question": f"QA-{i}"} for i, row in enumerate(original_rows)]
            audit = [{"id": "a1", "question": "AUDIT", "answer": "A-audit", "split": "probe_audit"}]
            seed = [{"text": "seed", "sources": ["S1"]}]
            option = {"id": "p1-option", "cost": 8, "gain": 1.0,
                      "unit": {"text": "alias", "sources": ["S1"], "source_probe_id": "p1"}}
            history = {"history_id": "synthetic", "seed": seed, "fit": original_rows,
                       "audit": audit, "groups": [[option]], "qb_ids": [] if qb_missing else ["p1"]}
            protocol = {"histories": {"synthetic": history}, "adapter_root": "unused",
                        "v2_root": "unused", "environment": {}}
            bundle = {"manifest": {"records": [{"original": original_rows[0],
                         "id": "p1", "diagnostic_question": "QB"}]}}
            out.joinpath("protocol.json").write_text(json.dumps(protocol))
            out.joinpath("views").mkdir()
            out.joinpath("views", "synthetic.json").write_text(json.dumps(bundle))
            events: list[str] = []

            def retrieve(index: object, memory: list[dict], questions: list[str]) -> list[dict]:
                for question in questions:
                    if question == "QB":
                        self.assertTrue((out / "memory_lock.json").exists())
                        events.append("qb_after_lock")
                text = "augmented" if len(memory) > len(seed) else "seed"
                return [{"context": text, "source_ids": ["S1"], "read_tokens": 1} for _ in questions]

            class Scorer:
                def __init__(self, *args: object) -> None:
                    self.ntok = lambda text: len(text.split())

                def score(self, jobs: list[dict]) -> None:
                    for job in jobs:
                        question, context = json.loads(job["user"])
                        if question == "QA-0":
                            lp = -1.0 if context == "augmented" else -3.0
                        elif question == "QA-1":
                            lp = -5.0 if context == "augmented" and other_fit_harmed else -1.0
                        elif question == "Q0-one":
                            lp = -1.0 if context else -3.0
                        elif question == "QB":
                            if not (out / "memory_lock.json").exists():
                                raise AssertionError("QB used before final memory lock")
                            lp = -10.0 if context == "augmented" else -1.0
                        else:
                            lp = -1.0
                        job["score"] = {"mean_logprob": lp, "answer_tokens": 1,
                                        "sum_logprob": lp, "token_logprobs": [lp]}

                def generate(self, controls: dict) -> dict:
                    return {key: {"text": job["answer"], "source_answer_f1": 1.0}
                            for key, job in controls.items()}

            def allocate(groups: list[list[dict]], budget: int, quantum: int) -> tuple[list[dict], dict]:
                selected = [copy.deepcopy(group[0]) for group in groups if group]
                return selected, {"actual_tokens": sum(option["cost"] for option in selected)}

            support = SimpleNamespace(
                transfer_runtime=SimpleNamespace(Scorer=Scorer),
                runtime_meter=SimpleNamespace(set_runtime_context=lambda *a, **k: None),
                source_views_v2=SimpleNamespace(fit_rows=lambda *a: copy.deepcopy(fit)),
                recursive_memory=SimpleNamespace(retrieve=retrieve),
                evidence_utility=SimpleNamespace(reader_user=lambda q, c: json.dumps([q, c])),
                budgeted_evidence=SimpleNamespace(multiple_choice_budget=allocate),
            )
            with patch.object(pilot, "verified", return_value=protocol), \
                    patch.object(pilot, "support", return_value=(support, {})), \
                    patch.object(pilot, "checked_views", return_value=bundle), \
                    patch.object(pilot, "index_runtime", return_value=object()), \
                    patch.object(pilot, "calibration_status", return_value={"status": "ready"}), \
                    patch.object(pilot, "generation_guard", return_value=None):
                pilot.score(out)
            return {
                "construction": json.loads(out.joinpath("construction", "synthetic.json").read_text()),
                "memory": json.loads(out.joinpath("memories", "synthetic.json").read_text()),
                "qb": json.loads(out.joinpath("qb", "synthetic.json").read_text()),
                "complete": json.loads(out.joinpath("COMPLETE.json").read_text()),
                "seed": seed, "events": events,
            }

    def test_own_query_gain_cannot_override_whole_set_loss(self) -> None:
        result = self._run(other_fit_harmed=True)
        self.assertFalse(result["construction"]["adopted"])
        self.assertLess(result["construction"]["whole_set_qa_delta"], 0)
        self.assertGreater(result["construction"]["option_details"][0]["marginal_value"], 0)
        self.assertEqual(result["memory"], result["seed"])
        self.assertTrue(result["events"])

    def test_negative_qb_is_recorded_without_changing_locked_selection(self) -> None:
        result = self._run(other_fit_harmed=False)
        self.assertTrue(result["construction"]["adopted"])
        self.assertLess(result["qb"]["mean_logprob_delta"], 0)
        self.assertEqual(len(result["memory"]), 2)
        self.assertFalse(result["qb"]["selection_allowed"])
        self.assertTrue(result["events"])

    def test_zero_qa_records_stop_before_scorer_or_memory_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="independent_", dir=PILOT_PATH.parent) as raw:
            out = Path(raw)
            out.joinpath("views").mkdir()
            out.joinpath("views", "synthetic.json").write_text("{}")
            protocol = {"histories": {"synthetic": {}}, "adapter_root": "unused", "v2_root": "unused"}
            no_model = SimpleNamespace()
            with patch.object(pilot, "verified", return_value=protocol), \
                    patch.object(pilot, "support", return_value=(no_model, {})), \
                    patch.object(pilot, "checked_views", return_value={"accepted_count": 0}), \
                    patch.object(pilot, "index_runtime", side_effect=AssertionError("must not initialize")):
                pilot.score(out)
            status = json.loads(out.joinpath("CALIBRATION_STATUS.json").read_text())
            self.assertEqual(status["status"], "insufficient_evidence")
            self.assertFalse(status["packed_memory_scoring_performed"])
            self.assertFalse(out.joinpath("memory_lock.json").exists())
            self.assertFalse(out.joinpath("COMPLETE.json").exists())

    def test_nonfinite_mapped_answerability_cannot_admit_option(self) -> None:
        before = {"context": "before", "score": {"mean_logprob": -3.0}}
        after = {"context": "after", "score": {"mean_logprob": -1.0}}
        mapped = {"score": {"mean_logprob": float("nan")}}
        empty = {"score": {"mean_logprob": -3.0}}
        with self.assertRaises(ValueError):
            pilot.value(after, before, mapped, empty)
        after["context"] = "before"
        self.assertEqual(pilot.value(after, before, mapped, empty), 0.0)

    def test_changed_question_or_denominator_refuses_comparison(self) -> None:
        before = [{"probe_id": "p", "question": "Q", "answer": "A", "score": {"mean_logprob": -2.0}}]
        after = copy.deepcopy(before)
        after[0]["question"] = "Another question"
        with self.assertRaises(ValueError):
            pilot.paired_delta(before, after)
        with self.assertRaises(ValueError):
            pilot.paired_delta(before, [])
    def test_no_qb_is_explicit_null_not_positive_evidence(self) -> None:
        result = self._run(other_fit_harmed=False, qb_missing=True)
        self.assertEqual(result["qb"]["count"], 0)
        self.assertIsNone(result["qb"]["mean_logprob_delta"])
        self.assertIsNone(result["qb"]["final_answers"]["mean_f1"])
        self.assertEqual(result["events"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)


