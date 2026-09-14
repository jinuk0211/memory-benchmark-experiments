"""Regression checks for complete coverage and safe Qwen-to-baseline handoff."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from verify_qwen_completion import (
    NotReady, qwen_source, reference_digest, require_idle_gpu,
    verify_arm, verify_artifacts,
)


class QwenCompletionTests(unittest.TestCase):
    def test_source_and_digest_match_original_implementation(self):
        import ast
        import hashlib
        import re

        path = Path(__file__).resolve().parents[2] / "generalization_20260908/source/refine.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ("session_data", "digest")]
        scope = {"json": json, "hashlib": hashlib, "re": re}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
        record = dict(
            haystack_dates=["2023/05/20 (Sat) 02:21"],
            haystack_sessions=[[dict(role="user", content="a\n\nb", has_answer=True),
                                dict(role="assistant", content="reply")]],
            question="must not appear", answer="must not appear",
        )
        sample = dict(conversation={
            "session_1_date_time": record["haystack_dates"][0],
            "session_1": [dict(dia_id=f"D1:{i}", speaker=turn["role"], text=turn["content"])
                          for i, turn in enumerate(record["haystack_sessions"][0], start=1)],
        })
        source = qwen_source(record)
        self.assertEqual(source, scope["session_data"](sample))
        self.assertEqual(reference_digest(source), scope["digest"](source))

    def test_full500_empty_results_kept_and_bad_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = root / "items"
            items.mkdir()
            expected = {
                f"q{i}": dict(question=f"Question{i}", question_date="date",
                              question_type="multi-session", answer="gold")
                for i in range(500)
            }
            rows = []
            for i, (qid, target) in enumerate(expected.items()):
                text = "" if i == 0 else "prediction"
                row = dict(question_id=qid, conversation_id=qid, method="seed",
                           question=target["question"], question_date="date",
                           question_type="multi-session", gold="gold", is_abstention=False,
                           hypothesis=text, prediction=text,
                           status="ok" if text else "generation_empty",
                           finish_reason="length" if i == 1 else "stop",
                           read_tokens=2048, output_tokens=96)
                rows.append(row)
                (items / f"{qid}.json").write_text(json.dumps([row]))
            path = root / "seed.jsonl"

            def save(values):
                path.write_text("\n".join(json.dumps(row) for row in values) + "\n")

            def replay_question(qid):
                return {"hypothesis": rows[int(qid[1:])]["hypothesis"]}

            save(rows)
            verified = verify_arm(path, "seed", expected, items, replay_question)
            self.assertEqual(verified["questions"], 500)
            self.assertEqual(verified["empty_generations"], 1)
            self.assertEqual(verified["length_stops"], 1)
            for bad in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
                save(bad)
                with self.assertRaisesRegex(ValueError, "exactly500"):
                    verify_arm(path, "seed", expected, items, replay_question)
            save(rows)
            (items / "q499.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "aggregate and per-item"):
                verify_arm(path, "seed", expected, items, replay_question)

    def test_partial_run_cannot_produce_completion_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "status.json").write_text(json.dumps(dict(phase="building_seed")))
            with self.assertRaises(NotReady):
                verify_artifacts(root, root, root / "not_read.json")

    def test_busy_gpu_and_failed_observation_do_not_pass(self):
        with patch("verify_qwen_completion.subprocess.run",
                   return_value=SimpleNamespace(stdout="26746\n")):
            with self.assertRaises(NotReady):
                require_idle_gpu()
        with patch("verify_qwen_completion.subprocess.run", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                require_idle_gpu()


if __name__ == "__main__":
    unittest.main()