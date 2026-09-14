"""Behavioral checks for ablation isolation, budgets, and selection."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ablation as runner


class FakeRuntime:
    @staticmethod
    def ntok(text: str) -> int:
        return len(text.split())


class AblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rt = FakeRuntime()
        self.sessions = [{"num": 1, "date": "3:00 pm on 15 June, 2023", "turns": [
            {"id": f"D1:{i}", "speaker": "Alice", "body": body, "text": f"D1:{i}: Alice: {body}"}
            for i, body in enumerate(("I visited London yesterday.", "I bought a book.",
                                      "The book was red.", "I read it last night."), 1)]}]
        self.initial = [{"text": "Alice visited London yesterday.", "kind": "fact",
                         "sources": ["D1:1"], "session": 1}]
        self.audited = self.initial + [{"text": "Audit-only fact: Alice bought a red book.",
                                       "kind": "fact", "sources": ["D1:2", "D1:3"], "session": 1}]
        self.plans = {
            "r06_calendar_month": {"recipe": {"operations": [{"op": "anchor_time", "style": "month"}]}},
            "r12_filter_current_best": {"recipe": {"operations": [{"op": "filter_social_facts"}]}},
            "r40_fused_four_turn": {"recipe": {"operations": [{"op": "fuse_evidence", "size": 4,
                                                               "overlap": 2, "anchor": True}]}}}

    def options(self, rt: Any, sessions: list, parent: list, rows: list) -> tuple[list, dict]:
        unit = copy.deepcopy(parent[0])
        unit.update(index_text="Where did Alice go?", source_probe_id="p1")
        option = {"id": "p1:0", "kind": "single", "gain": 1.0,
                  "cost": runner.storage_cost(unit, rt.ntok), "unit": unit}
        return [[option]], {option["id"]: option}

    def test_independent_removals_and_payload_preservation(self) -> None:
        initial_before, audited_before = copy.deepcopy(self.initial), copy.deepcopy(self.audited)
        with patch.object(runner, "make_options", self.options):
            memories, details = runner.all_memories(
                self.rt, self.sessions, self.initial, self.audited, [], self.plans, "conv-test")
        self.assertEqual(set(memories), {"ours", "no_cues", "no_audit", "no_temporal", "no_binding", "random_cues", "payload_keys"})
        self.assertEqual(self.initial, initial_before)
        self.assertEqual(self.audited, audited_before)
        def text(arm: str) -> str:
            return "\n".join(u["text"] for u in memories[arm])
        self.assertIn("Audit-only fact:", text("ours"))
        self.assertNotIn("Audit-only fact:", text("no_audit"))
        self.assertIn("original relative expression:", text("ours"))
        self.assertNotIn("original relative expression:", text("no_temporal"))
        self.assertIn("yesterday", text("no_temporal"))
        self.assertNotIn("Facts:\n", text("no_binding"))
        self.assertTrue(any(u["kind"] == "fact" for u in memories["no_binding"]))
        self.assertTrue(any(u["kind"] == "dialogue_block" for u in memories["no_binding"]))
        self.assertEqual([u["text"] for u in memories["ours"]],
                         [u["text"] for u in memories["payload_keys"]])
        self.assertEqual(len(memories["ours"]), len(memories["payload_keys"]))
        self.assertTrue(all(u.get("index_text", u["text"]) == u["text"]
                            for u in memories["payload_keys"]))
        self.assertEqual(memories["ours"][:len(memories["no_cues"])], memories["no_cues"])
        for arm in ("ours", "no_audit", "no_temporal", "no_binding"):
            self.assertLessEqual(details[arm]["storage_tokens"] - details[arm]["parent_tokens"], 2000)

    def test_random_budget_one_per_probe_and_reproducibility(self) -> None:
        groups, mapped = [], {}
        for group_id in range(30):
            group = []
            for j, kind in enumerate(("single", "full", "pair")):
                option = {"id": f"{group_id}:{j}", "kind": kind, "gain": 1,
                          "cost": 201, "unit": {"source_probe_id": str(group_id)}}
                group.append(option)
                mapped[option["id"]] = option
            groups.append(group)
        selected, stats = runner.random_select(groups, mapped, "conv-1")
        self.assertEqual((selected, stats), runner.random_select(groups, mapped, "conv-1"))
        self.assertTrue(all(o["kind"] != "pair" for o in selected))
        self.assertEqual(len(selected), len({o["unit"]["source_probe_id"] for o in selected}))
        self.assertLessEqual(stats["rounded_tokens"], 2000)
        self.assertLessEqual(stats["actual_tokens"], stats["rounded_tokens"])

    def test_input_fingerprint_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.json").write_text("original")
            runner.core.save(root / "manifest.json",
                             {"files": {"source.json": runner.sha(root / "source.json")}, "code": {}})
            runner.verify_inputs(root)
            (root / "source.json").write_text("tampered")
            with self.assertRaises(ValueError):
                runner.verify_inputs(root)


if __name__ == "__main__":
    unittest.main()
