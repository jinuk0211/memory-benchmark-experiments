"""Behavioral checks for matched construction, random controls, and immutable runs."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_revised as runner


class FakeRuntime:
    @staticmethod
    def ntok(text: str) -> int:
        return len(text.split())


class RevisedTests(unittest.TestCase):
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

    @staticmethod
    def options(rt, sessions, parent, rows):
        unit = copy.deepcopy(parent[0])
        unit.update(index_text="Where did Alice go?", source_probe_id="p1")
        option = {"id": "p1:0", "kind": "single", "gain": 1.0,
                  "cost": runner.storage_cost(unit, rt.ntok), "unit": unit}
        return [[option]], {option["id"]: option}

    def test_temporal_toggle_keeps_canonical_headers_and_unit_identity(self) -> None:
        parent, unresolved, bound = runner.build_parents(self.rt, self.sessions, self.audited, self.plans)
        self.assertEqual(len(parent), len(unresolved))
        for normal, raw in zip(parent, unresolved):
            self.assertEqual({k: v for k, v in normal.items() if k != "text"},
                             {k: v for k, v in raw.items() if k != "text"})
            self.assertTrue(normal["text"].startswith("(Recorded on: 15 June 2023) "))
            self.assertTrue(raw["text"].startswith("(Recorded on: 15 June 2023) "))
            self.assertNotIn("3:00 pm", raw["text"])
            self.assertNotIn("original relative expression:", raw["text"])
        self.assertIn("yesterday", "\n".join(u["text"] for u in unresolved))
        self.assertIn("original relative expression:", "\n".join(u["text"] for u in parent))
        self.assertTrue(any(u["kind"] == "fact" for u in parent))
        self.assertTrue(any("Facts:\n" in u["text"] for u in bound))

    def test_independent_arms_and_accounting(self) -> None:
        before = copy.deepcopy(self.audited)
        with patch.object(runner, "make_options", self.options):
            memories, details, accounts = runner.all_memories(
                self.rt, self.sessions, self.initial, self.audited, [], self.plans, "test")
        self.assertEqual(self.audited, before)
        self.assertEqual(set(memories), set(runner.ARMS))
        base = memories["no_cues"]
        for arm in ("ours", "payload_keys", *runner.RANDOM_SEEDS):
            self.assertEqual(memories[arm][:len(base)], base)
            self.assertEqual(accounts[arm]["parent_sha256"], accounts["no_cues"]["parent_sha256"])
        self.assertEqual([u["text"] for u in memories["ours"]],
                         [u["text"] for u in memories["payload_keys"]])
        self.assertTrue(all(u.get("index_text", u["text"]) == u["text"] for u in memories["payload_keys"]))
        self.assertEqual(accounts["payload_keys"]["distinct_key_tokens"], 0)
        self.assertNotIn("Audit-only", "\n".join(u["text"] for u in memories["no_audit"]))
        for account in accounts.values():
            self.assertEqual(account["total_stored_tokens"], account["parent_payload_tokens"] +
                             account["cue_payload_tokens"] + account["distinct_key_tokens"])
            self.assertLessEqual(account["cue_payload_tokens"] + account["distinct_key_tokens"], 2000)
        self.assertEqual(details["ours"]["selected_options"], details["payload_keys"]["selected_options"])

    def test_ten_random_seeds_reproduce_and_keep_eligible_budget(self) -> None:
        groups, mapped = [], {}
        for i in range(30):
            group = []
            for j, kind in enumerate(("single", "full", "pair")):
                option = {"id": f"{i}:{j}", "kind": kind, "gain": 1,
                          "cost": 201, "unit": {"source_probe_id": str(i)}}
                group.append(option)
                mapped[option["id"]] = option
            groups.append(group)
        sequences = set()
        for seed in runner.RANDOM_SEEDS.values():
            selected, stats = runner.random_select(groups, mapped, "conv-test", seed)
            self.assertEqual((selected, stats), runner.random_select(groups, mapped, "conv-test", seed))
            self.assertTrue(all(o["kind"] != "pair" for o in selected))
            self.assertEqual(len(selected), len({o["unit"]["source_probe_id"] for o in selected}))
            self.assertLessEqual(stats["rounded_tokens"], 2000)
            self.assertLessEqual(stats["actual_tokens"], stats["rounded_tokens"])
            sequences.add(tuple(stats["selected_options"]))
        self.assertEqual(len(sequences), 10)

    def test_frozen_protocol_rejects_changed_reader_or_population(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            base = {"reader": "legacy", "questions_sha256": "population-a", "memory_locks": {"c": "a"}}
            runner.frozen_save(path, base)
            runner.frozen_save(path, copy.deepcopy(base))
            for key, value in (("reader", "grounded"), ("questions_sha256", "population-b"),
                               ("memory_locks", {"c": "b"})):
                with self.assertRaises(ValueError):
                    runner.frozen_save(path, {**base, key: value})

    def test_manifest_rejects_input_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text("original", encoding="utf-8")
            runner.core.save(root / "manifest.json", {"files": {"source.json": runner.sha(source)}, "code": {}})
            runner.verify_inputs(root)
            source.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.verify_inputs(root)


    def test_native_receipts_keep_empty_or_length_answers_but_reject_bad_counts(self) -> None:
        for prediction, reason in (("", "stop"), ("unfinished", "length")):
            receipt = {"text": prediction, "finish_reason": reason,
                       "input_tokens": 25, "output_tokens": 96}
            runner.validate_receipt(receipt, prediction, 25)
            for field, invalid in (("input_tokens", True), ("output_tokens", -1),
                                   ("output_tokens", 97), ("input_tokens", 26), ("finish_reason", "abort")):
                with self.assertRaises(ValueError):
                    runner.validate_receipt({**receipt, field: invalid}, prediction, 25)


if __name__ == "__main__":
    unittest.main()
