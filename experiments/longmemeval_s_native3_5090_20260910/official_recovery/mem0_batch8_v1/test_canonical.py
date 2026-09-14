"""Full canonical source coverage oracle; no answer fields or model calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("batch8_canonical_tested", HERE / "runner.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class CanonicalTests(unittest.TestCase):
    def test_all_500_inputs_preserved_with_session_local_batch_counts(self):
        candidates = [r.ROOT / "longmemeval_s_cleaned.json",
                      r.ROOT.parents[1] / "MemoryData/datasets/LongMemEval/longmemeval_s_cleaned.json"]
        dataset = next((p for p in candidates if p.is_file()), None)
        if dataset is None:
            self.skipTest("Canonical dataset is not staged at either documented path")
        data = dataset.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), r.harness.DATA_SHA256)
        records = json.loads(data)
        self.assertEqual(len(records), 500)
        sessions = turns = pairs = batch_count = odd_sessions = partial_sessions = empty_turns = 0
        counts = {}
        for record in records:
            source = r.harness.source_only(record)
            batches = list(r.session_batches(source))
            for index, session in enumerate(source):
                selected = [(meta, messages) for meta, messages in batches if meta["session_index"] == index]
                self.assertEqual([message for _, messages in selected for message in messages], session["turns"])
                self.assertTrue(all(meta["session_id"] == session["session_id"] and meta["session_date"] == session["date"]
                                    and 1 <= len(messages) <= 16 for meta, messages in selected))
            ns = len(source)
            nt = sum(len(session["turns"]) for session in source)
            np = sum((len(session["turns"]) + 1) // 2 for session in source)
            counts[str(record["question_id"])] = (ns, nt, np, len(batches))
            empty_turns += sum(turn["content"] == "" for session in source for turn in session["turns"])
            sessions += ns
            turns += nt
            pairs += np
            batch_count += len(batches)
            odd_sessions += sum(len(session["turns"]) % 2 for session in source)
            partial_sessions += sum(bool(len(session["turns"]) % 16) for session in source)
        self.assertEqual((sessions, turns, pairs, batch_count, odd_sessions, partial_sessions),
                         (23867, 246750, 124345, 24365, 1940, 23732))
        self.assertEqual(empty_turns, 12)
        self.assertEqual(counts["e47becba"], (53, 550, 277, 53))
        self.assertEqual(counts["852ce960"][:2], (39, 396))


if __name__ == "__main__":
    unittest.main()
