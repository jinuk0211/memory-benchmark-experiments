"""Bounded child scheduling regressions; all child/model work is synthetic."""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_support import source_fixture

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("batch8_pool_tested", HERE / "runner.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class FakeChildren:
    def __init__(self, fail_qid=None, start_failure=None):
        self.fail_qid, self.start_failure = fail_qid, start_failure
        self.children, self.sealed = [], []
        self.max_live = 0

    def popen(self, argv, **kwargs):
        attempt = Path(argv[-1])
        qid = r.harness.read_json(attempt / "query.json")["question_id"]
        if qid == self.start_failure:
            raise OSError("synthetic spawn failure")
        owner = self
        class Child:
            pid = 1000 + len(owner.children)
            exited = False
            polls = 0
            waits = 0
            def finish(self):
                self.exited = True
                return 1 if qid == owner.fail_qid else 0
            def poll(self):
                self.polls += 1
                return self.finish() if self.polls >= 2 else None
            def wait(self):
                self.waits += 1
                return self.finish()
        child = Child()
        child.attempt, child.qid = attempt, qid
        self.children.append(child)
        self.max_live = max(self.max_live, sum(not c.exited for c in self.children))
        return child

    def finalize(self, attempt, identity):
        child = next(c for c in self.children if c.attempt == attempt)
        assert child.exited, "Cannot seal before child exit"
        self.sealed.append(child.qid)
        r.harness.save_json(attempt / "synthetic_sealed.json", {
            "identity": identity, "question_id": child.qid, "status": "generated", "hypothesis": "synthetic"})

    @staticmethod
    def verified(history, identity):
        for receipt in history.glob("attempt_*/synthetic_sealed.json"):
            prediction = r.harness.read_json(receipt)
            if prediction["identity"] != identity:
                raise ValueError("Synthetic identity differs")
            return prediction
        return None


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = source_fixture((2,))
        rows = [{"question_id": f"q{i}", "question": "synthetic question", "question_date": "2026/09/11",
                 "haystack_sessions": [s["turns"] for s in source],
                 "haystack_dates": [s["date"] for s in source],
                 "haystack_session_ids": [s["session_id"] for s in source]} for i in range(500)]
        dataset = self.root / "data.json"
        dataset.write_text(json.dumps(rows), encoding="utf-8")
        self.args = SimpleNamespace(dataset=dataset, run_dir=self.root / "run", source_root=self.root / "source",
            method="mem0", api_base="http://127.0.0.1:18083/v1", embedding_api_base="http://127.0.0.1:18084/v1",
            model="Qwen/Qwen3.5-9B", embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dims=384, tokenizer="/synthetic/tokenizer", ids_file=None, workers=4)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(r.harness, "DATA_SHA256", hashlib.sha256(dataset.read_bytes()).hexdigest()))
        self.stack.enter_context(patch.object(r, "validate_runtime"))
        self.stack.enter_context(patch.object(r, "source_hashes", return_value={"synthetic": "pinned"}))
        self.stack.enter_context(patch.object(r.time, "sleep"))

    def install(self, children):
        launch = self.stack.enter_context(patch.object(r.subprocess, "Popen", side_effect=children.popen))
        self.stack.enter_context(patch.object(r, "finalize_attempt", side_effect=children.finalize))
        self.stack.enter_context(patch.object(r, "verified_prediction", side_effect=children.verified))
        return launch

    def select(self, qids):
        self.args.ids_file = self.root / "ids.json"
        r.harness.save_json(self.args.ids_file, qids)

    def test_full_population_finishes_with_at_most_four_children(self):
        children = FakeChildren()
        launch = self.install(children)
        self.assertEqual(r.run(self.args), 0)
        self.assertEqual((launch.call_count, children.max_live, len(children.sealed)), (500, 4, 500))
        status = r.harness.read_json(self.args.run_dir / "status.json")
        self.assertEqual((status["planned"], status["population"], status["generated"], status["status"]),
                         (500, 500, 500, "generation_complete"))
        self.assertEqual(r.harness.read_json(self.args.run_dir / "current.json")["active"], [])

    def test_first_child_failure_stops_dispatch_and_drains_siblings(self):
        children = FakeChildren(fail_qid="q0")
        launch = self.install(children)
        self.assertEqual(r.run(self.args), 1)
        self.assertEqual(launch.call_count, 4)
        self.assertEqual(children.sealed, ["q1", "q2", "q3"])
        self.assertTrue(all(c.exited for c in children.children))
        status = r.harness.read_json(self.args.run_dir / "status.json")
        self.assertEqual((status["population"], status["generated"], status["failed"], status["status"]),
                         (500, 3, 1, "generation_incomplete"))

    def test_spawn_failure_still_drains_and_seals_started_children(self):
        children = FakeChildren(start_failure="q2")
        launch = self.install(children)
        self.assertEqual(r.run(self.args), 1)
        self.assertEqual(launch.call_count, 3)
        self.assertEqual(children.sealed, ["q0", "q1"])
        self.assertTrue(all(c.exited for c in children.children))

    def test_parent_publish_failure_preserves_original_error_and_successful_children(self):
        children = FakeChildren()
        launch = self.install(children)
        save = r.harness.save_json
        original_error = OSError("synthetic one-time aggregate storage failure")
        raised = False
        def fail_once(path, value):
            nonlocal raised
            if path.name == "status.json" and not raised:
                raised = True
                raise original_error
            return save(path, value)
        with patch.object(r.harness, "save_json", side_effect=fail_once):
            with self.assertRaises(OSError) as caught:
                r.run(self.args)
        self.assertIs(caught.exception, original_error)
        self.assertEqual(launch.call_count, 4)
        self.assertEqual(children.sealed, ["q0", "q1", "q2", "q3"])
        self.assertTrue(all(c.waits == 1 and c.exited for c in children.children))
        self.assertEqual(len(r.harness.read_json(self.args.run_dir / "predictions.json")), 4)

    def test_canary_keeps_500_protocol_and_reuses_cache_with_changed_workers(self):
        children = FakeChildren()
        launch = self.install(children)
        self.select(["q0"])
        self.args.workers = 1
        self.assertEqual(r.run(self.args), 0)
        protocol_bytes = (self.args.run_dir / "protocol.json").read_bytes()
        status = r.harness.read_json(self.args.run_dir / "status.json")
        self.assertEqual((status["population"], status["selected_count"], status["generated"], status["status"]),
                         (500, 1, 1, "subset_generation_complete"))
        protocol = json.loads(protocol_bytes)
        self.assertEqual(len(protocol["population_ids"]), 500)
        self.assertNotIn("workers", protocol["runtime"])
        self.assertNotIn("ids_file", protocol["runtime"])
        self.select(["q0", "q1"])
        self.args.workers = 2
        self.assertEqual(r.run(self.args), 0)
        self.assertEqual(launch.call_count, 2)
        self.assertEqual((self.args.run_dir / "protocol.json").read_bytes(), protocol_bytes)
        self.assertEqual([c.qid for c in children.children], ["q0", "q1"])

    def test_second_parent_rejected_before_protocol_or_worker_creation(self):
        launch = self.install(FakeChildren())
        with r.run_lock(self.args.run_dir):
            with self.assertRaises(OSError):
                r.run(self.args)
        launch.assert_not_called()
        self.assertFalse((self.args.run_dir / "protocol.json").exists())
        self.assertFalse((self.args.run_dir / "histories").exists())

    def test_invalid_selection_or_worker_count_never_launches(self):
        launch = self.install(FakeChildren())
        for selected in ([], ["q0", "q0"], ["absent"], [False]):
            with self.subTest(selected=selected):
                self.select(selected)
                with self.assertRaises(ValueError):
                    r.run(self.args)
        self.select(["q0"])
        for workers in (True, 0, 5):
            with self.subTest(workers=workers):
                self.args.workers = workers
                with self.assertRaises(ValueError):
                    r.run(self.args)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
