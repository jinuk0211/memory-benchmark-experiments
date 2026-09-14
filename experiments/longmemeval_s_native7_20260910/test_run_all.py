"""Queue gate tests use synthetic data and mock subprocesses; no inference."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_all as queue


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "data.json"
        self.dataset.write_text(json.dumps([{"question_id": f"qid-{i}"} for i in range(500)]), encoding="utf-8")
        self.data_sha = hashlib.sha256(self.dataset.read_bytes()).hexdigest()
        self.receipt = self.root / "runtime_receipt.json"
        self.receipt.write_text('{"status": "runtime_verified"}', encoding="utf-8")
        self.commands = {method: ["separate-python", f"native-{method}.py", "--dataset", str(self.dataset),
                                  "--run-dir", str(self.root / method)] for method in queue.METHODS}
        self.plan = {
            "dataset": str(self.dataset), "dataset_sha256": self.data_sha,
            "runtime_receipt": {"path": str(self.receipt),
                                "sha256": hashlib.sha256(self.receipt.read_bytes()).hexdigest()},
            "runner_commands": self.commands}
        self.plan_path = self.root / "plan.json"
        queue.save_json(self.plan_path, self.plan)
        self.state_dir = self.root / "queue"

    def status(self, method, count):
        if method == "lightmem":
            return {"method": method, "selected": count, "completed": count, "population": 500,
                    "failed": [], "generation_complete": count == 500}
        if method == "higmem":
            return {"method": method, "selected": count, "results": count, "failed_ids": [],
                    "run_complete": True, "invalid_native_answers": 0, "benchmark_complete": count == 500}
        return {"method": method, "planned": count, "generated": count,
                "failed": 0, "status": "generation_complete"}

    def write_status(self, method, count):
        name = "completion.json" if method == "higmem" else "status.json"
        queue.save_json(self.root / method / name, self.status(method, count))

    def test_all_seven_share_first_history_then_each_complete_500(self):
        calls = []

        def execute(command, **kwargs):
            method = command[1].removeprefix("native-").removesuffix(".py")
            smoke = "--ids-file" in command
            self.assertEqual(queue.argument(command, "--run-dir"), str(self.root / method))
            if smoke:
                self.assertEqual(queue.read_json(Path(queue.argument(command, "--ids-file"))), ["qid-0"])
            self.assertFalse(kwargs["check"])
            calls.append((method, smoke))
            self.write_status(method, 1 if smoke else 500)
            return SimpleNamespace(returncode=0)

        with patch.object(queue, "DATA_SHA256", self.data_sha), \
             patch.object(queue.subprocess, "run", side_effect=execute):
            self.assertEqual(queue.run(self.plan_path, self.state_dir), 0)
        self.assertEqual(calls, [(method, True) for method in queue.METHODS] +
                         [(method, False) for method in queue.METHODS])
        status = queue.read_json(self.state_dir / "status.json")
        self.assertEqual(status["generated"], 3500)
        self.assertTrue(status["officialjudge_pending"])
        self.assertEqual(status["officially_judged"], 0)

    def test_first_smoke_failure_prevents_all_full_runs(self):
        with patch.object(queue, "DATA_SHA256", self.data_sha), \
             patch.object(queue.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as execute:
            self.assertEqual(queue.run(self.plan_path, self.state_dir), 1)
        self.assertEqual(execute.call_count, 1)
        self.assertIn("--ids-file", execute.call_args.args[0])
        self.assertEqual(queue.read_json(self.state_dir / "status.json")["phase"], "smoke")
        self.assertIn("exited 7", (self.state_dir / "errors.log").read_text())

    def test_stale_incomplete_status_cannot_pass_zero_exit(self):
        self.write_status("e_mem", 0)
        with patch.object(queue, "DATA_SHA256", self.data_sha), \
             patch.object(queue.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as execute:
            self.assertEqual(queue.run(self.plan_path, self.state_dir), 1)
        self.assertEqual(execute.call_count, 1)
        self.assertIn("fresh completion", queue.read_json(self.state_dir / "status.json")["error"])

    def test_fresh_incomplete_status_stops_queue(self):
        def execute(*_args, **_kwargs):
            self.write_status("e_mem", 0)
            return SimpleNamespace(returncode=0)
        with patch.object(queue, "DATA_SHA256", self.data_sha), \
             patch.object(queue.subprocess, "run", side_effect=execute) as execute_mock:
            self.assertEqual(queue.run(self.plan_path, self.state_dir), 1)
        self.assertEqual(execute_mock.call_count, 1)
        self.assertIn("incomplete", queue.read_json(self.state_dir / "status.json")["error"])

    def test_modified_runtime_receipt_refuses_any_launch(self):
        self.receipt.write_text('{"status":"runtime_verified","modified":true}', encoding="utf-8")
        with patch.object(queue, "DATA_SHA256", self.data_sha), patch.object(queue.subprocess, "run") as execute:
            self.assertEqual(queue.run(self.plan_path, self.state_dir), 1)
        execute.assert_not_called()

    def test_duplicate_population_refuses_any_launch(self):
        self.dataset.write_text(json.dumps([{"question_id": "same"}] * 500), encoding="utf-8")
        changed_hash = hashlib.sha256(self.dataset.read_bytes()).hexdigest()
        self.plan["dataset_sha256"] = changed_hash
        queue.save_json(self.plan_path, self.plan)
        with patch.object(queue, "DATA_SHA256", changed_hash), patch.object(queue.subprocess, "run") as execute:
            self.assertEqual(queue.run(self.plan_path, self.state_dir), 1)
        execute.assert_not_called()

    def test_missing_method_refuses_any_launch(self):
        self.plan["runner_commands"].pop("mem0")
        queue.save_json(self.plan_path, self.plan)
        with patch.object(queue, "DATA_SHA256", self.data_sha), patch.object(queue.subprocess, "run") as execute:
            self.assertEqual(queue.run(self.plan_path, self.state_dir), 1)
        execute.assert_not_called()


    def test_unverified_receipt_refuses_launch_even_with_matching_hash(self):
        self.receipt.write_text('{"status":"runtime_pending"}', encoding="utf-8")
        self.plan["runtime_receipt"]["sha256"] = hashlib.sha256(self.receipt.read_bytes()).hexdigest()
        queue.save_json(self.plan_path, self.plan)
        with patch.object(queue, "DATA_SHA256", self.data_sha), patch.object(queue.subprocess, "run") as execute:
            self.assertEqual(queue.run(self.plan_path, self.state_dir), 1)
        execute.assert_not_called()

    def test_stale_success_status_cannot_pass_zero_exit(self):
        self.write_status("e_mem", 1)
        with patch.object(queue, "DATA_SHA256", self.data_sha), \
             patch.object(queue.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as execute:
            self.assertEqual(queue.run(self.plan_path, self.state_dir), 1)
        self.assertEqual(execute.call_count, 1)
        self.assertIn("fresh completion", queue.read_json(self.state_dir / "status.json")["error"])

    def test_full_method_failure_stops_remaining_methods(self):
        calls = []

        def execute(command, **_kwargs):
            method = command[1].removeprefix("native-").removesuffix(".py")
            smoke = "--ids-file" in command
            calls.append((method, smoke))
            if not smoke and method == "simplemem":
                return SimpleNamespace(returncode=9)
            self.write_status(method, 1 if smoke else 500)
            return SimpleNamespace(returncode=0)

        with patch.object(queue, "DATA_SHA256", self.data_sha), \
             patch.object(queue.subprocess, "run", side_effect=execute):
            self.assertEqual(queue.run(self.plan_path, self.state_dir), 1)
        self.assertEqual(calls, [(method, True) for method in queue.METHODS] +
                         [("e_mem", False), ("simplemem", False)])
        status = queue.read_json(self.state_dir / "status.json")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["phase"], "full")
        self.assertEqual(status["method"], "simplemem")
        self.assertNotEqual(status.get("generated"), 3500)
        self.assertEqual(len(status["completed_steps"]), 8)

    def test_higmem_invalid_answer_stops_before_any_full_run(self):
        calls = []

        def execute(command, **_kwargs):
            method = command[1].removeprefix("native-").removesuffix(".py")
            self.assertIn("--ids-file", command)
            calls.append(method)
            self.write_status(method, 1)
            if method == "higmem":
                invalid_status = self.status(method, 1)
                invalid_status["invalid_native_answers"] = 1
                queue.save_json(self.root / method / "completion.json", invalid_status)
            return SimpleNamespace(returncode=0)

        with patch.object(queue, "DATA_SHA256", self.data_sha), \
             patch.object(queue.subprocess, "run", side_effect=execute):
            self.assertEqual(queue.run(self.plan_path, self.state_dir), 1)
        self.assertEqual(calls, list(queue.METHODS))
        self.assertEqual(queue.read_json(self.state_dir / "status.json")["status"], "failed")

    def test_reduced_population_and_shared_output_directories_refuse_launch(self):
        original_plan = json.loads(json.dumps(self.plan))
        for change in ("split_limit", "equals_limit", "shared_directory"):
            with self.subTest(change=change):
                changed = json.loads(json.dumps(original_plan))
                command = changed["runner_commands"]["e_mem"]
                if change == "split_limit":
                    command.extend(["--limit", "1"])
                elif change == "equals_limit":
                    command.append("--limit=1")
                else:
                    command[command.index("--run-dir") + 1] = str(self.root / "simplemem")
                queue.save_json(self.plan_path, changed)
                with patch.object(queue, "DATA_SHA256", self.data_sha), patch.object(queue.subprocess, "run") as execute:
                    self.assertEqual(queue.run(self.plan_path, self.root / change), 1)
                execute.assert_not_called()

if __name__ == "__main__":
    unittest.main()

