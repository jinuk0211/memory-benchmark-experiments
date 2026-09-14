"""CPU-only tests for source isolation and upstream HiGMem QA routing."""
from pathlib import Path
import tempfile
import unittest
import native_higmem as target


class FakeSystem:
    def __init__(self):
        self.calls = []

    def add_turn(self, **kwargs):
        self.calls.append(("add_turn", kwargs))

    def finalize_memory_build(self):
        self.calls.append(("finalize",))

    def build_indices(self):
        self.calls.append(("indices",))

    def retrieve_for_query(self, **kwargs):
        self.calls.append(("retrieve", kwargs))
        return "NATIVE FULL CONTEXT", {"mode": "full"}

    def _get_llm_json_response(self, prompt, schema, **kwargs):
        self.calls.append(("native_answer", prompt, schema, kwargs))
        return {"answer": "Native final answer"}


class NativeHiGMemTests(unittest.TestCase):
    def test_source_preserves_all_turns_and_excludes_annotations(self):
        record = {"haystack_sessions": [[{"role": "user", "content": "original", "has_answer": True},
                                         {"role": "assistant", "content": "reply"}]],
                  "haystack_dates": ["2025/01/01"], "haystack_session_ids": ["session-a"],
                  "question": "DO NOT INGEST", "answer": "SECRET", "question_type": "temporal-reasoning",
                  "answer_session_ids": ["session-a"]}
        source = target.source_only(record)
        self.assertEqual(source, [{"session_id": "session-a", "date": "2025/01/01", "turns": [
            {"role": "user", "content": "original"}, {"role": "assistant", "content": "reply"}]}])
        system = FakeSystem()
        self.assertEqual(target.build_source(system, source), 2)
        self.assertEqual(system.calls[-2:], [("finalize",), ("indices",)])
        self.assertEqual(system.calls[1][1], {"turn_id": "D1:2", "turn_content": "reply",
                                             "speaker": "assistant", "timestamp": "2025/01/01"})

    def test_answer_uses_native_rewrite_retrieve_prompt_and_json_generation(self):
        system = FakeSystem()
        controller = object()
        observed = []

        def rewrite(actual_controller, query):
            self.assertIs(actual_controller, controller)
            observed.append(query)
            return {"keyword_query": "native keywords", "profile_retrieval_keys": ["person"]}

        def prompt_builder(**kwargs):
            self.assertEqual(kwargs["category"], 0)
            self.assertEqual(kwargs["context"], "NATIVE FULL CONTEXT")
            return "UPSTREAM PROMPT"

        answer, trace = target.native_answer(system, controller, "When?", "2026/01/01", rewrite, prompt_builder)
        self.assertEqual(answer, "Native final answer")
        self.assertEqual(observed, ["Question date: 2026/01/01\nQuestion: When?"])
        self.assertEqual(system.calls[0][1]["keyword_query"], "native keywords")
        self.assertEqual(system.calls[0][1]["k_event"], 10)
        self.assertEqual(system.calls[1][1], "UPSTREAM PROMPT")
        self.assertEqual(system.calls[1][3]["caller"], "final_answer_generation")
        self.assertEqual(trace["context"], "NATIVE FULL CONTEXT")

    def test_full_population_and_explicit_subset_selection(self):
        records = [{"question_id": f"q{i}"} for i in range(500)]
        self.assertEqual(len(target.select_records(records, None, 0)), 500)
        self.assertEqual(target.select_records(records, None, 1), records[:1])
        with self.assertRaises(ValueError):
            target.select_records(records[:-1], None, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.json"
            path.write_text('["q3", "q1"]', encoding="utf-8")
            self.assertEqual(target.select_records(records, path, 0), [records[3], records[1]])
            path.write_text('["q3", "q3"]', encoding="utf-8")
            with self.assertRaises(ValueError):
                target.select_records(records, path, 0)

    def test_native_failures_are_not_successful_cached_predictions(self):
        for prediction in ("", "  ", "Could not generate answer.", " Could not generate answer. "):
            with self.subTest(prediction=prediction):
                self.assertFalse(target.is_native_answer(prediction))
                self.assertFalse(target.valid_generated({"status": "generated", "hypothesis": prediction,
                                                         "native_answer_valid": True}))
        self.assertTrue(target.is_native_answer("unknown"))
        self.assertTrue(target.valid_generated({"status": "generated", "hypothesis": "unknown",
                                                "native_answer_valid": True}))

    def test_failure_preserved_and_retried_before_successful_resume(self):
        import contextlib
        import hashlib
        import io
        import json
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"haystack_sessions": [[{"role": "user", "content": "source", "has_answer": True}]],
                   "haystack_dates": ["2025/01/01"], "haystack_session_ids": ["s1"],
                   "question": "When?", "question_date": "2026/01/01"}
            records = [{**row, "question_id": f"q{i}"} for i in range(500)]
            dataset = root / "dataset.json"
            dataset.write_text(json.dumps(records), encoding="utf-8")
            for name in target.SOURCE_FILES:
                (root / name).write_text("source fixture", encoding="utf-8")
            ids_file = root / "ids.json"
            ids_file.write_text('["q0"]', encoding="utf-8")
            argv = ["native_higmem.py", "--dataset", str(dataset), "--run-dir", str(root / "run"),
                    "--source-root", str(root), "--embedding-model", str(root), "--api-base", "http://fixture",
                    "--ids-file", str(ids_file)]
            controller = SimpleNamespace(llm=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0))
            answer = ["Could not generate answer."]
            created = []

            def make_system(*args):
                system = FakeSystem()
                system.turn_notes, system.events, system.profiles = {}, {}, {}
                system.executor = SimpleNamespace(shutdown=lambda **kwargs: None)
                system._get_llm_json_response = lambda *args, **kwargs: {"answer": answer[0]}
                created.append(system)
                return system

            loaded = (controller, make_system,
                      lambda *args: {"keyword_query": "native query", "profile_retrieval_keys": []},
                      lambda **kwargs: "native prompt")
            with patch.object(sys, "argv", argv), patch.object(target, "DATA_SHA256", hashlib.sha256(dataset.read_bytes()).hexdigest()), patch.object(target, "load_native", return_value=loaded), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(target.main(), 1)
                completion = json.loads((root / "run/completion.json").read_text())
                self.assertFalse(completion["run_complete"])
                self.assertFalse(completion["benchmark_complete"])
                self.assertEqual(completion["failed_ids"], ["q0"])
                self.assertEqual((root / "run/predictions.jsonl").read_text(), "")
                self.assertEqual(completion["valid_predictions"], 0)
                failed_path = next((root / "run/histories").glob("*/attempt_0001/prediction.json"))
                failed = json.loads(failed_path.read_text())
                self.assertEqual(failed["hypothesis"], "Could not generate answer.")
                self.assertEqual(failed["status"], "failed")
                answer[0] = "2025"
                self.assertEqual(target.main(), 0)
                self.assertEqual(json.loads(failed_path.read_text()), failed)
                self.assertTrue(failed_path.parent.parent.joinpath("attempt_0002/prediction.json").exists())
                self.assertEqual(target.main(), 0)
                self.assertEqual(len(created), 2)
                protocol_bytes = (root / "run/protocol.json").read_bytes()
                success_path = failed_path.parent.parent / "attempt_0002/prediction.json"
                success_bytes = success_path.read_bytes()
                with patch.object(sys, "argv", argv[:-2]):
                    self.assertEqual(target.main(), 0)
                self.assertEqual((root / "run/protocol.json").read_bytes(), protocol_bytes)
                self.assertEqual(success_path.read_bytes(), success_bytes)
                self.assertEqual(len(created), 501)  # One failed attempt plus 500 valid histories.
                completion = json.loads((root / "run/completion.json").read_text())
                self.assertTrue(completion["run_complete"])
                self.assertTrue(completion["benchmark_complete"])
                self.assertEqual(completion["selected"], 500)
                self.assertEqual(completion["valid_predictions"], 500)
                predictions = [json.loads(line) for line in (root / "run/predictions.jsonl").read_text().splitlines()]
                self.assertEqual({row["question_id"] for row in predictions}, {f"q{i}" for i in range(500)})

            with patch.object(sys, "argv", argv + ["--model", "different-model"]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    target.main()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

