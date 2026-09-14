"""CPU-only contracts plus execution of the unmodified official memory function."""
import argparse
import ast
import copy
import hashlib
import json
import logging
from pathlib import Path
import re
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import runner


def source_record(qid="q0"):
    return {"question_id": qid, "question": "Where?", "question_date": "2023/05/20 (Sat) 12:00",
            "answer": "DO_NOT_INGEST_GOLD", "question_type": "DO_NOT_INGEST_TYPE",
            "haystack_session_ids": ["s1", "s2"], "haystack_dates": ["2023/05/18 (Thu) 09:00", "2023/05/19 (Fri) 09:00"],
            "haystack_sessions": [[{"role": "user", "content": "A" * 5000, "has_answer": True},
                                    {"role": "assistant", "content": "A response", "has_answer": False},
                                    {"role": "user", "content": "Tail", "evidence": "secret"}],
                                   [{"role": "user", "content": "Next session"},
                                    {"role": "assistant", "content": "Another response"}]]}


def runtime(root):
    return argparse.Namespace(method="mem0", model="Qwen/Qwen3.5-9B",
                              embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dims=384,
                              api_base="http://127.0.0.1:18083/v1", embedding_api_base="http://127.0.0.1:18084/v1",
                              source_root=runner.ROOT / "source/MemoryData", tokenizer="/pinned/qwen",
                              dataset=root / "data.json", run_dir=root / "run", ids_file=root / "ids.json")


class CandidateTests(unittest.TestCase):
    def test_official_wheel_members_are_exact_and_tampering_is_rejected(self):
        hashes = runner.verify_official_source()
        self.assertEqual(hashes["vendor/mem0/memory/main.py"], "7c43b5defd6767f3a101a9bd5d605662e11f1d637462846b82c2ec5658b63371")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"wheel_file": "wheel", "wheel_sha256": runner.WHEEL_SHA256,
                        "files_sha256": {"vendor/mem0/fake.py": "0" * 64}}
            (root / "vendor/mem0").mkdir(parents=True)
            (root / "vendor/mem0/fake.py").write_text("altered")
            (root / "wheel").write_bytes(b"altered wheel")
            runner.harness.save_json(root / "source_manifest.json", manifest)
            with patch.object(runner, "HERE", root), self.assertRaisesRegex(ValueError, "bytes changed"):
                runner.verify_official_source()

    def test_pairs_preserve_full_roles_text_tail_and_session_boundaries_without_labels(self):
        original = source_record()
        before = copy.deepcopy(original)
        source = runner.harness.source_only(original)
        pairs = list(runner.session_pairs(source))
        self.assertEqual([len(messages) for _, messages in pairs], [2, 1, 2])
        self.assertEqual([meta["session_id"] for meta, _ in pairs], ["s1", "s1", "s2"])
        self.assertEqual(pairs[0][1][0], {"role": "user", "content": "A" * 5000})
        self.assertEqual(pairs[0][1][1]["role"], "assistant")
        self.assertEqual(pairs[0][0]["timestamp"], 1684400400)
        for _, messages in pairs:
            self.assertTrue(all(set(message) == {"role", "content"} for message in messages))
        self.assertNotIn("DO_NOT_INGEST", json.dumps(source))
        self.assertEqual(original, before)

    def test_invalid_turn_fails_instead_of_silently_dropping_text(self):
        for value in ["", "  ", None, 42, ["text"]]:
            record = source_record()
            record["haystack_sessions"][0][0]["content"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                list(runner.session_pairs(runner.harness.source_only(record)))

    def test_native_add_receives_original_pair_and_metadata_without_wrapper_prompt(self):
        memory = Mock()
        memory.add.return_value = {"results": []}
        with tempfile.TemporaryDirectory() as temporary:
            count = runner.ingest_pairs(memory, runner.harness.source_only(source_record()), "qid-store",
                                        Path(temporary) / "progress.json", runner.harness.NativeErrors())
        self.assertEqual(count, 3)
        self.assertEqual(memory.add.call_args_list[0].args[0][1], {"role": "assistant", "content": "A response"})
        for call in memory.add.call_args_list:
            self.assertEqual(call.kwargs["user_id"], "qid-store")
            self.assertIs(call.kwargs["infer"], True)
            self.assertIn("session_date", call.kwargs["metadata"])

    def test_swallowed_native_error_stops_construction_without_retrying_or_completing(self):
        errors = runner.harness.NativeErrors()
        memory = Mock()
        memory.add.side_effect = lambda *a, **kw: errors.records.append("Invalid JSON response")
        with tempfile.TemporaryDirectory() as temporary:
            progress = Path(temporary) / "progress.json"
            with self.assertRaises(RuntimeError):
                runner.ingest_pairs(memory, runner.harness.source_only(source_record()), "qid", progress, errors)
            self.assertEqual(runner.harness.read_json(progress)["completed_pairs"], 0)
        self.assertEqual(memory.add.call_count, 1)

    def test_model_guard_and_official_generation_defaults(self):
        args = runtime(Path("unused"))
        config = runner.memory_config(args, Path("state"))
        self.assertEqual(config["llm"]["config"]["max_tokens"], 2000)
        self.assertEqual(config["llm"]["config"]["temperature"], 0.1)
        self.assertEqual(config["llm"]["config"]["top_p"], 0.1)
        self.assertNotIn("custom_fact_extraction_prompt", config)
        self.assertNotIn("custom_update_memory_prompt", config)
        for key, value in [("model", "other"), ("embedding_dims", 1536),
                           ("api_base", "https://api.openai.com/v1"),
                           ("embedding_api_base", "http://user:secret@localhost/v1")]:
            modified = copy.copy(args)
            setattr(modified, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_runtime(modified)

    def test_failed_prediction_is_not_reused_and_next_attempt_preserves_old_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            history = Path(temporary)
            first = runner.harness.next_attempt(history)
            identity = {"protocol_sha256": "a", "source_sha256": "b", "query_sha256": "c"}
            runner.harness.save_json(first / "prediction.json", {"identity": identity, "status": "generated", "hypothesis": "fallback"})
            runner.harness.save_json(first / "failure.json", {"reason": "failed after writing"})
            old = (first / "failure.json").read_bytes()
            self.assertIsNone(runner.verified_prediction(history, identity))
            self.assertEqual(runner.harness.next_attempt(history).name, "attempt_0002")
            self.assertEqual((first / "failure.json").read_bytes(), old)

    def test_smoke_then_full_protocol_and_actual_exporter_compatible(self):
        import export_official
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = runtime(root)
            runner.harness.save_json(args.dataset, [source_record(f"q{i}") for i in range(500)])
            runner.harness.save_json(args.ids_file, ["q0"])
            sha = hashlib.sha256(args.dataset.read_bytes()).hexdigest()

            def generate(command, **kwargs):
                attempt = Path(command[-1])
                spec = runner.harness.read_json(attempt / "worker.json")
                query = runner.harness.read_json(attempt / "query.json")
                runner.harness.save_json(attempt / "prediction.json", {"identity": spec["identity"],
                    "question_id": query["question_id"], "status": "generated", "hypothesis": "Answer", "officially_judged": False})
                for name in ("native_config", "native_runtime", "native_answer_detail"):
                    runner.harness.save_json(attempt / f"{name}.json", {})
                runner.harness.save_json(attempt / "build_complete.json", {"identity": spec["identity"]})
                (attempt / "memory/qdrant").mkdir(parents=True)
                (attempt / "memory/history.db").write_bytes(b"CPU fake SQLite artifact")
                (attempt / "memory/qdrant/storage.sqlite").write_bytes(b"CPU fake Qdrant artifact")
                return SimpleNamespace(returncode=0)

            with patch.object(runner.harness, "DATA_SHA256", sha), patch.object(runner, "source_hashes", return_value={"verified": "hash"}), patch.object(runner.subprocess, "run", side_effect=generate) as process:
                self.assertEqual(runner.run(args), 0)
                protocol = (args.run_dir / "protocol.json").read_bytes()
                self.assertEqual(len(json.loads(protocol)["population_ids"]), 500)
                args.ids_file = None
                self.assertEqual(runner.run(args), 0)
                self.assertEqual((args.run_dir / "protocol.json").read_bytes(), protocol)
                self.assertEqual(process.call_count, 500)
            vendor = root / "evaluator"
            vendor.mkdir()
            (vendor / "mock.py").write_text("# CPU export test\n")
            upstream_hash = hashlib.sha256((vendor / "mock.py").read_bytes()).hexdigest()
            with patch.object(export_official, "DATA_SHA256", sha), patch.object(export_official, "UPSTREAM_HASHES", {"mock.py": upstream_hash}):
                result = export_official.export(argparse.Namespace(method="mem0", dataset=args.dataset,
                    run_dir=args.run_dir, vendor=vendor, output=root / "hypotheses.jsonl"))
            self.assertEqual(result["expected_count"], 500)
            self.assertEqual(result["official_judge_calls_by_exporter"], 0)
            self.assertEqual(set(json.loads((root / "hypotheses.jsonl").read_text().splitlines()[0])), {"question_id", "hypothesis"})

    def test_worker_rejects_spec_that_does_not_match_saved_run_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            attempt = runner.harness.next_attempt(run_dir / "histories/qid")
            protocol = {"runtime": {"model": "frozen"}, "source_files_sha256": {"a": "b"}}
            identity = {"protocol_sha256": runner.harness.digest(protocol)}
            runner.harness.save_json(run_dir / "protocol.json", protocol)
            runner.harness.save_json(attempt / "worker.json", {"identity": identity,
                "runtime": {"model": "changed"}, "source_files_sha256": protocol["source_files_sha256"]})
            with patch.object(runner, "dependency_versions") as dependencies:
                self.assertEqual(runner.worker(attempt), 1)
                dependencies.assert_not_called()
            self.assertIn("immutable run protocol", runner.harness.read_json(attempt / "failure.json")["traceback"])
            self.assertFalse((attempt / "prediction.json").exists())

    def test_completion_cache_detects_memory_build_and_source_mutation_or_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            history = Path(temporary)
            attempt = runner.harness.next_attempt(history)
            source, query = [], {"question_id": "qid", "question": "Where?", "question_date": "date"}
            identity = {"protocol_sha256": "a" * 64, "source_sha256": runner.harness.digest(source),
                        "query_sha256": runner.harness.digest(query)}
            for name, value in (("source", source), ("query", query), ("worker", {}),
                                ("native_config", {}), ("native_runtime", {}), ("native_answer_detail", {}),
                                ("build_complete", {"identity": identity}),
                                ("prediction", {"identity": identity, "question_id": "qid", "status": "generated", "hypothesis": "Answer"})):
                runner.harness.save_json(attempt / f"{name}.json", value)
            (attempt / "memory/qdrant").mkdir(parents=True)
            (attempt / "memory/history.db").write_bytes(b"history")
            (attempt / "memory/qdrant/storage.sqlite").write_bytes(b"native memory")
            runner.finalize_attempt(attempt, identity)
            self.assertEqual(runner.verified_prediction(history, identity)["hypothesis"], "Answer")
            for name in ("source.json", "build_complete.json", "memory/qdrant/storage.sqlite"):
                path = attempt / name
                original = path.read_bytes()
                with self.subTest(name=name, mutation="modified"):
                    path.write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "artifacts changed"):
                        runner.verified_prediction(history, identity)
                    path.write_bytes(original)
                with self.subTest(name=name, mutation="missing"):
                    path.unlink()
                    with self.assertRaisesRegex(ValueError, "artifacts changed"):
                        runner.verified_prediction(history, identity)
                    path.write_bytes(original)

    def test_invalid_dataset_and_duplicate_unknown_selected_ids_fail_before_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = runtime(root)
            runner.harness.save_json(args.dataset, [source_record(f"q{i}") for i in range(500)])
            with patch.object(runner.subprocess, "run") as process, self.assertRaises(ValueError):
                runner.run(args)
            process.assert_not_called()
            sha = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
            for selected in [["q0", "q0"], ["unknown"], [], "q0"]:
                runner.harness.save_json(args.ids_file, selected)
                with patch.object(runner.harness, "DATA_SHA256", sha), self.subTest(selected=selected), self.assertRaises(ValueError):
                    runner.run(args)


class OfficialCoreTests(unittest.TestCase):
    """Execute the actual frozen _add_to_vector_store with fake I/O only."""
    @classmethod
    def setUpClass(cls):
        vendor = runner.HERE / "vendor/mem0"
        namespace = runpy.run_path(str(vendor / "configs/prompts.py"))
        namespace.update(json=json, logging=logging, re=re, capture_event=lambda *a, **kw: None)
        utils = ast.parse((vendor / "memory/utils.py").read_text(encoding="utf-8"))
        selected = [n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name in ("parse_messages", "get_fact_retrieval_messages", "remove_code_blocks")]
        tree = ast.parse((vendor / "memory/main.py").read_text(encoding="utf-8"))
        memory = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Memory")
        selected.append(next(n for n in memory.body if isinstance(n, ast.FunctionDef) and n.name == "_add_to_vector_store"))
        exec(compile(ast.Module(body=selected, type_ignores=[]), "unaltered_official_functions", "exec"), namespace)
        cls.operation = staticmethod(namespace["_add_to_vector_store"])

    def memory(self, action_response):
        memory = SimpleNamespace(config=SimpleNamespace(custom_fact_extraction_prompt=None, custom_update_memory_prompt=None),
                                 llm=Mock(), embedding_model=Mock(), vector_store=Mock(), api_version="v1.1",
                                 _create_memory=Mock(return_value="new-id"), _update_memory=Mock(), _delete_memory=Mock())
        memory.llm.generate_response.side_effect = [json.dumps({"facts": ["fact"]}), action_response]
        memory.embedding_model.embed.return_value = [0.1] * 384
        memory.vector_store.search.return_value = [SimpleNamespace(id="old-id", payload={"data": "old fact"})]
        return memory

    def test_native_add_update_delete_none_actions_are_not_rewritten(self):
        actions = [{"event": "ADD", "text": "new fact"}, {"event": "UPDATE", "id": "0", "text": "updated"},
                   {"event": "DELETE", "id": "0", "text": "delete"}, {"event": "NONE", "id": "0", "text": "unchanged"}]
        memory = self.memory(json.dumps({"memory": actions}))
        result = self.operation(memory, [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "reply"}], {"session_id": "s"}, {"user_id": "qid"}, True)
        self.assertEqual([r["event"] for r in result], ["ADD", "UPDATE", "DELETE"])
        memory._update_memory.assert_called_once()
        memory._delete_memory.assert_called_once_with(memory_id="old-id")
        self.assertEqual(memory.llm.generate_response.call_count, 2)
        self.assertIn("assistant: reply", memory.llm.generate_response.call_args_list[0].kwargs["messages"][1]["content"])

    def test_native_malformed_json_and_bad_action_ids_log_failure_without_repair_calls(self):
        for response in ["{truncated", json.dumps({"memory": [{"event": "UPDATE", "id": "invalid", "text": "x"}]})]:
            memory = self.memory(response)
            errors = runner.harness.NativeErrors()
            logging.getLogger().addHandler(errors)
            try:
                result = self.operation(memory, [{"role": "user", "content": "hello"}], {}, {"user_id": "qid"}, True)
            finally:
                logging.getLogger().removeHandler(errors)
            self.assertEqual(result, [])
            self.assertTrue(errors.records)
            self.assertEqual(memory.llm.generate_response.call_count, 2)
            memory._update_memory.assert_not_called()


if __name__ == "__main__":
    unittest.main()