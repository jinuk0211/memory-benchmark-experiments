"""Behavioral native-path, source-boundary and restart tests; no model calls."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import native_five as runner


def row(qid="q1"):
    return {
        "question_id": qid, "question": "TARGET_QUESTION", "question_date": "2023/05/22 (Mon) 10:00",
        "answer": "SECRET_GOLD", "question_type": "SECRET_TYPE", "answer_session_ids": ["SECRET_EVIDENCE"],
        "haystack_session_ids": ["s1"], "haystack_dates": ["2023/05/20 (Sat) 02:21"],
        "haystack_sessions": [[{"role": "user", "content": "SOURCE_ONLY", "has_answer": True}]],
    }


class SourceBoundaryTests(unittest.TestCase):
    def test_annotations_and_query_never_enter_construction(self):
        record = row()
        source = runner.source_only(record)
        self.assertEqual(source, [{"session_id": "s1", "date": record["haystack_dates"][0],
                                  "turns": [{"role": "user", "content": "SOURCE_ONLY"}]}])
        record.update(answer="different", question="changed", answer_session_ids=[])
        self.assertEqual(source, runner.source_only(record))
        text = json.dumps(source)
        for secret in ("SECRET", "TARGET", "has_answer"):
            self.assertNotIn(secret, text)

    def test_large_turns_and_all_sessions_are_retained(self):
        source = runner.source_only(row())
        source[0]["turns"] = [{"role": "user", "content": "A" * 9000},
                              {"role": "assistant", "content": "B" * 8000}]
        source.append({"session_id": "s2", "date": "original", "turns": [
            {"role": "user", "content": "LAST_SESSION"}]})
        chunks = list(runner.source_chunks(source))
        self.assertEqual(len(chunks), 3)
        self.assertIn("A" * 9000, chunks[0][1])
        self.assertIn("B" * 8000, chunks[1][1])
        self.assertIn("LAST_SESSION", chunks[2][1])
        self.assertEqual(chunks[2][0], "original")

    def test_misaligned_history_is_rejected(self):
        record = row()
        record["haystack_dates"] = []
        with self.assertRaises(ValueError):
            runner.source_only(record)

    def test_ingest_passes_source_date_and_not_target_metadata(self):
        agent = SimpleNamespace(simplemem=Mock())
        runner.ingest(agent, "simplemem", "date", "SOURCE", "qid")
        agent.simplemem.add_chunk.assert_called_once_with("SOURCE", timestamp="date")


class NativePathTests(unittest.TestCase):
    def setUp(self):
        self.query = {key: row()[key] for key in ("question_id", "question", "question_date")}

    def test_emem_uses_upstream_chat_not_retrieval_plus_reader(self):
        manager = SimpleNamespace(chat=Mock(return_value="native"), last_queried_memory="evidence")
        adapter = SimpleNamespace(manager=manager, _storage_scope=lambda: nullcontext())
        answer, detail = runner.native_answer(SimpleNamespace(benchmark_memory=adapter), "e_mem", self.query)
        self.assertEqual(answer, "native")
        self.assertEqual(detail["last_queried_memory"], "evidence")
        manager.chat.assert_called_once()
        self.assertEqual(manager.chat.call_args.kwargs, {})

    def test_simplemem_and_amem_use_own_ask(self):
        for method in ("simplemem", "a_mem"):
            with self.subTest(method=method):
                adapter = SimpleNamespace(ask=Mock(return_value="native"))
                answer, _ = runner.native_answer(SimpleNamespace(**{method: adapter}), method, self.query)
                self.assertEqual(answer, "native")
                adapter.ask.assert_called_once()

    def test_library_methods_keep_original_integration_without_labels(self):
        for method in ("mem0", "langmem"):
            agent = SimpleNamespace(send_message=Mock(return_value={"output": "integrated"}))
            answer, _ = runner.native_answer(agent, method, self.query)
            self.assertEqual(answer, "integrated")
            kwargs = agent.send_message.call_args.kwargs
            self.assertFalse(kwargs["memorizing"])
            self.assertEqual(set(kwargs["eval_metadata"]), {"question_id", "question_date"})
            self.assertNotIn("SECRET", str(agent.send_message.call_args))



class NativePolicyTests(unittest.TestCase):
    def test_config_restores_upstream_simplemem_policy_without_source_cap(self):
        root = Path(__file__).resolve().parents[2] / "MemoryData"
        args = argparse.Namespace(
            method="simplemem", source_root=root, model="Qwen3.5-9B",
            api_base="http://llm/v1", embedding_api_base="http://embed/v1",
            embedding_model="all-MiniLM-L6-v2", embedding_dims=384, tokenizer="pinned-tokenizer")
        with patch.object(sys, "path", [str(root), *sys.path]):
            config, _ = runner.build_config(args, Path("attempt"))
        self.assertEqual(config["simplemem_window_size"], 40)
        self.assertTrue(config["simplemem_use_streaming"])
        self.assertEqual(config["simplemem_max_parallel_workers"], 16)
        self.assertEqual(config["simplemem_max_retrieval_workers"], 8)
        self.assertEqual(config["input_length_limit"], 10000000)
        self.assertEqual(config["temperature"], 0.7)
        self.assertEqual(config["embedding_model"], "all-MiniLM-L6-v2")

    def test_mem0_uses_documented_default_temperature(self):
        self.assertEqual(runner.POLICY_OVERRIDES["mem0"], {"temperature": 0.1})

    def test_langmem_override_is_scoped_and_preserves_other_keywords(self):
        module = types.ModuleType("langmem")
        original = Mock(return_value="manager")
        module.create_memory_store_manager = original
        adapter_module = types.ModuleType("methods.langmem.langmem_adapter")

        def factory(*_args, **_kwargs):
            return module.create_memory_store_manager(
                "model", enable_deletes=True, enable_inserts=True, query_limit=5)

        with patch.dict(sys.modules, {"langmem": module,
                                     "methods.langmem.langmem_adapter": adapter_module}):
            self.assertEqual(runner.create_agent(factory, {}, {}, Path("state"), "langmem"), "manager")
        original.assert_called_once_with("model", enable_deletes=False, enable_inserts=True, query_limit=5)
        self.assertIs(module.create_memory_store_manager, original)


class ResumeTests(unittest.TestCase):
    def test_failed_attempt_is_preserved_and_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            history = Path(folder)
            first = runner.next_attempt(history)
            runner.save_json(first / "failure.json", {"reason": "interrupted"})
            self.assertIsNone(runner.verified_prediction(history, {"source": "one"}))
            second = runner.next_attempt(history)
            self.assertNotEqual(first, second)
            self.assertTrue((first / "failure.json").exists())

    def test_completed_prediction_reused_only_for_same_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            history = Path(folder)
            attempt = runner.next_attempt(history)
            identity = {"source": "one", "query": "two"}
            prediction = {"identity": identity, "status": "generated", "hypothesis": "done"}
            runner.save_json(attempt / "prediction.json", prediction)
            self.assertEqual(runner.verified_prediction(history, identity), prediction)
            with self.assertRaises(ValueError):
                runner.verified_prediction(history, {"source": "changed"})

    def test_run_retries_failure_then_skips_completed_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / "dataset.json"
            dataset.write_text(json.dumps([row(f"q{i}") for i in range(500)]), encoding="utf-8")
            ids = root / "ids.json"
            ids.write_text('["q1"]', encoding="utf-8")
            args = argparse.Namespace(
                method="simplemem", dataset=dataset, run_dir=root / "run", ids_file=ids,
                source_root=root, api_base="http://llm/v1", embedding_api_base="http://embed/v1",
                model="Qwen3.5-9B", embedding_model="all-MiniLM-L6-v2", embedding_dims=384, tokenizer=None)
            attempts = []

            def execute(command, **_kwargs):
                attempt = Path(command[-1])
                attempts.append(attempt)
                worker = runner.read_json(attempt / "worker.json")
                self.assertNotIn("SECRET", json.dumps(runner.read_json(attempt / "source.json")))
                self.assertNotIn("dataset", worker["runtime"])
                if len(attempts) == 2:
                    runner.save_json(attempt / "prediction.json", {
                        "identity": worker["identity"], "question_id": "q1",
                        "status": "generated", "hypothesis": "ok"})
                return SimpleNamespace(returncode=1 if len(attempts) == 1 else 0)

            with patch.object(runner, "DATA_SHA256", hashlib.sha256(dataset.read_bytes()).hexdigest()), \
                 patch.object(runner, "source_hashes", return_value={"runner": "frozen"}), \
                 patch.object(runner.subprocess, "run", side_effect=execute):
                self.assertEqual(runner.run(args), 1)
                self.assertEqual(runner.run(args), 0)
                self.assertEqual(runner.run(args), 0)
            self.assertEqual(len(attempts), 2)
            self.assertTrue((attempts[0] / "failure.json").is_file())
            self.assertEqual(runner.read_json(root / "run/status.json")["generated"], 1)

    def test_protocol_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.json"
            runner.write_once(path, {"model": "original"})
            with self.assertRaises(ValueError):
                runner.write_once(path, {"model": "changed"})



class TokenizerRuntimeTests(unittest.TestCase):
    def setUp(self):
        # Import the real lightweight wrapper, without transformers or a model.
        path = Path(__file__).resolve().parents[2] / "MemoryData/utils/provider_utils.py"
        spec = importlib.util.spec_from_file_location("utils.provider_utils", path)
        self.provider = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.provider)
        active = patch.dict(sys.modules, {"utils.provider_utils": self.provider})
        active.start()
        self.addCleanup(active.stop)

    def test_hf_adapter_records_backend_and_exact_sample_ids(self):
        backend = SimpleNamespace(name_or_path="/pinned/qwen", encode=Mock(return_value=[17, 42, 300]))
        wrapped = self.provider.HuggingFaceTokenizerAdapter(backend)
        runtime = runner.tokenizer_runtime(wrapped, "/pinned/qwen")
        self.assertEqual(runtime["pinned_tokenizer_model"], "/pinned/qwen")
        self.assertEqual(runtime["tokenizer_adapter"]["class"], "HuggingFaceTokenizerAdapter")
        self.assertEqual(runtime["tokenizer_backend"]["name_or_path"], "/pinned/qwen")
        self.assertEqual(runtime["sample_encode_ids"], [17, 42, 300])
        backend.encode.assert_called_once_with(runtime["sample_text"], add_special_tokens=False)

    def test_generic_or_approximate_tokenizers_are_rejected(self):
        for fallback in (SimpleNamespace(encode=Mock(return_value=[1])), self.provider.ApproximateTokenizer()):
            with self.subTest(fallback=type(fallback).__name__):
                with self.assertRaisesRegex(RuntimeError, "refusing tokenizer fallback"):
                    runner.tokenizer_runtime(fallback, "/pinned/qwen")

    def test_invalid_hf_encode_result_is_rejected(self):
        for ids in ([], ["text"], [True]):
            with self.subTest(ids=ids):
                wrapped = self.provider.HuggingFaceTokenizerAdapter(SimpleNamespace(encode=Mock(return_value=ids)))
                with self.assertRaisesRegex(RuntimeError, "integer token sequence"):
                    runner.tokenizer_runtime(wrapped, "/pinned/qwen")


class CompletionGuardTests(unittest.TestCase):
    def test_blank_saved_answers_are_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            history = Path(folder)
            identity = {"source": "same", "query": "same"}
            for answer in (None, "", "  \n "):
                with self.subTest(answer=answer):
                    attempt = runner.next_attempt(history)
                    runner.save_json(attempt / "prediction.json", {
                        "identity": identity, "status": "generated", "hypothesis": answer})
                    self.assertIsNone(runner.verified_prediction(history, identity))

    def test_worker_records_blank_answers_as_failures(self):
        fake_metering = types.ModuleType("utils.request_metering")
        fake_metering.install_request_metering = lambda: None
        fake_metering.meter_operation = lambda *_args, **_kwargs: nullcontext()
        fake_agent_module = types.ModuleType("utils.agent")
        fake_agent_module.AgentWrapper = object
        for answer in (None, "", "  \n "):
            with self.subTest(answer=answer), tempfile.TemporaryDirectory() as folder:
                attempt = Path(folder) / "attempt"
                source = runner.source_only(row())
                query = {key: row()[key] for key in ("question_id", "question", "question_date")}
                identity = {"source_sha256": runner.digest(source), "query_sha256": runner.digest(query),
                            "protocol_sha256": "frozen-protocol"}
                hashes = {"native_five.py": "frozen"}
                runtime = {"source_root": folder, "method": "simplemem", "api_base": "http://unused/v1",
                           "embedding_api_base": "http://unused/v1", "model": "Qwen/Qwen3.5-9B"}
                runner.save_json(attempt / "worker.json", {
                    "runtime": runtime, "identity": identity, "source_files_sha256": hashes})
                runner.save_json(attempt / "source.json", source)
                runner.save_json(attempt / "query.json", query)
                agent = Mock()
                with patch.dict(sys.modules, {"utils.request_metering": fake_metering,
                                              "utils.agent": fake_agent_module}), \
                     patch.object(runner, "source_hashes", return_value=hashes), \
                     patch.object(runner, "build_config", return_value=({"tokenizer_model": "pinned-tokenizer"}, {})), \
                     patch.object(runner, "create_agent", return_value=agent), \
                     patch.object(runner, "tokenizer_runtime", return_value={"tokenizer": "fixture"}), \
                     patch.object(runner, "ingest"), \
                     patch.object(runner, "native_answer", return_value=(answer, None)), \
                     patch.object(runner.os, "chdir"), \
                     patch.dict(runner.os.environ), \
                     patch.object(sys, "path", list(sys.path)):
                    self.assertEqual(runner.worker(attempt), 1)
                self.assertFalse((attempt / "prediction.json").exists())
                failure = runner.read_json(attempt / "failure.json")
                self.assertIn("Native answer must be nonempty text", failure["traceback"])
                self.assertEqual(failure["identity"], identity)
                agent.close.assert_called_once()

    def test_one_question_smoke_extends_to_full500_without_protocol_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / "dataset.json"
            dataset.write_text(json.dumps([row(f"q{i}") for i in range(500)]), encoding="utf-8")
            ids = root / "ids.json"
            ids.write_text('["q0"]', encoding="utf-8")
            args = argparse.Namespace(
                method="simplemem", dataset=dataset, run_dir=root / "run", ids_file=ids,
                source_root=root, api_base="http://llm/v1", embedding_api_base="http://embed/v1",
                model="Qwen/Qwen3.5-9B", embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dims=384, tokenizer="pinned-tokenizer")
            native_calls = []

            def fake_worker(command, **_kwargs):
                attempt = Path(command[-1])
                worker = runner.read_json(attempt / "worker.json")
                query = runner.read_json(attempt / "query.json")
                native_calls.append((query["question_id"], worker["identity"]["protocol_sha256"]))
                runner.save_json(attempt / "prediction.json", {
                    "identity": worker["identity"], "question_id": query["question_id"],
                    "status": "generated", "hypothesis": "synthetic valid answer"})
                return SimpleNamespace(returncode=0)

            with patch.object(runner, "DATA_SHA256", hashlib.sha256(dataset.read_bytes()).hexdigest()), \
                 patch.object(runner, "source_hashes", return_value={"runner": "frozen"}), \
                 patch.object(runner.subprocess, "run", side_effect=fake_worker):
                self.assertEqual(runner.run(args), 0)
                protocol_path = args.run_dir / "protocol.json"
                frozen_protocol = protocol_path.read_bytes()
                q0_path = next((args.run_dir / "histories").glob("*/attempt_*/prediction.json"))
                frozen_q0 = q0_path.read_bytes()
                self.assertEqual(len(native_calls), 1)
                args.ids_file = None
                self.assertEqual(runner.run(args), 0)
            self.assertEqual(protocol_path.read_bytes(), frozen_protocol)
            self.assertEqual(q0_path.read_bytes(), frozen_q0)
            self.assertEqual(len(native_calls), 500)
            self.assertEqual(len({qid for qid, _ in native_calls}), 500)
            self.assertEqual(len({identity for _, identity in native_calls}), 1)
            self.assertEqual(len(runner.read_json(protocol_path)["population_ids"]), 500)
            status = runner.read_json(args.run_dir / "status.json")
            self.assertEqual((status["planned"], status["generated"], status["failed"]), (500, 500, 0))
            self.assertEqual(status["status"], "generation_complete")
if __name__ == "__main__":
    unittest.main()

