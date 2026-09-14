"""CPU native-code tests; fake transports/storage only, no model/API calls."""
import argparse
import ast
import concurrent.futures
from contextlib import nullcontext, redirect_stdout
from functools import partial
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import patch

import runner as r


def original_class(path, name, globals_):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), globals_)
    return globals_[name]


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.saved_modules = {key: value for key, value in sys.modules.items() if key == "config" or key.startswith("simplemem")}
        def restore():
            for key in list(sys.modules):
                if key == "config" or key.startswith("simplemem"):
                    del sys.modules[key]
            sys.modules.update(self.saved_modules)
        self.addCleanup(restore)
        self.cfg = r.configure({"api_base": "http://127.0.0.1:18083/v1", "embedding_model": str(self.root / r.MINILM_REVISION)}, self.root)
        models = r.load_file("simplemem_test_models", r.HERE / "upstream/simplemem/core/models/memory_entry.py")
        self.Dialogue, self.MemoryEntry = models.Dialogue, models.MemoryEntry
        self.requests, self.entries, self.events = [], [], []
        self.invalid_location = False
        self.plain_answer = None
        outer = self

        class Client:
            def __init__(self, api_key, base_url):
                self.base_url = base_url
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                if "Q&A assistant" in kwargs["messages"][0]["content"]:
                    value = {"answer": "native answer"}
                else:
                    value = [{"lossless_restatement": "native fact", "keywords": ["fact"],
                              "location": ["bad", "shape"] if outer.invalid_location else "room"}]
                text = outer.plain_answer if outer.plain_answer is not None and "Q&A assistant" in kwargs["messages"][0]["content"] else json.dumps(value)
                return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])])

        llm_globals = {"json": json, "List": List, "Dict": Dict, "Any": Any, "Optional": Optional,
                       "OpenAI": Client, "config": self.cfg}
        self.LLMClient = original_class(r.HERE / "upstream/simplemem/core/utils/llm_client.py", "LLMClient", llm_globals)
        self.client = self.LLMClient(api_key="EMPTY", model=r.MODEL, base_url=self.cfg.OPENAI_BASE_URL)
        self.store = SimpleNamespace(add_entries=lambda entries: self.entries.extend(entries))
        builder_globals = {"List": List, "Optional": Optional, "MemoryEntry": self.MemoryEntry,
                           "Dialogue": self.Dialogue, "LLMClient": self.LLMClient, "VectorStore": object,
                           "config": self.cfg, "json": json, "concurrent": concurrent, "partial": partial}
        self.MemoryBuilder = original_class(r.HERE / "upstream/simplemem/core/memory_builder.py", "MemoryBuilder", builder_globals)
        self.builder = self.MemoryBuilder(self.client, self.store, enable_parallel_processing=None, max_parallel_workers=None)
        self.calls = r.Calls(self.root / "calls.jsonl", "q", lambda *args, **kwargs: nullcontext())
        self.calls.bind(self.client, self.cfg.OPENAI_BASE_URL)

    def source(self, count):
        return [{"session_id": "s", "date": "2023/05/20 (Sat) 13:00", "turns": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(count)]}]

    def test_original_bulk_550_turns_has_15_native_windows_and_no_budget_override(self):
        source = self.source(550)
        source[0]["turns"][1]["content"] = "x" * 5001
        dialogues = r.convert_dialogues(source, self.Dialogue)
        self.assertEqual(len(dialogues), 550)
        self.assertEqual([x.dialogue_id for x in dialogues], list(range(1, 551)))
        self.assertEqual(dialogues[1].content, "x" * 5001)
        self.assertEqual(dialogues[1].speaker, "assistant")
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            self.builder.add_dialogues(dialogues)
            self.builder.process_remaining()
        self.calls.check()
        self.assertEqual(len(self.requests), 15)
        self.assertEqual(len(self.entries), 15)
        self.assertEqual(self.builder.processed_count, 578)  # 550 unique + native overlap.
        self.assertEqual((self.builder.window_size, self.builder.step_size, self.builder.max_parallel_workers), (40, 38, 16))
        self.assertTrue(all(x["model"] == r.MODEL and x["temperature"] == 0.1 and x["stream"] is True for x in self.requests))
        self.assertTrue(all("max_tokens" not in x and "response_format" not in x for x in self.requests))
        self.assertEqual(self.calls.errors, [])

    def test_original_location_schema_and_three_native_retries_preserved(self):
        self.invalid_location = True
        dialogues = r.convert_dialogues(self.source(3), self.Dialogue)
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            self.builder.add_dialogues(dialogues)
            self.builder.process_remaining()
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.entries, [])
        with self.assertRaises(RuntimeError):
            self.calls.check()
        self.assertTrue(any("memory_builder.py:227" in error for error in self.calls.errors))

    def test_parallel_thread_native_fallback_detected(self):
        self.invalid_location = True
        dialogues = r.convert_dialogues(self.source(81), self.Dialogue)
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            self.builder.add_dialogues(dialogues)
        self.assertEqual(len(self.requests), 9)  # 40,40,5 windows; each original three retries.
        self.assertTrue(any("memory_builder.py:433" in error for error in self.calls.errors))

    def test_native_system_ask_calls_original_answer_generator(self):
        answer_globals = {"List": List, "MemoryEntry": self.MemoryEntry, "LLMClient": self.LLMClient, "config": self.cfg}
        AnswerGenerator = original_class(r.HERE / "upstream/simplemem/core/answer_generator.py", "AnswerGenerator", answer_globals)
        system_globals = {"List": List, "Optional": Optional, "Dialogue": self.Dialogue, "MemoryEntry": self.MemoryEntry,
                          "config": self.cfg, "LLMClient": self.LLMClient, "EmbeddingModel": object,
                          "VectorStore": object, "MemoryBuilder": self.MemoryBuilder,
                          "HybridRetriever": object, "AnswerGenerator": AnswerGenerator}
        System = original_class(r.HERE / "upstream/main.py", "SimpleMemSystem", system_globals)
        system = System.__new__(System)
        system.hybrid_retriever = SimpleNamespace(retrieve=lambda question: self.events.append(question) or [self.MemoryEntry(lossless_restatement="evidence")])
        system.answer_generator = AnswerGenerator(self.client)
        self.calls.phase = "qa"
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            result = system.ask("Question date: D\nQuestion: Q")
        self.assertEqual(result, "native answer")
        self.assertEqual(self.events, ["Question date: D\nQuestion: Q"])
        self.assertIn("professional Q&A assistant", self.requests[-1]["messages"][0]["content"])

    def test_native_optional_retrieval_recovery_is_warning_not_fatal(self):
        scope = {"List": List, "Optional": Optional, "Dict": Dict, "Any": Any,
                 "MemoryEntry": self.MemoryEntry, "LLMClient": self.LLMClient,
                 "VectorStore": object, "config": self.cfg}
        Retriever = original_class(r.HERE / "upstream/simplemem/core/hybrid_retriever.py", "HybridRetriever", scope)
        retriever = Retriever.__new__(Retriever)
        def fail(*args, **kwargs):
            raise ValueError("native planning failed")
        retriever.llm_client = SimpleNamespace(chat_completion=fail)
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            queries = retriever._generate_targeted_queries("original question", {})
        self.assertEqual(queries, ["original question"])
        self.calls.check()
        self.assertTrue(any("hybrid_retriever.py:792" in event for event in self.calls.warnings))

    def test_native_raw_qa_recovery_is_retained(self):
        self.plain_answer = "plain native answer"
        scope = {"List": List, "MemoryEntry": self.MemoryEntry, "LLMClient": self.LLMClient, "config": self.cfg}
        Generator = original_class(r.HERE / "upstream/simplemem/core/answer_generator.py", "AnswerGenerator", scope)
        generator = Generator(self.client)
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            answer = generator.generate_answer("Q", [self.MemoryEntry(lossless_restatement="fact")])
        self.assertEqual(answer, "plain native answer")
        self.assertEqual(len(self.requests), 3)
        self.calls.check()
        self.assertTrue(any("answer_generator.py:81" in event for event in self.calls.warnings))

    def test_source_whitelist_and_config_closed_to_inherited_defaults(self):
        row = {"answer": "GOLD", "question_type": "LABEL", "haystack_sessions": [[
            {"role": "user", "content": "only content", "has_answer": True}]],
            "haystack_dates": ["D"], "haystack_session_ids": ["s"]}
        source = r.source_only(row)
        converted = r.convert_dialogues(source, self.Dialogue)
        self.assertEqual(converted[0].model_dump(), {"dialogue_id": 1, "speaker": "user", "content": "only content", "timestamp": "D"})
        self.assertEqual((self.cfg.WINDOW_SIZE, self.cfg.MAX_PARALLEL_WORKERS, self.cfg.MAX_RETRIEVAL_WORKERS), (40, 16, 8))
        self.assertFalse(self.cfg.USE_JSON_FORMAT)
        with self.assertRaises(RuntimeError):
            r.configure({"api_base": "http://127.0.0.1:18083/v1", "embedding_model": "elsewhere"}, self.root)


class ReceiptTests(unittest.TestCase):
    def test_success_artifacts_tamper_and_failed_attempt_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            history = Path(folder) / "history"
            attempt = r.next_attempt(history)
            identity = {"protocol_sha256": "p", "source_sha256": "s", "query_sha256": "q"}
            for name in ("source.json", "query.json", "worker.json", "memory.json", "dialogues.json",
                         "build_complete.json", "usage.json", "native_runtime.json"):
                r.save_json(attempt / name, {})
            (attempt / "llm_calls.jsonl").write_text("{}\n")
            r.save_json(attempt / "prediction.json", {"status": "generated", "identity": identity, "hypothesis": "answer"})
            r.save_json(attempt / "database/table.json", {"native_cache": True})
            r.seal(attempt, identity)
            self.assertEqual(r.verified(history, identity)["hypothesis"], "answer")
            with self.assertRaises(ValueError):
                r.verified(history, dict(identity, source_sha256="different"))
            r.save_json(attempt / "database/table.json", {"modified": True})
            with self.assertRaises(ValueError):
                r.verified(history, identity)
            next_attempt = r.next_attempt(history)
            r.save_json(next_attempt / "failure.json", {"error": "preserved"})
            self.assertTrue((next_attempt / "failure.json").exists())
            self.assertTrue((attempt / "completion.json").exists())

    def test_fts_required_for_nonempty_memory_only(self):
        backend = SimpleNamespace(count=lambda: 0, _fts_initialized=False)
        system = SimpleNamespace(vector_store=SimpleNamespace(backend=backend))
        r.require_fts(system)
        backend.count = lambda: 1
        with self.assertRaises(RuntimeError):
            r.require_fts(system)
        backend._fts_initialized = True
        r.require_fts(system)

    def test_original_sources_and_local_endpoint_guard(self):
        hashes = r.source_hashes()
        self.assertEqual(hashes["upstream/simplemem/core/memory_builder.py"], "ade5d9c252158ede2d0bd4ca3da027e913d6e88dbe616e00f5962178053c9dc7")
        self.assertEqual(hashes["upstream/simplemem/core/models/memory_entry.py"], "487209f5d6981a12568ec3f6f4597d2bd1437a82a0ef9af85c5dc755bc2947c8")
        with self.assertRaises(ValueError):
            r.local_endpoint("https://api.openai.com/v1")


if __name__ == "__main__":
    unittest.main()