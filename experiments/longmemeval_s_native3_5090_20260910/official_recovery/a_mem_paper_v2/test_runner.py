"""CPU-only behavioral checks for the original paper pipeline and durable harness."""
import argparse
from contextlib import nullcontext, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import runner as r


class Response:
    def __init__(self, value, finish="stop"):
        self.text = value if isinstance(value, str) else json.dumps(value)
        self.choices = [SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=self.text))]
        self.usage = {"prompt_tokens": 5, "completion_tokens": 7}

    def model_dump(self, **kwargs):
        return {"choices": [{"finish_reason": self.choices[0].finish_reason,
                             "message": {"content": self.text}}], "usage": self.usage}


class NativePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.requests = []
        self.embedding_calls = []
        self.fenced_metadata = False
        self.invalid_evolution = False
        self.invalid_metadata = False
        self.metadata_transport_error = False
        self.unhandled_evolution = False
        self.blank_qa = False
        self.environment = patch.dict(os.environ, OPENAI_BASE_URL="http://127.0.0.1:18083/v1", OPENAI_API_KEY="EMPTY")
        self.environment.start()
        self.addCleanup(self.environment.stop)
        outer = self

        class FakeOpenAI:
            def __init__(self, api_key):
                if api_key != "EMPTY":
                    raise AssertionError("Non-placeholder credential")
                self.base_url = os.environ["OPENAI_BASE_URL"]
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **kwargs):
                outer.requests.append(kwargs)
                keys = set(kwargs["response_format"]["json_schema"]["schema"]["properties"])
                if keys == {"keywords", "context", "tags"}:
                    if outer.metadata_transport_error:
                        raise ConnectionError("synthetic local transport failure")
                    if outer.invalid_metadata:
                        return Response("{", "length")
                    value = {"keywords": ["native"], "context": "native metadata", "tags": ["tag"]}
                    return Response("```json\n" + json.dumps(value) + "\n```" if outer.fenced_metadata else value)
                if "should_evolve" in keys:
                    if outer.unhandled_evolution:
                        return Response({"wrong_field": False})
                    if outer.invalid_evolution:
                        return Response("{", "length")
                    return Response({"should_evolve": False, "actions": [], "suggested_connections": [],
                                     "tags_to_update": [], "new_context_neighborhood": [], "new_tags_neighborhood": []})
                if keys == {"keywords"}:
                    return Response({"keywords": "rewritten-keywords"})
                return Response({"answer": " " if outer.blank_qa else "native answer"})

        class FakeEmbedding:
            max_seq_length = 256
            def __init__(self, name, device):
                outer.embedding_calls.append((name, device))
            def get_sentence_embedding_dimension(self):
                return 384
            def parameters(self):
                return [SimpleNamespace(dtype="torch.float32")]
            def encode(self, documents):
                return np.ones((len(documents), 384), dtype=np.float32)
            def get_config_dict(self):
                return {"model_name": "all-MiniLM-L6-v2"}

        modules = {}
        for name in ("openai", "sentence_transformers", "sklearn", "sklearn.metrics", "sklearn.metrics.pairwise"):
            modules[name] = ModuleType(name)
        modules["openai"].OpenAI = FakeOpenAI
        modules["sentence_transformers"].SentenceTransformer = FakeEmbedding
        modules["sklearn.metrics.pairwise"].cosine_similarity = lambda q, e: np.ones((len(q), len(e)))
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.calls = r.Calls(self.root / "calls.jsonl", "http://127.0.0.1:18083/v1")
        self.agent = r.make_agent(self.root / r.MINILM_REVISION, self.calls)

    def test_real_native_path_per_turn_rewrite_retrieval_qa_and_all_model_calls(self):
        record = {"haystack_sessions": [[{"role": "user", "content": "x" * 5000, "has_answer": True},
                                         {"role": "assistant", "content": "second", "has_answer": False}]],
                  "haystack_dates": ["2023/05/20 (Sat) 13:00"], "haystack_session_ids": ["s"],
                  "answer": "GOLD_SENTINEL", "question_type": "LABEL_SENTINEL"}
        sessions = r.source_only(record)
        with redirect_stdout(io.StringIO()):
            count = r.ingest_turns(self.agent, sessions, self.calls)
            answer, detail = r.answer(self.agent, {"question": "public question", "question_date": "2023/05/21", "question_id": "q"}, self.calls)
        self.assertEqual(count, 2)
        contents = [note.content for note in self.agent.memory_system.memories.values()]
        self.assertEqual(contents, ["Speaker usersays : " + "x" * 5000, "Speaker assistantsays : second"])
        self.assertEqual(answer, "native answer")
        self.assertEqual(len(self.requests), 6)  # Two metadata + two evolution + rewrite + QA.
        self.assertIn("cosmos", self.requests[-2]["messages"][1]["content"])
        self.assertIn("short phrase", self.requests[-1]["messages"][1]["content"])
        self.assertTrue(all(x["model"] == r.MODEL and x["max_tokens"] == 1000 and x["temperature"] == 0.7 for x in self.requests))
        self.assertEqual(self.agent.retrieve_k, 10)
        for forbidden in ("GOLD_SENTINEL", "LABEL_SENTINEL", "has_answer"):
            self.assertNotIn(forbidden, json.dumps(self.requests))
        self.assertIn("Question date: 2023/05/21", detail["prompt"])
        self.assertEqual(self.calls.errors, [])
        self.assertEqual({x[1] for x in self.embedding_calls}, {"cpu"})
        self.agent.memory_system.consolidate_memories()
        self.assertTrue(all(x[0] == str(self.root / r.MINILM_REVISION) for x in self.embedding_calls))

    def test_keyword_output_is_used_for_native_retrieval_at_k10(self):
        seen = []
        self.agent.retrieve_memory = lambda query, k: seen.append((query, k)) or "evidence"
        with redirect_stdout(io.StringIO()):
            answer, _ = r.answer(self.agent, {"question": "Q", "question_date": "D"}, self.calls)
        self.assertEqual(seen, [("rewritten-keywords", 10)])
        self.assertEqual(answer, "native answer")

    def test_native_fenced_metadata_is_accepted_without_repair(self):
        self.fenced_metadata = True
        sessions = [{"date": "2023/05/20 (Sat) 13:00", "turns": [{"role": "user", "content": "text"}]}]
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            self.assertEqual(r.ingest_turns(self.agent, sessions, self.calls), 1)
        self.assertEqual(next(iter(self.agent.memory_system.memories.values())).context, "native metadata")
        self.assertEqual(self.calls.errors, [])
        self.assertEqual(len(self.requests), 2)

    def test_native_json_fallback_counted_and_original_note_retained(self):
        self.invalid_evolution = True
        sessions = [{"date": "2023/05/20 (Sat) 13:00", "turns": [{"role": "user", "content": "text"}]}]
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            self.assertEqual(r.ingest_turns(self.agent, sessions, self.calls), 1)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.agent.memory_system.memories), 1)
        self.assertEqual(self.calls.errors, [])
        self.assertEqual(self.calls.degradation_counts(), {"native_handled_fallback_count": 1,
                         "native_metadata_fallback_count": 0, "native_evolution_fallback_count": 1})
        self.assertEqual(self.calls.handled_fallbacks, [{"kind": "evolution", "file": "upstream/memory_layer.py",
                         "function": "process_memory", "line": 827, "phase": "memory_write", "native_call_ordinal": 2}])
        self.assertIn("warning", self.calls.records[-1])

    def test_native_metadata_fallback_counted_once_without_fixing_upstream(self):
        self.invalid_metadata = True
        sessions = [{"date": "2023/05/20 (Sat) 13:00", "turns": [{"role": "user", "content": "text"}]}]
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            self.assertEqual(r.ingest_turns(self.agent, sessions, self.calls), 1)
        note = next(iter(self.agent.memory_system.memories.values()))
        self.assertEqual((note.context, note.keywords, note.tags), ("General", [], []))
        self.assertEqual(self.calls.errors, [])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.calls.degradation_counts(), {"native_handled_fallback_count": 1,
                         "native_metadata_fallback_count": 1, "native_evolution_fallback_count": 0})
        # The unchanged native inner exception formatter references unbound e;
        # its existing outer handler, not an added correction, returns defaults.
        self.assertEqual(self.calls.handled_fallbacks[0]["line"], 396)
        self.assertEqual(self.calls.handled_fallbacks[0]["native_call_ordinal"], 1)

    def test_all_550_turns_and_valid_qa_survive_counted_native_fallbacks(self):
        self.invalid_metadata = self.invalid_evolution = True
        turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"source turn {i}"} for i in range(550)]
        sessions = [{"date": "2023/05/20 (Sat) 13:00", "turns": turns}]
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            count = r.ingest_turns(self.agent, sessions, self.calls)
            hypothesis, _ = r.answer(self.agent, {"question": "public query", "question_date": "D"}, self.calls)
        self.assertEqual(count, 550)
        self.assertEqual([note.content for note in self.agent.memory_system.memories.values()],
                         ["Speaker " + turn["role"] + "says : " + turn["content"] for turn in turns])
        self.assertEqual(hypothesis, "native answer")
        self.assertEqual(len(self.requests), 1102)  # Exactly two native writes per turn, then rewrite/QA.
        self.assertEqual(self.calls.degradation_counts(), {"native_handled_fallback_count": 1100,
                         "native_metadata_fallback_count": 550, "native_evolution_fallback_count": 550})
        self.assertEqual(self.calls.errors, [])
        self.assertTrue(all(req["model"] == r.MODEL and req["max_tokens"] == 1000 and req["temperature"] == 0.7 for req in self.requests))

    def test_native_unhandled_error_and_metadata_transport_are_not_successes(self):
        sessions = [{"date": "2023/05/20 (Sat) 13:00", "turns": [{"role": "user", "content": "text"}]}]
        self.unhandled_evolution = True
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            with self.assertRaises(KeyError):
                r.ingest_turns(self.agent, sessions, self.calls)
        self.assertEqual(self.calls.handled_fallbacks, [])
        self.unhandled_evolution = False
        self.metadata_transport_error = True
        with redirect_stdout(io.StringIO()), r.native_fallbacks(self.calls):
            with self.assertRaises(Exception):
                r.ingest_turns(self.agent, sessions, self.calls)
        self.assertTrue(any("ConnectionError" in message for message in self.calls.errors))
        with self.assertRaises(RuntimeError):
            self.calls.check()

    def test_observer_preserves_original_native_requests_and_memory(self):
        self.invalid_metadata = self.invalid_evolution = True
        sessions = [{"date": "2023/05/20 (Sat) 13:00", "turns": [
                    {"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}]}]
        outcomes = []
        for observed in (False, True):
            self.requests.clear()
            calls = r.Calls(self.root / f"parity_{observed}.jsonl", self.calls.endpoint)
            agent = r.make_agent(self.root / r.MINILM_REVISION, calls)
            with redirect_stdout(io.StringIO()), r.native_fallbacks(calls) if observed else nullcontext():
                count = r.ingest_turns(agent, sessions, calls)
                hypothesis, _ = r.answer(agent, {"question": "public query", "question_date": "D"}, calls)
            outcomes.append({"count": count, "hypothesis": hypothesis,
                             "requests": json.dumps(self.requests, sort_keys=True),
                             "memory": [(n.content, n.context, n.keywords, n.tags, n.links) for n in agent.memory_system.memories.values()]})
            self.assertEqual(len(calls.handled_fallbacks), 4 if observed else 0)
            self.assertEqual(calls.errors, [])
        self.assertEqual(outcomes[0], outcomes[1])

    def test_trace_restored_after_handled_and_propagated_error(self):
        previous = sys.gettrace()
        with r.native_fallbacks(self.calls):
            pass
        self.assertIs(sys.gettrace(), previous)
        with self.assertRaisesRegex(ValueError, "propagated"):
            with r.native_fallbacks(self.calls):
                raise ValueError("propagated")
        self.assertIs(sys.gettrace(), previous)

    def test_worker_seals_degradation_and_preserves_fatal_diagnostics(self):
        for outcome in ("clean", "handled", "transport", "unhandled", "blank"):
            with self.subTest(outcome=outcome):
                self.invalid_metadata = self.invalid_evolution = outcome == "handled"
                self.metadata_transport_error = outcome == "transport"
                self.unhandled_evolution = outcome == "unhandled"
                self.blank_qa = outcome == "blank"
                attempt = r.next_attempt(self.root / outcome / "history")
                source = [{"date": "2023/05/20 (Sat) 13:00", "turns": [{"role": "user", "content": "source fact"}]}]
                query = {"question_id": "q", "question": "public query", "question_date": "D"}
                identity = {"protocol_sha256": "p", "source_sha256": r.digest(source), "query_sha256": r.digest(query)}
                for name, value in (("source.json", source), ("query.json", query), ("worker.json", {
                        "identity": identity, "source_files_sha256": r.source_hashes(),
                        "runtime": {"api_base": "http://127.0.0.1:18083/v1", "embedding_model": str(self.root / r.MINILM_REVISION)}})):
                    r.save_json(attempt / name, value)
                utils = ModuleType("utils"); utils.__path__ = []
                metering = ModuleType("utils.request_metering")
                metering.install_request_metering = lambda: None
                metering.meter_operation = lambda *args, **kwargs: nullcontext()
                with patch.dict(sys.modules, {"utils": utils, "utils.request_metering": metering}), redirect_stdout(io.StringIO()):
                    result = r.worker(attempt)
                success = outcome in {"clean", "handled"}
                self.assertEqual(result, 0 if success else 1)
                degradation = r.read_json(attempt / "native_degradation.json")
                expected_each = int(outcome == "handled")
                self.assertEqual(degradation["native_metadata_fallback_count"], expected_each)
                self.assertEqual(degradation["native_evolution_fallback_count"], expected_each)
                if success:
                    self.assertEqual(r.verified(attempt.parent, identity)["hypothesis"], "native answer")
                    receipt = r.read_json(attempt / "completion.json")
                    self.assertIn("native_degradation.json", receipt["files_sha256"])
                    for name in ("build_complete.json", "usage.json", "prediction.json"):
                        counts = r.read_json(attempt / name)
                        self.assertEqual(counts["native_metadata_fallback_count"], expected_each)
                        self.assertEqual(counts["native_evolution_fallback_count"], expected_each)
                else:
                    self.assertTrue((attempt / "failure.json").is_file())
                    self.assertFalse((attempt / "completion.json").exists())
                    self.assertIsNone(r.verified(attempt.parent, identity))
                    if outcome == "transport":
                        self.assertTrue(r.read_json(attempt / "failure.json")["native_errors"])

    def test_transport_exception_recorded_and_final_empty_answer_rejected(self):
        calls = r.Calls(self.root / "transport.jsonl", "http://127.0.0.1:18083/v1")
        def fail(**kwargs):
            raise ConnectionError("local model unavailable")
        controller = SimpleNamespace(model=r.MODEL, client=SimpleNamespace(base_url=calls.endpoint,
            chat=SimpleNamespace(completions=SimpleNamespace(create=fail))))
        calls.bind(controller)
        with self.assertRaises(ConnectionError):
            controller.client.chat.completions.create(model=r.MODEL, max_tokens=1000, temperature=0.7)
        self.assertEqual(len(calls.records), 1)
        self.agent.answer_question = lambda *args, **kwargs: ('{"answer":" "}', "prompt", "context")
        with self.assertRaises(ValueError):
            r.answer(self.agent, {"question": "Q", "question_date": "D"}, self.calls)

    def test_endpoint_and_model_escape_rejected(self):
        for value in ("https://api.openai.com/v1", "http://user:secret@127.0.0.1:80/v1", "http://127.0.0.1:80/v1?x=y"):
            with self.assertRaises(ValueError):
                r.local_endpoint(value)
        controller = self.agent.memory_system.llm_controller.llm
        controller.model = "gpt-4"
        with self.assertRaises(ValueError):
            self.calls.bind(controller)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def completed(self):
        history = self.root / "history"
        attempt = r.next_attempt(history)
        identity = {"protocol_sha256": "p", "source_sha256": "s", "query_sha256": "q"}
        for name in ("source.json", "query.json", "worker.json", "memory.json", "build_complete.json", "usage.json", "native_answer.json", "native_degradation.json"):
            r.save_json(attempt / name, {})
        (attempt / "llm_calls.jsonl").write_text("{}\n")
        (attempt / "retriever.pkl").write_bytes(b"cache")
        (attempt / "embeddings.npy").write_bytes(b"embedding-cache")
        prediction = {"identity": identity, "status": "generated", "hypothesis": "answer", "question_id": "q"}
        r.save_json(attempt / "prediction.json", prediction)
        r.seal(attempt, identity)
        return history, attempt, identity, prediction

    def test_success_resume_checks_identity_and_native_artifacts(self):
        history, attempt, identity, prediction = self.completed()
        self.assertEqual(r.verified(history, identity), prediction)
        with self.assertRaises(ValueError):
            r.verified(history, dict(identity, source_sha256="different"))
        r.save_json(attempt / "memory.json", {"changed": True})
        with self.assertRaises(ValueError):
            r.verified(history, identity)

    def test_degradation_artifact_is_mandatory_and_tamper_checked(self):
        history, attempt, identity, _ = self.completed()
        receipt = r.read_json(attempt / "completion.json")
        del receipt["files_sha256"]["native_degradation.json"]
        r.save_json(attempt / "completion.json", receipt)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            r.verified(history, identity)
        r.seal(attempt, identity)
        r.save_json(attempt / "native_degradation.json", {"tampered": True})
        with self.assertRaisesRegex(ValueError, "modified"):
            r.verified(history, identity)

    def test_unsealed_prediction_not_reused_and_failure_preserved(self):
        history = self.root / "history"
        first = r.next_attempt(history)
        r.save_json(first / "failure.json", {"error": "first failure"})
        r.save_json(first / "prediction.json", {"status": "generated", "hypothesis": "stale"})
        self.assertIsNone(r.verified(history, {}))
        second = r.next_attempt(history)
        self.assertNotEqual(first, second)
        self.assertEqual(r.read_json(first / "failure.json"), {"error": "first failure"})

    def test_run_failure_stops_before_next_history_and_writes_incomplete_status(self):
        records = [{"question_id": str(i), "question": "Q", "question_date": "D", "answer": "GOLD",
                    "haystack_sessions": [[{"role": "user", "content": "text", "has_answer": True}]],
                    "haystack_dates": ["2023/05/20 (Sat) 13:00"], "haystack_session_ids": ["s"]} for i in range(500)]
        dataset = self.root / "dataset.json"
        r.save_json(dataset, records)
        embedding = self.root / r.MINILM_REVISION
        embedding.mkdir()
        for name in ("config.json", "modules.json", "sentence_bert_config.json"):
            r.save_json(embedding / name, {})
        args = argparse.Namespace(dataset=dataset, ids_file=None, run_dir=self.root / "run",
                                  embedding_model=embedding, api_base="http://127.0.0.1:18083/v1")
        with patch.object(r, "DATA_SHA256", r.sha(dataset)), patch.object(r.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as child:
            self.assertEqual(r.run(args), 1)
        self.assertEqual(child.call_count, 1)
        status = r.read_json(args.run_dir / "status.json")
        self.assertEqual((status["planned"], status["generated"], status["failed"]), (500, 0, 1))
        self.assertEqual(status["status"], "generation_incomplete")
        worker_source = next(args.run_dir.glob("histories/*/attempt_*/source.json")).read_text()
        self.assertNotIn("GOLD", worker_source)
        self.assertNotIn("has_answer", worker_source)

    def test_exact_pinned_files_and_class_only_boundary(self):
        hashes = r.source_hashes()
        self.assertEqual(hashes["upstream/memory_layer.py"], "b9a5b5797881b25c5524b98a543b22a1bbbcc7f285437ca667a5673e7f293eba")
        path = self.root / "module.py"
        path.write_text("raise RuntimeError('must not execute')\nclass Native:\n    def call(self): return 7\n")
        scope = r.compile_classes(path, {"Native"}, {})
        self.assertEqual(scope["Native"]().call(), 7)


if __name__ == "__main__":
    unittest.main()