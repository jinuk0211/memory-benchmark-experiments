"""CPU-only regression checks; no model API calls or native database writes."""
import argparse
import asyncio
from contextlib import nullcontext
import copy
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("langmem_sessions_candidate", Path(__file__).with_name("runner.py"))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


class FakeResponse:
    def model_dump(self, **kwargs):
        return {"choices": [{"finish_reason": "tool_calls"}], "usage": {"total_tokens": 4}}


class Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = {"api_base": "http://127.0.0.1:18083/v1", "embedding_api_base": "http://127.0.0.1:18084/v1"}

    def test_official_sources_and_defaults_unchanged(self):
        hashes = r.source_hashes()
        self.assertEqual(sum(k.startswith("upstream/") for k in hashes), 38)
        import ast
        file = r.HERE / "upstream/langmem/src/langmem/knowledge/extraction.py"
        tree = ast.parse(file.read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_memory_store_manager")
        defaults = dict(zip((x.arg for x in function.args.kwonlyargs), function.args.kw_defaults))
        self.assertIs(ast.literal_eval(defaults["enable_deletes"]), False)
        self.assertEqual(ast.literal_eval(defaults["query_limit"]), 5)
        self.assertIn("DEFAULT_MAX_ATTEMPTS = 3", (r.HERE / "upstream/trustcall/trustcall/_base.py").read_text(encoding="utf-8"))

    def test_canonical_53_sessions_550_original_turns(self):
        dataset = r.HARNESS.parents[1] / "MemoryData/datasets/LongMemEval/longmemeval_s_cleaned.json"
        raw = dataset.read_bytes()
        self.assertEqual(r.sha(dataset), r.DATA_SHA256)
        record = next(x for x in json.loads(raw) if x["question_id"] == "e47becba")
        source = r.source_only(record)
        calls = []
        manager = SimpleNamespace(invoke=lambda value: calls.append(copy.deepcopy(value)) or [])
        counts = r.ingest_sessions(manager, source, r.NativeFailures(), self.root / "calls.jsonl")
        self.assertEqual(counts, {"sessions": 53, "source_turns": 550})
        self.assertEqual(len(calls), 53)
        for session, call in zip(source, calls):
            self.assertEqual(call["messages"][1:], session["turns"])
            self.assertEqual(call["messages"][0], {"role": "system", "content": f'Session {session["session_id"]} ({session["date"]})'})
            self.assertTrue(all(set(turn) == {"role", "content"} for turn in call["messages"]))
        self.assertEqual(sum(s["turns"][0]["role"] == "assistant" for s in source), 4)
        self.assertEqual(sum(len(s["turns"]) % 2 for s in source), 4)

    def test_labels_excluded_and_long_turn_not_split(self):
        row = {"haystack_sessions": [[{"role": "assistant", "content": "x" * 30000, "has_answer": True}]],
               "haystack_dates": ["public date"], "haystack_session_ids": ["s"],
               "answer": "SECRET_GOLD", "question_type": "SECRET_LABEL", "question": "SECRET_QUERY"}
        source = r.source_only(row)
        messages = r.session_messages(source[0])
        serialized = json.dumps(messages)
        self.assertNotIn("SECRET", serialized)
        self.assertNotIn("has_answer", serialized)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"], "x" * 30000)

    def test_only_exact_native_handled_drops_are_allowed(self):
        failures = r.NativeFailures()
        failures.emit(logging.LogRecord("native", logging.WARNING, "file", 1, "Retrying validation", (), None, "invoke"))
        failures.check()
        failures.emit(logging.LogRecord("native", logging.ERROR, failures.native_file, 823,
            "Could not apply patch: missing field", (), None, "_teardown"))
        failures.check()
        self.assertEqual(failures.degradation()["native_handled_patch_drop_count"], 1)
        for filename, line, function in [("other.py", 823, "_teardown"),
                (failures.native_file, 824, "_teardown"), (failures.native_file, 823, "other_function")]:
            with self.subTest(filename=filename, line=line, function=function):
                observer = r.NativeFailures()
                observer.emit(logging.LogRecord("native", logging.ERROR, filename, line,
                    "Could not apply patch: missing field", (), None, function))
                with self.assertRaises(RuntimeError):
                    observer.check()

    def test_handled_drop_keeps_all_53_sessions_550_turns(self):
        dataset = r.HARNESS.parents[1] / "MemoryData/datasets/LongMemEval/longmemeval_s_cleaned.json"
        self.assertEqual(r.sha(dataset), r.DATA_SHA256)
        row = next(x for x in json.loads(dataset.read_bytes()) if x["question_id"] == "e47becba")
        source = r.source_only(row)
        failures, calls = r.NativeFailures(), []
        def invoke(value):
            calls.append(copy.deepcopy(value))
            if len(calls) == 9:
                failures.emit(logging.LogRecord("extraction", logging.ERROR, failures.native_file, 823,
                    "Could not apply patch: synthetic invalid path", (), None, "_teardown"))
            return []
        counts = r.ingest_sessions(SimpleNamespace(invoke=invoke), source, failures, self.root / "sessions.jsonl")
        self.assertEqual(counts, {"sessions": 53, "source_turns": 550})
        self.assertEqual([c["messages"][1:] for c in calls], [s["turns"] for s in source])
        journal = [json.loads(x) for x in (self.root / "sessions.jsonl").read_text().splitlines()]
        self.assertEqual(len(journal), 53)
        self.assertEqual(sum(x["native_handled_patch_drops"] for x in journal), 1)
        self.assertEqual(journal[8]["native_handled_patch_drops"], 1)
        self.assertEqual(failures.degradation()["native_handled_patch_drop_count"], 1)

    def test_actual_native_teardown_drop_logs_are_counted_once(self):
        import ast
        import __future__
        import uuid
        path = r.HERE / "upstream/trustcall/trustcall/_base.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "_ExtractUpdates")
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_teardown")
        node.decorator_list = []
        class AI:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.id = None
        def fail_patch(*args):
            raise ValueError("can't replace a non-existent object ''")
        namespace = {"AIMessage": AI, "PatchDoc": type("PatchDoc", (), {}), "ToolCall": dict,
            "logger": logging.getLogger("native_teardown_test"), "uuid": uuid,
            "ls": SimpleNamespace(get_current_run_tree=lambda: None),
            "_ensure_patches": lambda args: args["patches"], "_apply_patch": fail_patch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path.resolve()), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        cases = [([("doc", "Memory", {"content": "old"})], 823),
                 ([], 788), ([("doc", None, {"content": "old"})], 797),
                 ([("doc", "Memory", {})], 829)]
        for existing, line in cases:
            with self.subTest(line=line):
                observer = r.NativeFailures()
                incoming = SimpleNamespace(content="", tool_calls=[
                    {"id": "bad", "name": "PatchDoc", "args": {"json_doc_id": "doc", "patches": ["invalid"]}},
                    {"id": "good", "name": "Memory", "args": {"content": "synthetic valid"}}])
                logging.getLogger().addHandler(observer)
                try:
                    with observer.observe():
                        result = namespace["_teardown"](SimpleNamespace(tool_choice="required"), incoming, existing)
                    observer.check()
                finally:
                    logging.getLogger().removeHandler(observer)
                self.assertEqual([x["id"] for x in result["messages"][0].tool_calls], ["good"])
                self.assertEqual(len(observer.handled_patch_drops), 1)
                self.assertEqual(observer.handled_patch_drops[0]["line"], line)
                self.assertEqual(observer.records, [])

    def test_qa_sync_and_async_requests_preserve_kwargs_and_local_binding(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                calls = r.Calls(self.root / f"calls_{asynchronous}.jsonl", self.runtime)
                calls.phase = "qa"
                seen = []
                response = FakeResponse()
                def sync(**kwargs):
                    seen.append(kwargs)
                    return response
                async def async_call(**kwargs):
                    return sync(**kwargs)
                client = SimpleNamespace(base_url=self.runtime["api_base"] + "/", api_key="EMPTY",
                    chat=SimpleNamespace(completions=SimpleNamespace(create=async_call if asynchronous else sync)),
                    embeddings=SimpleNamespace(create=async_call if asynchronous else sync))
                calls.bind(client, asynchronous)
                kwargs = {"model": r.MODEL, "temperature": 0, "max_tokens": None,
                          "messages": [{"role": "user", "content": "source"}],
                          "tools": [{"type": "function", "function": {"name": "PatchDoc"}}]}
                observed = client.chat.completions.create(**kwargs)
                if asynchronous: observed = asyncio.run(observed)
                self.assertIs(observed, response)
                self.assertEqual(seen, [kwargs])
                self.assertEqual(calls.count, 1)
                bad = dict(kwargs, model="external-model")
                with self.assertRaises(ValueError):
                    result = client.chat.completions.create(**bad)
                    if asynchronous: asyncio.run(result)
                self.assertEqual(len(seen), 1)

    def test_embedding_strings_only_and_external_urls_rejected(self):
        calls = r.Calls(self.root / "calls.jsonl", self.runtime)
        client = SimpleNamespace(base_url=self.runtime["embedding_api_base"], api_key="EMPTY")
        calls.validate("embedding", client, {"model": r.EMBEDDING_MODEL, "input": ["text", "other"]})
        with self.assertRaises(ValueError):
            calls.validate("embedding", client, {"model": r.EMBEDDING_MODEL, "input": [[123, 456]]})
        for url in ["https://api.openai.com/v1", "http://localhost:8080/v1?redirect=x", "http://user:secret@localhost:8080/v1"]:
            with self.assertRaises(ValueError): r.local_endpoint(url)

    def test_client_constructor_guard_covers_both_clients_and_restores(self):
        fake = ModuleType("openai")
        class Client:
            def __init__(self, base_url, api_key):
                self.base_url, self.api_key = base_url, api_key
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: FakeResponse()))
                self.embeddings = SimpleNamespace(create=lambda **kw: FakeResponse())
        class AsyncClient(Client):
            pass
        fake.OpenAI, fake.AsyncOpenAI = Client, AsyncClient
        before = (Client.__init__, AsyncClient.__init__)
        calls = r.Calls(self.root / "calls.jsonl", self.runtime)
        with patch.dict(sys.modules, {"openai": fake}), calls.clients():
            fake.OpenAI(base_url=self.runtime["api_base"], api_key="EMPTY")
            for cls in (fake.OpenAI, fake.AsyncOpenAI):
                with self.assertRaises(ValueError):
                    cls(base_url="https://api.openai.com/v1", api_key="EMPTY")
        self.assertEqual((Client.__init__, AsyncClient.__init__), before)


    def bounded_client(self, name, response, asynchronous=False):
        calls = r.Calls(self.root / f"{name}.jsonl", self.runtime)
        seen = []
        def sync(**kwargs):
            seen.append(copy.deepcopy(kwargs))
            if isinstance(response, Exception):
                raise response
            return SimpleNamespace(model_dump=lambda **unused: copy.deepcopy(response))
        async def async_call(**kwargs):
            return sync(**kwargs)
        client = SimpleNamespace(base_url=self.runtime["api_base"], api_key="EMPTY", max_retries=2,
            chat=SimpleNamespace(completions=SimpleNamespace(create=async_call if asynchronous else sync)),
            embeddings=SimpleNamespace(create=async_call if asynchronous else sync))
        other = SimpleNamespace(base_url=self.runtime["api_base"], api_key="EMPTY", max_retries=2)
        model = SimpleNamespace(root_client=client, root_async_client=other, max_tokens=None, max_retries=2)
        calls.bind(client, asynchronous)
        calls.configure_memory(model)
        return calls, client, model, seen

    @staticmethod
    def bounded_request():
        return {"model": r.MODEL, "temperature": 0, "max_completion_tokens": 8192,
            "stream": False, "tool_choice": "required", "messages": [{"role": "user", "content": "source fact"}],
            "tools": [{"type": "function", "function": {"name": "Memory", "parameters": {"type": "object"}}}]}

    @staticmethod
    def bounded_response(finish="tool_calls", tools=True):
        return {"choices": [{"finish_reason": finish, "message": {"content": None,
            "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "Memory", "arguments": "{}"}}] if tools else []}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112}}

    def test_bound_applies_only_to_memory_model_owned_clients(self):
        calls, client, model, seen = self.bounded_client("bound", self.bounded_response())
        self.assertEqual((model.max_tokens, model.max_retries, client.max_retries, model.root_async_client.max_retries), (8192, 0, 0, 0))
        request = self.bounded_request()
        result = client.chat.completions.create(**request)
        self.assertEqual(seen, [request])
        self.assertEqual(result.model_dump(), self.bounded_response())
        calls.check()
        qa_client = SimpleNamespace(base_url=self.runtime["api_base"], api_key="EMPTY", max_retries=2,
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: seen.append(kwargs) or FakeResponse())),
            embeddings=SimpleNamespace(create=lambda **kwargs: FakeResponse()))
        calls.bind(qa_client, False)
        calls.phase = "qa"
        qa_kwargs = {"model": r.MODEL, "temperature": 0.7, "max_tokens": 256, "messages": []}
        qa_client.chat.completions.create(**qa_kwargs)
        self.assertEqual(qa_client.max_retries, 2)
        self.assertEqual(seen[-1], qa_kwargs)

    def test_sync_async_memory_snapshot_is_exact_and_hash_bound_without_headers(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                calls, client, _, seen = self.bounded_client(f"snapshot_{asynchronous}", self.bounded_response(), asynchronous)
                request = self.bounded_request()
                request["extra_headers"] = {"Authorization": "SYNTHETIC_SECRET_HEADER"}
                result = client.chat.completions.create(**request)
                if asynchronous:
                    asyncio.run(result)
                self.assertEqual(seen, [request])
                events = [json.loads(line) for line in calls.path.read_text().splitlines()]
                self.assertEqual([event["event"] for event in events], ["begin", "end"])
                for kind, expected in (("request", {k: v for k, v in request.items() if k != "extra_headers"}), ("response", self.bounded_response())):
                    item = events[-1][kind + "_artifact"]
                    path = calls.path.parent / item["path"]
                    self.assertEqual(r.sha(path), item["sha256"])
                    self.assertEqual(r.read_json(path), expected)
                    self.assertNotIn("SYNTHETIC_SECRET_HEADER", path.read_text())
                self.assertNotIn("SYNTHETIC_SECRET_HEADER", calls.path.read_text())

    def test_bad_memory_responses_are_preserved_then_fatal_sync_and_async(self):
        for asynchronous in (False, True):
            for finish, tools in (("length", True), ("content_filter", True), ("stop", False)):
                with self.subTest(asynchronous=asynchronous, finish=finish, tools=tools):
                    name = f"reject_{asynchronous}_{finish}_{tools}"
                    payload = self.bounded_response(finish, tools)
                    calls, client, _, seen = self.bounded_client(name, payload, asynchronous)
                    with self.assertRaises(ValueError):
                        result = client.chat.completions.create(**self.bounded_request())
                        if asynchronous:
                            asyncio.run(result)
                    self.assertEqual(len(seen), 1)
                    events = [json.loads(line) for line in calls.path.read_text().splitlines()]
                    artifact = events[-1]["response_artifact"]
                    self.assertEqual(r.read_json(calls.path.parent / artifact["path"]), payload)
                    self.assertEqual(events[-1]["error_class"], "ValueError")
                    with self.assertRaises(RuntimeError):
                        calls.check()
                    with self.assertRaises(RuntimeError):
                        result = client.chat.completions.create(**self.bounded_request())
                        if asynchronous:
                            asyncio.run(result)
                    self.assertEqual(len(seen), 1)

    def test_transport_error_is_not_retried_and_swallowed_gate_stops_next_session(self):
        calls, client, _, seen = self.bounded_client("transport", TimeoutError("synthetic transport timeout"))
        def invoke(_payload):
            try:
                client.chat.completions.create(**self.bounded_request())
            except TimeoutError:
                return []
        source = [{"session_id": str(i), "date": "D", "turns": [{"role": "user", "content": "fact"}]} for i in range(2)]
        with self.assertRaises(RuntimeError):
            r.ingest_sessions(SimpleNamespace(invoke=invoke), source, r.NativeFailures(), self.root / "sessions.jsonl", calls)
        self.assertEqual(len(seen), 1)
        self.assertFalse((self.root / "sessions.jsonl").exists())
        self.assertEqual(client.max_retries, 0)

    def test_memory_policy_escape_and_missing_choices_are_fatal(self):
        for index, changes in enumerate(({"max_completion_tokens": None}, {"max_completion_tokens": 16384}, {"stream": True}, {"max_tokens": 8192})):
            calls, client, _, seen = self.bounded_client(f"escape_{index}", self.bounded_response())
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                client.chat.completions.create(**dict(self.bounded_request(), **changes))
            self.assertEqual(seen, [])
            with self.assertRaises(RuntimeError):
                calls.check()
        with self.assertRaises(ValueError):
            r.Calls.check_memory_response({"choices": []}, self.bounded_request())


    def test_raw_sdk_response_retained_and_original_parse_unmodified(self):
        import httpx
        import openai
        for finish in ("tool_calls", "length"):
            with self.subTest(finish=finish):
                received = []
                payload = dict(self.bounded_response(finish), id="synthetic", object="chat.completion", created=1, model=r.MODEL)
                wire = json.dumps(payload, separators=(",", ":")).encode()
                def transport(request):
                    received.append(request)
                    return httpx.Response(200, content=wire, headers={"content-type": "application/json"})
                with httpx.Client(transport=httpx.MockTransport(transport)) as http_client:
                    with openai.OpenAI(base_url=self.runtime["api_base"], api_key="EMPTY", http_client=http_client) as client:
                        calls = r.Calls(self.root / f"raw_{finish}.jsonl", self.runtime)
                        calls.bind(client, False)
                        calls.configure_memory(SimpleNamespace(root_client=client, root_async_client=client))
                        request = self.bounded_request()
                        request["max_tokens"] = request.pop("max_completion_tokens")
                        if finish == "length":
                            with self.assertRaisesRegex(ValueError, "truncated"):
                                client.chat.completions.with_raw_response.create(**request)
                        else:
                            raw = client.chat.completions.with_raw_response.create(**request)
                            parsed = raw.parse()
                            self.assertEqual(parsed.choices[0].message.tool_calls[0].function.name, "Memory")
                            self.assertIs(raw.parse(), parsed)
                self.assertEqual(len(received), 1)
                entry = json.loads(calls.path.read_text().splitlines()[-1])
                for kind, expected in (("request_http", received[0].content), ("response_http", wire)):
                    item = entry[kind + "_artifact"]
                    path = calls.path.parent / item["path"]
                    self.assertEqual(path.read_bytes(), expected)
                    self.assertEqual(r.sha(path), item["sha256"])
                self.assertEqual(entry["transport_max_retries"], 0)

    def test_native_manager_shares_model_and_binds_extractor_after_configuration(self):
        import ast
        import __future__
        path = r.HERE / "upstream/langmem/src/langmem/knowledge/extraction.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        def native_method(class_name, method, namespace):
            node = next(node for node in classes[class_name].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
            return namespace[method]
        class Model:
            pass
        class Memory:
            pass
        namespace = {"BaseChatModel": Model, "Memory": Memory, "_MEMORY_INSTRUCTIONS": "unchanged",
            "utils": SimpleNamespace(NamespaceTemplate=lambda x: x), "create_search_memory_tool": lambda **kwargs: object()}
        initialize_memory = native_method("MemoryManager", "__init__", namespace)
        def create_memory(model, **kwargs):
            manager = SimpleNamespace()
            initialize_memory(manager, model, **kwargs)
            return manager
        namespace["create_memory_manager"] = create_memory
        initialize_store = native_method("MemoryStoreManager", "__init__", namespace)
        model = Model()
        clients = [SimpleNamespace(base_url=self.runtime["api_base"], api_key="EMPTY", max_retries=2) for _ in range(2)]
        model.root_client, model.root_async_client = clients
        store = SimpleNamespace()
        initialize_store(store, model, store=object())
        self.assertIs(store.memory_manager.model, model)
        calls = r.Calls(self.root / "native_binding.jsonl", self.runtime)
        calls.configure_memory(store.model)
        seen = []
        class BoundaryReached(Exception):
            pass
        def extract(actual_model, **kwargs):
            seen.append((actual_model, actual_model.max_tokens, actual_model.root_client.max_retries, kwargs))
            raise BoundaryReached()
        namespace["create_extractor"] = extract
        manager = store.memory_manager
        manager._prepare_messages = lambda messages, steps: messages
        manager._prepare_existing = lambda existing: existing
        for name in ("invoke", "ainvoke"):
            invoke = native_method("MemoryManager", name, namespace)
            with self.assertRaises(BoundaryReached):
                result = invoke(manager, {"messages": [], "existing": []})
                if name == "ainvoke":
                    asyncio.run(result)
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(row[0] is model and row[1:3] == (8192, 0) for row in seen))

    def test_malformed_raw_json_is_preserved_and_stays_fatal_after_native_catch(self):
        import httpx
        request = httpx.Request("POST", self.runtime["api_base"] + "/chat/completions", json=self.bounded_request())
        wire = b'{"choices":['
        response = SimpleNamespace(http_response=httpx.Response(200, content=wire, request=request))
        calls = r.Calls(self.root / "malformed.jsonl", self.runtime)
        client = SimpleNamespace(base_url=self.runtime["api_base"], api_key="EMPTY", max_retries=2,
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)),
            embeddings=SimpleNamespace(create=lambda **kwargs: None))
        calls.bind(client, False)
        calls.configure_memory(SimpleNamespace(root_client=client, root_async_client=client))
        with self.assertRaises(json.JSONDecodeError):
            client.chat.completions.create(**self.bounded_request())
        with self.assertRaises(RuntimeError):
            calls.check()
        entry = json.loads(calls.path.read_text().splitlines()[-1])
        item = entry["response_http_artifact"]
        path = calls.path.parent / item["path"]
        self.assertEqual(path.read_bytes(), wire)
        self.assertEqual(r.sha(path), item["sha256"])
        self.assertEqual(entry["error_class"], "JSONDecodeError")

    def fixture_completion(self, history, identity):
        attempt = r.next_attempt(history)
        names = ["source.json", "query.json", "worker.json", "memory.json", "native_sessions.json",
                 "session_calls.jsonl", "build_complete.json", "usage.json", "native_runtime.json",
                 "native_config.json", "native_dataset_config.json", "llm_calls.jsonl", "native_degradation.json"]
        for name in names: r.save_json(attempt / name, {})
        r.save_json(attempt / "prediction.json", {"identity": identity, "status": "generated", "question_id": "q", "hypothesis": "answer"})
        r.seal(attempt, identity)
        return attempt

    def test_completed_cache_requires_identity_and_all_artifacts(self):
        identity = {"source_sha256": "source", "query_sha256": "query", "protocol_sha256": "protocol"}
        history = self.root / "history"
        attempt = self.fixture_completion(history, identity)
        self.assertEqual(r.verified(history, identity)["hypothesis"], "answer")
        with self.assertRaises(ValueError): r.verified(history, dict(identity, query_sha256="changed"))
        r.save_json(attempt / "extra.json", {})
        with self.assertRaisesRegex(ValueError, "inventory"): r.verified(history, identity)
        (attempt / "extra.json").unlink()
        r.save_json(attempt / "memory.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "artifact changed"): r.verified(history, identity)

    def test_failed_and_blank_predictions_are_never_reused(self):
        identity, history = {"protocol_sha256": "protocol"}, self.root / "history"
        failed = r.next_attempt(history)
        r.save_json(failed / "failure.json", {"reason": "preserved"})
        self.assertIsNone(r.verified(history, identity))
        attempt = self.fixture_completion(history, identity)
        self.assertEqual(attempt.name, "attempt_0002")
        prediction = r.read_json(attempt / "prediction.json")
        prediction["hypothesis"] = "  "
        r.save_json(attempt / "prediction.json", prediction)
        r.seal(attempt, identity)
        with self.assertRaisesRegex(ValueError, "Invalid completed"): r.verified(history, identity)
        self.assertEqual(r.read_json(failed / "failure.json"), {"reason": "preserved"})

    def test_wrong_native_module_path_is_refused(self):
        fake_langmem = ModuleType("langmem")
        fake_langmem.__file__ = str(self.root / "wrong.py")
        with patch.dict(sys.modules, {"langmem": fake_langmem}):
            with self.assertRaisesRegex(ValueError, "non-pinned"): r.imported_native()
            with self.assertRaisesRegex(ValueError, "already loaded"): r.load_native()


    def test_worker_uses_existing_qa_and_preserves_failure_attempt(self):
        for outcome in ("clean", "fatal_log", "handled", "exception", "blank"):
            with self.subTest(outcome=outcome):
                fail = outcome in {"fatal_log", "exception", "blank"}
                run_dir = self.root / outcome
                attempt = r.next_attempt(run_dir / "histories/h")
                source = [{"session_id": "s", "date": "public date", "turns": [{"role": "assistant", "content": "source fact"}]}]
                query = {"question_id": "q", "question": "public query", "question_date": "query date"}
                runtime = dict(self.runtime, model=r.MODEL, method="langmem", embedding_model=r.EMBEDDING_MODEL,
                               embedding_dims=384, tokenizer="pinned-tokenizer")
                protocol = {"runtime": runtime, "source_files_sha256": {"code": "hash"}}
                identity = {"protocol_sha256": r.digest(protocol), "source_sha256": r.digest(source), "query_sha256": r.digest(query)}
                for name, value in [("source.json", source), ("query.json", query), ("worker.json", {
                        "identity": identity, "runtime": runtime, "source_files_sha256": {"code": "hash"}})]:
                    r.save_json(attempt / name, value)
                r.save_json(run_dir / "protocol.json", protocol)
                seen, answered = [], []
                def invoke(payload):
                    seen.append(copy.deepcopy(payload))
                    if outcome == "fatal_log":
                        logging.getLogger("extraction").error("Could not apply patch: missing field")
                    if outcome == "exception":
                        raise ValueError("unhandled native exception")
                    if outcome == "handled":
                        logging.getLogger().handle(logging.LogRecord("extraction", logging.ERROR,
                            str((r.HERE / "upstream/trustcall/trustcall/_base.py").resolve()), 823,
                            "Could not apply patch: missing field", (), None, "_teardown"))
                    return []
                memory_clients = [SimpleNamespace(base_url=self.runtime["api_base"], api_key="EMPTY", max_retries=2) for _ in range(2)]
                manager = SimpleNamespace(invoke=invoke, query_limit=5, enable_deletes=False,
                    enable_inserts=True, query_model=None, phases=[], model=SimpleNamespace(model_name=r.MODEL, temperature=0,
                        root_client=memory_clients[0], root_async_client=memory_clients[1]))
                manager.memory_manager = SimpleNamespace(model=manager.model)
                def answer(message, **kwargs):
                    answered.append((message, kwargs))
                    return {"output": "  " if outcome == "blank" else "Existing QA answer"}
                agent = SimpleNamespace(benchmark_memory=SimpleNamespace(manager=manager, _live_items=lambda: []),
                    tokenizer=object(), save_agent=lambda: None, close=lambda: None, send_message=answer)
                utils = ModuleType("utils"); utils.__path__ = []
                metering = ModuleType("utils.request_metering")
                metering.install_request_metering = lambda: None
                metering.meter_operation = lambda *a, **kw: nullcontext()
                agent_module = ModuleType("utils.agent"); agent_module.AgentWrapper = object
                modules = {"utils": utils, "utils.request_metering": metering, "utils.agent": agent_module}
                with patch.dict(sys.modules, modules), patch.object(r.os, "chdir"), \
                     patch.object(r, "source_hashes", return_value={"code": "hash"}), \
                     patch.object(r, "load_native"), patch.object(r, "client_versions", return_value=r.CLIENT_VERSIONS), \
                     patch.object(r, "imported_native", return_value={"pinned": True}), \
                     patch.object(r.Calls, "clients", return_value=nullcontext()), \
                     patch.object(r.harness, "build_config", return_value=({"tokenizer_model": "pinned-tokenizer"}, {})), \
                     patch.object(r.harness, "create_agent", return_value=agent), \
                     patch.object(r.harness, "tokenizer_runtime", return_value={"pinned": True}):
                    code = r.worker(attempt)
                self.assertEqual(code, int(fail))
                degradation = r.read_json(attempt / "native_degradation.json")
                self.assertEqual(degradation["native_handled_patch_drop_count"], int(outcome == "handled"))
                self.assertEqual(seen[0]["messages"][1:], source[0]["turns"])
                self.assertNotIn("question", seen[0])
                self.assertTrue((attempt / "native_runtime.json").exists())
                if fail:
                    if outcome != "blank":
                        self.assertEqual(answered, [])
                    self.assertTrue((attempt / "failure.json").exists())
                    self.assertFalse((attempt / "completion.json").exists())
                else:
                    self.assertEqual(answered[0][0], "Question date: query date\nQuestion: public query")
                    self.assertIs(answered[0][1]["memorizing"], False)
                    self.assertEqual(r.verified(attempt.parent, identity)["hypothesis"], "Existing QA answer")

    def test_smoke_selection_expands_with_same_population_and_verified_reuse(self):
        records = [{"question_id": f"q{i}", "question": "query", "question_date": "date", "answer": "GOLD",
            "question_type": "LABEL", "haystack_sessions": [[{"role": "user", "content": "fact", "has_answer": True}]],
            "haystack_dates": ["date"], "haystack_session_ids": ["s"]} for i in range(500)]
        dataset = self.root / "dataset.json"; r.save_json(dataset, records)
        tokenizer = self.root / r.TOKENIZER_REVISION; tokenizer.mkdir()
        for name in ("tokenizer_config.json", "tokenizer.json"): r.save_json(tokenizer / name, {})
        selected = self.root / "selected.json"; r.save_json(selected, ["q0"])
        args = argparse.Namespace(dataset=dataset, run_dir=self.root / "run", ids_file=selected,
            tokenizer=tokenizer, **self.runtime)
        launched = []
        def subprocess_run(argv, **kwargs):
            attempt = Path(argv[-1]); launched.append(attempt)
            spec = r.read_json(attempt / "worker.json")
            self.assertNotIn("GOLD", (attempt / "source.json").read_text())
            self.assertNotIn("has_answer", (attempt / "source.json").read_text())
            for name in ["memory.json", "native_sessions.json", "session_calls.jsonl", "build_complete.json", "usage.json",
                         "native_runtime.json", "native_config.json", "native_dataset_config.json", "llm_calls.jsonl", "native_degradation.json"]:
                r.save_json(attempt / name, {})
            qid = r.read_json(attempt / "query.json")["question_id"]
            r.save_json(attempt / "prediction.json", {"identity": spec["identity"], "question_id": qid,
                "status": "generated", "hypothesis": "answer", "officially_judged": False})
            r.seal(attempt, spec["identity"])
            return SimpleNamespace(returncode=0)
        with patch.object(r, "DATA_SHA256", r.sha(dataset)), patch.object(r, "source_hashes", return_value={"code": "hash"}), \
             patch.object(r.subprocess, "run", side_effect=subprocess_run):
            self.assertEqual(r.run(args), 0)
            original_protocol = (args.run_dir / "protocol.json").read_bytes()
            r.save_json(selected, ["q0", "q1"])
            self.assertEqual(r.run(args), 0)
            self.assertEqual((args.run_dir / "protocol.json").read_bytes(), original_protocol)
            self.assertEqual(len(launched), 2)
            self.assertEqual(len(r.read_json(args.run_dir / "protocol.json")["population_ids"]), 500)
            self.assertEqual(r.read_json(args.run_dir / "status.json")["generated"], 2)
            args.ids_file = None
            self.assertEqual(r.run(args), 0)
            self.assertEqual(len(launched), 500)
            status = r.read_json(args.run_dir / "status.json")
            self.assertEqual((status["planned"], status["generated"], status["population"]), (500, 500, 500))
            self.assertEqual(status["failed"], 0)
            self.assertEqual(len(r.read_json(args.run_dir / "predictions.json")), 500)
            with patch.object(r, "source_hashes", return_value={"code": "changed"}):
                with self.assertRaisesRegex(ValueError, "Existing identity/input differs"): r.run(args)

    def original_ast(self, name, namespace):
        import ast
        import __future__
        path = r.HERE / "upstream/trustcall/trustcall/_base.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path.resolve()), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        return namespace[name]

    def test_original_repair_error_then_success_is_allowed(self):
        class AI:
            id = "ai"
            tool_calls = [{"id": "tool", "name": "Memory", "args": {"content": "old"}}]
        class Tool:
            id = "validation"
            tool_call_id = "tool"
        count = []
        def apply_patch(args, patches):
            count.append(patches)
            if len(count) == 1: raise ValueError("first repair invalid")
            return {"content": "repaired"}
        function = self.original_ast("_get_message_op", {"AIMessage": AI, "ToolMessage": Tool,
            "ls": SimpleNamespace(get_current_run_tree=lambda: None), "logger": logging.getLogger("trustcall_test"),
            "_ensure_patches": lambda value: value["patches"], "_apply_patch": apply_patch, "MessageOp": dict})
        failures = r.NativeFailures()
        logging.getLogger().addHandler(failures)
        try:
            with failures.observe():
                first = function([AI(), Tool()], {"patches": ["first"]}, "PatchFunctionErrors", "tool")
                second = function([AI(), Tool()], {"patches": ["second"]}, "PatchFunctionErrors", "tool")
            failures.check()
        finally:
            logging.getLogger().removeHandler(failures)
        self.assertEqual([x["op"] for x in first], ["delete"])
        self.assertEqual(second[0]["target"]["args"], {"content": "repaired"})
        self.assertEqual(len(failures.warnings), 1)

    def test_original_patch_sync_and_async_terminal_abort_is_failed(self):
        import ast
        import __future__
        path = r.HERE / "upstream/trustcall/trustcall/_base.py"
        cls = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef) and n.name == "_Patch")
        for name in ("invoke", "ainvoke"):
            with self.subTest(name=name):
                node = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
                namespace = {"Command": lambda **kwargs: SimpleNamespace(**kwargs)}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path.resolve()), "exec",
                             flags=__future__.annotations.compiler_flag), namespace)
                def fail(*args): raise TimeoutError("terminal transport failure")
                async def async_fail(*args): raise TimeoutError("terminal transport failure")
                instance = SimpleNamespace(bound=SimpleNamespace(invoke=fail, ainvoke=async_fail))
                failures = r.NativeFailures()
                with failures.observe():
                    result = namespace[name](instance, SimpleNamespace(messages=[]), {})
                    if name == "ainvoke": result = asyncio.run(result)
                self.assertEqual(result.goto, "__end__")
                with self.assertRaises(RuntimeError): failures.check()
                self.assertIn(1101 if name == "invoke" else 1084, [x["line"] for x in failures.records])

    def test_original_empty_memory_is_allowed_but_missing_final_message_is_not(self):
        class AI:
            id = "ai"
            additional_kwargs = {}
            tool_calls = []
        function = self.original_ast("filter_state", {"AIMessage": AI, "ExtractionOutputs": dict,
            "enable_deletes": False, "existing_schema_policy": False})
        failures = r.NativeFailures()
        with failures.observe():
            result = function({"msg_id": "ai", "messages": [AI()], "attempts": 1})
        self.assertEqual(result["responses"], [])
        failures.check()
        with failures.observe():
            function({"msg_id": "missing", "messages": [], "attempts": 3})
        with self.assertRaises(RuntimeError): failures.check()

if __name__ == "__main__":
    unittest.main()