"""Replay exact pinned return branches without network or model execution."""
import __future__
import ast
import asyncio
from contextlib import contextmanager
import copy
import json
import logging
from types import SimpleNamespace
import unittest
import uuid

from pydantic import BaseModel, Field
import test_runner as fixtures

r = fixtures.r
NATIVE = r.HERE / "upstream/trustcall/trustcall/_base.py"
EXTRACTION = r.HERE / "upstream/langmem/src/langmem/knowledge/extraction.py"


class AI:
    def __init__(self, tools=()):
        self.id = "ai"
        self.tool_calls = list(tools)
        self.additional_kwargs = {}


def pinned_node(path, name, parent=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if parent:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent)
    return next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name)


def compile_native(path, name, namespace, parent=None):
    node = pinned_node(path, name, parent)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path.resolve()), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def native_filter():
    memory = compile_native(EXTRACTION, "Memory", {"BaseModel": BaseModel, "Field": Field})
    return compile_native(NATIVE, "filter_state", {
        "AIMessage": AI, "ExtractionOutputs": dict, "enable_deletes": False,
        "existing_schema_policy": False, "validator": SimpleNamespace(schemas_by_name={"Memory": memory}),
        "logger": logging.getLogger("trustcall_v6_replay")})


def native_patch(name):
    return compile_native(NATIVE, name, {"Command": SimpleNamespace,
        "cast": lambda _, value: value, "AIMessage": AI}, "_Patch")


@contextmanager
def observed(failures):
    logging.getLogger().addHandler(failures)
    try:
        with failures.observe():
            yield
    finally:
        logging.getLogger().removeHandler(failures)


class NativeReturnTests(unittest.TestCase):
    setUp = fixtures.Tests.setUp
    bounded_client = fixtures.Tests.bounded_client
    bounded_request = staticmethod(fixtures.Tests.bounded_request)
    bounded_response = staticmethod(fixtures.Tests.bounded_response)
    fixture_completion = fixtures.Tests.fixture_completion

    def test_mixed_invalid_and_valid_tool_preserves_native_output_and_one_log(self):
        function = native_filter()
        message = AI([{"id": "invalid", "name": "Memory", "args": {}},
                      {"id": "valid", "name": "Memory", "args": {"content": "preserved fact"}}])
        state = {"msg_id": "ai", "messages": [message], "attempts": 3}
        baseline = function(state)
        failures = r.NativeFailures()
        failures.session_index = 2
        with observed(failures):
            result = function(state)
        failures.check()
        self.assertEqual(result, baseline)
        self.assertIs(result["messages"][0], message)
        self.assertEqual([x.content for x in result["responses"]], ["preserved fact"])
        self.assertEqual(result["response_metadata"], [{"id": "valid"}])
        self.assertEqual(len(failures.return_events), 1)
        self.assertEqual(len(failures.warnings), 1)
        event = failures.return_events[0]
        self.assertEqual((event["line"], event["event"], event["session_index"]), (468, "native_filter_drop", 2))
        self.assertEqual(event["exception_type"], "ValidationError")
        self.assertIn("content", event["message"])
        r.verify_return_events(failures.degradation())

    def test_empty_return_counts_once_each_and_empty_ai_is_different(self):
        function, failures = native_filter(), r.NativeFailures()
        with observed(failures):
            outputs = [function({"msg_id": "missing", "messages": [], "attempts": 3}) for _ in range(2)]
            empty_ai = function({"msg_id": "ai", "messages": [AI()], "attempts": 1})
        failures.check()
        self.assertEqual(outputs, [{"messages": [], "responses": [], "attempts": 3, "response_metadata": []}] * 2)
        self.assertEqual(len(empty_ai["messages"]), 1)
        self.assertEqual([e["id"] for e in failures.return_events], [1, 2])
        self.assertEqual(failures.degradation()["native_empty_extraction_count"], 2)

    def test_patch_caught_sync_and_async_exceptions_keep_command_and_call_count(self):
        for name in ("invoke", "ainvoke"):
            with self.subTest(name=name):
                calls = []
                def fail(messages, config):
                    calls.append((messages, config))
                    raise ValueError("native bound parser failure")
                async def async_fail(messages, config):
                    await asyncio.sleep(0)
                    return fail(messages, config)
                instance = SimpleNamespace(bound=SimpleNamespace(invoke=fail, ainvoke=async_fail))
                failures = r.NativeFailures()
                with observed(failures):
                    result = native_patch(name)(instance, SimpleNamespace(messages=["original"]), {"unchanged": True})
                    if name == "ainvoke":
                        result = asyncio.run(result)
                failures.check()
                self.assertEqual(vars(result), {"goto": "__end__"})
                self.assertEqual(calls, [(["original"], {"unchanged": True})])
                self.assertEqual(failures.degradation()["native_patch_abort_count"], 1)

    def test_caught_sdk_transport_and_content_filter_stay_fatal_without_retry(self):
        for asynchronous in (False, True):
            for response in (TimeoutError("synthetic transport"), self.bounded_response("content_filter", False)):
                with self.subTest(asynchronous=asynchronous, response=type(response).__name__):
                    calls, client, _, seen = self.bounded_client(
                        f"caught_{asynchronous}_{type(response).__name__}", response, asynchronous)
                    def call(*args):
                        return client.chat.completions.create(**self.bounded_request())
                    async def async_call(*args):
                        return await call(*args)
                    instance = SimpleNamespace(bound=SimpleNamespace(invoke=call, ainvoke=async_call))
                    failures = r.NativeFailures()
                    with observed(failures):
                        result = native_patch("ainvoke" if asynchronous else "invoke")(
                            instance, SimpleNamespace(messages=[]), {})
                        if asynchronous:
                            result = asyncio.run(result)
                    self.assertEqual(result.goto, "__end__")
                    self.assertEqual(failures.degradation()["native_patch_abort_count"], 1)
                    failures.check()
                    with self.assertRaises(RuntimeError):
                        calls.check()
                    self.assertEqual(seen, [self.bounded_request()])

    def test_patch_teardown_escaping_exception_is_not_swallowed_or_counted_return(self):
        for name in ("invoke", "ainvoke"):
            with self.subTest(name=name):
                async def success(*args):
                    return AI()
                def fail_teardown(*args):
                    raise LookupError("native teardown escaped")
                instance = SimpleNamespace(bound=SimpleNamespace(invoke=lambda *args: AI(), ainvoke=success),
                                           _tear_down=fail_teardown)
                state = SimpleNamespace(messages=[], tool_call_id="id", bump_attempt=True)
                failures = r.NativeFailures()
                with observed(failures), self.assertRaisesRegex(LookupError, "escaped"):
                    result = native_patch(name)(instance, state, {})
                    if name == "ainvoke":
                        asyncio.run(result)
                self.assertEqual(failures.return_events, [])

    def test_actual_memory_manager_empty_message_indexerror_still_escapes(self):
        function, count = native_filter(), []
        def extract(*args, **kwargs):
            count.append(1)
            return function({"msg_id": "missing", "messages": [], "attempts": 3})
        invoke = compile_native(EXTRACTION, "invoke", {
            "create_extractor": lambda *a, **kw: SimpleNamespace(invoke=extract), "uuid": uuid}, "MemoryManager")
        manager = SimpleNamespace(_prepare_messages=lambda messages, steps: messages,
            _prepare_existing=lambda existing: [], model=object(), schemas=[],
            enable_inserts=True, enable_updates=True, enable_deletes=False)
        failures = r.NativeFailures()
        with observed(failures), self.assertRaises(IndexError):
            invoke(manager, {"messages": ["original session"]})
        self.assertEqual(count, [1])
        self.assertEqual(failures.degradation()["native_empty_extraction_count"], 1)

    def test_session_order_inputs_and_unchanged_native_return_are_preserved(self):
        function, inputs = native_filter(), []
        def invoke(payload):
            inputs.append(copy.deepcopy(payload))
            tools = [{"id": "good", "name": "Memory", "args": {"content": "fact"}}]
            if len(inputs) == 1:
                tools.insert(0, {"id": "bad", "name": "Memory", "args": {}})
            return function({"msg_id": "ai", "messages": [AI(tools)], "attempts": 3})["responses"]
        source = [{"session_id": str(i), "date": "public date", "turns": [
            {"role": "assistant" if i == 0 else "user", "content": f"source {i}"}]} for i in range(2)]
        failures = r.NativeFailures()
        with observed(failures):
            counts = r.ingest_sessions(SimpleNamespace(invoke=invoke), source, failures, self.root / "sessions.jsonl")
        self.assertEqual(counts, {"sessions": 2, "source_turns": 2})
        self.assertEqual(inputs, [{"messages": r.session_messages(s)} for s in source])
        journal = [json.loads(x) for x in (self.root / "sessions.jsonl").read_text().splitlines()]
        self.assertEqual([x["native_return_events"] for x in journal], [1, 0])
        self.assertEqual(failures.degradation()["native_return_affected_session_indices"], [0])
        self.assertIs(failures.degradation()["lossless_writes"], False)

    def test_only_exact_caught_exception_log_is_allowed(self):
        for file, line, function, message in (("elsewhere.py", 468, "filter_state", ValueError("x")),
                (str(NATIVE.resolve()), 467, "filter_state", ValueError("x")),
                (str(NATIVE.resolve()), 468, "other", ValueError("x")),
                (str(NATIVE.resolve()), 468, "filter_state", "unhandled plain error")):
            with self.subTest(file=file, line=line, function=function):
                failures = r.NativeFailures()
                failures.emit(logging.LogRecord("native", logging.ERROR, file, line, message, (), None, function))
                with self.assertRaises(RuntimeError):
                    failures.check()
                self.assertEqual(failures.return_events, [])

    def test_new_ledger_is_mandatory_hash_bound_and_counters_checked_after_reseal(self):
        identity, history = {"protocol_sha256": "v6-only"}, self.root / "history"
        attempt = self.fixture_completion(history, identity)
        failures = r.NativeFailures()
        failures.session_index = 4
        failures.record_return("native_filter_drop", 468, "filter_state", exception_type="ValidationError", message="content missing")
        original = failures.degradation()
        r.save_json(attempt / "native_degradation.json", original)
        with self.assertRaises(ValueError):
            r.verified(history, identity)
        r.seal(attempt, identity)
        self.assertEqual(r.verified(history, identity)["hypothesis"], "answer")
        changes = [lambda d: d.pop("native_return_events"),
            lambda d: d.update(harness_version="langmem_native_sessions_v5"),
            lambda d: d["native_return_events"][0].update(line=468.0),
            lambda d: d.update(native_return_event_count=0),
            lambda d: d.update(native_filter_drop_count=True),
            lambda d: d.update(native_return_affected_session_indices=[]),
            lambda d: d["native_return_events"][0].update(id=2),
            lambda d: d["native_return_events"][0].update(line=467),
            lambda d: d["native_return_events"][0].update(file="other.py"),
            lambda d: d["native_return_events"][0].update(session_index=True),
            lambda d: d["native_return_events"][0].pop("exception_type"),
            lambda d: d.update(lossless_writes=True)]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                corrupt = copy.deepcopy(original)
                change(corrupt)
                r.save_json(attempt / "native_degradation.json", corrupt)
                r.seal(attempt, identity)
                with self.assertRaises((KeyError, ValueError)):
                    r.verified(history, identity)

    def test_calls_session_prompt_and_qa_code_remain_unchanged(self):
        old = r.HARNESS / "official_recovery/langmem_native_sessions_v5/runner.py"
        for name in ("Calls", "session_messages", "source_hashes", "client_versions", "seal", "run"):
            self.assertEqual(ast.dump(pinned_node(old, name)), ast.dump(pinned_node(r.HERE / "runner.py", name)))
        before, after = pinned_node(old, "worker"), pinned_node(r.HERE / "runner.py", "worker")
        for node in ast.walk(after):
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    value[:] = [statement for statement in value if not (
                        isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Attribute)
                        and ast.unparse(statement.targets[0]) == "failures.session_index")]
        self.assertEqual(ast.dump(before), ast.dump(after))


if __name__ == "__main__":
    unittest.main()
