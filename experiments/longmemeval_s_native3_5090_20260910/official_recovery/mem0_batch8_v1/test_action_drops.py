"""Exact pinned Mem0 AST replay; synthetic SDK/storage only, no network/model calls."""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import gzip
import importlib.util
import json
import logging
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_support import completed_fixture

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mem0_action_candidate", HERE / "runner.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
NATIVE = HERE / "vendor/mem0/memory/main.py"


def pinned_function(path, name, parent=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if parent:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent)
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def compiled(path, name, namespace, parent=None):
    node = pinned_function(path, name, parent)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path.resolve()), "exec"), namespace)
    return namespace[name]


STRIP = compiled(HERE / "vendor/mem0/memory/utils.py", "remove_code_blocks", {"re": re})
ADD = {"text": "synthetic retained fact", "event": "ADD"}


@contextmanager
def handler(observer):
    logging.getLogger().addHandler(observer)
    try:
        yield
    finally:
        logging.getLogger().removeHandler(observer)


class Replay:
    def __init__(self, attempt, payload, *, observe=True, failure=None, mutate_response=False):
        self.attempt = attempt
        attempt.mkdir(parents=True)
        self.observer = r.NativeObservations(attempt)
        self.observer.begin_batch(0)
        self.requests, self.effects = [], []
        responses = iter([{"facts": ["fact"]}, payload])
        def sdk(**kwargs):
            self.requests.append(copy.deepcopy(kwargs))
            if failure == "transport" and len(self.requests) == 2:
                raise TimeoutError("synthetic transport")
            content = json.dumps(next(responses))
            raw = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
            return SimpleNamespace(model_dump=lambda **kw: copy.deepcopy(raw),
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        client = SimpleNamespace(create=sdk)
        if observe:
            self.observer.observe(client)
        def generate(**kwargs):
            content = client.create(**kwargs).choices[0].message.content
            if mutate_response and len(self.requests) == 2:
                return json.dumps({"memory": [{"text": "fact", "event": "UPDATE", "id": "12"}]})
            return content
        def create(**kwargs):
            self.effects.append(("ADD", kwargs["data"]))
            if failure == "create":
                raise KeyError("12")
            return f"created-{len(self.effects)}"
        def update(**kwargs):
            self.effects.append(("UPDATE", kwargs["memory_id"]))
            if failure == "update":
                raise KeyError("12")
        def delete(**kwargs):
            self.effects.append(("DELETE", kwargs["memory_id"]))
            if failure == "delete":
                raise KeyError("12")
        self.memory = SimpleNamespace(config=SimpleNamespace(
            custom_fact_extraction_prompt=None, custom_update_memory_prompt=None),
            llm=SimpleNamespace(generate_response=generate),
            embedding_model=SimpleNamespace(embed=lambda *a, **kw: [0]),
            vector_store=SimpleNamespace(search=lambda **kw: [SimpleNamespace(id="real-uuid", payload={"data": "old"})]),
            _create_memory=create, _update_memory=update, _delete_memory=delete, api_version="v1.1")
        self.native = compiled(NATIVE, "_add_to_vector_store", {
            "json": json, "logging": logging, "remove_code_blocks": STRIP,
            "parse_messages": lambda value: value,
            "get_fact_retrieval_messages": lambda value: ("native system", "native source"),
            "get_update_memory_messages": lambda *args: "native update", "capture_event": lambda *a, **kw: None}, "Memory")
        self.observe = observe

    def run(self):
        with patch.dict(sys.modules, {"mem0.memory.main": SimpleNamespace(remove_code_blocks=STRIP)}), \
                handler(self.observer if self.observe else logging.NullHandler()):
            self.result = self.native(self.memory, [{"role": "assistant", "content": "source"}], {}, {}, True)
        if self.observe:
            self.observer.save()
        return self.result


class ActionDropTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def replay(self, name, payload, **kwargs):
        replay = Replay(self.root / name, payload, **kwargs)
        replay.run()
        return replay

    def test_missing_unmapped_numeric_and_unhashable_ids_keep_neighbor_effects(self):
        cases = [("UPDATE", {}, "KeyError"), ("UPDATE", {"id": "12"}, "KeyError"),
                 ("UPDATE", {"id": 0}, "KeyError"), ("UPDATE", {"id": []}, "TypeError"),
                 ("UPDATE", {"id": {}}, "TypeError"), ("DELETE", {}, "KeyError"),
                 ("DELETE", {"id": "12"}, "KeyError"), ("DELETE", {"id": None}, "KeyError"),
                 ("DELETE", {"id": []}, "TypeError")]
        for index, (event, fields, error_type) in enumerate(cases):
            with self.subTest(event=event, fields=fields):
                payload = {"memory": [ADD, {"event": event, "text": "bad action", **fields}, ADD]}
                baseline = self.replay(f"baseline{index}", payload, observe=False)
                replay = self.replay(f"observed{index}", payload)
                self.assertEqual(replay.result, baseline.result)
                self.assertEqual(replay.effects, baseline.effects)
                self.assertEqual(replay.requests, baseline.requests)
                self.assertEqual(len(replay.requests), 2)
                self.assertEqual(len(replay.result), 2)
                self.assertEqual(replay.observer.records, [])
                warning, = replay.observer.warnings
                self.assertEqual(warning["action_evidence"]["exception_type"], error_type)
                self.assertEqual(warning["action_evidence"]["origin_line"], 304 if event == "UPDATE" else 318)
                self.assertEqual(warning["action_evidence"]["mapping_keys"], ["0"])
                r.verify_degradation(replay.attempt)

    def test_nondict_entries_and_string_or_mapping_memory_follow_native_iteration(self):
        for index, container in enumerate(([None, "bad", 5, True, []], "bad", {"bad": 1})):
            with self.subTest(container=container):
                replay = self.replay(str(index), {"memory": container})
                self.assertEqual(replay.observer.records, [])
                self.assertEqual(len(replay.observer.warnings), len(container))
                self.assertTrue(all(w["action_evidence"]["origin_line"] == 286 for w in replay.observer.warnings))
                self.assertEqual(replay.result, [])
                r.verify_degradation(replay.attempt)

    def test_same_handler_in_native_worker_thread_has_authenticated_exception(self):
        replay = Replay(self.root / "thread", {"memory": [{"event": "UPDATE", "text": "fact", "id": "12"}, ADD]})
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(replay.run).result()
        self.assertEqual(len(result), 1)
        self.assertEqual(replay.observer.counts()["native_action_drops"], 1)
        self.assertEqual(replay.observer.records, [])

    def test_deep_same_keyerror_storage_is_fatal_despite_native_partial_writes(self):
        for operation in ("create", "update", "delete"):
            with self.subTest(operation=operation):
                entry = ADD if operation == "create" else {"id": "0", "event": operation.upper(), "text": "fact"}
                payload = {"memory": [entry, ADD]}
                baseline = self.replay(operation + "baseline", payload, failure=operation, observe=False)
                replay = self.replay(operation, payload, failure=operation)
                self.assertEqual(replay.result, baseline.result)
                self.assertEqual(replay.effects, baseline.effects)
                self.assertEqual(replay.requests, baseline.requests)
                self.assertTrue(replay.observer.records)
                self.assertEqual(replay.observer.warnings, [])
                with self.assertRaisesRegex(ValueError, "fatal"):
                    r.verify_degradation(replay.attempt)

    def test_sdk_transport_remains_fatal_after_native_catches_and_no_retry(self):
        replay = self.replay("transport", {"memory": []}, failure="transport")
        self.assertEqual(len(replay.requests), 2)
        self.assertEqual(replay.result, [])
        self.assertTrue(any("transport" in record for record in replay.observer.records))
        self.assertEqual(replay.observer.warnings, [])

    def test_forged_outside_native_log_and_mismatched_saved_response_are_fatal(self):
        replay = self.replay("good", {"memory": []})
        try:
            raise KeyError("12")
        except KeyError:
            with patch.dict(sys.modules, {"mem0.memory.main": SimpleNamespace(remove_code_blocks=STRIP)}):
                replay.observer.emit(logging.LogRecord("root", logging.ERROR, str(NATIVE), 329,
                    "Error in new_memories_with_actions: '12'", (), None, "_add_to_vector_store"))
        self.assertTrue(replay.observer.records)
        self.assertEqual(replay.observer.warnings, [])
        mismatched = self.replay("mismatch", {"memory": []}, mutate_response=True)
        self.assertEqual(mismatched.observer.warnings, [])
        self.assertTrue(mismatched.observer.records)

    def test_native_silent_noops_are_not_invented_drops(self):
        payload = {"memory": [{"event": "UPDATE", "id": "12", "text": ""},
                               {"event": "UNKNOWN", "text": "fact"}, {"event": "NONE", "text": "fact"}]}
        replay = self.replay("noops", payload)
        self.assertEqual(replay.result, [])
        self.assertEqual(replay.observer.warnings, [])
        self.assertEqual(replay.observer.records, [])
        r.verify_degradation(replay.attempt)

    def seal_replay(self):
        replay = self.replay("run/histories/qid/attempt_0001", {"memory": [ADD, {"event": "UPDATE", "text": "bad", "id": "12"}, ADD]})
        _, identity = completed_fixture(r, replay.attempt, keep_responses=True)
        r.finalize_attempt(replay.attempt, identity)
        return replay, identity

    def test_sealed_accounting_and_provenance_corruption_remain_invalid_after_reseal(self):
        replay, identity = self.seal_replay()
        attempt = replay.attempt
        self.assertEqual(r.verified_prediction(attempt.parent, identity)["hypothesis"], "answer")
        original = r.harness.read_json(attempt / "native_degradation.json")
        changes = [lambda d: d.update(implementation="mem0_native_pairs_v3"),
            lambda d: d.update(native_action_drops=0), lambda d: d.update(native_affected_responses=True),
            lambda d: d.update(fatal_errors=["real failure"]), lambda d: d.update(lossless_writes=True),
            lambda d: d["warnings"][0].update(warning_id=2), lambda d: d["warnings"][0].update(native_path="other.py"),
            lambda d: d["warnings"][0].update(native_line=331), lambda d: d["warnings"][0].update(batch_index=1),
            lambda d: d["warnings"][0].update(finish_reason="length"),
            lambda d: d["warnings"][0].update(call=999), lambda d: d["warnings"][0].update(response_sha256="0" * 64),
            lambda d: d["warnings"][0]["action_evidence"].update(native_traceback_frames=2),
            lambda d: d["warnings"][0]["action_evidence"].update(origin_line=290),
            lambda d: d["warnings"][0]["action_evidence"].update(exception_type="OSError"),
            lambda d: d["warnings"][0]["action_evidence"].update(response_entry=ADD),
            lambda d: d["warnings"][0]["action_evidence"].update(mapping_keys=[str(i) for i in range(13)])]
        receipt = r.harness.read_json(attempt / "completion.json")
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                corrupted = copy.deepcopy(original)
                change(corrupted)
                r.harness.save_json(attempt / "native_degradation.json", corrupted)
                r.harness.save_json(attempt / "completion.json", {**receipt, "files_sha256": r.attempt_files(attempt)})
                with self.assertRaises((KeyError, ValueError)):
                    r.verified_prediction(attempt.parent, identity)

    def test_original_response_evidence_is_required_and_hash_verified(self):
        replay = self.replay("evidence", {"memory": [{"id": "12", "event": "UPDATE", "text": "bad"}]})
        path = replay.attempt / "memory_responses.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            events = [json.loads(line) for line in stream]
        for records in (events[:1], [events[0], {**events[1], "response_sha256": "0" * 64}], [events[0], events[0], events[1]]):
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for event in records:
                    stream.write(json.dumps(event) + "\n")
            with self.assertRaises(ValueError):
                r.verify_degradation(replay.attempt)

    def test_native_prompts_telemetry_qa_configuration_unchanged(self):
        old = r.ROOT / "official_recovery/mem0_native_pairs_v4/runner.py"
        for name in ("memory_config", "dependency_versions", "disable_telemetry_threads", "create_agent", "validate_runtime"):
            self.assertEqual(ast.dump(pinned_function(old, name)), ast.dump(pinned_function(HERE / "runner.py", name)))
        old_observe = ast.dump(pinned_function(old, "observe", "NativeObservations")).replace("pair", "batch")
        self.assertEqual(old_observe, ast.dump(pinned_function(HERE / "runner.py", "observe", "NativeObservations")))
        r.verify_official_source()
        for content in ('{"memory": []}', '```json\n{"memory": []}\n```', ' [] ', 'null', 'invalid'):
            try:
                expected = json.loads(STRIP(content)), False
            except (ValueError, TypeError, AttributeError):
                expected = None, True
            self.assertEqual(r.parsed_response({"choices": [{"message": {"content": content}}]}), expected)


if __name__ == "__main__":
    unittest.main()
