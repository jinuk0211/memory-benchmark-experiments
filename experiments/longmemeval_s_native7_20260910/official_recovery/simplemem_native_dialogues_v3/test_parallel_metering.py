"""Real frozen metering + HTTPX MockTransport, no network or model calls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import runner as r

ENDPOINT = "http://127.0.0.1:18083/v1"
METER_ENV = {
    "METER_RUN_ID": "synthetic-protocol",
    "METER_METHOD": "simplemem",
    "METER_LLM_PROXY_ORIGIN": ENDPOINT,
    "METER_EMBEDDING_PROXY_ORIGIN": ENDPOINT,
    "METER_TIMING_JOURNAL": "",
}


class ParallelMeteringTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        path = r.HARNESS / "source/MemoryData/utils/request_metering.py"
        name = "simplemem_v2_real_metering_test"
        spec = importlib.util.spec_from_file_location(name, path)
        self.meter = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.meter
        self.addCleanup(sys.modules.pop, name, None)
        with patch.dict(os.environ, {key: "" for key in METER_ENV}):
            spec.loader.exec_module(self.meter)
        self.addCleanup(self.meter._reset_for_tests)
        self.journal = self.root / "timing.jsonl"
        self.meter.install_request_metering({**METER_ENV, "METER_TIMING_JOURNAL": str(self.journal)})
        self.requests, self.received, self.lock = [], [], threading.Lock()

    def client(self, barrier=None, failure=None, target=ENDPOINT):
        def transport(request):
            if barrier:
                barrier.wait(timeout=10)
            with self.lock:
                self.requests.append((dict(request.headers), threading.get_ident()))
            if failure:
                raise failure
            return httpx.Response(200, json={"text": "unaltered native response"})
        http = httpx.Client(transport=httpx.MockTransport(transport))
        self.addCleanup(http.close)
        def original(messages, *args, **kwargs):
            with self.lock:
                self.received.append((messages, args, kwargs))
            return http.post(target + "/chat/completions", json={"messages": messages}).json()["text"]
        return SimpleNamespace(model=r.MODEL, client=SimpleNamespace(base_url=ENDPOINT), chat_completion=original)

    def logs(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def assert_headers(self, phase, question_id=None):
        for headers, _ in self.requests:
            self.assertEqual(headers["x-meter-run-id"], "synthetic-protocol")
            self.assertEqual(headers["x-meter-method"], "simplemem")
            self.assertEqual(headers["x-meter-phase"], phase)
            self.assertEqual(headers["x-meter-sample-id"], "synthetic-qid")
            self.assertEqual(headers.get("x-meter-question-id"), question_id)

    def parallel_phase(self, phase, question_id=None):
        path = self.root / "calls.jsonl"
        calls = r.Calls(path)
        calls.phase = phase
        client = self.client(barrier=threading.Barrier(16))
        calls.bind(client, ENDPOINT)
        messages = [{"role": "user", "content": "synthetic input"}]
        kwargs = {"temperature": 0.1, "response_format": None}
        with self.meter.meter_operation(phase, sample_id="synthetic-qid", question_id=question_id):
            snapshot = self.meter.current_operation_snapshot()
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(client.chat_completion, messages, "kept positional", **kwargs) for _ in range(16)]
                answers = [future.result(timeout=15) for future in futures]
            self.assertIs(self.meter.current_operation_snapshot(), snapshot)
        self.assertIsNone(self.meter.current_operation_snapshot())
        self.assertEqual(answers, ["unaltered native response"] * 16)
        self.assertEqual(len({thread for _, thread in self.requests}), 16)
        self.assertEqual(len(self.received), 16)
        for actual_messages, args, actual_kwargs in self.received:
            self.assertIs(actual_messages, messages)
            self.assertEqual(args, ("kept positional",))
            self.assertEqual(actual_kwargs, kwargs)
        self.assert_headers(phase, question_id)
        self.assertEqual(calls.count, 16)
        rows = self.logs(path)
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertEqual(row["phase"], phase)
            self.assertEqual(row["messages"], messages)
            self.assertEqual(row["args"], ["kept positional"])
            self.assertEqual(row["kwargs"], kwargs)
            self.assertEqual(row["response"], "unaltered native response")
            self.assertGreaterEqual(row["seconds"], 0)
            self.assertNotIn("error", row)
        timing = self.logs(self.journal)
        self.assertEqual([row["event"] for row in timing], ["begin", "end"])
        self.assertEqual(timing[-1]["status"], "success")
        self.assertEqual(timing[0]["operation_id"], timing[1]["operation_id"])

    def test_16_memory_write_threads_inherit_outer_attribution(self):
        self.parallel_phase("memory_write")

    def test_16_qa_threads_inherit_question_attribution(self):
        self.parallel_phase("qa", "synthetic-qid")

    def test_v1_nested_scope_pattern_reproduces_failure_before_http(self):
        client = self.client()
        def old_observed():
            # Exact rejected ownership pattern from v1 Calls.bind.
            with self.meter.meter_operation("memory_write", sample_id="synthetic-qid"):
                return client.chat_completion([])
        with self.meter.meter_operation("memory_write", sample_id="synthetic-qid"):
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(old_observed) for _ in range(16)]
                for future in futures:
                    with self.assertRaisesRegex(RuntimeError, "one serial outer operation per process"):
                        future.result(timeout=5)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.received, [])
        self.assertIsNone(self.meter.current_operation_snapshot())

    def test_original_transport_error_is_logged_and_propagated_with_scope_cleanup(self):
        failure = httpx.ConnectError("synthetic transport failure")
        client = self.client(failure=failure)
        path = self.root / "failed_calls.jsonl"
        calls = r.Calls(path)
        calls.bind(client, ENDPOINT)
        with self.assertRaises(httpx.ConnectError) as caught:
            with self.meter.meter_operation("memory_write", sample_id="synthetic-qid"):
                client.chat_completion([], temperature=0.1)
        self.assertIs(caught.exception, failure)
        self.assertEqual(calls.count, 1)
        row = self.logs(path)[0]
        self.assertEqual(row["error"], "ConnectError: synthetic transport failure")
        self.assertNotIn("response", row)
        self.assert_headers("memory_write")
        self.assertIsNone(self.meter.current_operation_snapshot())
        self.assertEqual(self.logs(self.journal)[-1]["status"], "error")

    def test_headers_do_not_escape_allowlisted_origin(self):
        client = self.client(target="https://outside.invalid/v1")
        calls = r.Calls(self.root / "calls.jsonl")
        calls.bind(client, ENDPOINT)
        with self.meter.meter_operation("qa", sample_id="synthetic-qid", question_id="synthetic-qid"):
            client.chat_completion([])
        self.assertFalse(any(name.startswith("x-meter-") for name in self.requests[0][0]))

    def test_phase_switch_does_not_reuse_memory_write_snapshot(self):
        client = self.client()
        calls = r.Calls(self.root / "calls.jsonl")
        calls.bind(client, ENDPOINT)
        for phase, qid in (("memory_write", None), ("qa", "synthetic-qid")):
            calls.phase = phase
            with self.meter.meter_operation(phase, sample_id="synthetic-qid", question_id=qid):
                client.chat_completion([])
        self.assertEqual([headers["x-meter-phase"] for headers, _ in self.requests], ["memory_write", "qa"])
        self.assertNotIn("x-meter-question-id", self.requests[0][0])
        self.assertEqual(self.requests[1][0]["x-meter-question-id"], "synthetic-qid")
        self.assertIsNone(self.meter.current_operation_snapshot())

    def test_real_native_550_turn_bulk_works_inside_actual_outer_meter(self):
        # Reuse the existing unchanged-native fixture, now with the real outer scope.
        from test_runner import NativeTests
        case = NativeTests("test_original_bulk_550_turns_has_15_native_windows_and_no_budget_override")
        try:
            case.setUp()
            with self.meter.meter_operation("memory_write", sample_id="synthetic-qid"):
                case.test_original_bulk_550_turns_has_15_native_windows_and_no_budget_override()
            self.assertEqual(case.calls.count, 15)
            self.assertIsNone(self.meter.current_operation_snapshot())
        finally:
            case.doCleanups()


if __name__ == "__main__":
    unittest.main()
