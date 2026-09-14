import gzip
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mem0_v2_tested", HERE / "runner.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.h = r.NativeObservations(self.root)
        self.h.begin_session(0)
        self.patch = patch.dict(sys.modules, {"mem0.memory.main": types.SimpleNamespace(remove_code_blocks=lambda text: text)})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def response(self, content, phase="update", finish="stop"):
        raw = {"choices": [{"message": {"content": content}, "finish_reason": finish}], "usage": {"completion_tokens": 3}}
        seen = []
        result = types.SimpleNamespace(model_dump=lambda **kwargs: raw)
        def create(*args, **kwargs):
            seen.append((args, kwargs))
            return result
        client = types.SimpleNamespace(create=create)
        self.h.session_calls = 0 if phase == "facts" else 1
        self.h.observe(client)
        kwargs = {"model": "Qwen/Qwen3.5-9B", "max_tokens": 2000, "top_p": .1, "response_format": {"type": "json_object"}}
        self.assertIs(client.create(**kwargs), result)
        self.assertEqual(seen, [((), kwargs)])
        return raw

    def emit(self, line, native=True):
        path = HERE / "vendor/mem0/memory/main.py" if native else HERE / "unrelated.py"
        record = logging.LogRecord("root", logging.ERROR, str(path), line, "native test diagnostic", (), None, "_add_to_vector_store")
        self.h.emit(record)

    def test_malformed_update_counts_one_response_and_pair(self):
        self.response('{"memory":', finish="length")
        self.emit(278)
        self.emit(331)
        self.assertEqual(self.h.records, [])
        self.assertEqual(self.h.counts(), {"native_warning_logs": 2, "native_affected_responses": 1, "native_affected_sessions": 1, "native_action_drops": 0})
        self.h.save()
        self.assertEqual(json.loads((self.root / "native_degradation.json").read_text())["native_affected_sessions"], 1)

    def test_fact_missing_key_is_native_fallback(self):
        self.response("{}", phase="facts")
        self.emit(233)
        self.assertEqual(len(self.h.warnings), 1)
        self.assertFalse(self.h.records)

    def test_update_array_is_native_fallback(self):
        self.response("[]")
        self.emit(331)
        self.assertEqual(len(self.h.warnings), 1)
        self.assertFalse(self.h.records)

    def test_storage_and_transport_log_branches_remain_fatal(self):
        self.response('{"memory": []}')
        for line in (271, 329):
            self.emit(line)
        self.assertEqual(len(self.h.records), 2)
        self.assertFalse(self.h.warnings)

    def test_valid_response_cannot_justify_parse_warning(self):
        self.response('{"memory": []}')
        self.emit(278)
        self.emit(331)
        self.assertEqual(len(self.h.records), 2)

    def test_other_origin_and_prior_pair_cannot_be_whitelisted(self):
        self.response("[]")
        self.emit(331, native=False)
        self.h.begin_session(1)
        self.emit(331)
        self.assertEqual(len(self.h.records), 2)
        self.assertFalse(self.h.warnings)

    def test_sdk_exception_cannot_be_hidden_by_native_fallback(self):
        def create(**kwargs):
            raise TimeoutError("mock only")
        client = types.SimpleNamespace(create=create)
        self.h.observe(client)
        with self.assertRaises(TimeoutError):
            client.create()
        self.emit(278)
        self.emit(331)
        self.assertEqual(len(self.h.records), 3)
        self.assertFalse(self.h.warnings)

    def test_raw_sdk_response_retained_without_request_changes(self):
        raw = self.response('{"memory": []}')
        with gzip.open(self.root / "memory_responses.jsonl.gz", "rt", encoding="utf-8") as stream:
            evidence = json.loads(stream.read())
        self.assertEqual(evidence["response"], raw)
        self.assertEqual(evidence["phase"], "update")


if __name__ == "__main__":
    unittest.main()
