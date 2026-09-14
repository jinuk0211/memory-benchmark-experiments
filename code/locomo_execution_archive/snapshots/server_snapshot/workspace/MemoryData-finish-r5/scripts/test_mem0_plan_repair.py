import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods/mem0/source"))
from mem0.memory.main import Memory
from mem0.memory.action_plan_repair import plan_errors


class PlanRepairTests(unittest.TestCase):
    def invoke(self, plans, persistent=False):
        memory = Memory.__new__(Memory)
        memory.custom_fact_extraction_prompt = None
        memory.custom_update_memory_prompt = None
        memory.api_version = "v1.1"
        memory.embedding_model = Mock()
        memory.embedding_model.embed.return_value = [1.0]
        memory.vector_store = Mock()
        memory.vector_store.search.return_value = []
        memory._create_memory = Mock(return_value="new-memory")
        memory._update_memory = Mock()
        memory._delete_memory = Mock()
        calls = []
        def generate(**kwargs):
            # Every response, including corrections, precedes all mutations.
            memory._create_memory.assert_not_called()
            memory._update_memory.assert_not_called()
            memory._delete_memory.assert_not_called()
            calls.append(kwargs)
            if len(calls) == 1:
                return json.dumps({"facts": ["A fact"]})
            return json.dumps(plans[min(len(calls) - 2, len(plans) - 1)])
        memory.llm = Mock(generate_response=generate)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "BASELINE_STRICT_COMPARISON": "1", "MEM0_TELEMETRY": "false",
                "METER_AUXILIARY_JOURNAL": str(Path(directory) / "auxiliary.jsonl")}), \
                patch("mem0.memory.main.capture_event"), contextlib.redirect_stdout(io.StringIO()):
            if persistent:
                with self.assertRaisesRegex(ValueError, "after two corrections"):
                    memory._add_to_vector_store([{"role": "user", "content": "A fact"}], {}, {}, True)
                memory._create_memory.assert_not_called()
                result = None
            else:
                result = memory._add_to_vector_store([{"role": "user", "content": "A fact"}], {}, {}, True)
            journal = Path(directory) / "mem0_action_repairs.jsonl"
            events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        return result, calls, events, memory

    def test_valid_initial_plan_has_no_retry(self):
        plan = {"memory": [{"id": "0", "text": "A fact", "event": "ADD"}]}
        result, calls, events, memory = self.invoke([plan])
        self.assertEqual(len(calls), 2)
        self.assertEqual(events, [])
        self.assertEqual(result[0]["memory"], "A fact")
        memory._create_memory.assert_called_once()

    def test_missing_event_repaired_before_any_mutation(self):
        invalid = {"memory": [{"id": "0", "text": "A fact", "event": "ADD"},
                               {"id": "1", "text": "Other fact"}]}
        corrected = {"memory": [invalid["memory"][0], dict(invalid["memory"][1], event="ADD")]}
        result, calls, events, memory = self.invoke([invalid, corrected])
        self.assertEqual(len(result), 2)
        self.assertEqual(memory._create_memory.call_count, 2)
        self.assertEqual(len(calls), 3)
        self.assertIn("memory[1].event", calls[-1]["messages"][-1]["content"])
        self.assertEqual([e["accepted"] for e in events], [False, True])
        self.assertEqual(events[0]["plan"], invalid)

    def test_persistent_invalid_plan_never_applied(self):
        invalid = {"memory": [{"id": "0", "text": "A fact", "event": "ADD"}, {"text": "Missing event"}]}
        _, calls, events, _ = self.invoke([invalid], persistent=True)
        self.assertEqual(len(calls), 4)
        self.assertEqual([e["accepted"] for e in events], [False, False, False])

    def test_unknown_and_deleted_ids_rejected(self):
        self.assertTrue(plan_errors({"memory": [{"event": "UPDATE", "text": "A fact", "id": "missing"}]}, {"0": "uuid"}))
        self.assertTrue(plan_errors({"memory": [{"event": "DELETE", "text": "A fact", "id": "0"},
            {"event": "UPDATE", "text": "A fact", "id": "0"}]}, {"0": "uuid"}))

    def test_malformed_event_is_repairable(self):
        self.assertTrue(plan_errors({"memory": [{"event": [], "text": "A fact"}]}, {}))


if __name__ == "__main__":
    unittest.main()
