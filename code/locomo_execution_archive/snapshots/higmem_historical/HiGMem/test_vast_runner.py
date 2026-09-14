import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from run_vast import Journal, MeteredLLM, read_rows


class RunnerTests(unittest.TestCase):
    def test_concurrent_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            journal = Journal(Path(folder) / "usage.jsonl")
            with ThreadPoolExecutor(8) as pool:
                list(pool.map(lambda index: journal.append({"id": index}), range(200)))
            self.assertEqual({r["id"] for r in read_rows(journal.path)}, set(range(200)))

    def test_partial_tail_is_backed_up_and_recovered(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "log"
            path.write_bytes(b'{"ok":1}\n{"partial":')
            self.assertEqual(read_rows(path), [{"ok": 1}])
            Journal(path).append({"ok": 2})
            self.assertEqual(len(read_rows(path)), 2)
            self.assertEqual(len(list(Path(folder).glob("log.partial-*"))), 1)

    def test_corrupt_middle_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "log"
            path.write_bytes(b'bad\n{"ok":1}\n')
            with self.assertRaises(json.JSONDecodeError):
                read_rows(path)

    def test_complete_tail_without_newline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "log"
            path.write_bytes(b'{"ok":1}')
            read_rows(path)
            Journal(path).append({"ok": 2})
            self.assertEqual(len(read_rows(path)), 2)

    def test_usage_recorded_even_for_truncated_response(self):
        with tempfile.TemporaryDirectory() as folder:
            journal = Journal(Path(folder) / "usage")
            controller = object.__new__(MeteredLLM)
            controller.model, controller.journal = "test", journal
            controller.sample, controller.phase, controller.question = "s", "qa", 1
            controller.client = Mock()
            controller.client.chat.completions.create.return_value = SimpleNamespace(
                id="r", usage=SimpleNamespace(model_dump=lambda: dict(prompt_tokens=7, completion_tokens=3, total_tokens=10)),
                choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content='{}'))])
            with self.assertRaises(RuntimeError):
                controller.get_completion("test", {})
            self.assertEqual(read_rows(journal.path)[0]["usage"]["total_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
