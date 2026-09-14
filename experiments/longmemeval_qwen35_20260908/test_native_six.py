"""Boundary regressions without loading baseline models or calling an API."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from native_six import a_mem_candidates, ingest_chunk, retrieve_candidates, source_chunks


ROOT = Path(__file__).resolve().parents[2]


def native_method(path: Path, class_name: str, method_name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


class NativeBoundaryTests(unittest.TestCase):
    def test_complete_turn_chunking(self):
        long_text = "a\nb\n" * 3000
        sessions = [dict(session_id="session1", date="2023/05/20 (Sat) 02:21",
                         turns=[dict(role="user", content=long_text),
                                dict(role="assistant", content="retained assistant")])]
        chunks = list(source_chunks(sessions))
        self.assertEqual(len(chunks), 2)
        self.assertIn(long_text, chunks[0][1])
        self.assertIn("assistant: retained assistant", chunks[1][1])
        self.assertTrue(all("Session session1" in text for _, text in chunks))

    def test_native_a_mem_keeps_linked_neighbors_and_duplicates(self):
        root = SimpleNamespace(content="root", links=["neighbor"])
        neighbor = SimpleNamespace(content="neighbor\n\ninternal paragraphs", links=[])
        class Adapter:
            def _format_memory_note(self, note):
                return f"NOTE: {note.content}"
        namespace = {"parse_locomo_source_ids": lambda _: [],
                     "dedupe_preserve_order": lambda values: list(dict.fromkeys(values))}
        Adapter.retrieve_with_source_groups = native_method(
            ROOT / "MemoryData/methods/a_mem/a_mem_adapter.py",
            "AMemAdapter", "retrieve_with_source_groups", namespace)
        adapter = Adapter()
        adapter.retrieve_k = 2
        adapter.memory_system = SimpleNamespace(
            memories={"root": root, "neighbor": neighbor},
            retriever=SimpleNamespace(search=lambda query, k: [0, 1]),
        )
        before = adapter.retrieve_with_source_groups("q")[0]
        parts = a_mem_candidates(adapter, "q")
        self.assertEqual(parts, ["NOTE: root", "NOTE: neighbor\n\ninternal paragraphs",
                                 "NOTE: neighbor\n\ninternal paragraphs"])
        self.assertEqual("\n".join(parts), before)
        self.assertNotIn("_format_memory_note", vars(adapter))

    def test_a_mem_restores_formatter_on_error(self):
        def formatter(note):
            return str(note)
        adapter = SimpleNamespace(_format_memory_note=formatter,
                                  retrieve_with_source_groups=Mock(side_effect=RuntimeError("fail")))
        with self.assertRaises(RuntimeError):
            a_mem_candidates(adapter, "q")
        self.assertIs(adapter._format_memory_note, formatter)

    def test_ingestion_uses_source_time_and_no_question(self):
        source = "Session s (2023/05/20 (Sat) 02:21)\nuser: original"
        agent = SimpleNamespace(simplemem=Mock(), lightmem=Mock(), a_mem=Mock())
        for method in ("simplemem", "lightmem_direct", "a_mem"):
            ingest_chunk(agent, method, "2023/05/20 (Sat) 02:21", source, "qid")
        agent.simplemem.add_chunk.assert_called_once_with(
            source, timestamp="2023/05/20 (Sat) 02:21")
        agent.lightmem.add_chunk.assert_called_once_with(
            source, timestamp="2023/05/20 (Sat) 02:21")
        agent.a_mem.add_chunk.assert_called_once_with(source, timestamp="202305200221")

    def test_mem0_retrieves_using_same_history_namespace(self):
        agent = SimpleNamespace(
            memory=SimpleNamespace(search=Mock(return_value=[{"memory": "fact"}])),
            sub_dataset="longmemeval_s_official", retrieve_num=5,
            _normalize_mem0_search_results=lambda result: result,
        )
        result = retrieve_candidates(agent, "mem0", "where?", "date", "qid1")
        self.assertEqual(result, ["fact"])
        agent.memory.search.assert_called_once_with(
            query="Question date: date\nQuestion: where?",
            user_id="context_qid1_longmemeval_s_official", limit=5)


if __name__ == "__main__":
    unittest.main()