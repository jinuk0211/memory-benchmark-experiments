"""Exercise the actual frozen parser/schema and this unapplied patch in memory."""
from __future__ import annotations

import ast
import copy
from functools import partial
import hashlib
import json
from pathlib import Path
import re
import runpy
import tempfile
from types import SimpleNamespace
import unittest

from pydantic import ValidationError

from location_audit import JOURNAL_NAME, record_location_normalization

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parents[1]
SIMPLEMEM = RUN_ROOT / "source/MemoryData/methods/simplemem/source/SimpleMem"
SOURCE = SIMPLEMEM / "core/memory_builder.py"
MODEL = SIMPLEMEM / "models/memory_entry.py"
PATCH = HERE / "simplemem_location_list_v1.patch"


def apply_patch_in_memory(original: str, patch: str) -> str:
    """Apply exact unified-diff context; never write the frozen implementation."""
    source_lines = original.splitlines(keepends=True)
    output, cursor = [], 0
    in_hunk = False
    for line in patch.splitlines(keepends=True):
        if line.startswith(("--- ", "+++ ")):
            continue
        if line.startswith("@@ "):
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
            if match is None:
                raise AssertionError("Unsupported diff header")
            start = int(match.group(1)) - 1
            if start < cursor:
                raise AssertionError("Overlapping diff hunks")
            output.extend(source_lines[cursor:start])
            cursor, in_hunk = start, True
            continue
        if not in_hunk or not line.startswith((" ", "+", "-")):
            raise AssertionError("Unexpected patch content")
        if line[0] in " -":
            if cursor >= len(source_lines) or source_lines[cursor] != line[1:]:
                raise AssertionError("Frozen source does not match patch context")
            cursor += 1
        if line[0] in " +":
            output.append(line[1:])
    output.extend(source_lines[cursor:])
    return "".join(output)


def parser_node(tree: ast.Module) -> ast.FunctionDef:
    builder = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "MemoryBuilder")
    return next(node for node in builder.body
                if isinstance(node, ast.FunctionDef) and node.name == "_parse_llm_response")


def load_parser(source: str, memory_entry: type) -> object:
    node = copy.deepcopy(parser_node(ast.parse(source)))
    namespace = {"List": list, "MemoryEntry": memory_entry}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    parser = type("ActualFrozenParser", (), {
        "_parse_llm_response": namespace["_parse_llm_response"]})()
    # All fixtures are valid JSON arrays: transport/extract_json recovery is out of scope.
    parser.llm_client = SimpleNamespace(extract_json=json.loads)
    return parser


def response_for(location: object) -> str:
    return json.dumps([{"lossless_restatement": "Meeting at Seoul and Busan.",
                        "location": location, "keywords": ["meeting"],
                        "persons": [], "entities": [], "timestamp": None,
                        "topic": "meeting"}], ensure_ascii=False)


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_bytes = SOURCE.read_bytes()
        expected = (HERE / "original_memory_builder.sha256").read_text().strip()
        if hashlib.sha256(cls.original_bytes).hexdigest() != expected:
            raise AssertionError("Frozen parser changed since candidate was prepared")
        cls.original = SOURCE.read_text(encoding="utf-8-sig")
        cls.patched = apply_patch_in_memory(cls.original, PATCH.read_text(encoding="utf-8"))
        cls.memory_entry = runpy.run_path(str(MODEL))["MemoryEntry"]
        cls.model_bytes = MODEL.read_bytes()

    @classmethod
    def tearDownClass(cls) -> None:
        if SOURCE.read_bytes() != cls.original_bytes or MODEL.read_bytes() != cls.model_bytes:
            raise AssertionError("Tests must not modify frozen source/schema")

    def setUp(self) -> None:
        self.original_parser = load_parser(self.original, self.memory_entry)
        self.candidate = load_parser(self.patched, self.memory_entry)

    def test_only_parser_ast_changes(self) -> None:
        original, patched = ast.parse(self.original), ast.parse(self.patched)
        proposed_node = parser_node(patched)
        proposed_node.body = copy.deepcopy(parser_node(original).body)
        self.assertEqual(ast.dump(original), ast.dump(patched))
        # This covers extraction prompt, temperature, retry loops and all other methods.

    def test_normal_string_and_none_are_unchanged_without_hook(self) -> None:
        for location in (None, "Seoul", "", "서울, 부산"):
            with self.subTest(location=location):
                raw = response_for(location)
                before = self.original_parser._parse_llm_response(raw, [1])
                after = self.candidate._parse_llm_response(raw, [1])
                self.assertEqual([x.model_dump(exclude={"entry_id"}) for x in before],
                                 [x.model_dump(exclude={"entry_id"}) for x in after])

    def test_original_schema_rejects_location_lists(self) -> None:
        for location in ([], ["Seoul"], ["Seoul", "Busan"]):
            with self.subTest(location=location), self.assertRaises(ValidationError):
                self.original_parser._parse_llm_response(response_for(location), [1])

    def test_order_duplicates_empty_and_raw_response_are_journaled(self) -> None:
        cases = [["서울", "부산", "서울"], [], [""], ["A, B", "C"]]
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            attempt = Path(temporary)
            self.candidate.location_normalization_audit = partial(record_location_normalization, attempt)
            raw_responses = []
            for location in cases:
                raw = "  " + response_for(location) + "\n"
                raw_responses.append(raw)
                entries = self.candidate._parse_llm_response(raw, [1])
                self.assertEqual(entries[0].location, ", ".join(location))
                self.assertEqual(json.loads(raw)[0]["location"], location)
            lines = (attempt / JOURNAL_NAME).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(cases))
            for line, original, raw in zip(lines, cases, raw_responses):
                self.assertEqual(json.loads(line), {
                    "policy": "simplemem_location_list_v1",
                    "event": "location_representation_normalized", "entry_index": 0,
                    "raw_response": raw, "original_location": original,
                    "normalized_location": ", ".join(original)})

    def test_mixed_entries_record_actual_item_index(self) -> None:
        items = [json.loads(response_for(value))[0]
                 for value in ("Existing", ["Seoul", "Busan"], None)]
        events = []
        self.candidate.location_normalization_audit = events.append
        entries = self.candidate._parse_llm_response(json.dumps(items), [1, 2, 3])
        self.assertEqual([entry.location for entry in entries], ["Existing", "Seoul, Busan", None])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["entry_index"], 1)

    def test_invalid_types_keep_original_validation_failure_without_audit(self) -> None:
        events = []
        self.candidate.location_normalization_audit = events.append
        for location in (3, True, {"city": "Seoul"}, ["Seoul", 1], [None], [["Seoul"]]):
            with self.subTest(location=location):
                errors = []
                for parser in (self.original_parser, self.candidate):
                    with self.assertRaises(ValidationError) as caught:
                        parser._parse_llm_response(response_for(location), [1])
                    errors.append(caught.exception.errors())
                self.assertEqual(errors[0], errors[1])
        self.assertEqual(events, [])

    def test_missing_hook_rejects_normalization(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires an audit hook"):
            self.candidate._parse_llm_response(response_for(["Seoul"]), [1])

    def test_journal_write_failure_propagates_without_normalized_result(self) -> None:
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            absent = Path(temporary) / "absent_attempt"
            self.candidate.location_normalization_audit = partial(record_location_normalization, absent)
            with self.assertRaises(FileNotFoundError):
                self.candidate._parse_llm_response(response_for(["Seoul"]), [1])
            self.assertFalse(absent.exists())


if __name__ == "__main__":
    unittest.main()
