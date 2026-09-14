"""Check HiGMem's source and context boundaries without model initialization."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from native_higmem import retrieve_candidates, source_turns


class HiGMemBoundaryTests(unittest.TestCase):
    def test_source_order_roles_and_complete_text(self):
        source = [dict(session_id="s1", date="2023/05/20 (Sat) 02:21",
                       turns=[dict(role="user", content="a\n\nb"),
                              dict(role="assistant", content="reply")]),
                  dict(session_id="s2", date="2023/05/21 (Sun) 02:21",
                       turns=[dict(role="user", content="later")])]
        rows = source_turns(source)
        self.assertEqual([r[0] for r in rows], ["session0_turn0", "session0_turn1", "session1_turn0"])
        self.assertEqual([r[2] for r in rows], ["user", "assistant", "user"])
        self.assertEqual(rows[0][1], "a\n\nb")
        self.assertEqual(rows[2][3], source[1]["date"])

    def test_native_chronological_units_preserve_internal_blank_lines(self):
        early = SimpleNamespace(id="a", timestamp="2023/05/20", speaker="user",
                                content="a\n\nb", context="early context")
        late = SimpleNamespace(id="b", timestamp="2023/05/21", speaker="assistant",
                               content="c", context="late context")
        # Extract exact native turn formatting from its AST so this test does not
        # merely duplicate the boundary implementation's formatting expression.
        import ast
        path = Path(__file__).resolve().parents[2] / "HiGMem/fphm_core.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        expressions = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "turn_string" for t in n.targets)]
        expr = next(n for n in expressions if "--- Turn Start ---" in ast.unparse(n))
        code = compile(ast.Expression(body=expr), str(path), "eval")
        expected = [eval(code, {"t": turn}) for turn in (early, late)]
        trace = dict(mode="full", relevant_turn_ids=["b", "a"])
        system = SimpleNamespace(
            turn_notes={"a": early, "b": late},
            _get_llm_json_response=Mock(return_value=dict(keyword_query="native rewritten",
                                                        profile_retrieval_keys=[])),
            retrieve_for_query=Mock(return_value=("\n\n".join(expected), trace)),
        )
        with patch.dict(sys.modules, {"prompts": SimpleNamespace(
                QUERY_REWRITING_PROMPT="Rewrite {original_query}")}):
            parts, returned_trace = retrieve_candidates(system, "what?", "date")
        self.assertEqual(parts, expected)
        self.assertIs(returned_trace, trace)
        system.retrieve_for_query.assert_called_once_with(
            "Question date: date\nQuestion: what?", "native rewritten", [], 0, 10, 10,
            return_trace=True)


if __name__ == "__main__":
    unittest.main()