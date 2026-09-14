"""Exercise the real v15 main body without loading a GPU model."""
import argparse
import ast
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SOURCE = Path(__file__).parents[1] / "scripts" / "run_system_v15.py"


def run_main(monkeypatch, tmp_path, comparison, long_raw=False):
    trace = []

    class Engine:
        def __init__(self, cfg):
            self.cfg = cfg
            self.tok = lambda text, **kwargs: SimpleNamespace(input_ids=text.split())
            self.context = {}

        def set_context(self, **kwargs):
            self.context = kwargs
            trace.append(("context", kwargs))

        def generate(self, system, prompts, max_tokens, **kwargs):
            trace.append(("generate", system, len(prompts), max_tokens, self.context))
            return ["answer"] * len(prompts)

    class Embedding:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 3))

    class Index:
        def __init__(self, emb, texts):
            self.texts = texts

        def search(self, query, k):
            return list(range(min(k, len(self.texts)))), np.ones(min(k, len(self.texts)))

    def configure(cfg, output):
        trace.append(("configure", str(output)))
        cfg.comparison = True
        return cfg

    def embedding(cfg):
        trace.append(("embedding", cfg.comparison))
        return Embedding()

    qa = lambda: SimpleNamespace(question="Question?", answer="answer", category=1, sessions=lambda: [1])

    def memory(*args, **kwargs):
        units = [{"text": "fact", "session": 1, "turns": [1], "tok": 1}]
        text = "Speaker: " + ("word " * 39000 if long_raw else "sentence.")
        raw = [{"text": text, "session": 1, "turn": 1, "tok": 2}]
        stats = [{"deferred": 0}]
        return units, raw, {1: 0.2}, stats, {}

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    module = ast.Module(body=[main], type_ignores=[])
    namespace = {
        "argparse": argparse, "os": os, "np": np, "pd": pd, "time": time,
        "Config": lambda: SimpleNamespace(locomo_path="unused", embed_model="native"),
        "comparison_config": configure, "load_embedding": embedding,
        "load_locomo": lambda path: {"conv-test": {"sessions": [], "qa": [qa(), qa()]}},
        "Engine": Engine, "extract_facts": lambda *args, **kwargs: [],
        "build_memory": memory, "Index": Index, "split_sentences": lambda text: [text],
        "REFORM_SYS": "reform", "READER_SYS": "reader",
        "reader_prompt": lambda context, question: f"{context}\n{question}",
        "f1n": lambda pred, gold: 1.0, "lenient": lambda pred, gold: 1,
    }
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Embedding))
    argv = [str(SOURCE), "--out", str(tmp_path), "--budgets", "400,1600,4000"]
    if comparison:
        argv.append("--comparison")
    monkeypatch.setattr(sys, "argv", argv)
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102 - Checked-in main AST only.
    namespace["main"]()
    return pd.read_csv(tmp_path / "items.csv"), trace


def test_comparison_preserves_native_policies_and_caps(monkeypatch, tmp_path):
    rows, trace = run_main(monkeypatch, tmp_path, True)
    assert len(rows) == 74
    assert len(rows[rows.config == "full"]) == 30
    assert len(rows[rows.config == "no_adaptive"]) == 22
    assert len(rows[rows.config == "no_residual"]) == 22
    assert [x[3] for x in trace if x[:2] == ("generate", "reader")] == [32, 32, 32]
    assert [x[3] for x in trace if x[:2] == ("generate", "reform")] == [60, 60, 60]
    assert ("embedding", True) in trace
    assert list(rows.groupby(["config", "question_ordinal"]).size()) == [15, 15, 11, 11, 11, 11]
    assert {x[1]["phase"] for x in trace if x[0] == "context"} >= {
        "fact_extraction", "memory_build", "retrieval", "query_reform", "qa"
    }


def test_default_does_not_enable_comparison(monkeypatch, tmp_path):
    rows, trace = run_main(monkeypatch, tmp_path, False)
    assert len(rows) == 74
    assert not [x for x in trace if x[0] in {"configure", "embedding"}]


def test_native_qa_identity_keeps_duplicate_questions_distinct(monkeypatch, tmp_path):
    rows, _ = run_main(monkeypatch, tmp_path, True)
    certified = rows[(rows.config == "full") & (rows.policy == "certified")]
    assert certified.question.nunique() == 1
    assert list(certified.question_ordinal) == [0, 1]


def test_comparison_keeps_long_raw_and_default_preserves_native_skip(monkeypatch, tmp_path):
    native, _ = run_main(monkeypatch, tmp_path / "native", False, long_raw=True)
    comparison, trace = run_main(monkeypatch, tmp_path / "comparison", True, long_raw=True)
    assert len(native) == 72
    assert len(comparison) == 74
    assert len(comparison[comparison.policy == "full_raw"]) == 2
    reader = [x for x in trace if x[:2] == ("generate", "reader")]
    assert len(reader[0][4]["items"]) == 30
    assert reader[0][4]["items"][14] == {"policy": "full_raw", "question_ordinal": 0}
