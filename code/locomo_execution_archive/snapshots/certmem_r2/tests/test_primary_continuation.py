"""Native main/build/retrieval CPU tests; never load models or rewrite dataset files."""

import argparse
import ast
import copy
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "scripts/run_system_v15.py"
IDS = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44",
       "conv-47", "conv-48", "conv-49", "conv-50"]
REMAINING = IDS[6:]


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if
             isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
             or isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102


@pytest.fixture
def native(monkeypatch, tmp_path):
    trace = []
    conversations = {}
    for cid in IDS:
        turns = [SimpleNamespace(speaker="Alice", text=f"I keep a cat in {cid}. It is black. It is friendly."),
                 SimpleNamespace(speaker="Alice", text="I enjoy tea. I drink tea each morning. I also like music.")]
        session = SimpleNamespace(conv_id=cid, num=1, date="1 May 2023", turns=turns)
        qa = SimpleNamespace(question="What does Alice enjoy?", answer="tea", category=1,
                             sessions=lambda: [1], question_date="2 May 2023")
        conversations[cid] = {"sessions": [session], "qa": [qa, copy.copy(qa)]}

    class Engine:
        def __init__(self, cfg):
            trace.append(("engine",))
            self.context = {}
            self.tok = lambda text, **kwargs: SimpleNamespace(input_ids=text.split())

        def set_context(self, **fields):
            self.context = fields

        def generate(self, system, prompts, max_tokens, **kwargs):
            trace.append(("generation", system, list(prompts), max_tokens, copy.deepcopy(self.context), kwargs))
            text = "Alice tea" if system == ns["REFORM_SYS"] else "Alice likes tea."
            return [text] * len(prompts)

    class Embedding:
        def __init__(self):
            trace.append(("embedding",))

        def encode(self, texts, **kwargs):
            trace.append(("encode", list(texts), kwargs))
            vectors = np.asarray([[1.0, float(len(text) % 7 + 1), float("tea" in text)] for text in texts])
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def facts(engine, sessions, win):
        trace.append(("facts", sessions[0].conv_id, win))
        return [[{"fact": "Alice keeps a cat", "turn_start": 1, "turn_end": 1},
                 {"fact": "Alice enjoys tea", "turn_start": 2, "turn_end": 2}]]

    def host(name, engine, cap):
        trace.append(("host", name, cap))

        def compress(session, state):
            trace.append(("compress", session.conv_id, cap))
            return ["Alice keeps a cat.", "Alice enjoys tea."]

        return SimpleNamespace(compress=compress)

    def entail(engine, block, fact_texts):
        trace.append(("entail", block, fact_texts))
        return [True, False] if len(fact_texts) == 2 else [True] * len(fact_texts)

    ns = {"argparse": argparse, "os": os, "np": np, "pd": pd, "time": time, "re": re,
          "copy": copy, "BM25Okapi": BM25Okapi, "Engine": Engine,
          "Config": lambda: SimpleNamespace(locomo_path="canonical-original-data"),
          "comparison_config": lambda cfg, out: cfg,
          "load_embedding": lambda cfg: Embedding(),
          "load_locomo": lambda path: conversations,
          "extract_facts": facts, "get_host": host, "check_entailment": entail,
          "f1n": lambda pred, gold: 1.0, "lenient": lambda pred, gold: 1}
    definitions(ROOT / "certmem/hosts/base.py", {"MemoryState"}, ns)
    definitions(ROOT / "scripts/run_read_v9.py", {"Index", "reader_prompt", "READER_SYS", "STOP", "content"}, ns)
    definitions(ROOT / "scripts/run_recover_v7.py", {"split_sentences"}, ns)
    definitions(SOURCE, {"main", "build_memory", "REFORM_SYS", "PROFILE_SYS"}, ns)
    counter = 0

    def run(*extra):
        nonlocal counter
        counter += 1
        trace.clear()
        out = tmp_path / str(counter)
        monkeypatch.setattr(sys, "argv", [str(SOURCE), "--out", str(out),
                            "--budgets", "400,1600,4000", *extra])
        ns["main"]()
        return pd.read_csv(out / "items.csv"), copy.deepcopy(trace)

    return SimpleNamespace(run=run, trace=trace, conversations=conversations, namespace=ns)


def readers(trace, system):
    return [event for event in trace if event[:2] == ("generation", system)]


def test_primary_prompts_meta_and_shared_work_match_original_full_grid(native):
    selector = ",".join(REMAINING)
    grid, before = native.run("--comparison", "--configs", "full", "--conversation-ids", selector)
    primary, after = native.run("--comparison", "--configs", "full", "--conversation-ids", selector, "--primary-only")
    ns = native.namespace
    filtered = grid[grid.policy.isin(["certified", "full_raw"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(primary, filtered)
    assert len(primary) == 2 * 2 * len(REMAINING)
    for old, new in zip(readers(before, ns["READER_SYS"]), readers(after, ns["READER_SYS"]), strict=True):
        selected = [(prompt, metadata) for prompt, metadata in zip(old[2], old[4]["items"], strict=True)
                    if metadata["policy"] in {"certified", "full_raw"}]
        assert new[2] == [item[0] for item in selected]
        assert new[4]["items"] == [item[1] for item in selected]
        assert new[3] == old[3] == 32
        assert new[5] == old[5] == {"think": False}
    assert [event for event in before if event[0] not in {"encode", "generation"}] == [
        event for event in after if event[0] not in {"encode", "generation"}]
    assert [event for event in before if event[0] == "generation" and event[1] != ns["READER_SYS"]] == [
        event for event in after if event[0] == "generation" and event[1] != ns["READER_SYS"]]
    # Two indexes + two multi-sentence turns are shared; exactly three encodes per QA remain.
    grid_encodes = [event for event in before if event[0] == "encode"]
    primary_encodes = [event for event in after if event[0] == "encode"]
    assert len(grid_encodes) == len(REMAINING) * (4 + 12 * 2)
    assert len(primary_encodes) == len(REMAINING) * (4 + 3 * 2)
    for index in range(len(REMAINING)):
        old = grid_encodes[index * 28:(index + 1) * 28]
        new = primary_encodes[index * 10:(index + 1) * 10]
        assert new == old[:4] + old[4:7] + old[16:19]


def test_selector_skips_completed_contexts_before_all_model_work_and_keeps_canonical_order(native):
    rows, trace = native.run("--comparison", "--configs", "full", "--primary-only",
                             "--conversation-ids", ",".join(reversed(REMAINING)))
    assert list(rows.conv_id.unique()) == REMAINING
    assert [(event[1], event[2]) for event in trace if event[0] == "facts"] == [(cid, 3) for cid in REMAINING]
    assert {event[1] for event in trace if event[0] == "compress"} == set(REMAINING)
    generations = [event for event in trace if event[0] == "generation"]
    assert {event[4]["conv_id"] for event in generations} == set(REMAINING)
    assert set(rows.policy) == {"certified", "full_raw"}


def test_remaining_four_contexts_keep_all_655_qa_with_duplicate_text_identity(native):
    for cid, count in zip(REMAINING, [150, 191, 156, 158], strict=True):
        native.conversations[cid]["qa"] = [copy.copy(native.conversations[cid]["qa"][0]) for _ in range(count)]
    rows, _ = native.run("--comparison", "--configs", "full", "--primary-only", "--conversation-ids", ",".join(REMAINING))
    assert len(rows) == 1310
    assert len(rows.drop_duplicates(["conv_id", "policy", "question_ordinal"])) == 1310
    assert rows.groupby("policy").size().to_dict() == {"certified": 655, "full_raw": 655}


@pytest.mark.parametrize("extra", [
    ["--primary-only"],
    ["--primary-only", "--comparison"],
    ["--primary-only", "--comparison", "--configs", "no_adaptive"],
    ["--primary-only", "--comparison", "--configs", "full,full"],
    ["--primary-only", "--comparison", "--configs", "full", "--no_raw"],
    ["--conversation-ids", ""],
    ["--conversation-ids", "conv-47,"],
    ["--conversation-ids", "conv-47,conv-47"],
    ["--conversation-ids", "unknown"],
    ["--conversation-ids", "conv-47,unknown"],
    ["--conversation-ids", "conv-47", "--limit_convs", "1"],
    ["--conversation-ids", "conv-47", "--dataset", "longmemeval"],
])
def test_invalid_scope_fails_before_engine_embedding_or_fact_work(native, extra):
    with pytest.raises(SystemExit) as caught:
        native.run(*extra)
    assert caught.value.code == 2
    assert native.trace == []


def test_default_grid_keeps_all_37_policies_per_qa(native):
    rows, trace = native.run("--comparison")
    assert len(rows) == len(IDS) * 2 * 37
    assert list(rows.conv_id.unique()) == IDS
    assert rows.groupby(["conv_id", "config", "question_ordinal"]).size().unique().tolist() == [15, 11]
    assert len(readers(trace, native.namespace["READER_SYS"])) == len(IDS) * 3
