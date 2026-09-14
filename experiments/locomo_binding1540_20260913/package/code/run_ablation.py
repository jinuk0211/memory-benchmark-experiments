"""Matched full-LoCoMo evidence-binding ablation using frozen source-only operations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "vendor/source"))
import refine as core  # noqa: E402
from continuous import anchor_units, dialogue_blocks  # noqa: E402
from continuous_v2 import construct as apply_recipe  # noqa: E402
from parent_evidence import construct as select_cues, make_options  # noqa: E402
from budgeted_evidence import storage_cost  # noqa: E402

ARMS = ("ours", "no_binding")
READ_BUDGET = 2048
ADD_BUDGET = 2000
SALT = "locomo-ablation300-20260913-v1"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_save(path: Path, value: Any) -> None:
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        core.save(path, value)


def verify_inputs(inputs: Path) -> dict:
    manifest = read(inputs / "manifest.json")
    for name, digest in manifest["files"].items():
        if sha(inputs / name) != digest:
            raise ValueError(f"Input fingerprint mismatch: {name}")
    for name, digest in manifest["code"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Code fingerprint mismatch: {name}")
    return manifest


def status(out: Path, phase: str, **fields: Any) -> None:
    row = {"phase": phase, "unix": time.time(), **fields}
    core.save(out / "status.json", row)
    print(json.dumps(row), flush=True)


def random_select(groups: list, mapped: dict, cid: str) -> tuple[list, dict]:
    """Shuffle the same eligible options; greedily admit one per probe."""
    options = [(i, mapped[o["id"]]) for i, group in enumerate(groups)
               for o in group if o["kind"] != "pair" and o["id"] in mapped]
    rng = random.Random(core.digest([SALT, "random-cues", cid]))
    rng.shuffle(options)
    selected, used, rounded = [], set(), 0
    for group_id, option in options:
        cost = math.ceil(option["cost"] / 8) * 8
        if group_id not in used and rounded + cost <= ADD_BUDGET:
            selected.append(option)
            used.add(group_id)
            rounded += cost
    return selected, {"selected_options": [o["id"] for o in selected],
                      "eligible_options": len(options),
                      "rounded_tokens": rounded,
                      "actual_tokens": sum(o["cost"] for o in selected),
                      "policy": "seeded random option-order greedy; one per probe"}


def backend(rt: Any, sessions: list, seed: list, plans: dict, arm: str) -> list:
    memory = copy.deepcopy(seed)
    if arm != "no_temporal":
        memory = apply_recipe(rt, sessions, memory, plans["r06_calendar_month"]["recipe"])
    memory = apply_recipe(rt, sessions, memory, plans["r12_filter_current_best"]["recipe"])
    if arm == "no_binding":
        facts = [u for u in memory if u["kind"] in ("fact", "extractive_fact", "profile_fact")]
        blocks = anchor_units(sessions, dialogue_blocks(sessions, 4, 2), "month")
        return core.dedupe(blocks + facts)
    recipe = copy.deepcopy(plans["r40_fused_four_turn"]["recipe"])
    if arm == "no_temporal":
        for operation in recipe["operations"]:
            if operation["op"] == "fuse_evidence":
                operation["anchor"] = False
    return apply_recipe(rt, sessions, memory, recipe)


def all_memories(rt: Any, sessions: list, initial: list, audited: list,
                 rows: list, plans: dict, cid: str) -> tuple[dict, dict]:
    seed = core.build(rt, sessions, "dialogue_residual", audited)
    unaudited_seed = core.build(rt, sessions, "dialogue_residual", initial)
    memories, details = {}, {}
    for arm in ("ours", "no_audit", "no_temporal", "no_binding"):
        parent = backend(rt, sessions, unaudited_seed if arm == "no_audit" else seed, plans, arm)
        groups, mapped = make_options(rt, sessions, parent, rows)
        memory, receipt = select_cues(parent, groups, mapped, {}, {}, rt.ntok,
                                      "parent_single", ADD_BUDGET)
        if memory[:len(parent)] != parent:
            raise ValueError("Cue construction modified base memory")
        memories[arm], details[arm] = memory, receipt
        if arm == "ours":
            memories["no_cues"] = copy.deepcopy(parent)
            random_options, random_receipt = random_select(groups, mapped, cid)
            memories["random_cues"] = copy.deepcopy(parent) + [
                copy.deepcopy(o["unit"]) for o in random_options]
            details["random_cues"] = random_receipt
            payload = copy.deepcopy(memory)
            for unit in payload:
                if "index_text" in unit:
                    unit["index_text"] = unit["text"]
            memories["payload_keys"] = payload
    source_ids = {t["id"] for s in sessions for t in s["turns"]}
    for arm, units in memories.items():
        covered = {sid for u in units for sid in u["sources"]}
        if covered != source_ids:
            raise ValueError(f"Source coverage changed in {cid}/{arm}")
        if arm == "payload_keys" and [u["text"] for u in units] != [u["text"] for u in memories["ours"]]:
            raise ValueError("Payload-key control changed evidence")
    return memories, details


def make_runtime(args: argparse.Namespace, stage: str) -> Any:
    from transfer_runtime import Runtime
    options = SimpleNamespace(stage=stage, out=args.out / f"{stage}_{args.shard}",
                              model="Qwen/Qwen3.5-9B",
                              embed_model="sentence-transformers/all-MiniLM-L6-v2",
                              embed_batch_size=64, seed=20260907)
    return Runtime(options, read(ROOT / "environment.json"))


def prepare(args: argparse.Namespace) -> None:
    manifest = verify_inputs(args.inputs)
    sessions_by_id = read(args.inputs / "source_sessions.json")
    plans = read(args.inputs / "plans.json")
    utility = read(args.inputs / "source_utility.json")
    out = args.out / f"prepare_{args.shard}"
    frozen_save(out / "input_lock.json", manifest)
    rt = make_runtime(args, "prepare")
    for i, (cid, sessions) in enumerate(sorted(sessions_by_id.items())):
        if i % args.shards != args.shard:
            continue
        lock_path = args.out / "locks" / (cid + ".json")
        if lock_path.exists():
            lock = read(lock_path)
            for arm, digest in lock["memories"].items():
                if sha(args.out / "memories" / arm / (cid + ".json")) != digest:
                    raise ValueError("Completed memory changed")
            continue
        status(out, "initial_extraction", conversation=cid)
        initial = core.build(rt, sessions, "session10")
        frozen_save(out / "initial" / (cid + ".json"), initial)
        status(out, "omission_audit", conversation=cid)
        audited = core.build(rt, sessions, "audit", initial)
        frozen_save(out / "audited" / (cid + ".json"), audited)
        status(out, "ablation_construction", conversation=cid)
        memories, details = all_memories(rt, sessions, initial, audited, utility[cid], plans, cid)
        hashes, counts = {}, {}
        for arm in ARMS:
            path = args.out / "memories" / arm / (cid + ".json")
            frozen_save(path, memories[arm])
            hashes[arm] = sha(path)
            counts[arm] = sum(storage_cost(u, rt.ntok) for u in memories[arm])
        frozen_save(args.out / "construction" / (cid + ".json"), details)
        frozen_save(lock_path, {"source_sha256": core.digest(sessions), "memories": hashes,
                               "stored_tokens": counts, "benchmark_qa_used": False,
                               "initial_sha256": core.digest(initial),
                               "audited_sha256": core.digest(audited)})
        status(out, "history_complete", conversation=cid, stored_tokens=counts)
    status(out, "prepare_complete")


def evaluate(args: argparse.Namespace) -> None:
    import numpy as np
    from rank_bm25 import BM25Okapi
    verify_inputs(args.inputs)
    sources = read(args.inputs / "source_sessions.json")
    for cid in sources:
        lock = read(args.out / "locks" / (cid + ".json"))
        if lock["source_sha256"] != core.digest(sources[cid]) or lock["benchmark_qa_used"]:
            raise ValueError("Source construction lock failed")
        for arm, digest in lock["memories"].items():
            if sha(args.out / "memories" / arm / (cid + ".json")) != digest:
                raise ValueError("Memory fingerprint changed")
    questions = read(args.inputs / "reader_questions.json")
    rt = make_runtime(args, "evaluate")
    out = args.out / f"evaluate_{args.shard}"
    def ranks(scores: Any) -> Any:
        order = np.argsort(-np.asarray(scores), kind="stable")
        result = np.empty(len(order), dtype=np.int64)
        result[order] = np.arange(len(order))
        return result
    for i, cid in enumerate(sorted(sources)):
        if i % args.shards != args.shard:
            continue
        selected = [q for q in questions if q["conv_id"] == cid]
        queries = rt.encode([q["question"] for q in selected], query=True)
        for arm in ARMS:
            path = args.out / "predictions" / arm / (cid + ".json")
            if path.exists():
                old = read(path)
                if [r["id"] for r in old] != [r["id"] for r in selected]:
                    raise ValueError("Completed prediction population changed")
                continue
            status(out, "reader", conversation=cid, arm=arm, n=len(selected))
            units = read(args.out / "memories" / arm / (cid + ".json"))
            keys = [u.get("index_text", u["text"]) for u in units]
            dense = rt.encode(keys)
            sparse = BM25Okapi([core.lexical(k) or ["_empty"] for k in keys])
            stored = sum(storage_cost(u, rt.ntok) for u in units)
            metadata, prompts = [], []
            for q, query in zip(selected, queries):
                score = 1 / (60 + ranks(dense @ query)) + 1 / (
                    60 + ranks(sparse.get_scores(core.lexical(q["question"]))))
                order = np.argsort(-score, kind="stable")[:120]
                context, hits, tokens = core.pack(units, order, rt.ntok, READ_BUDGET)
                prompt = f'Conversation memory:\n{context}\n\nQuestion: {q["question"]}\nAnswer:'
                prompts.append(prompt)
                metadata.append({**q, "arm": arm, "context": context,
                                 "context_sha256": core.digest(context),
                                 "read_tokens": tokens, "stored_tokens": stored,
                                 "memory_indices": [int(x) for x in hits]})
            predictions = rt.generate(core.READER, prompts, max_tokens=96)
            if len(predictions) != len(selected):
                raise ValueError("Reader omitted predictions")
            rows = []
            for row, prompt, prediction in zip(metadata, prompts, predictions):
                key = core.digest([rt.model_meta, rt.args.seed, core.READER, prompt, 96, False])
                native = read(rt.cache / "generations" / (key + ".json"))
                if native["text"] != prediction or not isinstance(prediction, str):
                    raise ValueError("Native response mismatch")
                rows.append({**row, "prediction": prediction,
                             "finish_reason": native["finish_reason"],
                             "input_tokens": native["input_tokens"],
                             "output_tokens": native["output_tokens"],
                             "generation_cache_key": key})
            frozen_save(path, rows)
    status(out, "evaluation_complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "prepare", "evaluate"))
    parser.add_argument("--inputs", type=Path, default=ROOT.parent / "input")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "run")
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")
    manifest = verify_inputs(args.inputs)
    contract = {"manifest": manifest, "shards": args.shards, "arms": list(ARMS),
                "read_budget": READ_BUDGET, "extra_budget": ADD_BUDGET}
    if args.stage == "freeze":
        frozen_save(args.out / "protocol.json", contract)
        return
    if read(args.out / "protocol.json") != contract:
        raise ValueError("Run protocol changed; choose a new output directory")
    (prepare if args.stage == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
