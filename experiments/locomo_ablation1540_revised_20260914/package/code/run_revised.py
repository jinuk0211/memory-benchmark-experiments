"""Source-only matched ablations of the no-binding LoCoMo memory."""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "vendor/source"))
import refine as core  # noqa: E402
from continuous import anchor_units, dialogue_blocks, day_label, parse_date  # noqa: E402
from continuous_v2 import construct as apply_recipe  # noqa: E402
from parent_evidence import construct as select_cues, make_options  # noqa: E402
from budgeted_evidence import storage_cost  # noqa: E402

RANDOM_SEEDS = {f"random_{i:02d}": 2026091400 + i for i in range(10)}
ARMS = ("ours", "no_cues", "no_audit", "no_temporal", "with_binding", "payload_keys", *RANDOM_SEEDS)
READ_BUDGET = 2048
ADD_BUDGET = 2000
SALT = "locomo-ablation1540-revised-20260914-v1"
GROUNDED_READER = (
    "Answer the question using the provided conversation memory as evidence. "
    "You may combine supported facts with relevant common or world knowledge to draw reasonable inferences. "
    "Do not invent personal events, identities, or details absent from the evidence. "
    "Give a concise exact answer at the level requested by the question, such as a name, date, "
    "country, state, number, or named concept. For a list question include every supported item, "
    "separated by commas. Use session dates to interpret relative times, but distinguish the event "
    "date from the date someone talked about it. If the answer cannot reasonably be inferred, "
    "answer unknown. No explanation or introductory text."
)
READERS = {"legacy": core.READER, "grounded": GROUNDED_READER}


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


class TokenRuntime:
    """Exact tokenizer only: cached facts make GPU generation unnecessary."""

    def __init__(self, tokenizer: Path, expected_sha: str) -> None:
        from tokenizers import Tokenizer
        if sha(tokenizer) != expected_sha:
            raise ValueError("Pinned tokenizer fingerprint changed")
        self.tok = Tokenizer.from_file(str(tokenizer))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.ntok = lru_cache(maxsize=100000)(
            lambda text: len(self.tok.encode(text, add_special_tokens=False).ids))


def canonical_unresolved(sessions: list, units: list) -> list:
    """Keep the normalized metadata/header; change only relative-time body expansion."""
    dates = {s["num"]: s["date"] for s in sessions}
    result = []
    for original in units:
        if original.get("calendar_anchored"):
            raise ValueError("Expected cached source units before calendar normalization")
        unit = copy.deepcopy(original)
        date = dates[unit["session"]]
        normalized = parse_date(date)
        body = re.sub(r"^\(Session date: [^\n]*?\)\s*", "", unit["text"])
        unit["text"] = f"(Recorded on: {day_label(normalized) if normalized else date}) " + body
        unit["calendar_anchored"] = True
        result.append(unit)
    return result


def build_parents(rt: Any, sessions: list, facts: list, plans: dict) -> tuple[list, list, list]:
    """Return normal, unresolved, and bound parents from identical cached facts."""
    seed = core.build(rt, sessions, "dialogue_residual", facts)
    normalized_seed = anchor_units(sessions, seed, "month")
    unresolved_seed = canonical_unresolved(sessions, seed)
    normalized = apply_recipe(rt, sessions, seed, plans["r06_calendar_month"]["recipe"])
    filtered = apply_recipe(rt, sessions, normalized, plans["r12_filter_current_best"]["recipe"])
    blocks = dialogue_blocks(sessions, 4, 2)
    normalized_blocks = anchor_units(sessions, blocks, "month")
    unresolved_blocks = canonical_unresolved(sessions, blocks)
    counterpart = {}
    for norm, raw in zip(normalized_seed + normalized_blocks, unresolved_seed + unresolved_blocks):
        counterpart.setdefault(core.digest(norm), raw)
    selected_facts = [u for u in filtered if u["kind"] in ("fact", "extractive_fact", "profile_fact")]
    parent = core.dedupe(normalized_blocks + selected_facts)
    # Select identity/order using the normalized parent; disabling expansion cannot
    # silently change filtering, deduplication, or unit boundaries.
    unresolved = [copy.deepcopy(counterpart[core.digest(u)]) for u in parent]
    bound = apply_recipe(rt, sessions, filtered, plans["r40_fused_four_turn"]["recipe"])
    return parent, unresolved, bound


def random_select(groups: list, mapped: dict, cid: str, seed: int) -> tuple[list, dict]:
    options = [(i, mapped[o["id"]]) for i, group in enumerate(groups)
               for o in group if o["kind"] != "pair" and o["id"] in mapped]
    rng = random.Random(core.digest([SALT, "random-cues", seed, cid]))
    rng.shuffle(options)
    selected, used, rounded = [], set(), 0
    for group_id, option in options:
        cost = math.ceil(option["cost"] / 8) * 8
        if group_id not in used and rounded + cost <= ADD_BUDGET:
            selected.append(option)
            used.add(group_id)
            rounded += cost
    return selected, {"selected_options": [o["id"] for o in selected],
                      "eligible_options": len(options), "rounded_tokens": rounded,
                      "actual_tokens": sum(o["cost"] for o in selected), "seed": seed,
                      "policy": "seeded random option-order greedy; one per probe"}


def accounting(units: list, parent: list, ntok: Any) -> dict:
    if units[:len(parent)] != parent:
        raise ValueError("Cue construction modified parent memory")
    parent_tokens = sum(ntok(u["text"]) for u in parent)
    cue_tokens = sum(ntok(u["text"]) for u in units[len(parent):])
    key_tokens = sum(ntok(u["index_text"]) for u in units
                     if u.get("index_text", u["text"]) != u["text"])
    total = sum(storage_cost(u, ntok) for u in units)
    if parent_tokens + cue_tokens + key_tokens != total:
        raise ValueError("Storage accounting failed")
    if cue_tokens + key_tokens > ADD_BUDGET:
        raise ValueError("Cue cap exceeded")
    return {"parent_payload_tokens": parent_tokens, "cue_payload_tokens": cue_tokens,
            "distinct_key_tokens": key_tokens, "total_stored_tokens": total,
            "parent_unit_count": len(parent), "cue_unit_count": len(units) - len(parent),
            "parent_sha256": core.digest(parent)}


def all_memories(rt: Any, sessions: list, initial: list, audited: list,
                 rows: list, plans: dict, cid: str) -> tuple[dict, dict, dict]:
    parent, unresolved, bound = build_parents(rt, sessions, audited, plans)
    unaudited, _, _ = build_parents(rt, sessions, initial, plans)
    parents = {"ours": parent, "no_audit": unaudited,
               "no_temporal": unresolved, "with_binding": bound}
    memories, details = {}, {}
    for arm, base in tuple(parents.items()):
        groups, mapped = make_options(rt, sessions, base, rows)
        memory, receipt = select_cues(base, groups, mapped, {}, {}, rt.ntok,
                                      "parent_single", ADD_BUDGET)
        memories[arm], details[arm] = memory, receipt
        if arm == "ours":
            memories["no_cues"] = copy.deepcopy(base)
            parents["no_cues"] = base
            payload = copy.deepcopy(memory)
            for unit in payload:
                if "index_text" in unit:
                    unit["index_text"] = unit["text"]
            memories["payload_keys"] = payload
            parents["payload_keys"] = base
            details["payload_keys"] = {"selected_options": receipt["selected_options"],
                                        "policy": "same units and payloads; keys set to payload"}
            for random_arm, seed in RANDOM_SEEDS.items():
                selected, random_receipt = random_select(groups, mapped, cid, seed)
                memories[random_arm] = copy.deepcopy(base) + [copy.deepcopy(o["unit"]) for o in selected]
                details[random_arm], parents[random_arm] = random_receipt, base
    source_ids = {t["id"] for s in sessions for t in s["turns"]}
    accounts = {}
    for arm in ARMS:
        units = memories[arm]
        if {sid for u in units for sid in u["sources"]} != source_ids:
            raise ValueError(f"Source coverage changed: {cid}/{arm}")
        accounts[arm] = accounting(units, parents[arm], rt.ntok)
    if [u["text"] for u in memories["ours"]] != [u["text"] for u in memories["payload_keys"]]:
        raise ValueError("Payload-key control changed evidence")
    if [(u["kind"], u["session"], u["sources"]) for u in parent] != [
            (u["kind"], u["session"], u["sources"]) for u in unresolved]:
        raise ValueError("Temporal control changed unit identities")
    return memories, details, accounts


def prepare(args: argparse.Namespace) -> None:
    manifest = verify_inputs(args.inputs)
    rt = TokenRuntime(args.tokenizer, manifest["tokenizer_sha256"])
    sources, plans, utility = (read(args.inputs / name) for name in
                               ("source_sessions.json", "plans.json", "source_utility.json"))
    frozen_save(args.out / "protocol.json", {"stage": "prepare", "manifest": manifest,
                "arms": list(ARMS), "random_seeds": RANDOM_SEEDS,
                "read_budget": READ_BUDGET, "extra_budget": ADD_BUDGET})
    for cid, sessions in sorted(sources.items()):
        initial = read(args.inputs / "cached/initial" / f"{cid}.json")
        audited = read(args.inputs / "cached/audited" / f"{cid}.json")
        status(args.out, "construction", conversation=cid)
        memories, details, accounts = all_memories(rt, sessions, initial, audited, utility[cid], plans, cid)
        archived_path = args.inputs / "archived_no_binding" / f"{cid}.json"
        if memories["ours"] != read(archived_path):
            raise ValueError(f"No-binding baseline did not reproduce: {cid}")
        hashes = {}
        for arm in ARMS:
            path = args.out / "memories" / arm / f"{cid}.json"
            frozen_save(path, memories[arm])
            if arm == "ours" and sha(path) != sha(archived_path):
                # Preserve original immutable bytes after full semantic equality.
                shutil.copyfile(archived_path, path)
            hashes[arm] = sha(path)
        frozen_save(args.out / "construction" / f"{cid}.json", details)
        frozen_save(args.out / "locks" / f"{cid}.json", {
            "source_sha256": core.digest(sessions), "benchmark_qa_used": False,
            "initial_sha256": core.digest(initial), "audited_sha256": core.digest(audited),
            "memories": hashes, "stored_tokens": {a: accounts[a]["total_stored_tokens"] for a in ARMS},
            "accounting": accounts, "archived_no_binding_sha256": sha(archived_path)})
        status(args.out, "history_complete", conversation=cid)
    status(args.out, "prepare_complete")


def population(inputs: Path, name: str) -> list:
    return read(inputs / ("reader_questions.json" if name == "full1540" else "dev300_questions.json"))


def verify_memories(inputs: Path, memory_root: Path) -> dict:
    sources = read(inputs / "source_sessions.json")
    hashes = {}
    for cid, sessions in sources.items():
        path = memory_root / "locks" / f"{cid}.json"
        lock = read(path)
        if lock["source_sha256"] != core.digest(sessions) or lock["benchmark_qa_used"]:
            raise ValueError("Source construction lock failed")
        if set(lock["memories"]) != set(ARMS):
            raise ValueError("Incomplete matched memory arms")
        for arm, digest in lock["memories"].items():
            if sha(memory_root / "memories" / arm / f"{cid}.json") != digest:
                raise ValueError("Memory fingerprint changed")
        hashes[cid] = sha(path)
    return hashes


def evaluation_contract(args: argparse.Namespace) -> dict:
    manifest = verify_inputs(args.inputs)
    questions = population(args.inputs, args.population)
    return {"stage": "evaluate", "manifest": manifest, "arms": list(args.arms),
            "random_seeds": RANDOM_SEEDS, "population": args.population,
            "questions_sha256": core.digest(questions), "question_count": len(questions),
            "reader": args.reader, "reader_system": READERS[args.reader],
            "environment": read(ROOT / "environment.json"),
            "read_budget": READ_BUDGET, "extra_budget": ADD_BUDGET,
            "max_output_tokens": 96, "shards": args.shards,
            "memory_locks": verify_memories(args.inputs, args.memory_root),
            "retrieval": "dense+BM25 stable RRF k=60 zero-based; top120; whole payload skip/continue",
            "packing": "legacy", "exposure": "exploratory; all histories and full1540 previously exposed",
            "candidate_configurations": ["legacy", "grounded"],
            "selection_rule": "Report both dev300 configurations; freeze chosen reader before full ablations."}


def make_runtime(args: argparse.Namespace) -> Any:
    from transfer_runtime import Runtime
    options = SimpleNamespace(stage="evaluate", out=args.out / f"evaluate_{args.shard}",
                              model="Qwen/Qwen3.5-9B",
                              embed_model="sentence-transformers/all-MiniLM-L6-v2",
                              embed_batch_size=64, seed=20260907)
    return Runtime(options, read(ROOT / "environment.json"))


def validate_receipt(native: dict, prediction: str, expected_input_tokens: int) -> None:
    """Keep empty and length-ended outputs, but reject malformed native metadata."""
    if not isinstance(prediction, str) or native["text"] != prediction:
        raise ValueError("Native response mismatch")
    for field in ("input_tokens", "output_tokens"):
        if type(native[field]) is not int or native[field] < 0:
            raise ValueError(f"Invalid native {field}")
    if native["input_tokens"] != expected_input_tokens or native["output_tokens"] > 96:
        raise ValueError("Native token counts disagree with the frozen request")
    if native["finish_reason"] not in {"stop", "length"}:
        raise ValueError("Native generation ended with an error instead of a completed answer")


def evaluate(args: argparse.Namespace) -> None:
    import numpy as np
    from rank_bm25 import BM25Okapi
    contract = evaluation_contract(args)
    frozen_save(args.out / "protocol.json", contract)
    questions = population(args.inputs, args.population)
    rt = make_runtime(args)
    out = args.out / f"evaluate_{args.shard}"
    system = READERS[args.reader]
    def ranks(scores: Any) -> Any:
        order = np.argsort(-np.asarray(scores), kind="stable")
        result = np.empty(len(order), dtype=np.int64)
        result[order] = np.arange(len(order))
        return result
    sources = read(args.inputs / "source_sessions.json")
    for i, cid in enumerate(sorted(sources)):
        if i % args.shards != args.shard:
            continue
        selected = [q for q in questions if q["conv_id"] == cid]
        queries = rt.encode([q["question"] for q in selected], query=True)
        for arm in args.arms:
            path = args.out / "predictions" / arm / f"{cid}.json"
            if path.exists():
                old = read(path)
                if [r["id"] for r in old] != [r["id"] for r in selected]:
                    raise ValueError("Completed prediction population changed")
                continue
            status(out, "reader", conversation=cid, arm=arm, n=len(selected))
            units = read(args.memory_root / "memories" / arm / f"{cid}.json")
            keys = [u.get("index_text", u["text"]) for u in units]
            dense = rt.encode(keys)
            sparse = BM25Okapi([core.lexical(k) or ["_empty"] for k in keys])
            stored = sum(storage_cost(u, rt.ntok) for u in units)
            expected = read(args.memory_root / "locks" / f"{cid}.json")["stored_tokens"][arm]
            if stored != expected:
                raise ValueError("CPU preparation and GPU runtime tokenizer accounting differ")
            metadata, prompts = [], []
            for q, query in zip(selected, queries):
                score = 1 / (60 + ranks(dense @ query)) + 1 / (
                    60 + ranks(sparse.get_scores(core.lexical(q["question"]))))
                order = np.argsort(-score, kind="stable")[:120]
                context, hits, tokens = core.pack(units, order, rt.ntok, READ_BUDGET)
                if tokens > READ_BUDGET or rt.ntok(context) != tokens:
                    raise ValueError("Reader evidence cap mismatch")
                prompt = f'Conversation memory:\n{context}\n\nQuestion: {q["question"]}\nAnswer:'
                prompts.append(prompt)
                metadata.append({**q, "arm": arm, "context": context,
                                 "context_sha256": core.digest(context),
                                 "read_tokens": tokens, "stored_tokens": stored,
                                 "memory_indices": [int(x) for x in hits]})
            predictions = rt.generate(system, prompts, max_tokens=96)
            if len(predictions) != len(selected):
                raise ValueError("Reader omitted predictions")
            rows = []
            for row, prompt, prediction in zip(metadata, prompts, predictions):
                key = core.digest([rt.model_meta, rt.args.seed, system, prompt, 96, False])
                native = read(rt.cache / "generations" / f"{key}.json")
                chat = rt.tok.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                validate_receipt(native, prediction, rt.ntok(chat))
                rows.append({**row, "prediction": prediction,
                             "finish_reason": native["finish_reason"],
                             "input_tokens": native["input_tokens"],
                             "output_tokens": native["output_tokens"], "generation_cache_key": key})
            frozen_save(path, rows)
    status(out, "evaluation_complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "freeze", "evaluate"))
    parser.add_argument("--inputs", type=Path, default=ROOT.parent / "input")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, default=ROOT.parent / "run")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--reader", choices=READERS, default="legacy")
    parser.add_argument("--population", choices=("full1540", "dev300"), default="full1540")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or len(args.arms) != len(set(args.arms)):
        raise ValueError("Invalid shard or repeated arms")
    if args.stage == "prepare":
        if args.tokenizer is None:
            raise ValueError("CPU preparation needs --tokenizer tokenizer.json")
        prepare(args)
    elif args.stage == "freeze":
        frozen_save(args.out / "protocol.json", evaluation_contract(args))
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
