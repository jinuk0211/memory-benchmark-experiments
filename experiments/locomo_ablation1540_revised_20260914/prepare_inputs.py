"""Package source-only cached inputs and locally isolate canonical LoCoMo gold."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "locomo_ablation300_20260913"
FULL = ROOT.parent / "locomo_binding1540_20260913"
DATASET = ROOT.parent / "recursive_minilm_20260909/data/locomo10.json"
DATASET_SHA = "cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993"
TOKENIZER = ROOT.parents[1] / "model_staging/qwen35_9b_c202236/tokenizer.json"
sys.path.insert(0, str(ROOT / "package/code"))
import run_revised as runner  # noqa: E402


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    runner.frozen_save(path, value)


def copy_checked(source: Path, target: Path, expected: str | None = None) -> None:
    digest = sha(source)
    if expected is not None and digest != expected:
        raise ValueError(f"Archived artifact fingerprint changed: {source}")
    if target.exists() and sha(target) != digest:
        raise ValueError(f"Frozen target differs: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main() -> None:
    if sha(DATASET) != DATASET_SHA:
        raise ValueError("Canonical dataset fingerprint changed")
    prior_manifest = read(PRIOR / "package/input/manifest.json")
    for name, digest in prior_manifest["files"].items():
        if sha(PRIOR / "package/input" / name) != digest:
            raise ValueError(f"Prior input changed: {name}")
    if read(PRIOR / "collected/run/protocol.json")["manifest"] != prior_manifest:
        raise ValueError("Prior protocol differs")
    inputs, code = ROOT / "package/input", ROOT / "package/code"
    for name in ("source_sessions.json", "plans.json", "source_utility.json"):
        copy_checked(PRIOR / "package/input" / name, inputs / name, prior_manifest["files"][name])
    sources = read(inputs / "source_sessions.json")
    provenance = {}
    for i, cid in enumerate(sorted(sources)):
        lock_path = PRIOR / "collected/run/locks" / f"{cid}.json"
        lock = read(lock_path)
        if lock["benchmark_qa_used"] or lock["source_sha256"] != runner.core.digest(sources[cid]):
            raise ValueError("Prior source-only construction contract failed")
        for stage in ("initial", "audited"):
            source = PRIOR / "collected/run" / f"prepare_{i % 2}" / stage / f"{cid}.json"
            if runner.core.digest(read(source)) != lock[f"{stage}_sha256"]:
                raise ValueError("Cached extraction differs from prior lock")
            copy_checked(source, inputs / "cached" / stage / f"{cid}.json")
        source = PRIOR / "collected/run/memories/no_binding" / f"{cid}.json"
        copy_checked(source, inputs / "archived_no_binding" / f"{cid}.json", lock["memories"]["no_binding"])
        provenance[cid] = {"prior_lock_sha256": sha(lock_path),
                           "archived_no_binding_sha256": sha(source),
                           "initial_digest": lock["initial_sha256"], "audited_digest": lock["audited_sha256"]}
    questions, gold = [], []
    for sample in read(DATASET):
        cid = str(sample["sample_id"])
        for index, qa in enumerate(sample["qa"]):
            if int(qa["category"]) not in (1, 2, 3, 4):
                continue
            row = {"id": f"{cid}:{index}", "conv_id": cid, "qa_index": index,
                   "category": int(qa["category"]), "question": qa["question"]}
            questions.append(row)
            gold.append({**row, "gold": str(qa["answer"])})
    questions.sort(key=lambda r: (r["conv_id"], r["qa_index"]))
    gold.sort(key=lambda r: (r["conv_id"], r["qa_index"]))
    if len(questions) != 1540 or len({q["id"] for q in questions}) != 1540:
        raise ValueError("Expected unique full1540 population")
    dev = read(PRIOR / "package/input/reader_questions.json")
    if len(dev) != 300 or not all(q in questions for q in dev):
        raise ValueError("Fixed development300 population changed")
    save(inputs / "reader_questions.json", questions)
    save(inputs / "dev300_questions.json", dev)
    save(ROOT / "evaluation/gold.json", gold)
    selection = [{k: q[k] for k in ("id", "conv_id", "qa_index", "category")} for q in questions]
    save(inputs / "selection.json", {"records": selection, "selected": len(questions),
         "source_population": 1540, "category_counts": dict(Counter(str(q["category"]) for q in questions)),
         "selection": "All category1-4 questions; historically exposed exploratory evaluation"})
    save(inputs / "lineage.json", {"prior_protocol_sha256": sha(PRIOR / "collected/run/protocol.json"),
         "prior_manifest_sha256": sha(PRIOR / "package/input/manifest.json"), "cached_source_origin": provenance,
         "reader_development_population": "same frozen prior300; included in full1540; not unseen holdout",
         "candidate_configurations": list(runner.READERS),
         "random_seeds": runner.RANDOM_SEEDS,
         "selection_rule": "Evaluate both reader prompts on fixed300; choose grounded only if overall F1 improves; otherwise legacy. Freeze before full1540 component ablations."})
    files = {p.relative_to(inputs).as_posix(): sha(p) for p in sorted(inputs.rglob("*"))
             if p.is_file() and p.name != "manifest.json"}
    # Parent-owned pilot scripts have their own contract and are not part of memory construction.
    names = [code / "run_revised.py", code / "test_revised.py", code / "PROTOCOL.md", code / "environment.json"]
    names += [p for p in (code / "vendor").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    code_hashes = {p.relative_to(code).as_posix(): sha(p) for p in sorted(names)}
    save(inputs / "manifest.json", {"files": files, "code": code_hashes,
         "dataset_sha256": DATASET_SHA, "sample_count": 1540, "arms": list(runner.ARMS),
         "tokenizer_sha256": sha(TOKENIZER), "random_seeds": runner.RANDOM_SEEDS})
    print(json.dumps({"questions": len(questions), "arms": list(runner.ARMS),
                      "cached_histories": len(sources), "gold_is_local_only": True}))


if __name__ == "__main__":
    main()
