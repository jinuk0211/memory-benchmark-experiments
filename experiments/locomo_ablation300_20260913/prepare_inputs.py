"""Prepare source-only inputs and a deterministic stratified 300-question sample."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SALT = "locomo-ablation300-20260913-v1"
SOURCE = WORKSPACE / "generalization_20260908/source"
IMPORTED = WORKSPACE / "migration_20260910/seed_parent_ablation/runs/seed_parent_locomo_qwen35_minilm_r2/imported_source"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def allocate(counts: dict, total: int) -> dict:
    population = sum(counts.values())
    result = {key: value * total // population for key, value in counts.items()}
    order = sorted(counts, key=lambda key: (-(counts[key] * total % population), str(key)))
    for key in order[:total - sum(result.values())]:
        result[key] += 1
    return result


def select(records: list) -> list:
    categories = Counter(r["category"] for r in records)
    quotas = allocate(categories, 300)
    selected = []
    for category, n in sorted(quotas.items()):
        population = [r for r in records if r["category"] == category]
        conv_quotas = allocate(Counter(r["conv_id"] for r in population), n)
        for cid, count in sorted(conv_quotas.items()):
            pool = [r for r in population if r["conv_id"] == cid]
            pool.sort(key=lambda r: (hashlib.sha256(
                (SALT + "\0" + r["id"]).encode()).hexdigest(), r["id"]))
            selected.extend(pool[:count])
    selected.sort(key=lambda r: (r["conv_id"], r["qa_index"]))
    if len(selected) != 300 or len({r["id"] for r in selected}) != 300:
        raise ValueError("Invalid 300-question selection")
    return selected


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_core", SOURCE / "refine.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    dataset_path = WORKSPACE / "experiments/recursive_minilm_20260909/data/locomo10.json"
    if sha(dataset_path) != "cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993":
        raise ValueError("Canonical LoCoMo data changed")
    dataset = read(dataset_path)
    sources = read(IMPORTED / "source_sessions.json")
    if {str(s["sample_id"]): core.session_data(s) for s in dataset} != sources:
        raise ValueError("Imported source sessions differ from canonical LoCoMo")
    lock = read(IMPORTED / "memory_lock.json")
    if lock["benchmark_questions_used"] or lock["source_sha256"] != core.digest(sources):
        raise ValueError("Historical source lock mismatch")
    original_protocol = read(IMPORTED / "protocol.json")
    if lock["protocol_sha256"] != core.digest(original_protocol):
        raise ValueError("Historical source protocol mismatch")
    records, labels = [], {}
    for sample in dataset:
        cid = str(sample["sample_id"])
        for index, qa in enumerate(sample["qa"]):
            if int(qa["category"]) not in (1, 2, 3, 4):
                continue
            identity = f"{cid}:{index}"
            records.append({"id": identity, "conv_id": cid,
                            "qa_index": index, "category": int(qa["category"])})
            labels[identity] = {"question": qa["question"], "gold": str(qa["answer"])}
    if len(records) != 1540 or len(sources) != 10:
        raise ValueError("Expected 1540 questions across ten conversations")
    selection = select(records)
    package = ROOT / "package"
    inputs, code = package / "input", package / "code"
    inputs.mkdir(parents=True, exist_ok=True)
    code.mkdir(parents=True, exist_ok=True)
    save(inputs / "selection.json", {"salt": SALT, "records": selection,
         "category_counts": dict(Counter(r["category"] for r in selection)),
         "conversation_counts": dict(Counter(r["conv_id"] for r in selection)),
         "selection": "Largest-remainder category quotas then conversation quotas; SHA256 ID order",
         "source_population": 1540, "selected": 300,
         "interpretation": "Previously exposed LoCoMo histories; exploratory component comparison"})
    save(inputs / "source_sessions.json", sources)
    save(inputs / "reader_questions.json",
         [{**r, "question": labels[r["id"]]["question"]} for r in selection])
    save(ROOT / "evaluation/gold.json",
         [{**r, **labels[r["id"]]} for r in selection])
    lineage = read(IMPORTED.parent / "import_lineage.json")
    if sha(IMPORTED.parent / "import_lineage.json") != "1d0b71d7b723b27a41f3b46c65154c291623906566221c2c8b2dd3960b6f1c6c":
        raise ValueError("Historical import lineage fingerprint changed")
    for relative, receipt in lineage["files"].items():
        if sha(IMPORTED / relative) != receipt["sha256"]:
            raise ValueError(f"Historical import file changed: {relative}")
    utility = defaultdict(list)
    ids = read(IMPORTED / "utility_selection.json")["ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate utility IDs")
    for identity in ids:
        row = read(IMPORTED / "utility/items" / (identity.replace(":", "_") + ".json"))
        if row["id"] != identity or row["split"] != "probe_fit" or row["conv_id"] not in sources:
            raise ValueError("Invalid source-only utility identity")
        utility[row["conv_id"]].append(row)
    if set(utility) != set(sources):
        raise ValueError("Utility history coverage mismatch")
    save(inputs / "source_utility.json", dict(utility))
    plans = {name: read(SOURCE / "runs/continuous_v3/plans" / (name + ".json"))
             for name in ("r06_calendar_month", "r12_filter_current_best", "r40_fused_four_turn")}
    save(inputs / "plans.json", plans)
    vendor = ROOT / "vendor/source"
    vendor.mkdir(parents=True, exist_ok=True)
    source_names = [p.name for p in (WORKSPACE / "experiments/reuse_memory_20260913/vendor/source").glob("*.py")]
    for name in source_names:
        origin = (WORKSPACE / "experiments/reuse_memory_20260913/vendor/source" / name
                  if name == "frozen_partition.py" else SOURCE / name)
        shutil.copy2(origin, vendor / name)
    for name in ("transfer_runtime.py", "runtime_meter.py", "chat_tokenizer_compat.py"):
        shutil.copy2(WORKSPACE / "experiments/reuse_memory_20260913/vendor" / name, ROOT / "vendor" / name)
    shutil.copy2(WORKSPACE / "experiments/reuse_memory_20260913/environment.json", ROOT / "environment.json")
    for name in ("run_ablation.py", "test_ablation.py", "environment.json", "PROTOCOL.md"):
        shutil.copy2(ROOT / name, code / name)
    shutil.copytree(ROOT / "vendor", code / "vendor", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    manifest = {"files": {p.relative_to(inputs).as_posix(): sha(p)
                          for p in sorted(inputs.glob("*.json")) if p.name != "manifest.json"},
                "code": {p.relative_to(code).as_posix(): sha(p) for p in sorted(code.rglob("*"))
                         if p.is_file() and "__pycache__" not in p.parts},
                "original_dataset_sha256": sha(dataset_path),
                "original_source_protocol_sha256": sha(IMPORTED / "protocol.json"),
                "original_model": original_protocol.get("model"),
                "source_utility_count": len(ids), "answers_uploaded": False}
    save(inputs / "manifest.json", manifest)
    print(json.dumps({"selected": 300, "histories": 10, "utility_rows": len(ids),
                      "category_counts": dict(Counter(r["category"] for r in selection)),
                      "input_bytes": sum(p.stat().st_size for p in inputs.glob("*"))}))


if __name__ == "__main__":
    main()
