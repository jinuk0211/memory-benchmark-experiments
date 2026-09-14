"""Validate paired LoCoMo1540 predictions and export the requested ablation table."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "lightmem_analysis_deps"))
sys.path.insert(0, str(ROOT / "vendor/source"))
import refine as core  # noqa: E402

ARMS = ("ours", "no_binding")
LABELS = ("Our method", "w/o evidence binding")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    storage: dict[str, int] = {}
    for row in rows:
        prior = storage.setdefault(row["conv_id"], row["stored_tokens"])
        if prior != row["stored_tokens"]:
            raise ValueError("Within-history storage changed")
    return {"n": len(rows), "histories": len(storage),
            "f1": statistics.mean(r["official_f1"] for r in rows) * 100,
            "stored_tokens": statistics.mean(storage.values()),
            "read_tokens": statistics.mean(r["read_tokens"] for r in rows),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
            "empty_answers": sum(not r["prediction"].strip() for r in rows),
            "category_f1": {str(c): statistics.mean(
                r["official_f1"] for r in rows if r["category"] == c) * 100
                for c in sorted({r["category"] for r in rows})}}


def interval(base: list[dict], variant: list[dict]) -> list[float]:
    import numpy as np
    lookup = {r["id"]: r["official_f1"] for r in base}
    groups: dict[str, list[float]] = {}
    for row in variant:
        groups.setdefault(row["conv_id"], []).append((row["official_f1"] - lookup[row["id"]]) * 100)
    values = list(groups.values())
    rng = np.random.default_rng(20260913)
    draws = [statistics.mean(x for i in rng.integers(0, len(values), len(values))
                             for x in values[i]) for _ in range(5000)]
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "collected/run")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    manifest = read(ROOT / "package/input/manifest.json")
    contract = read(args.run / "protocol.json")
    if contract != {"manifest": manifest, "shards": 2, "arms": list(ARMS),
                    "read_budget": 2048, "extra_budget": 2000}:
        raise ValueError("Downloaded protocol differs from frozen local contract")
    for name in ("reader_questions.json", "selection.json"):
        if sha(ROOT / "package/input" / name) != manifest["files"][name]:
            raise ValueError(f"Frozen local input changed: {name}")
    if sha(ROOT / "vendor/source/refine.py") != manifest["code"]["vendor/source/refine.py"]:
        raise ValueError("Frozen official scorer changed")
    dataset_path = ROOT.parents[1] / "experiments/recursive_minilm_20260909/data/locomo10.json"
    if sha(dataset_path) != manifest["original_dataset_sha256"]:
        raise ValueError("Canonical LoCoMo dataset changed")
    dataset = {str(row["sample_id"]): row for row in read(dataset_path)}
    selected = read(ROOT / "package/input/selection.json")["records"]
    canonical_gold = []
    for record in selected:
        qa = dataset[record["conv_id"]]["qa"][record["qa_index"]]
        if int(qa["category"]) != record["category"]:
            raise ValueError("Selection category differs from canonical question")
        canonical_gold.append({**record, "question": qa["question"], "gold": str(qa["answer"])})
    gold = read(ROOT / "evaluation/gold.json")
    if gold != canonical_gold:
        raise ValueError("Evaluation gold differs from canonical1540 labels")
    expected = {r["id"]: r for r in gold}
    histories = sorted({r["conv_id"] for r in gold})
    if len(gold) != 1540 or len(expected) != 1540 or len(histories) != 10:
        raise ValueError("Expected1540 unique questions in10 histories")
    scores, summary = {}, {}
    for arm, label in zip(ARMS, LABELS):
        rows = []
        for cid in histories:
            lock = read(args.run / "locks" / (cid + ".json"))
            path = args.run / "memories" / arm / (cid + ".json")
            if lock != read(ROOT / "package/input/locks" / (cid + ".json")):
                raise ValueError("Run lock differs from frozen import")
            if sha(path) != manifest["files"][f"memories/{arm}/{cid}.json"]:
                raise ValueError("Memory differs from original frozen memory")
            if sha(path) != lock["memories"][arm] or lock["benchmark_qa_used"]:
                raise ValueError("Memory lock failed")
            part = read(args.run / "predictions" / arm / (cid + ".json"))
            if any(r["stored_tokens"] != lock["stored_tokens"][arm] for r in part):
                raise ValueError("Stored-token metadata mismatch")
            rows.extend(part)
        if len(rows) != 1540 or {r["id"] for r in rows} != set(expected):
            raise ValueError(f"Incomplete/duplicate population: {arm}")
        for row in rows:
            reference = expected[row["id"]]
            if any(row[k] != reference[k] for k in ("question", "category", "conv_id", "qa_index")):
                raise ValueError("Question identity mismatch")
            if row["arm"] != arm or not 0 <= row["read_tokens"] <= 2048:
                raise ValueError("Arm or read-budget mismatch")
            if core.digest(row["context"]) != row["context_sha256"]:
                raise ValueError("Context fingerprint mismatch")
            shard = histories.index(row["conv_id"]) % 2
            native = read(args.run / f"evaluate_{shard}/cache/generations" /
                          (row["generation_cache_key"] + ".json"))
            if native["text"] != row["prediction"] or any(
                native[k] != row[k] for k in ("finish_reason", "input_tokens", "output_tokens")
            ):
                raise ValueError("Native reader receipt mismatch")
            row["gold"] = reference["gold"]
            row["official_f1"] = core.f1(row["prediction"], row["gold"], row["category"])
        scores[arm] = rows
        summary[arm] = {"label": label, **summarize(rows)}
    for arm in ARMS:
        summary[arm]["delta_pp"] = summary[arm]["f1"] - summary["ours"]["f1"]
        summary[arm]["paired_95ci_pp"] = interval(scores["ours"], scores[arm])
    args.out.mkdir(parents=True, exist_ok=True)
    core.save(args.out / "RESULTS.json", {
        "selection": read(ROOT / "package/input/selection.json"), "results": summary,
        "protocol_sha256": sha(args.run / "protocol.json"),
        "scorer_sha256": sha(ROOT / "vendor/source/refine.py"),
        "interpretation": "Single fresh construction, same1540 previously exposed questions; "
                          "exact prior memories; content-addressed reader-cache reuse."})
    core.save(args.out / "scored_predictions.json", scores)
    fields = ["label", "f1", "delta_pp", "stored_tokens", "read_tokens"]
    with (args.out / "table.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary[arm] for arm in ARMS)
    tex = [r"\begin{table}[t]", r"\centering\small",
           r"\caption{Evidence-binding ablation on all 1,540 LoCoMo questions with Qwen3.5-9B.",
           r"Differences are relative to the complete method on the same 1,540 questions.",
           r"Stored tokens are averaged over ten histories; read tokens over 1540 questions.",
           r"Both variants use the same frozen memories as the 300-question experiment",
           r"and a 2,048-token evidence budget.}",
           r"\label{tab:exp-ablation-binding1540}", r"\setlength{\tabcolsep}{4pt}",
           r"\renewcommand{\arraystretch}{1.12}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
           r"\textbf{Variant} & \textbf{F1 (\%)} & $\boldsymbol{\Delta}$\textbf{ (pp)} & \textbf{Stored tokens} & \textbf{Read tokens} \\",
           r"\midrule"]
    md = ["| Variant | F1 (%) | Delta (pp) | Stored tokens | Read tokens |",
          "|---|---:|---:|---:|---:|"]
    for i, arm in enumerate(ARMS):
        row = summary[arm]
        label = r"\textbf{Our method}" if arm == "ours" else row["label"]
        tex.append(f'{label} & {row["f1"]:.4f} & {row["delta_pp"]:+.4f} & '
                   f'{row["stored_tokens"]:,.1f} & {row["read_tokens"]:.1f} ' + r"\\")
        if i == 0:
            tex.append(r"\midrule")
        md.append(f'| {row["label"]} | {row["f1"]:.4f} | {row["delta_pp"]:+.4f} | '
                  f'{row["stored_tokens"]:,.1f} | {row["read_tokens"]:.1f} |')
    tex.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}", ""])
    (args.out / "table.tex").write_text("\n".join(tex), encoding="utf-8")
    (args.out / "RESULTS_KO.md").write_text(
        "LoCoMo1540 실측 결과. 저장량은 10대화 평균, 읽기 토큰은 1540문항 평균.\n\n" + "\n".join(md) +
        "\n\n기존 300문항 실험에서 동결한 메모리를 그대로 사용한 전체 1,540문항 결과다. 이전 아카이브의 55.5278과 다른 fresh-memory 비교이며, 두 조건 모두 같은 메모리 구성과 평가 설정을 사용한다.\n",
        encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
