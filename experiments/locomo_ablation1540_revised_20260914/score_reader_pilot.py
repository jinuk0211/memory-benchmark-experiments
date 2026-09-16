"""Validate both frozen reader pilots and commit the predeclared reader choice."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import statistics

import report_results as report

ROOT = Path(__file__).resolve().parent


def choose_reader(results: dict) -> str:
    if set(results) != {"legacy", "grounded"} or any(
            result["n"] != 300 or not 0 <= result["f1"] <= 100 for result in results.values()):
        raise ValueError("Both complete 300-question reader results are required")
    return "grounded" if results["grounded"]["f1"] > results["legacy"]["f1"] else "legacy"


def canonical_gold(inputs: Path, manifest: dict) -> dict:
    dataset_path = ROOT.parent / "recursive_minilm_20260909/data/locomo10.json"
    if report.sha(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Canonical dataset fingerprint changed")
    rows = []
    for sample in report.read(dataset_path):
        cid = str(sample["sample_id"])
        for index, qa in enumerate(sample["qa"]):
            if int(qa["category"]) in report.CATEGORIES:
                rows.append({"id": f"{cid}:{index}", "conv_id": cid, "qa_index": index,
                             "category": int(qa["category"]), "question": qa["question"],
                             "gold": str(qa["answer"])})
    rows.sort(key=lambda row: (row["conv_id"], row["qa_index"]))
    if len(rows) != 1540 or report.read(ROOT / "evaluation/gold.json") != rows:
        raise ValueError("Local evaluation gold differs from all canonical 1,540 labels")
    questions = [{key: value for key, value in row.items() if key != "gold"} for row in rows]
    if report.read(inputs / "reader_questions.json") != questions:
        raise ValueError("Full reader question identities differ from canonical gold")
    return {row["id"]: row for row in rows}


def score_rows(rows: list[dict], gold: dict, expected_ids: set[str],
               f1: Callable[[str, str, int], float]) -> tuple[list[dict], dict]:
    if len(rows) != 300 or len({row["id"] for row in rows}) != 300 or {
            row["id"] for row in rows} != expected_ids:
        raise ValueError("Expected all 300 unique frozen pilot questions")
    scored = []
    for row in rows:
        reference = gold[row["id"]]
        if (not isinstance(row["prediction"], str)
                or any(row[key] != reference[key] for key in ("conv_id", "category", "question"))):
            raise ValueError("Pilot question identity or response type changed")
        scored.append({**row, "gold": reference["gold"],
                       "official_f1": f1(row["prediction"], reference["gold"], row["category"])})
    result = {"n": len(scored), "f1": statistics.mean(row["official_f1"] for row in scored) * 100,
              "category_f1": {str(category): statistics.mean(
                  row["official_f1"] for row in scored if row["category"] == category) * 100
                  for category in sorted({row["category"] for row in scored})},
              "category_counts": dict(Counter(str(row["category"]) for row in scored)),
              "finish_reasons": dict(Counter(row["finish_reason"] for row in scored)),
              "empty_answers": sum(not row["prediction"].strip() for row in scored),
              "unknown_answers": sum(row["prediction"].strip().casefold().rstrip(".") == "unknown"
                                     for row in scored),
              "read_tokens": statistics.mean(row["read_tokens"] for row in scored)}
    return scored, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "package/input")
    parser.add_argument("--pilot", type=Path, default=ROOT / "collected/reader_pilot")
    parser.add_argument("--pilot-inputs", type=Path, default=ROOT / "package/pilot_input")
    parser.add_argument("--tokenizer", type=Path, default=ROOT.parents[1] / "model_staging/qwen35_9b_c202236/tokenizer.json")
    parser.add_argument("--selection", type=Path, default=ROOT / "READER_SELECTION.json")
    parser.add_argument("--out", type=Path, default=ROOT / "pilot_analysis")
    args = parser.parse_args()

    code = args.inputs.parent / "code"
    manifest = report.read(args.inputs / "manifest.json")
    for name, digest in manifest["files"].items():
        if report.sha(args.inputs / name) != digest:
            raise ValueError(f"Frozen input fingerprint mismatch: {name}")
    for name, digest in manifest["code"].items():
        if report.sha(code / name) != digest:
            raise ValueError(f"Frozen code fingerprint mismatch: {name}")
    runner = report.load_runner(code)
    import nltk
    from nltk.stem import porter
    if nltk.__version__ != "3.9.1":
        raise ValueError("Pilot scoring requires pinned nltk==3.9.1")
    gold = canonical_gold(args.inputs, manifest)
    expected_ids = {row["id"] for row in report.read(args.inputs / "dev300_questions.json")}
    scored, results = {}, {}
    for reader in ("legacy", "grounded"):
        scored[reader], results[reader] = score_rows(
            report.read(args.pilot / f"{reader}.json"), gold, expected_ids, runner.core.f1)
    chosen = choose_reader(results)
    protocol_sha = report.sha(args.pilot / "protocol.json")
    selection = {
        "candidate_configurations": ["legacy", "grounded"], "population": "dev300", "arms": ["ours"],
        "criterion": "highest_overall_f1_tie_legacy", "results": results, "chosen_reader": chosen,
        "frozen_before_full1540": True,
        "dev_protocol_sha256": {reader: protocol_sha for reader in results},
        "dev_prediction_sha256": {reader: report.sha(args.pilot / f"{reader}.json") for reader in results},
        "interpretation": "Finite predeclared reader selection on the previously exposed 300-question subset; "
                          "all 300 remain in the 1,540-question exploratory evaluation.",
        "diagnostic_policy": "Category scores, empty/length responses and exact unknown-answer counts do not affect selection.",
        "scorer_sha256": report.sha(code / "vendor/source/refine.py"), "nltk_version": nltk.__version__,
        "porter_sha256": report.sha(Path(porter.__file__)),
    }
    report.validate_selection(selection, chosen)
    tokenizer = runner.TokenRuntime(args.tokenizer, manifest["tokenizer_sha256"])
    validation = report.validate_pilot(args, runner, gold, tokenizer, selection)
    if args.selection.exists():
        previous = report.read(args.selection)
        created = previous["created_at_utc"]
    else:
        created = datetime.now(timezone.utc).isoformat()
    selection["created_at_utc"] = created
    # The caller commits this record before starting any full-set evaluation.
    # Existing choices are immutable; later data cannot silently replace them.
    runner.frozen_save(args.selection, selection)
    args.out.mkdir(parents=True, exist_ok=True)
    report.write_json(args.out / "SCORED_PILOT.json", scored)
    report.write_json(args.out / "VALIDATION.json", validation)
    report.write_csv(args.out / "per_question_scores.csv",
                     ["reader", "id", "conv_id", "category", "question", "gold", "prediction",
                      "official_f1", "finish_reason", "read_tokens", "input_tokens", "output_tokens",
                      "generation_cache_key"],
                     [row for reader in ("legacy", "grounded") for row in scored[reader]])
    report.write_json(args.out / "SUMMARY.json", selection)
    print({"chosen_reader": chosen, "legacy_f1": results["legacy"]["f1"],
           "grounded_f1": results["grounded"]["f1"], "validated_responses": 600,
           "selection_sha256": report.sha(args.selection)})


if __name__ == "__main__":
    main()
