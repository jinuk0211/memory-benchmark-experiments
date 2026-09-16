"""Prepare one validated full-LoCoMo answer set for the existing mini judge; no API calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from typing import Any

import report_results as report

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "recursive_minilm_20260909/data/locomo10.json"
DATASET_SHA256 = "cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993"
JUDGE_RUNNER = ROOT.parents[1] / "outputs/locomo_gpt4omini_judge_20260911/run_judge.py"
JUDGE_RUNNER_SHA256 = "92e1e70671a5eb4bd96a00a6b96c53dd08b29aa4a026920936eabc2fd313b69f"
PROMPT_SHA256 = "62395dd312a631dfd9355026a0b69cc936018274c3198b6365b5c2a5c9bca9e0"
READER_MODEL = "Qwen/Qwen3.5-9B"
READER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
JUDGE_MODEL = "gpt-4o-mini"
REQUIRED_RETURNED_MODEL = "gpt-4o-mini-2024-07-18"
CATEGORY_COUNTS = {1: 282, 2: 321, 3: 96, 4: 841}


def canonical_rows(dataset: Path) -> dict[str, dict]:
    """Load the unchanged 1,540 category-1--4 canonical questions and raw gold strings."""
    if report.sha(dataset) != DATASET_SHA256:
        raise ValueError("Canonical dataset fingerprint changed")
    rows = {}
    for sample in report.read(dataset):
        cid = str(sample["sample_id"])
        for index, qa in enumerate(sample["qa"]):
            if qa["category"] not in CATEGORY_COUNTS:
                continue
            qid = f"{cid}:{index}"
            if qid in rows:
                raise ValueError("Duplicate canonical question identity")
            rows[qid] = {"id": qid, "conv_id": cid, "qa_index": index,
                         "category": qa["category"], "question": qa["question"],
                         "gold": str(qa["answer"])}
    if (len(rows) != 1540 or len({row["conv_id"] for row in rows.values()}) != 10
            or Counter(row["category"] for row in rows.values()) != CATEGORY_COUNTS):
        raise ValueError("Expected the complete canonical 1,540-question population")
    return dict(sorted(rows.items(), key=lambda item: (item[1]["conv_id"], item[1]["qa_index"])))


def native_request_key(model: dict, system: str, row: dict) -> str:
    """Match the frozen reader's JSON digest, including its unchanged prompt and seed."""
    user = report.reader_request(row["context"], row["question"])
    request = [model, 20260907, system, user, 96, False]
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validated_rows(results: Path, run: Path, arm: str, dataset: Path = DATASET) -> tuple[list[dict], dict]:
    """Recheck selected rows against canonical gold, sealed predictions and native receipts."""
    gold = canonical_rows(dataset)
    result = report.read(results / "RESULTS.json")
    validation = report.read(results / "VALIDATION.json")
    protocol = report.read(run / "protocol.json")
    if (validation.get("status") != "PASS" or result.get("validation") != validation
            or result.get("protocol") != protocol or validation.get("questions_per_arm") != 1540
            or validation.get("histories") != 10
            or validation.get("protocol_sha256") != report.sha(run / "protocol.json")
            or protocol.get("population") != "full1540" or protocol.get("question_count") != 1540
            or protocol.get("max_output_tokens") != 96 or arm not in protocol.get("arms", [])
            or validation.get("arms") != len(protocol["arms"])
            or validation.get("prediction_count") != 1540 * validation["arms"]):
        raise ValueError("Complete validated full1540 report/protocol provenance is required")
    model = protocol.get("environment", {}).get("models", {}).get(READER_MODEL, {})
    if model.get("revision") != READER_REVISION:
        raise ValueError("Unexpected reader model or pinned revision")
    shards = protocol.get("shards")
    if type(shards) is not int or not 1 <= shards <= 10 or not isinstance(protocol.get("reader_system"), str):
        raise ValueError("Invalid reader runtime contract")
    native_fields = [key for key in ("native_receipt_sha256", "native_reader_sha256") if key in validation]
    if len(native_fields) != 1:
        raise ValueError("Expected a round1 or round2 native receipt hash map")
    native_field = native_fields[0]
    if native_field == "native_reader_sha256" and shards != 1:
        raise ValueError("Round2 native receipts require its single reader shard")
    rows = report.read(results / "scored_predictions.json").get(arm, [])
    indexed = {row["id"]: row for row in rows}
    if len(rows) != 1540 or len(indexed) != 1540 or set(indexed) != set(gold):
        raise ValueError("Selected arm must contain all 1,540 unique canonical question IDs")
    histories = sorted({row["conv_id"] for row in gold.values()})
    prediction_hashes, native_hashes = {}, {}
    for cid in histories:
        path = run / "predictions" / arm / f"{cid}.json"
        name = f"{arm}/{cid}"
        if report.sha(path) != validation.get("prediction_file_sha256", {}).get(name):
            raise ValueError("Original prediction file differs from the validated report")
        original_rows = report.read(path)
        expected_ids = [qid for qid, reference in gold.items() if reference["conv_id"] == cid]
        if [row["id"] for row in original_rows] != expected_ids:
            raise ValueError("Original prediction population is incomplete or reordered")
        prediction_hashes[name] = report.sha(path)
        shard = histories.index(cid) % shards
        for original in original_rows:
            row = indexed[original["id"]]
            if (any(row.get(key) != value for key, value in gold[row["id"]].items())
                    or type(row.get("qa_index")) is not int or type(row.get("category")) is not int
                    or row.get("arm") != arm):
                raise ValueError("Question, identity, category, gold or arm differs from canonical input")
            if {key: value for key, value in row.items() if key not in {"gold", "official_f1"}} != original:
                raise ValueError("Scored prediction changed the original answer or metadata")
            score = row.get("official_f1")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid validated F1 value")
            key = native_request_key(model, protocol["reader_system"], row)
            receipt_path = run / f"evaluate_{shard}/cache/generations" / f"{key}.json"
            receipt_name = f"evaluate_{shard}/{key}" if native_field == "native_receipt_sha256" else key
            if report.sha(receipt_path) != validation[native_field].get(receipt_name):
                raise ValueError("Native reader receipt fingerprint changed")
            report.validate_receipt(row, report.read(receipt_path), key)
            native_hashes[receipt_name] = report.sha(receipt_path)
    summary = result.get("results", {}).get(arm, {})
    if (summary.get("n") != 1540 or summary.get("histories") != 10
            or not math.isclose(summary.get("f1", -1), statistics.mean(row["official_f1"] for row in rows) * 100,
                                rel_tol=0, abs_tol=1e-9)):
        raise ValueError("Selected answer set does not match its validated result aggregate")
    provenance = {"results_directory": str(results.resolve()), "run_directory": str(run.resolve()),
                  "files_sha256": {name: report.sha(results / name) for name in
                                   ("RESULTS.json", "VALIDATION.json", "scored_predictions.json")},
                  "protocol_sha256": report.sha(run / "protocol.json"),
                  "prediction_file_sha256": prediction_hashes, "native_receipt_sha256": native_hashes}
    return [indexed[qid] for qid in gold], provenance


def load_judge() -> Any:
    """Import the existing runner without invoking its CLI or making an API request."""
    if report.sha(JUDGE_RUNNER) != JUDGE_RUNNER_SHA256:
        raise ValueError("Existing judge runner fingerprint changed")
    spec = importlib.util.spec_from_file_location("saved_locomo_mini_judge", JUDGE_RUNNER)
    if spec is None or spec.loader is None:
        raise ValueError("Existing mini judge runner is unavailable")
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    if judge.MODEL != JUDGE_MODEL or judge.digest(judge.load_prompt().encode("utf-8")) != PROMPT_SHA256:
        raise ValueError("Existing judge model or exact official prompt changed")
    return judge


def prepare(results: Path, run: Path, arm: str, output: Path, dataset: Path = DATASET) -> dict:
    """Write a new immutable input set only after all source checks succeed."""
    source, provenance = validated_rows(results, run, arm, dataset)
    judge = load_judge()
    rows = [{"id": row["id"], "method": arm, "conversation_id": row["conv_id"],
             "qa_index": row["qa_index"], "category": row["category"], "question": row["question"],
             "gold_answer": row["gold"], "generated_answer": row["prediction"],
             "official_f1": row["official_f1"]} for row in source]
    judge.validate_rows(rows)
    raw = "".join(judge.canonical_json(row) + "\n" for row in rows).encode("utf-8")
    manifest = {"status": "PREPARED_NO_API_CALLS", "method": arm, "rows": 1540, "histories": 10,
                "categories": dict(Counter(row["category"] for row in rows)),
                "input_sha256": judge.digest(raw), "prompt_sha256": PROMPT_SHA256,
                "model": JUDGE_MODEL, "required_returned_model": REQUIRED_RETURNED_MODEL,
                "api_settings": {"temperature": 0.0, "response_format": {"type": "json_object"},
                                 "max_tokens": "omitted as upstream"},
                "gold_preprocessing": "str(canonical answer); no category-specific F1 preprocessing",
                "source_predictions_modified": False, "source_validation": provenance,
                "dataset_path": str(dataset.resolve()), "dataset_sha256": report.sha(dataset),
                "judge_runner_path": str(JUDGE_RUNNER), "judge_runner_sha256": report.sha(JUDGE_RUNNER),
                "prices_per_million_usd": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
                "price_source": "https://developers.openai.com/api/docs/models/gpt-4o-mini",
                "price_checked_date": "2026-09-14"}
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; do not replace prepared or judged inputs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "input.jsonl").write_bytes(raw)
    (output / "official_accuracy_prompt.txt").write_bytes(judge.load_prompt().encode("utf-8"))
    report.write_json(output / "input_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", default="ours")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.results, args.run, args.arm, args.output_dir)
    print(json.dumps({"status": manifest["status"], "rows": manifest["rows"],
                      "model": manifest["model"], "input_sha256": manifest["input_sha256"]}))


if __name__ == "__main__":
    main()
