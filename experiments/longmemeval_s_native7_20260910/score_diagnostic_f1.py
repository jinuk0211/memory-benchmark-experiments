"""CPU-only diagnostic token F1 for verified full-500 LongMemEval-S exports."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
METHODS = ("e_mem", "simplemem", "langmem", "mem0", "a_mem", "lightmem", "higmem")
METRIC_VERSION = "generalization_20260908.refine.generic_f1.v1"
ORIGINAL_SOURCE_SHA256 = "71384cac6aaa260d6df4f24f605797a8a886cd51514a83c9c1d041eff99f14fa"


# Copied verbatim from generalization_20260908/source/refine.py:98.
def generic_f1(pred, gold):
    def norm(x):
        return [w for w in re.sub(r'[^a-z0-9 ]', ' ', str(x).lower()).split()
                if w not in {'a', 'an', 'the'}]
    p, g = norm(pred), norm(gold)
    common = sum((Counter(p) & Counter(g)).values())
    return 2 * common / (len(p) + len(g)) if common else 0.0


def file_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mean = sum(row["diagnostic_token_f1"] for row in rows) / len(rows) if rows else None
    return {"count": len(rows), "question_macro_f1": mean,
            "question_macro_f1_100": mean * 100 if mean is not None else None}


def score(dataset: Path, hypotheses: Path, output: Path) -> dict[str, Any]:
    """Read gold/type only here; never modify generation artifacts or call a model."""
    if output.exists():
        raise FileExistsError("Existing scores are preserved; choose a fresh output path")
    receipt_path = hypotheses.with_name(hypotheses.name + ".receipt.json")
    inputs = {path: path.read_bytes() for path in (dataset, hypotheses, receipt_path)}
    if file_hash(inputs[dataset]) != DATA_SHA256:
        raise ValueError("Canonical LongMemEval-S cleaned SHA-256 mismatch")
    records = json.loads(inputs[dataset])
    if not isinstance(records, list) or len(records) != 500:
        raise ValueError("Expected 500 canonical reference records")
    ids = [row["question_id"] for row in records]
    if any(not isinstance(qid, str) or not qid for qid in ids) or len(set(ids)) != 500:
        raise ValueError("Reference IDs must be 500 unique nonempty strings")
    by_id = {row["question_id"]: row for row in records}
    receipt = json.loads(inputs[receipt_path])
    if (receipt.get("schema") != "native-seven-official-hypotheses-v1"
            or receipt.get("scope") != "full_canonical_500"
            or receipt.get("expected_count") != 500
            or receipt.get("expected_question_ids") != ids
            or receipt.get("dataset_sha256") != DATA_SHA256
            or receipt.get("hypotheses_sha256") != file_hash(inputs[hypotheses])
            or receipt.get("answer_postprocessing") != "none"
            or receipt.get("method") not in METHODS):
        raise ValueError("A matching export_official.py full-500 receipt is required")
    rows = []
    predictions = {}
    lines = inputs[hypotheses].decode("utf-8").splitlines()
    if len(lines) != 500:
        raise ValueError("Hypotheses must contain exactly 500 JSONL records")
    for line in lines:
        prediction = json.loads(line)
        if not isinstance(prediction, dict) or set(prediction) != {"question_id", "hypothesis"}:
            raise ValueError("Prediction records must contain only question_id and hypothesis")
        qid, answer = prediction["question_id"], prediction["hypothesis"]
        if not isinstance(qid, str) or qid in predictions or qid not in by_id:
            raise ValueError("Duplicate, unknown or malformed prediction ID")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Blank or invalid predictions cannot be scored as a complete run")
        predictions[qid] = answer
    if set(predictions) != set(ids):
        raise ValueError("Prediction coverage differs from all 500 canonical IDs")
    for qid in ids:
        reference = by_id[qid]
        category = reference["question_type"]
        if not isinstance(category, str) or not category:
            raise ValueError("Missing canonical question type")
        rows.append({"question_id": qid, "question_type": category,
                     "is_abstention": qid.endswith("_abs"),
                     "diagnostic_token_f1": generic_f1(predictions[qid], reference["answer"])})
    summary = {
        "overall": describe(rows),
        "by_question_type": {category: describe([row for row in rows if row["question_type"] == category])
                             for category in sorted({row["question_type"] for row in rows})},
        "by_answerability": {name: describe([row for row in rows if row["is_abstention"] == value])
                             for name, value in (("answerable", False), ("abstention", True))},
        "abstention_included_in_overall": True,
        "overall_weighting": "Equal weight per question, not an unweighted average of question-type means",
    }
    result = {
        "schema": "longmemeval-s-diagnostic-token-f1-v1", "method": receipt["method"],
        "metric_version": METRIC_VERSION, "official_accuracy": None,
        "metric_definition": "Lowercase ASCII alphanumeric tokens; remove a/an/the; multiset overlap F1; zero when overlap is zero",
        "metric_origin": {"file": "generalization_20260908/source/refine.py", "function": "generic_f1",
                          "file_sha256": ORIGINAL_SOURCE_SHA256, "reuse": "verbatim standalone function copy"},
        "limitations": "Diagnostic lexical overlap, not official judge accuracy or an abstention correctness test; no stemming or semantic matching",
        "abstention_rule": "question_id ends with _abs, matching the existing transfer_data.py convention",
        "summary": summary, "questions": rows,
        "provenance": {"inputs_sha256": {str(path): file_hash(raw) for path, raw in inputs.items()},
                       "scorer_sha256": file_hash(Path(__file__).read_bytes()),
                       "external_api_calls": 0, "judge_calls": 0,
                       "generation_artifacts_modified": False},
    }
    for path, raw in inputs.items():
        if file_hash(path.read_bytes()) != file_hash(raw):
            raise ValueError(f"Input changed during scoring: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True,
                        help="export_official.py JSONL; matching .receipt.json is required")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.dataset, args.hypotheses, args.output)
    print(json.dumps({"output": str(args.output), "method": result["method"],
                      "diagnostic_summary": result["summary"], "official_accuracy": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
