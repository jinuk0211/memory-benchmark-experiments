"""Canonicalize raw LoCoMo outputs and compute the pinned official token F1."""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import sys
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_index(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, str], tuple[int, dict[str, Any]]]]:
    samples = json.loads(path.read_text(encoding="utf-8-sig"))
    by_index: dict[tuple[str, int], dict[str, Any]] = {}
    by_question_id: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        for index, qa in enumerate(sample["qa"]):
            if int(qa["category"]) not in (1, 2, 3, 4):
                continue
            by_index[(sample_id, index)] = qa
            question_id = str(qa.get("question_id") or f"{sample_id}_qa{index}")
            by_question_id[(sample_id, question_id)] = (index, qa)
    if len(by_index) != 1540:
        raise ValueError(f"Canonical dataset has {len(by_index)} eligible questions, expected 1540")
    return by_index, by_question_id


def canonical_row(sample: str, index: int, qa: dict[str, Any], prediction: str, **extra: Any) -> dict[str, Any]:
    if not isinstance(prediction, str):
        raise ValueError("Prediction is not a string")
    return {
        "sample": sample,
        "index": index,
        "question": qa["question"],
        "answer": qa["answer"],
        "category": qa["category"],
        "evidence": qa["evidence"],
        "prediction": prediction,
        **extra,
    }


def common_rows(root: Path, dataset: Path, expected: dict[tuple[str, int], dict[str, Any]],
                by_question_id: dict[tuple[str, str], tuple[int, dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[Path]]:
    files = sorted(root.rglob("*_results.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one raw result JSON below {root}, found {len(files)}")
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    memorydata = Path(__file__).resolve().parent / "MemoryData"
    sys.path.insert(0, str(memorydata))
    from benchmark.memoryagentbench.prompts.benchmark_templates import get_template
    template = get_template(
        payload["dataset_config"]["sub_dataset"],
        "query",
        payload["agent_config"]["agent_name"],
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for record in payload["data"]:
        if record.get("status") == "failed":
            raise ValueError("Raw common-baseline result contains a failed query")
        metadata = record.get("eval_metadata") or {}
        lookup_key = (str(metadata.get("sample_id")), str(metadata.get("question_id")))
        if lookup_key not in by_question_id or metadata.get("dataset") != "locomo_qa":
            raise ValueError(f"Unknown QA identity: {lookup_key}")
        index, qa = by_question_id[lookup_key]
        key = (lookup_key[0], index)
        if key in seen:
            raise ValueError(f"Duplicate QA identity: {key}")
        seen.add(key)
        recorded_question = record.get("query")
        valid_questions = {qa["question"], template.format(question=qa["question"])}
        if recorded_question not in valid_questions:
            raise ValueError(f"Recorded query does not match pinned template: {key}")
        answer = record.get("answer")
        if isinstance(answer, list) and len(answer) == 1 and not isinstance(qa["answer"], list):
            answer = answer[0]
        if (
            str(answer) != str(qa["answer"])
            or str(metadata.get("category")) != str(qa["category"])
            or metadata.get("evidence") != qa["evidence"]
        ):
            raise ValueError(f"Raw metadata differs from canonical QA: {key}")
        rows.append(canonical_row(key[0], index, qa, record.get("output"), seconds=record.get("query_time_len")))
    rows.sort(key=lambda row: list(expected).index((row["sample"], row["index"])))
    return rows, files


def ours_rows(root: Path, expected: dict[tuple[str, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Path]]:
    files = sorted(root.glob("*.json"))
    if not files:
        raise ValueError(f"No ours predictions found below {root}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for path in files:
        for record in json.loads(path.read_text(encoding="utf-8")):
            key = (str(record.get("conv_id")), int(record.get("qa_index")))
            if key not in expected or key in seen:
                raise ValueError(f"Unknown or duplicate ours QA identity: {key}")
            seen.add(key)
            qa = expected[key]
            if record.get("question") != qa["question"] or int(record.get("category")) != int(qa["category"]):
                raise ValueError(f"Ours metadata differs from canonical QA: {key}")
            rows.append(canonical_row(key[0], key[1], qa, record.get("prediction"),
                                      read_tokens=record.get("read_tokens"),
                                      stored_tokens=record.get("stored_tokens")))
    order = {key: position for position, key in enumerate(expected)}
    rows.sort(key=lambda row: order[(row["sample"], row["index"])])
    return rows, files


def score(rows: list[dict[str, Any]], scorer: Path) -> tuple[list[float], dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("official_locomo_fp16", scorer)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load official scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()):
        values, _, _ = module.eval_question_answering(rows, eval_key="prediction", metric="f1")
    scores = [float(value) for value in values]
    if len(scores) != len(rows):
        raise ValueError("Official scorer returned the wrong number of rows")
    by_category: dict[int, list[float]] = defaultdict(list)
    for row, value in zip(rows, scores):
        row["official_f1"] = value
        by_category[int(row["category"])].append(value)
    return scores, {
        str(category): {
            "n": len(by_category[category]),
            "f1": statistics.mean(by_category[category]) if by_category[category] else None,
        }
        for category in (1, 2, 3, 4)
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("common", "ours"))
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1540)
    args = parser.parse_args()

    expected, by_question_id = dataset_index(args.dataset)
    rows, sources = (
        common_rows(args.raw, args.dataset, expected, by_question_id)
        if args.kind == "common"
        else ours_rows(args.raw, expected)
    )
    if len(rows) != args.expected_count:
        raise ValueError(f"Found {len(rows)} predictions, expected {args.expected_count}")
    scores, categories = score(rows, args.scorer)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = args.output / "scored_predictions.json"
    write_json(predictions, rows)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "complete": len(rows) == args.expected_count,
        "kind": args.kind,
        "n": len(rows),
        "official_f1": statistics.mean(scores),
        "by_category": categories,
        "empty_predictions": sum(not row["prediction"].strip() for row in rows),
        "dataset_sha256": sha256(args.dataset),
        "scorer_sha256": sha256(args.scorer),
        "raw_sources": [{"path": str(path), "sha256": sha256(path)} for path in sources],
        "prediction_field": "raw model output; no gold-dependent answer selection",
        "scored_predictions_sha256": sha256(predictions),
    }
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()