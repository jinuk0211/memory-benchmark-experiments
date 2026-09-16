"""Resumable full-context LoCoMo runner using the archived full_raw protocol."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from openai import OpenAI

READER_SYSTEM = (
    "You answer questions about a long conversation using only the memory notes provided. "
    "Reply with the shortest exact answer only: a name, a date written like '7 May 2023', a number "
    "written in digits, or a short noun phrase. No sentence, no explanation. "
    "If the notes do not contain the answer, reply: unknown"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def build_context(sample: dict[str, Any]) -> str:
    conversation = sample["conversation"]
    rows = []
    session_keys = [
        key for key in conversation
        if re.fullmatch(r"session_\d+", key) and isinstance(conversation[key], list)
    ]
    session_keys.sort(key=lambda key: int(key.split("_")[1]))
    for key in session_keys:
        number = int(key.split("_")[1])
        date = conversation.get(f"session_{number}_date_time", "")
        for turn in conversation[key]:
            rows.append(f"[session {number}] ({date}) {turn['speaker']}: {turn['text']}")
    return "\n".join(rows)


def score_and_report(dataset: list[dict[str, Any]], rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "HiGMem"))
    from official_locomo_evaluation import eval_question_answering

    expected = {
        (str(sample["sample_id"]), index)
        for sample in dataset
        for index, qa in enumerate(sample["qa"])
        if int(qa["category"]) in (1, 2, 3, 4)
    }
    unique = {(row["sample"], int(row["index"])): row for row in rows}
    if len(unique) != len(rows):
        raise RuntimeError("Duplicate full-context predictions")
    if set(unique) - expected:
        raise RuntimeError("Prediction outside canonical population")
    ordered = [unique[key] for key in sorted(unique)]
    scores, _, _ = eval_question_answering(ordered) if ordered else ([], None, None)
    by_category: dict[int, list[float]] = defaultdict(list)
    scored = []
    for row, value in zip(ordered, scores):
        item = {**row, "official_f1": float(value)}
        scored.append(item)
        by_category[int(row["category"])].append(float(value))
    report = {
        "method": "full-context",
        "expected": len(expected),
        "evaluated": len(ordered),
        "complete": set(unique) == expected,
        "official_f1": sum(scores) / len(scores) if scores else None,
        "official_f1_x100": 100 * sum(scores) / len(scores) if scores else None,
        "by_category": {
            str(category): {"count": len(values), "f1_x100": 100 * sum(values) / len(values)}
            for category, values in sorted(by_category.items())
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "scored_predictions.json").write_text(
        json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemma-3-12b-it")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    raw = args.dataset.read_bytes()
    dataset = json.loads(raw)
    population = [
        (sample, index, qa)
        for sample in dataset
        for index, qa in enumerate(sample["qa"])
        if int(qa["category"]) in (1, 2, 3, 4)
    ]
    if len(population) != 1540:
        raise RuntimeError(f"Expected 1540 questions, found {len(population)}")
    if args.max_questions:
        population = population[: args.max_questions]

    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": "full-context",
        "model": args.model,
        "checkpoint": os.environ.get(
            "LOCOMO_CHECKPOINT",
            "ggml-org/gemma-3-12b-it-GGUF/gemma-3-12b-it-f16.gguf",
        ),
        "model_sha256": os.environ.get("LOCOMO_MODEL_SHA256", "unknown"),
        "precision": "F16",
        "kv_cache": "F16",
        "server_context_size": 49152,
        "base_url": args.base_url,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "reader_system": READER_SYSTEM,
        "max_answer_tokens": 32,
        "temperature": 0,
        "categories": [1, 2, 3, 4],
        "context_serialization": "[session N] (date) speaker: text",
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Full-context protocol changed; use a new output directory")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    predictions_path = args.output / "predictions.jsonl"
    rows = read_jsonl(predictions_path)
    done = {(row["sample"], int(row["index"])) for row in rows}
    if not args.report_only:
        client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1200, max_retries=0)
        contexts = {str(sample["sample_id"]): build_context(sample) for sample in dataset}
        for position, (sample, index, qa) in enumerate(population, 1):
            key = (str(sample["sample_id"]), index)
            if key in done:
                continue
            prompt = f"Memory notes:\n{contexts[key[0]]}\n\nQuestion: {qa['question']}\nAnswer:"
            error = None
            for attempt in range(3):
                started = time.time()
                try:
                    result = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": READER_SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                        max_tokens=32,
                    )
                    prediction = (result.choices[0].message.content or "").strip()
                    usage = result.usage.model_dump() if result.usage else None
                    row = {
                        "sample": key[0],
                        "index": index,
                        "question": qa["question"],
                        "category": int(qa["category"]),
                        "answer": qa["answer"],
                        "prediction": prediction,
                        "finish_reason": result.choices[0].finish_reason,
                        "usage": usage,
                        "seconds": time.time() - started,
                    }
                    append_jsonl(predictions_path, row)
                    done.add(key)
                    print(f"FULL_CONTEXT {position}/{len(population)} {key[0]}:{index}", flush=True)
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    print(f"FULL_CONTEXT_RETRY {key[0]}:{index} attempt={attempt + 1} error={exc}", flush=True)
                    time.sleep(2 ** attempt)
            if error is not None:
                raise error
        client.close()

    report = score_and_report(dataset, read_jsonl(predictions_path), args.output)
    print(json.dumps(report, indent=2), flush=True)
    if not args.max_questions and not report["complete"]:
        raise SystemExit("Full-context run incomplete")


if __name__ == "__main__":
    main()