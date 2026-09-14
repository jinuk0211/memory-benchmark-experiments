"""Freeze score-blind common50 and the exact official GPT-4o-mini payloads."""
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-4o-mini-2024-07-18"
SOURCE = Path("D:/MemoryData/migration_20260910/official_longmemeval/src/evaluation/evaluate_qa.py")
SOURCE_SHA = "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251"
PREDICTIONS = Path("D:/MemoryData/outputs/longmemeval_overlap_latest_20260912/common_baseline_predictions.json")
PREDICTIONS_SHA = "f21dcd996f10577b9611e39dbef73c65a90df489eb0ace95cd98067b5f3f368e"
REFERENCE = Path("D:/MemoryData/MemoryData/datasets/LongMemEval/longmemeval_s_cleaned.json")
REFERENCE_SHA = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
METHODS = ("full_context", "langmem", "simplemem", "lightmem")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def freeze(name: str, value: Any) -> None:
    path = ROOT / name
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Frozen artifact differs: " + name)
    else:
        with path.open("xb") as out:
            out.write(raw)


def prepare() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    prediction_bytes = PREDICTIONS.read_bytes()
    reference_bytes = REFERENCE.read_bytes()
    source_bytes = SOURCE.read_bytes()
    if (sha(prediction_bytes), sha(reference_bytes), sha(source_bytes)) != (
            PREDICTIONS_SHA, REFERENCE_SHA, SOURCE_SHA):
        raise ValueError("Prediction, canonical dataset, or official source hash mismatch")
    predictions = json.loads(prediction_bytes)
    reference_all = json.loads(reference_bytes)
    indexed = {}
    for method in METHODS:
        method_rows = predictions[method]
        indexed[method] = {r["question_id"]: r for r in method_rows}
        if len(method_rows) != 52 or len(indexed[method]) != 52:
            raise ValueError("Expected52 distinct completed predictions")
        if any(not isinstance(r["hypothesis"], str) or not r["hypothesis"].strip()
               for r in method_rows):
            raise ValueError("Incomplete prediction")
    common = set.intersection(*(set(indexed[m]) for m in METHODS))
    if len(common) != 52:
        raise ValueError("Methods must share exactly52 IDs")
    eligible = [r for r in reference_all if r["question_id"] in common]
    if len(eligible) != 52:
        raise ValueError("Canonical dataset ID mismatch")
    selected = eligible[:50]
    selection = {
        "rule": "First50 of shared52 in canonical dataset order; no answers or scores used",
        "reference_sha256": REFERENCE_SHA, "common_count": 52, "selected_count": 50,
        "question_ids": [r["question_id"] for r in selected],
        "excluded_question_ids": [r["question_id"] for r in eligible[50:]],
        "question_type_counts": dict(Counter(r["question_type"] for r in selected)),
    }
    freeze("selection.json", selection)
    function = next(n for n in ast.parse(source_bytes.decode("utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "get_anscheck_prompt")
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    prompt = namespace["get_anscheck_prompt"]
    references = [{k: r[k] for k in ("question_id", "question_type", "question", "answer")}
                  for r in selected]
    rows, requests = [], {}
    for method in METHODS:
        for ref in references:
            hypothesis = indexed[method][ref["question_id"]]["hypothesis"]
            payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt(
                ref["question_type"], ref["question"], ref["answer"], hypothesis,
                abstention="_abs" in ref["question_id"])}],
                "n": 1, "temperature": 0, "max_tokens": 10}
            key = sha(canonical(payload).encode("utf-8"))
            rows.append({"method": method, "question_id": ref["question_id"],
                         "hypothesis": hypothesis, "payload_hash": key})
            requests[key] = payload
    protocol = {
        "schema": "longmemeval-common50-gpt4omini-v1", "model": MODEL,
        "selection": selection, "source_predictions": str(PREDICTIONS),
        "source_predictions_sha256": PREDICTIONS_SHA,
        "reference": str(REFERENCE), "reference_sha256": REFERENCE_SHA,
        "upstream_source": str(SOURCE), "upstream_sha256": SOURCE_SHA,
        "upstream_commit": "9e0b455f4ef0e2ab8f2e582289761153549043fc",
        "generation_rerun": False, "answer_postprocessing": "none",
        "judge_parser": "exact official expression: 'yes' in stripped response.lower(); only complete yes/no with optional terminal period accepted",
        "deduplication": "identical full API payloads share one verdict; no GPT-4o cache reused",
        "endpoint": "https://api.openai.com/v1", "methods": list(METHODS),
        "input_rows": len(rows), "unique_requests": len(requests),
        "inputs_sha256": sha(canonical(rows).encode("utf-8")),
        "requests_sha256": sha(canonical(requests).encode("utf-8")),
    }
    for name, value in (("references50.json", references), ("inputs.json", rows),
                        ("requests.json", requests), ("protocol.json", protocol)):
        freeze(name, value)
    print(canonical({"input_rows": len(rows), "unique_requests": len(requests),
                     "selection": selection, "api_calls_this_prepare": 0}), flush=True)
    return rows, requests, protocol


if __name__ == "__main__":
    prepare()
