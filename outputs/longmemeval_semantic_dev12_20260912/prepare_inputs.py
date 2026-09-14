"""Freeze existing predictions and exact upstream LongMemEval judge requests."""
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "longmemeval_pilot_baseline_comparison_20260912"
SOURCE = Path("D:/MemoryData/migration_20260910/official_longmemeval/src/evaluation/evaluate_qa.py")
SOURCE_SHA = "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251"
REFERENCE = Path("D:/MemoryData/MemoryData/datasets/LongMemEval/longmemeval_s_cleaned.json")
REFERENCE_SHA = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
MODEL = "gpt-4o-2024-08-06"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_frozen(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen artifact differs: {path.name}")
    else:
        with path.open("xb") as out:
            out.write(data)


def prompt_function() -> Callable[..., str]:
    raw = SOURCE.read_bytes()
    if sha(raw) != SOURCE_SHA:
        raise ValueError("Official evaluator hash mismatch")
    tree = ast.parse(raw.decode("utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "get_anscheck_prompt")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["get_anscheck_prompt"]


def prepare() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    old = read_json(OLD / "baseline_matched_scores.json")
    supplemental = read_json(OLD / "comparison_sources.json")
    ids = old["old12_question_ids"]
    if len(ids) != 12 or len(set(ids)) != 12:
        raise ValueError("Expected exactly the original 12 distinct IDs")
    reference_bytes = REFERENCE.read_bytes()
    if sha(reference_bytes) != REFERENCE_SHA:
        raise ValueError("Canonical dataset hash mismatch")
    references = {r["question_id"]: {k: r[k] for k in
                  ("question_id", "question_type", "question", "answer")}
                  for r in json.loads(reference_bytes) if r["question_id"] in ids}
    if set(references) != set(ids):
        raise ValueError("Missing canonical reference")
    make_prompt = prompt_function()
    rows, bindings = [], []
    for spec in supplemental["own_sources"]:
        path = Path(spec["path"])
        raw = path.read_bytes()
        if sha(raw) != spec["sha256"].lower():
            raise ValueError("Prediction source hash mismatch")
        bindings.append({"path": str(path), "sha256": sha(raw)})
        predictions = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines()]
        if len(predictions) != 12 or {p["question_id"] for p in predictions} != set(ids):
            raise ValueError("Own method coverage mismatch")
        for prediction in predictions:
            ref = references[prediction["question_id"]]
            if (prediction["question"], prediction["gold"], prediction["question_type"]) != (
                    ref["question"], ref["answer"], ref["question_type"]):
                raise ValueError("Question, gold, or type mismatch")
            rows.append({"method": spec["method"], "question_id": ref["question_id"],
                         "hypothesis": prediction["hypothesis"]})
    for method in old["methods"]:
        for prediction in method["rows"]:
            ref = references[prediction["question_id"]]
            if prediction["gold"] != ref["answer"]:
                raise ValueError("Baseline gold mismatch")
            rows.append({"method": method["method"], "question_id": ref["question_id"],
                         "hypothesis": prediction["hypothesis"]})
    for prediction in supplemental["extra_baseline_scores"]:
        ref = references[prediction["question_id"]]
        if prediction["gold"] != ref["answer"]:
            raise ValueError("Extra baseline gold mismatch")
        rows.append({k: prediction[k] for k in ("method", "question_id", "hypothesis")})
    if len(rows) != 135 or len({(r["method"], r["question_id"]) for r in rows}) != 135:
        raise ValueError("Unexpected prediction count or duplicates")
    requests = {}
    for row in rows:
        if not isinstance(row["hypothesis"], str):
            raise ValueError("Hypothesis must be an unmodified string")
        ref = references[row["question_id"]]
        payload = {"model": MODEL, "messages": [{"role": "user", "content": make_prompt(
            ref["question_type"], ref["question"], ref["answer"], row["hypothesis"],
            abstention="_abs" in ref["question_id"])}], "n": 1, "temperature": 0,
            "max_tokens": 10}
        key = sha(canonical(payload).encode("utf-8"))
        row["payload_hash"] = key
        requests[key] = payload
    manifest = {
        "schema": "longmemeval-semantic-dev12-v1", "judge_model": MODEL,
        "endpoint": "https://api.openai.com/v1",
        "upstream_commit": "9e0b455f4ef0e2ab8f2e582289761153549043fc",
        "upstream_source": str(SOURCE), "upstream_sha256": SOURCE_SHA,
        "reference": str(REFERENCE), "reference_sha256": REFERENCE_SHA,
        "expected_question_ids": ids, "rows": len(rows), "unique_requests": len(requests),
        "coverage": dict(sorted(Counter(r["method"] for r in rows).items())),
        "source_bindings": bindings + [
            {"path": str(OLD / name), "sha256": sha((OLD / name).read_bytes())}
            for name in ("baseline_matched_scores.json", "comparison_sources.json")],
        "generation_rerun": False, "answer_postprocessing": "none",
        "prompt_extraction": "unchanged official function compiled from pinned AST",
        "deduplication": "identical complete API payload, including gold and hypothesis",
        "parser": "strict yes/no; invalid responses pending, never scored as no",
        "official_implementation": "same prompt/model/parameters; custom resumable transport",
        "requests_sha256": sha(canonical(requests).encode("utf-8")),
        "inputs_sha256": sha(canonical(rows).encode("utf-8")),
        "baseline_missing_predictions": {
            m: [qid for qid in ids if not any(r["method"] == m and r["question_id"] == qid
                 for r in rows)] for m in ("langmem", "simplemem", "lightmem")},
    }
    for name, value in (("references12.json", list(references.values())),
                        ("inputs.json", rows), ("requests.json", requests),
                        ("protocol.json", manifest)):
        write_frozen(ROOT / name, value)
    print(canonical({"rows": len(rows), "unique_requests": len(requests),
                     "coverage": manifest["coverage"], "api_calls": 0}))
    return rows, requests, manifest


if __name__ == "__main__":
    prepare()
