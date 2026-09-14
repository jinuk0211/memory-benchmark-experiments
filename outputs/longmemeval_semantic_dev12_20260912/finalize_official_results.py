"""Apply the pinned upstream verdict expression to preserved GPT-4o responses."""
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_inputs import MODEL, ROOT, SOURCE, SOURCE_SHA, canonical, read_json
from run_semantic_judge import summarize


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize() -> dict[str, Any]:
    protocol = read_json(ROOT / "protocol.json")
    inputs = read_json(ROOT / "inputs.json")
    requests = read_json(ROOT / "requests.json")
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError("Pinned official evaluator changed")
    for data, field in ((inputs, "inputs_sha256"), (requests, "requests_sha256")):
        if hashlib.sha256(canonical(data).encode("utf-8")).hexdigest() != protocol[field]:
            raise ValueError("Frozen input or requests changed")
    for key, payload in requests.items():
        if hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest() != key:
            raise ValueError("Request payload digest mismatch")
    journal = ROOT / "judge_responses.jsonl"
    receipts = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    keys = [row["payload_hash"] for row in receipts]
    if len(keys) != len(set(keys)) or set(keys) != set(requests):
        raise ValueError("Need exactly one preserved response for every frozen request")
    cache: dict[str, Any] = {}
    for receipt in receipts:
        content = receipt.get("content")
        if (receipt.get("returned_model") != MODEL
                or receipt.get("finish_reason") != "stop"
                or not isinstance(content, str)
                or content.strip().lower() not in {"yes", "no", "yes.", "no."}):
            raise ValueError("Incomplete model response or unsupported text")
        usage = receipt.get("usage")
        if not isinstance(usage, dict) or any(
            type(usage.get(field)) is not int or usage[field] < 0
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise ValueError("Missing or invalid usage")
        if usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
            raise ValueError("Token usage does not add up")
        # Exact evaluate_qa.py expression: eval_response is stripped first upstream.
        label = "yes" in content.strip().lower()
        cache[receipt["payload_hash"]] = {**receipt, "label": label}
    result = summarize(inputs, cache)
    refs = {ref["question_id"]: ref for ref in read_json(ROOT / "references12.json")}
    manual = read_json(ROOT / "manual_judgments.json")
    manual_by_key = {(r["method"], r["question_id"]): r for r in manual["rows"]}
    disagreements = []
    for row in result["scored_rows"]:
        row["question_type"] = refs[row["question_id"]]["question_type"]
        row["raw_verdict"] = cache[row["payload_hash"]]["content"]
        key = (row["method"], row["question_id"])
        previous = manual_by_key.get(key)
        if previous and previous["correct"] != row["correct"]:
            disagreements.append({
                "method": row["method"], "question_id": row["question_id"],
                "manual_correct": previous["correct"], "gpt4o_correct": row["correct"],
                "question": refs[row["question_id"]]["question"],
                "gold": refs[row["question_id"]]["answer"],
                "hypothesis": row["hypothesis"], "raw_verdict": row["raw_verdict"],
            })
    result.update({
        "schema": "longmemeval-dev12-real-gpt4o-official-rubric-v1",
        "evaluation": "Actual GPT-4o API; pinned official prompt, model, parameters and label expression",
        "transport": "custom resumable runner, not an execution of the upstream CLI",
        "parser": "'yes' in eval_response.lower(), after strip; exact upstream expression",
        "parser_recovery": "No new calls; 28 valid terminal-period verdicts recovered from preserved receipts",
        "unique_requests": len(requests), "unique_judged": len(cache),
        "api_attempts": len(receipts), "http_errors": 0,
        "reported_usage": {
            field: sum(r["usage"][field] for r in receipts)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "returned_models": sorted({r["returned_model"] for r in receipts}),
        "all_finish_reason_stop": True,
        "upstream_commit": protocol["upstream_commit"], "upstream_sha256": SOURCE_SHA,
        "source_files": {name: digest(ROOT / name) for name in
                         ("inputs.json", "requests.json", "protocol.json",
                          "judge_responses.jsonl", "manual_judgments.json")},
        "manual_disagreements": disagreements,
    })
    destination = ROOT / "official_results.json"
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != encoded:
        raise ValueError("Existing finalized result differs; preserve it")
    destination.write_text(encoded, encoding="utf-8")
    with (ROOT / "official_scores.csv").open("w", encoding="utf-8-sig", newline="") as out:
        fields = ("method", "question_id", "question_type", "correct", "raw_verdict", "payload_hash")
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["scored_rows"])
    print(canonical({k: v for k, v in result.items()
                     if k not in {"scored_rows", "manual_disagreements", "source_files"}}))
    print(canonical({"manual_disagreements": disagreements}))
    return result


if __name__ == "__main__":
    finalize()
