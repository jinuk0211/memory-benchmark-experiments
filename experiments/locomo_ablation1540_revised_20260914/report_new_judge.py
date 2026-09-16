"""Export complete, provenance-checked mini accuracy for one validated full-LoCoMo Ours run."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any

import prepare_new_judge as prep

CATEGORIES = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop"}
FILES = ("input.jsonl", "input_manifest.json", "official_accuracy_prompt.txt", "run_config.json",
         "api_events.jsonl", "judge_cache.jsonl", "scores.jsonl", "summary.json")
PRICES = {"input": 0.15, "cached_input": 0.075, "output": 0.60}


def read_jsonl(path: Path) -> list[dict]:
    """Read without repairing or modifying the runner's immutable evidence."""
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Unfinished journal boundary: {path.name}")
    rows = [json.loads(line) for line in raw.splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Non-object journal record: {path.name}")
    return rows


def same_number(actual: Any, expected: float) -> bool:
    return (type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9))


def validate_input(directory: Path, judge: Any) -> tuple[list[dict], dict, dict]:
    """Revalidate original native reader evidence rather than trusting copied judge F1 values."""
    manifest = prep.report.read(directory / "input_manifest.json")
    if (manifest.get("status") != "PREPARED_NO_API_CALLS" or manifest.get("method") != "ours"
            or manifest.get("rows") != 1540 or manifest.get("histories") != 10
            or manifest.get("categories") != {str(k): v for k, v in prep.CATEGORY_COUNTS.items()}
            or manifest.get("model") != prep.JUDGE_MODEL
            or manifest.get("required_returned_model") != prep.REQUIRED_RETURNED_MODEL
            or manifest.get("source_predictions_modified") is not False
            or manifest.get("judge_runner_path") != str(prep.JUDGE_RUNNER)
            or manifest.get("judge_runner_sha256") != prep.JUDGE_RUNNER_SHA256
            or manifest.get("dataset_sha256") != prep.DATASET_SHA256
            or manifest.get("prompt_sha256") != prep.PROMPT_SHA256
            or manifest.get("prices_per_million_usd") != PRICES
            or manifest.get("api_settings") != {"temperature": 0.0,
                "response_format": {"type": "json_object"}, "max_tokens": "omitted as upstream"}):
        raise ValueError("Prepared Ours full1540 input/model provenance differs")
    raw = (directory / "input.jsonl").read_bytes()
    prompt = judge.load_prompt()
    if (judge.digest(raw) != manifest.get("input_sha256")
            or (directory / "official_accuracy_prompt.txt").read_bytes() != prompt.encode("utf-8")):
        raise ValueError("Prepared input or exact prompt fingerprint changed")
    source = manifest["source_validation"]
    results, run = Path(source["results_directory"]), Path(source["run_directory"])
    native, provenance = prep.validated_rows(results, run, "ours", Path(manifest["dataset_path"]))
    if provenance != source:
        raise ValueError("Selected validated source/native fingerprints changed")
    rows = [{"id": row["id"], "method": "ours", "conversation_id": row["conv_id"],
             "qa_index": row["qa_index"], "category": row["category"], "question": row["question"],
             "gold_answer": row["gold"], "generated_answer": row["prediction"],
             "official_f1": row["official_f1"]} for row in native]
    expected_raw = "".join(judge.canonical_json(row) + "\n" for row in rows).encode("utf-8")
    if raw != expected_raw:
        raise ValueError("Judge inputs differ from the unchanged selected predictions or F1")
    config = prep.report.read(directory / "run_config.json")
    expected_config = {"input_sha256": manifest["input_sha256"], "prompt_sha256": prep.PROMPT_SHA256,
                       "model": prep.JUDGE_MODEL, "temperature": 0.0,
                       "response_format": {"type": "json_object"}, "prompt_source": str(judge.PROMPT_SOURCE),
                       "source_sha256": prep.report.sha(judge.PROMPT_SOURCE),
                       "deduplication": "Identical full API payloads share a judgment across methods"}
    if config != expected_config:
        raise ValueError("Frozen judge run configuration differs")
    selected = prep.report.read(results / "RESULTS.json")["results"]["ours"]
    if (set(selected.get("category_f1", {})) != {str(category) for category in CATEGORIES}
            or any(not same_number(selected["category_f1"][str(category)], statistics.mean(
                row["official_f1"] for row in rows if row["category"] == category) * 100)
                for category in CATEGORIES)):
        raise ValueError("Selected category F1 differs from its validated predictions")
    return rows, manifest, selected


def validate_receipt(record: dict, judge: Any, *, selected: bool) -> None:
    """Validate native identity/model/usage; only complete labels with stop may score a row."""
    if (record.get("kind") != "response" or not isinstance(record.get("response_id"), str)
            or not record["response_id"] or record.get("returned_model") != prep.REQUIRED_RETURNED_MODEL
            or record.get("prompt_sha256") != prep.PROMPT_SHA256):
        raise ValueError("Native judge receipt identity, model or prompt differs")
    usage = record.get("usage")
    if (not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
        raise ValueError("Native judge token usage is absent or inconsistent")
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0)
    if (type(cached) is not int or not 0 <= cached <= usage["prompt_tokens"]
            or record.get("prices_per_million_usd") != PRICES
            or not same_number(record.get("cost_usd"), judge.usage_cost(usage, PRICES))
            or record.get("unknown_cost_upper_bound_usd") != 0):
        raise ValueError("Native judge usage accounting differs")
    try:
        label = judge.parse_label(record.get("response"))
    except (ValueError, TypeError):
        label = None
    if record.get("label") != label:
        raise ValueError("Native raw response label differs from saved label")
    if selected and (label is None or record.get("finish_reason") != "stop"
                     or usage["prompt_tokens"] == 0 or usage["completion_tokens"] == 0):
        raise ValueError("Selected native judgment must have a valid label, usage and stop")


def validate_judgments(directory: Path, rows: list[dict], judge: Any) -> tuple[list[dict], list[dict], dict]:
    """Match every logical item to one exact payload receipt and independently check aggregates."""
    prompt = judge.load_prompt()
    keys = [judge.payload_hash(judge.make_payload(row, prompt)) for row in rows]
    expected = set(keys)
    events, cache = read_jsonl(directory / "api_events.jsonl"), read_jsonl(directory / "judge_cache.jsonl")
    responses, cached = {}, {}
    for event in events:
        if (event.get("payload_hash") not in expected or event.get("prompt_sha256") != prep.PROMPT_SHA256
                or event.get("kind") not in ("response", "error")):
            raise ValueError("Journal contains an unrelated or invalid request")
        if event["kind"] == "response":
            validate_receipt(event, judge, selected=False)
            identity = event["response_id"]
            if identity in responses:
                raise ValueError("Duplicate native response identity")
            responses[identity] = event
        unknown = event.get("unknown_cost_upper_bound_usd")
        if type(unknown) not in (int, float) or not math.isfinite(unknown) or unknown < 0:
            raise ValueError("Invalid unresolved billing bound")
    for record in cache:
        validate_receipt(record, judge, selected=True)
        key = record.get("payload_hash")
        if key not in expected or key in cached:
            raise ValueError("Duplicate or unrelated native cached request")
        if responses.get(record["response_id"]) != record:
            raise ValueError("Cached judgment differs from its native response event")
        cached[key] = record
    if set(cached) != expected:
        raise ValueError("Missing judgments: all 1,540 logical rows must be complete")
    valid_events = [event for event in responses.values() if event.get("label") in ("CORRECT", "WRONG")]
    if any(event["label"] != cached[event["payload_hash"]]["label"] for event in valid_events):
        raise ValueError("Conflicting native labels for one payload")
    scored = [{**row, "payload_hash": key, "judge_label": cached[key]["label"],
               "judge_score": int(cached[key]["label"] == "CORRECT")}
              for row, key in zip(rows, keys, strict=True)]
    if read_jsonl(directory / "scores.jsonl") != scored:
        raise ValueError("Saved scores differ from input/native receipt mapping")
    categories = {}
    for category, name in CATEGORIES.items():
        count = prep.CATEGORY_COUNTS[category]
        correct = sum(row["judge_score"] for row in scored if row["category"] == category)
        categories[str(category)] = {"name": name, "expected": count, "scored": count,
                                     "correct": correct, "accuracy_pct": 100 * correct / count}
    correct = sum(row["judge_score"] for row in scored)
    method = {"expected": 1540, "scored": 1540, "correct": correct, "complete": True,
              "accuracy_pct": 100 * correct / 1540, "categories": categories}
    tokens = {key: sum(event["usage"][key] for event in responses.values())
              for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    tokens["cached_prompt_tokens"] = sum((event["usage"].get("prompt_tokens_details") or {}).get(
        "cached_tokens", 0) for event in responses.values())
    native_summary = {"status": "complete", "logical_rows": 1540, "scored_rows": 1540,
                      "unique_payloads": len(expected), "cached_unique_payloads": len(expected),
                      "billed_responses": len(responses), "error_events": len(events) - len(responses),
                      "token_usage": tokens, "returned_models": dict(Counter(
                          event["returned_model"] for event in responses.values())), "methods": {"ours": method}}
    saved = prep.report.read(directory / "summary.json")
    if any(saved.get(key) != value for key, value in native_summary.items()):
        raise ValueError("Complete judge summary differs from native counts or accuracy")
    cost = sum(event["cost_usd"] for event in responses.values())
    unknown = sum(event["unknown_cost_upper_bound_usd"] for event in events)
    for key, value in {"cost_usd": cost, "unknown_cost_upper_bound_usd": unknown,
                       "budget_committed_usd": cost + unknown}.items():
        if not same_number(saved.get(key), value):
            raise ValueError("Judge summary billing differs from native journal")
        native_summary[key] = value
    mapped = [{**row, "response_id": cached[row["payload_hash"]]["response_id"],
               "native_receipt_sha256": judge.digest(judge.canonical_json(
                   cached[row["payload_hash"]]).encode("utf-8"))} for row in scored]
    return mapped, list(responses.values()), native_summary


def export(directory: Path, output: Path) -> dict:
    """Publish only after all offline checks pass; preserve original artifacts and metrics."""
    before = {name: prep.report.sha(directory / name) for name in FILES}
    judge = prep.load_judge()
    rows, manifest, selected = validate_input(directory, judge)
    mapped, receipts, summary = validate_judgments(directory, rows, judge)
    if before != {name: prep.report.sha(directory / name) for name in FILES}:
        raise ValueError("Judge run changed during validation; wait for a stable completed run")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty report directory")
    accuracy = summary["methods"]["ours"]
    validation = {"status": "PASS", "logical_rows": 1540, "histories": 10,
                  "categories": prep.CATEGORY_COUNTS, "unique_payloads": summary["unique_payloads"],
                  "selected_response_model": prep.REQUIRED_RETURNED_MODEL,
                  "selected_finish_reason": "stop", "native_responses": len(receipts),
                  "source_validation": manifest["source_validation"], "judge_directory": str(directory.resolve()),
                  "judge_files_sha256": before, "judge_runner_sha256": prep.JUDGE_RUNNER_SHA256}
    result = {"validation": validation, "method": "ours", "source_metrics": selected,
              "gpt4omini_accuracy": accuracy, "native_judge_summary": summary}
    output.mkdir(parents=True, exist_ok=True)
    prep.report.write_json(output / "JUDGE_VALIDATION.json", validation)
    prep.report.write_json(output / "RESULTS.json", result)
    for name, records in (("mapped_judgments.jsonl", mapped), ("native_judgments.jsonl", receipts)):
        (output / name).write_text("".join(judge.canonical_json(row) + "\n" for row in records), encoding="utf-8")
    cells = [selected["category_f1"][str(category)] for category in CATEGORIES]
    cells.extend((selected["f1"], accuracy["accuracy_pct"]))
    latex = r"& \textbf{Ours} & " + " & ".join(f"{value:.2f}" for value in cells) + r" \\" + "\n"
    (output / "table_main_ours.tex").write_text(latex, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.judge_dir, args.output_dir)
    print(json.dumps({"status": "PASS", "rows": 1540,
                      "accuracy_pct": result["gpt4omini_accuracy"]["accuracy_pct"]}))


if __name__ == "__main__":
    main()
