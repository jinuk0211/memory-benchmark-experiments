"""Score preserved and recovered LangMem QA without rewriting either attempt."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import locomo_server_queue as queue
from scripts.audit_langmem_reuse import audit_reuse
from scripts.finalize_locomo_comparison import (
    get_template,
    normalize_records,
    running_writers,
)
from scripts.score_locomo_comparison import (
    build_report,
    expected_questions,
    read_jsonl,
    runtime_metrics,
    summarize_usage,
    validate_predictions,
    write_report,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compatible_prior(plan, prior_path):
    """Only the documented recovery repairs may differ from the original plan."""
    prior_path = Path(prior_path).resolve()
    if plan["file_sha256"].get(str(prior_path)) != digest(prior_path):
        raise ValueError("The original execution plan is unpinned or changed")
    prior = json.loads(prior_path.read_text())
    if plan.get("dtype") != "float16" or any(plan.get(key) != prior.get(key) for key in
            ("model", "model_revision", "dtype", "embedding_path", "dataset", "scorer")):
        raise ValueError("Original and recovered model/data/precision conditions differ")
    if (prior["run_id"] != plan["reused_langmem"]["run_id"]
            or (Path(prior["output"]) / "langmem").resolve() != Path(plan["reused_langmem"]["source"]).resolve()
            or prior["run_id"] == plan["run_id"]):
        raise ValueError("Original and recovered attempt provenance is inconsistent")
    allowed = {str((ROOT / name).resolve()) for name in (
        "methods/a_mem/source/a_mem/memory_layer.py", "methods/e-mem/src/agent/base.py",
        "methods/mem0/source/mem0/memory/main.py", "methods/simplemem/source/SimpleMem/database/vector_store.py",
        "scripts/locomo_server_queue.py", "scripts/metered_openai_proxy.py", "scripts/run_locomo_parallel.py")}
    if any(plan["file_sha256"].get(path) != value and path not in allowed
           for path, value in prior["file_sha256"].items()):
        raise ValueError("An undocumented source/config change prevents shard reuse")
    return prior


def shard_records(source, indices, expected, hashes):
    """Verify original global indices and the raw outputs behind completion markers."""
    samples = list(dict.fromkeys(sample for sample, _ in expected))
    global_ids = {(sample, str(qa.get("question_id") or f"{sample}_qa{index}")): offset
                  for offset, ((sample, index), qa) in enumerate(expected.items())}
    result = []
    for index in indices:
        context = source / f"context_{index:02d}"
        marker_path = context / "completion.json"
        marker = json.loads(marker_path.read_text())
        raw = Path(marker["result_path"]).resolve()
        sample = samples[index]
        count = sum(key[0] == sample for key in expected)
        if (marker.get("context_index") != index or marker.get("sample_id") != sample
                or marker.get("questions") != count or not raw.is_relative_to(context)):
            raise ValueError("A completion marker has wrong original context identity")
        rows = json.loads(raw.read_text())["data"]
        if len(rows) != count:
            raise ValueError("A completed context has a changed QA count")
        for row in rows:
            metadata = row.get("eval_metadata") or {}
            key = (sample, metadata.get("question_id"))
            if (metadata.get("sample_id") != sample or type(row.get("query_id")) is not int
                    or row["query_id"] != global_ids.get(key) or type(row.get("context_id")) is not int
                    or row["context_id"] != index):
                raise ValueError("Raw QA global/context identity changed")
        hashes.update({str(marker_path): digest(marker_path), str(raw): digest(raw)})
        result.extend(rows)
    return result


def finalize(plan, prior_path, destination):
    """Audit two disjoint partitions, then score their exact 1540-question union."""
    destination = Path(destination).resolve()
    old = Path(plan["reused_langmem"]["source"]).resolve()
    new = (Path(plan["output"]) / "langmem").resolve()
    if destination.exists():
        raise FileExistsError("Refusing to overwrite a composite evaluation")
    if any(destination.is_relative_to(source) or source.is_relative_to(destination) for source in (old, new)):
        raise ValueError("Composite output must be separate from both original attempts")
    if running_writers(old) or running_writers(new):
        raise RuntimeError("LangMem artifact writers are still live")
    queue.validate_plan(plan)
    prior = compatible_prior(plan, prior_path)
    reused = audit_reuse(plan)
    status = json.loads((new / "parallel_status.json").read_text())
    attempt = json.loads((new / "selected_attempt.json").read_text())
    indices = list(range(4, 10))
    if (status.get("selected_contexts_complete") is not True or status.get("failed") is not False
            or status.get("active_contexts") or status.get("context_indices") != indices
            or attempt.get("run_id") != plan["run_id"] or attempt.get("method") != "langmem"
            or attempt.get("requires_composite") is not True or attempt.get("context_indices") != indices):
        raise ValueError("The new attempt has not completed its exact selected contexts")
    expected = expected_questions(plan["dataset"], 1540)
    hashes = {}
    old_records = shard_records(old, range(4), expected, hashes)
    new_records = shard_records(new, indices, expected, hashes)
    if json.loads((new / "selected_results.json").read_text())["data"] != new_records:
        raise ValueError("Merged recovered results differ from the original context outputs")
    selected = {key: qa for key, qa in expected.items() if key[0] not in reused["samples"]}
    original_usage, new_usage = read_jsonl(old / "usage.jsonl"), read_jsonl(new / "usage.jsonl")
    old_usage = [row for row in original_usage if row["sample_id"] in reused["samples"]]
    queue.audit_coverage(new_usage, selected, plan["run_id"], "langmem")
    usage = old_usage + new_usage
    usage_summary = summarize_usage(usage, True)
    if not usage_summary["complete"] or usage_summary["truncated_requests"]:
        raise ValueError("Composite usage is incomplete, duplicated or truncated")
    item = next(item for item in plan["methods"] if item["method"] == "langmem")
    agent = yaml.safe_load(Path(item["agent_config"]).read_text())
    config = yaml.safe_load(Path(item["dataset_config"]).read_text())
    template = get_template(config["sub_dataset"], "query", agent["agent_name"])
    records = normalize_records(old_records + new_records, expected, template)
    destination.mkdir(parents=True, exist_ok=False)
    normalized = destination / "normalized_results.json"
    write_report(normalized, {"data": records})
    predictions = queue.canonical_predictions([normalized], expected)
    valid, issues, missing = validate_predictions(predictions, expected)
    if issues or missing or any(not row["prediction"].strip() for row in valid):
        raise ValueError("Composite does not contain every original nonempty QA exactly once")
    prediction_path, usage_path = destination / "predictions.jsonl", destination / "composite_usage.jsonl"
    for path, rows in ((prediction_path, predictions), (usage_path, usage)):
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = build_report(plan["dataset"], prediction_path, plan["scorer"], usage_path)
    attempts = [runtime_metrics([], source / "telemetry.json", active["hourly_rate_usd"])
                for source, active in ((old, prior), (new, plan))]
    report.update(method="langmem", composite=True, run_ids=[prior["run_id"], plan["run_id"]],
                  quality_complete=True, coverage_complete=True,
                  partitions=[{"run_id": prior["run_id"], "questions": len(old_records), "reused": True},
                              {"run_id": plan["run_id"], "questions": len(new_records), "reused": False}],
                  gross_attempt_usage=summarize_usage(original_usage + new_usage, True))
    report["attempt_runtime"] = [{"run_id": active["run_id"], "includes_old_unfinished_work": source == old,
                                  **metrics} for source, active, metrics in zip((old, new), (prior, plan), attempts)]
    report["gross_attempt_cost"] = {
        key: sum(values) if all(value is not None for value in values) else None
        for key in ("estimated_rental_cost_usd", "gpu_energy_wh")
        for values in [[attempt[key] for attempt in attempts]]}
    report["accounting_note"] = ("Composite usage covers reused QA plus new generation. Gross attempt usage/cost "
        "also includes old unfinished work; do not add these overlapping totals together. GPU time/energy and "
        "rental estimates are not allocated proportionally to QA. Setup, idle time and provider bandwidth/storage "
        "charges are not a complete invoice. Original run IDs and raw artifacts remain unchanged.")
    for path in (Path(prior_path), old / "usage.jsonl", old / "telemetry.json", new / "usage.jsonl",
                 new / "telemetry.json", new / "selected_results.json", new / "parallel_status.json",
                 new / "selected_attempt.json", Path(__file__)):
        hashes[str(path.resolve())] = digest(path)
    report["raw_sources_sha256"] = hashes
    report["native_behavior"] = {"trustcall_patch_error_log_events": sum(
        path.read_text(encoding="utf-8", errors="replace").count("Could not apply patch")
        for source, context_indices in ((old, range(4)), (new, indices))
        for index in context_indices for path in (source / f"context_{index:02d}").glob("worker.log")),
        "note": "Native Trustcall patch-error log events, not a count of uniquely lost facts."}
    if not report["complete"]:
        raise ValueError("Composite failed the original scorer/accounting completeness gates")
    write_report(destination / "report.json", report)
    return report


def wait_for_generation(plan: dict, service: str) -> None:
    """Wait only while the real queue is live; never accept an open artifact set."""
    output = Path(plan["output"])
    source = output / "langmem"
    while True:
        if (source / "selected_attempt.json").is_file() and not running_writers(source):
            return
        state_path = output / "recovery_status.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if any(item.get("method") == "langmem" and item.get("state") == "failed"
               for item in state.get("methods", [])):
            raise RuntimeError("LangMem generation failed; composite cannot be produced")
        if state.get("method") not in (None, "langmem"):
            raise RuntimeError("Queue advanced without closed LangMem completion artifacts")
        if queue.supervisor_state(service) not in {"RUNNING", "STARTING", "STOPPING"}:
            raise RuntimeError("Queue terminated before LangMem artifacts were closed")
        time.sleep(30)


def main(argv: list[str] | None = None) -> int:
    """Audit prior provenance before waiting, then run the unchanged full scorer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prior-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--queue", default="locomo-recovery-50156979")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a composite evaluation")
    queue.validate_plan(plan)
    compatible_prior(plan, args.prior_plan)
    audit_reuse(plan)
    if args.wait:
        wait_for_generation(plan, args.queue)
    report = finalize(plan, args.prior_plan, args.output)
    print(json.dumps({key: report[key] for key in ("complete", "evaluated", "official_f1", "run_ids")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
