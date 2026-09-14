"""Score untouched comparison artifacts after the Supervisor queue exits.

The benchmark stores template-wrapped questions. Accept only an exact rendering
of the pinned template, then give a separate normalized copy to the original
metadata validator. This never changes inference, raw answers or queue files.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.memoryagentbench.prompts.benchmark_templates import get_template
from scripts import locomo_server_queue as queue
from scripts.score_locomo_comparison import (
    build_report, expected_questions, read_jsonl, write_report,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_records(records, expected, template):
    """Validate the entire recorded prompt; never strip arbitrary surrounding text."""
    lookup = {
        (sample, str(qa.get("question_id") or f"{sample}_qa{index}")): qa
        for (sample, index), qa in expected.items()
    }
    normalized = []
    for record in records:
        metadata = record.get("eval_metadata") or {}
        key = (metadata.get("sample_id"), metadata.get("question_id"))
        if key not in lookup:
            raise ValueError("Unknown original question identity")
        question = lookup[key]["question"]
        if record.get("query") != template.format(question=question):
            raise ValueError("Recorded question differs from the exact pinned prompt template")
        normalized.append(dict(record, query=question, recorded_query=record["query"]))
    return normalized


def running_writers(run_directory):
    """Detect orphaned benchmark/proxy processes, not the shared inference server."""
    writers = []
    scripts = {"main.py", "run_locomo_parallel.py", "metered_openai_proxy.py"}
    prefix = str(Path(run_directory).resolve()) + "/"
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            arguments = path.read_bytes().decode().split("\0")
        except FileNotFoundError:
            continue
        if (any(Path(arg).name in scripts for arg in arguments)
                and any(arg.startswith(prefix) for arg in arguments)):
            writers.append(int(path.parent.name))
    return writers


def wait_for_queue(plan, wait):
    while True:
        state = queue.supervisor_state("locomo-comparison-queue")
        writers = running_writers(plan["output"])
        if state == "EXITED" and not writers:
            return
        if state not in {"RUNNING", "STARTING", "STOPPING", "EXITED"}:
            raise RuntimeError(f"Queue did not exit normally: {state}")
        if not wait:
            raise RuntimeError(f"Queue or artifact writers are still live: {state}, {writers}")
        time.sleep(30)


def finalize_method(plan, item, expected, destination):
    method = item["method"]
    source = Path(plan["output"]) / method
    status = json.loads((source / "parallel_status.json").read_text())
    if status.get("complete") is not True or status.get("failed") is not False:
        raise ValueError("The benchmark did not complete all conversation workers")
    if status.get("active_contexts"):
        raise ValueError("The benchmark still reports active contexts")
    raw_path = source / "parallel_results.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    agent = yaml.safe_load(Path(item["agent_config"]).read_text())
    dataset = yaml.safe_load(Path(item["dataset_config"]).read_text())
    template = get_template(dataset["sub_dataset"], "query", agent["agent_name"])
    records = normalize_records(raw["data"], expected, template)
    destination.mkdir(parents=True, exist_ok=False)
    normalized = destination / "normalized_results.json"
    write_report(normalized, {"data": records})
    rows = queue.canonical_predictions([normalized], expected)
    predictions = destination / "predictions.jsonl"
    with predictions.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    usage_path = source / "usage.jsonl"
    telemetry_path = source / "telemetry.json"
    report = build_report(plan["dataset"], predictions, plan["scorer"], usage_path,
                          telemetry_path, plan.get("hourly_rate_usd"))
    report.update(method=method, run_id=plan["run_id"], context_workers=plan["context_workers"])
    report["quality_complete"] = report["usage"]["truncated_requests"] == 0 and report["empty_predictions"] == 0
    try:
        queue.audit_coverage(read_jsonl(usage_path), expected, plan["run_id"], method)
        report["coverage_complete"] = True
    except ValueError as exc:
        report.update(coverage_complete=False, coverage_error=str(exc))
    try:
        report["auxiliary"] = queue.auxiliary_metrics(plan, method, source)
    except (ValueError, OSError) as exc:
        report["auxiliary"] = {"applicable": True, "complete": False, "error": str(exc)}
    report["complete"] = (report["complete"] and report["quality_complete"]
                          and report["coverage_complete"] and report["auxiliary"]["complete"])
    report["normalization"] = {
        "rule": "Exact pinned benchmark query template; all other original metadata gates unchanged",
        "template": template, "rows": len(records), "raw_artifacts_modified": False,
        "raw_results_path": str(raw_path), "raw_results_sha256": digest(raw_path),
        "usage_sha256": digest(usage_path), "telemetry_sha256": digest(telemetry_path),
        "finalizer_sha256": digest(__file__),
    }
    report["telemetry_notes"] = {
        key: value for key, value in json.loads(telemetry_path.read_text()).items()
        if key.endswith("note") or key in {"gpu_samples", "missing_gpu_samples"}
    }
    if method == "langmem":
        report["native_behavior"] = {
            "trustcall_patch_error_log_events": sum(
                path.read_text(encoding="utf-8", errors="replace").count("Could not apply patch")
                for path in source.glob("context_*/worker.log")),
            "note": "Native Trustcall can omit invalid patches and continue. Log events are not "
                    "a count of uniquely lost facts. No memory, answer or native settings were changed by this finalizer.",
        }
    write_report(destination / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    queue.validate_plan(plan)
    if plan.get("context_workers", 1) <= 1:
        raise ValueError("This finalizer requires isolated parallel context artifacts")
    print("Waiting for the existing comparison queue; inference and raw artifacts remain untouched.", flush=True)
    wait_for_queue(plan, args.wait)
    queue.validate_plan(plan)
    source_status = json.loads((Path(plan["output"]) / "queue_status.json").read_text())
    if source_status.get("state") not in {"complete", "complete_with_skips", "failed"}:
        raise RuntimeError("Queue exited without recording its terminal method outcomes")
    expected = expected_questions(plan["dataset"], 1540)
    args.output.mkdir(parents=True, exist_ok=False)
    if not queue.higmem_ready(queue.supervisor_state("higmem-eval"), plan["higmem_run"],
                              expected, args.output / "higmem_accounting.json"):
        raise RuntimeError("HiGMem has not completed")
    results = []
    for item in plan["methods"]:
        method = item["method"]
        if method in plan.get("skip_methods", {}):
            results.append({"method": method, "state": "skipped", "reason": plan["skip_methods"][method]})
            continue
        source = Path(plan["output"]) / method
        status_path = source / "parallel_status.json"
        if not status_path.exists() or json.loads(status_path.read_text()).get("complete") is not True:
            results.append({"method": method, "state": "benchmark_failed", "source": str(source)})
            continue
        try:
            report = finalize_method(plan, item, expected, args.output / method)
            results.append({"method": method, "state": "complete" if report["complete"] else "audit_failed",
                            "official_f1": report["official_f1"], "report": str(args.output / method / "report.json")})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            results.append({"method": method, "state": "audit_failed", "error": str(exc)})
        print(json.dumps(results[-1]), flush=True)
    audit_failed = any(row["state"] == "audit_failed" for row in results)
    summary = {
        "finalization_complete": not audit_failed,
        "all_six_complete": all(row["state"] == "complete" for row in results),
        "run_id": plan["run_id"], "plan_sha256": digest(args.plan), "methods": results,
        "source_queue_status": source_status,
        "note": "Separate postprocessing only. Failed/skipped benchmarks have no valid final score. "
                "Initial queue metadata failures caused solely by the exact query wrapper are re-audited here, "
                "without rerunning generation or changing raw artifacts.",
    }
    write_report(args.output / "summary.json", summary)
    return 1 if audit_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
