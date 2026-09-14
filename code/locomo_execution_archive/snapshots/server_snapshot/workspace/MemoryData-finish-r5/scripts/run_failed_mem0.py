"""Rebuild only failed full conversations after the preserved Mem0 run exits."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import locomo_server_queue as queue
from scripts.score_locomo_comparison import expected_questions, write_report
from scripts.finalize_langmem_recovery import shard_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--prior-plan", required=True)
    parser.add_argument("--after-service", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    prior = json.loads(Path(args.prior_plan).read_text())
    queue.validate_plan(plan)
    while queue.supervisor_state(args.after_service) in {"RUNNING", "STARTING", "STOPPING"}:
        time.sleep(30)
    queue.validate_plan(plan)
    expected = expected_questions(plan["dataset"], 1540)
    old = (Path(prior["output"]) / "mem0").resolve()
    status = json.loads((old / "parallel_status.json").read_text())
    if status.get("active_contexts"):
        raise RuntimeError("Prior run still reports active workers")
    indices, preserved = [], []
    for index in range(10):
        endings = [event for event in status["events"] if event["context"] == index and event["event"] == "end"]
        if len(endings) != 1:
            raise RuntimeError(f"Prior context {index} has no unique terminal worker event")
        if endings[0]["exit_code"] != 0:
            indices.append(index)
        else:
            shard_records(old, [index], expected, {})
            preserved.append(index)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "selection.json", {"prior_plan": args.prior_plan,
        "prior_output": str(old), "rebuild_contexts": indices, "preserved_contexts": preserved,
        "note": "Each rebuilt conversation starts at original turn zero and includes every original QA."})
    if not indices:
        write_report(output / "status.json", {"state": "no_repair_needed", "complete": False})
        return 0
    write_report(output / "status.json", {"state": "running", "complete": False, "contexts": indices})
    queue.wait_vllm(plan)
    with queue.service([plan["server_python"], str(ROOT / "scripts/serve_minilm_embeddings.py"),
                        "--model-path", plan["embedding_path"]], output / "embedding_server.log") as encoder:
        queue.wait_health("http://127.0.0.1:18081", encoder)
        item = next(item for item in plan["methods"] if item["method"] == "mem0")
        try:
            report = queue.run_method(plan, item, expected, context_indices=indices)
        except Exception as exc:
            write_report(output / "status.json", {"state": "needs_repair", "complete": False,
                "contexts": indices, "error_type": type(exc).__name__, "error": str(exc)})
            raise
    write_report(output / "status.json", {"state": "awaiting_composite", "complete": False, "attempt": report})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
