"""Run the isolated LightMem direct preset once after existing benchmark jobs exit."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_locomo_comparison as finalizer
from scripts import locomo_server_queue as queue
from scripts.retry_simplemem_after_queue import require_free_ports
from scripts.run_locomo_live_probes import run_probe


def validate_direct(plan, *, unused=True):
    queue.validate_plan(plan)
    enabled = [item for item in plan["methods"] if item["method"] not in plan.get("skip_methods", {})]
    if len(enabled) != 1 or enabled[0]["method"] != "lightmem" or plan.get("context_workers", 1) <= 1:
        raise ValueError("Only LightMem with isolated context workers may run")
    item = enabled[0]
    agent = yaml.safe_load(Path(item["agent_config"]).read_text(encoding="utf-8-sig"))
    dataset = yaml.safe_load(Path(item["dataset_config"]).read_text(encoding="utf-8-sig"))
    settings = {
        "agent_name": "Agentic_memory_lightmem", "model": queue.MODEL,
        "lightmem_ingest_mode": "direct", "lightmem_messages_use": "user_only",
        "agent_chunk_size": 4096, "retrieve_num": 10,
        "lightmem_comparison_mode": False, "lightmem_pre_compress": False,
        "lightmem_topic_segment": False, "lightmem_metadata_generate": False,
        "lightmem_text_summary": False, "lightmem_embedding_backend": "openai",
        "lightmem_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "lightmem_embedding_dims": 384, "provider": "openai_compatible",
    }
    data_settings = {
        "dataset": "LoCoMo", "sub_dataset": "locomo_qa", "chunk_size": 4096,
        "locomo_repeat_session_header": False, "max_test_samples": 10,
        "locomo_categories": [1, 2, 3, 4], "generation_max_length": 1024,
    }
    for config, required in ((agent, settings), (dataset, data_settings)):
        for key, value in required.items():
            if type(config.get(key)) is not type(value) or config[key] != value:
                raise ValueError(f"Direct LightMem preset mismatch: {key}")
    if (Path(dataset["test_files"]).resolve() != Path(plan["dataset"]).resolve()
            or len(json.loads(Path(plan["dataset"]).read_text())) != 10):
        raise ValueError("Direct LightMem requires all ten original dataset conversations")
    for key in ("wait_for_services", "prior_outputs"):
        if not isinstance(plan.get(key), list) or not plan[key] or any(
                not isinstance(value, str) or not value.strip() for value in plan[key]):
            raise ValueError(f"An explicit nonempty {key} list is required")
    paths = [Path(plan[key]) for key in ("output", "probe_output", "controller_output")]
    prior = [Path(value) for value in plan["prior_outputs"]]
    if any(not path.is_absolute() for path in paths + prior):
        raise ValueError("Artifact paths must be absolute")
    paths, prior = [path.resolve() for path in paths], [path.resolve() for path in prior]
    for index, left in enumerate(paths):
        for right in paths[index + 1:] + prior:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("New artifact directories must not overlap each other or prior outputs")
    if unused and any(path.exists() for path in paths):
        raise FileExistsError("Refusing to reuse full, probe, or controller artifacts")
    return item


def final_report(report, probe_path, raw_error):
    if report.get("complete") is not True or report.get("expected") != 1540 or report.get("evaluated") != 1540:
        raise ValueError("Full LightMem report did not pass all 1540-QA gates")
    categories = report.get("by_category", {})
    if set(categories) != {"1", "2", "3", "4"}:
        raise ValueError("Macro F1 requires exactly four complete categories")
    for row in categories.values():
        if (type(row.get("count")) is not int or row["count"] <= 0 or row["count"] != row.get("expected")
                or type(row.get("f1")) not in (int, float) or not math.isfinite(row["f1"])
                or not 0 <= row["f1"] <= 1):
            raise ValueError("Macro F1 requires exactly four complete categories")
    if sum(row["count"] for row in categories.values()) != 1540:
        raise ValueError("Category coverage does not total 1540 questions")
    return dict(report, micro_official_f1=report["official_f1"],
                macro_official_f1=sum(row["f1"] for row in categories.values()) / 4,
                variant="MemoryData direct preset in the common Qwen3.5-9B/MiniLM environment; not exact paper reproduction",
                diagnostic_probe_report=str(probe_path), raw_run_error=raw_error)


def run(plan_path, check_only=False):
    plan = json.loads(plan_path.read_text())
    plan_digest = finalizer.digest(plan_path)
    item = validate_direct(plan)
    if check_only:
        print("LightMem direct plan valid; no inference or artifact writes performed.", flush=True)
        return 0
    controller = Path(plan["controller_output"])
    controller.mkdir(parents=True, exist_ok=False)

    def status(state, **details):
        queue.write_report(controller / "status.json", dict(
            state=state, complete=state == "complete", run_id=plan["run_id"], **details))

    def revalidate():
        if finalizer.digest(plan_path) != plan_digest:
            raise ValueError("The LightMem plan changed after controller startup")
        validate_direct(plan, unused=False)
        states = {name: queue.supervisor_state(name) for name in plan["wait_for_services"]}
        if any(state != "EXITED" for state in states.values()):
            raise RuntimeError(f"A prior Supervisor job restarted before inference: {states}")
        for previous in plan["prior_outputs"]:
            if finalizer.running_writers(previous):
                raise RuntimeError(f"Prior artifact writers remain active: {previous}")
        require_free_ports()
        queue.wait_vllm(plan)

    def encoder(output):
        return queue.service([plan["server_python"], str(ROOT / "scripts/serve_minilm_embeddings.py"),
                              "--model-path", plan["embedding_path"]], output / "embedding_server.log")

    try:
        while True:
            states = {name: queue.supervisor_state(name) for name in plan["wait_for_services"]}
            status("waiting", services=states)
            if any(state not in {"RUNNING", "STARTING", "STOPPING", "EXITED"} for state in states.values()):
                raise RuntimeError(f"Prior Supervisor job did not exit normally: {states}")
            if all(state == "EXITED" for state in states.values()):
                break
            time.sleep(30)
        revalidate()
        probe_output = Path(plan["probe_output"])
        probe_output.mkdir(parents=True, exist_ok=False)
        probe_plan = dict(plan, output=str(probe_output), run_id=plan["run_id"] + "-probe")
        status("live_probe")
        with encoder(probe_output) as process:
            queue.wait_health("http://127.0.0.1:18081", process)
            probe_env = {
                "LIGHTMEM_MODEL": queue.MODEL, "LIGHTMEM_BASE_URL": queue.LLM_ORIGIN + "/v1",
                "LIGHTMEM_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
                "LIGHTMEM_EMBEDDING_DIMENSION": "384",
                "METER_AUXILIARY_JOURNAL": str(probe_output / "lightmem/auxiliary.jsonl"),
            }
            previous_env = {key: os.environ.get(key) for key in probe_env}
            os.environ.update(probe_env)
            try:
                probe = run_probe(probe_plan, "lightmem", None, sample_index=0)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        if probe.get("passed") is not True or probe.get("diagnostic_only") is not True or probe.get("qa_count") != 3:
            raise RuntimeError("LightMem diagnostic probe did not pass; full inference was not started")
        revalidate()
        output = Path(plan["output"])
        output.mkdir(parents=True, exist_ok=False)
        expected = queue.expected_questions(plan["dataset"], 1540)
        status("running_full", diagnostic_probe_report=str(probe_output / "lightmem/report.json"))
        raw_error = None
        with encoder(output) as process:
            queue.wait_health("http://127.0.0.1:18081", process)
            try:
                report = queue.run_method(plan, item, expected)
            except ValueError as exc:
                if str(exc) != "Result metadata differs from the original QA":
                    raise
                raw_error = str(exc)
                parallel = json.loads((output / "lightmem/parallel_status.json").read_text())
                if (parallel.get("complete") is not True or parallel.get("failed") is not False
                        or parallel.get("active_contexts") or finalizer.running_writers(output)):
                    raise RuntimeError("Full inference did not finish; refusing metadata-only finalization") from exc
                report = finalizer.finalize_method(plan, item, expected, controller / "audited/lightmem")
        report = final_report(report, probe_output / "lightmem/report.json", raw_error)
        queue.write_report(controller / "report.json", report)
        status("complete", report=str(controller / "report.json"),
               micro_official_f1=report["micro_official_f1"], macro_official_f1=report["macro_official_f1"])
        return 0
    except Exception as exc:
        status("failed", error_type=type(exc).__name__, error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    return run(args.plan, args.check_only)


if __name__ == "__main__":
    raise SystemExit(main())
