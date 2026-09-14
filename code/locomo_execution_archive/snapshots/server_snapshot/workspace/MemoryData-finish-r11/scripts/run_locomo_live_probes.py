"""Sequential live-model probes; outputs are diagnostics, never full benchmark scores."""

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.memoryagentbench.prompts.benchmark_templates import get_template
from scripts import locomo_server_queue as queue
from scripts.finalize_locomo_comparison import normalize_records
from scripts.score_locomo_comparison import (
    read_jsonl,
    summarize_usage,
    validate_predictions,
    write_report,
)

PROBES = (("e_mem", None), ("mem0", 80), ("a_mem", 40))


def prefix_dataset(dataset: list, turns: int | None) -> list:
    """Copy one conversation prefix; keep original QA order/IDs and session metadata."""
    if turns is not None and (type(turns) is not int or turns <= 0):
        raise ValueError("Probe turns must be a positive integer or None")
    sample = copy.deepcopy(dataset[0])
    if turns is not None:
        conversation = sample["conversation"]
        sessions = sorted((key for key in conversation if re.fullmatch(r"session_\d+", key)),
                          key=lambda key: int(key.split("_")[1]))
        remaining = turns
        for key in sessions:
            if remaining:
                selected = conversation[key][:remaining]
                conversation[key] = selected
                remaining -= len(selected)
            else:
                del conversation[key]
                conversation.pop(key + "_date_time", None)
    return [sample]


def check_gate(rows: list, issues: list, missing: list, usage: dict) -> None:
    """Require all three nonempty answers and complete, failure-free metering."""
    if (len(rows) != 3 or issues or missing or any(not row["prediction"].strip() for row in rows)
            or usage.get("complete") is not True or usage.get("failed_requests") != 0
            or usage.get("truncated_requests") != 0):
        raise ValueError("Live probe failed exact QA or metering integrity gates")


def run_probe(plan: dict, method: str, turns: int | None) -> dict:
    """Reuse pinned agent settings; only test input size and QA count are reduced."""
    output = Path(plan["output"]) / method
    output.mkdir(exist_ok=False)
    item = next(item for item in plan["methods"] if item["method"] == method)
    source = json.loads(Path(plan["dataset"]).read_text())
    data = prefix_dataset(source, turns)
    write_report(output / "test_dataset.json", data)
    config = yaml.safe_load(Path(item["dataset_config"]).read_text())
    config.update(test_files=str(output / "test_dataset.json"), max_test_samples=1)
    # JSON is also YAML; this test-only directory was exclusively created above.
    write_report(output / "dataset.yaml", config)
    expected = dict([(key, qa) for key, qa in queue.expected_questions(plan["dataset"], 1540).items()
                     if key[0] == source[0]["sample_id"]][:3])
    env = os.environ.copy()
    env.update(OPENAI_API_KEY="local-comparison", OPENAI_BASE_URL=queue.LLM_ORIGIN + "/v1",
               EMBEDDING_BASE_URL=queue.EMBED_ORIGIN + "/v1", METER_RUN_ID=plan["run_id"], METER_METHOD=method,
               METER_LLM_PROXY_ORIGIN=queue.LLM_ORIGIN, METER_EMBEDDING_PROXY_ORIGIN=queue.EMBED_ORIGIN,
               METER_TIMING_JOURNAL=str(output / "timing.jsonl"), BASELINE_STRICT_COMPARISON="1",
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
               HF_HOME=plan["hf_home"])
    proxy = [plan["server_python"], str(ROOT / "scripts/metered_openai_proxy.py"),
             "--run-id", plan["run_id"], "--method", method]
    llm_path, embed_path = output / "llm_usage.jsonl", output / "embedding_usage.jsonl"
    command = [plan["client_python"], str(ROOT / "main.py"), "--agent_config", item["agent_config"],
               "--dataset_config", str(output / "dataset.yaml"), "--artifact_root", str(output / "artifacts"),
               "--max_test_queries_ablation", "3", "--fail-on-query-error"]
    with queue.service(proxy + ["--journal", str(llm_path), "--upstream-base-url", "http://127.0.0.1:18080/v1",
                               "--port", "18082", "--comparison-policy"], output / "llm_proxy.log") as llm, \
         queue.service(proxy + ["--journal", str(embed_path), "--upstream-base-url", "http://127.0.0.1:18081/v1",
                               "--port", "18083"], output / "embedding_proxy.log") as embed:
        queue.wait_health(queue.LLM_ORIGIN, llm)
        queue.wait_health(queue.EMBED_ORIGIN, embed)
        start = time.monotonic()
        samples = [queue.gpu_sample()]
        with queue.service(command, output / "benchmark.log", env) as process:
            while process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                samples.append(queue.gpu_sample())
            exit_code = process.returncode
        queue.drain_proxies([llm, embed])
        wall = time.monotonic() - start
    write_report(output / "gpu_samples.json", samples)
    write_report(output / "telemetry.json", queue.telemetry(samples, read_jsonl(output / "timing.jsonl"), wall, output))
    if exit_code != 0:
        raise RuntimeError(f"{method} probe exited with code {exit_code}")
    results = list((output / "artifacts").rglob("*_results.json"))
    if len(results) != 1:
        raise ValueError("Expected one closed live-probe result")
    agent = yaml.safe_load(Path(item["agent_config"]).read_text())
    records = normalize_records(json.loads(results[0].read_text())["data"], expected,
                                get_template(config["sub_dataset"], "query", agent["agent_name"]))
    normalized = output / "normalized_results.json"
    write_report(normalized, {"data": records})
    predictions = queue.canonical_predictions([normalized], expected)
    valid, issues, missing = validate_predictions(predictions, expected)
    usage_rows = read_jsonl(llm_path) + read_jsonl(embed_path)
    usage = summarize_usage(usage_rows, True)
    queue.audit_coverage(usage_rows, expected, plan["run_id"], method)
    check_gate(valid, issues, missing, usage)
    report = {"passed": True, "method": method, "diagnostic_only": True, "run_id": plan["run_id"],
              "qa_count": 3, "memory_turns": sum(len(value) for key, value in data[0]["conversation"].items()
                                                 if re.fullmatch(r"session_\d+", key)),
              "usage": usage, "full_dataset_sha256": hashlib.sha256(Path(plan["dataset"]).read_bytes()).hexdigest(),
              "source_sha256": plan["file_sha256"]}
    write_report(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    queue.validate_plan(plan)
    queue.wait_vllm(plan)
    output = Path(plan["output"])
    outcomes = []
    with queue.service([plan["server_python"], str(ROOT / "scripts/serve_minilm_embeddings.py"),
                        "--model-path", plan["embedding_path"]], output / "embedding_server.log") as encoder:
        queue.wait_health("http://127.0.0.1:18081", encoder)
        for method, turns in PROBES:
            write_report(output / "probe_status.json", {"state": "running", "method": method, "outcomes": outcomes})
            try:
                outcomes.append(run_probe(plan, method, turns))
            except Exception as error:  # noqa: BLE001 - record explicit failure and test the other methods
                outcomes.append({"method": method, "passed": False, "error": str(error), "error_type": type(error).__name__})
    passed = len(outcomes) == 3 and all(row["passed"] for row in outcomes)
    write_report(output / "probe_status.json", {"state": "complete" if passed else "failed", "passed": passed,
                                               "outcomes": outcomes})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
