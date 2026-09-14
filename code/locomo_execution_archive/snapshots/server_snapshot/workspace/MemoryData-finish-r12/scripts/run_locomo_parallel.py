"""Batch independent LoCoMo contexts without changing their memory/QA order."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_locomo_comparison import read_jsonl, write_report


def load_contexts(args):
    import main as runner

    config_args = SimpleNamespace(
        agent_config=args.agent_config, dataset_config=args.dataset_config,
        artifact_root=str(args.artifact_root), chunk_size_ablation=0,
        max_test_queries_ablation=0, force=False, retry_failed_queries=False,
    )
    agent, dataset, result_path = runner.setup_configs_and_directories(config_args)
    if dataset.get("dataset", "").lower() != "locomo":
        raise ValueError("This runner only supports the selected LoCoMo comparison")
    started, contexts, questions = runner.create_agent_and_fetch_data(agent, dataset)
    if len(contexts) != args.context_count or len(questions) != args.context_count:
        raise ValueError("Unexpected context count from the unchanged dataset loader")
    if sum(map(len, questions)) != 1540:
        raise ValueError("The parallel runner requires all 1540 selected questions")
    return runner, agent, dataset, result_path, started, contexts, questions


def run_context(args):
    runner, agent, dataset, path, started, contexts, questions = load_contexts(args)
    index = args.context_index
    if not 0 <= index < len(contexts):
        raise ValueError("Context index is outside the original loaded dataset")
    metrics, results, completed, skipped = runner.initialize_progress_tracking(
        path, dataset, questions, False, False,
    )
    # Keep original global query IDs even though only one context runs here.
    offset = sum(len(group) for group in questions[:index])
    _, results, _, stopped = runner.process_context(
        index, contexts[index], questions[index], agent, dataset,
        metrics, results, offset, completed, skipped, 0, path, [], started,
        False, len(contexts),
    )
    sample = runner._resolve_context_sample_id(index, questions[index])
    if (stopped or len(results) != len(questions[index])
            or any(row.get("status") == "failed" for row in results)
            or any((row.get("eval_metadata") or {}).get("sample_id") != sample for row in results)):
        raise RuntimeError(f"Context {index} did not produce its exact successful QA set")
    write_report(args.output / "completion.json", {
        "context_index": index, "sample_id": sample,
        "questions": len(results), "result_path": str(Path(path).resolve()),
    })


def worker_command(args, index, directory):
    return [sys.executable, "-u", str(Path(__file__).resolve()),
            "--agent-config", args.agent_config, "--dataset-config", args.dataset_config,
            "--artifact-root", str(directory / "artifacts"), "--output", str(directory),
            "--context-count", str(args.context_count), "--context-index", str(index)]


def merge_journals(directories, name, destination):
    rows = []
    for directory in directories:
        source = directory / name
        if source.exists():
            rows.extend(read_jsonl(source))
    if rows:
        with Path(destination).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_parallel(args):
    indices = getattr(args, "context_indices", None)
    if indices is None:
        indices = list(range(args.context_count))
    if (not indices or indices != sorted(set(indices))
            or any(type(i) is not int or not 0 <= i < args.context_count for i in indices)):
        raise ValueError("Invalid context selection: require sorted, unique original context indices")
    full_dataset = indices == list(range(args.context_count))
    directories = {i: args.output / f"context_{i:02d}" for i in indices}
    for directory in directories.values():
        # A new measured run must never silently reuse state or overwrite a shard.
        directory.mkdir(parents=True, exist_ok=False)
    active, events = {}, []
    next_index, failed = 0, False
    try:
        while active or next_index < len(indices):
            while next_index < len(indices) and len(active) < args.workers:
                index = indices[next_index]
                directory = directories[index]
                env = os.environ.copy()
                env["METER_TIMING_JOURNAL"] = str(directory / "timing.jsonl")
                env["METER_AUXILIARY_JOURNAL"] = str(directory / "auxiliary.jsonl")
                log = (directory / "worker.log").open("ab")
                try:
                    process = subprocess.Popen(worker_command(args, index, directory),
                                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                finally:
                    log.close()
                active[index] = process
                events.append({"context": index, "event": "start", "time": time.time(), "pid": process.pid})
                print(json.dumps(events[-1]), flush=True)
                next_index += 1
            for index, process in list(active.items()):
                code = process.poll()
                if code is not None:
                    events.append({"context": index, "event": "end", "time": time.time(), "exit_code": code})
                    print(json.dumps(events[-1]), flush=True)
                    del active[index]
                    failed |= code != 0
            write_report(args.output / "parallel_status.json", {
                "workers": args.workers, "active_contexts": list(active),
                "events": events, "failed": failed, "complete": False,
            })
            if active:
                time.sleep(1)
    finally:
        for process in active.values():
            if process.poll() is None:
                process.terminate()
        for process in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for name, variable in (("timing.jsonl", "METER_TIMING_JOURNAL"),
                               ("auxiliary.jsonl", "METER_AUXILIARY_JOURNAL")):
            merge_journals(directories.values(), name, os.environ.get(variable, str(args.output / name)))
    if failed:
        raise RuntimeError("A context worker failed; preserved shard logs and state explain the failure")
    data, samples = [], set()
    for index, directory in directories.items():
        marker = json.loads((directory / "completion.json").read_text())
        if marker["context_index"] != index or marker["sample_id"] in samples:
            raise ValueError("Worker completion IDs are duplicated or inconsistent")
        samples.add(marker["sample_id"])
        rows = json.loads(Path(marker["result_path"]).read_text())["data"]
        if len(rows) != marker["questions"]:
            raise ValueError("Worker result count changed after completion")
        data.extend(rows)
    if full_dataset and len(data) != 1540:
        raise ValueError("Combined results do not contain all 1540 selected QA")
    result_name = "parallel_results.json" if full_dataset else "selected_results.json"
    write_report(args.output / result_name, {"data": data})
    write_report(args.output / "parallel_status.json", {
        "workers": args.workers, "active_contexts": [], "events": events,
        "failed": False, "complete": full_dataset,
        "selected_contexts_complete": True, "context_indices": indices,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--context-index", type=int)
    parser.add_argument("--context-indices", nargs="+", type=int)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= args.context_count <= 10:
        parser.error("Require 1 <= workers <= context-count <= 10")
    if args.inspect_only:
        runner, _, _, _, _, contexts, questions = load_contexts(args)
        print(json.dumps([{"index": i, "sample_id": runner._resolve_context_sample_id(i, group),
                           "turns": len(contexts[i]), "qa": len(group)} for i, group in enumerate(questions)]))
    elif args.context_index is not None:
        run_context(args)
    else:
        run_parallel(args)


if __name__ == "__main__":
    main()
