"""Run the user-approved five-method population after verified service startup."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from materialize_inputs import write_once
from prepare_inputs import METHODS
from run_history import read_json, validate_service_receipt

EXPECTED_METHODS = ("e_mem", "simplemem", "mem0", "langmem", "a_mem")
PATH_ARGS = ("inputs", "reference", "source-root", "higmem-root", "tokenizer", "runtime-receipt")


def work_items(manifest: dict, protocol: dict) -> list[tuple[str, str]]:
    if tuple(METHODS) != EXPECTED_METHODS or tuple(manifest["required_methods"]) != EXPECTED_METHODS:
        raise ValueError("Only the five user-approved methods may run")
    ids = [row["question_id"] for row in manifest["questions"]]
    if len(ids) != 500 or len(set(ids)) != 500 or ids != protocol["selection"]["selected_ids"]:
        raise ValueError("Expected the exact frozen500 question order")
    return [(method, ids[0]) for method in METHODS] + [
        (method, qid) for method in METHODS for qid in ids[1:]]


def run(config: dict) -> None:
    root = Path(config["run_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt = read_json(Path(config["runtime_receipt"]))
    validate_service_receipt(receipt)
    manifest = read_json(Path(config["inputs"]) / "manifest.json")
    protocol = read_json(Path(config["reference"]))
    jobs = work_items(manifest, protocol)
    write_once(root / "launch_config.json", config)
    write_once(root / "population.json", dict(methods=list(METHODS),
        question_ids=protocol["selection"]["selected_ids"],
        planned_histories=len(jobs), official_judge_complete=False))
    lock = root / "population.running.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(__import__("os").getpid()))
    try:
        common = []
        for name in PATH_ARGS:
            common += ["--" + name, str(Path(config[name.replace("-", "_")]).resolve())]
        for name in ("llm-url", "embedding-url"):
            common += ["--" + name, config[name.replace("-", "_")]]
        for index, (method, qid) in enumerate(jobs, 1):
            if shutil.disk_usage(root).free < 4 * 1024**3:
                raise RuntimeError("Less than4GiB free; existing artifacts preserved")
            output = root / method / qid / "attempt1"
            output.mkdir(parents=True, exist_ok=True)
            command = [config["client_python"], "-B", str(Path(__file__).with_name("run_history.py")),
                       "--method", method, "--question-id", qid, "--output", str(output), *common]
            event = dict(index=index, total=len(jobs), method=method, question_id=qid)
            print(json.dumps(dict(event, status="history_start")), flush=True)
            with (output / "process.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            with (root / "execution.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(dict(event, exit_code=result.returncode, time=time.time())) + "\n")
            if result.returncode:
                raise RuntimeError(f"{method}/{qid} failed; preserved at {output}")
            print(json.dumps(dict(event, status="history_complete")), flush=True)
        for method in METHODS:
            rows = []
            for qid in protocol["selection"]["selected_ids"]:
                row = read_json(root / method / qid / "attempt1" / "prediction.json")
                if row["method"] != method or row["question_id"] != qid or not isinstance(row["hypothesis"], str):
                    raise ValueError("Prediction identity mismatch")
                rows.append(row)
            write_once(root / f"{method}.json", rows)
        write_once(root / "generation_complete.json", dict(
            status="five_baseline_generation_complete", methods=list(METHODS),
            questions_per_method=500, total_predictions=2500, official_judge_complete=False))
    finally:
        lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    args = parser.parse_args()
    run(read_json(args.launch_config))


if __name__ == "__main__":
    main()
