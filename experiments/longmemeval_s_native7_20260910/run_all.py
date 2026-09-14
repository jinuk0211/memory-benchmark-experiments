"""Sequential native-seven gate: one complete shared history, then all 500."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

METHODS = ("e_mem", "simplemem", "langmem", "mem0", "a_mem", "lightmem", "higmem")
DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def argument(command: list[str], name: str) -> str:
    if command.count(name) != 1 or command.index(name) + 1 >= len(command):
        raise ValueError(f"Command requires one separate {name} value")
    return command[command.index(name) + 1]


def validate_plan(plan: dict[str, Any]) -> tuple[list[str], dict[str, Path]]:
    raw = Path(plan["dataset"]).read_bytes()
    if plan["dataset_sha256"] != DATA_SHA256 or hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("Canonical LongMemEval-S dataset hash mismatch")
    ids = [item["question_id"] for item in json.loads(raw)]
    if len(ids) != 500 or any(not isinstance(qid, str) or not qid for qid in ids) or len(set(ids)) != 500:
        raise ValueError("Expected exactly 500 unique question IDs")
    receipt_spec = plan["runtime_receipt"]
    receipt_bytes = Path(receipt_spec["path"]).read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != receipt_spec["sha256"]:
        raise ValueError("Verified runtime receipt hash mismatch")
    if json.loads(receipt_bytes).get("status") != "runtime_verified":
        raise ValueError("Runtime source/GPU/service verification gate is not ready")
    commands = plan["runner_commands"]
    if set(commands) != set(METHODS):
        raise ValueError("Launch plan must contain exactly the seven native methods")
    directories = {}
    for method, command in commands.items():
        if not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command):
            raise ValueError(f"{method}: command must be an argv string array")
        if any(arg.split("=")[0] in {"--ids-file", "--limit", "--max-samples", "--max-queries", "--max-context-chunks"} for arg in command):
            raise ValueError(f"{method}: full command must not limit the population")
        if Path(argument(command, "--dataset")).resolve() != Path(plan["dataset"]).resolve():
            raise ValueError(f"{method}: command uses another dataset")
        if "--method" in command and argument(command, "--method") != method:
            raise ValueError(f"{method}: command names another method")
        directories[method] = Path(argument(command, "--run-dir")).resolve()
    if len(set(directories.values())) != 7:
        raise ValueError("Each method must have its own run directory")
    return ids, directories


def check_complete(method: str, status: dict[str, Any], expected: int) -> None:
    if status.get("method") != method:
        raise RuntimeError(f"{method}: completion status belongs to another method")
    if method == "lightmem":
        valid = (status.get("selected") == expected and status.get("completed") == expected
                 and status.get("population") == 500 and status.get("failed") == []
                 and (expected != 500 or status.get("generation_complete") is True))
    elif method == "higmem":
        valid = (status.get("selected") == expected and status.get("results") == expected
                 and status.get("failed_ids") == [] and status.get("run_complete") is True
                 and status.get("invalid_native_answers") == 0
                 and (expected != 500 or status.get("benchmark_complete") is True))
    else:
        valid = (status.get("planned") == expected and status.get("generated") == expected
                 and status.get("failed") == 0 and status.get("status") == "generation_complete")
    if not valid:
        raise RuntimeError(f"{method}: incomplete or invalid {expected}-history result: {status}")


def run(plan_path: Path, state_dir: Path) -> int:
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"status": "preflight", "officialjudge_pending": True, "completed_steps": []}
    try:
        plan = read_json(plan_path)
        ids, directories = validate_plan(plan)
        snapshot = state_dir / "launch_plan.json"
        if snapshot.exists() and read_json(snapshot) != plan:
            raise ValueError("Saved launch plan changed; use another queue directory")
        save_json(snapshot, plan)
        smoke_ids = state_dir / "smoke_ids.json"
        if smoke_ids.exists() and read_json(smoke_ids) != [ids[0]]:
            raise ValueError("Saved shared smoke history changed")
        save_json(smoke_ids, [ids[0]])
        state.update(population=500, methods=list(METHODS), smoke_question_id=ids[0])
        for phase, expected in (("smoke", 1), ("full", 500)):
            for method in METHODS:
                command = list(plan["runner_commands"][method])
                if phase == "smoke":
                    command.extend(["--ids-file", str(smoke_ids)])
                status_path = directories[method] / ("completion.json" if method == "higmem" else "status.json")
                previous = status_path.stat().st_mtime_ns if status_path.exists() else None
                log = state_dir / "logs" / f"{time.time_ns()}_{phase}_{method}.log"
                log.parent.mkdir(exist_ok=True)
                state.update(status="running", phase=phase, method=method, log=str(log))
                save_json(state_dir / "status.json", state)
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f"{phase}/{method} exited {result.returncode}; log: {log}")
                if not status_path.exists() or status_path.stat().st_mtime_ns == previous:
                    raise RuntimeError(f"{phase}/{method} did not write a fresh completion status")
                check_complete(method, read_json(status_path), expected)
                state["completed_steps"].append({"phase": phase, "method": method, "generated": expected})
                save_json(state_dir / "status.json", state)
                print(f"{phase}/{method}: {expected} complete", flush=True)
        total = sum(step["generated"] for step in state["completed_steps"] if step["phase"] == "full")
        if total != 3500:
            raise RuntimeError("Expected exactly 7 x 500 generated answers")
        state.update(status="generation_complete", generated=total, officially_judged=0)
        save_json(state_dir / "status.json", state)
        return 0
    except Exception as error:
        state.update(status="failed", error=str(error))
        save_json(state_dir / "status.json", state)
        with (state_dir / "errors.log").open("a", encoding="utf-8") as stream:
            stream.write(traceback.format_exc() + "\n")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    return run(args.plan, args.state_dir)


if __name__ == "__main__":
    raise SystemExit(main())
