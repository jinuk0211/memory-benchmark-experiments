"""Two isolated, one-question processes of the unchanged native LightMem runner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
UPSTREAM_SHA256 = "32912049141844f3d174a731093d87fbf8b086984e53cdbca245a6099e807f07"
REUSE_ID = "e47becba"
MIN_FREE_BYTES = 2 * 1024**3


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def argument(command: list[str], name: str) -> str:
    if command.count(name) != 1 or command.index(name) + 1 >= len(command):
        raise ValueError(f"Expected one {name} argument")
    return command[command.index(name) + 1]


def preflight(args: argparse.Namespace) -> tuple[list[str], dict, dict, list[str], Path]:
    command = read(args.plan)["runner_commands"]["lightmem"]
    if (not isinstance(command, list) or len(command) < 2
            or any(not isinstance(part, str) for part in command)
            or Path(command[1]).name != "native_lightmem.py"):
        raise ValueError("Expected the frozen native LightMem argv")
    flags = {"--dataset", "--run-dir", "--source-root", "--api-base", "--model",
             "--embedding-model", "--compressor-model"}
    if len(command) != 2 + 2 * len(flags) or set(command[2::2]) != flags:
        raise ValueError("Runner arguments may not add limits or change native behavior")
    original = Path(argument(command, "--run-dir")).resolve()
    protocol_path = original / "protocol.json"
    if digest(protocol_path) != args.protocol_sha256:
        raise ValueError("Verified original protocol hash differs")
    protocol = read(protocol_path)
    dataset = Path(argument(command, "--dataset"))
    upstream = Path(argument(command, "--source-root")) / "experiments/longmemeval/run_lightmem_qwen.py"
    if (digest(dataset) != DATA_SHA256 or protocol["dataset_sha256"] != DATA_SHA256
            or digest(upstream) != UPSTREAM_SHA256 or protocol["upstream_sha256"] != UPSTREAM_SHA256
            or digest(Path(command[1])) != protocol["runner_sha256"]):
        raise ValueError("Frozen data/source/runner hash mismatch")
    for key in ("model", "embedding_model", "compressor_model"):
        if argument(command, "--" + key.replace("_", "-")) != protocol[key]:
            raise ValueError(f"Command/protocol mismatch: {key}")
    rows = read(dataset)
    ids = [row["question_id"] for row in rows]
    if (len(ids) != 500 or len(set(ids)) != 500
            or any(not isinstance(qid, str) or not qid or Path(qid).name != qid
                   or qid in (".", "..") for qid in ids) or REUSE_ID not in ids):
        raise ValueError("Expected the full 500 canonical safe question IDs")
    return ids, {row["question_id"]: row for row in rows}, protocol, command, original


def validate_prediction(target: Path, item: dict, protocol: dict) -> bytes:
    if read(target.parent / "protocol.json") != protocol:
        raise ValueError(f"Lane/original protocol mismatch: {target.parent}")
    raw = (target / "prediction.json").read_bytes()
    prediction = json.loads(raw)
    if (prediction.get("question_id") != item["question_id"]
            or not isinstance(prediction.get("hypothesis"), str)
            or not prediction["hypothesis"].strip()
            or {"answer", "reference", "reference_answer", "autoeval_label", "label",
                "question_type"}.intersection(prediction)):
        raise ValueError(f"Invalid native prediction: {target}")
    for key in ("dataset_sha256", "model", "embedding_model", "upstream_sha256"):
        if prediction.get(key) != protocol[key]:
            raise ValueError(f"Prediction/protocol mismatch: {target}/{key}")
    attempt = prediction.get("attempt")
    if not isinstance(attempt, str) or not attempt.startswith("attempt_") or Path(attempt).name != attempt:
        raise ValueError("Unsafe or missing native attempt")
    source = {key: item[key] for key in ("haystack_dates", "haystack_session_ids")}
    source["haystack_sessions"] = [[{"role": turn["role"], "content": turn["content"]}
                                    for turn in session] for session in item["haystack_sessions"]]
    if read(target / attempt / "source.json") != source:
        raise ValueError(f"Native source differs from the complete canonical history: {target}")
    if read(target / attempt / "construction.json")["source_turns_supplied"] != sum(
            len(session) for session in source["haystack_sessions"]):
        raise ValueError(f"Native construction source count differs: {target}")
    return raw


def schedule(ids: list[str], rows: dict, protocol: dict, command: list[str],
             state_dir: Path, original: Path, workers: int, lock_fd: int,
             retry_failed: bool = False) -> dict[str, Path]:
    lanes = state_dir / "lightmem_lanes"
    lanes.mkdir(exist_ok=True)
    for lane in range(workers):
        (lanes / f"lane_{lane}").mkdir(exist_ok=True)
    journal_path = state_dir / "lightmem_attempts.json"
    journal = read(journal_path) if journal_path.exists() else {}
    if not set(journal).issubset(ids):
        raise ValueError("Attempt journal contains noncanonical IDs")
    completed = {}
    for qid in ids:
        targets = ([original / qid] if qid == REUSE_ID else [])
        targets += list(lanes.glob(f"lane_*/{qid}"))
        candidates = [target for target in targets if (target / "prediction.json").exists()]
        if len(candidates) > 1:
            raise ValueError(f"Duplicate completed ID: {qid}")
        if candidates:
            validate_prediction(candidates[0], rows[qid], protocol)
            completed[qid] = candidates[0]
    if REUSE_ID in ids and REUSE_ID not in completed:
        raise ValueError("Verified original smoke prediction is required")
    pending = [qid for qid in ids if qid not in completed and (qid not in journal
               or retry_failed and journal[qid]["attempt"] < 2)]
    failures = {qid: entry for qid, entry in journal.items() if qid not in completed}
    active = {}
    disk_blocked = False

    def status() -> None:
        save(state_dir / "lightmem_status.json", {
            "method": "lightmem", "planned": len(ids), "population": 500,
            "generated": len(completed), "failed": len(failures), "failed_ids": list(failures),
            "inflight": [job[0] for job in active.values()], "pending": len(pending),
            "status": "disk_blocked" if disk_blocked else (
                "generation_complete" if len(completed) == 500 else "running" if active or pending else "incomplete"),
            "generation_complete": len(completed) == 500,
            "official_judge_pending": True, "time": time.time()})

    status()
    while pending or active:
        for lane in range(workers):
            if lane in active or not pending or disk_blocked:
                continue
            if shutil.disk_usage(state_dir).free < MIN_FREE_BYTES:
                disk_blocked = True
                break
            qid = pending.pop(0)
            run_dir = lanes / f"lane_{lane}"
            ids_file = state_dir / f"lightmem_lane_{lane}_ids.json"
            save(ids_file, [qid])
            argv = list(command)
            argv[argv.index("--run-dir") + 1] = str(run_dir)
            argv += ["--ids-file", str(ids_file)]
            log = state_dir / f"lightmem_{qid}_{time.time_ns()}.log"
            previous = journal.get(qid)
            history = (previous["history"] + [
                {key: value for key, value in previous.items() if key != "history"}]) if previous else []
            journal[qid] = {"lane": lane, "status": "dispatched", "log": str(log), "time": time.time(),
                            "attempt": previous["attempt"] + 1 if previous else 1, "history": history}
            failures.pop(qid, None)
            save(journal_path, journal)
            try:
                with log.open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                               pass_fds=(lock_fd,))
                active[lane] = (qid, process, run_dir / qid)
            except OSError as error:
                journal[qid].update(status="failed", error=str(error))
                failures[qid] = journal[qid]
                if retry_failed and journal[qid]["attempt"] < 2:
                    pending.append(qid)
                save(journal_path, journal)
        for lane, (qid, process, target) in list(active.items()):
            code = process.poll()
            if code is None and time.time() - journal[qid]["time"] >= 5400:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                code = 124
            if code is None:
                continue
            del active[lane]
            try:
                if code:
                    raise RuntimeError(f"Native process exited {code}")
                validate_prediction(target, rows[qid], protocol)
                completed[qid] = target
                journal[qid].update(status="generated", returncode=code)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                journal[qid].update(status="failed", returncode=code, error=str(error))
                failures[qid] = journal[qid]
                if retry_failed and journal[qid]["attempt"] < 2:
                    pending.append(qid)
            save(journal_path, journal)
        status()
        if disk_blocked and not active:
            break
        if active:
            time.sleep(2)
    return completed


def aggregate(ids: list[str], rows: dict, protocol: dict, completed: dict[str, Path],
              destination: Path) -> None:
    if len(ids) != 500 or len(set(ids)) != 500 or set(completed) != set(ids):
        raise ValueError("Aggregation requires exact full canonical 500 coverage")
    validated = {qid: validate_prediction(completed[qid], rows[qid], protocol) for qid in ids}
    if destination.exists():
        if (read(destination / "protocol.json") != protocol
                or {p.parent.name for p in destination.glob("*/prediction.json")} != set(ids)
                or any((destination / qid / "prediction.json").read_bytes() != validated[qid] for qid in ids)
                or read(destination / "status.json").get("generation_complete") is not True):
            raise ValueError("Existing aggregate differs; it was preserved")
        return
    staging = destination.with_name(f".{destination.name}.assembling_{time.time_ns()}")
    staging.mkdir()
    save(staging / "protocol.json", protocol)
    provenance = {}
    for qid in ids:
        target = staging / qid
        target.mkdir()
        (target / "prediction.json").write_bytes(validated[qid])
        provenance[qid] = {"source_dir": str(completed[qid].resolve()),
                           "prediction_sha256": hashlib.sha256(validated[qid]).hexdigest()}
    save(staging / "aggregation_receipt.json", {"question_ids": ids, "sources": provenance})
    save(staging / "status.json", {"method": "lightmem", "selected": 500, "completed": 500,
         "population": 500, "failed": [], "generation_complete": True, "official_judge_pending": True})
    staging.rename(destination)


def finish_outputs(ids: list[str], command: list[str], vendor: Path,
                   destination: Path, state_dir: Path) -> None:
    base = Path(command[1]).parent
    dataset = Path(argument(command, "--dataset"))
    output = state_dir / "lightmem_full500.hypotheses.jsonl"
    receipt_path = output.with_name(output.name + ".receipt.json")

    def preserve_partial(paths: tuple[Path, ...]) -> None:
        stamp = time.time_ns()
        for path in paths:
            if path.exists():
                path.rename(path.with_name(f"{path.name}.interrupted_{stamp}"))

    def validate_export(hypotheses: Path, receipt_file: Path) -> None:
        receipt = read(receipt_file)
        if (receipt.get("schema") != "native-seven-official-hypotheses-v1"
                or receipt.get("method") != "lightmem" or receipt.get("scope") != "full_canonical_500"
                or receipt.get("expected_question_ids") != ids
                or receipt.get("hypotheses_sha256") != digest(hypotheses)
                or receipt.get("exporter_sha256") != digest(base / "export_official.py")
                or not receipt.get("inputs_sha256")
                or any(digest(Path(path)) != expected for path, expected in receipt["inputs_sha256"].items())):
            raise ValueError("Existing export provenance differs; all outputs preserved")

    if output.exists() != receipt_path.exists():
        preserve_partial((output, receipt_path))
    if receipt_path.exists():
        try:
            read(receipt_path)
        except json.JSONDecodeError:
            preserve_partial((output, receipt_path))
    if not output.exists():
        staging = state_dir / f"lightmem_export_attempt_{time.time_ns()}"
        staging.mkdir()
        staged_output = staging / output.name
        staged_receipt = staging / receipt_path.name
        subprocess.run([command[0], str(base / "export_official.py"), "--method", "lightmem",
                        "--run-dir", str(destination), "--dataset", str(dataset),
                        "--vendor", str(vendor), "--output", str(staged_output)], check=True)
        validate_export(staged_output, staged_receipt)
        staged_output.rename(output)
        staged_receipt.rename(receipt_path)
    validate_export(output, receipt_path)
    score_path = state_dir / "lightmem_diagnostic_f1.json"
    if score_path.exists():
        try:
            read(score_path)
        except json.JSONDecodeError:
            preserve_partial((score_path,))
    candidate = score_path
    if not score_path.exists():
        candidate = state_dir / f"lightmem_diagnostic_attempt_{time.time_ns()}.json"
        subprocess.run([command[0], str(base / "score_diagnostic_f1.py"), "--dataset",
                        str(dataset), "--hypotheses", str(output), "--output", str(candidate)], check=True)
    score = read(candidate)
    expected_inputs = {str(path): digest(path) for path in (dataset, output, receipt_path)}
    if (score.get("schema") != "longmemeval-s-diagnostic-token-f1-v1" or score.get("method") != "lightmem"
            or score.get("provenance", {}).get("inputs_sha256") != expected_inputs
            or score.get("provenance", {}).get("scorer_sha256") != digest(base / "score_diagnostic_f1.py")
            or score.get("summary", {}).get("overall", {}).get("count") != 500):
        raise ValueError("Existing diagnostic provenance differs; all outputs preserved")
    if candidate != score_path:
        candidate.rename(score_path)


def main() -> int:
    import fcntl  # Runtime is Linux; import here so CPU-only unit tests also run on Windows.

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--retry-failed", action="store_true", help="Allow at most two total attempts per ID")
    args = parser.parse_args()
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "lightmem_queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ids, rows, protocol, command, original = preflight(args)
        base = Path(command[1]).parent
        support = {str(base / name): digest(base / name) for name in (
            "export_official.py", "score_diagnostic_f1.py", "run_all.py")}
        config = {"command": command, "protocol_sha256": args.protocol_sha256,
                  "canonical_ids": ids, "maximum_attempts_per_question": 2, "support_sha256": support}
        config_path = args.state_dir / "lightmem_queue_config.json"
        if config_path.exists() and read(config_path) != config:
            raise ValueError("Saved queue configuration differs")
        save(config_path, config)
        completed = schedule(ids, rows, protocol, command, args.state_dir, original,
                             args.workers, lock.fileno(), args.retry_failed)
        if len(completed) != 500:
            return 1
        preflight(args)
        if any(digest(Path(path)) != expected for path, expected in support.items()):
            raise ValueError("Frozen export/scoring support changed during generation")
        destination = original.parent / "lightmem_fast_native2"
        aggregate(ids, rows, protocol, completed, destination)
        finish_outputs(ids, command, args.vendor, destination, args.state_dir)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
