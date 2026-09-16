"""Bounded four-lane scheduling of the unchanged, sealed LangMem v6 worker."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

P = Path("/workspace/priority_fullcontext_langmem_20260912/resume_v100")
ROOT = Path("/workspace/longmemeval_s_native3_5090_20260910")
OLD = Path("/workspace/priority_fullcontext_langmem_20260912/old_native/runs/langmem_native_sessions_v6")
NATIVE = ROOT / "official_recovery/langmem_native_sessions_v6/runner.py"
NATIVE_SHA = "477c513ed0ed5f35cc37fcb5beb875dd3a2c2a8ba392539c8ac95ae7f2fca787"
OUT = ROOT / "runs/langmem_3090_resume_v6"
MAX_ATTEMPTS = 2
WORKER_TIMEOUT = 7200


def load_native():
    if hashlib.sha256(NATIVE.read_bytes()).hexdigest() != NATIVE_SHA:
        raise ValueError("Frozen LangMem worker changed")
    spec = importlib.util.spec_from_file_location("resume_native_langmem", NATIVE)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


def no_orphans():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if str(NATIVE).encode() in argv and b"--worker-dir" in argv:
            raise ValueError("Existing native LangMem worker must be reconciled before restart")


def job_identity(native, row, protocol):
    source = native.source_only(row)
    query = {key: row[key] for key in ("question_id", "question", "question_date")}
    return source, query, {"protocol_sha256": native.digest(protocol),
                          "source_sha256": native.digest(source), "query_sha256": native.digest(query)}


def history(root, qid):
    return root / "histories" / hashlib.sha256(qid.encode()).hexdigest()[:24]


def verified(native, target, identity, qid):
    result = native.verified(target, identity)
    if result is not None and result.get("question_id") != qid:
        raise ValueError("Verified prediction question ID mismatch")
    return result

def prepare(native):
    if native.sha(ROOT / "longmemeval_s_cleaned.json") != native.DATA_SHA256:
        raise ValueError("Canonical dataset changed")
    records = native.read_json(ROOT / "longmemeval_s_cleaned.json")
    ids = [row["question_id"] for row in records]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("Expected 500 canonical IDs")
    protocol = native.read_json(OLD / "protocol.json")
    if (protocol["population_ids"] != ids or protocol["source_files_sha256"] != native.source_hashes()
            or protocol["policy"] != native.POLICY or protocol["dataset_sha256"] != native.DATA_SHA256):
        raise ValueError("Recovered LangMem protocol/source differs")
    runtime = protocol["runtime"]
    if (runtime["api_base"] != "http://127.0.0.1:18083/v1"
            or runtime["embedding_api_base"] != "http://127.0.0.1:18084/v1"
            or runtime["model"] != native.MODEL or runtime["embedding_model"] != native.EMBEDDING_MODEL
            or runtime["embedding_dims"] != 384):
        raise ValueError("Recovered runtime endpoints or models differ")
    for name, expected in protocol["tokenizer_config_sha256"].items():
        if native.sha(Path(runtime["tokenizer"]) / name) != expected:
            raise ValueError("Tokenizer changed")
    native.client_versions()
    shards = [ids[lane::4] for lane in range(4)]
    manifest = {"instance_id": "51116981", "protocol_sha256": native.digest(protocol),
                "shards": shards, "worker_sha256": NATIVE_SHA,
                "coordinator_sha256": native.sha(Path(__file__)),
                "server_script_sha256": native.sha(Path("/workspace/priority_fullcontext_langmem_20260912/serve_priority.sh")),
                "router_sha256": native.sha(Path("/workspace/priority_fullcontext_langmem_20260912/langmem_router.py")),
                "max_attempts_per_history_in_recovered_ledger": MAX_ATTEMPTS,
                "worker_timeout_seconds": WORKER_TIMEOUT,
                "latest_ada_raw_results_unavailable": True}
    native.write_once(P / "langmem_shards.json", manifest)
    restored = 0
    for lane in range(4):
        run = OUT / f"lane{lane}"
        run.mkdir(parents=True, exist_ok=True)
        native.write_once(run / "protocol.json", protocol)
        native.write_once(run / "ids.json", shards[lane])
    for index, row in enumerate(records):
        qid = row["question_id"]
        old = history(OLD, qid)
        new = history(OUT / f"lane{index % 4}", qid)
        source, query, identity = job_identity(native, row, protocol)
        if old.exists():
            saved = verified(native, old, identity, qid)
            if not new.exists():
                shutil.copytree(old, new)
            if saved is not None:
                if verified(native, new, identity, qid) != saved:
                    raise ValueError("Restored completed artifact mismatch")
                restored += 1
    if restored != 92:
        raise ValueError(f"Expected 92 verified recovered LangMem histories, found {restored}")
    return records, protocol, restored


def run_one(native, row, lane, protocol, lock_fd):
    root = OUT / f"lane{lane}"
    target = history(root, row["question_id"])
    source, query, identity = job_identity(native, row, protocol)
    prior = verified(native, target, identity, row["question_id"])
    if prior is not None:
        return prior
    attempts = sorted(target.glob("attempt_*")) if target.exists() else []
    if attempts and not (attempts[-1] / "failure.json").exists():
        raise ValueError("Uncertain prior worker attempt requires reconciliation")
    if len(attempts) >= MAX_ATTEMPTS:
        raise ValueError("Attempt budget exhausted")
    attempt = native.next_attempt(target)
    for name, value in (("source.json", source), ("query.json", query),
                        ("worker.json", {"identity": identity, "runtime": protocol["runtime"],
                                         "source_files_sha256": protocol["source_files_sha256"]})):
        native.save_json(attempt / name, value)
    environment = os.environ.copy()
    for name in ("CONTAINER_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(name, None)
    started = time.time()
    native.save_json(attempt / "dispatch_execution.json", {
        "started_at": started, "instance_id": "51116981", "lane": lane,
        "coordinator_sha256": native.sha(Path(__file__))})
    with (attempt / "console.log").open("w", encoding="utf-8") as output:
        process = subprocess.Popen([sys.executable, str(NATIVE), "--worker-dir", str(attempt)],
                   stdout=output, stderr=subprocess.STDOUT, env=environment,
                   start_new_session=True, pass_fds=(lock_fd,))
        native.save_json(OUT / "executions" / f"{row['question_id']}_{attempt.name}.json", {
            "pid": process.pid, "attempt": str(attempt), "started_at": started,
            "intent_sha256": native.sha(attempt / "dispatch_execution.json")})
        try:
            code = process.wait(timeout=WORKER_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            completed_at_boundary = verified(native, target, identity, row["question_id"])
            if completed_at_boundary is not None:
                return completed_at_boundary
            if not (attempt / "failure.json").exists():
                native.save_json(attempt / "failure.json", {
                    "identity": identity, "error": "Worker wall-time budget exceeded",
                    "seconds": time.time() - started, "finished_at": time.time()})
            raise ValueError("Worker wall-time budget exceeded")
    prediction = verified(native, target, identity, row["question_id"])
    if code or prediction is None:
        raise ValueError(f"Native worker failed: exit={code}")
    return prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CONTAINER_ID") != "51116981":
        raise ValueError("Wrong instance")
    with (P / "langmem.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        no_orphans()
        native = load_native()
        records, protocol, restored = prepare(native)
        done, errors, queues = {}, [], [[] for _ in range(4)]
        for index, row in enumerate(records):
            _, _, identity = job_identity(native, row, protocol)
            saved = verified(native, history(OUT / f"lane{index % 4}", row["question_id"]), identity, row["question_id"])
            if saved:
                done[row["question_id"]] = saved
            else:
                queues[index % 4].append(row)

        def publish(state):
            native.save_json(OUT / "predictions.json", [done[r["question_id"]] for r in records if r["question_id"] in done])
            native.save_json(OUT / "status.json", {"status": state, "population": 500,
                "generated": len(done), "restored": restored, "failed": len(errors),
                "updated_at": time.time(), "officially_judged": 0})
            native.save_json(OUT / "failures.json", errors)

        if args.prepare_only or args.verify_only:
            publish("generation_complete" if len(done) == 500 else "prepared")
            print(json.dumps({"verified": len(done), "restored": restored, "remaining": 500 - len(done)}), flush=True)
            return 0 if args.prepare_only or len(done) == 500 else 1
        publish("running")
        with ThreadPoolExecutor(max_workers=4) as pool:
            inflight = {}
            for lane in range(4):
                if queues[lane]:
                    row = queues[lane].pop(0)
                    inflight[pool.submit(run_one, native, row, lane, protocol, lock.fileno())] = (lane, row)
            while inflight:
                finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in finished:
                    lane, row = inflight.pop(future)
                    try:
                        done[row["question_id"]] = future.result()
                    except Exception as error:
                        errors.append({"question_id": row["question_id"], "error": str(error)})
                    publish("running" if len(errors) < 8 else "draining_after_errors")
                    print(json.dumps({"generated": len(done), "failed": len(errors), "question_id": row["question_id"]}), flush=True)
                    if queues[lane] and len(errors) < 8:
                        following = queues[lane].pop(0)
                        inflight[pool.submit(run_one, native, following, lane, protocol, lock.fileno())] = (lane, following)
        success = len(done) == 500 and not errors
        publish("generation_complete" if success else "generation_incomplete")
        if success:
            native.save_json(OUT / "COMPLETE.json", {"generated": 500, "recovered": restored,
                "predictions_sha256": native.sha(OUT / "predictions.json"),
                "finished_at": time.time(), "officially_judged": 0})
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
