"""Durably observe a running queue without modifying its pinned code or artifacts."""

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.locomo_server_queue import gpu_sample, supervisor_state
from scripts.score_locomo_comparison import (
    runtime_metrics,
    summarize_usage,
    write_report,
)


def read_closed_rows(path: Path) -> tuple[list[dict], bool]:
    """Do not parse a request record that its writer has not finished appending."""
    if not path.exists():
        return [], False
    data = path.read_bytes()
    lines = data.split(b"\n")
    return [json.loads(line) for line in lines[:-1] if line.strip()], bool(lines[-1])


def read_state(output: Path) -> dict:
    """Read the queue's atomically replaced status, including pre-start absence."""
    path = output / "recovery_status.json"
    return json.loads(path.read_text()) if path.exists() else {}


def capture(plan: dict, journal: Path, session: str) -> dict:
    """Flush every GPU sample independently of queue/proxy shutdown behavior."""
    before = read_state(Path(plan["output"]))
    sample = gpu_sample()
    after = read_state(Path(plan["output"]))
    stable = before.get("method") == after.get("method") and before.get("state") == after.get("state")
    sample.update(timestamp=time.time(), run_id=plan["run_id"], session_id=session,
                  method=after.get("method") if stable and after.get("state") == "running" else None,
                  attribution_uncertain=not stable)
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sample, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sample


def observed_gpu(rows: list[dict], method: str, hourly_rate: float) -> dict:
    """Integrate only adjacent, same-session observations; never extrapolate."""
    good = [row for row in rows if row.get("method") == method and not row.get("unavailable")]
    seconds = energy = 0.0
    for first, last in pairwise(rows):
        if (first.get("method") != method or last.get("method") != method
                or first.get("session_id") != last.get("session_id")
                or first.get("unavailable") or last.get("unavailable")):
            continue
        elapsed = last["monotonic_s"] - first["monotonic_s"]
        if 0 < elapsed <= 15:
            seconds += elapsed
            energy += elapsed * (first["power_w"] + last["power_w"]) / 7200
    return {"whole_attempt_coverage": False, "covered_seconds": seconds,
            "gpu_energy_wh": energy if seconds else None,
            "estimated_rental_cost_usd": seconds * hourly_rate / 3600 if seconds else None,
            "peak_vram_mib": max((row["vram_mib"] for row in good), default=None),
            "mean_gpu_utilization_percent": sum(row["utilization"] for row in good) / len(good) if good else None,
            "samples": len(good), "note": "Observed intervals only; gaps and method/session boundaries excluded. "
            "5-second power samples integrated by trapezoids, not a hardware energy counter. "
            "Not an additional charge: overlaps primary runtime metrics. CPU energy and provider extras excluded."}


def snapshot(plan: dict, output: Path, samples: list[dict]) -> None:
    """Publish cumulative usage plus separately labelled primary and partial GPU metrics."""
    for item in plan["methods"]:
        method = item["method"]
        source = Path(plan["output"]) / method
        if not source.is_dir():
            continue
        usage, pending = [], False
        for name in ("llm_usage.jsonl", "embedding_usage.jsonl"):
            rows, tail = read_closed_rows(source / name)
            usage.extend(rows)
            pending |= tail
        if any(row.get("run_id") != plan["run_id"] or row.get("method") != method for row in usage):
            raise ValueError(f"Usage provenance mismatch: {method}")
        phases = defaultdict(lambda: defaultdict(float))
        timing_paths = list(source.glob("context_*/timing.jsonl")) or [source / "timing.jsonl"]
        for path in timing_paths:
            rows, _ = read_closed_rows(path)
            for row in rows:
                if row.get("event") == "end":
                    phases[row["phase"]][row["status"]] += row["duration_s"]
        primary = source / "telemetry.json"
        report = {"run_id": plan["run_id"], "method": method, "updated_at": time.time(),
                  "whole_attempt_metering_complete": False, "journal_has_inflight_tail": pending,
                  "usage": summarize_usage(usage, True), "phase_seconds": dict(phases),
                  "phase_note": "Completed operation durations by status, summed across overlapping workers; "
                  "not wall time. Active and interrupted operations may lack end records.",
                  "primary_runtime": runtime_metrics([], primary, plan["hourly_rate_usd"]) if primary.exists() else None,
                  "observed_gpu": observed_gpu(samples, method, plan["hourly_rate_usd"]),
                  "note": "Live snapshot, not a completed evaluation. Includes journaled failed requests. "
                  "Missing/unrecorded requests cannot be inferred as zero. Original artifacts are unchanged."}
        write_report(output / f"{method}.json", report)


def main(argv: list[str] | None = None) -> int:
    """Run as an independent Supervisor job; no model API calls or artifact edits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue", required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if args.output.resolve().is_relative_to(Path(plan["output"]).resolve()):
        raise ValueError("Observer must write outside the original experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    session, samples, next_snapshot = str(uuid.uuid4()), [], 0.0
    while True:
        samples.append(capture(plan, args.output / "gpu_samples.jsonl", session))
        state = supervisor_state(args.queue)
        stopped = state not in {"RUNNING", "STARTING"}
        if stopped or time.monotonic() >= next_snapshot:
            error = None
            try:
                snapshot(plan, args.output, samples)
            except (ValueError, OSError, KeyError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            write_report(args.output / "observer_status.json", {"updated_at": time.time(),
                         "queue_state": state, "snapshot_error": error, "samples": len(samples),
                         "run_id": plan["run_id"], "session_id": session})
            if stopped:
                return 1 if error else 0
            next_snapshot = time.monotonic() + 60
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
