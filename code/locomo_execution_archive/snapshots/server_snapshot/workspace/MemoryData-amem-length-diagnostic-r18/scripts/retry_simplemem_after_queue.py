"""Run the isolated SimpleMem compatibility retry after the primary queue exits."""

import argparse
import errno
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_locomo_comparison as finalizer
from scripts import locomo_server_queue as queue


def validate_retry(previous: dict, retry: dict, final_output: Path) -> None:
    """Keep the original experiment settings and reject artifact path overlap."""
    shared = ("model", "dtype", "model_revision", "embedding_path", "dataset",
              "scorer", "context_workers", "hf_home", "higmem_run", "server_python",
              "hourly_rate_usd", "methods")
    if any(previous[key] != retry[key] for key in shared):
        raise ValueError("Retry changed a shared experiment setting")
    if previous["client_python"] == retry["client_python"]:
        raise ValueError("Retry must use its isolated compatibility environment")
    if previous["run_id"] == retry["run_id"]:
        raise ValueError("Retry must have a separate metering run ID")
    enabled = [item["method"] for item in retry["methods"]
               if item["method"] not in retry.get("skip_methods", {})]
    if enabled != ["simplemem"]:
        raise ValueError("Only SimpleMem may run in this retry")
    paths = [Path(previous["output"]), Path(retry["output"]), final_output]
    if any(not path.is_absolute() for path in paths):
        raise ValueError("Artifact paths must be absolute")
    resolved = [path.resolve() for path in paths]
    for index, left in enumerate(resolved):
        for right in resolved[index + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Artifact directories must not overlap")
    if any(path.exists() for path in resolved[1:]):
        raise FileExistsError("Refusing to reuse retry or finalization output")


def require_free_ports() -> None:
    """Do not contend with another embedding server or either usage proxy."""
    for port in (18081, 18082, 18083):
        with socket.socket() as probe:
            probe.settimeout(2)
            if probe.connect_ex(("127.0.0.1", port)) != errno.ECONNREFUSED:
                raise RuntimeError(f"Cannot establish that comparison port {port} is free")


def run_retry(previous_path: Path, retry_path: Path, final_output: Path,
              check_only: bool = False) -> int:
    """Wait, execute once, then apply the existing full scoring/accounting gates."""
    plans = [json.loads(path.read_text()) for path in (previous_path, retry_path)]
    previous, retry = plans
    hashes = {path: finalizer.digest(path) for path in (previous_path, retry_path)}
    for plan in plans:
        queue.validate_plan(plan)
    validate_retry(previous, retry, final_output)
    if check_only:
        print("SimpleMem retry plan valid; runtime ports are checked only after waiting.", flush=True)
        return 0
    print("Waiting for the primary queue and all its artifact writers to exit.", flush=True)
    finalizer.wait_for_queue(previous, True)
    if any(finalizer.digest(path) != digest for path, digest in hashes.items()):
        raise ValueError("A plan changed while the retry was waiting")
    for plan in plans:
        queue.validate_plan(plan)
    validate_retry(previous, retry, final_output)
    require_free_ports()
    result = subprocess.run([retry["server_python"], str(ROOT / "scripts/locomo_server_queue.py"),
                             "--plan", str(retry_path)], cwd=ROOT, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Retry queue terminated abnormally: {result.returncode}")
    # Exit 1 can be the known exact-template metadata mismatch; re-audit raw data.
    result = subprocess.run([retry["server_python"], str(ROOT / "scripts/finalize_locomo_comparison.py"),
                             "--plan", str(retry_path), "--output", str(final_output)],
                            cwd=ROOT, check=False)
    if result.returncode != 0:
        return result.returncode
    summary = json.loads((final_output / "summary.json").read_text())
    outcomes = [row for row in summary["methods"] if row["method"] == "simplemem"]
    complete = (summary.get("finalization_complete") is True and len(outcomes) == 1
                and outcomes[0].get("state") == "complete")
    print(json.dumps({"simplemem_retry_complete": complete, "final_output": str(final_output)}), flush=True)
    return 0 if complete else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-plan", required=True, type=Path)
    parser.add_argument("--retry-plan", required=True, type=Path)
    parser.add_argument("--final-output", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    return run_retry(args.previous_plan, args.retry_plan, args.final_output, args.check_only)


if __name__ == "__main__":
    raise SystemExit(main())
