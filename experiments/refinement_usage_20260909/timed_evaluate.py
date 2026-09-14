"""Explicit, process-local timing for an unchanged evaluation runner."""
import argparse
import functools
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any

from evaluation_timing import TimingRecorder

ALLOWED_RUNNERS = {"run_transfer.py", "run_recursive.py", "run_recursive_v2.py"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_timed(runner: Path, arguments: list[str]) -> dict[str, Any]:
    runner = runner.resolve()
    if runner.name not in ALLOWED_RUNNERS or not runner.is_file():
        raise ValueError("Choose an existing supported evaluation runner")
    if not arguments or arguments[0] != "evaluate":
        raise ValueError("The timing launcher accepts only the evaluate stage")
    for name in ("run_transfer", "run_recursive", "transfer_runtime", "runtime_meter"):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve().parent != runner.parent:
            raise ValueError("Use a fresh process; a different project module is loaded: " + name)
    module_name = "_timed_evaluation_target"
    previous_module = sys.modules.get(module_name)
    previous_argv, previous_path = sys.argv, sys.path[:]
    api, original = None, None
    calls = 0
    receipt_root = None
    timed = None
    try:
        sys.path.insert(0, str(runner.parent))
        sys.argv = [str(runner), *arguments]
        spec = importlib.util.spec_from_file_location(module_name, runner)
        if spec is None or spec.loader is None:
            raise ValueError("Could not load the chosen runner")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        api = module if runner.name == "run_transfer.py" else module.baseline_api
        api_path = Path(api.__file__).resolve()
        if api_path != runner.parent / "run_transfer.py":
            raise ValueError("Evaluation API was imported from a different project")
        original = api.evaluate_sample
        original_source = inspect.getsourcefile(original)
        if original_source is None or Path(original_source).resolve() != api_path:
            raise ValueError("The evaluator is not the original project function")
        metadata = {
            "runner": str(runner), "runner_sha256": file_hash(runner),
            "runner_arguments": arguments[:],
            "evaluator_source": str(api_path), "evaluator_source_sha256": file_hash(api_path),
            "evaluator_function_sha256": hashlib.sha256(inspect.getsource(original).encode("utf-8")).hexdigest(),
            "evaluator_function_hash_format": "UTF-8 inspect.getsource(original)",
            "launcher_sha256": file_hash(Path(__file__)),
            "recorder_sha256": file_hash(Path(inspect.getfile(TimingRecorder))),
            "cache_policy": "Existing runtime cache lifecycle; no cache cleared or rewritten",
            "scope": "Original evaluate_sample call: embedding, retrieval, packing, generation and row assembly; excludes model initialization, outer item-cache skip checks, caller output persistence and timing sidecar writes",
        }

        @functools.wraps(original)
        def measured(rt: Any, units: Any, sample: dict, method: str, dataset: str) -> Any:
            nonlocal calls, receipt_root, timed
            target = Path(rt.args.out).resolve() / "evaluation_timing"
            if timed is None:
                receipt_root = target
                timed = TimingRecorder(target, metadata).wrap(original)
            elif target != receipt_root:
                raise ValueError("Runtime output changed within one timed launch")
            calls += 1
            return timed(rt, units, sample, method, dataset)

        api.evaluate_sample = measured
        try:
            module.main()
        except Exception as exc:
            if runner.name == "run_transfer.py":
                print("TRANSFER_FAILED " + repr(exc), flush=True)
            raise
        return {"evaluation_calls": calls, "receipt_root": str(receipt_root) if receipt_root else None,
                "note": "Zero calls does not establish zero evaluation time; existing item caches may have skipped evaluation"}
    finally:
        if api is not None and original is not None:
            api.evaluate_sample = original
        sys.argv = previous_argv
        sys.path[:] = previous_path
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("runner_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.runner_arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    print(json.dumps(run_timed(args.runner, arguments)))


if __name__ == "__main__":
    main()
