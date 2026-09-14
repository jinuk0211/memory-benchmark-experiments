"""Check process isolation and original QA ordering in the parallel launcher."""

import json
import hashlib
import sys
from types import SimpleNamespace

import pytest

from scripts import run_locomo_parallel as parallel
from scripts import locomo_server_queue as queue


def test_worker_uses_original_context_and_global_query_offset(tmp_path, monkeypatch):
    calls = []
    path = tmp_path / "results.json"
    questions = [["a", "b", "c"], ["d", "e"]]

    def process(*args):
        calls.append(args)
        rows = [{"eval_metadata": {"sample_id": "conv-original"}} for _ in args[2]]
        path.write_text(json.dumps({"data": rows}))
        return {}, rows, 5, False

    runner = SimpleNamespace(
        initialize_progress_tracking=lambda *args: ({}, [], set(), set()),
        process_context=process,
        _resolve_context_sample_id=lambda index, group: "conv-original",
    )
    monkeypatch.setattr(parallel, "load_contexts", lambda args: (
        runner, {"agent": 1}, {"dataset": 1}, path, 1.0, [["turn0"], ["turn1", "turn2"]], questions,
    ))
    parallel.run_context(SimpleNamespace(context_index=1, output=tmp_path))
    assert len(calls) == 1
    assert calls[0][0:3] == (1, ["turn1", "turn2"], ["d", "e"])
    assert calls[0][7] == 3  # Original global query offset, not shard-local zero.
    assert calls[0][10] == 0  # No QA limit.
    assert calls[0][-2:] == (False, 2)  # No forced memory rebuild; original context count.
    marker = json.loads((tmp_path / "completion.json").read_text())
    assert marker["context_index"] == 1
    assert marker["questions"] == 2


@pytest.fixture
def stub_workers(tmp_path, monkeypatch):
    script = tmp_path / "stub_worker.py"
    script.write_text('''import json, os, pathlib, sys, time
index, directory = int(sys.argv[1]), pathlib.Path(sys.argv[2])
started = time.time()
assert pathlib.Path(os.environ["METER_TIMING_JOURNAL"]).parent == directory
with open(os.environ["METER_TIMING_JOURNAL"], "w") as f:
    f.write(json.dumps({"context": index, "event": "start"}) + "\\n")
if os.environ.get("FAIL_CONTEXT") == str(index):
    raise SystemExit(7)
time.sleep(0.5)
rows = [{"context": index, "ordinal": n} for n in range(770)]
result = directory / "results.json"
result.write_text(json.dumps({"data": rows}))
(directory / "completion.json").write_text(json.dumps({
    "context_index": index, "sample_id": str(index), "questions": 770,
    "result_path": str(result), "started": started, "ended": time.time()}))
''')
    monkeypatch.setattr(parallel, "worker_command", lambda args, index, directory:
                        [sys.executable, str(script), str(index), str(directory)])
    return SimpleNamespace(output=tmp_path / "run", context_count=2, workers=2)


def test_processes_overlap_and_merge_in_original_context_order(stub_workers):
    args = stub_workers
    parallel.run_parallel(args)
    markers = [json.loads((args.output / f"context_{i:02d}" / "completion.json").read_text())
               for i in range(2)]
    assert max(row["started"] for row in markers) < min(row["ended"] for row in markers)
    rows = json.loads((args.output / "parallel_results.json").read_text())["data"]
    assert len(rows) == 1540
    assert rows[:770] == [{"context": 0, "ordinal": i} for i in range(770)]
    assert rows[770:] == [{"context": 1, "ordinal": i} for i in range(770)]
    assert len(parallel.read_jsonl(args.output / "timing.jsonl")) == 2
    assert json.loads((args.output / "parallel_status.json").read_text())["complete"]


def test_failed_worker_preserves_logs_and_never_emits_complete_results(stub_workers, monkeypatch):
    monkeypatch.setenv("FAIL_CONTEXT", "0")
    with pytest.raises(RuntimeError, match="context worker failed"):
        parallel.run_parallel(stub_workers)
    assert not (stub_workers.output / "parallel_results.json").exists()
    assert (stub_workers.output / "context_00" / "worker.log").exists()
    assert parallel.read_jsonl(stub_workers.output / "timing.jsonl")


def test_existing_worker_state_is_not_overwritten(stub_workers):
    saved = stub_workers.output / "context_00"
    saved.mkdir(parents=True)
    marker = saved / "existing-data.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        parallel.run_parallel(stub_workers)
    assert marker.read_text() == "preserve"


def test_malformed_worker_journal_is_not_silently_dropped(tmp_path):
    shard = tmp_path / "context_00"
    shard.mkdir()
    (shard / "timing.jsonl").write_text('{"unfinished":')
    with pytest.raises(ValueError, match="invalid JSON"):
        parallel.merge_journals([shard], "timing.jsonl", tmp_path / "merged.jsonl")
    assert not (tmp_path / "merged.jsonl").exists()


@pytest.mark.parametrize("skips,valid", [
    ({"a_mem": "Observed full-context output truncation in preserved attempt"}, True),
    ({"a_mem": ""}, False), ({"invented": "failure"}, False), (["a_mem"], False),
])
def test_only_explicit_known_method_exclusions_are_accepted(tmp_path, monkeypatch, skips, valid):
    source = tmp_path / "pinned.json"
    source.write_text("{}")
    plan = {"schema_version": 1, "model": queue.MODEL, "output": str(tmp_path), "skip_methods": skips,
            "methods": [{"method": method, "agent_config": str(source), "dataset_config": str(source)}
                        for method in queue.METHODS],
            **{key: str(source) for key in ("dataset", "scorer", "client_python", "server_python")},
            "file_sha256": {str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()}}
    monkeypatch.setattr(queue, "expected_questions", lambda *args: {})
    if valid:
        queue.validate_plan(plan)
    else:
        with pytest.raises(ValueError, match="Skipped methods"):
            queue.validate_plan(plan)
