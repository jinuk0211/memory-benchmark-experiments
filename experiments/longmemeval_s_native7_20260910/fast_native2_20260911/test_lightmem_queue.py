"""Behavioral checks for the external LightMem queue; no GPU or native imports."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lightmem_queue as queue


@pytest.fixture
def setup(tmp_path, monkeypatch):
    ids = [queue.REUSE_ID, "q1", "q2", "q3"]
    protocol = {"dataset_sha256": queue.DATA_SHA256, "upstream_sha256": queue.UPSTREAM_SHA256,
                "model": "Qwen/Qwen3.5-9B", "embedding_model": "/models/minilm"}
    rows = {qid: {"question_id": qid, "haystack_dates": ["2026-01-01"],
                 "haystack_session_ids": ["s1"], "haystack_sessions": [[
                     {"role": "user", "content": "All of the original history"},
                     {"role": "assistant", "content": "Preserve this too"}]]} for qid in ids}
    original = tmp_path / "original"
    state = tmp_path / "state"
    original.mkdir()
    state.mkdir()
    command = ["python", "native_lightmem.py", "--run-dir", str(original), "--dataset", "canonical.json"]
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * 1024**3))
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: None)

    def write_result(target, qid):
        target.mkdir(parents=True, exist_ok=True)
        queue.save(target.parent / "protocol.json", protocol)
        attempt = target / "attempt_1"
        attempt.mkdir(exist_ok=True)
        source = {key: rows[qid][key] for key in (
            "haystack_sessions", "haystack_dates", "haystack_session_ids")}
        queue.save(attempt / "source.json", source)
        queue.save(attempt / "construction.json", {"source_turns_supplied": 2})
        result = dict(protocol, question_id=qid, hypothesis="native answer", attempt="attempt_1")
        queue.save(target / "prediction.json", result)

    write_result(original / queue.REUSE_ID, queue.REUSE_ID)
    return SimpleNamespace(ids=ids, rows=rows, protocol=protocol, original=original,
                           state=state, command=command, write_result=write_result)


def fake_native(monkeypatch, setup, failed=()):
    launched, running, peak = [], set(), []

    def popen(argv, **kwargs):
        selected = queue.read(Path(queue.argument(argv, "--ids-file")))
        assert len(selected) == 1
        qid = selected[0]
        lane = Path(queue.argument(argv, "--run-dir"))
        assert lane not in running, "Two processes wrote the same lane concurrently"
        assert kwargs["pass_fds"] == (123,)
        running.add(lane)
        peak.append(len(running))
        launched.append((qid, lane, argv))

        class Process:
            calls = 0

            def poll(self):
                self.calls += 1
                if self.calls == 1:
                    return None
                running.remove(lane)
                if qid in failed:
                    return 1
                setup.write_result(lane / qid, qid)
                return 0

        return Process()

    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    return launched, peak


def run(setup, retry=False):
    return queue.schedule(setup.ids, setup.rows, setup.protocol, setup.command,
                          setup.state, setup.original, 2, 123, retry)


def test_disjoint_lanes_preserve_canonical_order_and_reuse_smoke(setup, monkeypatch):
    original_bytes = (setup.original / queue.REUSE_ID / "prediction.json").read_bytes()
    launched, peak = fake_native(monkeypatch, setup)
    completed = run(setup)
    assert list(completed) == setup.ids
    assert [job[0] for job in launched] == setup.ids[1:]
    assert max(peak) == 2
    for _, lane, argv in launched:
        restored = argv[:-2]
        restored[restored.index("--run-dir") + 1] = str(setup.original)
        assert restored == setup.command
        assert lane.parent == setup.state / "lightmem_lanes"
    assert (setup.original / queue.REUSE_ID / "prediction.json").read_bytes() == original_bytes
    status = queue.read(setup.state / "lightmem_status.json")
    assert status["generated"] == 4
    assert status["generation_complete"] is False
    assert status["inflight"] == []


def test_failure_does_not_block_other_ids_or_retry_without_permission(setup, monkeypatch):
    launched, _ = fake_native(monkeypatch, setup, failed={"q1"})
    completed = run(setup)
    assert set(completed) == {queue.REUSE_ID, "q2", "q3"}
    assert [job[0] for job in launched] == ["q1", "q2", "q3"]
    status = queue.read(setup.state / "lightmem_status.json")
    assert status["failed_ids"] == ["q1"]
    assert status["generated"] == 3 and not status["generation_complete"]
    launched.clear()
    assert run(setup) == completed
    assert launched == []
    resumed, _ = fake_native(monkeypatch, setup)
    assert set(run(setup, retry=True)) == set(setup.ids)
    assert [job[0] for job in resumed] == ["q1"]
    record = queue.read(setup.state / "lightmem_attempts.json")["q1"]
    assert record["attempt"] == 2 and record["history"][0]["status"] == "failed"


def test_automatic_retry_is_bounded_across_restarts(setup, monkeypatch):
    launched, _ = fake_native(monkeypatch, setup, failed={"q1"})
    run(setup, retry=True)
    assert [job[0] for job in launched] == ["q1", "q2", "q3", "q1"]
    launched.clear()
    run(setup, retry=True)
    assert launched == []
    assert queue.read(setup.state / "lightmem_status.json")["failed_ids"] == ["q1"]


def test_disk_floor_prevents_new_dispatch(setup, monkeypatch):
    launched, _ = fake_native(monkeypatch, setup)
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024))
    assert set(run(setup)) == {queue.REUSE_ID}
    assert launched == []
    status = queue.read(setup.state / "lightmem_status.json")
    assert status["status"] == "disk_blocked" and status["pending"] == 3


def test_prediction_validation_rejects_truncated_history(setup):
    target = setup.original / queue.REUSE_ID
    source_path = target / "attempt_1" / "source.json"
    source = queue.read(source_path)
    source["haystack_sessions"][0].pop()
    queue.save(source_path, source)
    with pytest.raises(ValueError, match="complete canonical history"):
        queue.validate_prediction(target, setup.rows[queue.REUSE_ID], setup.protocol)


def test_duplicate_completed_id_is_rejected(setup):
    target = setup.state / "lightmem_lanes" / "lane_0" / queue.REUSE_ID
    setup.write_result(target, queue.REUSE_ID)
    with pytest.raises(ValueError, match="Duplicate completed ID"):
        run(setup)


def test_full500_aggregation_is_exact_preserving_and_restart_safe(tmp_path, monkeypatch):
    ids = [f"q{i}" for i in range(500)]
    rows = {qid: {"question_id": qid} for qid in ids}
    completed = {qid: tmp_path / "sources" / qid for qid in ids}
    protocol = {"dataset_sha256": queue.DATA_SHA256}
    raw = {qid: json.dumps({"question_id": qid, "hypothesis": " Unchanged  answer "}).encode() for qid in ids}
    monkeypatch.setattr(queue, "validate_prediction", lambda target, item, expected: raw[item["question_id"]])
    destination = tmp_path / "aggregate"
    with pytest.raises(ValueError, match="exact full canonical 500"):
        queue.aggregate(ids, rows, protocol, dict(list(completed.items())[:-1]), destination)
    assert not destination.exists()
    queue.aggregate(ids, rows, protocol, completed, destination)
    assert {p.parent.name for p in destination.glob("*/prediction.json")} == set(ids)
    assert all((destination / qid / "prediction.json").read_bytes() == raw[qid] for qid in ids)
    assert queue.read(destination / "aggregation_receipt.json")["question_ids"] == ids
    queue.aggregate(ids, rows, protocol, completed, destination)
    damaged = destination / ids[0] / "prediction.json"
    damaged.write_bytes(b"divergent preserved data")
    with pytest.raises(ValueError, match="preserved"):
        queue.aggregate(ids, rows, protocol, completed, destination)
    assert damaged.read_bytes() == b"divergent preserved data"


@pytest.mark.parametrize("interrupted", [None, "missing_receipt", "truncated_score"])
def test_resume_after_scoring_failure_reuses_verified_export(tmp_path, monkeypatch, interrupted):
    ids = [f"q{i}" for i in range(500)]
    dataset = tmp_path / "canonical.json"
    dataset.write_text("canonical fixture", encoding="utf-8")
    for name in ("export_official.py", "score_diagnostic_f1.py"):
        (tmp_path / name).write_text("frozen utility", encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    command = ["python", str(tmp_path / "native_lightmem.py"), "--dataset", str(dataset)]
    output = state / "lightmem_full500.hypotheses.jsonl"
    receipt_path = output.with_name(output.name + ".receipt.json")
    score_path = state / "lightmem_diagnostic_f1.json"
    calls = []
    if interrupted == "missing_receipt":
        output.write_text("partial export from interrupted process", encoding="utf-8")
    elif interrupted == "truncated_score":
        score_path.write_text('{"summary":', encoding="utf-8")

    def execute(argv, check):
        assert check is True
        calls.append(Path(argv[1]).name)
        if calls[-1] == "export_official.py":
            staged_output = Path(queue.argument(argv, "--output"))
            staged_receipt = staged_output.with_name(staged_output.name + ".receipt.json")
            staged_output.write_text("unchanged full export", encoding="utf-8")
            queue.save(staged_receipt, {
                "schema": "native-seven-official-hypotheses-v1", "method": "lightmem",
                "scope": "full_canonical_500", "expected_question_ids": ids,
                "hypotheses_sha256": queue.digest(staged_output),
                "exporter_sha256": queue.digest(tmp_path / "export_official.py"),
                "inputs_sha256": {str(dataset): queue.digest(dataset)}})
        elif calls.count("score_diagnostic_f1.py") == 1:
            raise queue.subprocess.CalledProcessError(1, argv)
        else:
            queue.save(Path(queue.argument(argv, "--output")), {
                "schema": "longmemeval-s-diagnostic-token-f1-v1", "method": "lightmem",
                "provenance": {"inputs_sha256": {
                    str(path): queue.digest(path) for path in (dataset, output, receipt_path)},
                    "scorer_sha256": queue.digest(tmp_path / "score_diagnostic_f1.py")},
                "summary": {"overall": {"count": 500}}})

    monkeypatch.setattr(queue.subprocess, "run", execute)
    args = (ids, command, tmp_path / "vendor", tmp_path / "aggregate", state)
    with pytest.raises(queue.subprocess.CalledProcessError):
        queue.finish_outputs(*args)
    original = output.read_bytes()
    queue.finish_outputs(*args)
    queue.finish_outputs(*args)
    assert calls == ["export_official.py", "score_diagnostic_f1.py", "score_diagnostic_f1.py"]
    assert output.read_bytes() == original
    if interrupted == "missing_receipt":
        preserved = list(state.glob(output.name + ".interrupted_*"))
        assert len(preserved) == 1
        assert preserved[0].read_text(encoding="utf-8") == "partial export from interrupted process"
    elif interrupted == "truncated_score":
        preserved = list(state.glob(score_path.name + ".interrupted_*"))
        assert len(preserved) == 1
        assert preserved[0].read_text(encoding="utf-8") == '{"summary":'
    dataset.write_text("changed canonical fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="preserved"):
        queue.finish_outputs(*args)
