"""The compatibility retry must not overlap the live queue or weaken its gates."""

import errno
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import retry_simplemem_after_queue as retry_module


@pytest.fixture
def plans(tmp_path):
    shared = {key: key for key in ("model", "dtype", "model_revision", "embedding_path",
              "dataset", "scorer", "hf_home", "higmem_run", "server_python")}
    previous = dict(shared, context_workers=4, hourly_rate_usd=0.4,
                    methods=[{"method": "simplemem"}, {"method": "lightmem"}],
                    output=str(tmp_path / "primary"), client_python="original-python", run_id="primary")
    retry = dict(previous, output=str(tmp_path / "retry"), client_python="isolated-python",
                 run_id="retry", skip_methods={"lightmem": "Already attempted in the primary queue"})
    return previous, retry, tmp_path / "finalized"


@pytest.mark.parametrize("change", ["model_revision", "dtype", "embedding_path", "dataset",
                                   "context_workers", "methods", "client_python", "run_id", "skip_methods"])
def test_rejects_changed_comparison_or_retry_scope(plans, change):
    previous, retry, output = plans
    if change in ("client_python", "run_id"):
        retry[change] = previous[change]
    elif change == "skip_methods":
        retry[change] = {}
    else:
        retry[change] = "changed"
    with pytest.raises(ValueError):
        retry_module.validate_retry(previous, retry, output)


@pytest.mark.parametrize("overlap", ["same", "nested", "parent", "final"])
def test_rejects_overlapping_artifact_directories(plans, overlap):
    previous, retry, output = plans
    if overlap == "same":
        retry["output"] = previous["output"]
    elif overlap == "nested":
        retry["output"] = str(Path(previous["output"]) / "nested")
    elif overlap == "parent":
        retry["output"] = str(Path(previous["output"]).parent)
    else:
        output = Path(retry["output"]) / "final"
    with pytest.raises(ValueError, match="overlap"):
        retry_module.validate_retry(previous, retry, output)


def test_existing_output_is_never_reused(plans):
    previous, retry, output = plans
    output.mkdir()
    with pytest.raises(FileExistsError):
        retry_module.validate_retry(previous, retry, output)


@pytest.mark.parametrize("connection_result", [0, errno.ETIMEDOUT, errno.ECONNREFUSED])
def test_occupied_or_unobservable_ports_are_rejected(monkeypatch, connection_result):
    addresses = []
    class Probe:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def settimeout(self, timeout):
            pass
        def connect_ex(self, address):
            addresses.append(address)
            return connection_result
    monkeypatch.setattr(retry_module.socket, "socket", Probe)
    if connection_result == errno.ECONNREFUSED:
        retry_module.require_free_ports()
        assert addresses == [("127.0.0.1", port) for port in (18081, 18082, 18083)]
    else:
        with pytest.raises(RuntimeError, match="port"):
            retry_module.require_free_ports()


def arrange_run(plans, tmp_path, monkeypatch, queue_code=1, outcome="complete", finalizer_code=0):
    previous, retry, output = plans
    paths = [tmp_path / "previous.json", tmp_path / "retry.json"]
    for path, plan in zip(paths, (previous, retry)):
        path.write_text(json.dumps(plan))
    events = []
    monkeypatch.setattr(retry_module.queue, "validate_plan", lambda plan: events.append("validate"))
    monkeypatch.setattr(retry_module.finalizer, "wait_for_queue", lambda plan, wait: events.append("wait"))
    monkeypatch.setattr(retry_module, "require_free_ports", lambda: events.append("ports"))
    def execute(command, **kwargs):
        if command[1].endswith("locomo_server_queue.py"):
            assert "wait" in events and "ports" in events
            events.append("run")
            return SimpleNamespace(returncode=queue_code)
        events.append("finalize")
        output.mkdir()
        (output / "summary.json").write_text(json.dumps({"finalization_complete": True,
            "methods": [{"method": "simplemem", "state": outcome}]}))
        return SimpleNamespace(returncode=finalizer_code)
    monkeypatch.setattr(retry_module.subprocess, "run", execute)
    return paths, output, events


def test_waits_then_runs_and_reaudits_metadata_failure(plans, tmp_path, monkeypatch):
    paths, output, events = arrange_run(plans, tmp_path, monkeypatch)
    assert retry_module.run_retry(*paths, output) == 0
    assert [item for item in events if item != "validate"] == ["wait", "ports", "run", "finalize"]


@pytest.mark.parametrize("outcome", ["benchmark_failed", "audit_failed", "skipped"])
def test_postprocessing_success_is_not_benchmark_success(plans, tmp_path, monkeypatch, outcome):
    paths, output, _ = arrange_run(plans, tmp_path, monkeypatch, outcome=outcome)
    assert retry_module.run_retry(*paths, output) == 1


def test_abnormal_queue_exit_is_not_finalized(plans, tmp_path, monkeypatch):
    paths, output, events = arrange_run(plans, tmp_path, monkeypatch, queue_code=-9)
    with pytest.raises(RuntimeError, match="abnormally"):
        retry_module.run_retry(*paths, output)
    assert "finalize" not in events


def test_finalizer_failure_is_propagated(plans, tmp_path, monkeypatch):
    paths, output, _ = arrange_run(plans, tmp_path, monkeypatch, finalizer_code=3)
    assert retry_module.run_retry(*paths, output) == 3


def test_plan_changes_during_wait_fail_before_execution(plans, tmp_path, monkeypatch):
    paths, output, events = arrange_run(plans, tmp_path, monkeypatch)
    monkeypatch.setattr(retry_module.finalizer, "wait_for_queue", lambda *args: paths[1].write_text("{}"))
    with pytest.raises(ValueError, match="changed while"):
        retry_module.run_retry(*paths, output)
    assert "run" not in events


def test_check_only_neither_waits_nor_starts_services(plans, tmp_path, monkeypatch):
    paths, output, events = arrange_run(plans, tmp_path, monkeypatch)
    assert retry_module.run_retry(*paths, output, check_only=True) == 0
    assert set(events) == {"validate"}
