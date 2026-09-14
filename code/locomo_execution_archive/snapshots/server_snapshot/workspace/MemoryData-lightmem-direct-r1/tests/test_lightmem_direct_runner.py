"""Behavioral gates for the isolated direct LightMem controller."""

from contextlib import contextmanager
import json
from pathlib import Path

import pytest
import yaml

from scripts import run_lightmem_direct_r1 as runner


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    agent = {
        "agent_name": "Agentic_memory_lightmem", "model": runner.queue.MODEL,
        "lightmem_ingest_mode": "direct", "lightmem_messages_use": "user_only",
        "agent_chunk_size": 4096, "retrieve_num": 10, "provider": "openai_compatible",
        "lightmem_comparison_mode": False, "lightmem_pre_compress": False,
        "lightmem_topic_segment": False, "lightmem_metadata_generate": False,
        "lightmem_text_summary": False, "lightmem_embedding_backend": "openai",
        "lightmem_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "lightmem_embedding_dims": 384,
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([{"sample_id": str(i)} for i in range(10)]))
    dataset = {
        "dataset": "LoCoMo", "sub_dataset": "locomo_qa", "chunk_size": 4096,
        "locomo_repeat_session_header": False, "max_test_samples": 10,
        "locomo_categories": [1, 2, 3, 4], "generation_max_length": 1024,
        "test_files": str(dataset_path),
    }
    for name, config in (("agent", agent), ("dataset", dataset)):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(config))
    plan = {
        "run_id": "direct-r1", "model": runner.queue.MODEL, "context_workers": 4,
        "dataset": str(dataset_path), "server_python": "python", "embedding_path": "minilm",
        "methods": [dict(method=method, agent_config=str(tmp_path / "agent.yaml"),
                         dataset_config=str(tmp_path / "dataset.yaml")) for method in runner.queue.METHODS],
        "skip_methods": {method: "Only LightMem authorized" for method in runner.queue.METHODS if method != "lightmem"},
        "output": str(tmp_path / "full"), "probe_output": str(tmp_path / "probe"),
        "controller_output": str(tmp_path / "controller"), "prior_outputs": [str(tmp_path / "prior")],
        "wait_for_services": ["prior-benchmark"],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    # Shared pin/original-QA validators have their own regression suite; these tests isolate orchestration.
    monkeypatch.setattr(runner.queue, "validate_plan", lambda plan: None)
    monkeypatch.setattr(runner.queue, "expected_questions", lambda *args: {"expected": 1540})
    monkeypatch.setattr(runner.queue, "supervisor_state", lambda name: "EXITED")
    monkeypatch.setattr(runner.finalizer, "running_writers", lambda path: [])
    monkeypatch.setattr(runner, "require_free_ports", lambda: None)
    monkeypatch.setattr(runner.queue, "wait_vllm", lambda plan: None)
    monkeypatch.setattr(runner.queue, "wait_health", lambda *args: None)
    events = []

    @contextmanager
    def service(command, log_path):
        events.append("encoder_start")
        yield object()
        events.append("encoder_stop")

    def probe(probe_plan, method, turns, *, sample_index):
        events.append("probe")
        assert probe_plan["run_id"] != plan["run_id"]
        assert probe_plan["output"] == plan["probe_output"]
        assert method == "lightmem" and turns is None and sample_index == 0
        return {"passed": True, "diagnostic_only": True, "qa_count": 3}

    def full(full_plan, item, expected):
        events.append("full")
        assert item["method"] == "lightmem" and full_plan["output"] == plan["output"]
        return report()

    monkeypatch.setattr(runner.queue, "service", service)
    monkeypatch.setattr(runner, "run_probe", probe)
    monkeypatch.setattr(runner.queue, "run_method", full)
    return plan_path, plan, events


def report():
    return {
        "complete": True, "expected": 1540, "evaluated": 1540, "official_f1": 0.39,
        "by_category": {str(index + 1): dict(count=count, expected=count, f1=score)
                        for index, (count, score) in enumerate(zip([400, 400, 400, 340], [.1, .3, .5, .7]))},
    }


def test_check_only_does_not_create_artifacts_or_infer(prepared):
    path, plan, events = prepared
    assert runner.run(path, check_only=True) == 0
    assert not events
    assert not Path(plan["controller_output"]).exists()


@pytest.mark.parametrize("file,key,value", [
    ("agent", "lightmem_ingest_mode", "pipeline"), ("agent", "lightmem_comparison_mode", True),
    ("agent", "retrieve_num", 60), ("agent", "lightmem_messages_use", "hybrid"),
    ("agent", "lightmem_embedding_dims", 768), ("dataset", "chunk_size", 1),
    ("dataset", "locomo_repeat_session_header", True), ("dataset", "generation_max_length", 4096),
    ("dataset", "locomo_categories", [1, 2, 3, 4, 5]),
])
def test_invalid_preset_never_starts_inference(prepared, file, key, value):
    path, _, events = prepared
    config_path = path.parent / f"{file}.yaml"
    config = yaml.safe_load(config_path.read_text())
    config[key] = value
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=key):
        runner.run(path)
    assert not events


@pytest.mark.parametrize("key", ["output", "probe_output", "controller_output"])
def test_existing_artifacts_are_never_reused(prepared, key):
    path, plan, events = prepared
    Path(plan[key]).mkdir()
    with pytest.raises(FileExistsError, match="reuse"):
        runner.run(path)
    assert not events


def test_paths_cannot_overlap_prior_artifacts(prepared):
    _, plan, _ = prepared
    plan["output"] = str(Path(plan["prior_outputs"][0]) / "new")
    with pytest.raises(ValueError, match="overlap"):
        runner.validate_direct(plan)


def test_smoke_failure_blocks_full_inference_and_records_failure(prepared, monkeypatch):
    path, plan, events = prepared
    monkeypatch.setattr(runner, "run_probe", lambda *args, **kwargs: {"passed": False})
    with pytest.raises(RuntimeError, match="probe did not pass"):
        runner.run(path)
    assert "full" not in events
    assert not Path(plan["output"]).exists()
    assert json.loads((Path(plan["controller_output"]) / "status.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("gate", ["writers", "ports", "inference", "stopped"])
def test_runtime_prerequisites_block_probe_and_full(prepared, monkeypatch, gate):
    path, _, events = prepared
    def failed(*args):
        raise RuntimeError("runtime unavailable")
    if gate == "writers":
        monkeypatch.setattr(runner.finalizer, "running_writers", lambda path: [123])
    elif gate == "ports":
        monkeypatch.setattr(runner, "require_free_ports", failed)
    elif gate == "inference":
        monkeypatch.setattr(runner.queue, "wait_vllm", failed)
    else:
        monkeypatch.setattr(runner.queue, "supervisor_state", lambda name: "STOPPED")
    with pytest.raises(RuntimeError):
        runner.run(path)
    assert not events


def test_success_runs_probe_and_full_separately_and_reports_both_averages(prepared):
    path, plan, events = prepared
    assert runner.run(path) == 0
    assert events == ["encoder_start", "probe", "encoder_stop", "encoder_start", "full", "encoder_stop"]
    result = json.loads((Path(plan["controller_output"]) / "report.json").read_text())
    assert result["micro_official_f1"] == .39
    assert result["macro_official_f1"] == pytest.approx(.4)
    assert result["raw_run_error"] is None
    assert "not exact paper reproduction" in result["variant"]


@pytest.mark.parametrize("complete", [False, True])
def test_exact_metadata_boundary_finalizes_completed_raw_run_without_reinference(prepared, monkeypatch, complete):
    path, plan, events = prepared
    def full(*args):
        events.append("full")
        source = Path(plan["output"]) / "lightmem"
        source.mkdir()
        (source / "parallel_status.json").write_text(json.dumps(
            {"complete": complete, "failed": False, "active_contexts": []}))
        raise ValueError("Result metadata differs from the original QA")
    def finalize(*args):
        events.append("finalize")
        return report()
    monkeypatch.setattr(runner.queue, "run_method", full)
    monkeypatch.setattr(runner.finalizer, "finalize_method", finalize)
    if complete:
        assert runner.run(path) == 0
        assert events.count("finalize") == 1
    else:
        with pytest.raises(RuntimeError, match="did not finish"):
            runner.run(path)
        assert "finalize" not in events
    assert events.count("full") == 1


def test_macro_requires_all_categories_complete():
    result = report()
    result["by_category"]["4"]["count"] -= 1
    with pytest.raises(ValueError, match="complete categories"):
        runner.final_report(result, Path("probe/report.json"), None)


def test_waits_for_normal_exit_before_starting_encoder(prepared, monkeypatch):
    path, _, events = prepared
    states = iter(["RUNNING", "EXITED", "EXITED", "EXITED"])
    monkeypatch.setattr(runner.queue, "supervisor_state", lambda name: next(states))
    def sleep(seconds):
        assert seconds == 30 and not events
    monkeypatch.setattr(runner.time, "sleep", sleep)
    assert runner.run(path) == 0
    assert events.count("full") == 1


def test_changed_source_pin_after_wait_blocks_inference(prepared, monkeypatch):
    path, _, events = prepared
    calls = []
    def validate(plan):
        calls.append(plan)
        if len(calls) > 1:
            raise ValueError("Pinned file changed")
    monkeypatch.setattr(runner.queue, "validate_plan", validate)
    with pytest.raises(ValueError, match="Pinned file changed"):
        runner.run(path)
    assert not events


def test_changed_plan_after_probe_blocks_full_inference(prepared, monkeypatch):
    path, plan, events = prepared
    def probe(*args, **kwargs):
        path.write_text(json.dumps(dict(plan, run_id="changed")))
        return {"passed": True, "diagnostic_only": True, "qa_count": 3}
    monkeypatch.setattr(runner, "run_probe", probe)
    with pytest.raises(ValueError, match="plan changed"):
        runner.run(path)
    assert "full" not in events


def test_restarted_prior_job_blocks_inference(prepared, monkeypatch):
    path, _, events = prepared
    states = iter(["EXITED", "RUNNING"])
    monkeypatch.setattr(runner.queue, "supervisor_state", lambda name: next(states))
    with pytest.raises(RuntimeError, match="restarted"):
        runner.run(path)
    assert not events


@pytest.mark.parametrize("ambient_exists", [False, True])
@pytest.mark.parametrize("probe_fails", [False, True])
def test_probe_pins_full_run_environment_and_restores_ambient(prepared, monkeypatch, ambient_exists, probe_fails):
    path, plan, _ = prepared
    expected = {
        "LIGHTMEM_MODEL": runner.queue.MODEL,
        "LIGHTMEM_BASE_URL": runner.queue.LLM_ORIGIN + "/v1",
        "LIGHTMEM_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "LIGHTMEM_EMBEDDING_DIMENSION": "384",
        "METER_AUXILIARY_JOURNAL": str(Path(plan["probe_output"]) / "lightmem/auxiliary.jsonl"),
    }
    for key in expected:
        if ambient_exists:
            monkeypatch.setenv(key, "conflicting-ambient-" + key)
        else:
            monkeypatch.delenv(key, raising=False)
    previous = {key: runner.os.environ.get(key) for key in expected}
    def probe(*args, **kwargs):
        assert {key: runner.os.environ.get(key) for key in expected} == expected
        if probe_fails:
            raise RuntimeError("Diagnostic probe failed")
        return {"passed": True, "diagnostic_only": True, "qa_count": 3}
    monkeypatch.setattr(runner, "run_probe", probe)
    if probe_fails:
        with pytest.raises(RuntimeError, match="Diagnostic probe failed"):
            runner.run(path)
    else:
        assert runner.run(path) == 0
    assert {key: runner.os.environ.get(key) for key in expected} == previous
