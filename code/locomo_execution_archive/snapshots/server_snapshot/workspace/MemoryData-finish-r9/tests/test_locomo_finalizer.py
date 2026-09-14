"""Exact query-template normalization must never mask changed QA or outputs."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts import finalize_locomo_comparison as finalizer
from scripts import score_locomo_comparison as scoring


@pytest.fixture
def example():
    qa = dict(question="When?", answer=2023, category=2, evidence=["D1:1"])
    expected = {("conv-1", 0): qa}
    template = finalizer.get_template("locomo_qa", "query", "Agentic_memory_langmem")
    record = dict(query=template.format(question=qa["question"]), answer=["2023"], output="2023",
                  query_time_len=0.4, eval_metadata=dict(dataset="locomo_qa", sample_id="conv-1",
                  question_id="conv-1_qa0", category="2", evidence=["D1:1"]))
    return expected, template, record


def test_exact_template_is_normalized_in_a_copy_only(example, tmp_path):
    expected, template, record = example
    original = copy.deepcopy(record)
    normalized = finalizer.normalize_records([record], expected, template)
    assert record == original
    assert normalized[0]["recorded_query"] == original["query"]
    assert normalized[0]["query"] == "When?"
    path = tmp_path / "normalized.json"
    finalizer.write_report(path, {"data": normalized})
    rows = finalizer.queue.canonical_predictions([path], expected)
    assert rows[0]["answer"] == 2023
    assert rows[0]["prediction"] == original["output"]


@pytest.mark.parametrize("query", [
    "When?", "Who?", "prefix When? suffix",
    "Search Archival Memory and answer the question as concisely as you can, using a single phrase if possible.\n\n Who? \n\n Answer:",
])
def test_changed_or_merely_contained_questions_are_rejected(example, query):
    expected, template, record = example
    record["query"] = query
    with pytest.raises(ValueError, match="exact pinned prompt"):
        finalizer.normalize_records([record], expected, template)


@pytest.mark.parametrize("change", ["answer", "evidence", "output", "category"])
def test_original_metadata_gates_still_reject_changes(example, tmp_path, change):
    expected, template, record = example
    if change == "answer":
        record["answer"] = ["2024"]
    elif change == "output":
        record["output"] = ["2023"]
    else:
        record["eval_metadata"][change] = ["wrong"] if change == "evidence" else "1"
    path = tmp_path / "normalized.json"
    finalizer.write_report(path, {"data": finalizer.normalize_records([record], expected, template)})
    with pytest.raises(ValueError, match="metadata differs"):
        finalizer.queue.canonical_predictions([path], expected)


def test_unknown_identity_is_rejected(example):
    expected, template, record = example
    record["eval_metadata"]["question_id"] = "conv-1_qa999"
    with pytest.raises(ValueError, match="Unknown original"):
        finalizer.normalize_records([record], expected, template)


@pytest.mark.parametrize("state, writers", [("RUNNING", []), ("EXITED", [123]), ("STOPPED", [])])
def test_never_finalize_live_or_abnormally_stopped_queue(monkeypatch, state, writers):
    monkeypatch.setattr(finalizer.queue, "supervisor_state", lambda name: state)
    monkeypatch.setattr(finalizer, "running_writers", lambda path: writers)
    with pytest.raises(RuntimeError):
        finalizer.wait_for_queue({"output": "/unused"}, False)


def test_terminal_queue_without_writers_is_ready(monkeypatch):
    monkeypatch.setattr(finalizer.queue, "supervisor_state", lambda name: "EXITED")
    monkeypatch.setattr(finalizer, "running_writers", lambda path: [])
    finalizer.wait_for_queue({"output": "/unused"}, False)


def test_end_to_end_uses_original_gates_and_preserves_raw_artifacts(example, tmp_path, monkeypatch):
    expected, template, record = example
    source = tmp_path / "source" / "langmem"
    source.mkdir(parents=True)
    raw_path = source / "parallel_results.json"
    finalizer.write_report(raw_path, {"data": [record]})
    original_bytes = raw_path.read_bytes()
    finalizer.write_report(source / "parallel_status.json", {"complete": True, "failed": False, "active_contexts": []})
    usage = dict(schema_version=1, request_id="request", response_id="response", run_id="run", method="langmem",
                 sample_id="conv-1", question_id="0", phase="qa", request_kind="chat_completion", success=True,
                 usage_status="reported", usage=dict(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                 finish_reasons=["stop"])
    (source / "usage.jsonl").write_text(json.dumps(usage) + "\n")
    finalizer.write_report(source / "telemetry.json", {"wall_seconds": 60, "gpu_energy_wh": None})
    dataset_path = tmp_path / "dataset.json"
    finalizer.write_report(dataset_path, [{"sample_id": "conv-1", "qa": [expected[("conv-1", 0)]]}])
    agent_path, config_path, scorer_path = [tmp_path / name for name in ("agent.yaml", "dataset.yaml", "scorer.py")]
    agent_path.write_text(yaml.safe_dump({"agent_name": "Agentic_memory_langmem"}))
    config_path.write_text(yaml.safe_dump({"sub_dataset": "locomo_qa"}))
    scorer_path.write_text("# Scoring is stubbed; original metadata and accounting are real.\n")
    monkeypatch.setattr(scoring, "official_scores", lambda rows, path: [1.0 for row in rows])
    monkeypatch.setattr(finalizer, "build_report", lambda *args: scoring.build_report(*args, expected_count=1))
    plan = dict(output=str(source.parent), dataset=str(dataset_path), scorer=str(scorer_path),
                run_id="run", context_workers=4, hourly_rate_usd=0.4)
    item = dict(method="langmem", agent_config=str(agent_path), dataset_config=str(config_path))
    destination = tmp_path / "finalized" / "langmem"
    report = finalizer.finalize_method(plan, item, expected, destination)
    assert report["complete"] is True
    assert report["evaluated"] == 1 and report["official_f1"] == 1
    assert report["usage"]["reported_tokens"]["total_tokens"] == 12
    assert report["runtime"]["gpu_energy_wh"] is None
    assert report["normalization"]["raw_results_sha256"] == finalizer.digest(raw_path)
    assert raw_path.read_bytes() == original_bytes
    with pytest.raises(FileExistsError):
        finalizer.finalize_method(plan, item, expected, destination)
