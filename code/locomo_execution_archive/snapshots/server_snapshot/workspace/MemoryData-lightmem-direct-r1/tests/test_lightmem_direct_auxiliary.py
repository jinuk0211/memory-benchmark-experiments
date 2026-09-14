"""Direct ingestion may omit compressor telemetry only with explicit pinned evidence."""

import hashlib
import json

import pytest
import yaml

from scripts import locomo_server_queue as queue


def make_plan(tmp_path, **overrides):
    config = {
        "agent_name": "Agentic_memory_lightmem",
        "lightmem_ingest_mode": "direct",
        "lightmem_comparison_mode": False,
        "lightmem_pre_compress": False,
        "lightmem_topic_segment": False,
        "lightmem_metadata_generate": False,
        "lightmem_text_summary": False,
    }
    config.update(overrides)
    path = tmp_path / "lightmem.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([
        {"sample_id": "sample", "conversation": {"session_1": [{}, {}]}}
    ]), encoding="utf-8")
    return {
        "run_id": "run",
        "dataset": str(dataset),
        "methods": [{"method": "lightmem", "agent_config": str(path)}],
        "file_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()},
    }


@pytest.mark.parametrize("journal_exists", [False, True])
def test_direct_no_auxiliary_activity_is_not_applicable(tmp_path, journal_exists):
    plan = make_plan(tmp_path)
    if journal_exists:
        (tmp_path / "auxiliary.jsonl").touch()
    result = queue.auxiliary_metrics(plan, "lightmem", tmp_path)
    assert result["applicable"] is False
    assert result["complete"] is True
    assert result["ingest_mode"] == "direct"
    assert result["agent_config_sha256"] == next(iter(plan["file_sha256"].values()))


def test_direct_unexpected_auxiliary_activity_is_rejected(tmp_path):
    plan = make_plan(tmp_path)
    (tmp_path / "auxiliary.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected auxiliary"):
        queue.auxiliary_metrics(plan, "lightmem", tmp_path)


@pytest.mark.parametrize("flag", [
    "lightmem_comparison_mode", "lightmem_pre_compress", "lightmem_topic_segment",
    "lightmem_metadata_generate", "lightmem_text_summary",
])
@pytest.mark.parametrize("value", [True, None, "false", 0])
def test_direct_flags_must_be_explicit_false(tmp_path, flag, value):
    plan = make_plan(tmp_path, **{flag: value})
    with pytest.raises(ValueError, match="explicitly disabled"):
        queue.auxiliary_metrics(plan, "lightmem", tmp_path)


@pytest.mark.parametrize("failure", ["missing_pin", "changed_config", "wrong_agent"])
def test_direct_requires_pinned_lightmem_config(tmp_path, failure):
    plan = make_plan(tmp_path)
    if failure == "missing_pin":
        plan["file_sha256"] = {}
    elif failure == "changed_config":
        path = tmp_path / "lightmem.yaml"
        path.write_text(path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    else:
        plan = make_plan(tmp_path, agent_name="Agentic_memory_mem0")
    with pytest.raises(ValueError, match="pinned agent config|explicitly disabled"):
        queue.auxiliary_metrics(plan, "lightmem", tmp_path)


@pytest.mark.parametrize("mode", ["pipeline", None])
@pytest.mark.parametrize("journal_exists", [False, True])
def test_pipeline_or_unspecified_mode_still_requires_auxiliary_coverage(tmp_path, mode, journal_exists):
    plan = make_plan(tmp_path, lightmem_ingest_mode=mode)
    if journal_exists:
        (tmp_path / "auxiliary.jsonl").touch()
    with pytest.raises((ValueError, FileNotFoundError)):
        queue.auxiliary_metrics(plan, "lightmem", tmp_path)


def test_ambiguous_lightmem_configs_cannot_attest_direct_mode(tmp_path):
    plan = make_plan(tmp_path)
    plan["methods"].append(dict(plan["methods"][0]))
    with pytest.raises(ValueError, match="one explicit agent config"):
        queue.auxiliary_metrics(plan, "lightmem", tmp_path)
