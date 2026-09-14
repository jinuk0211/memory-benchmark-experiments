"""CPU-only behavior tests for refined evaluation with native construction dependencies."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import run_transfer as runner
from test_run_transfer import FakeRuntime


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    samples = [{
        "sample_id": str(cid),
        "conversation": {"session_1": [
            {"speaker": "Alice", "text": "My degree is history.", "dia_id": "D1:1"},
        ]},
        "qa": [{"question": "What is my degree?", "answer": "history", "category": 2}],
    } for cid in range(2)]
    data, environment, out = (tmp_path / name for name in ("data.json", "environment.json", "out"))
    runner.core.save(data, samples)
    runner.core.save(environment, {"models": {
        "Qwen/Qwen3.5-9B": {"path": "cpu-only-model"},
        "sentence-transformers/all-MiniLM-L6-v2": {"path": "cpu-only-embedding"},
    }})
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "cpu-test")
    monkeypatch.setattr(runner, "source_plans", lambda: {
        "r24_source_qa_cards": {"recipe": {"operations": [{}]}},
    })
    runtime = FakeRuntime(out / "cache")
    runtime_factory = Mock(return_value=runtime)
    monkeypatch.setattr(runner, "Runtime", runtime_factory)
    build = Mock(side_effect=lambda rt, history, style, *base: [{
        "text": "My degree is history.", "sources": ["D1:1"], "kind": style, "session": 1,
    }])
    backend = Mock(side_effect=lambda rt, history, seed, plans: [
        {**seed[0], "kind": "r40"},
    ])
    augment = Mock(side_effect=lambda rt, history, parent, rows: (
        [{**parent[0], "kind": "refined"}], {"source_rows": len(rows)}, {},
    ))
    monkeypatch.setattr(runner.core, "build", build)
    monkeypatch.setattr(runner, "backend", backend)
    monkeypatch.setattr(runner, "augment", augment)
    monkeypatch.setattr(runner, "generate_source_probes", lambda rt, sessions, operation: (
        {"records": []}, {"records": [
            {"id": cid + ":probe", "conv_id": cid, "split": "probe_fit"} for cid in sessions
        ]},
    ))
    monkeypatch.setattr(runner, "Scorer", lambda *args: SimpleNamespace(ntok=runtime.ntok))
    monkeypatch.setattr(runner, "subset_jobs", lambda *args: ([], "cpu-test-skip"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: SimpleNamespace(
            encode=lambda text, **kwargs: text.split(),
        )),
    ))
    paired = Mock(return_value={"cpu_test": True})
    monkeypatch.setattr(runner, "paired_summary", paired)

    def run(stage, refined_only=True):
        argv = ["run_transfer.py", stage, "--dataset", "locomo", "--data", str(data),
                "--out", str(out), "--environment", str(environment)]
        if refined_only:
            argv.append("--refined-only")
        monkeypatch.setattr(sys, "argv", argv)
        runner.main()

    def construct(refined_only=True):
        for stage in ("plan", "prepare", "score", "construct"):
            run(stage, refined_only)

    return SimpleNamespace(
        run=run, construct=construct, out=out, data=data, samples=samples,
        runtime=runtime, runtime_factory=runtime_factory, build=build,
        backend=backend, augment=augment, paired=paired,
    )


def test_refined_plan_pins_scope_without_claiming_pairwise_comparison(workflow):
    workflow.run("plan")
    protocol = runner.read(workflow.out / "protocol.json")
    assert protocol["config"]["refined_only"] is True
    assert protocol["evaluation_methods"] == ["s_parent_single_2000"]
    assert protocol["methods"] == list(runner.METHODS)
    assert protocol["primary_contrast"] is None
    assert protocol["secondary_contrast"] is None
    assert protocol["config"]["seed"] == 20260907
    assert protocol["read_budget"] == 2048
    assert protocol["extra_storage_budget"] == 2000
    assert protocol["max_answer_tokens"] == 96
    assert protocol["target_selection"] is False


def test_protocol_pins_meter_and_preserves_existing_source_pins(workflow):
    workflow.run("plan")
    hashes = runner.read(workflow.out / "protocol.json")["source_hashes"]
    required = {"run_transfer.py", "transfer_runtime.py", "transfer_data.py",
                "chat_tokenizer_compat.py", "runtime_meter.py"}
    required.update(str(path.relative_to(runner.ROOT)) for path in (runner.ROOT / "source").glob("*.py"))
    assert set(hashes) == required
    for name, expected in hashes.items():
        assert hashlib.sha256((runner.ROOT / name).read_bytes()).hexdigest() == expected


def test_every_stage_retains_scope_and_native_three_memory_lock(workflow):
    workflow.run("plan")
    expected_protocol = (workflow.out / "protocol.json").read_bytes()
    for stage in ("prepare", "score", "construct", "evaluate"):
        workflow.run(stage)
        assert (workflow.out / "protocol.json").read_bytes() == expected_protocol
    assert [call.args[2] for call in workflow.build.call_args_list] == [
        "session10", "audit", "dialogue_residual",
    ] * 2
    assert workflow.backend.call_count == 2
    assert workflow.augment.call_count == 2
    assert all(call.args[2][0]["kind"] == "r40" for call in workflow.augment.call_args_list)
    sessions = runner.read(workflow.out / "source_sessions.json")
    memories = {method: {cid: runner.read(workflow.out / "memories" / method / (cid + ".json"))
                         for cid in sessions} for method in runner.METHODS}
    lock = runner.read(workflow.out / "memory_lock.json")
    assert lock["memories_sha256"] == runner.core.digest(memories)
    assert lock["benchmark_questions_used"] is False
    assert not workflow.paired.called
    assert not (workflow.out / "paired_diagnostics.json").exists()
    assert sorted(path.name for path in workflow.out.glob("*.jsonl")) == ["s_parent_single_2000.jsonl"]
    assert sorted(path.name for path in (workflow.out / "items").iterdir()) == ["s_parent_single_2000"]


def test_refined_only_generates_all_1540_qa_once_without_seed_or_r40_readers(workflow):
    samples = []
    for cid in range(10):
        sample = copy.deepcopy(workflow.samples[0])
        sample["sample_id"] = str(cid)
        sample["qa"] *= 154
        sample["qa"].insert(1, {"question": "Excluded", "answer": "unused", "category": 5})
        samples.append(sample)
    runner.core.save(workflow.data, samples)
    workflow.construct()
    workflow.run("evaluate")
    rows = [json.loads(line) for line in (workflow.out / "s_parent_single_2000.jsonl").read_text().splitlines()]
    expected_ids = {f"{cid}:{index}" for cid in range(10) for index in range(155) if index != 1}
    assert len(rows) == 1540
    assert {row["question_id"] for row in rows} == expected_ids
    assert all(row["method"] == "s_parent_single_2000" for row in rows)
    assert all(row["official_f1"] == 1 for row in rows)
    assert len(workflow.runtime.calls) == 10
    assert sum(len(call[1]) for call in workflow.runtime.calls) == 1540
    assert {call[2] for call in workflow.runtime.calls} == {96}
    workflow.run("evaluate")
    assert len(workflow.runtime.calls) == 10
    assert not workflow.paired.called


def test_default_mode_keeps_all_three_evaluations_and_pairwise_diagnostics(workflow):
    workflow.construct(refined_only=False)
    workflow.run("evaluate", refined_only=False)
    protocol = runner.read(workflow.out / "protocol.json")
    assert protocol["config"]["refined_only"] is False
    assert protocol["evaluation_methods"] == list(runner.METHODS)
    assert protocol["primary_contrast"] == "s_parent_single_2000 minus r40_fused_four_turn"
    assert protocol["secondary_contrast"] == "s_parent_single_2000 minus seed"
    assert sorted(path.stem for path in workflow.out.glob("*.jsonl")) == sorted(runner.METHODS)
    assert len(workflow.runtime.calls) == 6
    assert workflow.paired.call_count == 2
    assert set(runner.read(workflow.out / "paired_diagnostics.json")["comparisons"]) == set(runner.METHODS[:2])


@pytest.mark.parametrize("stage", ["plan", "prepare", "score", "construct", "evaluate"])
def test_scope_cannot_change_between_stages(workflow, stage):
    workflow.run("plan")
    original = (workflow.out / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        workflow.run(stage, refined_only=False)
    assert (workflow.out / "protocol.json").read_bytes() == original
    workflow.runtime_factory.assert_not_called()


@pytest.mark.parametrize("method", ["seed", "r40_fused_four_turn"])
def test_refined_evaluation_still_rejects_changed_construction_dependencies(workflow, method):
    workflow.construct()
    path = workflow.out / "memories" / method / "0.json"
    runner.core.save(path, [{"text": "tampered dependency"}])
    workflow.runtime_factory.reset_mock()
    with pytest.raises(ValueError, match="Memory lock mismatch"):
        workflow.run("evaluate")
    workflow.runtime_factory.assert_not_called()
    assert not list(workflow.out.glob("*.jsonl"))


def test_state_replaces_meter_context_without_changing_saved_fields(tmp_path, monkeypatch):
    context = Mock()
    monkeypatch.setattr(runner, "set_runtime_context", context, raising=False)
    runner.state(tmp_path, "evaluating", method="s_parent_single_2000", conversation="42")
    runner.state(tmp_path, "generation_complete", questions=1540)
    assert context.call_args_list == [
        (("evaluating",), {"method": "s_parent_single_2000", "conversation": "42"}),
        (("generation_complete",), {"questions": 1540}),
    ]
    saved = runner.read(tmp_path / "status.json")
    assert saved["phase"] == "generation_complete"
    assert saved["questions"] == 1540
    assert "method" not in saved
    assert "conversation" not in saved


def test_source_score_meter_context_identifies_probe_and_conversation(workflow, monkeypatch):
    workflow.run("plan")
    workflow.run("prepare")
    context = Mock()
    monkeypatch.setattr(runner, "set_runtime_context", context, raising=False)
    workflow.run("score")
    scoring = [call.kwargs for call in context.call_args_list if call.args == ("scoring_source",)]
    assert scoring == [
        {"completed": 0, "total": 2, "probe_id": "0:probe", "conversation": "0"},
        {"completed": 1, "total": 2, "probe_id": "1:probe", "conversation": "1"},
    ]
