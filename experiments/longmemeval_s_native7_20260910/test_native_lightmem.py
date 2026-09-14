"""Native LightMem contract tests with all model and network calls replaced."""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
import types

import pytest


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = ROOT / "lightmem_official_20260909/upstream"
UPSTREAM = UPSTREAM_ROOT / "experiments/longmemeval/run_lightmem_qwen.py"
INITIAL = {"add_input_prompt": [], "add_output_prompt": [], "api_call_nums": 0}


@pytest.fixture
def adapter():
    path = Path(__file__).with_name("native_lightmem.py")
    spec = importlib.util.spec_from_file_location("native7_lightmem_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingMemory:
    def __init__(self):
        self.add_calls = []
        self.retrieve_calls = []

    def add_memory(self, **kwargs):
        self.add_calls.append(copy.deepcopy(kwargs))
        return {"call": len(self.add_calls)}

    def retrieve(self, question, limit):
        self.retrieve_calls.append((question, limit))
        return ["first retained fact", "second retained fact"]


def turn(role, content="source"):
    return {"role": role, "content": content}


@pytest.mark.parametrize("sessions", [
    [[turn("assistant"), turn("user"), turn("assistant"), turn("user")],
     [turn("user"), turn("assistant")]],
    [[turn("user"), turn("assistant")], []],
    [[turn("user"), turn("assistant"), turn("user"), turn("user")]],
    [[turn("system"), turn("assistant")]],
])
def test_pairing_and_flush_match_pinned_driver(adapter, sessions):
    source = {"haystack_sessions": sessions, "haystack_dates": [
        f"2023-01-{index + 1:02d}" for index in range(len(sessions))
    ]}
    unchanged = copy.deepcopy(source)
    native = RecordingMemory()
    upstream_text = UPSTREAM.read_text(encoding="utf-8-sig")
    fragment = upstream_text[upstream_text.index("    for session, timestamp in zip(sessions, timestamps):"):
                             upstream_text.index("    time_end = time.time()")]
    scope = {"sessions": copy.deepcopy(sessions), "timestamps": source["haystack_dates"],
             "lightmem": native, "INIT_RESULT": INITIAL, "results_list": []}
    exec(compile(ast.parse(textwrap.dedent(fragment)), str(UPSTREAM), "exec"), scope)
    wrapped = RecordingMemory()
    result = adapter.build_memory(wrapped, source)
    assert wrapped.add_calls == native.add_calls
    assert source == unchanged
    assert result["native_add_results"] == scope["results_list"]
    assert result["source_turns_supplied"] == sum(map(len, sessions))
    passed = sum(len(call["messages"]) for call in wrapped.add_calls)
    assert result["turns_passed_to_native"] == passed
    assert result["native_pairing_skipped_turns"] == sum(map(len, sessions)) - passed


def test_session_date_mismatch_fails_before_native_call(adapter):
    memory = RecordingMemory()
    with pytest.raises(ValueError, match="Session/date lengths differ"):
        adapter.build_memory(memory, {"haystack_sessions": [[turn("user")]], "haystack_dates": []})
    assert not memory.add_calls


def test_native_prompt_and_top20_are_preserved(adapter):
    memory = RecordingMemory()
    prompts = []
    reader = types.SimpleNamespace(call=lambda messages: prompts.append(messages) or "native answer")
    answer, memories = adapter.answer_question(memory, reader, {
        "question": "What was retained?", "question_date": "2023/01/01 (Sun) 12:00",
    })
    assert answer == "native answer"
    assert memory.retrieve_calls == [("What was retained?", 20)]
    assert prompts == [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content":
         "Question time:2023/01/01 (Sun) 12:00 and question:What was retained?\n"
         "Please answer the question based on the following memories: "
         "first retained fact\nsecond retained fact"},
    ]]
    assert memories == ["first retained fact", "second retained fact"]


def test_pinned_factories_use_original_configuration_and_decoder(adapter, monkeypatch, tmp_path):
    configs, clients, requests = [], [], []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            clients.append(kwargs)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            requests.append(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="native response"))])

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    fake_lightmem = types.ModuleType("lightmem.memory.lightmem")
    fake_lightmem.LightMemory = types.SimpleNamespace(from_config=lambda config: configs.append(config) or config)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "lightmem.memory.lightmem", fake_lightmem)
    monkeypatch.setattr(sys, "path", list(sys.path))
    scope = adapter.load_native(UPSTREAM_ROOT, "http://127.0.0.1:18081/v1", "Qwen/Qwen3.5-9B",
                                "/pinned/minilm", "/pinned/compressor", tmp_path)
    assert "llm_judge" not in scope
    scope["load_lightmem"]("question-id")
    config = configs[0]
    assert config["messages_use"] == "user_only"
    assert config["update"] == "offline"
    assert config["pre_compress"] is config["topic_segment"] is True
    assert config["pre_compressor"]["configs"]["llmlingua_config"]["model_name"] == "/pinned/compressor"
    assert config["text_embedder"]["configs"]["model"] == "/pinned/minilm"
    assert config["memory_manager"]["configs"]["max_tokens"] == 16000
    assert config["embedding_retriever"]["configs"]["path"] == f"{tmp_path}/question-id"
    reader = scope["LLMModel"]("Qwen/Qwen3.5-9B", "local", "http://127.0.0.1:18081/v1")
    assert reader.call([{"role": "user", "content": "hello"}]) == "native response"
    assert len(clients) == 1
    assert requests[0]["max_tokens"] == 2000
    assert requests[0]["temperature"] == 0.0
    assert requests[0]["top_p"] == 0.8
    assert requests[0]["stream"] is False


def test_changed_upstream_rejected_before_import(adapter, tmp_path):
    source = tmp_path / "experiments/longmemeval/run_lightmem_qwen.py"
    source.parent.mkdir(parents=True)
    source.write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash changed"):
        adapter.load_native(tmp_path, "unused", "unused", "unused", "unused", tmp_path)


@pytest.fixture
def run_case(adapter, monkeypatch, tmp_path):
    data = [{"question_id": f"q{index}", "question": "BENCHMARK_QUESTION",
             "question_date": "2023-01-02", "question_type": "EVAL_TYPE",
             "answer": "GOLD_ANSWER", "answer_session_ids": ["session0"],
             "haystack_sessions": [[{"role": "user", "content": "source only", "has_answer": True},
                                     {"role": "assistant", "content": "source reply"}]],
             "haystack_dates": ["2023-01-01"], "haystack_session_ids": ["session0"]}
            for index in range(500)]
    dataset, ids, run_dir = tmp_path / "data.json", tmp_path / "ids.json", tmp_path / "run"
    dataset.write_text(json.dumps(data), encoding="utf-8")
    ids.write_text('["q0"]', encoding="utf-8")
    monkeypatch.setattr(adapter, "DATA_SHA256", hashlib.sha256(dataset.read_bytes()).hexdigest())
    memories, load_calls = [], []
    control = {"answer": "valid native answer", "failure": None}

    def load_native(*args):
        load_calls.append(args)

        def load_memory(_qid):
            memory = RecordingMemory()
            memories.append(memory)
            return memory

        def read(_messages):
            if control["failure"]:
                raise control["failure"]
            return control["answer"]

        return {"load_lightmem": load_memory,
                "LLMModel": lambda *args: types.SimpleNamespace(call=read)}

    monkeypatch.setattr(adapter, "load_native", load_native)
    argv = ["native_lightmem.py", "--dataset", str(dataset), "--run-dir", str(run_dir),
            "--source-root", str(UPSTREAM_ROOT), "--api-base", "http://127.0.0.1:18081/v1",
            "--embedding-model", "/pinned/minilm", "--compressor-model", "/pinned/compressor",
            "--ids-file", str(ids)]
    monkeypatch.setattr(sys, "argv", argv)
    return types.SimpleNamespace(module=adapter, data=data, dataset=dataset, ids=ids,
                                 run_dir=run_dir, memories=memories, calls=load_calls,
                                 control=control, argv=argv)


def test_construction_receives_only_source_fields(run_case):
    case = run_case
    case.module.main()
    saved = next((case.run_dir / "q0").glob("attempt_*/source.json"))
    source = json.loads(saved.read_text(encoding="utf-8"))
    assert set(source) == {"haystack_sessions", "haystack_dates", "haystack_session_ids"}
    assert all(set(t) == {"role", "content"} for session in source["haystack_sessions"] for t in session)
    assert all(set(t) == {"role", "content", "time_stamp"}
               for call in case.memories[0].add_calls for t in call["messages"])
    assert "BENCHMARK_QUESTION" not in saved.read_text(encoding="utf-8")
    assert "GOLD_ANSWER" not in saved.read_text(encoding="utf-8")


def test_successful_resume_skips_inference_and_keeps_partial_status(run_case):
    case = run_case
    case.module.main()
    first = (case.run_dir / "q0/prediction.json").read_bytes()
    case.module.main()
    assert len(case.calls) == 1
    assert (case.run_dir / "q0/prediction.json").read_bytes() == first
    status = json.loads((case.run_dir / "status.json").read_text())
    assert status["completed"] == 1
    assert status["population"] == 500
    assert status["generation_complete"] is False


def test_changed_resume_protocol_rejected(run_case):
    case = run_case
    case.module.main()
    case.argv.extend(["--compressor-model", "/different/compressor"])
    with pytest.raises(ValueError, match="protocol differs"):
        case.module.main()
    assert len(case.calls) == 1


def test_changed_prediction_identity_rejected(run_case):
    case = run_case
    case.module.main()
    path = case.run_dir / "q0/prediction.json"
    value = json.loads(path.read_text())
    value["embedding_model"] = "another embedding"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume configuration mismatch"):
        case.module.main()
    assert len(case.calls) == 1


def test_failed_attempt_is_preserved_and_retry_uses_new_memory(run_case):
    case = run_case
    case.control["failure"] = RuntimeError("simulated service failure")
    with pytest.raises(SystemExit) as raised:
        case.module.main()
    assert raised.value.code == 1
    failure = next((case.run_dir / "q0").glob("attempt_*/failure.json"))
    preserved = failure.read_bytes()
    assert not (case.run_dir / "q0/prediction.json").exists()
    case.control["failure"] = None
    case.module.main()
    assert failure.read_bytes() == preserved
    assert len(list((case.run_dir / "q0").glob("attempt_*"))) == 2
    assert len(case.memories) == 2
    assert case.calls[0][-1] != case.calls[1][-1]


@pytest.mark.parametrize("answer", [None, "", "  \n "])
def test_empty_answer_is_not_published_as_completed(run_case, answer):
    case = run_case
    case.control["answer"] = answer
    with pytest.raises(SystemExit):
        case.module.main()
    assert not (case.run_dir / "q0/prediction.json").exists()
    assert list((case.run_dir / "q0").glob("attempt_*/failure.json"))


def test_canonical_dataset_hash_guard(run_case):
    case = run_case
    case.dataset.write_bytes(case.dataset.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Unexpected LongMemEval-S dataset"):
        case.module.main()
    assert not case.calls


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_all_500_unique_questions_required(run_case, monkeypatch, mode):
    case = run_case
    data = case.data[:-1] if mode == "missing" else case.data[:-1] + [case.data[0]]
    case.dataset.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(case.module, "DATA_SHA256", hashlib.sha256(case.dataset.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="500 unique"):
        case.module.main()
    assert not case.calls


@pytest.mark.parametrize("ids", [["q0", "q0"], ["unknown"]])
def test_invalid_selected_ids_rejected(run_case, ids):
    case = run_case
    case.ids.write_text(json.dumps(ids), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid selected IDs"):
        case.module.main()
    assert not case.calls


def test_protocol_binds_current_runner_bytes(run_case):
    case = run_case
    case.module.main()
    protocol = (case.run_dir / "protocol.json").read_text(encoding="utf-8")
    runner_hash = hashlib.sha256(Path(case.module.__file__).read_bytes()).hexdigest()
    assert runner_hash in protocol

