"""Exercise the HTTP contract without loading model weights or opening a port."""

import base64
import importlib.util
import json
from pathlib import Path
import sys
import types

from fastapi.testclient import TestClient
import numpy as np
import pytest


class FakeMiniLM:
    max_seq_length = 256
    dimension = 384

    def __init__(self):
        self.encode_calls = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def tokenize(self, inputs):
        lengths = [min(len(text.split()) + 2, 256) for text in inputs]
        mask = np.zeros((len(inputs), max(lengths)), dtype=np.int64)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        return {"attention_mask": mask}

    def encode(self, inputs, **kwargs):
        self.encode_calls.append((list(inputs), kwargs))
        vectors = np.zeros((len(inputs), 384), dtype=np.float32)
        for row, text in enumerate(inputs):
            vectors[row, sum(text.encode("utf-8")) % 384] = 1.0
        return vectors


@pytest.fixture
def service(monkeypatch):
    encoder = FakeMiniLM()
    constructor_calls = []

    def load_model(path, **kwargs):
        constructor_calls.append((path, kwargs))
        return encoder

    fake_package = types.ModuleType("sentence_transformers")
    fake_package.SentenceTransformer = load_model
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_package)
    path = Path(__file__).with_name("serve_minilm.py")
    spec = importlib.util.spec_from_file_location("native7_minilm_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, encoder, constructor_calls


@pytest.mark.parametrize("inputs", ["one input", ["first input", "second input"]])
def test_float_and_base64_return_identical_vectors(service, tmp_path, inputs):
    module, encoder, calls = service
    journal = tmp_path / "usage.jsonl"
    with TestClient(module.create_app("/pinned/minilm", journal)) as client:
        outputs = {}
        for encoding in ("float", "base64"):
            response = client.post("/v1/embeddings", json={
                "model": module.MODEL_NAME,
                "input": inputs,
                "encoding_format": encoding,
                "dimensions": 384,
            })
            assert response.status_code == 200
            outputs[encoding] = response.json()

    expected_inputs = [inputs] if isinstance(inputs, str) else inputs
    assert calls == [("/pinned/minilm", {"device": "cpu", "local_files_only": True})]
    assert len(outputs["float"]["data"]) == len(expected_inputs)
    for index, (plain, encoded) in enumerate(zip(
        outputs["float"]["data"], outputs["base64"]["data"], strict=True
    )):
        assert plain["index"] == encoded["index"] == index
        decoded = np.frombuffer(base64.b64decode(encoded["embedding"]), dtype="<f4")
        assert decoded.shape == (384,)
        np.testing.assert_array_equal(decoded, np.asarray(plain["embedding"], dtype=np.float32))
        assert float(np.linalg.norm(decoded)) == pytest.approx(1.0)
    assert encoder.encode_calls == [
        (expected_inputs, {"normalize_embeddings": True, "show_progress_bar": False})
    ] * 2
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize("inputs", [[1, 2, 3], [[1, 2], [3]], 42, ["text", 123]])
def test_token_ids_are_rejected_before_native_inference(service, tmp_path, inputs):
    module, encoder, _ = service
    journal = tmp_path / "usage.jsonl"
    with TestClient(module.create_app("/pinned/minilm", journal)) as client:
        response = client.post("/v1/embeddings", json={
            "model": module.MODEL_NAME, "input": inputs,
        })
    assert response.status_code == 422
    assert encoder.encode_calls == []
    assert not journal.exists()


@pytest.mark.parametrize("field,value", [
    ("model", "another-embedding-model"),
    ("dimensions", 128),
    ("encoding_format", "int8"),
    ("input", []),
])
def test_unsupported_requests_do_not_invoke_encoder(service, tmp_path, field, value):
    module, encoder, _ = service
    journal = tmp_path / "usage.jsonl"
    payload = {"model": module.MODEL_NAME, "input": "valid text", field: value}
    with TestClient(module.create_app("/pinned/minilm", journal)) as client:
        response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 400
    assert encoder.encode_calls == []
    assert not journal.exists()


@pytest.mark.parametrize("attribute,value", [("max_seq_length", 512), ("dimension", 768)])
def test_startup_rejects_non_native_window_or_dimension(service, tmp_path, attribute, value):
    module, encoder, _ = service
    setattr(encoder, attribute, value)
    with pytest.raises(RuntimeError, match="Unexpected MiniLM window or dimensions"):
        module.create_app("/pinned/minilm", tmp_path / "usage.jsonl")
    assert encoder.encode_calls == []


def test_usage_counts_nonpadding_tokens_after_native_window(service, tmp_path):
    module, _, _ = service
    journal = tmp_path / "usage.jsonl"
    inputs = ["two words", " ".join(["word"] * 300)]
    with TestClient(module.create_app("/pinned/minilm", journal)) as client:
        response = client.post("/v1/embeddings", json={
            "model": module.MODEL_NAME, "input": inputs,
        })
    assert response.status_code == 200
    assert response.json()["usage"] == {"prompt_tokens": 260, "total_tokens": 260}
    receipt = json.loads(journal.read_text(encoding="utf-8"))
    assert receipt["input_count"] == 2
    assert receipt["prompt_tokens"] == 260
    assert receipt["max_seq_length"] == 256
    assert receipt["device"] == "cpu"
    assert receipt["dtype"] == "float32"
    assert receipt["elapsed_seconds"] >= 0


def test_models_endpoint_reports_only_expected_backbone(service, tmp_path):
    module, encoder, _ = service
    with TestClient(module.create_app("/pinned/minilm", tmp_path / "usage.jsonl")) as client:
        response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": [
        {"id": module.MODEL_NAME, "object": "model"}
    ]}
    assert encoder.encode_calls == []
