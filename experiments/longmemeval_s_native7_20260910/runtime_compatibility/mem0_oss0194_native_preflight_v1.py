"""Public Mem0 OSS runtime probe only; not recovery candidate or benchmark execution."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import traceback
import urllib.request
from urllib.parse import urlsplit

ROOT = Path("/workspace/longmemeval_s_native7_20260910")
ENV = ROOT / ".venv-mem0-oss0194"
QWEN = "Qwen/Qwen3.5-9B"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_BASE = "http://127.0.0.1:18083/v1"
EMBED_BASE = "http://127.0.0.1:18084/v1"
HF = Path("/workspace/.hf_home/hub")
QWEN_PATH = HF / "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MINILM_PATH = HF / "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PROTECTED = {
    "runtime_receipt.json": "ad0ac34ce4a68101d39b466e5c208a338123868fc008979c97a85cf7f180897d",
    "source/source_snapshot.json": "c0448a9a90dfa71bcb62f80a190d2270e943cc36f04f610eb20271ec6951cc70",
}
TEXTS = ["The native runtime is ready.", "Bananas grow on tropical plants.",
         "hello " * 1024 + "suffix alpha", "hello " * 1024 + "suffix beta"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def protected_hashes():
    return {name: sha(ROOT / name) for name in PROTECTED}


def local_base(value, expected):
    parsed = urlsplit(str(value))
    require(str(value).rstrip("/") == expected and parsed.scheme == "http"
            and parsed.hostname == "127.0.0.1" and not parsed.username
            and not parsed.password and not parsed.query and not parsed.fragment,
            "Client must use the fixed localhost endpoint")


def environment(proof):
    require(sys.flags.isolated == 1, "Invoke the probe with Python -I")
    require(Path(sys.prefix).resolve() == ENV.resolve() and sys.prefix != sys.base_prefix,
            "Use the new isolated .venv-mem0-oss0194")
    require(sys.implementation.name == "cpython" and sys.version_info[:3] == (3, 11, 16),
            "Expected CPython 3.11.16")
    config = (ENV / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    require("include-system-site-packages = false" in config, "System site packages must be disabled")
    for key in list(os.environ):
        if key.startswith("OPENROUTER"):
            del os.environ[key]
    os.environ.update(
        OPENAI_API_KEY="EMPTY", MEM0_TELEMETRY="false", MEM0_DIR=str(proof / "mem0_home"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", HF_HOME=str(proof / "hf_home"),
        XDG_CACHE_HOME=str(proof / "cache"), TIKTOKEN_CACHE_DIR=str(proof / "tiktoken"),
        PYTHONDONTWRITEBYTECODE="1", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
    )
    require(importlib.util.find_spec("torch") is None, "Torch must not be importable")
    distributions = {d.metadata["Name"].lower().replace("_", "-"): d.version
                     for d in importlib.metadata.distributions()}
    require("torch" not in distributions and not any(n.startswith("nvidia-") for n in distributions),
            "Unexpected Torch/GPU distribution")
    require(distributions.get("mem0ai") == "0.1.94", "Expected installed mem0ai 0.1.94")
    return {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
            "isolated_venv": True, "torch_absent": True, "mem0ai_version": distributions["mem0ai"],
            "local_only_offline_environment": True}


def installed_mem0():
    import mem0
    package = Path(mem0.__file__).resolve().parent
    require(package.is_relative_to(ENV.resolve()), "mem0 import escaped the new environment")
    require(mem0.__version__ == "0.1.94", "Imported mem0 version differs")
    files = {"mem0/" + str(p.relative_to(package)).replace(os.sep, "/"): sha(p)
             for p in sorted(package.rglob("*.py")) if p.is_file()}
    require(bool(files), "No installed mem0 Python files")
    distribution = importlib.metadata.distribution("mem0ai")
    metadata = [f for f in distribution.files or [] if str(f).endswith(".dist-info/METADATA")]
    require(len(metadata) == 1, "Expected exactly one mem0ai METADATA file")
    metadata_path = Path(distribution.locate_file(metadata[0])).resolve()
    require(metadata_path.is_relative_to(ENV.resolve()), "METADATA escaped the new environment")
    files[str(metadata[0]).replace(os.sep, "/")] = sha(metadata_path)
    return {"version": mem0.__version__, "import_path": str(package),
            "python_file_count": len(files) - 1, "files_sha256": files}


def qdrant_probe(proof):
    from mem0.vector_stores.qdrant import Qdrant
    database = proof / "synthetic_qdrant"
    require(not database.exists(), "Synthetic Qdrant path must be fresh")
    options = dict(collection_name="public_synthetic_probe", embedding_model_dims=384,
                   path=str(database), on_disk=True)
    a, b = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
    va, vb = [1.0] + [0.0] * 383, [0.0, 1.0] + [0.0] * 382
    pa = {"user_id": "public-probe", "data": "Synthetic alpha", "age": 20}
    pb = {"user_id": "public-probe", "data": "Synthetic beta", "age": 40}
    store = None
    try:
        store = Qdrant(**options)
        store.insert(vectors=[va, vb], payloads=[pa, pb], ids=[a, b])
        hits = store.search(query="public synthetic", vectors=va, limit=2)
        require([str(h.id) for h in hits] == [a, b] and hits[0].score > hits[1].score,
                "Native query_points ranking differs")
        filtered = store.search(query="public synthetic", vectors=va, limit=2,
                                filters={"user_id": "public-probe", "age": {"gte": 18, "lte": 25}})
        require([str(h.id) for h in filtered] == [a], "user_id + numeric range filter failed")
        require(not store.search(query="public synthetic", vectors=va, limit=2,
                                 filters={"user_id": "other-public-user"}), "user_id isolation failed")
        require(store.get(a).payload == pa, "Native get payload differs")
        records, offset = store.list(filters={"user_id": "public-probe"}, limit=10)
        require({str(r.id) for r in records} == {a, b} and offset is None, "Native list differs")
        require(store.col_info().points_count == 2, "Native collection count differs")
        updated = {**pa, "data": "Synthetic alpha updated", "age": 21}
        store.update(vector_id=a, vector=va, payload=updated)
        require(store.get(a).payload == updated, "Native payload update failed")
        store.client.close()
        store = None
        store = Qdrant(**options)
        require(store.get(a).payload == updated and store.get(b).payload == pb,
                "Data did not persist across close/reopen")
        store.delete(vector_id=b)
        require(store.get(b) is None and store.get(a).payload == updated, "Synthetic delete affected wrong point")
        require(store.col_info().points_count == 1, "Delete count differs")
        return {"path": str(database), "on_disk": True, "dimensions": 384,
                "ranking_ids": [str(h.id) for h in hits], "ranking_scores": [float(h.score) for h in hits],
                "filters_get_list_info_update_reopen_delete": "passed", "remaining_id": a}
    finally:
        if store is not None:
            store.client.close()


def tokenizer_probe():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN_PATH), local_files_only=True, trust_remote_code=False)
    tokens = tokenizer.encode("Public tokenizer runtime probe.", add_special_tokens=True)
    require(tokens and all(isinstance(t, int) for t in tokens), "Qwen tokenizer encode failed")
    files = {str(p.relative_to(QWEN_PATH)): sha(p) for p in sorted(QWEN_PATH.rglob("*"))
             if p.is_file() and p.suffix in (".json", ".txt", ".jinja", ".model")}
    require("tokenizer.json" in files and "tokenizer_config.json" in files, "Pinned tokenizer files missing")
    require(importlib.util.find_spec("torch") is None, "Tokenizer probe unexpectedly requires Torch")
    return {"snapshot": str(QWEN_PATH), "class": type(tokenizer).__name__, "token_ids": tokens,
            "files_sha256": files, "local_files_only": True, "torch_model_loaded": False}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise RuntimeError("Redirect rejected by localhost-only probe")


def direct_embeddings():
    local_base(EMBED_BASE, EMBED_BASE)
    payload = {"model": MINILM, "input": TEXTS, "dimensions": 384, "encoding_format": "float"}
    request = urllib.request.Request(EMBED_BASE + "/embeddings", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=120) as response:
        return json.load(response)


def finite_vectors(vectors):
    for vector in vectors:
        require(len(vector) == 384 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x) for x in vector), "Expected 384 finite embedding values")
        require(abs(sum(x * x for x in vector) - 1.0) < 1e-4, "Embedding is not normalized")


def embedding_probe(proof):
    from transformers import AutoTokenizer
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.openai import OpenAIEmbedding
    tokenizer = AutoTokenizer.from_pretrained(str(MINILM_PATH), local_files_only=True, trust_remote_code=False)
    raw = [len(tokenizer(t, truncation=False, add_special_tokens=True)["input_ids"]) for t in TEXTS]
    truncated = [tokenizer(t, truncation=True, max_length=256, add_special_tokens=True)["input_ids"] for t in TEXTS]
    require(min(raw[2:]) > 256 and len(truncated[2]) == 256 and truncated[2] == truncated[3],
            "Synthetic suffixes must be beyond the native 256-token window")
    embedder = OpenAIEmbedding(BaseEmbedderConfig(model=MINILM, embedding_dims=384,
                                                api_key="EMPTY", openai_base_url=EMBED_BASE))
    try:
        local_base(embedder.client.base_url, EMBED_BASE)
        require(embedder.config.model == MINILM and embedder.config.embedding_dims == 384,
                "Native embedding configuration differs")
        sdk = []
        for index, text in enumerate(TEXTS):
            vector = embedder.embed(text, memory_action="add")
            write_new(proof / f"embedding_sdk_{index}.json", {"embedding_repr": repr(vector)})
            sdk.append(vector)
        result = direct_embeddings()
        write_new(proof / "embedding_http_response.json", {"response_repr": repr(result)})
        require(result.get("model") == MINILM and len(result.get("data", [])) == 4,
                "HTTP embedding model/count mismatch")
        require([v.get("index") for v in result["data"]] == list(range(4)), "HTTP embedding order differs")
        expected_usage = {"prompt_tokens": sum(len(t) for t in truncated),
                          "total_tokens": sum(len(t) for t in truncated)}
        require(result.get("usage") == expected_usage, "Embedding usage does not match native 256-token truncation")
        http = [v["embedding"] for v in result["data"]]
        finite_vectors(sdk)
        finite_vectors(http)
        differences = [max(abs(a - b) for a, b in zip(left, right)) for left, right in zip(sdk, http)]
        require(max(differences) <= 1e-6, "Native Mem0 SDK and HTTP vectors differ")
        require(max(abs(a - b) for a, b in zip(sdk[2], sdk[3])) <= 1e-6,
                "Suffix beyond native window changed the embedding")
        require(max(abs(a - b) for a, b in zip(sdk[0], sdk[1])) > 1e-6,
                "Distinct short texts unexpectedly share a vector")
        return {"base_url": EMBED_BASE, "model": MINILM, "dimensions": 384,
                "raw_tokens": raw, "truncated_tokens": [len(t) for t in truncated],
                "sdk_http_max_abs_difference": differences, "tolerance": 1e-6,
                "finite_normalized": True, "suffix_beyond_256_ignored": True,
                "sdk_vectors": sdk, "http_float_vectors": http, "http_usage": result.get("usage"),
                "expected_post_truncation_usage": expected_usage,
                "sdk_embedding_calls": 4, "direct_http_embedding_calls": 1}
    finally:
        embedder.client.close()


def llm_probe(proof):
    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.llms.openai import OpenAILLM
    config = BaseLlmConfig(model=QWEN, api_key="EMPTY", openai_base_url=CHAT_BASE)
    defaults = {k: getattr(config, k) for k in ("max_tokens", "temperature", "top_p")}
    require(defaults == {"max_tokens": 2000, "temperature": 0.1, "top_p": 0.1}, "Native LLM defaults changed")
    llm = OpenAILLM(config)
    try:
        local_base(llm.client.base_url, CHAT_BASE)
        require(llm.config.model == QWEN, "Native LLM model differs")
        ready = llm.generate_response(messages=[{"role": "user", "content": "Reply with the single word READY."}])
        write_new(proof / "llm_text_response.json", {"response": ready})
        require(isinstance(ready, str) and ready.strip() == "READY", "READY connection probe failed")
        raw_json = llm.generate_response(messages=[{"role": "user", "content": 'Return only the JSON object {"ready": true}.'}],
                                         response_format={"type": "json_object"})
        write_new(proof / "llm_json_response.json", {"response": raw_json})
        require(isinstance(raw_json, str), "Expected JSON response text")
        parsed_json = json.loads(raw_json)
        require(isinstance(parsed_json, dict) and set(parsed_json) == {"ready"}
                and parsed_json["ready"] is True, "JSON connection probe failed")
        return {"base_url": CHAT_BASE, "model": QWEN, "native_defaults": defaults,
                "text_response": ready, "json_response": raw_json, "sdk_chat_calls": 2,
                "official_judge": False, "memory_add_called": False}
    finally:
        llm.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-dir", type=Path, required=True)
    args = parser.parse_args()
    proof = args.proof_dir
    base = ROOT / "runtime_compatibility_receipts"
    require(ROOT.is_dir() and base.is_dir(), "Fixed runtime root/receipt directory is missing")
    require(proof.is_absolute() and proof.parent.resolve() == base.resolve()
            and not proof.exists() and not proof.is_symlink(), "Use a fresh direct child of runtime_compatibility_receipts")
    require(base.resolve().is_relative_to(ROOT.resolve()), "Receipt directory escapes the fixed root")
    proof.mkdir(exist_ok=False)
    receipt = {"scope": "public_mem0_oss0194_runtime_preflight_not_recovery_candidate_or_benchmark",
               "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "proof_dir": str(proof), "probe_sha256": sha(Path(__file__)), "stages": {},
               "recovery_source_bundle_transferred": False, "benchmark_questions_run": 0,
               "collected_mem0_and_tokenizer_sha_comparison": "pending independent manifest comparison"}
    write_new(proof / "started.json", receipt)
    exit_code = 1
    try:
        receipt["protected_before_sha256"] = protected_hashes()
        require(receipt["protected_before_sha256"] == PROTECTED, "Existing runtime proof hashes differ")
        stages = [("environment", lambda: environment(proof)), ("installed_mem0", installed_mem0),
                  ("qdrant", lambda: qdrant_probe(proof)), ("tokenizer", tokenizer_probe),
                  ("embeddings", lambda: embedding_probe(proof)), ("llm", lambda: llm_probe(proof))]
        for name, operation in stages:
            stage = {"started_at_utc": datetime.now(timezone.utc).isoformat()}
            try:
                stage.update(status="passed", result=operation())
            except Exception:
                stage.update(status="failed", traceback=traceback.format_exc())
                raise
            finally:
                stage["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                receipt["stages"][name] = stage
                write_new(proof / (name + ".json"), stage)
        receipt["status"] = "passed"
        exit_code = 0
    except Exception:
        receipt.update(status="failed", traceback=traceback.format_exc())
    finally:
        try:
            receipt["protected_after_sha256"] = protected_hashes()
            require(receipt["protected_after_sha256"] == PROTECTED == receipt.get("protected_before_sha256"),
                    "Existing runtime/source proof changed during preflight")
            receipt["protected_files_unchanged"] = True
        except Exception:
            receipt.update(status="failed", protected_files_unchanged=False,
                           preservation_traceback=traceback.format_exc())
            exit_code = 1
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_new(proof / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(proof / "receipt.json")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
