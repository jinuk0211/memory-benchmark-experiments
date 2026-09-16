"""Run the frozen no-binding algorithm with Gemma through local OpenAI-compatible APIs."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any

import numpy as np
from openai import OpenAI
import requests

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "ours_source" / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "vendor"))
sys.path.insert(0, str(CODE / "vendor" / "source"))

import run_ablation as frozen  # noqa: E402
import refine as core  # noqa: E402


class ApiRuntime:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.np = np
        self.cache = Path(args.out) / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.chat_base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        self.embedding_base = os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
        self.model = os.environ.get("LOCOMO_MODEL", "gemma-3-12b-it")
        self.embedding_model = os.environ.get(
            "LOCOMO_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model_meta = {
            "name": self.model,
            "checkpoint": "ggml-org/gemma-3-12b-it-GGUF/gemma-3-12b-it-f16.gguf",
            "precision": "F16",
            "kv_cache": "F16",
            "sha256": os.environ.get("LOCOMO_MODEL_SHA256", "unknown"),
            "backend": "llama.cpp",
        }
        self.embed_meta = {
            "name": self.embedding_model,
            "backend": "fastembed-onnx-cpu",
            "dimension": 384,
        }
        self.chat = OpenAI(base_url=self.chat_base, api_key="EMPTY", timeout=1200, max_retries=0)
        self.embedding = OpenAI(
            base_url=self.embedding_base, api_key="EMPTY", timeout=1200, max_retries=0
        )
        self.tokenize_url = self.chat_base.removesuffix("/v1") + "/tokenize"
        self.ntok = lru_cache(maxsize=100000)(self._token_count)

    def _token_count(self, text: str) -> int:
        response = requests.post(
            self.tokenize_url,
            json={"content": text, "add_special": False, "with_pieces": False},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"Unexpected tokenizer response: {payload}")
        return len(tokens)

    def generate(self, system: str, users: list[str], max_tokens: int = 512) -> list[str]:
        results: list[str] = []
        for position, user in enumerate(users, 1):
            key = core.digest([self.model_meta, self.args.seed, system, user, max_tokens, False])
            path = self.cache / "generations" / f"{key}.json"
            if path.exists():
                results.append(json.loads(path.read_text(encoding="utf-8"))["text"])
                continue
            error = None
            for attempt in range(3):
                started = time.time()
                try:
                    response = self.chat.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0,
                        max_tokens=max_tokens,
                    )
                    choice = response.choices[0]
                    text = (choice.message.content or "").strip()
                    usage = response.usage.model_dump() if response.usage else {}
                    core.save(
                        path,
                        {
                            "text": text,
                            "finish_reason": choice.finish_reason,
                            "input_tokens": int(usage.get("prompt_tokens", 0)),
                            "output_tokens": int(usage.get("completion_tokens", 0)),
                            "seconds": time.time() - started,
                            "backend": "llama.cpp",
                        },
                    )
                    results.append(text)
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    print(
                        f"OURS_GENERATION_RETRY {position}/{len(users)} "
                        f"attempt={attempt + 1} error={exc}",
                        flush=True,
                    )
                    time.sleep(2 ** attempt)
            if error is not None:
                raise error
            print(f"OURS_GEN {position}/{len(users)} max_tokens={max_tokens}", flush=True)
        return results

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        key = core.digest([self.embed_meta, texts, query, "api-v1"])
        path = self.cache / "embeddings" / f"{key}.npy"
        if path.exists():
            return np.load(path)
        vectors = []
        for start in range(0, len(texts), 128):
            batch = texts[start : start + 128]
            response = self.embedding.embeddings.create(
                model=self.embedding_model,
                input=batch,
                encoding_format="float",
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        array = array / np.maximum(norms, 1e-12)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, array)
        temporary.replace(path)
        core.save(path.with_suffix(".json"), {"records": len(texts), "query": query, "dimension": 384})
        return array


def make_runtime(args, stage: str) -> ApiRuntime:
    options = SimpleNamespace(
        stage=stage,
        out=args.out / f"{stage}_{args.shard}",
        model="gemma-3-12b-it",
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        embed_batch_size=128,
        seed=20260907,
    )
    return ApiRuntime(options)


def build_no_binding_only(
    rt: Any,
    sessions: list,
    initial: list,
    audited: list,
    rows: list,
    plans: dict,
    cid: str,
) -> tuple[dict, dict]:
    """Construct exactly the requested no-binding arm."""
    seed = core.build(rt, sessions, "dialogue_residual", audited)
    parent = frozen.backend(rt, sessions, seed, plans, "no_binding")
    groups, mapped = frozen.make_options(rt, sessions, parent, rows)
    memory, receipt = frozen.select_cues(
        parent, groups, mapped, {}, {}, rt.ntok, "parent_single", frozen.ADD_BUDGET
    )
    if memory[: len(parent)] != parent:
        raise ValueError("Cue construction modified base memory")
    source_ids = {turn["id"] for session in sessions for turn in session["turns"]}
    covered = {source_id for unit in memory for source_id in unit["sources"]}
    if covered != source_ids:
        raise ValueError(f"Source coverage changed in {cid}/no_binding")
    return {"no_binding": memory}, {"no_binding": receipt}


def main() -> None:
    frozen.ARMS = ("no_binding",)
    frozen.all_memories = build_no_binding_only
    frozen.make_runtime = make_runtime
    frozen.main()


if __name__ == "__main__":
    main()