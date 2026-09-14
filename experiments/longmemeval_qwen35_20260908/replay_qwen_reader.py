"""Replay the frozen reader from existing caches, with no inference or writes."""
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def reference_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def replay(
    units: list[dict[str, Any]], target: dict[str, Any], protocol: dict[str, Any],
    cache: Path, count_tokens: Callable[[str], int],
) -> dict[str, Any]:
    """Match evaluate_sample's cached dense/BM25 RRF, packing and generation key."""
    import numpy as np
    from rank_bm25 import BM25Okapi

    if not units:
        raise ValueError("Cannot replay an empty memory")

    def lexical(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def embedding(texts: list[str], query: bool) -> Any:
        key = reference_digest([
            protocol["embedding"], texts, query, protocol["config"]["embed_batch_size"],
            "oom_backoff_v1",
        ])
        values = np.load(cache / "embeddings" / f"{key}.npy", allow_pickle=False)
        if (values.ndim != 2 or values.shape[0] != len(texts)
                or not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all()):
            raise ValueError("Invalid cached embedding matrix")
        return values

    def ranks(scores: Any) -> Any:
        order = np.argsort(-scores, kind="stable")
        positions = np.empty(len(order), dtype=np.int64)
        positions[order] = np.arange(len(order))
        return positions

    indexes = [unit.get("index_text", unit["text"]) for unit in units]
    vectors = embedding(indexes, False)
    query = embedding([target["question"]], True)[0]
    sparse = BM25Okapi([lexical(text) or ["_empty"] for text in indexes])
    fused = 1 / (60 + ranks(vectors @ query)) + 1 / (
        60 + ranks(np.asarray(sparse.get_scores(lexical(target["question"])))))
    order = np.argsort(-fused, kind="stable")[:min(len(units), 120)]
    parts, selected, total = [], [], 0
    for index in order:
        candidate = "\n\n".join(parts + [units[index]["text"]])
        cost = count_tokens(candidate)
        if cost <= 2048:
            parts.append(units[index]["text"])
            selected.append(index)
            total = cost
    context = "\n\n".join(parts)
    user = (f"Conversation memory:\n{context}\n\nQuestion date: {target['question_date']}\n"
            f"Question: {target['question']}\nAnswer:")
    key = reference_digest([
        protocol["model"], protocol["config"]["seed"], protocol["reader_prompt"], user, 96, False,
    ])
    generation = json.loads((cache / "generations" / f"{key}.json").read_bytes())
    if not isinstance(generation["text"], str):
        raise ValueError("Invalid cached generation text")
    storage = sum(count_tokens(unit["text"]) + (count_tokens(index) if index != unit["text"] else 0)
                  for unit, index in zip(units, indexes))
    return dict(
        context=context, source_ids=sorted({sid for i in selected for sid in units[i]["sources"]}),
        read_tokens=total, memory_tokens=storage, memory_units=len(units),
        hypothesis=generation["text"], input_tokens=generation["input_tokens"],
        output_tokens=generation["output_tokens"], finish_reason=generation["finish_reason"],
    )