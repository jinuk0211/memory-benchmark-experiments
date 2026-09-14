"""Shared final-reader boundary for the controlled LongMemEval comparison."""
from collections.abc import Callable, Sequence
import hashlib
from typing import Any

MODEL = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
READER_PROMPT_SHA256 = "c6f1c0ea029349c0b10967d340d600ce37f89930b5baf0d3ef3ce5549a0b62c3"
DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"


def validate_reference(protocol: dict[str, Any]) -> None:
    """Reject a reader reference from a different experimental condition."""
    expected = {
        "dataset_sha256": DATA_SHA256, "read_budget": 2048,
        "max_answer_tokens": 96, "target_selection": False,
        "history_truncation": False,
    }
    for key, value in expected.items():
        if type(protocol.get(key)) is not type(value) or protocol[key] != value:
            raise ValueError(f"Reference mismatch: {key}")
    if protocol["config"]["model"] != MODEL:
        raise ValueError("Reference model mismatch")
    if protocol["model"]["revision"] != MODEL_REVISION:
        raise ValueError("Reference model revision mismatch")
    if not isinstance(protocol.get("reader_prompt"), str) or not protocol["reader_prompt"].strip():
        raise ValueError("Missing reference reader prompt")
    if hashlib.sha256(protocol["reader_prompt"].encode("utf-8")).hexdigest() != READER_PROMPT_SHA256:
        raise ValueError("Reference reader prompt changed")


def reader_request(
    contexts: Sequence[str], question: str, question_date: str,
    count_tokens: Callable[[str], int], protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pack native ranked candidates with the existing whole-unit read policy.

    Source construction and native retrieval precede this boundary. The caller
    uses the pinned Qwen tokenizer without special tokens. Gold answers and
    benchmark evidence annotations are deliberately not inputs.
    """
    validate_reference(protocol)
    if isinstance(contexts, str):
        raise TypeError("Pass ranked contexts, not one concatenated string")
    if not isinstance(question, str) or not isinstance(question_date, str):
        raise TypeError("Question and date must be strings")
    selected, parts, total = [], [], 0
    for index, text in enumerate(contexts):
        if not isinstance(text, str):
            raise TypeError("Retrieved contexts must be strings")
        candidate = "\n\n".join(parts + [text])
        cost = count_tokens(candidate)
        if cost <= protocol["read_budget"]:
            selected.append(index)
            parts.append(text)
            total = cost
    context = "\n\n".join(parts)
    user = (f"Conversation memory:\n{context}\n\n"
            f"Question date: {question_date}\nQuestion: {question}\nAnswer:")
    request = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": protocol["reader_prompt"]},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0, "max_tokens": protocol["max_answer_tokens"],
        "seed": protocol["config"]["seed"],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    evidence = {
        "retrieved_candidates": list(contexts), "selected_indices": selected,
        "context": context, "read_tokens": total,
    }
    return request, evidence
