"""Exact LongMemEval rubrics, strict verdict parsing, and resumable item cache.

Rubrics: https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py
The MIT-licensed upstream prompts are reproduced below. Only complete yes/no
verdicts are accepted; judge infrastructure errors remain pending.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib import error, parse, request

JUDGE_MODEL = "gpt-4o-2024-08-06"
TYPES = (
    "single-session-user", "single-session-assistant", "single-session-preference",
    "multi-session", "temporal-reasoning", "knowledge-update",
)
COMMON = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no."
)
END = (
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
TEMPLATES = {
    "single-session-user": COMMON + " " + END,
    "temporal-reasoning": COMMON + (
        " In addition, do not penalize off-by-one errors for the number of days. "
        "If the question asks for the number of days/weeks/months, etc., and the model "
        "makes off-by-one errors (e.g., predicting 19 days when the answer is 18), "
        "the model's response is still correct. "
    ) + END,
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, "
        "answer no. If the response contains some previous information along with an "
        "updated answer, the response should be considered as correct as long as the "
        "updated answer is the required answer."
    ) + END,
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and "
        "a response from a model. Please answer yes if the response satisfies the desired "
        "response. Otherwise, answer no. The model does not need to reflect all the "
        "points in the rubric. The response is correct as long as it recalls and "
        "utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
        "Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? "
        "Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from "
        "a model. Please answer yes if the model correctly identifies the question as "
        "unanswerable. The model could say that the information is incomplete, or some "
        "other information is given but the asked information is not.\n\nQuestion: {}"
        "\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify "
        "the question as unanswerable? Answer yes or no only."
    ),
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def make_prompt(reference: dict[str, Any], hypothesis: str) -> str:
    kind = reference["question_type"]
    if kind not in TYPES:
        raise ValueError("unsupported_question_type")
    if "_abs" in reference["question_id"]:
        kind = "abstention"
    elif kind in ("single-session-assistant", "multi-session"):
        kind = "single-session-user"
    return TEMPLATES[kind].format(reference["question"], reference["answer"], hypothesis)


def parse_verdict(value: Any) -> bool:
    if not isinstance(value, str) or value.strip().lower() not in {"yes", "no"}:
        raise ValueError("invalid_judge_verdict")
    return value.strip().lower() == "yes"


def make_cache_key(reference: dict[str, Any], hypothesis: str, endpoint: str) -> str:
    return digest({
        "question_id": reference["question_id"], "question": reference["question"],
        "answer": reference["answer"], "question_type": reference["question_type"],
        "hypothesis": hypothesis, "model": JUDGE_MODEL, "temperature": 0,
        "max_tokens": 10, "n": 1, "prompt_hash": digest(make_prompt(reference, hypothesis)),
        "endpoint_hash": digest(endpoint.rstrip("/")), "parser_version": "strict-yes-no-v1",
    })


def read_cache(path: Path, key: str, question_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        verdict = parse_verdict(row["verdict"])
        valid = (
            row["cache_key"] == key and row["question_id"] == question_id
            and row["autoeval_label"]["model"] == JUDGE_MODEL
            and isinstance(row["autoeval_label"]["label"], bool)
            and row["autoeval_label"]["label"] == verdict
            and isinstance(row["hypothesis"], str)
        )
    except (ValueError, KeyError, TypeError):
        raise ValueError("invalid_judge_cache") from None
    if not valid:
        raise ValueError("judge_cache_mismatch")
    return row


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def request_verdict(prompt: str, endpoint: str, api_key: str) -> bool:
    payload = {
        "model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}],
        "n": 1, "temperature": 0, "max_tokens": 10,
    }
    req = request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            result = json.load(response)
    except error.HTTPError as exc:
        raise ValueError(f"judge_http_error_{exc.code}") from None
    except (error.URLError, TimeoutError, OSError, ValueError):
        raise ValueError("judge_transport_or_json_error") from None
    try:
        if result["model"] != JUDGE_MODEL:
            raise ValueError("judge_model_mismatch")
        choice = result["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("judge_response_truncated")
        return parse_verdict(choice["message"]["content"])
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ValueError("invalid_judge_response_schema") from None


def summarize(
    references: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]], errors: list[dict[str, str]],
) -> dict[str, Any]:
    counts = Counter(ref["question_type"] for ref in references.values())
    by_type = {}
    for kind in TYPES:
        subset = [row for row in rows if references[row["question_id"]]["question_type"] == kind]
        correct = sum(row["autoeval_label"]["label"] for row in subset)
        by_type[kind] = {
            "expected": counts[kind], "judged": len(subset), "correct": correct,
            "accuracy_on_judged": correct / len(subset) if subset else None,
        }
    abs_rows = [row for row in rows if "_abs" in row["question_id"]]
    complete = len(rows) == len(references)
    accuracy = sum(row["autoeval_label"]["label"] for row in rows) / len(rows) if rows else None
    macro = (
        sum(value["accuracy_on_judged"] for value in by_type.values()) / len(TYPES)
        if complete and all(value["judged"] for value in by_type.values()) else None
    )
    return {
        "judge_model": JUDGE_MODEL, "complete": complete,
        "dataset_expected": len(references), "predictions_available": len(predictions),
        "judged": len(rows), "pending": len(references) - len(rows),
        "missing_predictions": len(references) - len(predictions),
        "pending_judgments": len(predictions) - len(rows),
        "complete_for_predictions": len(rows) == len(predictions),
        "overall_accuracy": accuracy if complete else None,
        "accuracy_on_judged": accuracy, "task_averaged_accuracy": macro,
        "by_type": by_type,
        "abstention": {
            "expected": sum("_abs" in qid for qid in references), "judged": len(abs_rows),
            "accuracy_on_judged": sum(row["autoeval_label"]["label"] for row in abs_rows) / len(abs_rows)
                if abs_rows else None,
        },
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Maximum new calls; cached results still load.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    endpoint = os.environ.get(args.base_url_env, "")
    api_key = os.environ.get(args.api_key_env, "")
    if not endpoint or (not args.cache_only and not api_key):
        parser.error("Endpoint and API-key environment variables are required (cache-only needs endpoint).")
    parsed_endpoint = parse.urlsplit(endpoint)
    if (parsed_endpoint.scheme not in {"https", "http"} or not parsed_endpoint.netloc
            or parsed_endpoint.username or parsed_endpoint.password
            or parsed_endpoint.query or parsed_endpoint.fragment):
        parser.error("Endpoint must be an HTTP(S) API root without embedded credentials, query or fragment.")
    reference_list = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    references = {ref["question_id"]: ref for ref in reference_list}
    if len(references) != len(reference_list):
        raise ValueError("duplicate_reference_ids")
    predictions = {}
    for line in args.predictions.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row["question_id"]
        if qid in predictions or qid not in references or not isinstance(row["hypothesis"], str):
            raise ValueError("invalid_or_duplicate_prediction")
        predictions[qid] = row
    cache_dir = args.cache_dir or args.output.with_suffix(args.output.suffix + ".cache")
    rows, errors = [], []
    calls = 0
    allow_calls = True
    for qid, prediction in predictions.items():
        ref = references[qid]
        hypothesis = prediction["hypothesis"]
        key = make_cache_key(ref, hypothesis, endpoint)
        cache_path = cache_dir / (key + ".json")
        try:
            row = read_cache(cache_path, key, qid)
            if row is None:
                if args.cache_only or not allow_calls or (args.limit is not None and calls >= args.limit):
                    continue
                calls += 1
                label = request_verdict(make_prompt(ref, hypothesis), endpoint, api_key)
                row = {
                    "question_id": qid, "hypothesis": hypothesis, "cache_key": key,
                    "verdict": "yes" if label else "no",
                    "autoeval_label": {"model": JUDGE_MODEL, "label": label},
                }
                write_atomic(cache_path, json.dumps(row, ensure_ascii=False) + "\n")
            if row["hypothesis"] != hypothesis:
                raise ValueError("judge_cache_prediction_mismatch")
            rows.append(row)
        except (ValueError, OSError) as exc:
            errors.append({
                "question_id": qid,
                "error": str(exc) if isinstance(exc, ValueError) else "judge_cache_io_error",
            })
            allow_calls = False  # Read remaining cache, but stop new API spending.
    summary = summarize(references, predictions, rows, errors)
    write_atomic(args.output, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_atomic(args.output.with_suffix(args.output.suffix + ".summary.json"), json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 2 if errors or summary["pending_judgments"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
