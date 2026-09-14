"""Lossless LongMemEval adaptation and paired, cluster-aware transfer statistics.

Only split_sample(sample)[0] may be exposed to a model. Labels, original
session IDs and answer annotations remain in the separate evaluation payload.
This module has no third-party dependencies and never modifies the dataset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

SUBSET_SEED = "frozen-transfer-20260908"


def _unique_ids(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Every row requires a nonempty string {key}")
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def adapt_longmemeval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve every session/turn; use position IDs that carry no gold signal."""
    _unique_ids(rows, "question_id")
    samples = []
    for row in rows:
        qid = row["question_id"]
        sessions = row["haystack_sessions"]
        dates = row["haystack_dates"]
        session_ids = row["haystack_session_ids"]
        if not len(sessions) == len(dates) == len(session_ids):
            raise ValueError(f"Unaligned session/date/ID arrays for {qid}")
        if not isinstance(row["question_type"], str):
            raise ValueError(f"question_type must be a string for {qid}")
        conversation = {}
        evidence = []
        session_map = []
        gold_sessions = set(row["answer_session_ids"])
        if not gold_sessions.issubset(set(session_ids)):
            raise ValueError(f"Unknown answer session IDs for {qid}")
        for number, (session, date, original_id) in enumerate(
            zip(sessions, dates, session_ids), start=1
        ):
            key = f"session_{number}"
            conversation[f"{key}_date_time"] = date
            turns = []
            for turn_number, turn in enumerate(session, start=1):
                if not isinstance(turn["content"], str) or not isinstance(turn["role"], str):
                    raise ValueError(f"Nontext role/content for {qid}, {key}")
                turn_id = f"D{number}:{turn_number}"
                turns.append({
                    "dia_id": turn_id, "speaker": turn["role"], "text": turn["content"]
                })
                if turn.get("has_answer") is True:
                    evidence.append(turn_id)
            conversation[key] = turns
            # A list preserves duplicate original session IDs without losing history.
            session_map.append({
                "session": key,
                "original_id": original_id,
                "is_answer_session": original_id in gold_sessions,
            })
        samples.append({
            "sample_id": qid,
            "conversation": conversation,
            "qa": [{
                "question": row["question"],
                "question_date": row["question_date"],
                "answer": row["answer"],
                "category": row["question_type"],
                "question_type": row["question_type"],
                "evidence": evidence,
                "is_abstention": qid.endswith("_abs"),
            }],
            "evaluation_metadata": {
                "question_id": qid,
                "cluster_id": qid.removesuffix("_abs"),
                "session_map": session_map,
                "answer_session_ids": list(row["answer_session_ids"]),
            },
        })
    return samples


def load_longmemeval(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load all samples by default; an explicit limit uses fixed hash selection.

    limit limits questions, never a question's history. For a saved subset
    manifest, use deterministic_subset(load_longmemeval(path), n) directly.
    """
    with Path(path).open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("LongMemEval input must be a JSON array")
    samples = adapt_longmemeval(rows)
    return samples if limit is None else deterministic_subset(samples, limit)[0]


def split_sample(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return independent JSON-serializable model input and evaluation metadata.

    Pass model_input['conversation'] alone to query-independent memory
    construction; question and question_date are reader inputs only.
    """
    qa = sample["qa"]
    if len(qa) != 1:
        raise ValueError("LongMemEval samples must contain exactly one question")
    model_input = {
        "conversation": copy.deepcopy(sample["conversation"]),
        "question": qa[0]["question"],
        "question_date": qa[0]["question_date"],
    }
    metadata = copy.deepcopy(sample["evaluation_metadata"])
    metadata.update({key: copy.deepcopy(qa[0][key]) for key in (
        "answer", "category", "question_type", "evidence", "is_abstention"
    )})
    return model_input, metadata


def deterministic_subset(
    samples: list[dict[str, Any]], n: int, seed: str = SUBSET_SEED
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select IDs by seeded SHA-256 order, independently of labels or file order."""
    by_id = _unique_ids(samples, "sample_id")
    if isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= len(samples):
        raise ValueError("Subset size must be an integer from 1 to dataset size")
    if not isinstance(seed, str):
        raise ValueError("Subset seed must be a string")
    ordered = sorted(by_id, key=lambda qid: (
        hashlib.sha256(f"{seed}\0{qid}".encode()).hexdigest(), qid
    ))
    selected_ids = ordered[:n]
    population_bytes = json.dumps(sorted(by_id), separators=(",", ":")).encode()
    manifest = {
        "selection": "seeded-sha256-id-order-v1",
        "seed": seed,
        "population_count": len(samples),
        "population_ids_sha256": hashlib.sha256(population_bytes).hexdigest(),
        "selected_ids": selected_ids,
        "selected_count": n,
    }
    return [by_id[qid] for qid in selected_ids], manifest


def _score(row: dict[str, Any], score_key: str) -> tuple[float, bool]:
    failed = row.get("status") in {"error", "failed", "failure"}
    failed = failed or ("prediction" in row and not str(row["prediction"] or "").strip())
    if failed:
        return 0.0, True
    value = row.get(score_key)
    if not isinstance(value, (bool, int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Missing/invalid {score_key}; unjudged rows cannot be summarized")
    return float(value), False


def _percentile(values: list[float], fraction: float) -> float:
    index = fraction * (len(values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def paired_summary(
    baseline_rows: list[dict[str, Any]],
    refined_rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any] | list[str],
    dataset: str = "longmemeval",
    id_key: str = "question_id",
    score_key: str = "score",
    group_key: str = "conversation_id",
    n_bootstrap: int = 10000,
    seed: int = 20260908,
) -> dict[str, Any]:
    """Strict paired micro effect with a percentile cluster-bootstrap 95% CI.

    Rows must exactly cover the same explicit manifest. Missing judgments raise;
    explicit generation failures or empty predictions score zero in the fixed
    denominator. LongMemEval clusters pair qid with qid_abs. LoCoMo requires the
    conversation ID on each row. No target labels select a method.
    """
    baseline = _unique_ids(baseline_rows, id_key)
    refined = _unique_ids(refined_rows, id_key)
    ids = manifest["selected_ids"] if isinstance(manifest, dict) else manifest
    if not isinstance(ids, list) or not ids or any(
        not isinstance(qid, str) or not qid for qid in ids
    ):
        raise ValueError("A nonempty explicit selected_ids manifest is required")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in evaluation manifest")
    expected = set(ids)
    for label, rows in (("baseline", baseline), ("refined", refined)):
        if set(rows) != expected:
            raise ValueError(
                f"{label} coverage mismatch: missing={sorted(expected - set(rows))}, "
                f"extra={sorted(set(rows) - expected)}"
            )
    if dataset not in {"longmemeval", "locomo"}:
        raise ValueError("dataset must be longmemeval or locomo")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer >= 2")
    pairs = []
    clusters = defaultdict(list)
    failures = {"baseline": 0, "refined": 0}
    for qid in sorted(ids):
        before, after = baseline[qid], refined[qid]
        question_type = before.get("question_type", before.get("category"))
        other_type = after.get("question_type", after.get("category"))
        if question_type != other_type:
            raise ValueError(f"Mismatched question type for {qid}")
        is_abstention = qid.endswith("_abs") if dataset == "longmemeval" else False
        for row in (before, after):
            if "is_abstention" in row and row["is_abstention"] != is_abstention:
                raise ValueError(f"Inconsistent abstention metadata for {qid}")
        if dataset == "longmemeval":
            group = qid.removesuffix("_abs")
        else:
            group = before.get(group_key)
            if group is None or group != after.get(group_key):
                raise ValueError(f"Missing/mismatched {group_key} for {qid}")
            group = str(group)
        baseline_score, baseline_failed = _score(before, score_key)
        refined_score, refined_failed = _score(after, score_key)
        failures["baseline"] += baseline_failed
        failures["refined"] += refined_failed
        pair = (baseline_score, refined_score, question_type, is_abstention)
        pairs.append(pair)
        clusters[group].append(refined_score - baseline_score)

    def describe(items: list[tuple[float, float, Any, bool]]) -> dict[str, Any]:
        count = len(items)
        before = sum(item[0] for item in items) / count if count else None
        after = sum(item[1] for item in items) / count if count else None
        return {
            "n": count, "baseline": before, "refined": after,
            "effect_pp": (after - before) * 100 if count else None,
        }

    cluster_totals = [(sum(clusters[key]), len(clusters[key])) for key in sorted(clusters)]
    random_state = random.Random(seed)
    boot = []
    for _ in range(n_bootstrap):
        sampled = random_state.choices(cluster_totals, k=len(cluster_totals))
        boot.append(100 * sum(total for total, _ in sampled) / sum(n for _, n in sampled))
    boot.sort()
    types = sorted({pair[2] for pair in pairs if pair[2] is not None}, key=str)
    by_type = {
        str(kind): describe([pair for pair in pairs if pair[2] == kind]) for kind in types
    }
    result = describe(pairs)
    result.update({
        "dataset": dataset,
        "cluster_count": len(clusters),
        "ci95_pp": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "ci_method": "paired-cluster-percentile-bootstrap",
        "bootstrap_resamples": n_bootstrap,
        "bootstrap_seed": seed,
        "wins": sum(pair[1] > pair[0] for pair in pairs),
        "losses": sum(pair[1] < pair[0] for pair in pairs),
        "ties": sum(pair[1] == pair[0] for pair in pairs),
        "generation_failures_counted_zero": failures,
        "by_question_type": by_type,
        "abstention": describe([pair for pair in pairs if pair[3]]),
        "answerable": describe([pair for pair in pairs if not pair[3]]),
        "manifest_ids_sha256": hashlib.sha256(
            json.dumps(sorted(ids), separators=(",", ":")).encode()
        ).hexdigest(),
    })
    if by_type:
        result["macro_by_question_type"] = {
            arm: sum(item[arm] for item in by_type.values()) / len(by_type)
            for arm in ("baseline", "refined", "effect_pp")
        }
    return result

