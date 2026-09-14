"""Replay locked LoCoMo audit contexts while changing only the answer reader.

The source run already fixed memory construction, retrieval and packing. This
script reuses each exact source context and question, with the frozen READER
instruction. A plan locks all three conditions before a target model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from functools import lru_cache
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
import refine as core
from transfer_data import paired_summary

METHODS = ("seed", "r40_fused_four_turn", "s_parent_single_2000")
EXPECTED_COUNT = 100


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def locked(path: Path, value: Any) -> None:
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen reader-transfer artifact changed: {path}")
        return
    core.save(path, value)


def user_prompt(row: dict[str, Any]) -> str:
    """Identical to frozen refine.evaluate and evaluate_indexed.evaluate."""
    return f"Conversation memory:\n{row['context']}\n\nQuestion: {row['question']}\nAnswer:"


def load_source_artifacts(source_run: Path) -> tuple[dict, dict, dict, dict]:
    """Validate all 300 source rows against the original 100-question manifest."""
    paths = {
        "protocol": source_run / "protocol.json",
        "memory_lock": source_run / "memory_selection_locked.json",
        **{method: source_run / f"{method}_question_audit_items.json" for method in METHODS},
    }
    protocol, memory_lock = read(paths["protocol"]), read(paths["memory_lock"])
    if memory_lock["protocol_sha256"] != core.digest(protocol):
        raise ValueError("Source memory lock does not match its protocol")
    if memory_lock["construction_used_benchmark_questions_or_answers"] is not False:
        raise ValueError("Source construction must be question blind")
    if protocol["methods"] != list(METHODS) or memory_lock["methods"] != list(METHODS):
        raise ValueError("Source method conditions changed")
    records = protocol["audit_manifest"]["records"]
    expected = {record["id"]: record for record in records}
    if len(records) != EXPECTED_COUNT or len(expected) != EXPECTED_COUNT:
        raise ValueError("Source audit manifest must contain 100 unique IDs")
    sources = {}
    reference_qa = {}
    for method in METHODS:
        rows = read(paths[method])
        by_id = {row["id"]: row for row in rows}
        if len(rows) != EXPECTED_COUNT or len(by_id) != len(rows) or set(by_id) != set(expected):
            raise ValueError(f"Source ID coverage mismatch: {method}")
        for qid, row in by_id.items():
            record = expected[qid]
            for key in ("conv_id", "qa_index", "category", "split"):
                if row[key] != record[key]:
                    raise ValueError(f"Source manifest metadata mismatch: {method}/{qid}/{key}")
            if row["candidate"] != method:
                raise ValueError(f"Source candidate label mismatch: {method}/{qid}")
            if not isinstance(row["context"], str) or not isinstance(row["question"], str):
                raise ValueError(f"Source context/question must be text: {method}/{qid}")
            qa_identity = (row["question"], row["gold"], row["category"], row["conv_id"])
            if method == METHODS[0]:
                reference_qa[qid] = qa_identity
            elif reference_qa[qid] != qa_identity:
                raise ValueError(f"Source QA differs across methods: {method}/{qid}")
        # The source manifest, rather than output availability, fixes all row IDs.
        sources[method] = [by_id[record["id"]] for record in records]
    return sources, protocol, memory_lock, paths


class ReaderRuntime(core.Runtime):
    """Frozen generation implementation, initialized without an unused embedder."""

    def __init__(self, args: argparse.Namespace, environment: dict) -> None:
        import torch
        from vllm import LLM

        torch.set_num_threads(8)
        self.args = args
        self.cache = args.out / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model_meta = environment["models"][args.model]
        self.llm = LLM(
            model=self.model_meta["path"], dtype="bfloat16", max_model_len=8192,
            gpu_memory_utilization=0.78, max_num_seqs=24, max_num_batched_tokens=8192,
            enable_prefix_caching=True, enforce_eager=True, seed=args.seed,
        )
        self.tok = self.llm.get_tokenizer()
        self.ntok = lru_cache(maxsize=100000)(
            lambda text: len(self.tok.encode(text, add_special_tokens=False))
        )


def evaluate_rows(rt: Any, source_rows: list[dict], method: str) -> list[dict]:
    """Generate one prediction per fixed source ID; preserve every context byte."""
    prompts = [user_prompt(row) for row in source_rows]
    predictions = rt.generate(core.READER, prompts, max_tokens=96)
    if len(predictions) != len(source_rows) or any(not isinstance(p, str) for p in predictions):
        raise ValueError("Missing generation outputs; reader transfer is incomplete")
    rows = []
    for source, prompt, prediction in zip(source_rows, prompts, predictions):
        key = core.digest([rt.model_meta, rt.args.seed, core.READER, prompt, 96, False])
        cache_path = rt.cache / "generations" / (key + ".json")
        generation = read(cache_path)
        if generation["text"] != prediction:
            raise ValueError("Generation cache does not match returned prediction")
        rows.append({
            **source,
            "question_id": source["id"],
            "conversation_id": str(source["conv_id"]),
            "method": method,
            "source_reader_prediction": source["prediction"],
            "source_reader_official_f1": source["official_f1"],
            "prediction": prediction,
            "hypothesis": prediction,
            "status": "ok" if prediction.strip() else "generation_empty",
            "official_f1": core.f1(prediction, source["gold"], int(source["category"])),
            "generic_token_f1": core.generic_f1(prediction, source["gold"]),
            "source_context_sha256": text_sha256(source["context"]),
            "replayed_context_sha256": text_sha256(source["context"]),
            "user_prompt_sha256": text_sha256(prompt),
            "source_context_tokens": source["read_tokens"],
            "source_memory_tokens": source["memory_tokens"],
            "target_context_tokens": rt.ntok(source["context"]),
            "generation_cache_key": key,
            "generation_cache_sha256": sha256(cache_path),
            "input_tokens": generation["input_tokens"],
            "output_tokens": generation["output_tokens"],
            "finish_reason": generation["finish_reason"],
        })
    return rows


def build_protocol(
    args: argparse.Namespace, environment: dict, sources: dict,
    source_protocol: dict, source_memory_lock: dict, paths: dict,
) -> dict:
    source_hashes = source_protocol["source_sha256"]
    for name in ("refine.py", "evaluate_indexed.py"):
        if sha256(ROOT / "source" / name) != source_hashes[name]:
            raise ValueError(f"Frozen source evaluator hash mismatch: {name}")
    model = environment["models"][args.model]
    model_path = Path(model["path"])
    if not model.get("revision") or not model_path.is_dir():
        raise ValueError("Target model requires a pinned revision and existing snapshot path")
    source_model = source_protocol["args"]["model"]
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "vllm", "transformers", "nltk", "numpy")
    }
    code_paths = [ROOT / "run_reader_transfer.py", ROOT / "transfer_data.py", ROOT / "source" / "refine.py",
                  ROOT / "source" / "evaluate_indexed.py"]
    return {
        "experiment": "exact-context-cross-reader-transfer",
        "source_run": str(args.source_run.resolve()),
        "source_artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "source_protocol_sha256": core.digest(source_protocol),
        "source_memory_lock": source_memory_lock,
        "source_writer_model": {
            "name": source_model,
            **source_protocol["environment"]["models"][source_model],
        },
        "source_reader_model": source_model,
        "target_reader_model": {"name": args.model, **model},
        "environment": {"path": str(args.environment.resolve()), "sha256": sha256(args.environment)},
        "target_model_config_sha256": {
            name: sha256(model_path / name)
            for name in ("config.json", "generation_config.json", "tokenizer_config.json",
                         "tokenizer.json", "special_tokens_map.json")
            if (model_path / name).is_file()
        },
        "packages": packages,
        "source_code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths},
        "methods": list(METHODS),
        "selected_ids": [row["id"] for row in sources[METHODS[0]]],
        "question_count": EXPECTED_COUNT,
        "reader_system_prompt": core.READER,
        "contexts_sha256": {
            method: {row["id"]: text_sha256(row["context"]) for row in rows}
            for method, rows in sources.items()
        },
        "user_prompts_sha256": {
            method: {row["id"]: text_sha256(user_prompt(row)) for row in rows}
            for method, rows in sources.items()
        },
        "generation": {"seed": args.seed, "temperature": 0, "max_tokens": 96,
                       "max_model_len": 8192, "enable_thinking": False},
        "primary_contrast": "s_parent_single_2000 minus r40_fused_four_turn",
        "secondary_contrast": "s_parent_single_2000 minus seed",
        "metric": "frozen LoCoMo category-specific F1",
        "changed_role": "answer reader only",
        "token_accounting": {
            "source_context_tokens": "Original source-reader tokenizer; contexts are not repacked",
            "source_memory_tokens": "Original source-writer/reader tokenizer",
            "target_context_tokens": "Target reader tokenizer over the unchanged context",
            "input_tokens": "Target reader complete chat-template prompt tokens",
            "legacy_row_fields": "read_tokens and memory_tokens retain original source values",
        },
        "memory_construction_replayed": False,
        "retrieval_or_packing_replayed": False,
        "target_outcomes_used_for_selection": False,
        "source_audit_limitation": source_protocol["audit_manifest"]["limitation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "evaluate"), default="plan")
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.source_run.resolve()):
        raise ValueError("Output must be outside the immutable source run")
    environment = read(args.environment)
    sources, source_protocol, source_memory_lock, paths = load_source_artifacts(args.source_run)
    protocol = build_protocol(args, environment, sources, source_protocol, source_memory_lock, paths)
    locked(args.out / "protocol.json", protocol)
    state = {"phase": "planned", "pid": os.getpid(), "time": time.time(),
             "questions": EXPECTED_COUNT, "conditions": list(METHODS), "model": args.model}
    core.save(args.out / "status.json", state)
    if args.stage == "plan":
        print(json.dumps(state), flush=True)
        return
    rt = ReaderRuntime(args, environment)
    results = {}
    for method in METHODS:
        core.save(args.out / "status.json", {**state, "phase": "evaluating", "method": method})
        rows = evaluate_rows(rt, sources[method], method)
        locked(args.out / f"{method}_question_audit_items.json", rows)
        results[method] = rows
    ids = protocol["selected_ids"]
    source_rows = {
        method: [{**row, "question_id": row["id"], "conversation_id": str(row["conv_id"])}
                 for row in rows]
        for method, rows in sources.items()
    }

    def comparisons(rows: dict) -> dict:
        return {
            baseline: paired_summary(
                rows[baseline], rows[METHODS[2]], manifest=ids,
                dataset="locomo", score_key="official_f1",
            )
            for baseline in METHODS[:2]
        }

    summaries = {}
    for method, rows in results.items():
        values = core.summarize(rows)
        values["source_context_tokens"] = values.pop("read_tokens")
        values["source_memory_tokens"] = values.pop("memory_tokens")
        values["target_context_tokens"] = sum(row["target_context_tokens"] for row in rows) / len(rows)
        summaries[method] = values
    summary = {
        "protocol_sha256": core.digest(protocol),
        "source_writer_model": protocol["source_writer_model"],
        "source_reader_model": protocol["source_reader_model"],
        "target_reader_model": protocol["target_reader_model"],
        "summaries": summaries,
        "token_accounting": protocol["token_accounting"],
        "target_reader_comparisons": comparisons(results),
        "source_reader_comparisons": comparisons(source_rows),
        "context_replay_proof": {
            method: {
                "n": len(rows),
                "unchanged_contexts": sum(
                    row["source_context_sha256"] == row["replayed_context_sha256"]
                    == protocol["contexts_sha256"][method][row["question_id"]]
                    for row in rows
                ),
                "unchanged_user_prompts": sum(
                    row["user_prompt_sha256"] == protocol["user_prompts_sha256"][method][row["question_id"]]
                    for row in rows
                ),
            } for method, rows in results.items()
        },
        "limitation": protocol["source_audit_limitation"],
    }
    locked(args.out / "reader_transfer_results.json", summary)
    core.save(args.out / "status.json", {
        **state, "phase": "complete", "time": time.time(),
        "result": str(args.out / "reader_transfer_results.json"),
    })
    print(json.dumps({"phase": "complete", "questions_per_method": EXPECTED_COUNT}), flush=True)


if __name__ == "__main__":
    main()

