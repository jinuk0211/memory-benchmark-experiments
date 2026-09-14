"""Run the pinned official LightMem LongMemEval pipeline without its bundled judge."""

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
UPSTREAM_SHA256 = "32912049141844f3d174a731093d87fbf8b086984e53cdbca245a6099e807f07"


def load_native(source_root: Path, api_base: str, model_name: str,
                embedding_path: str, compressor_path: str, cache_root: Path) -> dict:
    """Load the two unmodified upstream factories, excluding its 10-item driver/judge."""
    upstream = source_root / "experiments/longmemeval/run_lightmem_qwen.py"
    raw = upstream.read_bytes()
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Official LightMem source hash changed")
    sys.path.insert(0, str(source_root / "src"))
    from openai import OpenAI
    from lightmem.memory.lightmem import LightMemory
    scope = {"OpenAI": OpenAI, "LightMemory": LightMemory,
             "LLMLINGUA_MODEL_PATH": compressor_path,
             "EMBEDDING_MODEL_PATH": embedding_path,
             "LLM_MODEL": model_name, "API_KEY": "local",
             "API_BASE_URL": api_base, "QDRANT_DATA_DIR": str(cache_root)}
    text = raw.decode("utf-8-sig")
    factories = text[text.index("class LLMModel:"):text.index("\nllm_judge =")]
    tree = ast.parse(factories, filename=str(upstream))
    names = {"LLMModel", "load_lightmem"}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("Missing native LightMem factories")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(upstream), "exec"), scope)
    return scope


def build_memory(lightmem, source: dict) -> dict:
    """Retain the official user/assistant pairing, flush, compression, and extraction."""
    sessions = [[{"role": turn["role"], "content": turn["content"]} for turn in session]
                for session in source["haystack_sessions"]]
    dates = source["haystack_dates"]
    if len(sessions) != len(dates):
        raise ValueError("Session/date lengths differ")
    initial = {"add_input_prompt": [], "add_output_prompt": [], "api_call_nums": 0}
    results = []
    supplied = sum(len(session) for session in sessions)
    used = 0
    started = time.time()
    for session, timestamp in zip(sessions, dates):
        while session and session[0]["role"] != "user":
            session.pop(0)
        num_turns = len(session) // 2
        for turn_idx in range(num_turns):
            messages = session[turn_idx * 2:turn_idx * 2 + 2]
            if (len(messages) < 2 or messages[0]["role"] != "user"
                    or messages[1]["role"] != "assistant"):
                continue
            for message in messages:
                message["time_stamp"] = timestamp
            last = session is sessions[-1] and turn_idx == num_turns - 1
            result = lightmem.add_memory(messages=messages, force_segment=last, force_extract=last)
            used += len(messages)
            if result != initial:
                results.append(result)
    return {"construction_seconds": time.time() - started,
            "source_turns_supplied": supplied, "turns_passed_to_native": used,
            "native_pairing_skipped_turns": supplied - used, "native_add_results": results}


def answer_question(lightmem, reader, query: dict) -> tuple[str, list[str]]:
    memories = lightmem.retrieve(query["question"], limit=20)
    joined_memories = "\n".join(memories)
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content":
                 f"Question time:{query['question_date']} and question:{query['question']}\n"
                 f"Please answer the question based on the following memories: {joined_memories}"}]
    return reader.call(messages), memories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--embedding-model", required=True, help="Pinned local MiniLM path")
    parser.add_argument("--compressor-model", required=True)
    parser.add_argument("--ids-file", type=Path)
    args = parser.parse_args()
    if args.model != "Qwen/Qwen3.5-9B":
        raise ValueError("This run is frozen to Qwen3.5-9B")
    raw = args.dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("Unexpected LongMemEval-S dataset")
    data = json.loads(raw)
    if len(data) != 500 or len({row["question_id"] for row in data}) != 500:
        raise ValueError("Expected exactly 500 unique question IDs")
    all_ids = {row["question_id"] for row in data}
    selected = json.loads(args.ids_file.read_text(encoding="utf-8")) if args.ids_file else list(all_ids)
    if len(selected) != len(set(selected)) or not set(selected).issubset(all_ids):
        raise ValueError("Invalid selected IDs")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    expected = {"dataset_sha256": DATA_SHA256, "upstream_sha256": UPSTREAM_SHA256,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model": args.model, "embedding_model": args.embedding_model,
                "compressor_model": args.compressor_model, "retrieve_limit": 20,
                "native_reader_max_tokens": 2000, "native_reader_temperature": 0.0,
                "native_reader_top_p": 0.8, "messages_use": "user_only"}
    protocol_path = args.run_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Existing run protocol differs; use a separate run directory")
    else:
        protocol_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    failed = []
    for item in data:
        qid = item["question_id"]
        if qid not in selected:
            continue
        if Path(qid).name != qid or qid in (".", ".."):
            raise ValueError("Unsafe question ID")
        target = args.run_dir / qid
        if (target / "prediction.json").exists():
            previous = json.loads((target / "prediction.json").read_text(encoding="utf-8"))
            if (previous.get("dataset_sha256") != DATA_SHA256 or previous.get("model") != args.model
                    or previous.get("question_id") != qid
                    or previous.get("embedding_model") != args.embedding_model
                    or previous.get("upstream_sha256") != UPSTREAM_SHA256
                    or not isinstance(previous.get("hypothesis"), str)
                    or not previous["hypothesis"].strip()):
                raise ValueError("Resume configuration mismatch")
            continue
        target.mkdir(exist_ok=True)
        attempt = target / f"attempt_{time.time_ns()}"
        attempt.mkdir()
        source = {key: item[key] for key in ("haystack_sessions", "haystack_dates", "haystack_session_ids")}
        source["haystack_sessions"] = [[{"role": turn["role"], "content": turn["content"]}
                                        for turn in session] for session in item["haystack_sessions"]]
        query = {key: item[key] for key in ("question", "question_date")}
        (attempt / "source.json").write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
        try:
            native = load_native(args.source_root, args.api_base, args.model,
                                 args.embedding_model, args.compressor_model, attempt / "qdrant")
            memory = native["load_lightmem"](qid)
            reader = native["LLMModel"](args.model, "local", args.api_base)
            build = build_memory(memory, source)
            (attempt / "construction.json").write_text(json.dumps(build, ensure_ascii=False), encoding="utf-8")
            answer, memories = answer_question(memory, reader, query)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Native reader returned no valid answer")
            result = {"question_id": qid, "hypothesis": answer, "model": args.model,
                      "embedding_model": args.embedding_model, "dataset_sha256": DATA_SHA256,
                      "upstream_sha256": UPSTREAM_SHA256, "attempt": attempt.name,
                      "retrieved_memories": memories, "official_judge_pending": True}
            temporary = target / "prediction.tmp"
            temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, target / "prediction.json")
            del memory, reader
        except Exception as error:
            (attempt / "failure.json").write_text(json.dumps({"question_id": qid,
                "type": type(error).__name__, "error": str(error)}), encoding="utf-8")
            failed.append(qid)
            break
    completed = sum((args.run_dir / qid / "prediction.json").exists() for qid in selected)
    status = {"method": "lightmem", "selected": len(selected), "completed": completed,
              "failed": failed, "population": 500, "official_judge_pending": True,
              "generation_complete": completed == 500, "time": time.time()}
    (args.run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()