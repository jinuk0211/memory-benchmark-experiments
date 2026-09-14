"""Synthetic closed-attempt evidence for CPU-only verifier tests."""
import gzip
import hashlib
import json


def source_fixture(lengths=(17, 3)):
    return [{"session_id": f"s{index}", "date": "2026/09/11 (Fri) 10:00",
             "turns": [{"role": "user" if turn % 2 == 0 else "assistant",
                        "content": f"  original session {index} turn {turn}\n"} for turn in range(length)]}
            for index, length in enumerate(lengths)]


def completed_fixture(r, attempt, *, source=None, keep_responses=False):
    attempt.mkdir(parents=True, exist_ok=True)
    source = source if source is not None else source_fixture((2,))
    query = {"question_id": "qid", "question": "synthetic question", "question_date": "2026/09/11"}
    runtime = {"method": "mem0", "source_root": str(r.ROOT / "source/MemoryData"),
               "api_base": "http://127.0.0.1:18083/v1", "embedding_api_base": "http://127.0.0.1:18084/v1",
               "model": "Qwen/Qwen3.5-9B", "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
               "embedding_dims": 384, "tokenizer": "/synthetic/pinned/tokenizer"}
    protocol = r.protocol_for(runtime, ["qid"] + [f"q{i}" for i in range(499)],
                              r.source_hashes(r.ROOT / "source/MemoryData"))
    identity = {"protocol_sha256": r.harness.digest(protocol), "source_sha256": r.harness.digest(source),
                "query_sha256": r.harness.digest(query)}
    r.harness.save_json(attempt.parents[2] / "protocol.json", protocol)
    batches = list(r.session_batches(source))
    if not keep_responses:
        with gzip.open(attempt / "memory_responses.jsonl.gz", "wt", encoding="utf-8") as stream:
            for index, _ in enumerate(batches):
                for offset, phase in enumerate(("facts", "update")):
                    raw = {"choices": [{"message": {"content": json.dumps({"facts": []} if phase == "facts" else {"memory": []})},
                                        "finish_reason": "stop"}]}
                    event = {"call": index * 2 + offset + 1, "batch_index": index, "phase": phase,
                             "response": raw, "response_sha256": hashlib.sha256(json.dumps(raw, ensure_ascii=False).encode()).hexdigest()}
                    stream.write(json.dumps(event) + "\n")
    with gzip.open(attempt / "memory_responses.jsonl.gz", "rt", encoding="utf-8") as stream:
        responses = [json.loads(line) for line in stream]
    journal = [{**r.batch_input(identity["source_sha256"], metadata, messages),
                "chat_calls": sum(event["batch_index"] == i for event in responses), "seconds": 0.1}
               for i, (metadata, messages) in enumerate(batches)]
    (attempt / "batch_calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in journal), encoding="utf-8")
    progress = {key: value for key, value in journal[-1].items() if key not in ("chat_calls", "seconds")}
    values = {"source.json": source, "query.json": query,
              "worker.json": {"identity": identity, "runtime": runtime, "source_files_sha256": protocol["source_files_sha256"]},
              "native_config.json": {}, "native_runtime.json": {}, "native_answer_detail.json": {},
              "native_telemetry.json": {"posthog_version": "3.25.0", "send": False, "sync_mode": True,
                                        "disabled": True, "created_clients": 2},
              "build_complete.json": {"identity": identity, "batches": len(batches), "sessions": len(source),
                                      "turns": sum(len(s["turns"]) for s in source),
                                      "pairs": sum((len(s["turns"]) + 1) // 2 for s in source)},
              "construction_progress.json": {**progress, "completed_batches": len(batches), "status": "added", "native_result": []},
              "prediction.json": {"identity": identity, "question_id": "qid", "status": "generated",
                                  "hypothesis": "answer", "implementation": r.VERSION, "native_answer_path": r.ANSWER_PATH}}
    for name, value in values.items():
        r.harness.save_json(attempt / name, value)
    for name in ("memory/history.db", "memory/qdrant/closed.sqlite", "console.log"):
        path = attempt / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic closed artifact")
    if not (attempt / "native_degradation.json").exists():
        r.NativeObservations(attempt).save()
    return attempt, identity
