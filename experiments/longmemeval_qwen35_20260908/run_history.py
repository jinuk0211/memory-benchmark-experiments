"""Run one isolated native baseline history with the controlled final reader.

A verified post-Qwen service receipt is mandatory. No services are launched here.
Each output directory is a single build attempt; incomplete builds are preserved.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from controlled_reader import DATA_SHA256, MODEL, MODEL_REVISION, reader_request, validate_reference
from materialize_inputs import write_once
from native_six import (
    EMBEDDING_MODEL, EMBEDDING_REVISION, build_config, ingest_chunk,
    process_environment, retrieve_candidates, source_chunks,
)
from prepare_inputs import METHODS, REFERENCE_SHA256, digest


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def validate_service_receipt(receipt: dict[str, Any]) -> None:
    """Require recorded evidence from the separate service/completion gate."""
    expected = {
        "status": "baseline_services_verified_after_qwen_complete",
        "model": MODEL, "model_revision": MODEL_REVISION, "dtype": "bfloat16",
        "embedding_model": EMBEDDING_MODEL, "embedding_revision": EMBEDDING_REVISION,
        "embedding_dimensions": 1024, "qwen_completed_questions_per_arm": 500,
        "qwen_completed_arms": 3, "gemma_held": True,
    }
    for key, value in expected.items():
        if type(receipt.get(key)) is not type(value) or receipt[key] != value:
            raise ValueError(f"Unverified baseline service condition: {key}")



def verify_files(root: Path, hashes: dict[str, str]) -> None:
    """Check the source/tokenizer snapshot frozen by the launch receipt."""
    if not hashes:
        raise ValueError("Missing frozen source/tokenizer hashes")
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Frozen file is outside its declared root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen file changed: {relative}")
def run(args: argparse.Namespace) -> None:
    if args.method not in METHODS:
        raise ValueError("Method excluded from the user-approved LongMemEval comparison")
    receipt = read_json(args.runtime_receipt)
    validate_service_receipt(receipt)
    if (receipt["llm_url"], receipt["embedding_url"]) != (args.llm_url, args.embedding_url):
        raise ValueError("Endpoints differ from the verified service receipt")
    for root, field in ((args.source_root, "source_files_sha256"),
                        (args.higmem_root, "higmem_files_sha256"),
                        (args.tokenizer, "tokenizer_files_sha256"),
                        (Path(__file__).parent, "harness_files_sha256")):
        verify_files(root, receipt[field])
    protocol_bytes = args.reference.read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Current Qwen protocol changed")
    protocol = json.loads(protocol_bytes)
    validate_reference(protocol)
    manifest = read_json(args.inputs / "manifest.json")
    if digest(manifest) != receipt["input_manifest_sha256"]:
        raise ValueError("Frozen input manifest changed")
    if (manifest["reference_protocol_sha256"] != REFERENCE_SHA256
            or manifest["dataset_sha256"] != DATA_SHA256):
        raise ValueError("Input/reference mismatch")
    entries = {entry["question_id"]: entry for entry in manifest["questions"]}
    if len(entries) != 500 or set(entries) != set(protocol["selection"]["selected_ids"]):
        raise ValueError("Input population is not the frozen500")
    entry = entries[args.question_id]
    source = read_json(args.inputs / "sources" / f'{entry["source_sha256"]}.json')
    if digest(source) != entry["source_sha256"]:
        raise ValueError("Source input changed")
    # Resolve before changing cwd: native adapters resolve their own imports there.
    args.output.mkdir(parents=True, exist_ok=True)
    state = args.output / "memory"
    state.mkdir(exist_ok=True)
    os.environ.update(process_environment(state, args.llm_url, args.embedding_url))
    os.environ.update(
        METER_RUN_ID=receipt["run_id"], METER_METHOD=args.method,
        METER_LLM_PROXY_ORIGIN=args.llm_url,
        METER_EMBEDDING_PROXY_ORIGIN=args.embedding_url,
        METER_TIMING_JOURNAL=str(args.output / "timing.jsonl"),
    )
    os.chdir(args.source_root)
    sys.path.insert(0, str(args.source_root))
    from utils.request_metering import meter_operation
    from utils.provider_utils import HuggingFaceTokenizerAdapter
    from transformers import AutoTokenizer
    from openai import OpenAI

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.tokenizer.name != MODEL_REVISION:
        raise ValueError("Use the pinned tokenizer snapshot directory")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))
    identity = {
        "method": args.method, "question_id": args.question_id,
        "source_sha256": entry["source_sha256"], "query_sha256": entry["query_sha256"],
        "reference_sha256": REFERENCE_SHA256, "runtime_receipt_sha256": digest(receipt),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_once(args.output / "identity.json", identity)
    if (args.output / "prediction.json").exists():
        prediction = read_json(args.output / "prediction.json")
        if any(prediction.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing prediction identity differs")
        return
    reader_path = args.output / "reader_input.json"
    cached_reader = reader_path.exists()
    if cached_reader:
        lock = read_json(args.output / "reader_input_lock.json")
        if lock != dict(identity, reader_input_sha256=digest(read_json(reader_path))):
            raise ValueError("Cached reader input changed")
    elif any((args.output / name).exists() for name in ("build_started.json", "build_complete.json")):
        raise RuntimeError("Prior build preserved; use a fresh attempt directory")
    with (args.output / "running.lock").open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    started = time.monotonic()
    agent = system = llm = None
    cleanup = ExitStack()
    cleanup.callback((args.output / "running.lock").unlink)
    try:
        client = OpenAI(base_url=args.llm_url, api_key="EMPTY", max_retries=0, timeout=600)
        cleanup.callback(client.close)
        embedding_client = OpenAI(base_url=args.embedding_url, api_key="EMPTY", max_retries=0, timeout=600)
        cleanup.callback(embedding_client.close)
        # Endpoint identity is checked again for every history, without generation.
        if MODEL not in {model.id for model in client.models.list().data}:
            raise ValueError("Chat endpoint model mismatch")
        if EMBEDDING_MODEL not in {model.id for model in embedding_client.models.list().data}:
            raise ValueError("Embedding endpoint model mismatch")
        if not cached_reader:
            write_once(args.output / "build_started.json", identity)
            with meter_operation("memory_init", sample_id=args.question_id):
                if args.method == "higmem":
                    sys.path.insert(0, str(args.higmem_root))
                    import native_higmem
                    from run_vast import Journal, MeteredLLM, save_memory
                    llm = MeteredLLM(
                        argparse.Namespace(model=MODEL, api_base=args.llm_url),
                        Journal(args.output / "higmem_usage.jsonl"), args.question_id, "construction")
                    cleanup.callback(llm.client.close)
                    system = native_higmem.create_system(
                        llm, embedding_client, args.question_id, args.output / "logs")
                    cleanup.callback(system.executor.shutdown, wait=True)
                else:
                    from utils.agent import AgentWrapper
                    config, dataset_config = build_config(
                        args.method, state, args.tokenizer, args.llm_url, args.embedding_url)
                    write_once(args.output / "native_config.json", config)

                    class ControlledAgent(AgentWrapper):
                        def _initialize_tokenizer(self):
                            return HuggingFaceTokenizerAdapter(tokenizer)

                    agent = ControlledAgent(config, dataset_config, load_agent_from=str(state))
                    cleanup.callback(agent.close)
            with meter_operation("memory_write", sample_id=args.question_id):
                if system is not None:
                    turns = native_higmem.source_turns(source)
                    for index, turn in enumerate(turns, start=1):
                        system.add_turn(*turn)
                        if index % 20 == 0:
                            save_memory(system, state / "higmem.pkl", index, False)
                    unit_count = len(turns)
                else:
                    unit_count = 0
                    for date, text in source_chunks(source):
                        ingest_chunk(agent, args.method, date, text, args.question_id)
                        unit_count += 1
            with meter_operation("memory_finalize", sample_id=args.question_id):
                if system is not None:
                    system.finalize_memory_build()
                    system.build_indices()
                    checkpoint = state / "higmem.pkl"
                    save_memory(system, checkpoint, unit_count, True)
                    extra = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
                else:
                    agent.save_agent()
                    extra = {}
                write_once(args.output / "build_complete.json", dict(
                    identity, input_units=unit_count,
                    source_turns=sum(len(session["turns"]) for session in source), **extra))
        query = read_json(args.inputs / "queries" / f'{entry["query_sha256"]}.json')
        if digest(query) != entry["query_sha256"] or query["question_id"] != args.question_id:
            raise ValueError("Read-time query changed")
        if cached_reader:
            saved_input = read_json(reader_path)
            contexts, trace = saved_input["audit"]["retrieved_candidates"], saved_input["trace"]
        else:
            with meter_operation("retrieval", sample_id=args.question_id, question_id=args.question_id):
                if system is not None:
                    llm.phase, llm.question = "retrieval", args.question_id
                    contexts, trace = native_higmem.retrieve_candidates(
                        system, query["question"], query["question_date"])
                else:
                    contexts = retrieve_candidates(
                        agent, args.method, query["question"], query["question_date"], args.question_id)
                    trace = None
        request, audit = reader_request(
            contexts, query["question"], query["question_date"], count_tokens, protocol)
        write_once(reader_path, dict(request=request, audit=audit, trace=trace))
        write_once(args.output / "reader_input_lock.json",
                   dict(identity, reader_input_sha256=digest(read_json(reader_path))))
        with meter_operation("final_reader", sample_id=args.question_id, question_id=args.question_id):
            response = client.chat.completions.create(**request)
        response_path = args.output / f"reader_response.{time.time_ns()}.json"
        write_once(response_path, response.model_dump())
        choice = response.choices[0]
        if choice.finish_reason not in ("stop", "length") or choice.message.content is None:
            raise ValueError("Invalid final reader response")
        if response.usage is None:
            raise ValueError("Final reader usage is missing")
        write_once(args.output / "prediction.json", dict(
            identity, hypothesis=choice.message.content.strip(), response_file=response_path.name,
            finish_reason=choice.finish_reason, usage=response.usage.model_dump(),
            read_tokens=audit["read_tokens"], seconds=time.monotonic() - started))
        print(json.dumps({"status": "history_complete", "method": args.method,
                          "question_id": args.question_id}), flush=True)
    finally:
        cleanup.close()



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--question-id", required=True)
    for name in ("inputs", "reference", "source-root", "higmem-root",
                 "tokenizer", "output", "runtime-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--llm-url", required=True)
    parser.add_argument("--embedding-url", required=True)
    args = parser.parse_args()
    for name in ("inputs", "reference", "source_root", "higmem_root",
                 "tokenizer", "output", "runtime_receipt"):
        setattr(args, name, getattr(args, name).resolve())
    run(args)


if __name__ == "__main__":
    main()