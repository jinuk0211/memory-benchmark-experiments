"""Resumable HiGMem / Qwen experiment; scores use the official LoCoMo evaluator."""
import argparse
import copy
import hashlib
import json
import importlib.metadata
import pickle
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI
from fphm_core import FPHMSystem
from load_dataset import parse_conversation
import prompts


EVENT_AFFILIATION_MAX_TOKENS = 4096


def constrain_schema_for_caller(schema, caller):
    """Bound the one small JSON response that can otherwise grow indefinitely."""
    if caller != "_decide_event_affiliation":
        return schema, None
    constrained = copy.deepcopy(schema)
    root = constrained["schema"]
    properties = root["properties"]
    root["additionalProperties"] = False
    properties["reasoning"]["maxLength"] = 2048
    affiliations = properties["affiliations"]
    affiliations.update(maxItems=11, uniqueItems=True)
    affiliations["items"]["maxLength"] = 128
    return constrained, EVENT_AFFILIATION_MAX_TOKENS


class Journal:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def append(self, row):
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()


class MeteredLLM:
    def __init__(self, args, journal, sample, phase, question=None):
        self.client = OpenAI(base_url=args.api_base, api_key="EMPTY", max_retries=0, timeout=600)
        self.model, self.journal = args.model, journal
        self.sample, self.phase, self.question = sample, phase, question

    def get_completion(self, prompt, response_format, temperature=0.0, max_tokens=None):
        started = time.time()
        common = dict(sample=self.sample, phase=self.phase, question=self.question, started=started)
        try:
            request = dict(
                model=self.model,
                messages=[{"role": "system", "content": "You must respond with a JSON object."},
                          {"role": "user", "content": prompt}],
                response_format=response_format, temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if max_tokens is not None:
                request["max_tokens"] = max_tokens
            result = self.client.chat.completions.create(**request)
        except Exception as exc:
            self.journal.append(dict(common, seconds=time.time()-started, error=str(exc)))
            raise
        usage = result.usage.model_dump() if result.usage else None
        self.journal.append(dict(common, seconds=time.time()-started, usage=usage,
                                 response_id=result.id, finish_reason=result.choices[0].finish_reason))
        if usage is None:
            raise RuntimeError("Server omitted token usage")
        if result.choices[0].finish_reason != "stop":
            raise RuntimeError("Generation truncated")
        text = result.choices[0].message.content
        if not isinstance(json.loads(text), dict):
            raise ValueError("Expected JSON object")
        return text


class StrictFPHM(FPHMSystem):
    """Keep upstream algorithm but stop instead of silently swallowing failed model calls."""
    def _get_llm_json_response(self, prompt, schema, caller, temperature=0.0):
        import jsonschema
        constrained_schema, max_tokens = constrain_schema_for_caller(schema, caller)
        for attempt in range(3):
            try:
                text = self.llm.llm.get_completion(
                    prompt, {"type": "json_schema", "json_schema": constrained_schema},
                    temperature, max_tokens=max_tokens)
                parsed = json.loads(text)
                jsonschema.validate(parsed, constrained_schema["schema"])
                self.logger.log("llm_call", dict(caller_function=caller, prompt=prompt,
                                                raw_response=text, parsed_response=parsed))
                return parsed
            except Exception as exc:
                self.logger.log("call_error", dict(caller=caller, attempt=attempt, error=str(exc)))
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


def read_rows(path):
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    rows, offset = [], 0
    for index, line in enumerate(lines):
        try:
            if line.strip():
                rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise
            backup = path.with_name(path.name + f".partial-{time.time_ns()}")
            backup.write_bytes(line)
            with path.open("r+b") as stream:
                stream.truncate(offset)
            print(f"Recovered interrupted journal tail: {backup}", flush=True)
        offset += len(line)
    if lines and not lines[-1].endswith(b"\n") and offset == path.stat().st_size:
        with path.open("ab") as stream:
            stream.write(b"\n")
    return rows


def save_memory(system, path, count, complete):
    state = {name: getattr(system, name) for name in
             ("turn_notes", "events", "profiles", "last_updated_event_id", "recent_turns_window")}
    state.update(count=count, complete=complete)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(state, stream)
    temporary.replace(path)


def answer_question(args, system, sample_id, index, qa, usage, predictions):
    llm = MeteredLLM(args, usage, sample_id, "qa", index)
    view = system.spawn_qa_view(SimpleNamespace(llm=llm), executor_workers=4)
    view.__class__ = StrictFPHM
    started = time.time()
    try:
        query = view._get_llm_json_response(
            prompts.QUERY_REWRITING_PROMPT.format(original_query=qa["question"]),
            {"name": "response", "schema": {"type": "object", "properties": {
                "keyword_query": {"type": "string"},
                "profile_retrieval_keys": {"type": "array", "items": {"type": "string"}}},
                "required": ["keyword_query", "profile_retrieval_keys"]}}, caller="generate_keyword_query")
        if not query["keyword_query"]:
            raise RuntimeError("Empty search query")
        context, trace = view.retrieve_for_query(qa["question"], query["keyword_query"], [],
                                                 0, 10, 10, return_trace=True)
        if qa["category"] == 2:
            prompt = (f"Based on the context: {context}, answer the following question. "
                      "Use DATE of CONVERSATION to answer with an approximate date. "
                      "Please generate the shortest possible answer, using words from the conversation "
                      f"where possible, and avoid using any subjects. Question: {qa['question']} Short answer:")
        else:
            prompt = (f"Based on the context: {context}, write an answer in the form of a short phrase "
                      "for the following question. Answer with exact words from the context whenever possible. "
                      f"Question: {qa['question']} Short answer:")
        response = view._get_llm_json_response(
            prompt, {"name": "response", "schema": {"type": "object", "properties": {
                "answer": {"type": "string"}}, "required": ["answer"]}}, caller="final_answer_generation")
        predictions.append(dict(sample=sample_id, index=index, question=qa["question"],
                                category=qa["category"], answer=qa["answer"], prediction=response["answer"],
                                evidence=qa.get("evidence", []), retrieved=trace["relevant_turn_ids"],
                                seconds=time.time()-started))
        print(f"QA {sample_id}:{index} complete", flush=True)
    finally:
        view.executor.shutdown(wait=True)
        llm.client.close()


def process_sample(args, sample, usage, predictions, finished, qa_pool):
    sample_id = sample["sample_id"]
    llm = MeteredLLM(args, usage, sample_id, "construction")
    system = StrictFPHM(SimpleNamespace(llm=llm), sample_id, use_character_profile=False,
                        use_event_metadata_mode=True, ablation_no_link=True,
                        k_event_affiliation=10, log_dir=str(args.output / "logs"))
    checkpoint = args.output / f"{sample_id}.pkl"
    count, complete = 0, False
    if checkpoint.exists():
        with checkpoint.open("rb") as stream:
            state = pickle.load(stream)
        count, complete = state.pop("count"), state.pop("complete")
        system.__dict__.update(state)
        system.build_indices()
    conversation = parse_conversation(sample["conversation"])
    turns = [(turn, session.date_time) for sid, session in sorted(conversation.sessions.items())
             for turn in sorted(session.turns, key=lambda item: int(item.dia_id.split(":")[1]))]
    try:
        if not complete:
            for position in range(count, len(turns)):
                turn, timestamp = turns[position]
                system.add_turn(turn.dia_id, turn.text, turn.speaker, timestamp)
                if (position + 1) % 20 == 0:
                    save_memory(system, checkpoint, position + 1, False)
                    print(f"BUILD {sample_id} {position + 1}/{len(turns)}", flush=True)
            system.finalize_memory_build()
            system.build_indices()
            save_memory(system, checkpoint, len(turns), True)
        selected = [(index, qa) for index, qa in enumerate(sample["qa"]) if qa["category"] in (1, 2, 3, 4)]
        if args.pilot_questions:
            selected = selected[:args.pilot_questions]
        jobs = [qa_pool.submit(answer_question, args, system, sample_id, index, qa, usage, predictions)
                for index, qa in selected if (sample_id, index) not in finished]
        for job in as_completed(jobs):
            job.result()
    finally:
        system.executor.shutdown(wait=True)
        llm.client.close()


def report(args, expected, elapsed):
    from official_locomo_evaluation import eval_question_answering
    rows = read_rows(args.output / "predictions.jsonl")
    unique = {(row["sample"], row["index"]): row for row in rows}
    if len(unique) != len(rows):
        raise RuntimeError("Duplicate predictions")
    if set(unique) - expected:
        raise RuntimeError("Predictions outside selected QA set")
    if rows:
        scores, _, _ = eval_question_answering(rows)
    else:
        scores = []
    category = defaultdict(list)
    for row, score in zip(rows, scores):
        row["official_f1"] = float(score)
        category[row["category"]].append(float(score))
    tokens = defaultdict(lambda: dict(prompt_tokens=0, completion_tokens=0, total_tokens=0, calls=0))
    errors, missing, truncated = 0, 0, 0
    for row in read_rows(args.output / "usage.jsonl"):
        if "error" in row:
            errors += 1
            continue
        if row.get("usage") is None:
            missing += 1
            continue
        totals = tokens[row["phase"]]
        totals["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[key] += row["usage"][key]
        truncated += row["finish_reason"] != "stop"
    result = dict(expected=len(expected), evaluated=len(rows), complete=set(unique)==expected,
                  official_f1=float(sum(scores)/len(scores)) if scores else None,
                  by_category={key: dict(count=len(values), f1=sum(values)/len(values))
                               for key, values in category.items()}, server_tokens=dict(tokens),
                  failed_requests=errors, missing_usage=missing, truncated_requests=truncated,
                  invocation_wall_seconds=elapsed,
                  token_note="Includes retries and prior invocations in this directory. Failed requests without usage are unmetered.")
    (args.output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "scored_predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--api-base", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--dataset", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument("--output", type=Path, default=Path("vast_run"))
    parser.add_argument("--sample-workers", type=int, default=10)
    parser.add_argument("--qa-workers", type=int, default=8)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--pilot-questions", type=int, default=0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = json.loads(args.dataset.read_text(encoding="utf-8"))
    assert sum(qa["category"] in (1, 2, 3, 4) for sample in samples for qa in sample["qa"]) == 1540
    manifest = dict(model=args.model, dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                    thinking=False, max_tokens=None, temperature=0, k_event=10, k_turn=10,
                    profiles=False, links=False, event_mode="metadata", categories=[1, 2, 3, 4],
                    event_affiliation_safety={"max_tokens": EVENT_AFFILIATION_MAX_TOKENS,
                                              "reasoning_max_chars": 2048,
                                              "affiliations_max_items": 11,
                                              "affiliation_max_chars": 128},
                    evaluator_sha256=hashlib.sha256(Path("official_locomo_evaluation.py").read_bytes()).hexdigest(),
                    code_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in ("run_vast.py", "fphm_core.py", "prompts.py", "memory_layer.py")})
    manifest["serving_config"] = Path("vast_vllm.conf").read_text()
    manifest["versions"] = {name: importlib.metadata.version(name)
                            for name in ("vllm", "torch", "transformers", "sentence-transformers", "openai")}
    manifest["model_revision"] = Path("/workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/refs/main").read_text().strip()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Incompatible experiment manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.sample:
        samples = [sample for sample in samples if sample["sample_id"] == args.sample]
        if not samples:
            raise ValueError("Unknown sample")
    expected = {(sample["sample_id"], index) for sample in samples
                for index in [i for i, qa in enumerate(sample["qa"])
                              if qa["category"] in (1, 2, 3, 4)][:args.pilot_questions or None]}
    started = time.time()
    failures = []
    if not args.report_only:
        usage = Journal(args.output / "usage.jsonl")
        predictions = Journal(args.output / "predictions.jsonl")
        read_rows(usage.path)
        finished = {(row["sample"], row["index"]) for row in read_rows(predictions.path)}
        with ThreadPoolExecutor(args.qa_workers) as qa_pool, ThreadPoolExecutor(args.sample_workers) as build_pool:
            jobs = [build_pool.submit(process_sample, args, sample, usage, predictions, finished, qa_pool)
                    for sample in samples]
            for job in as_completed(jobs):
                try:
                    job.result()
                except Exception as exc:
                    failures.append(str(exc))
                    print(f"SAMPLE FAILED: {exc}", flush=True)
    result = report(args, expected, time.time()-started)
    if not result["complete"] or failures:
        raise SystemExit("Evaluation incomplete")


if __name__ == "__main__":
    main()
