"""Resume exact LightMem LoCoMo GPT-4o-mini judging of saved predictions."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

PROMPT_SOURCE = Path("D:/MemoryData/lightmem_official_20260909/upstream/experiments/locomo/llm_judge.py")
MODEL = "gpt-4o-mini"
CATEGORIES = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_prompt(path: Path = PROMPT_SOURCE) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ACCURACY_PROMPT"
            for target in node.targets
        ):
            prompt = ast.literal_eval(node.value)
            if not isinstance(prompt, str):
                raise ValueError("Official ACCURACY_PROMPT is not a string")
            return prompt
    raise ValueError("Official ACCURACY_PROMPT was not found")


def make_payload(row: dict, prompt: str) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt.format(
            question=row["question"], gold_answer=row["gold_answer"],
            generated_answer=row["generated_answer"],
        )}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }


def payload_hash(payload: dict) -> str:
    return digest(canonical_json(payload).encode("utf-8"))


def parse_label(content: str | None) -> str:
    if not isinstance(content, str):
        raise ValueError("Judge response has no text")
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content.strip(), re.DOTALL)
    data = json.loads(match.group(1) if match else content.strip())
    if not isinstance(data, dict) or data.get("label") not in ("CORRECT", "WRONG"):
        raise ValueError("Judge response must contain a CORRECT or WRONG label")
    return data["label"]


def validate_rows(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Input has no rows")
    ids: set[str] = set()
    method_ids: dict[str, set[tuple]] = defaultdict(set)
    golds: dict[tuple, tuple] = {}
    for row in rows:
        for field in ("id", "method", "conversation_id", "question", "gold_answer", "generated_answer"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"Input field {field} must be a string")
        if row["id"] in ids:
            raise ValueError("Duplicate input row id")
        ids.add(row["id"])
        if type(row.get("qa_index")) is not int or row["qa_index"] < 0:
            raise ValueError("Input qa_index must be a nonnegative integer")
        if type(row.get("category")) is not int or row["category"] not in CATEGORIES:
            raise ValueError("Input categories must be 1 through 4")
        identity = (row["conversation_id"], row["qa_index"])
        if identity in method_ids[row["method"]]:
            raise ValueError("Duplicate method/question identity")
        method_ids[row["method"]].add(identity)
        gold = (row["category"], row["question"], row["gold_answer"])
        if identity in golds and golds[identity] != gold:
            raise ValueError("Methods do not share the same question/category/gold")
        golds[identity] = gold
    question_sets = list(method_ids.values())
    if any(questions != question_sets[0] for questions in question_sets[1:]):
        raise ValueError("Methods do not cover identical question sets")


def read_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records = []
    offset = 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt checkpoint: {path.name}") from None
            # Preserve an interrupted final write before repairing the append boundary.
            tail = path.with_name(path.name + ".interrupted-tail")
            tail.write_bytes(raw[offset:])
            path.write_bytes(raw[:offset])
            break
        if not isinstance(record, dict):
            raise ValueError(f"Invalid checkpoint record: {path.name}")
        records.append(record)
        offset += len(line)
    if records and raw and not raw.endswith(b"\n") and offset == len(raw):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def usage_cost(usage: dict, prices: dict) -> float:
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    if not 0 <= cached <= prompt_tokens or completion_tokens < 0:
        raise ValueError("Invalid API token usage")
    return ((prompt_tokens - cached) * prices["input"] + cached * prices["cached_input"]
            + completion_tokens * prices["output"]) / 1_000_000


def attempt_cost_upper_bound(payload: dict, prices: dict) -> float:
    # UTF-8 bytes bound ordinary prompt tokens; 16,384 is this model's output limit.
    return ((len(canonical_json(payload).encode("utf-8")) + 128) * prices["input"]
            + 16_384 * prices["output"]) / 1_000_000


def rate_headers(headers: Any) -> dict:
    names = ("x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
             "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
             "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens", "retry-after")
    return {name: str(headers[name])[:128] for name in names if headers is not None and name in headers}


def safe_error(error: Exception) -> dict:
    status = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    code = body.get("code") if isinstance(body, dict) else None
    if code is None and isinstance(body, dict) and isinstance(body.get("error"), dict):
        code = body["error"].get("code")
    # Only permit known identifier characters; never serialize exception text or headers.
    safe_code = code if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_]{1,64}", code) else None
    return {"error_type": type(error).__name__, "status_code": status, "error_code": safe_code}


def is_transient(error: Exception) -> bool:
    details = safe_error(error)
    if details["error_code"] in ("insufficient_quota", "billing_hard_limit_reached"):
        return False
    status = details["status_code"]
    return (isinstance(error, (APIConnectionError, APITimeoutError)) or status == 429
            or (isinstance(status, int) and 500 <= status <= 599))


class Journal:
    def __init__(self, output_dir: Path, prices: dict, prompt_sha: str):
        self.output_dir = output_dir
        self.prices = prices
        self.prompt_sha = prompt_sha
        self.lock = threading.Lock()
        self.events = read_journal(output_dir / "api_events.jsonl")
        if any(event.get("kind") == "response" and event.get("usage") is None for event in self.events):
            raise ValueError("A prior response has unknown usage; resolve billing before resuming")
        self.cache: dict[str, dict] = {}
        cache_records = read_journal(output_dir / "judge_cache.jsonl")
        recorded_response_ids = {event.get("response_id") for event in self.events
                                 if event.get("kind") == "response"}
        recovered_events = []
        for record in cache_records:
            if record.get("usage") is None:
                raise ValueError("Cached response has unknown token usage")
            if not record.get("response_id"):
                raise ValueError("Cached response is missing its billing identity")
            if record["response_id"] not in recorded_response_ids:
                recovered_events.append(record)
                recorded_response_ids.add(record["response_id"])
        self.events.extend(recovered_events)
        # Recover an already billed valid response if interrupted before the cache append.
        for record in self.events + cache_records:
            if record.get("kind") != "response" or record.get("label") not in ("CORRECT", "WRONG"):
                continue
            key = record["payload_hash"]
            if record.get("prompt_sha256") != prompt_sha:
                raise ValueError("Checkpoint prompt hash mismatch")
            if key in self.cache and self.cache[key]["label"] != record["label"]:
                raise ValueError("Conflicting cached labels for the same request")
            self.cache[key] = record
        unique_billed = {event["response_id"]: event for event in self.events if event.get("kind") == "response"}
        self.spent = sum(float(event.get("cost_usd", 0)) for event in unique_billed.values())
        self.unknown_cost_upper_bound = sum(float(event.get("unknown_cost_upper_bound_usd", 0))
                                            for event in self.events)
        self.event_file = (output_dir / "api_events.jsonl").open("a", encoding="utf-8", newline="\n")
        for record in recovered_events:
            self.event_file.write(canonical_json(record) + "\n")
        self.event_file.flush()
        os.fsync(self.event_file.fileno())
        self.cache_file = (output_dir / "judge_cache.jsonl").open("a", encoding="utf-8", newline="\n")
        recorded = {record.get("payload_hash") for record in cache_records}
        for key, record in self.cache.items():
            if key not in recorded:
                self.cache_file.write(canonical_json(record) + "\n")
        self.cache_file.flush()

    def append(self, record: dict) -> None:
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **record,
                  "prompt_sha256": self.prompt_sha}
        with self.lock:
            self.event_file.write(canonical_json(record) + "\n")
            self.event_file.flush()
            os.fsync(self.event_file.fileno())
            self.events.append(record)
            self.spent += float(record.get("cost_usd", 0))
            self.unknown_cost_upper_bound += float(record.get("unknown_cost_upper_bound_usd", 0))
            if record.get("label") in ("CORRECT", "WRONG"):
                self.cache_file.write(canonical_json(record) + "\n")
                self.cache_file.flush()
                os.fsync(self.cache_file.fileno())
                self.cache[record["payload_hash"]] = record

    def close(self) -> None:
        self.event_file.close()
        self.cache_file.close()


def request_judgment(client: Any, payload: dict, journal: Journal, *, attempts: int = 5,
                     sleep=time.sleep) -> dict:
    key = payload_hash(payload)
    for attempt in range(1, attempts + 1):
        try:
            raw_response = client.chat.completions.with_raw_response.create(**payload)
            response = raw_response.parse()
            limits = rate_headers(raw_response.headers)
        except Exception as error:
            transient = is_transient(error)
            details = safe_error(error)
            limits = rate_headers(getattr(getattr(error, "response", None), "headers", None))
            status = details["status_code"]
            billing_unknown = (isinstance(error, (APIConnectionError, APITimeoutError))
                               or (isinstance(status, int) and status >= 500))
            journal.append({"kind": "error", "payload_hash": key, "attempt": attempt,
                            "transient": transient, "rate_limits": limits,
                            "unknown_cost_upper_bound_usd": attempt_cost_upper_bound(payload, journal.prices)
                            if billing_unknown else 0, **details})
            if not transient or attempt == attempts:
                return {"ok": False, "payload_hash": key, **details}
            try:
                delay = max(float(limits.get("retry-after", 0)), min(2 ** (attempt - 1), 30))
            except ValueError:
                delay = min(2 ** (attempt - 1), 30)
            sleep(min(delay, 60))
            continue
        usage = response.usage.model_dump() if response.usage is not None else None
        content = response.choices[0].message.content if response.choices else None
        label = None
        try:
            label = parse_label(content)
        except (ValueError, TypeError):
            pass
        record = {
            "kind": "response", "payload_hash": key, "attempt": attempt,
            "label": label if usage is not None else None,
            "response": content, "returned_model": response.model, "rate_limits": limits,
            "response_id": response.id, "request_id": getattr(response, "_request_id", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": response.choices[0].finish_reason if response.choices else None,
            "usage": usage, "prices_per_million_usd": journal.prices,
            "cost_usd": usage_cost(usage, journal.prices) if usage is not None else 0,
            "unknown_cost_upper_bound_usd": attempt_cost_upper_bound(payload, journal.prices)
            if usage is None else 0,
        }
        # Persist billed responses even when invalid; no failed row is scored as WRONG.
        journal.append(record)
        if usage is None:
            return {"ok": False, "payload_hash": key, "error_type": "MissingTokenUsage"}
        if label is None:
            return {"ok": False, "payload_hash": key, "error_type": "InvalidJudgeLabel"}
        return {"ok": True, "payload_hash": key}
    raise AssertionError("Unreachable retry state")


def summarize(rows: list[dict], keys: list[str], journal: Journal, reason: str) -> dict:
    expected = Counter(row["method"] for row in rows)
    methods: dict[str, dict] = {}
    scored_rows = []
    for row, key in zip(rows, keys, strict=True):
        cached = journal.cache.get(key)
        scored_rows.append({**row, "payload_hash": key,
                            "judge_label": cached["label"] if cached else None,
                            "judge_score": int(cached["label"] == "CORRECT") if cached else None})
    for method, total in expected.items():
        selected = [row for row in scored_rows if row["method"] == method]
        done = [row for row in selected if row["judge_score"] is not None]
        complete = len(done) == total
        categories = {}
        for category, name in CATEGORIES.items():
            cat_rows = [row for row in selected if row["category"] == category]
            cat_done = [row for row in cat_rows if row["judge_score"] is not None]
            correct = sum(row["judge_score"] for row in cat_done)
            categories[str(category)] = {
                "name": name, "expected": len(cat_rows), "scored": len(cat_done), "correct": correct,
                "accuracy_pct": 100 * correct / len(cat_rows) if complete and cat_rows else None,
            }
        correct = sum(row["judge_score"] for row in done)
        methods[method] = {"expected": total, "scored": len(done), "correct": correct,
                           "complete": complete,
                           "accuracy_pct": 100 * correct / total if complete else None,
                           "categories": categories}
    temporary = journal.output_dir / "scores.jsonl.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in scored_rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(journal.output_dir / "scores.jsonl")
    response_events = [event for event in journal.events if event.get("kind") == "response"]
    tokens = {name: sum(int((event.get("usage") or {}).get(name, 0)) for event in response_events)
              for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    tokens["cached_prompt_tokens"] = sum(
        int(((event.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        for event in response_events)
    complete = all(method["complete"] for method in methods.values())
    summary = {
        "status": "complete" if complete else "incomplete", "stop_reason": reason,
        "logical_rows": len(rows), "scored_rows": sum(value["scored"] for value in methods.values()),
        "unique_payloads": len(set(keys)), "cached_unique_payloads": len(set(keys) & journal.cache.keys()),
        "billed_responses": len(response_events), "error_events": len(journal.events) - len(response_events),
        "cost_usd": journal.spent, "unknown_cost_upper_bound_usd": journal.unknown_cost_upper_bound,
        "budget_committed_usd": journal.spent + journal.unknown_cost_upper_bound, "token_usage": tokens,
        "returned_models": dict(Counter(event.get("returned_model") for event in response_events)),
        "methods": methods,
    }
    write_json(journal.output_dir / "summary.json", summary)
    return summary


def run_evaluation(rows: list[dict], prompt: str, output_dir: Path, client: Any, *, workers: int = 8,
                   limit: int | None = None, budget_usd: float = 5.0,
                   prices: dict | None = None, input_sha: str | None = None) -> dict:
    validate_rows(rows)
    if workers < 1 or budget_usd <= 0 or (limit is not None and limit < 1):
        raise ValueError("Workers, budget, and limit must be positive")
    prices = prices or {"input": 0.15, "cached_input": 0.075, "output": 0.60}
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_sha = digest(prompt.encode("utf-8"))
    config = {"input_sha256": input_sha or digest(canonical_json(rows).encode("utf-8")),
              "prompt_sha256": prompt_sha, "model": MODEL, "temperature": 0.0,
              "response_format": {"type": "json_object"}, "prompt_source": str(PROMPT_SOURCE),
              "source_sha256": digest(PROMPT_SOURCE.read_bytes()),
              "deduplication": "Identical full API payloads share a judgment across methods"}
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if any(prior.get(key) != config[key] for key in ("input_sha256", "prompt_sha256", "model")):
            raise ValueError("Resume input/prompt/model does not match the existing run")
    else:
        write_json(config_path, config)
    payloads = {}
    keys = []
    for row in rows:
        payload = make_payload(row, prompt)
        key = payload_hash(payload)
        keys.append(key)
        payloads.setdefault(key, payload)
    journal = Journal(output_dir, prices, prompt_sha)
    pending_keys = [key for key in payloads if key not in journal.cache]
    if limit is not None:
        pending_keys = pending_keys[:limit]
    submitted = 0
    completed = 0
    reason = "all_requested_payloads_processed"
    last_progress = 0.0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active: dict[Any, float] = {}
            stopped = False
            while submitted < len(pending_keys) or active:
                while not stopped and submitted < len(pending_keys) and len(active) < workers:
                    key = pending_keys[submitted]
                    payload = payloads[key]
                    # Reserve all five possible attempts, including ambiguously billed retries.
                    reserve = 5 * attempt_cost_upper_bound(payload, prices)
                    if journal.spent + journal.unknown_cost_upper_bound + sum(active.values()) + reserve > budget_usd:
                        if not active:
                            stopped = True
                            reason = "budget_reservation_limit"
                        break
                    future = pool.submit(request_judgment, client, payload, journal)
                    active[future] = reserve
                    submitted += 1
                if not active:
                    break
                done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    result = future.result()
                    completed += 1
                    if not result["ok"]:
                        stopped = True
                        reason = result["error_type"]
                        print(canonical_json({"event": "request_failed", **result}), flush=True)
                now = time.monotonic()
                if done and (now - last_progress >= 15 or not active):
                    print(canonical_json({"event": "progress", "completed_this_run": completed,
                          "pending_this_run": len(pending_keys), "cached_unique": len(journal.cache),
                          "required_unique": len(payloads), "cost_usd": round(journal.spent, 6)}), flush=True)
                    last_progress = now
                if stopped and not active:
                    break
        summary = summarize(rows, keys, journal, reason)
        print(canonical_json({key: value for key, value in summary.items() if key != "methods"}), flush=True)
        return summary
    finally:
        journal.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--api-key-stdin", action="store_true", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--input-price", type=float, default=0.15)
    parser.add_argument("--cached-input-price", type=float, default=0.075)
    parser.add_argument("--output-price", type=float, default=0.60)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    validate_rows(rows)
    prompt = load_prompt()
    expected_prompt_sha = "62395dd312a631dfd9355026a0b69cc936018274c3198b6365b5c2a5c9bca9e0"
    manifest = json.loads((args.input.parent / "input_manifest.json").read_text(encoding="utf-8-sig"))
    if digest(prompt.encode("utf-8")) != expected_prompt_sha or manifest["prompt_sha256"] != expected_prompt_sha:
        raise ValueError("Official prompt does not match the verified immutable upstream")
    if manifest["input_sha256"] != digest(raw):
        raise ValueError("Input does not match its audited manifest")
    # TTY getpass disables echo. Plain pipe input is read without any prompt or logging.
    key = getpass.getpass("API key: ") if sys.stdin.isatty() else sys.stdin.readline().strip()
    if not key:
        print('{"event":"fatal","error_type":"MissingAPIKey"}', flush=True)
        return 2
    prices = {"input": args.input_price, "cached_input": args.cached_input_price, "output": args.output_price}
    if any(value < 0 for value in prices.values()):
        raise ValueError("Prices cannot be negative")
    try:
        with OpenAI(api_key=key, max_retries=0, timeout=90.0, base_url="https://api.openai.com/v1") as client:
            del key
            summary = run_evaluation(rows, prompt, args.output_dir, client, workers=args.workers,
                                     limit=args.limit, budget_usd=args.budget_usd, prices=prices,
                                     input_sha=digest(raw))
        return 0 if summary["status"] == "complete" or args.limit is not None else 1
    except Exception as error:
        print(canonical_json({"event": "fatal", **safe_error(error)}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())