"""Judge frozen LongMemEval answers with the exact upstream payloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
from typing import Any
import sys

from openai import OpenAI
from prepare_inputs import MODEL, ROOT, canonical, prepare


def parse_label(content: Any) -> bool:
    if not isinstance(content, str) or content.strip().lower() not in ("yes", "no", "yes.", "no."):
        raise ValueError("Non yes/no response; leave pending")
    return "yes" in content.strip().lower()


def load_cache(path: Path, requests: dict[str, Any]) -> dict[str, Any]:
    cache = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = row["payload_hash"]
        if key not in requests:
            raise ValueError("Journal contains an unknown payload")
        if row.get("status") != "ok":
            continue
        if (row["returned_model"] != MODEL or row.get("finish_reason") != "stop"
                or type(row["label"]) is not bool
                or parse_label(row["content"]) != row["label"]):
            raise ValueError("Invalid saved model or verdict")
        if key in cache:
            raise ValueError("Duplicate successful payload in journal")
        cache[key] = row
    return cache


def summarize(rows: list[dict[str, Any]], cache: dict[str, Any]) -> dict[str, Any]:
    methods = {}
    scored = []
    for row in rows:
        result = cache.get(row["payload_hash"])
        label = result["label"] if result else None
        scored.append({**row, "correct": label})
        method = methods.setdefault(row["method"], {"available": 0, "judged": 0,
                                                   "correct": 0})
        method["available"] += 1
        if label is not None:
            method["judged"] += 1
            method["correct"] += label
    for method in methods.values():
        method["expected"] = 50
        method["missing_predictions"] = 50 - method["available"]
        method["pending_judgments"] = method["available"] - method["judged"]
        method["accuracy_on_available"] = (
            method["correct"] / method["judged"] if method["judged"] == method["available"]
            and method["judged"] else None)
        method["accuracy_on_same50"] = (
            method["correct"] / 50 if method["judged"] == 50 else None)
    return {"judge_model": MODEL, "scope": "common50 single-session-user completed subset, not full500",
            "methods": methods, "scored_rows": scored}


def judge_one(client: OpenAI, key: str, payload: dict[str, Any]) -> dict[str, Any]:
    row = {"payload_hash": key, "utc": datetime.now(timezone.utc).isoformat()}
    try:
        response = client.chat.completions.create(**payload)
    except Exception as exc:
        row.update(status="request_error", error_type=type(exc).__name__,
                   http_status=getattr(exc, "status_code", None))
        return row
    row.update(returned_model=response.model, response_id=response.id,
               usage=response.usage.model_dump() if response.usage else None)
    try:
        choice = response.choices[0]
        row.update(content=choice.message.content, finish_reason=choice.finish_reason)
        if response.model != MODEL or choice.finish_reason != "stop":
            raise ValueError("Model mismatch or incomplete output")
        row.update(label=parse_label(choice.message.content), status="ok")
    except (ValueError, IndexError, AttributeError) as exc:
        row.update(status="invalid_response", error_type=type(exc).__name__)
    return row


def write_summary(rows: list[dict[str, Any]], requests: dict[str, Any],
                  cache: dict[str, Any], journal: Path) -> None:
    summary = summarize(rows, cache)
    receipts = ([json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
                if journal.exists() else [])
    summary["unique_requests"] = len(requests)
    summary["unique_judged"] = len(cache)
    summary["api_attempts"] = len(receipts)
    summary["reported_usage"] = {
        field: sum((r.get("usage") or {}).get(field, 0) for r in receipts)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
    summary["missing_usage_attempts"] = sum(not r.get("usage") for r in receipts)
    for name, value in (("results.json", summary),):
        temporary = ROOT / (name + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(ROOT / name)
    print(canonical({k: v for k, v in summary.items() if k != "scored_rows"}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--api-key-stdin", action="store_true")
    args = parser.parse_args()
    rows, requests, _ = prepare()
    journal = ROOT / "judge_responses.jsonl"
    cache = load_cache(journal, requests)
    pending = {key: payload for key, payload in requests.items() if key not in cache}
    if not args.run or not pending:
        write_summary(rows, requests, cache, journal)
        return 0
    credential = os.environ.get("OPENAI_API_KEY", "")
    if args.api_key_stdin:
        credential = (getpass.getpass("OpenAI API key (hidden): ") if sys.stdin.isatty()
                      else sys.stdin.readline().strip())
    if not credential:
        print("AUTHENTICATION_REQUIRED: set OPENAI_API_KEY or use --api-key-stdin",
              flush=True)
        write_summary(rows, requests, cache, journal)
        return 2
    with OpenAI(api_key=credential, base_url="https://api.openai.com/v1",
                timeout=45.0, max_retries=0) as client:
        del credential
        # Check one real judgment before dispatching the rest.
        items = list(pending.items())
        first_key, first_payload = items.pop(0)
        first = judge_one(client, first_key, first_payload)
        with journal.open("a", encoding="utf-8", newline="\n") as out:
            out.write(canonical(first) + "\n")
            out.flush()
            os.fsync(out.fileno())
            if first["status"] != "ok":
                write_summary(rows, requests, cache, journal)
                print("FIRST_REQUEST_FAILED: inspect sanitized judge_responses.jsonl")
                return 1
            cache[first_key] = first
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(judge_one, client, key, payload) for key, payload in items]
                for future in as_completed(futures):
                    row = future.result()
                    out.write(canonical(row) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    if row["status"] == "ok":
                        cache[row["payload_hash"]] = row
                    print(canonical({"judged": len(cache), "total": len(requests),
                                     "last_status": row["status"]}), flush=True)
    write_summary(rows, requests, cache, journal)
    return 0 if len(cache) == len(requests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
