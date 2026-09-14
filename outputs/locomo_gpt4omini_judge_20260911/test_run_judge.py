"""Offline behavioral tests; never contact a network or require credentials."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import AuthenticationError, APITimeoutError, RateLimitError

SPEC = importlib.util.spec_from_file_location("run_judge", Path(__file__).with_name("run_judge.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
PRICES = {"input": 0.15, "cached_input": 0.075, "output": 0.60}
PROMPT = "Q: {question}\nG: {gold_answer}\nA: {generated_answer}"


def row(method="A", answer="a"):
    return {"id": method + ":c:0", "method": method, "conversation_id": "c", "qa_index": 0,
            "category": 1, "question": "q", "gold_answer": "g", "generated_answer": answer}


def response(label="CORRECT", usage=True, identity="resp1"):
    tokens = {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
              "prompt_tokens_details": {"cached_tokens": 20}}
    parsed = SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: tokens) if usage else None,
                             choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"label": label})),
                                                      finish_reason="stop")],
                             model="gpt-4o-mini-2024-07-18", id=identity,
                             _request_id="request1", system_fingerprint="fp")
    return SimpleNamespace(parse=lambda: parsed, headers={"x-ratelimit-remaining-requests": "499"})


def client(*outcomes):
    create = Mock(side_effect=outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create)))), create


def journal(path):
    return runner.Journal(path, PRICES, runner.digest(PROMPT.encode()))


@pytest.mark.parametrize("content", ['{"label":"correct"}', '{"label":"WRONG or CORRECT"}',
                                      '[]', '{}', 'not json', None])
def test_strict_labels(content):
    with pytest.raises((ValueError, TypeError)):
        runner.parse_label(content)


def test_official_payload_is_unmodified():
    prompt = runner.load_prompt()
    assert runner.digest(prompt.encode()) == "62395dd312a631dfd9355026a0b69cc936018274c3198b6365b5c2a5c9bca9e0"
    payload = runner.make_payload(row(), prompt)
    assert payload == {"model": "gpt-4o-mini", "temperature": 0.0,
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "user", "content": prompt.format(
                           question="q", gold_answer="g", generated_answer="a")} ]}
    assert runner.payload_hash(payload) == runner.payload_hash(runner.make_payload(row("B"), prompt))


def test_deduplication_and_resume_do_not_rebill(tmp_path):
    api, create = client(response())
    rows = [row(), row("B")]
    summary = runner.run_evaluation(rows, PROMPT, tmp_path, api)
    assert create.call_count == 1
    assert summary["scored_rows"] == 2
    assert summary["methods"]["A"]["accuracy_pct"] == 100
    resumed, resumed_create = client()
    second = runner.run_evaluation(rows, PROMPT, tmp_path, resumed)
    assert resumed_create.call_count == 0
    assert second["cost_usd"] == summary["cost_usd"]
    assert second["status"] == "complete"


def test_transient_429_retries_but_auth_fails_fast(tmp_path):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    limited = RateLimitError("safe", response=httpx.Response(429, request=request),
                             body={"code": "rate_limit_exceeded"})
    api, create = client(limited, response())
    ledger = journal(tmp_path)
    sleeps = []
    result = runner.request_judgment(api, runner.make_payload(row(), PROMPT), ledger, sleep=sleeps.append)
    assert result["ok"] and create.call_count == 2 and sleeps == [1]
    ledger.close()
    failure = AuthenticationError("secret must never be logged", response=httpx.Response(401, request=request),
                                  body={"code": "invalid_api_key"})
    api, create = client(failure)
    ledger = journal(tmp_path)
    result = runner.request_judgment(api, runner.make_payload(row(answer="different"), PROMPT), ledger)
    ledger.close()
    assert not result["ok"] and create.call_count == 1
    assert "secret must never be logged" not in (tmp_path / "api_events.jsonl").read_text()


def test_invalid_label_is_billed_but_never_wrong(tmp_path):
    api, _ = client(response(label="MAYBE"))
    summary = runner.run_evaluation([row()], PROMPT, tmp_path, api, workers=1)
    assert summary["status"] == "incomplete"
    assert summary["methods"]["A"]["accuracy_pct"] is None
    assert summary["scored_rows"] == 0 and summary["cost_usd"] > 0
    assert summary["stop_reason"] == "InvalidJudgeLabel"


def test_missing_usage_is_not_cached_and_blocks_resume(tmp_path):
    api, _ = client(response(usage=False))
    summary = runner.run_evaluation([row()], PROMPT, tmp_path, api, workers=1)
    assert summary["scored_rows"] == 0
    assert summary["unknown_cost_upper_bound_usd"] > 0
    with pytest.raises(ValueError, match="unknown usage"):
        journal(tmp_path)


def test_cache_only_billing_recovery_counted_once(tmp_path):
    api, _ = client(response())
    original = runner.run_evaluation([row()], PROMPT, tmp_path, api)
    (tmp_path / "api_events.jsonl").write_text("")
    api, create = client()
    recovered = runner.run_evaluation([row()], PROMPT, tmp_path, api)
    assert recovered["cost_usd"] == original["cost_usd"]
    assert recovered["billed_responses"] == 1 and create.call_count == 0
    twice = runner.run_evaluation([row()], PROMPT, tmp_path, api)
    assert twice["cost_usd"] == original["cost_usd"]


def test_timeout_attempt_conservatively_consumes_budget(tmp_path):
    timeout = APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    api, create = client(timeout, response())
    ledger = journal(tmp_path)
    payload = runner.make_payload(row(), PROMPT)
    result = runner.request_judgment(api, payload, ledger, sleep=lambda seconds: None)
    assert result["ok"] and create.call_count == 2
    assert ledger.unknown_cost_upper_bound == runner.attempt_cost_upper_bound(payload, PRICES)
    known, unknown = ledger.spent, ledger.unknown_cost_upper_bound
    ledger.close()
    ledger = journal(tmp_path)
    assert ledger.spent == known and ledger.unknown_cost_upper_bound == unknown
    ledger.close()
    api, create = client(response(identity="resp2"))
    summary = runner.run_evaluation([row(), row("B", answer="different")], PROMPT,
                                    tmp_path / "low-budget", api, budget_usd=0.001)
    assert create.call_count == 0 and summary["stop_reason"] == "budget_reservation_limit"


def test_partial_smoke_has_no_incomplete_accuracy_then_resumes(tmp_path):
    rows = [row(), row("B", answer="different")]
    api, _ = client(response())
    first = runner.run_evaluation(rows, PROMPT, tmp_path, api, limit=1, workers=1)
    assert first["status"] == "incomplete" and first["methods"]["B"]["accuracy_pct"] is None
    api, _ = client(response(label="WRONG", identity="resp2"))
    second = runner.run_evaluation(rows, PROMPT, tmp_path, api, workers=1)
    assert second["status"] == "complete"
    assert second["methods"]["B"]["accuracy_pct"] == 0


def test_resume_rejects_changed_input(tmp_path):
    api, _ = client(response())
    runner.run_evaluation([row()], PROMPT, tmp_path, api)
    with pytest.raises(ValueError, match="Resume input"):
        runner.run_evaluation([row(answer="changed")], PROMPT, tmp_path, api)


def test_interrupted_tail_is_preserved_and_valid_prefix_used(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"ok":1}\n{"partial":')
    assert runner.read_journal(path) == [{"ok": 1}]
    assert path.read_bytes() == b'{"ok":1}\n'
    assert path.with_name("events.jsonl.interrupted-tail").read_bytes() == b'{"partial":'