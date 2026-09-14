# SimpleMem native Dialogues v2: parallel request metering

Local candidate only. This version fixes the instrumentation failure in v1 without changing the original SimpleMem method. No v2 deployment, model request, GPU smoke, or full-500 run has been performed by these tests. The deployed/frozen v1, its failed attempt, and shared `request_metering.py` remain unchanged.

## Exact change

The v1 worker opens one outer `meter_operation("memory_write", sample_id=qid)` around native bulk construction. Its `Calls.bind` wrapper then attempted another metering scope inside each native worker thread. The shared meter deliberately rejects concurrent outer scopes, so every native window exhausted its three retries before reaching HTTP. The observed v1 failure contained 45 such errors for 15 windows.

In v2, `Calls.bind` invokes the original `chat_completion(messages, *args, **kwargs)` directly. It retains the existing per-call response/error, elapsed time, phase, lock, and count logging. The now-unused private `Calls` constructor arguments were removed. The outer `memory_init`, `memory_write`, and `qa` scopes remain in the worker.

The frozen meter already exposes the active process-wide immutable operation snapshot to HTTPX calls from internal worker threads. This preserves run ID, method, phase, and sample ID on every allowlisted physical request; QA also carries the question ID. The native ThreadPool still has 16 construction workers and 8 retrieval workers. No serialization lock, environment opt-out, new context implementation, retry, or fallback was added. Logical per-call timing remains in `llm_calls.jsonl`; `timing.jsonl` records the outer operation boundaries.

## Preserved source and method

`upstream/` and `source_manifest.json` are byte-identical to v1: all 20 files from official `aiming-lab/SimpleMem` commit `db80b6a7c591e0ea730a058e9f5fc4eb06572299`. The original parser, schema, prompts, native retry/fallback branches, window size 40, overlap 2, streaming, planning, reflection and caller-specific generation settings are unchanged. A list-valued `location` still fails the native `Optional[str]` schema. The original client still sends no `max_tokens` override.

The unchanged original converter maps each canonical turn to one Dialogue with its role, exact content, and session date. The 53-session/550-turn smoke input maps to 15 overlapping windows: 14 × 40 and 1 × 18. The original overlap counter may therefore be 578. `canonical_smoke_input_audit.json` is retained as the prior input-mapping evidence; it is not a v2 execution receipt. Gold answers, question types and turn labels remain excluded from source/query payloads.

The existing model condition remains local Qwen/Qwen3.5-9B plus the pinned MiniLM snapshot on CPU FP32, 384 dimensions and native 256-token truncation. No runtime package or model change is introduced here. The native `SimpleMemSystem.ask` path and public question-date prefix are unchanged.

Native terminal memory-window loss, discarded futures, DB errors and failed-answer sentinel remain fatal. Original recoverable retrieval and raw-text answer fallbacks remain warnings. The FTS guard still verifies creation, not full row coverage: after a sequential fallback, compare actual Tantivy coverage against all DB rows separately. The original backend does not recreate its Tantivy index after every later insertion; this candidate does not alter that behavior.

## Local tests

```text
python -B -m unittest discover -s official_recovery/simplemem_native_dialogues_v2 -p "test_*.py" -v
```

The retained 10 tests exercise unchanged original conversion, bulk memory creation, schema/retry behavior, native QA/fallbacks, label isolation, FTS guard and completion-seal integrity. Seven new tests load the actual frozen `request_metering.py` with HTTPX MockTransport:

- 16 simultaneously blocked worker threads retain exact memory-write and QA headers.
- The old nested-scope pattern fails before HTTP while the new wrapper succeeds.
- Positional/keyword arguments and responses pass through unchanged; errors propagate and are logged.
- Per-call records remain complete, outer scopes clean up, phases do not leak, and headers stay on allowlisted origins.
- The real native 550-turn/15-window fixture succeeds inside the actual outer metering scope.

All transports are fake. These tests do not establish live model success, semantic memory coverage, FTS row coverage, final-answer accuracy, or benchmark completion.

## New protocol and execution path

Use a new run directory such as `runs/simplemem_native_dialogues_v2`. The existing protocol hashes the candidate `runner.py`, so v2 has a different protocol digest even though `source_manifest.json`, upstream files, source/query data and all policy values are unchanged. Reusing v1's run directory fails its existing immutable protocol check. Do not import v1 predictions or failed attempts into v2.

Before any remote execution, the root must bind the entire v2 candidate and unchanged model/source dependencies to a new immutable runtime verification receipt and update the concrete execution plan. Existing runtime receipts do not attest the new runner. Those manifests, plans and remote actions are outside this candidate's changes.

A prospective smoke command uses the already prepared `.venv-simplemem-native0253` environment and a new run path:

```text
.venv-simplemem-native0253/bin/python official_recovery/simplemem_native_dialogues_v2/runner.py --method simplemem --dataset longmemeval_s_cleaned.json --run-dir runs/simplemem_native_dialogues_v2 --api-base http://127.0.0.1:18083/v1 --model Qwen/Qwen3.5-9B --embedding-model /workspace/.hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --ids-file queue/smoke_ids.json
```

A subsequent full run omits `--ids-file` and retains the v2 run directory to reuse only a verified successful v2 smoke. CLI flags and export schemas remain unchanged. Exclude `__pycache__` from bundles.
