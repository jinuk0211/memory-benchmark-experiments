# LangMem native session candidate (local, not deployed)

This candidate restores structured conversation input to the authentic LangMem library. It is an explicitly documented LongMemEval integration; it is not an official LangMem LongMemEval paper reproduction. No official LongMemEval/LoCoMo evaluation driver was found in the pinned LangMem repository. Its examples accept user/assistant message lists and permit application-selected conversation processing boundaries.

## Provenance

- LangMem repository: https://github.com/langchain-ai/langmem
- Commit: `9d033b47d9ce53e37e92c92241b0496c0278932e`; project version `0.0.30`.
- `upstream/langmem/` contains all 24 original `src/langmem/*.py` files plus LICENSE, README, pyproject and the background/delayed-processing guides. Every file is an unchanged official byte copy.
- Trustcall is the original PyPI `trustcall==0.0.39` wheel. Wheel SHA256: `d7da42e0bba816c0539b2936dfed90ffb3ea8d789e548e73865d416f8ac4ee64`.
- `upstream/trustcall/` preserves the wheel's package, distribution metadata and license bytes. `source_manifest.json` records all 38 source/metadata files, URLs, individual SHA256 values, and wheel members.
- The runner imports these pinned modules before the existing adapter and validates the actual loaded module paths and hashes. An already imported native package or a different source file is refused.
- The existing experiment's frozen LangMem Python files also match this official revision after CRLF/LF normalization. The problem being restored is the experiment input adapter, not a claim that the LangMem core was rewritten.

## Input mapping and retained behavior

| Component | Candidate policy |
| --- | --- |
| Construction input | One `manager.invoke({"messages": ...})` per original public session, in dataset order |
| Source turns | Original `role` and `content` only; no rewriting, role reassignment, character splitting, truncation or reordering |
| Public date | One separate system message `Session {session_id} ({date})` precedes that session's turns |
| Why metadata is a message | The original manager has no dedicated timestamp argument; this is explicit cross-dataset metadata mapping, not a new extraction instruction |
| Source isolation | `answer`, `question_type`, turn-level `has_answer` and all unlisted fields are removed before worker input is created; the question is passed only during QA |
| Store | Existing per-history `InMemoryStore`, namespace `("benchmark", "memories")`, MiniLM embedding endpoint and `fields=["content"]` |
| Memory settings | Original default instructions/schema; inserts and updates enabled, public factory default `enable_deletes=False`, `query_limit=5`, no query model, no extra phases, `max_steps=1` |
| Trustcall | Original tools, parser, JSONPatch logic and default `max_attempts=3`; no new repair prompt, normalization, retries or output limit |
| Memory temperature | Existing benchmark adapter's `ChatOpenAI(temperature=0)`, retained; this is not claimed to be an upstream universal default |
| Final QA | Existing `native_five.native_answer` -> `AgentWrapper._handle_benchmark_memory_agent` -> native store retrieval -> existing benchmark packing and QA prompts |
| QA settings | Existing `retrieve_num=10`, question date, context packing, final model parameters and strict over-context error policy unchanged |

The old path used `native_five.source_chunks` to pack approximately 4096 characters of complete turns, then `LangMemAdapter.add_chunk` presented each bundle as a single user message. This changed the role formatting and message boundaries used by native `get_conversation` and `get_dialated_windows`.

The canonical shared smoke history `e47becba` has 53 sessions and 550 turns (273 user, 277 assistant). Four sessions start with assistant and four have odd lengths. The candidate makes 53 native manager calls and keeps all 550 turns. It does not impose artificial user/assistant pairs. The previous path made 170 single-user bundle calls.

The original adapter already passes the store and namespace explicitly when constructing the manager. Direct `manager.invoke` needs no extra storage context. The old `add_chunk` source-ID parsing/pruning is bypassed together with its text flattening. LongMemEval session IDs are metadata, not LoCoMo dialogue-evidence IDs; no fabricated source-ID evaluation metadata is attached. Native memory save/retrieval and existing QA integration are reused.

## Local model routing and runtime boundary

The runner only permits explicit loopback HTTP `/v1` endpoints and `EMPTY` credentials. Both `OpenAI` and `AsyncOpenAI` client creation and their chat/embedding calls are guarded. The chat model is fixed to `Qwen/Qwen3.5-9B`; embeddings are fixed to `sentence-transformers/all-MiniLM-L6-v2`, dimension 384. Embedding requests must contain original strings, not tokenizer ID arrays. Wrapper observations return original native results unchanged, including valid responses that native parsing or repair must handle. No API/model kwargs, timeout, stream policy or retry settings are changed. LangSmith tracing is disabled for this offline benchmark.

The actual shared client environment reported by the root operator is CPython 3.12.14, LangMem 0.0.30, Trustcall 0.0.39, langchain-openai 0.3.35, langchain-core 0.3.86, langgraph 1.0.1, OpenAI 2.54.0. The runner checks these six package versions and records the actual Python version. It does not assume Python 3.11 or install/upgrade packages. Existing client dependency requirements must remain satisfied; actual import and `pip check` verification are runtime checks.

Qwen tokenizer revision is fixed to `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; the existing HuggingFaceTokenizerAdapter guard rejects approximate/tiktoken fallback. Tokenizer files and source identity are included in the protocol. MiniLM weight revision, CPU FP32/native 256-token embedding behavior, Qwen weights/FP16 and `enable_thinking=false` must additionally be verified by the root's new recovery runtime receipt. Standalone CPU tests or this runner do not prove live service weights, precision or behavior. `source_manifest.json` and `runner.py` are present for `official_recovery/verify_recovery_runtime.py`'s candidate gate.

## Native failures, original repairs and transport

Observation does not modify original functions or return values. Original validation/tool repair continues normally. Specifically, `_get_message_op` errors at Trustcall `_base.py:1476/1480` can be repaired by the next native attempt; these are retained as warnings, and successful recovery remains admissible.

The observer monitors the exact pinned source in both the main thread and worker threads. These terminal branches reject an attempt:

- `_ExtractUpdates._teardown`: discarded PatchDocs, including a missing target or failed patch application (795/804/823/829).
- `filter_state`: no final AI message (435) or a final schema validation failure (468).
- `_Patch.invoke/ainvoke`: caught transport/other exception leading directly to `Command(goto="__end__")` (1101/1084).
- Other native ERROR records, uncaught exceptions, incomplete attempts and a blank final QA answer.

A valid model response with no useful memories is allowed. No-op patches and ordinary native validation retries are not converted into extra model calls or arbitrary failures. The observer's terminal source-line map is bound to the unchanged official file hash.

The prior stopped LangMem attempt had three 600-second proxy ReadTimeout events and subsequently three discarded PatchDoc errors. Session input restoration is independent of that transport problem. It provides no guarantee of solving timeouts or invalid patches; a full session can be longer than a former bundle. This candidate does not change the proxy timeout or generation cap and does not accept lost patches to obtain a score.

## Artifacts and resume

Every question runs in a fresh subprocess. The canonical dataset must have exactly 500 unique IDs and SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. Optional selected IDs affect status/selection only, while the immutable protocol always contains the full population. A smoke can therefore be reused during a subsequent full run with the identical protocol.

Each attempt preserves source/query/worker input, the exact native session message lists, completed-session journal, native configs/module paths/package versions, memory snapshot/native saved files, model-call observations, timings, usage and prediction or failure. `completion.json` seals the complete artifact inventory and every artifact hash except the self-referential receipt and the parent's still-open console log. Added, missing or modified artifacts, identity drift, a failure file or a blank prediction invalidate reuse. Failed attempts are preserved and a new attempt rebuilds from the beginning; no partial native memory state is resumed.

`predictions.json`, `failures.json` and `status.json` use the existing native-five `langmem` format. The official exporter can consume completed all-500 predictions after its own checks. `officially_judged` remains false/zero. Queue wait on a shared GPU is included in wall time; the local proxy journal is the authoritative physical-call/token record, including SDK retries.

Example command shape, **not executed or authorized for deployment by this candidate**:

```bash
/workspace/longmemeval_s_native7_20260910/.venv-client/bin/python \
  /workspace/longmemeval_s_native7_20260910/official_recovery/langmem_native_sessions_v1/runner.py \
  --dataset /workspace/longmemeval_s_native7_20260910/longmemeval_s_cleaned.json \
  --run-dir /workspace/longmemeval_s_native7_20260910/official_recovery_runs/langmem_native_sessions_v1 \
  --api-base http://127.0.0.1:18083/v1 \
  --embedding-api-base http://127.0.0.1:18084/v1 \
  --tokenizer /workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --ids-file /workspace/longmemeval_s_native7_20260910/queue/smoke_ids.json
```

After a separately authorized deployment and runtime verification, the same command with the same run directory can resume failed work. Removing `--ids-file` selects all 500 without changing the population identity. Existing failed legacy `runs/langmem` must not be mixed with this new protocol.

## CPU validation

`python -m unittest discover -s <candidate-directory> -p test_runner.py -v`

Fifteen tests pass: exact official hashes/defaults; canonical 53 calls/550 turns; assistant-first/odd sessions and long-turn preservation; label isolation; synchronous/asynchronous local routing and original kwargs; raw-string embeddings; authentic AST repair error then successful recovery; authentic synchronous/asynchronous terminal abort; empty-memory versus missing-final-message distinction; existing QA invocation; failed-attempt preservation; artifact seals; source/module rejection; and smoke-to-larger-selection verified reuse. CLI `--help` also passes.

These are CPU/mocked transport tests, not GPU performance results, live native imports or a completed smoke. No remote changes, API calls, deployment, full-500 run or official evaluation were performed while creating this candidate. It is not part of the pending A-MEM/SimpleMem/Mem0 combined transfer archive.