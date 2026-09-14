# LangMem native sessions v2: retain upstream handled patch losses

This is an explicitly versioned LongMemEval integration of authentic LangMem, not an official LangMem LongMemEval paper reproduction. It keeps the v1 source-session mapping, model settings, retrieval, QA, native defaults, and native library bytes. The only execution-policy change is that the harness records four pinned Trustcall _ExtractUpdates._teardown branches that discard a PatchDoc and return normally instead of aborting the entire history after that normal return.

## Why v2 exists

The RTX 5090 v1 smoke stopped in the harness's ingest_sessions -> failures.check, after the ninth manager.invoke returned. The first eight sessions were journaled. All nine physical Qwen requests returned HTTP 200 with tool_calls finish reason, no proxy error, and the correct Qwen3.5-9B model. Trustcall logged "Could not apply patch: can't replace a non-existent object ''"; its unchanged code catches this exception, drops that operation, and preserves other valid tool calls. The harness imposed a stricter lossless-write requirement than the native baseline.

The exact generated patch was not captured: the existing call observer did not serialize the returned response wrapper, so all nine chat response fields are null. The content-free proxy journal independently has response model, finish reason, HTTP status, and token usage. We do not claim to have reconstructed the real patch or diagnosed the server parser. A synthetic CPU test of the actual native function AST reproduces the same logged error and proves its handled-return behavior.

The failed v1 attempt and protocol remain unchanged. A v2 run must use its own fresh directory, runs/langmem_native_sessions_v2; it cannot reuse a v1 completion. The runner source hash and explicit policy.harness_version change its immutable protocol identity.

## Retained provenance and configuration

- LangMem commit 9d033b47d9ce53e37e92c92241b0496c0278932e, version 0.0.30.
- Original Trustcall 0.0.39 wheel SHA256 d7da42e0bba816c0539b2936dfed90ffb3ea8d789e548e73865d416f8ac4ee64.
- All 38 upstream source/metadata files, licenses, and source_manifest.json are copied byte-for-byte from v1. Trustcall _base.py SHA256 is a2102f4a66dd724e028412f7730a205de5f6e731a43076f3fce77e5f6b2e907e.
- Qwen/Qwen3.5-9B, tokenizer revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, native memory temperature 0. The external runtime receipt must still verify actual FP16 weights, thinking off, and runtime flags.
- MiniLM-L6-v2, 384 dimensions, original string embedding inputs. CPU FP32 and native 256-token behavior remain external runtime checks.
- One native manager call per full original public session, original roles/content/order, one separate public date/session system message. No flattening, source truncation, relabeling, added extraction instruction, or query during construction.
- Original inserts/updates enabled, deletes disabled, query_limit=5, no query model or extra phases, max_steps=1. Trustcall original default max_attempts=3.
- Existing benchmark retrieve_num=10, context packing, QA prompt, output cap, model parameters and strict over-context policy remain unchanged.
- Client versions and local-only loopback/EMPTY credential routing remain v1's pinned checks. No native function, API argument, response, retry, timeout, parser, or JSON Patch is modified.
- The canonical dataset must contain all 500 unique IDs with SHA256 d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442. The query, answer, question_type, and has_answer are excluded from construction.

## Handled losses versus terminal failures

Only exact pinned source path, function _teardown, log line and message prefix match the allowlist:

| Native log line | Native handled outcome |
| --- | --- |
| 788 | Missing target ID in existing schema list |
| 797 | Malformed or missing target schema kind |
| 823 | Patch application exception |
| 829 | Missing or empty target schema |

These branches continue exactly as upstream implements them. The observer records one native_handled_patch_drop per native log and excludes the corresponding trace branches from fatal tracking, so trace plus logger does not count one dropped operation twice. A drop is a complete discarded PatchDoc, not a count of individual JSON Patch entries. Ordinary native repair warnings are still retained.

Unhandled exceptions, other native ERROR records, invalid manager return type, blank QA, missing final AI message (filter_state:435), final schema-validation failure (filter_state:468), and native caught transport/other exceptions terminating _Patch (1084/1101) remain fatal. This deliberately narrow v2 does not silently reclassify unknown failures. There is no extra repair prompt, fallback memory algorithm, insert-only mode, output normalization or corrective retry.

## Artifacts, denominator and interpretation

Every completed session retains its input hash, source-turn count, native put count, timing, and handled-drop count in session_calls.jsonl. Each attempt must include sealed native_degradation.json with drop records and total count. build_complete.json and usage.json retain the total count; usage.json also retains native warnings. Failed attempts preserve diagnostics too. lossless_writes is false when a handled drop is observed and null otherwise; absence of this warning never proves semantic completeness.

Completion means all selected source sessions were processed and a nonblank native QA answer was generated. It does not mean that every requested memory update succeeded. The canonical shared smoke must still process all 53 sessions and 550 source turns. Native lost updates can reduce quality and must remain part of the reported baseline behavior.

Full500 requires 500 generated predictions, zero unresolved failed histories, and the immutable 500-ID population. Nothing filters warning-bearing examples from the denominator. A missing answer is incomplete, not a smaller successful evaluation. Official LongMemEval evaluation is a separate step using the pinned official code; no score is produced or claimed by these CPU tests.

## Validation and launch boundary

Run python -B -m unittest discover -s <candidate-directory> -p test_runner.py -v.

The regression suite covers all four actual native caught-drop AST branches and one-count accounting, all 53 sessions/550 turns after a ninth-session drop, existing native repair behavior, sync/async transport aborts, unknown/error paths, worker-level handled drop plus QA versus fatal/unhandled/blank cases, source hashes/defaults, label isolation, endpoint guards, artifact seals and a mocked full500 run preserving every ID. Tests do not make GPU/model API calls.

The v2 candidate is locally implemented. The parent operator owns independent Python review, runtime receipt binding, deployment, a real full-history smoke, queue integration, and the full500 launch. This file is not evidence of a completed live run.