# LangMem native sessions v6 candidate

Inactive integration candidate; this package does not deploy or launch anything. Place the complete directory at `official_recovery/langmem_native_sessions_v6` for server tests. The nested local candidate layout keeps the same harness-root relationship. All v5 artifacts remain separate; v6 protocol/source identities do not accept v5 completions.

## Correction and interpretation

The v5 observer turned four pinned native return/drop branches into question failures. v6 preserves the returned upstream values and counts each observation once:

- Trustcall `filter_state`, line 468: the actual caught exception log records one `native_filter_drop`, including its exception class and message. The native loop continues and retains valid responses. The caught block includes schema validation and response metadata assembly; this counter does not claim every exception was a schema error or exactly one successful write was lost.
- `filter_state`, line 435: one `native_empty_extraction` for the completed empty-output return, rather than duplicated multiline trace events. If the original LangMem manager subsequently raises `IndexError` when accessing the absent final message, that exception still fails the worker.
- `_Patch.ainvoke` line 1084 and `_Patch.invoke` line 1101: one `native_patch_abort` for each completed `Command(goto="__end__")` return, excluding async suspension events. No native retry or repair is added. SDK transport, malformed response, and content-filter failures are still captured independently by unchanged `Calls.check`, including when native `_Patch` catches them. Exceptions escaping native code, including `_tear_down`, still fail normally.

`native_degradation.json` is mandatory and sealed. It records contiguous event IDs, pinned path/line/function, source-session attribution, per-kind totals, affected sessions, and `lossless_writes: false` whenever losses were observed. `verified()` checks the v6 ledger version, event evidence, derived totals, and sealed bytes; resealing inconsistent counters does not make them valid. Session journals record each session's return-event count. Existing PatchDoc and raw response warning artifacts remain unchanged.

Completion means all source sessions were processed and QA was generated, not lossless memory construction. A native returned result may contain partial or zero new writes. The observer does not invent recovered memories, claim write counts, suppress escaping errors, or make question-specific exceptions.

## Fidelity

The 38 pinned upstream files and source manifest are byte-identical to v5. Memory and QA request handling (`Calls`), source-session mapping, native prompts, session order, token limits, sampling, SDK retries, native graph defaults, QA, and cache population remain unchanged. The memory serving adaptation remains 8,192 output tokens, SDK retries 0, and the fixed Qwen non-thinking recipe already disclosed in v5. This is a cross-dataset LangMem library integration, not an official LangMem LongMemEval driver. It generates answers only; no official judge scores are claimed here.

## CPU verification

Run from the experiment root:

```text
python -m unittest discover -s langmem_v6_candidate/langmem_native_sessions_v6 -p test_*.py -v
```

For the deployed package use `-s official_recovery/langmem_native_sessions_v6`. Local verification: all 42 tests passed in 50.227 seconds on Python 3.12.11. The 32 existing regression tests cover source fidelity, canonical 500-question cache protocol, original sessions, fixed memory serving, QA isolation, SDK failures, raw-response evidence, and native handling. Ten new tests execute original pinned ASTs with unchanged source line numbers and synthetic dependencies: mixed invalid/valid Pydantic Memory tools; empty returns; sync/async patch aborts; caught SDK transport/content-filter errors; escaping teardown errors; the real MemoryManager downstream empty-message failure; unchanged per-session inputs/call counts; log provenance; mandatory/corrupt sealed ledgers; and v5/v6 AST equality of Calls and QA/worker behavior. No model, HTTP, or remote calls are made. Actual pinned server-environment tests remain a rollout prerequisite.

Ruff finds only the two inherited E402 bootstrap imports in the runner; the new focused tests pass Ruff. The test AST replay intentionally compiles only trusted hash-pinned local source.

## Frozen source hashes

- v6 runner: `477c513ed0ed5f35cc37fcb5beb875dd3a2c2a8ba392539c8ac95ae7f2fca787`
- v5 reference runner: `a65da06dd85d4e2ebaabe0ef9b282cc6bda81d7ff21da01cc9a9c01720b15e77`
- pinned Trustcall `_base.py`: `a2102f4a66dd724e028412f7730a205de5f6e731a43076f3fce77e5f6b2e907e`
- source manifest: `13d65563d44db7b9f5e63edbe0db09a7d530bb6d0db0de0ad83e0573ba1a7c94`
