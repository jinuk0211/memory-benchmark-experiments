# Mem0 OSS 0.1.94 native pairs v4 candidate

Inactive source package for independent review and transfer to the separately assigned Mem0 GPU task. Do not activate this candidate on the existing LangMem GPU. No services, controller, provider guard, or runtime are included or activated by this package. The nested local candidate layout has the same harness-root relationship as `official_recovery/mem0_native_pairs_v4`.

## Exact correction

Pinned Mem0 `memory/main.py` catches each action exception at lines 328-329, continues its loop, and returns accumulated native results at line 339. The v3 observer incorrectly made every line-329 log fatal, including errors accessing a malformed response entry or looking up a nonexistent temporary memory ID before a storage call.

v4 accepts only an authenticated pre-storage loss at that catch. It requires the current pair's original update response, the exact pinned native log path/function/line, an active exception whose traceback has exactly one frame in `_add_to_vector_store`, the native catch's identical exception object, matching parsed response, and matching current entry/mapping evidence. Supported origins are line 286 (`AttributeError` on a non-dict entry), line 304 (missing/unmapped/unhashable UPDATE ID), and line 318 (unmapped/unhashable DELETE ID). The numeric mapping comes from actual retrieved native memory; there are no fixed IDs or question-specific exceptions. `sys.exception()` is read inside synchronous logging, including when native work runs in its worker thread. No global tracing hook is installed.

The original loop, returned values, and valid writes before or after a dropped entry remain intact. A caught deeper exception from `_create_memory`, `_update_memory`, `_delete_memory`, embeddings, vector storage, or SQLite remains fatal to this integration, even when its message is the same KeyError and some side effects already occurred. SDK transport/response failures remain fatal. Unknown origins and errors that escape native code are not suppressed. Existing native JSON/shape fallbacks at lines 233, 278 and 331 remain as previously disclosed. Missing `memory`, empty text, `NONE`, and unknown actions preserve native silent behavior; this observer does not invent drop counts for unlogged cases. Other failures outside the three authenticated action origins remain failures.

## Artifact meaning and version separation

`native_degradation.json` is versioned and mandatory. Per-log warnings include ordered IDs, pair/call attribution, native location, raw-response hash and finish reason. Action warnings also retain the actual JSON entry, temporary mapping keys, exception class/message and origin. Counts are `native_warning_logs`, `native_affected_responses`, `native_affected_pairs`, and `native_action_drops`. The old `native_rejected_responses` label is replaced because a partly applied response was not rejected in its entirety. Affected responses can preserve partial native writes; neither zero loss nor recovered writes is claimed.

The existing gzip response journal remains unchanged. Both post-exit finalization and cache verification now check the degradation version, fatal state, warning totals/provenance, raw response sequence/hash links and action evidence. Recomputing receipt hashes around inconsistent counters or provenance does not make them valid. `lossless_writes` is false when any warning exists. Completion still means source-pair processing plus generated QA; official judging is separate.

v4 has a new implementation/protocol/source identity and does not reuse v3 completions. Preserve all older completed and failed artifacts separately. The protocol `native_fallbacks` disclosure string changed, so a coordinator that constructs an exact expected protocol must use the v4 string. Coordinator fixtures now need valid versioned degradation JSON and a valid gzip journal rather than `{}` or arbitrary bytes. An empty gzip journal is sufficient for synthetic zero-warning fixtures.

## Unchanged execution

All 97 vendor files, wheel, source manifest and requirements are byte-identical to v3. The original physical SDK create wrapper is AST-identical. Memory configuration, native prompts, source-pair ordering, telemetry opt-out fix, embedding/model calls, retries, worker lifecycle and QA are unchanged. This remains the documented cross-dataset OSS integration, not a cloud-paper reproduction. No memory reconstruction, repair, retry or model request is added by the observer correction.

## Verification receipt

All 26 CPU tests passed in 2.630 seconds on local Python 3.12.11: 16 inherited observation/telemetry tests and 10 focused tests. The latter execute exact pinned native AST code with original file/line locations and synthetic SDK/storage dependencies. They verify missing/unmapped/numeric/unhashable IDs, malformed entries/containers, valid neighboring writes, native worker-thread exception provenance, unchanged native outputs/request arguments/counts, deeper same-KeyError storage faults with partial side effects, transport failure, forged outside-native logs, mismatched saved responses, silent no-ops, corrupted sealed accounting/provenance, and raw response corruption. No model, HTTP, GPU or real database operations were made. Actual pinned server-environment tests and a full-history preflight remain prerequisites for the receiving task's activation.

Ruff reports only the inherited E402 harness-bootstrap import in runner.py; all three test files pass. Existing telemetry tests changed only their synthetic completion evidence; observation tests changed only the renamed/added count expectation. The saved native source was never patched.

## Frozen source hashes

- runner.py: `a47c0f059654f02a00a5f8b9ae60c508aeac79a4bb56aba19e0095f4b2b95d48`
- test_action_drops.py: `332bad3c0a677192072b90c9b8894d48efa964f749c2e5215ef8f9316ef11088`
- test_observations.py: `c38826469725118255d22b97cc6564351796fa7e9988789d34e02ee9fc4c0dd8`
- test_telemetry.py: `83be1d6731e0e16adfd42590e39be97519137523a8fd4cb4e18a040f7ee8225e`
- source_manifest.json: `343b0ef929aaaf3c0b3a790df6c60d98dcc7ec4e1189fb6d18a4d8b5c941f1a8`
- Pinned vendor memory/main.py: `7c43b5defd6767f3a101a9bd5d605662e11f1d637462846b82c2ec5658b63371`

Copied README_V3.md and REVIEW.md are historical documents; this README describes the current candidate. This package's code is frozen pending independent review. No live deployment has occurred.
