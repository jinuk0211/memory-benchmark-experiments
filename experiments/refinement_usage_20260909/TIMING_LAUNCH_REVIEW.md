# Timing-enabled v2 launch review

Status: LOCAL PREVIEW, NOT APPROVED, NOT DEPLOYED. No v2 GPU evaluation or real evaluation-function wall-time measurement exists yet. Source task `01a080b3-dfac-7b41-bffe-5836b1c205fb` owns the scientific queue and final deployment.

## Exact proposed transfer

[V2_TIMED_TRANSFER_CONTENTS.json](V2_TIMED_TRANSFER_CONTENTS.json) lists 13 files, 79,212 bytes, targeting `root@ssh2.vast.ai:23065 /workspace/recursive_minilm_20260909/`. Manifest SHA256: `7b4b9fddf8ec024865d98ae3375e0d4659fcffca563172dba4bbc21ac50452e3`. The manifest itself is a local inventory, not one of the 13 transferred files.

The original eight prepared v2 files remain unchanged on disk. Seven use the original source paths; only the queue's proposed source is a separate timing-enabled copy. Five additions are the four reviewed Python timing/test files and `metrics/timing_source.sha256`. No data, predictions or credentials are included. The original eight-file external transfer was explicitly rejected by automatic approval review; this 13-file proposal expands that pending scope and must not silently inherit a narrower approval.

## Review and deployment requirements

Independent review verified the exact evaluate-only diff, all original eight file hashes, 13 distinct destinations, all current file hashes and byte counts. Git Bash syntax validation passed. Independent Python review and 19 CPU tests passed for the four addon files, including real process spawning.

The queue preserves construction, predecessor decision, GPU-idle, disk-space and existing source-integrity checks. The only evaluation-path addition verifies `metrics/timing_source.sha256` and invokes the explicit timing launcher. A zero-call cached resume is not a measured zero-second evaluation.

Before any authorized start, the owner must compare all deployed file bytes with the proposal inventory and generate `recursive_v2_source.sha256` for all 13 proposed destination paths, including `metrics/timing_source.sha256`. This generated operational receipt is not currently in the transfer inventory. It must pin the NEW queue hash `8f8a3f071b1092eccdde8b17c7499f846f46d859e6c22d1a0fcde182045c9f6c`; the old queue hash would fail correctly. Binding the timing checksum itself prevents an unpinned replacement checksum from accepting changed timing code. Verify this receipt before registration/start. Existing `launch_source.sha256` (125 entries) and `recursive_source.sha256` (six entries) do not overlap any of these 13 paths and should stay byte-identical.

After authorized transfer, run the same two CPU test modules from the remote `metrics` directory with the existing environment and CUDA masked. The Linux tests and source verification are still pending; Windows CPU success does not establish remote execution success. Supervisor remains the persistent-job owner, and the existing queue's gates determine whether GPU work may start. Do not start a second queue or restart the stopped old full500 jobs.

The evaluation command from the project directory is:

```bash
/workspace/generalization_20260908/modern/.venv/bin/python metrics/timed_evaluate.py --runner /workspace/recursive_minilm_20260909/run_recursive_v2.py -- evaluate
```

Use this through the reviewed queue, which performs its preceding gates. This document is not authorization to execute it now.

## Measurement and reporting

The monotonic interval covers the original evaluate_sample function: document/query embedding, retrieval, ranking, packing, generation, existing native telemetry I/O, token accounting, and row assembly/F1. It excludes model initialization, outer item-cache checks, caller output persistence and timing-sidecar writes. Label the result evaluation-function wall time, amortized seconds per question; it is neither full-process E2E nor single-request latency. Do not add native inference time to this enclosing interval.

UUID session/invocation directories preserve starts, completions and errors separately. A completion links the started-file SHA256, question IDs/count and the canonical in-memory returned-row digest. Runner, evaluator file/function, launcher and recorder hashes are recorded. Canonical row hashes differ from raw JSONL file hashes. Logging fails closed: start failure prevents evaluation; completion logging failure can prevent completed rows reaching the caller. Original evaluator exceptions remain unchanged even if error logging also fails.

The native usage report for completed Seed/r40/Refined/v1 is immutable and complete. Its existing native inference time is a separate measurement. The timer cannot reconstruct old unmeasured evaluation-function durations. The separate local aggregate_evaluation_timing.py reporter is implemented and independently reviewed, with 19 focused CPU tests passing. It is not part of this 13-file external transfer. Its population is exactly the supplied final-answer JSONL; canonical benchmark completeness and model-weight provenance remain separate checks. After future results are collected, verify complete canonical coverage and digests, account for failed/repeated attempts, and leave missing/ambiguous averages null. Actual v2 token accounting and final answers will be verified from that run's collected native receipts; the older hardcoded report is not evidence for v2.

## Fresh authoritative state

Read-only SSH check at 2026-09-09T01:58:37Z: old full500 STOPPED, recursive v1 EXITED, recursive Gemma18 EXITED, v2 has no Supervisor process. Both the v2 runner and the timing launcher are absent remotely. There was no upload, queue registration, GPU inference or process mutation in this preparation.

## Local reporting proof

The reporter reconstructs each receipt's returned-row list from original final JSONL rows in receipt question-ID order, preserving every field. It checks actual started-file bytes, terminal linkage, meaningful model/source metadata, timing values, and complete nonoverlapping coverage. Missing, invalid, conflicting or ambiguous attempts withhold the successful attributable mean. Distinct observed successful and failed attempt seconds remain separate; no retry is selected by UUID or file timestamp.

Root ran the actual completed v1 file with all 1,540 questions through the CLI against its absent timing sidecar directory. The expected result was verified: mean seconds is null, not zero. See [TIMING_INTEGRATION_CHECK.json](TIMING_INTEGRATION_CHECK.json) and [untimed v1 report](reports/v1_evaluation_timing_unmeasured/EVALUATION_TIMING.md). This demonstrates missing-data handling; it is not a new timing measurement. The original 13-file proposal, original eight owner files, and completed native usage JSON report hashes remained unchanged.

Local invocation after future evidence collection:

```text
python aggregate_evaluation_timing.py --predictions FINAL_ANSWERS_JSONL --sidecar-root COLLECTED_TIMING_ROOT --method EXACT_METHOD_ID --out NEW_REPORT_DIRECTORY
```

The output directory must be new and outside the sidecar input. Both JSON and Markdown reports are written. LLM/embedding native receipts must still be collected and aggregated separately for each actual new run; do not apply the old completed-run totals to v2.
