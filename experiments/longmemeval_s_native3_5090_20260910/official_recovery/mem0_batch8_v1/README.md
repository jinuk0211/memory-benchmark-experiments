# Mem0-batch8 v1

This is an explicitly batched comparison variant of `mem0_native_pairs_v4`, not an unchanged reproduction of its pair-at-a-time input condition. Each call to official `Memory.add(..., infer=True)` receives up to eight consecutive original dialogue pairs (at most 16 turns) from one original session. Remaining pairs and a final singleton are retained. Original empty and whitespace-only string turns are retained at their original positions; non-string turns and unsupported roles are rejected. No truncation, summarization, cross-session merge, or role/content rewriting is performed.

The baseline runner SHA-256 is `a47c0f059654f02a00a5f8b9ae60c508aeac79a4bb56aba19e0095f4b2b95d48`. The copied official vendor, wheel, source manifest and dependency requirements are byte-identical to that baseline. Official fact extraction, update behavior, model settings, search, and benchmark QA integration are unchanged. Session ID, date and UTC timestamp remain metadata; this version adds batch index and original half-open turn range to that metadata. The implementation ID is `mem0_batch8_v1`, and the display name is `Mem0-batch8`.

## Canonical input counts

The full 500-question source SHA-256 is `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.

| Original source | Count |
| --- | ---: |
| Sessions | 23,867 |
| Turns | 246,750 |
| Session-local pairs, including final singletons | 124,345 |
| Batch8 add calls | 24,365 |
| Odd-length sessions | 1,940 |
| Original empty-string turns, preserved verbatim | 12 |
| Sessions ending in a batch shorter than 16 turns | 23,732 |

The nominal two-chat-call budget is 48,730 rather than 248,690, about 5.103 times fewer calls. This is a call-count comparison, not a wall-time or accuracy claim. Actual chat calls are observed separately for each batch; native fallback behavior is retained. Pair counts use a separate ceiling for each session. For example, question `e47becba` has 53 sessions, 550 turns, 277 session-local pairs, and 53 batch8 adds.

## Fresh run and optional canary

Use the existing pinned Mem0 client environment and a fresh directory, for example `runs/mem0_batch8_v1`. Do not copy earlier native or other-variant results or databases into it. The unchanged canonical dataset and complete original question histories are always required. Local endpoints and tokenizer below are deployment placeholders to replace with the actual pinned paths:

```sh
python official_recovery/mem0_batch8_v1/runner.py \
  --dataset longmemeval_s_cleaned.json \
  --run-dir runs/mem0_batch8_v1 \
  --source-root source/MemoryData \
  --api-base http://127.0.0.1:LLM_PORT/v1 \
  --embedding-api-base http://127.0.0.1:EMBED_PORT/v1 \
  --model Qwen/Qwen3.5-9B \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-dims 384 \
  --tokenizer /path/to/pinned/tokenizer \
  --workers 1
```

An optional `--ids-file canary.json` containing `["852ce960"]` selects that one full history for dispatch. The protocol still includes all 500 canonical question IDs; source, query, and source-code identities remain unchanged. After verified completion, rerun the same command without `--ids-file` to reuse the canary and dispatch the remaining 499. Changing `--workers` also leaves scientific identity unchanged. The default is one worker. The bounded pool accepts one through four independent question processes sharing the supplied endpoint; the deployment owner chooses a suitable concurrency for the actual GPU.

One process-held lock owns the run directory through preflight, dispatch, child draining and publication. The first worker or launch failure stops further dispatch, while already started children finish and their successful artifacts are verified and sealed. A parent-side error similarly waits for existing children and attempts to preserve successful completed work before re-raising the original error. An incomplete or failed attempt is preserved. Restart allocates a new whole-question attempt; the batch journal is evidence, not an in-place partial-database restart mechanism.

## Completion and cache proof

`batch_calls.jsonl` contains exactly one newline-terminated record per successful add, in original source order. Each record binds the source hash, batch index, an immutable copy of session/range metadata, message hash, turn count, pair count, actual observed chat-call count, and elapsed seconds. Native mutation of the metadata argument cannot alter this input snapshot. Native empty results and authenticated nonfatal fallbacks remain valid when the original evidence verifies.

`verify_batch_coverage(attempt, identity)` reconstructs batches from `source.json` and checks exact coverage, order, range, hashes, integer accounting, observed response attribution, final progress and build totals. `validated_attempt_files(attempt, identity)` also verifies protocol/source/worker identity, native observations, telemetry opt-out, prediction identity, and all required closed-worker artifacts. Both `finalize_attempt(attempt, identity)` and `verified_prediction(history, identity)` use this semantic verifier, so a rehashed receipt cannot legitimize a missing, duplicated or malformed batch journal. Old implementation identities are rejected.

Only a parent that has observed worker exit calls `finalize_attempt`; the receipt inventories all closed SQLite, Qdrant, response, journal and other attempt files. `verified_prediction` returns a generated prediction or `None` for unsealed attempts, and raises for incompatible or corrupt completed evidence. An external collector should reconstruct the expected canonical source/query identity and call this function; it must not infer completion merely from file counts.

Protocol construction is exposed as `protocol_for(runtime, population_ids, source_hashes(source_root))`. Runtime fields are `method`, `source_root`, `api_base`, `embedding_api_base`, `model`, `embedding_model`, `embedding_dims`, and `tokenizer`. Dataset, source-file hashes and all 500 ordered IDs are bound in the protocol. Scheduling fields, selected IDs and worker count are excluded.

The parent writes canonical-order `predictions.json`, `failures.json`, `status.json` and `current.json`. Status always includes `planned=500`, `population=500`, selected count and verified generated count. A successful canary reports `subset_generation_complete`; only 500 verified predictions report `generation_complete`. `current.json` lists active question IDs, attempt paths and child PIDs. This runner performs no server Stop action.

## CPU verification

```sh
python -B -m unittest discover -s official_recovery/mem0_batch8_v1 -p 'test_*.py' -v
```

Tests exercise exact source preservation, final singletons, immutable metadata, failed adds, malformed or resealed journal rejection in both completion and cache paths, native fallback/transport/storage regressions, unchanged native configuration/QA code, full canonical source counts, pool bounds, failure draining, successful-result preservation on parent errors, canary reuse and exclusive run ownership. All model and child execution in scheduling tests is synthetic. The canonical-data test skips only if neither documented canonical dataset path is present.
