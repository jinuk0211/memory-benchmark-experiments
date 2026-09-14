# SimpleMem original bulk-Dialogue recovery v1

This is a local, unlaunched candidate under a separate source/protocol version. Existing smoke failures, pending location-normalization proposals, deployed source and launch plans remain preserved. This candidate does not normalize locations or modify the original memory schema.

## Pinned original code

`upstream/` contains 20 exact files from [aiming-lab/SimpleMem commit db80b6a7c591e0ea730a058e9f5fc4eb06572299](https://github.com/aiming-lab/SimpleMem/tree/db80b6a7c591e0ea730a058e9f5fc4eb06572299): the complete `simplemem/core` subtree, `main.py`, `test_locomo10.py`, the sample configuration, requirements, README and license. `source_manifest.json` records immutable source URLs and byte SHA-256 hashes. In particular:

- `simplemem/core/memory_builder.py`: `ade5d9c252158ede2d0bd4ca3da027e913d6e88dbe616e00f5962178053c9dc7`
- `simplemem/core/models/memory_entry.py`: `487209f5d6981a12568ec3f6f4597d2bd1437a82a0ef9af85c5dc755bc2947c8`

No upstream source file is edited. The original core is imported through a closed namespace package path, avoiding a different installed SimpleMem/router/evolver. The original `SimpleMemSystem` constructor and methods run directly. Only `LoCoMoTester.convert_to_dialogues` is AST-isolated, unchanged, from the benchmark file; the evaluator and its judge are never imported or executed.

## What this corrects

The deployed adapter fed each 4096-character bundle to `add_dialogue(speaker='Benchmark', content=bundle)`. Thus a configured window of 40 meant 40 multi-turn bundles. The pinned benchmark instead creates one `Dialogue` per turn and calls `system.add_dialogues(dialogues)` once, followed by `system.finalize()`.

This runner follows the original bulk path. Canonical session order is represented by ordered integer keys for the original converter's sorted iteration. Each turn retains its original role as `speaker`, its exact content, and the session date string as `timestamp`; original session IDs remain in the source receipt. Nothing is shortened, concatenated, normalized or dropped. The canonical e47becba input audit verifies 53 sessions and 550 distinct turns with exact role/content/date/order equality and zero model calls. The source digest is `a8e29b2cdca6a8ed3293867275b7dcdd4d15f8e545f109d18838dfbb3523115b`.

The old vendor also added provider-tokenizer pre-splitting, context-limit splitting and strict-retry shortcuts. These are not included. The official core's own parser, schema, native three attempts and parallel/sequential recovery remain unchanged. A list-valued `location` still fails the original `Optional[str]` schema; tests verify that no normalization has been added.

The original configuration sample remains window=40, overlap=2, construction workers=16, retrieval workers=8, streaming=True, planning=True, reflection=True, reflection rounds=2, semantic/keyword/structured top-k=25/5/5 and JSON mode=False. The pinned Settings defaults fill only fields absent from the official sample, preventing unrelated environment variables from changing policy. There is no generic temperature override. Every call retains the original caller-specific temperature. The original LLMClient sends **no max_tokens field**; its unused Settings default does not justify introducing a new output cap.

For 550 turns, native bulk processing makes 15 windows: fourteen of size 40 and one of size 18. Its processed_count is 578 because it counts 28 overlap appearances. That counter is not evidence of 578 unique turns or complete successful coverage. `dialogues.json` preserves the unique 550-turn input, while native terminal-failure observations detect dropped windows.

## Model binding and native answer

The common backbone is Qwen/Qwen3.5-9B at the supplied loopback metered OpenAI endpoint, with an EMPTY key and OpenRouter key unset. All three components share the original system's single LLMClient. The runner records each logical call around the original method, including calls made by native worker threads; the proxy journal supplies physical usage, stream finish reasons and SDK/native retry accounting. No provider or paid evaluation API is called.

MiniLM uses the original `EmbeddingModel` → SentenceTransformer standard path with its original `normalize_embeddings=True`. The local snapshot is fixed to `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU FP32, 384 dimensions, native 256-token embedding limit. Loading another model or taking an embedding fallback is rejected. This explicitly replaces the original sample's Qwen embedding backbone to meet the common model condition; it does not replace the retrieval algorithm with an API/common reader.

After native construction/finalize, the runner calls the original `SimpleMemSystem.ask`. Its HybridRetriever planning, three retrieval views, reflection and AnswerGenerator remain intact. The public question argument contains `Question date: ...\nQuestion: ...`; the original prompt templates are unchanged. Gold answers, question types and per-turn `has_answer` never enter the worker's source/query payloads.

## Failures, designed recovery and artifacts

The observer traces original code branches on both the main thread and parallel worker threads; it never changes their return values. Final memory-window loss after native retries, discarded parallel futures, backend failures and the explicit failed-answer sentinel make the history fail. Native optional planning/original-query/incomplete/empty-additional-query recoveries are recorded as warnings. Original raw-text QA recovery remains accepted, and original valid empty-memory/no-relevant-information behavior is retained. No corrective request or extra schema restriction is added by the observer.

The official LanceDB backend uses Tantivy full-text indexing. When stored memory is nonempty, `_fts_initialized` must be true after the original index-creation call; swallowed FTS/backend errors are also observed. This prevents reporting a successful three-view method when the lexical component is absent. Empty valid memory is not failed merely for lacking an index.

Each history runs in a separate process. Canonical all-500 byte hash and IDs, source code hashes, protocol, source and query digests are immutable. Failed attempts remain in place; retries create a new attempt and rebuild that history from the beginning. Successful predictions are reused only after all sealed artifacts verify, including native source/dialogues/memories, the database files, logical calls and prediction. No cache from the previous vendor run is imported.

`status.json`, `predictions.json` and identity fields use the existing native-five `simplemem` schema; a complete all-500 run can pass the existing run-all/export-official gates. A shared smoke subset and later full run have the same full-population protocol. Any failed history stops this candidate runner and leaves an incomplete status. Official judging remains pending.

## Verification and deployment boundary

CPU tests execute the actual pinned converter, MemoryBuilder, streaming LLMClient, SimpleMemSystem.ask and AnswerGenerator with fake transport/storage. They verify 550-turn packing/15 windows, no output-cap override, original location-schema rejection and three retries, parallel failure detection, allowed retrieval/raw-answer recovery, native QA, source-label exclusion, FTS gating and sealed-cache tamper rejection. The real canonical input mapping is independently recorded in `canonical_smoke_input_audit.json`.

```text
python -m unittest discover -s official_recovery/simplemem_native_dialogues_v1 -p test_runner.py -v
```

The runtime still needs a new root-verified receipt binding these source hashes to the actual Qwen/MiniLM weight hashes, FP16 serving, thinking-disabled policy and local services. A configured model name cannot prove the weights behind a live endpoint. No real GPU, LanceDB or Tantivy validation has been run by these CPU tests.

The core imports OpenAI, sentence-transformers, numpy, pydantic, dateparser, LanceDB, pyarrow and Tantivy. Original requirements specify LanceDB 0.25.3, pyarrow 22.0.0, Pydantic 2.12.0, OpenAI 2.3.0 and SentenceTransformers 5.1.1; Tantivy is unpinned. The common experiment runtime may use different compatible library versions and must record them. In that runtime verify the original `create_fts_index(use_tantivy=True, tokenizer_name='en_stem', replace=True)` and retrieval APIs before claiming successful native execution. Do not silently switch to another backend or remove FTS.

An eventual gated smoke command has this form; it has not been executed:

```text
.venv-client/bin/python official_recovery/simplemem_native_dialogues_v1/runner.py --method simplemem --dataset source/MemoryData/datasets/LongMemEval/longmemeval_s_cleaned.json --run-dir runs/simplemem_native_dialogues_v1 --api-base http://127.0.0.1:18083/v1 --model Qwen/Qwen3.5-9B --embedding-model /workspace/.hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --ids-file queue/smoke_ids.json
```

The root's gated full run omits `--ids-file` and retains the same protocol/run directory to reuse a verified successful smoke. The candidate reuses sibling native harness I/O/source-whitelist functions and frozen request metering, all included in protocol hashes. Exclude `__pycache__` from deployment bundles.