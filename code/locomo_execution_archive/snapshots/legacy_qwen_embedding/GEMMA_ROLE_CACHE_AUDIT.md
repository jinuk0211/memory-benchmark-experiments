# Gemma role and cache audit

Read-only audit on 2026-09-08 of `/workspace/generalization_20260908/modern/runs/gemma4_locomo3_fullrole`. **Model-role provenance passes.** A bounded duplicate-cache caveat affects two unused pair diagnostics; it is described below. At the last check, evaluation was running on conv-42 and final507 results were not yet available. This document does not report final scores.

## Actual model and smoke

The actual protocol selects **google/gemma-4-E4B-it**, revision `ee0ef6023621cff504d758262d4e04895a5af4a2`, path `/workspace/.hf_home/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2`.

- Writer preparation and reader evaluation instantiate `transfer_runtime.Runtime`, whose constructor selects `environment['models'][args.model]` for vLLM.
- Likelihood scoring and source control generation instantiate the overridden `transfer_runtime.Scorer` constructor, which selects the same model. The historical 8B constructor in the inherited class is not executed.
- Construction uses the selected Gemma tokenizer for token costs; it does not call another generative model.
- **Qwen/Qwen3-Embedding-0.6B**, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, is intentionally fixed for retrieval across arms/models.

The protocol records torch2.10.0+cu129, vLLM0.19.1, Transformers5.5.3, sentence-transformers5.2.0, NumPy2.2.6 and rank-bm25 0.2.2; `language_model_only=True`, chat-tokenizer default `return_dict=False`, reader budget2048, extra storage2000, answer limit96 and model context8192. All **105 source hashes** match actual remote files.

`runs/smoke_gemma4_1788850468070700123/gpu_smoke.json` records `gpu_smoke_passed` for the same model revision. The synthetic prompt's 135 token IDs matched generation serialization; NLL and source-only synthetic answer generation completed. The smoke records `enable_thinking=False` and explicitly uses no benchmark outcomes.

## Completed writer, scorer and memory provenance

All checks below read existing files and perform CPU reconstruction only; no model was loaded or inference rerun.

- **496 writer requests** resolved through Gemma-specific generation cache keys. Replaying frozen parsing from these cached strings exactly reconstructed all **three seed memories** and the entire **255-window source-generation dump / 180-probe pool**.
- All **6,976 scored source subsets** have their reconstructed Gemma NLL cache key. **6,974 score objects match exactly**; the remaining two are the pair diagnostics described below. All **966 named source-generation controls** match their Gemma cache text and finish reason.
- Each checked writer, likelihood and source-generation prompt was also hashed with historical Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` and Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`: **zero corresponding foreign-model keys** occur in this run's lookup locations.
- `cache`, `cache/generations` and `utility/cache` are nonsymlink paths resolving within this Gemma run. Generation and scorer keys include full selected model path/revision. The protocol is checked for exact equality across stages; the source-history and all-memory digests match `memory_lock.json`, which records `benchmark_questions_used=false`.

## Duplicate diagnostic cache caveat

Two NLL cache keys each correspond to exactly two named subsets: **pair** and **answer_removed**, with identical serialized prompt/answer. The later answer_removed entry matches the final cache value; the pair's saved item score differs by **+0.2516002655029297** and **+0.012847583692444431 nats/token**, respectively. Answer-token lengths agree. Neither key has an empty, full or single counterpart.

The frozen scorer queues duplicate cache misses independently and writes each result to the same key, so a later result can replace an earlier diagnostic cache value. This observed discrepancy is consistent with repeated evaluation under different batch contexts; its numerical cause was not tested on GPU. It prevents claiming exact equality for every NLL item/cache pair.

**All empty/full/single scores and all control generations match their current caches.** Actual parent_single construction reads the saved utility items; admission uses full-minus-empty likelihood and full-generation consistency, while `parent_evidence.construct` explicitly skips pair options. Thus these two duplicate diagnostic differences do not change any consumed score or the current locked parent_single memory. No cache, item, threshold or method was changed to address them. Future replay claims must preserve the distinction between immutable recorded item scores and last-written diagnostic cache values.

## Reader and coverage

At the last check, conv-50 was complete for all three arms: each has **158 rows and 158 unique IDs**, and all **474 row-to-cache checks** resolve to the correct Gemma key and matching prediction. No corresponding Qwen3.5 or old8B reader keys were found. Evaluation had moved to conv-42 seed.

The frozen dataset retains category1-4 counts conv-42 199, conv-47 150 and conv-50 158, totaling **507**. The runner's final paired summaries require exact manifest coverage and reject duplicate, missing or extra IDs before `generation_complete`. Final507 F1 and predeclared377/100/507 summaries remain to be audited when the report exists. Identical prompts may reuse within-model cached answers; cache-hit counts do not reduce evaluation coverage.

## Verified artifact hashes

Paths are relative to `/workspace/generalization_20260908/modern/` unless stated otherwise.

| Artifact | SHA256 |
| --- | --- |
| `runs/gemma4_locomo3_fullrole/protocol.json` | `bf956995dcf7ef05afac315c44e1734e0c2727604c5882954bed7c6dd8d5f74e` |
| `runs/gemma4_locomo3_fullrole/memory_lock.json` | `6ac2da48d6bf0d6179682efcc21a98755f4151f2a1c691cff53b044a767e27da` |
| `runs/smoke_gemma4_1788850468070700123/gpu_smoke.json` | `363b7ea2972ad78f2eda0e96e8b89a6f0b666e98a5cc648a3656ab6a7aa06aa1` |
| `run_transfer.py` | `de4bb873999341828ef492a03e2016e5b8effc7661f3a8be121d86e0c5487b32` |
| `transfer_runtime.py` | `8005f287161eafc50df532dce1c6f98dafb722c6d324194a0667d4502d5ff30f` |
| `source/refine.py` | `71384cac6aaa260d6df4f24f605797a8a886cd51514a83c9c1d041eff99f14fa` |
| `source/run_evidence_utility.py` | `534749083cd534deb00944d893ef9ed43fcb624a14fd5d9f560669d99ad03edd` |

Protocol dataset: `/workspace/generalization_20260908/locomo_transfer/locomo3_unchanged_samples.json`, SHA256 `696e2090c4c419229c7243d088f6a69e8d30507c9313c2dd7d2aa6cffd636c0d`.

Only this local audit document was written. No GPU process, service, experiment artifact or production code was changed.
