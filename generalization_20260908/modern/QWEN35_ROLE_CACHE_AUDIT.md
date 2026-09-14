# Qwen3.5 role and cache audit

Read-only audit of `/workspace/generalization_20260908/modern/runs/qwen35_locomo3_fullrole` on 2026-09-08. Evaluation was still running; this audit does not claim completed 507-question results or inspect gold failure examples.

## Actual model roles

The remote `protocol.json` selects **Qwen/Qwen3.5-9B**, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, with snapshot path `/workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

- `transfer_runtime.Runtime.__init__` uses `environment['models'][args.model]` for its vLLM engine. `run_transfer.py` uses this runtime for both writer preparation and reader evaluation.
- `transfer_runtime.Scorer.__init__` independently uses the same selected model for its likelihood and control-generation engine. It overrides the inherited constructor; the old Qwen3-8B model literal in `source/run_evidence_utility.py` is not executed.
- The construct stage uses the selected Qwen3.5 tokenizer for token costs and does not load another generative model.
- The embedding model is intentionally fixed at **Qwen/Qwen3-Embedding-0.6B**, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, for all arms and transfer models.

The actual protocol records torch `2.10.0+cu129`, vLLM `0.19.1`, Transformers `5.5.3`, sentence-transformers `5.2.0`, NumPy `2.2.6`, and rank-bm25 `0.2.2`; `language_model_only=True` and the reviewed list-return chat tokenizer wrapper are active.

## Cache isolation and the 77-generation count

`source/refine.py:261` returns an output slot for every requested prompt, reads any existing generation cache, and sends only cache misses to vLLM. Its `GEN ... / N` denominator is the number of missing prompts. The key includes complete model metadata (path and revision), seed, system prompt, full user prompt including retrieved context, output limit, and the nonthinking flag. Method name is intentionally absent: identical prompts within a model/run reuse the same generation.

For **conv-50**, all three completed item files have **158 rows, 158 unique question IDs, and exact expected coverage**. The refined arm has **81 contexts identical to r40**, zero identical to seed. All 81 corresponding predictions match r40. Therefore **77 new generations + 81 cache hits = 158 evaluated questions**.

All 158 rows in each of seed, r40, and refined were independently matched to their reconstructed Qwen3.5 cache key and stored prediction: **474 matching row-to-cache checks, zero prediction mismatches**. None of these prompts had a corresponding old Qwen3-8B key in the run cache, using its historical revision `b968826d9c46dd6066d109eabc6255188de91218`.

The generation cache and utility cache resolve inside this run directory and are not symlinks. Scoring keys also include selected model metadata, source seed, prompt and answer. `protocol.json` is checked for exact equality on each stage/resume; model/config/source changes require a new run. `memory_lock.json` matches the current protocol and all three memory arms, and records `benchmark_questions_used=false`. No old-model cache lookup or observed output leakage was found.

## Coverage enforcement

The unchanged three-conversation dataset contributes category 1-4 question counts **conv-42: 199, conv-47: 150, conv-50: 158**, totaling **507**. `run_transfer.py` retains one row for every selected question, including cache hits. Its final manifest is passed to `transfer_data.paired_summary`, which rejects duplicate IDs and any missing or extra IDs in either baseline or refined arm. Both baseline comparisons must succeed before `generation_complete` is written. Cache-hit counts do not reduce the denominator.

## Verified artifacts and hashes

Remote prefix: `/workspace/generalization_20260908/modern/`.

| File | SHA256 |
| --- | --- |
| `runs/qwen35_locomo3_fullrole/protocol.json` | `54eb61046be0f4a6f67755fbb6250ad3f81dc2edd6ccc396f92c23447988dc26` |
| `run_transfer.py` | `de4bb873999341828ef492a03e2016e5b8effc7661f3a8be121d86e0c5487b32` |
| `transfer_runtime.py` | `8005f287161eafc50df532dce1c6f98dafb722c6d324194a0667d4502d5ff30f` |
| `transfer_data.py` | `f4089630c57a27c739bdaf52cdb3c491fb763dad7f875eba2eaa474109582648` |
| `source/refine.py` | `71384cac6aaa260d6df4f24f605797a8a886cd51514a83c9c1d041eff99f14fa` |
| `source/run_evidence_utility.py` | `534749083cd534deb00944d893ef9ed43fcb624a14fd5d9f560669d99ad03edd` |
| `chat_tokenizer_compat.py` (recorded by protocol) | `b50ed0f335ca2cd7a14957968d5d279fae8556da69a03c9b18f8f83db0108f4c` |

The five runner/runtime/coverage/frozen source hashes were rechecked against actual remote files and all matched the protocol. Dataset path: `/workspace/generalization_20260908/locomo_transfer/locomo3_unchanged_samples.json`; protocol dataset SHA256 `696e2090c4c419229c7243d088f6a69e8d30507c9313c2dd7d2aa6cffd636c0d`.

No experiment, source code, GPU process, or service was changed during this audit.
