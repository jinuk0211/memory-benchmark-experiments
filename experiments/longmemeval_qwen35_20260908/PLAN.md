# Qwen3.5-9B LongMemEval baseline comparison

User objective (2026-09-08): complete the currently running Qwen3.5-9B LongMemEval-S cleaned 500-question Seed/r40/Refined answer generation; hold subsequent Gemma execution; then evaluate the existing baselines on the same 500 questions. Preserve all existing results and memories.

## Scope and ownership

- Current Qwen run: /workspace/generalization_20260908/full_transfer/runs/qwen35_lme500_fullrole on ssh2.vast.ai:23065. The existing Test refinement generalization task owns its queue and the Gemma hold. A follow-up carrying this explicit override was sent on 2026-09-08; Gemma hold verified at 2026-09-08T13:24:39Z in full_transfer/recovery_lineage/gemma_hold_20260908/HOLD_APPLIED_RECEIPT.json. The live Qwen processes and all110 source hashes remain unchanged. The current parsed queue intentionally exits75 at the first Gemma smoke; this is a user hold and must not be bypassed by failure recovery.
- This comparison task owns baseline preparation, its new isolated outputs, and comparison reporting. Do not modify the other server's CertMem/A-MEM experiments or share their writable memory stores.
- Required external methods after the explicit 2026-09-08 user correction: E-Mem, SimpleMem, Mem0, LangMem, A-MEM. The user excluded LightMem direct and HiGMem no-profile/no-link before any baseline execution. Seed/r40/Refined remain the internal comparison arms. No further method or question may be silently omitted after observing scores or operational difficulty. scope_five.json records the execution scope and existing strict-repair conditions.
- Execution order after the Qwen completion gate: the first frozen question for E-Mem, SimpleMem, Mem0, LangMem, A-MEM as full-history integration checks, then each method's remaining499. Each method's full500 results are required; a failed method remains incomplete and keeps its error artifacts.

## Common evaluation contract

- Dataset: canonical LongMemEval-S cleaned at D:/MemoryData/MemoryData/datasets/LongMemEval/longmemeval_s_cleaned.json; SHA256 d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442.
- Exactly the same 500 question IDs as the running Qwen protocol, including 30 abstention cases. Paired reporting clusters by the 471 base question IDs.
- Every original session and turn is available to memory construction. Only source role/content, session IDs and session dates are construction inputs. Questions, answers, question types, has_answer and answer_session_ids are excluded; question and question date become available only during retrieval/answering.
- Generative model in all roles: Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, nonquantized BF16 to match the current Qwen transfer run. Record all method-specific internal generation settings and verify the actual served model and dtype.
- Embedding: Qwen/Qwen3-Embedding-0.6B revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3, 1024 dimensions. Record actual query/document preprocessing per method; keeping the model common does not imply identical retrieval algorithms.
- Controlled final reader: exact system prompt from the current Qwen protocol, exact user envelope `Conversation memory: ... / Question date: ... / Question: ... / Answer:`, temperature0, thinking disabled, max96 output tokens.
- Final memory text budget: 2048 tokens under the pinned Qwen tokenizer, including separators, whole-unit greedy packing that skips oversized units and continues. This is a controlled-reader configuration, not a claim to reproduce every upstream default. Preserve both retrieved candidates and the actual packed context.
- Preserve each baseline's memory construction and native retrieval logic. E-Mem's internal block reasoning and aggregation remain part of E-Mem and must be separately metered. A common final reader budget is not a claim of equal total inference cost.
- Keep raw source ingestion complete. Do not repurpose input_length_limit=2048: that controls broader requests and can truncate source material. The final-reader cap needs an explicit path separate from native construction/internal generation limits.
- Record total stored text/index tokens, all generation/embedding requests, prompt/output tokens, elapsed time and available GPU measurements. Do not sum duplicate counters or treat missing usage as zero.
- Primary outcome: accuracy using the same official LongMemEval rubric and pinned gpt-4o-2024-08-06 judge for all arms. API configuration is currently unavailable; no judge call has been made. Diagnostic token F1 is reported separately, never substituted for official accuracy.
- Report all500, six question types, abstention subset, and paired differences with uncertainty. Missing/failed answers stay visible; no success-only denominator or score-driven retries.

## Verified integration findings and remaining work

- Existing MemoryData/scripts/run_six_baselines.py supports the cleaned LongMemEval loader and six methods. HiGMem has a separate runner and needs its LongMemEval source adapter.
- Existing generic settings are not directly comparable: generation_max_length256, several native prompt paths, tiktoken fallback, and different memory-fitting logic. Mem0 and A-MEM use their own answer paths; the other four methods use the common helper. Controlled final-reader integration must cover all seven methods and verify actual outgoing requests.
- Existing loader retains complete turns and dates while excluding answer-bearing annotation fields. Its chunk_size is a character target and it does not split a single long turn; it is not a model token bound.
- LightMem's direct variant and A-MEM's original fallback policy must be pinned to the reviewed working implementation before source freeze. Do not import experimental repair/reconstruction behavior silently.
- Preparation status: no baseline GPU job or external judge has been launched; no existing source/results/memory was modified by this task.

## Execution gates

1. Verify the current Qwen process remains live and the applied control actually prevents a Gemma launch.
2. Freeze independent baseline source/config snapshots and package versions. Build the explicit controlled-reader path and audit source/QA separation, exact500 coverage and metering.
3. Wait for the current Qwen run to produce all three500-answer files and a valid completion state. Verify hashes and preserve the run. No replacement inference or duplicate Qwen job.
4. Start baseline services only after the current GPU work has exited and the GPU is idle. Use separate state/output directories and a durable process supervisor.
5. Verify small predeclared integration cases without selecting methods/settings from answer scores; then run all500 for all seven methods. Preserve failures and recover their operational causes without target-error tuning.
6. Complete the common official judging and paired comparison. The goal remains active until the requested runs and verified results exist.


## Candidate preparation checkpoint

Candidate v1 was explicitly approved, transferred and SHA256-verified in /workspace/lme_baselines_20260908. Its archived seven-method scope is historical and must not be launched; the user subsequently restricted execution to five methods. A reviewed five-method harness update is required. No baseline is queued or running yet. Source-only inputs have been materialized for all500. Native build/retrieval attempts are preserved on failure; only a hash-locked final-reader input is retried within the same attempt.
