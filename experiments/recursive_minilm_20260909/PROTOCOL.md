# Recursive MiniLM experiment — 2026-09-09 KST

User objective: improve our memory method on LoCoMo with Qwen3.5-9B and MiniLM, beat the recorded baselines under disclosed comparable conditions, and check transfer to Gemma4-E4B on a small LongMemEval subset.

## Fixed comparison conditions

- LoCoMo: original10 conversations,1540 category1–4 questions. Canonical file SHA256 cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993. Official scorer SHA256 8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd.
- Writer, reader and source likelihood scorer: Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, FP16, no quantization. Generation temperature0, thinking disabled. Native reader/writer seed20260907 and likelihood scorer POLICY seed20260908 remain distinct.
- Embedding: sentence-transformers/all-MiniLM-L6-v2 revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41, CPUFP32,384dimensions, native256-token limit; normalized vectors and no Qwen query prefix.
- Original method: seed -> r40_fused_four_turn -> s_parent_single_2000. Original source-only construction, RRF60/top120, read2048, added storage2000, answer96, prompt and utility thresholds preserved for the starting comparison. Source NLL uses chunked prefill512 for GPU memory capacity; changed GPU batch numerics are not claimed bitwise equivalent to older schedules.
- Compare baseline F1 using canonical official scoring. Current recorded E-Mem56.96 is the highest completed reference; SimpleMem38.19, LightMem46.60, HiGMem40.33, Mem036.50, LangMem29.93. A-MEM pending. Baselines have documented implementation variants and differing native reader flows; same backbone/embedding alone is not exact end-to-end compute parity.

## Development and transfer boundaries

- Seven LoCoMo conversations (all except conv-42,conv-47,conv-50) are development. Those three are reporting-only for this new search. All ten conversation histories and prior aggregate results have historical exposure; none is claimed pristine/unseen.
- Memory construction and recursive checks receive sessions and source-generated probes only. Benchmark questions, gold answers, category and evidence annotations are absent from the construction process. Benchmark QA can be used to measure/select generic algorithm candidates on the seven development conversations, never to write example-specific memory or rules.
- Each candidate recipe and source hash is committed before its construction/evaluation. Freeze the accepted recipe before Gemma transfer. Keep unsuccessful candidates, fixed denominators, cost and regression reports. A baseline win is an empirical goal, not a guaranteed result.
- LongMemEval: preselect18 questions by type and seeded ID hash,3 per each of six types, using one abstention where available. Exclude previously evaluated12-question clusters. Use complete histories, never truncate/choose answer-containing sessions. Selection may use type/abstention metadata for stratification, but no question text, answers or outcomes.
- The first new Gemma transfer check compares the original method with the frozen accepted improvement on identical18 histories using MiniLM. LongMemEval results are evaluation-only; do not retune a method on these18 and reuse them as untouched generalization evidence.
- Official LongMemEval judge: pinned gpt-4o-2024-08-06 if authorized credentials are available. Diagnostic token F1 is reported separately and never called official accuracy. A small transfer pilot can show a limited observed effect; it cannot prove broad generalization.

## Execution

Independent root: /workspace/recursive_minilm_20260909. Existing full500 and old five-baseline queues are superseded; preserve their files and stop only the full500 experiment at a completed-history checkpoint. Never stop the Vast instance, management services, or the other server's experiments. Never control local PC power.

The first supervised job builds and evaluates the original method under MiniLM. Subsequent recursive stages and Gemma testing will be registered after their concrete implementations are reviewed and frozen. This document is a protocol, not evidence of a running job or successful result.
