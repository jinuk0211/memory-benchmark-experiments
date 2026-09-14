# LoCoMo comparison

Canonical LoCoMo 10 conversations, all 1,540 category 1-4 questions; official F1 multiplied by 100. Results are measured scores, not paper-reported numbers.

| Method | All 1,540 | Multi-hop 282 | Temporal 321 | Open-domain 96 | Single-hop 841 |
|---|---:|---:|---:|---:|---:|
| E-Mem | 56.96 | 44.87 | 54.67 | 30.64 | 64.89 |
| seed | 56.26 | 46.08 | 41.76 | 13.57 | 70.08 |
| Recursive source rehearsal v1 | 55.86 | 38.76 | 50.48 | 16.01 | 68.19 |
| s_parent_single_2000 | 55.53 | 38.13 | 50.64 | 15.58 | 67.79 |
| r40_fused_four_turn | 55.52 | 39.68 | 49.70 | 14.09 | 67.78 |
| LightMem (direct-store variant) | 46.60 | 42.01 | 34.36 | 24.31 | 55.35 |
| HiGMem (profile/link disabled) | 40.33 | 27.95 | 26.40 | 12.70 | 52.96 |
| SimpleMem | 38.19 | 29.57 | 24.87 | 23.33 | 47.86 |
| A-MEM (original failure policy) | 36.85 | 28.98 | 29.09 | 17.43 | 44.66 |
| Mem0 | 36.50 | 37.58 | 13.29 | 24.53 | 46.36 |
| LangMem | 29.93 | 28.37 | 27.27 | 21.65 | 32.42 |

The completed Refined method (s_parent_single_2000) scores 55.5278, versus r40 55.5224 (+0.0054 points) and E-Mem 56.9567 (-1.4289 points). Recursive v1 is complete at 55.8560: +0.3282 points versus Refined, -0.4037 versus Seed, and -1.1007 versus E-Mem. The predefined transfer gate failed; Gemma inference did not start. Source-round acceptance and CPU representation checks do not establish benchmark improvement.

Qwen3.5-9B and MiniLM are fixed for this run. Existing external baseline implementations retain their native reader flows; these rows do not establish equal inference budgets or exact paper reproduction. LightMem and HiGMem are implementation variants and remain excluded from the planned LongMemEval baseline comparison. E-Mem and SimpleMem prediction coverage is complete; their usage accounting is incomplete.

All ten LoCoMo conversations were historically exposed during development. The development subset excludes conv-42, conv-47, and conv-50; those three are reporting conversations, not a pristine holdout. Gemma4-E4B LongMemEval transfer outcomes have not been evaluated.

Recursive v1 compared with Refined:

| Population | Questions | F1 change (points) | Wins | Losses | Ties | Paired cluster bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|---|
| all1540 | 1540 | +0.3282 | 60 | 47 | 1433 | [+0.0271, +0.7727] |
| development7 | 1033 | +0.4273 | 49 | 35 | 949 | [-0.0056, +1.0518] |
| reporting3_previously_exposed | 507 | +0.1263 | 11 | 12 | 484 | [+0.0000, +0.3217] |

Intervals are exploratory: only10conversation clusters overall, historical LoCoMo exposure and previous development trials. These results do not establish out-of-dataset generalization. Source-round audit:7accepted changes in6conversations, including one accepted round2; no cue rewrites were selected.

Source reports:

- [Complete recursive v1 official report](receipts/recursive_v1/OFFICIAL_COMPARISON.json)
- [Actual prediction hashes and Gemma non-execution](receipts/recursive_v1/FINAL_INTEGRITY.json)
- [Current three-method official report](receipts/qwen35_baseline/BASELINE_COMPARISON.json)
- E-Mem: D:/MemoryData/results/locomo/emem/emem-final-report-50156979-r11.json
- LightMem (direct-store variant): D:/MemoryData/results/locomo/lightmem/lightmem-final-report-20260907.json
- HiGMem (profile/link disabled): D:/MemoryData/results/locomo/higmem/higmem-final-report-20260907.json
- SimpleMem: D:/MemoryData/simplemem-final-report-20260908-r25.json
- Mem0: D:/MemoryData/results/locomo/mem0/mem0-final-report-50156979-r5.json
- LangMem: D:/MemoryData/results/locomo/langmem/langmem-final-report-50156979-r1.json

A-MEM completion update: all1,540predictions and usage accounting are complete, F1=36.8459. The original metadata/evolution prompts, schemas,1,000-token output cap and JSON-failure skip policy were restored;3,569evolution updates were skipped after JSON failures. The original report retains complete=false for an external backup boundary, while benchmark_complete=true; a separate verified PC receipt confirms backup completion. This is an instrumented native-failure-policy comparison, not a claim of an unmodified upstream implementation or successful evolution on every call. `A-MEM final summary` (historical source: `amem_final_results_r26/results_summary.md`) · [Verified reference](receipts/A_MEM_COMPLETED_REFERENCE.json).
