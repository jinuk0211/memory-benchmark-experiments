# Qwen3.5-9B + MiniLM LoCoMo recursive comparison

Official category-specific F1,0–100. Same1540 questions, canonical scorer.

| Method | All1540 | Multi-hop | Temporal | Open-domain | Single-hop |
|---|---:|---:|---:|---:|---:|
| seed | 56.260 | 46.080 | 41.762 | 13.565 | 70.080 |
| r40_fused_four_turn | 55.522 | 39.684 | 49.703 | 14.092 | 67.784 |
| s_parent_single_2000 | 55.528 | 38.129 | 50.644 | 15.576 | 67.787 |
| recursive_source_rehearsal_v1 | 55.856 | 38.758 | 50.483 | 16.014 | 68.188 |

Gemma LongMemEval transfer remains pending. Prior LoCoMo history exposure is disclosed in PROTOCOL.md.
Recorded external baseline references: E-Mem56.96, LightMem46.60, HiGMem40.33, SimpleMem38.19, Mem036.50, LangMem29.93. Their native reader flows and implementation variants differ.
