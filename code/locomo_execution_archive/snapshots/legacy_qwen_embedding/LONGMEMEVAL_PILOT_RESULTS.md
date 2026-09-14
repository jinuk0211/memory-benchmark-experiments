# Fixed12 LongMemEval pilot results

Both predeclared12-question pilots have completed answer generation. The frozen refined method shows **no answer change over r40 on Qwen3.5-9B**, and a **token-F1 increase confined to one question on Gemma4-E4B-it**. Official LongMemEval judging and full500 results are still required before drawing a data-generalization conclusion.

| Model | Seed F1 ×100 | r40 F1 ×100 | Refined F1 ×100 | Refined − r40 | Refined − Seed |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 56.8983 | 40.6781 | 40.6781 | 0.0000 | −16.2202 |
| Gemma4-E4B-it | 42.0306 | 37.5035 | 43.0590 | +5.5556 | +1.0284 |

Each cell averages the same12 question IDs. Every arm has12 valid, nonempty outputs; all36 outputs per model finished with `stop`. Both completed-result audits recomputed all saved F1 values against canonical answers without discrepancy. These scores are diagnostic token F1, **not official accuracy**.

| Model | Primary comparison wins / losses / ties | Exactly identical r40/refined hypotheses |
|---|---:|---:|
| Qwen3.5-9B | 0 / 0 / 12 | 12 / 12 |
| Gemma4-E4B-it | 1 / 0 / 11 | 11 / 12 |

For Gemma, the only changed r40/refined hypothesis and score is `gpt4_7de946e7`; its change accounts for the whole+5.5556 aggregate difference. For Qwen, all12 answer strings are exactly equal, not merely tied in F1. Preserve the negative Qwen refined−Seed result alongside the positive Gemma diagnostic.

The model-specific execution histories differ: Qwen used writer/reader/scorer context8192 with the original scorer scheduling; Gemma's completed pilot imports its verified writer65536 preparation and recomputes all scoring with chunk512, while reader/scorer context remains8192. Both keep the same frozen algorithm, input subset, embedding, read2048/add2000/answer96 limits and original unelided pilot scoring. Scheduling changes do not imply bitwise numerical equivalence. This table describes those recorded executions and is not a runtime-matched model ranking.

Coverage is single-session-user1, single-session-assistant3, multi-session4 and temporal-reasoning4. There are zero preference, knowledge-update or abstention cases. The subset is a feasibility pilot, not representative coverage of full500, and was not chosen or altered using its answer scores. No new confidence interval or target-based refinement was introduced for this comparison.

Audit sources:

- [Qwen completed-result audit](/D:/MemoryData/generalization_20260908/modern/verified_qwen35_lme12_results/PILOT_RESULTS_AUDIT.md).
- [Gemma completed-result audit](/D:/MemoryData/generalization_20260908/modern/verified_gemma4_lme12_scorer512_results/PILOT_RESULTS_AUDIT.md).
- [Gemma scorer execution receipt](/D:/MemoryData/generalization_20260908/modern/verified_gemma4_lme12_scorer512/RECEIPT.md).

The Qwen pilot result and the Gemma execution recovery do not replace the full500 evaluation. The experiment remains a fixed-method model/data transfer test.