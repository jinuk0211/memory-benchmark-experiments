# Source-only feasibility result

Neither candidate is promoted. This is one source history per dataset, with only three LoCoMo QB questions and two LME QB questions. These are not benchmark or official LongMemEval scores.

| Source history | Accepted QA | Proposed QA likelihood delta | Audit F1 Seed → proposal | Pilot final memory | QB count | QB F1 Seed → final |
|---|---:|---:|---:|---|---:|---:|
| LoCoMo conv-49 | 11 | +0.634724 | 0.183333 → 0.086806 | Exact Seed rollback | 3 | 0.666667 → 0.666667 |
| LME d6062bb9 | 10 | +0.004246 | 0.777778 → 0.777778 | Two options adopted within pilot | 2 | 1.000000 → 1.000000 |

LoCoMo improved source-fit likelihood/F1 but harmed the separate source-audit set (likelihood delta −0.282377). The predeclared gate therefore returned exactly Seed. LME changed two of ten QA packed contexts, with a small likelihood gain and unchanged F1. All four LME audit inputs were byte-identical, so their raw +0.000416 difference does not establish evidence improvement.

Raw QB likelihood deltas remain −0.002030 for LoCoMo and −0.002272 for LME. All three LoCoMo QB inputs are exactly identical between Seed/final; those differences cannot show memory harm. For LME, one QB context changed (delta −0.004545) and one stayed identical (delta about +0.000002). F1 did not improve. The source QB evidence does not support promoting either candidate.

The paired receipt compares context, full user prompt and answer. Identical-input NLL variation is consistent with the frozen scorer scheduling duplicate uncached keys together; the exact numerical/kernel cause is not established. Recorded scores, adoption gates and memories are preserved without retrospective correction or rerun.

New native consumption was **0.567989 million LLM tokens** (564,479 prompt + 3,510 completion) and **2.538858 million embedding tokens**. Native LLM inference totaled 62.10 seconds and native embedding inference 95.54 seconds. Source stage wall times were 45.73 seconds for calibration and 210.55 seconds for scoring. These are source construction/calibration/diagnostic costs, not benchmark seconds per question. Imported Seed, source-probe and source-utility preparation costs are additional.

The reviewed accounting tool reports complete evidence with zero gaps, zero failed native calls and zero incomplete usage/timing calls. Both stages and the supervisor launcher exited zero. At 09:05:54 UTC, the supervisor was EXITED, the GPU compute-process query was empty and PIDs 6467/6530/6686 were absent. Teardown warning/error text is preserved in the stage logs; it follows Shutdown complete and did not correspond to a failed stage or missing receipt.

The local archive contains 33 verified files (2.04 MB compressed). `FINAL_SOURCE_PILOT_REPORT.json` records exact archive, protocol, locks, completion and cost-report SHA256 values. `SOURCE_PILOT_DIAGNOSIS.json` contains unchanged raw deltas and per-question input-equality checks; `collected/` holds the source artifacts. No target QA, benchmark judge, retry, tuning, or promotion was performed by this agent.
