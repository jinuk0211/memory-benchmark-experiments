# Historical refinement count — 2026-09-08

The verified original development ledger contains **126 evaluated candidate/control configurations**, after removing copied history entries, the initial baseline, and two explicit replay-only configurations. This is a count of attempted comparisons, not 126 accepted improvements or 126 independent algorithmic ideas. **Seven candidates were recorded as accepted improvements** by the historical dev-selection logic. Since the current generalization phase froze `s_parent_single_2000`, there have been **zero additional algorithm refinements based on its target results**.

The authoritative read-only scope is `/workspace/locomo-refinement/runs`. All 18 currently present immediate-child `history.json` files were enumerated and examined. They contain 215 recorded rows. Deduplicating each named configuration by its recipe/parent identity and complete dev-summary digest yields 129 configurations; repeated names have zero identity or score collisions. The 86 excess rows are overlapping copied histories in pilot100/v2/v3 and continuous_v1/v2/v3; they are not additional candidate evaluations for this count.

Every one of the 129 distinct configurations has an existing named `_dev_items.json` result file with exactly 70 rows, 70 unique question IDs, finite F1 values in [0,1], and a mean matching its ledger F1 within 1e-12. No missing or mismatching result file was found. This verifies recorded completed dev comparisons, not the number of actual GPU launches, cache misses, failed attempts, or scientific improvements.

The exclusions are `r00_session10` (initial baseline), `r51_replay_reference` (empty operations; parent `r12_filter_current_best`), and `r52_replay_seed` (empty operations; parent `seed`). The remaining 126 include controls and ablations such as raw/copy variants. Those remain evaluated configurations even when they yield the same memory or score as another arm. Their names do not prove distinct semantic algorithms.

| Historical group | Candidate/control configurations after exclusions |
|---|---:|
| Initial pilot (r01–r04) | 4 |
| Continuous series, deduplicated | 59 |
| Budgeted evidence | 11 |
| Parent evidence | 14 |
| Retrieval contract | 1 |
| Cross-view selection | 1 |
| Identity consolidation | 9 |
| Contrastive events | 4 |
| Source index | 3 |
| Transfer index | 4 |
| Provenance payload | 4 |
| Coverage routes | 4 |
| Identity routes | 6 |
| Read budget | 2 |
| **Total** | **126** |

The seven historically accepted candidates are `r02_audit`, `r03_dialogue_residual`, `r05_calendar_anchor`, `r06_calendar_month`, `r12_filter_current_best`, `r40_fused_four_turn`, and `s_parent_single_2000`. The baseline also has `accepted=true`, so merely counting unique accepted rows would incorrectly give eight improvements. Acceptance was development-set selection; it is not statistical significance or established generalization.

The largest evaluated numeric recipe prefix is **r68**, specifically `r68_fused_key_both_payload`. `r40` was a selected winner, not the final attempted recipe. Numbering has gaps, lettered r50 variants, m-prefixed interventions, and later b/s/t/v/w/x/y/z/aa/ab/ac/ad families. Therefore neither 40 nor 68 is the total trial count. The ledger's execution order also differs from numeric order.

Two additional `source_history.json` files describe nested optimization within the retrieval-contract and cross-view configurations already counted above. Retrieval contract records three loop decisions (two accepted changes, then a stop); cross-view records two decisions (one accepted change, then a stop). These five decisions are not five additional dev-evaluated methods and are not added to 126. This audit does not count every internal proposal, failed/uncommitted attempt, GPU rerun, or benchmark audit/generalization evaluation.

The current writer-capacity change to 65,536 and scorer token-chunk change to 512 are two infrastructure recovery changes, not methodological refinements. They preserve the frozen algorithm but are separately recorded execution variants. The historical accidental Qwen3-8B development record is not relabeled as Qwen3.5-9B validation.

Metadata-only evidence retained locally:

- `refinement_history_count_metadata.json`: per-ledger SHA256, configuration names, duplicate-identity checks and counts.
- `refinement_history_results_audit.json`: per-result path/size/SHA256 and count/score validation, exact replay-plan metadata, source-loop summaries, and complete history-file inventory.

No source conversation, question, gold-answer, or generated-answer bodies were copied into these audit artifacts. No remote files, services, model settings, or GPU jobs were changed.
Metadata SHA256: 285c88266820e603a498ae5c5edb50b08219d887f5e4f68b44fa92e1ed905900
Result audit SHA256: 6fd0fbd9ae55060cf23e60856783b3b5a22249eab609a134f0fd7f765979f9fe
