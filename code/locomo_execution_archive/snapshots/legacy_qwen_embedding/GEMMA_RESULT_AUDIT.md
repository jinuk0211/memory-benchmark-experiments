# Gemma LoCoMo result audit

Independent read-only audit on 2026-09-08 of `/workspace/generalization_20260908/modern/runs/gemma4_locomo3_fullrole`. **PASS:** final answer coverage, F1, declared populations, paired summaries, source hashes and memory locks are consistent. Generation completed at 2026-09-08 07:25 UTC. No target-outcome tuning or failure-example mining was performed.

## Integrity and independent reconstruction

- Every arm has **507 rows, 507 unique canonical question IDs, status ok for all507, and zero empty predictions**. Conversation counts are conv-42 199, conv-47 150, conv-50 158. All rows agree with the original question, gold, category, conversation and method; hypothesis and prediction fields match.
- Recomputed all **1,521 stored F1 scores** with frozen `source/refine.py::f1` against canonical LoCoMo gold: **zero mismatches** (absolute tolerance 1e-12).
- Rebuilt all507, primary377 and secondary100 IDs directly from canonical `data/locomo10.json`, original `runs/pilot100_v3/manifest.json` and `research_data/question_audit100_manifest.json`. Primary377 is the complement of recorded pilot/audit questions within all507; secondary100 is the original audit population. Every ID matches the predeclared population file; the remaining30 are earlier pilot questions.
- Actual model is **google/gemma-4-E4B-it**, revision `ee0ef6023621cff504d758262d4e04895a5af4a2`. All **105 source hashes** match remote files. The protocol, source-history and all-memory digests match `memory_lock.json`, whose `benchmark_questions_used` is false. Run status is `generation_complete`, questions507.
- Independently recomputed the six comparisons without calling the reporting or paired-summary functions: question means, equal-category macro means, paired deltas, win/loss/tie counts and 95% percentile intervals from 10,000 conversation-cluster bootstrap samples, seed20260908. All **42 numerical/summary checks** match the saved report.

## Final scores

Main metric: **question-mean F1**, 0-100, with equal weight per question. Equal-category macro F1 is supplemental and does not replace the predeclared main metric.

| Frozen population | n | Seed | r40 | Refined |
| --- | ---: | ---: | ---: | ---: |
| Primary fresh questions | 377 | 46.221787 | 50.558209 | 50.741006 |
| Secondary prior audit | 100 | 49.190897 | 52.337933 | 50.864701 |
| All generated questions | 507 | 47.182449 | 51.790694 | 51.672307 |

Differences and 95% CIs are in percentage points.

| Population | Refined - r40 [95% CI] | Refined - seed [95% CI] |
| --- | --- | --- |
| Primary377 | +0.182797 [-2.196818, +1.498880] | +4.519219 [+3.307449, +5.197618] |
| Secondary100 | -1.473232 [-2.626263, 0.000000] | +1.673804 [-1.971090, +7.565853] |
| All507 | -0.118387 [-2.144841, +0.736058] | +4.489857 [+2.416071, +5.760654] |

Primary refined-versus-r40 outcomes: **18 wins, 15 losses, 344 ties**. Per-conversation primary effects are:

| Conversation | n | Refined - r40 | Refined - seed |
| --- | ---: | ---: | ---: |
| conv-42 | 155 | +0.849051 pp | +5.197618 pp |
| conv-47 | 107 | -2.196818 pp | +4.838862 pp |
| conv-50 | 115 | +1.498880 pp | +3.307449 pp |

Supplemental equal-category macro F1, in seed/r40/refined order: primary **34.336911 / 37.682041 / 39.631255**; secondary **33.343190 / 35.129865 / 33.509099**; all507 **34.103529 / 37.526785 / 38.746961**.

The full refined bundle exceeds seed on the primary and all507 populations. Its incremental effect over r40 is small and uncertain on primary377, negative on audit100 and slightly negative on all507. These results therefore do **not** establish a stable incremental parent-single benefit across the declared populations. They must not be replaced by the more favorable supplemental category-macro result. All three histories were previously exposed, and three clusters provide limited evidence about unseen conversations. LongMemEval remains necessary for the data-generalization claim. Added storage and question-key augmentation remain confounded in the bundled comparison.

## Separate pair-cache diagnostic caveat

The earlier role audit found two duplicate source-likelihood keys shared by pair and answer_removed diagnostics. The saved pair scores differ from the later cache writes by 0.2516002655 and 0.0128475837 nats/token. Neither key has an empty/full/single counterpart; all consumed empty/full/single scores and all source-control generation texts matched their caches. The current locked parent_single memory is unaffected, and no artifact was changed.

This caveat concerns source diagnostic replay. It is **separate from final answer scoring**: all1,521 final benchmark F1 values were independently regenerated and matched. See [GEMMA_ROLE_CACHE_AUDIT.md](../../../../generalization_20260908/modern/GEMMA_ROLE_CACHE_AUDIT.md) for the bounded provenance finding.

## Artifact hashes

Remote run prefix: `/workspace/generalization_20260908/modern/runs/gemma4_locomo3_fullrole/`.

| Artifact | SHA256 |
| --- | --- |
| `predeclared_locomo_results.json` | `f549e95629db4222f078a5f75fac3594c064192a537d36068fe375bd8041c133` |
| `protocol.json` | `bf956995dcf7ef05afac315c44e1734e0c2727604c5882954bed7c6dd8d5f74e` |
| `memory_lock.json` | `6ac2da48d6bf0d6179682efcc21a98755f4151f2a1c691cff53b044a767e27da` |
| `seed.jsonl` | `7bc62f276053f639e7bb784d2fe0f36f7dcdb0a505aa7c3df45a6ab2f8e0bc18` |
| `r40_fused_four_turn.jsonl` | `384257b3b2aedf2074e65301c5765886817329b4e4a04a2d69200fda886cf16f` |
| `s_parent_single_2000.jsonl` | `a4fffefe0f12c17f0963423252313aa1d4f3c236d67ef5c584f59518985e9542` |
| `locomo_transfer/evaluation_populations.json` | `615b81dad77a64dafcea718f60472bc4b0a526d0f19cafe290c520fb0c17d631` |
| `locomo_transfer/locomo3_unchanged_samples.json` | `696e2090c4c419229c7243d088f6a69e8d30507c9313c2dd7d2aa6cffd636c0d` |

Original input hashes were rechecked: LoCoMo10 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`; pilot100 manifest `12e4e8970b324a1ce25163192fab336e7fe21fa97a9f8e1dca70beb4a11f537e`; audit100 manifest `5c928b820cd7bf192ec8c094405865ee0d06239bff380893f3849a3ee91dd989`.

Only this local audit document was written. No GPU process, service, source code, benchmark artifact or method was modified.
