# Qwen3.5 LoCoMo result audit

Independent read-only audit on 2026-09-08 of `/workspace/generalization_20260908/modern/runs/qwen35_locomo3_fullrole`. **PASS:** no coverage, score, population, protocol, source-hash or memory-lock inconsistency found. Root's `QWEN35_RESULTS.md` agrees with the audited figures and scope.

## Integrity and independent reconstruction

- All three arms contain exactly **507 rows and 507 unique canonical question IDs**: conv-42 199, conv-47 150, conv-50 158. All 1,521 rows have the correct method, conversation, category, original question and original gold. Hypothesis and prediction fields agree. There are zero empty predictions.
- Recomputed every stored score with frozen `source/refine.py::f1` against the canonical dataset: **1,521 checks, zero mismatches** (absolute tolerance 1e-12). No failure examples were selected or inspected for tuning.
- Rebuilt populations directly from the original dataset, `runs/pilot100_v3/manifest.json`, and `research_data/question_audit100_manifest.json`: all507; primary377 = all507 minus recorded pilot and audit IDs; secondary100 = original audit IDs. All reconstructed IDs exactly match the frozen population artifact. Primary377 and audit100 are disjoint; the remaining30 are prior pilot questions.
- All **105 protocol source hashes** match actual remote files. Protocol model is **Qwen/Qwen3.5-9B**, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Protocol, source-history and memory digests match `memory_lock.json`, which records `benchmark_questions_used=false`. Status is `generation_complete`, questions507.
- Independently recomputed means, category macro means, paired differences, win/loss/tie counts and CIs without calling `report_locomo` or `paired_summary`. Used the declared 10,000 conversation-cluster bootstrap resamples, seed20260908, percentile interpolation. All **42 numerical/summary checks across six comparisons** match the report.

## Results

The main metric is the **question-mean F1** on a 0-100 scale; each question has equal weight. It is distinct from the supplemental equal-category macro mean.

| Frozen population | n | Seed | r40 | Refined |
| --- | ---: | ---: | ---: | ---: |
| Primary fresh questions | 377 | 54.094655 | 55.687063 | 56.305168 |
| Secondary prior audit | 100 | 54.314622 | 52.978814 | 53.287775 |
| All generated questions | 507 | 54.834016 | 56.070440 | 56.578669 |

Differences and 95% CIs are in percentage points.

| Population | Refined - r40 [95% CI] | Refined - seed [95% CI] |
| --- | --- | --- |
| Primary377 | +0.618105 [-0.090814, +1.664471] | +2.210514 [+0.554301, +3.682036] |
| Secondary100 | +0.308962 [-0.372960, +1.027822] | -1.026847 [-6.917880, +3.035113] |
| All507 | +0.508228 [-0.055480, +1.133585] | +1.744653 [+0.771088, +3.161145] |

Primary refined-versus-r40 outcomes: **13 wins, 7 losses, 357 ties**. Its per-conversation differences are conv-42 **-0.090814 pp** (n155), conv-47 **+0.520446 pp** (n107), conv-50 **+1.664471 pp** (n115). Thus the small positive aggregate is not uniform across all three conversations.

Supplemental equal-category macro F1, in seed/r40/refined order: primary **41.227249 / 43.188148 / 43.547126**; secondary **39.468512 / 35.996529 / 36.536126**; all507 **41.331497 / 42.328779 / 42.683977**.

The primary incremental effect over r40 is positive but its CI includes zero. The audit100 refined-versus-seed result is negative. All populations use the same three previously exposed conversation histories; fresh377 means fresh recorded evaluation questions, not unseen histories. With only three clusters, these bootstrap intervals provide limited evidence about broader populations. This audit establishes result integrity and intended-model execution; cross-model/data generalization still requires Gemma and LongMemEval results. Added storage and question-key augmentation remain confounded in the bundle comparison, as documented in `METHOD_CLAIM_AUDIT.md`.

## Artifact hashes

Remote run prefix: `/workspace/generalization_20260908/modern/runs/qwen35_locomo3_fullrole/`.

| Artifact | SHA256 |
| --- | --- |
| `predeclared_locomo_results.json` | `8f256ce83544bdcc6237e55516260f8d62b41102852ec11de8f767a900952041` |
| `protocol.json` | `54eb61046be0f4a6f67755fbb6250ad3f81dc2edd6ccc396f92c23447988dc26` |
| `seed.jsonl` | `5f96ce7311092acd594ed5ecc5fbecc7caac2d5496bc19dfe7d3cf58259f32f8` |
| `r40_fused_four_turn.jsonl` | `f9f62a799e2bb744fa89107e7348f39a6561051a71ee5ca8d20bf2dd4955af8d` |
| `s_parent_single_2000.jsonl` | `3d38348a74ca5692f5e9ab357a3f0fbd87397f6e859588376a0e7c372be5eb64` |
| `locomo_transfer/evaluation_populations.json` | `615b81dad77a64dafcea718f60472bc4b0a526d0f19cafe290c520fb0c17d631` |
| `locomo_transfer/locomo3_unchanged_samples.json` | `696e2090c4c419229c7243d088f6a69e8d30507c9313c2dd7d2aa6cffd636c0d` |

Rechecked original inputs under `/workspace/locomo-refinement/`: `data/locomo10.json` SHA256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`; `runs/pilot100_v3/manifest.json` `12e4e8970b324a1ce25163192fab336e7fe21fa97a9f8e1dca70beb4a11f537e`; `research_data/question_audit100_manifest.json` `5c928b820cd7bf192ec8c094405865ee0d06239bff380893f3849a3ee91dd989`.

Only this local documentation file was written. No experiment, source code, GPU process, service or target-dependent rule was changed.
