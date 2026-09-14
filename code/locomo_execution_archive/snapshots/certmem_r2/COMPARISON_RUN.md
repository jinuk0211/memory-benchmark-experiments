# LoCoMo comparison execution (2026-09-08)

The comparison entrypoint is the first v15 recipe in `scripts/run_day3.sh`:
all ten original LoCoMo conversations, 1,540 category 1–4 QA, configurations
`full,no_adaptive,no_residual`, and budgets `400,1600,4000`.
The original source archive was retained separately before integration.

Only `--comparison` enables the following operational changes:

- Pinned Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
  FP16, no quantization, non-thinking; MiniLM-L6-v2 revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU, native 256-token limit.
- `generation_config=vllm`, prefix caching, 48 sequences, 8,192 batched tokens
  and 0.92 GPU utilization match the observed existing comparison server.
  This is comparison-server alignment, not equivalence to native HF `auto`.
- 49,152 context capacity, with explicit overflow rejection, not truncation.
  The native 39,000-token `full_raw` skip is lifted only in comparison mode:
  all 1,540 naive questions are evaluated; original mode would omit 453.
  Measured maximum native chat prompt plus its 32-token answer cap is 40,343.
- Actual prompts, raw responses, token IDs, finish reasons, and phase/QA identities
  are saved for each generation batch. Native output caps and trimming remain
  unchanged. Length and empty results are retained, not silently repaired.
- Encoder forward attention-mask tokens use the same primary token definition
  as the earlier MiniLM server: post-native-truncation, excluding padding.
  Forward counts/latencies and API request counts/latencies are different units.
  Padding work is separate. Pre-truncation input counts are not measured here.
- Fresh output only; existing runtime records are never overwritten or replayed.

The official comparison metric is calculated by `scripts/score_certmem_full.py`
using the unchanged previously used LoCoMo evaluator, SHA256
`8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd`.
It validates all 37 policy groups against the original 1,540 QA identities,
including empty predictions. Native date/number-normalized F1 and lenient
scores remain reference metrics and must not replace official F1 in comparisons.

Dataset SHA256:
`cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993`.
Expected reader outputs: 56,980 (37 × 1,540), with full/full_raw at 1,540.
GPU energy is sampled whole-device energy; rental cost is a wall-time estimate,
not an invoice. Unknown failed work stays unknown, not zero. Closed token audit
and official scoring are separate from native process/output completion.

A-MEM code changes are prepared separately. CertMem has execution priority.
Do not shut down or otherwise change power state of the user's PC.
