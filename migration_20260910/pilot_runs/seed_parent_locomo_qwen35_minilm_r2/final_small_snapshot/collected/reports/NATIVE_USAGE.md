# Native usage for one attempt

Attempt: seed_parent_locomo_qwen35_minilm_r2 (quiescent)

| Scope | LLM tokens (M) | Embedding tokens (M) | Eval seconds/question |
|---|---:|---:|---:|
| seed_parent_single_2000 | 3.322521 | 0.697131 | 0.483280 |

Observed lower bounds: LLM 3.322521 M; embedding 0.697131 M.

Official LongMemEval accuracy: pending separate upstream bridge.

- Only new reader native invocations are counted. Source construction and utility are imported, not free.
- Fresh reader caches are caller-declared; imported source artifacts are bound separately by complete hashes.
- Inherited construction/scoring costs are unavailable in this incremental report; see the original run cost evidence.
- Counts are native observed invocations, including retries/failures; unobserved provider-internal retry costs are unknown.
- Native LLM totals include prompt and completion IDs, including likelihood stub completion tokens.
- Embedding totals use actual nonpadding native forward inputs after the 256-token window; padded tokens are separate.
- Wrapper/encode-attempt and row generation_batch_seconds are excluded from native sums.
- Only complete quiescent coverage receives diagnostic score and complete totals. Known lower bounds remain available otherwise.
- Timing is paired evaluate_sample wall time, not full-job end-to-end time; it is never added to nested native duration.
- Official LongMemEval accuracy requires the separate actual upstream judge bridge. F1 is auxiliary only.
- Source/protocol hashes are checked; model weight files, canonical dataset contents and memory payloads are not rehashed here.

Evidence gaps: 0
