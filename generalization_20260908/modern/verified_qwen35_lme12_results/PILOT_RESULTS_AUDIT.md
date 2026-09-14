# Qwen3.5 LME12 completed-result audit

PASS for the bounded coverage, metadata and diagnostic metric checks. Generation completed at 2026-09-08 09:12:51 UTC with `questions=12` and `official_judge_pending=true`. These token-F1 values are nonrepresentative feasibility diagnostics, not official LongMemEval accuracy or full500 generalization evidence.

Seven exactly named completed files were read back from `/workspace/generalization_20260908/modern/runs/qwen35_lme12`; each local SHA256 matches the independently read remote SHA256 below. No API judging, GPU inference, production-code changes, uploads or service actions were performed. No active Gemma, score or cache files were read.

## Verified checks

- Exactly 36 rows: 12 unique rows per arm, matching the fixed manifest IDs and order. Missing, duplicate, empty and failed rows: zero in all arms. All 36 outputs have `finish_reason=stop`.
- Canonical question, original answer string, question type/category, question date, abstention flag, conversation ID and method label match. Prediction and hypothesis fields match.
- Recomputed all 36 scores using the unchanged `core.generic_f1` and canonical answers. Maximum absolute difference from saved F1: 0.
- The completed protocol is byte-identical to the independently verified prepare protocol: Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, all model roles and frozen budgets/settings unchanged.
- Memory-lock protocol and source digests match the completed protocol and the previously verified complete `source_sessions.json`. Individual memory files were outside this audit; their recorded content digest is `e8c876baa0331f2879854fb7d2ce530a0015ab101eb53c35fe0ef8865f2b8d4f`.
- Existing paired mean, delta, win/loss/tie and failure-count fields agree with direct recomputation. No new bootstrap or confidence interval was computed.

## Diagnostic results

| Arm | Rows | Mean token F1 (0–1) |
|---|---:|---:|
| seed | 12 | 0.5689830946 |
| r40_fused_four_turn | 12 | 0.4067807136 |
| s_parent_single_2000 | 12 | 0.4067807136 |

| Paired contrast | F1 delta ×100 | Wins / losses / ties |
|---|---:|---:|
| parent-single minus r40, primary | 0.0000000000 | 0 / 0 / 12 |
| parent-single minus seed, secondary | -16.2202380952 | 2 / 3 / 7 |

The 12-ID selection contains single-session-user 1, single-session-assistant 3, multi-session 4 and temporal-reasoning 4, with no abstentions. It does not represent all six question types or the full dataset. The values above do not establish an official accuracy gain, absence of gain, or transfer to full500. The existing `paired_diagnostics.json` remains unchanged, including its original stored interval fields; this audit only verified its requested scalar diagnostics.

Canonical dataset SHA256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. Reused source histories: `../verified_qwen35_lme12_prepare/source_sessions.json`; prior date/role/turn and source-pool coverage audit: `../verified_qwen35_lme12_prepare/PREPARE_AUDIT.md`.

## Readback SHA256

```text
1d6b425947034d588107ad9e8b959d1c720c89538e719c7430f11a058819ec5f  status.json
d2e360039843ec2276dcb502f594c371d780b3895b599ef8958a8d33e5ca0be3  protocol.json
392dd2daef7b669754b9b461d613adac971eae41a323fd6a03cdd26b6c86e99e  memory_lock.json
3e8fdeefbe074f4e290ca22c418df72486c9268b5c79a652b63682bf049c03d8  seed.jsonl
d2ae5e2525ec1114fb65a7e39ceb48923efdabe0673b8d686598fc5dbe086a75  r40_fused_four_turn.jsonl
42800f44714e11d3c7c18427367988ba51cfb8576253ce6c798af68a4c28073f  s_parent_single_2000.jsonl
cfc3a3158fffefc6b5423a581b06af4e2a2bf1f5915c4d6723896b0345d2ef2c  paired_diagnostics.json
```
