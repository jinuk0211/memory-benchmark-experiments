# Gemma LME12 completed-result audit

PASS for the bounded coverage, canonical metadata, provenance and diagnostic metric checks. Generation completed on 2026-09-08 at 12:34:06 UTC; independent local checks completed at 12:42:29 UTC. These are feasibility-pilot token-F1 diagnostics, not official LongMemEval accuracy or full500 generalization evidence.

The run is `/workspace/generalization_20260908/full_transfer/runs/gemma4_lme12_scorer512`. The five completed result/status files match the remote SHA256 values independently supplied by root. Protocol and memory lock are byte-identical to their copies in the verified source bundle; the protocol also matches the earlier reviewed scorer512 receipt. All seven hashes appear below.

Verified checks:

- Exactly 36 rows: 12 per arm, with the same predeclared IDs and order. Missing, duplicate, empty and failed outputs: zero. All 36 finish reasons are `stop`; prediction and hypothesis fields agree.
- Question, answer string, type/category, question date, abstention flag, conversation ID and method match the canonical dataset. Every reported source ID exists in its selected source history, and saved any/all answer-session recall flags agree with canonical session metadata.
- All 36 diagnostic scores were recalculated with the hash-pinned `source/refine.py::generic_f1` and canonical answers. Maximum absolute error from saved scores: **0**.
- The dataset SHA256 and deterministic selection manifest match the canonical500 input and fixed12 subset. Protocol digest and memory-lock source digest match both the retained source-session file and independently adapted canonical histories.
- Saved global paired means, deltas, wins/losses/ties and failure counts match direct recomputation within 1e-12. Per-type counts and means also match. No new bootstrap or confidence interval was computed.
- Logged read/output lengths satisfy the fixed2048/96 limits and reader context8192. This audit checks recorded metadata; it does not rerun tokenization or retrieval.
- Gemma model revision `ee0ef6023621cff504d758262d4e04895a5af4a2`, embedding revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, generation seed20260907 and read2048/add2000/answer96 remain fixed. Writer65536/reader8192/scorer8192 and explicit scorer chunk512 recovery are recorded separately from the preserved failed executions.

| Arm | Rows | Mean token F1 (0–1) | F1 ×100 |
|---|---:|---:|---:|
| Seed | 12 | 0.4203058459 | 42.03058459 |
| r40 | 12 | 0.3750346819 | 37.50346819 |
| Refined / parent-single | 12 | 0.4305902374 | 43.05902374 |

| Contrast | Difference in F1 ×100 | Wins / losses / ties | Exactly identical hypotheses |
|---|---:|---:|---:|
| Refined − r40, primary | +5.55555556 | 1 / 0 / 11 | 11 / 12 |
| Refined − Seed, secondary | +1.02843915 | 2 / 2 / 8 | 7 / 12 |

The entire refined−r40 token-F1 increase comes from one question, `gpt4_7de946e7`; the remaining11 hypotheses are exactly identical. Against Seed, five hypotheses change but only four token-F1 scores change: string equality and F1 ties were checked separately.

The12 selection covers single-session-user1, single-session-assistant3, multi-session4 and temporal-reasoning4. It contains no preference questions, knowledge-update questions or abstentions. Its small, incomplete coverage and the single-question primary improvement do not establish a generalization effect. Official judging remains pending; no token-F1 value is labeled accuracy.

Provenance boundaries: this audit verified the memory lock's protocol/source bindings and equality to the source bundle. The recorded aggregate memory digest is `5167021b7522a871a53a0cc705fee00b3696d0f987499c6d8f9c207d19614259`. The separate source-memory audit owns complete construction, utility and memory replay; that work was not duplicated here. No target outcomes were used to change a rule, threshold, arm or method.

Canonical input: `D:/MemoryData/MemoryData/datasets/LongMemEval/longmemeval_s_cleaned.json`; SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.

```text
35f6a7899e6fa771364f41eda8a2980557cdb4997a2ad817ea54d70fb1d544ec  memory_lock.json
7fb898064959c75941c681ffa41bba3d00a807ca8c1d2ec0a6e4b58b3b6c8751  paired_diagnostics.json
0bb020fd69642a0db3737d63e0b508164ef0eb756a6ab886f90bbf1f6ec0e35c  protocol.json
0392c661694e8ea428e4fe6839366f747e611038260ed8db3340abbaf647c6ef  r40_fused_four_turn.jsonl
842a58dcb53f5a1b3763d8e865c68c052bb26e50d1bb6c5be1b6f60e961d5fa9  s_parent_single_2000.jsonl
940a54bb8a73a759f65a5d22fb23bf5973f4803fb272bfb3491ee7db0e61e653  seed.jsonl
9159b4647478b0f9195f72210fc9965acce8fe24456bf769e3c68210d0ba4144  status.json
```

Only local result checks and documentation were performed. No remote access, GPU/API generation, runtime code changes or service actions were used in this audit.