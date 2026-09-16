# LoCoMo300 component ablations

User requested five missing ablations on LoCoMo300 (2026-09-13).

- Seven arms: ours, no_cues, no_audit, no_temporal, no_binding, random_cues, payload_keys.
- Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a; FP16, no quantization/thinking, temperature0, seed20260907, TP2. Existing4x3090 instance; two disjoint two-GPU workers.
- MiniLM-L6-v2 1110a243fdf4706b3f48f1d95db1a4f5529b4d41; CPU FP32, normalized384dim, native256tokens.
- Fixed300: largest-remainder category quotas then conversation quotas within category; SHA256(salt+NUL+ID) ordering. Salt locomo-ablation300-20260913-v1. Select using IDs/categories only, excluding category5. Previously exposed histories; exploratory comparison.
- Fresh session10 extraction generated once/history. Audit uses that initial memory. Audited arms share audited memory; no_audit uses the exact pre-audit extraction plus unchanged source-dialogue residual.
- Frozen base: r06_calendar_month -> r12_filter_current_best -> r40_fused_four_turn (4turn/2overlap,640token attachment cap).
- no_temporal omits anchor_time AND disables fuse_evidence anchor. Session dates/source-relative wording remain. no_binding retains date-normalized4turn/2overlap blocks and filtered facts separately; no fusion cap applies to separate entries.
- Reuse original source-only Qwen3.5-9B utility records, admission and realized-candidate rules. Historical probe generation/scoring costs are inherited, not zero. Remap/reselect evidence for changed parents.
- Utility cue selection: mapped positive-utility single/full candidates, complete payload<=2048, one/probe, additional2000tokens with8token upward cost rounding. Retain base. Count payload and distinct key storage.
- random_cues: same eligible pool before utility Pareto pruning; seeded random option order, greedy one/probe admission within the same rounded cap. One fixed draw/history; no outcome-based seed selection. Report realized storage.
- payload_keys: exact ours entries/payload/order/multiplicity; replace question index_text with payload text, no dedupe or reselection. Count reduced distinct-key storage.
- Reader: dense+BM25 RRF offset60, stable0-based ranks, top120, complete-payload skip/continue packing<=2048 evidence tokens, fixed reader prompt,96outputtokens. Empty and length-ended responses remain in denominator.
- Lock all70 memory files before reading evaluation questions. Gold stays local outside deployment package. Source constructors receive only sessions/probes/recipes; reader receives question text.
- Report official category-specific F1x100, fresh300-question ours-relative delta, storage averaged once per10 histories and read tokens averaged per300 questions. Report categories, paired conversation-bootstrap CI and response completion audit.
- Native runtime evidence and content-addressed cache retained; identical requests may reuse results. Shared preparation/embedding/generation costs are separate from table storage/read metrics.
- Fresh reconstruction/current TP2 execution differs from archived1540 run; 55.5278 is not substituted as the300-question baseline.
