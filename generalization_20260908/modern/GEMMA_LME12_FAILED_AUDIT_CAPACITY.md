# Gemma LME12 failed audit capacity check

Read-only CPU reconstruction of the completed source-side artifacts for `gpt4_b5700ca9` in `/workspace/generalization_20260908/modern/runs/gemma4_lme12`. No target questions or answers, active runtime cache, inference, GPU operations, service changes, or remote writes were used. Only this item's 52 completed session10 writer caches and exact audit-key existence checks were accessed.

The strict 8,192-token writer limit fails on audit request 13, corresponding to session 13, source IDs `D13:1`–`D13:12`. Its exact native-chat input has **7,757 tokens**; the frozen audit generation reserve is **700**, giving **8,457**, or **265 over the limit**. This is the only failing request and the maximum among all 52 audit requests.

| Check | Result |
|---|---:|
| Source sessions / reconstructed session10 cache hits | 52 / 52 |
| Reconstructed parent units | 446 |
| Session10 parsing fallbacks | 1, session 13 |
| Audit requests / existing exact audit caches | 52 / 0 |
| Requests exceeding 8,192 including reserve | 1 |
| Failing request original-block tokens | 3,676 |
| Failing request existing-memory tokens / units | 3,947 / 12 |
| Exact templated input / generation reserve / total | 7,757 / 700 / 8,457 |
| Native rendered-encode versus chat token-ID parity | All 52 pass |
| Final seed-memory file exists | No |

Session 13's completed writer output was nonblank, 419 output tokens, and ended with `finish_reason=stop`; it was not reported as a truncated generation. Frozen fact parsing nevertheless yielded no accepted facts, so `core.build(..., 'session10')` reproduced `EXTRACTION_FALLBACK 1/52 chunks` and retained its 12 raw source units. The subsequent frozen audit prompt includes both the original dialogue and those source-overlapping existing memory units. This reconstructed prompt establishes the concrete capacity failure; no claim about why the model produced unparsable text is made.

`Runtime.generate` scans uncached prompts and checks input plus reserve before invoking generation for the collected batch. Consequently, failure at request 13 prevents new generation for the entire audit call, consistent with zero exact audit-cache files. The audit maximum above is an actual reconstruction from this failed parent, not a hypothetical source-only window estimate. The failed strict-8K outputs remain unchanged.

Provenance:

- Model: `google/gemma-4-E4B-it`, revision `ee0ef6023621cff504d758262d4e04895a5af4a2`; tokenizer loaded locally from that pinned snapshot, with frozen nonthinking prompt construction and the reviewed `ListChatTokenizer` compatibility wrapper.
- Actual run `protocol.json` SHA256: `63f4b90439d94b0a41769db83cca230aac2a95f1bf6401cd7d933a7a457d1be8`.
- Frozen core SHA256: `71384cac6aaa260d6df4f24f605797a8a886cd51514a83c9c1d041eff99f14fa` (checked against the protocol before reconstruction).
- Reconstructed parent semantic digest (frozen `core.digest`): `d88942880d6f46476866c3f4b82cc5bf667885a3860e0a79507d888181c64136`.
- The 52 completed session10 cache-file inventory digest: `5e893fc576d4c7dc2c384f2912fa88ddb75a6ff1e72a955664af52ea765c74ee`.

This check is confined to one source item and establishes a strict writer-capacity failure. It does not establish benchmark accuracy, full500 portability, or the behavior of any larger-capacity recovery run.
