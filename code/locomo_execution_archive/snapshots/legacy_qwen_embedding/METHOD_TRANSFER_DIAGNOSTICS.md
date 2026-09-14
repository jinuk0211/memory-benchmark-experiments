# Source-only method transfer diagnostics

Read-only CPU audit on 2026-09-08 of completed Qwen3.5 and Gemma LoCoMo construction artifacts. **No benchmark questions, gold answers, failure examples, F1 result files or new inference were used in this diagnostic.** Counts use each model's saved source utility items, parent/refined memories and construction details.

Pinned models/tokenizers: Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; google/gemma-4-E4B-it revision `ee0ef6023621cff504d758262d4e04895a5af4a2`. Tokenizer files were loaded locally with network access disabled at the loader and CUDA masked for the audit process. Frozen `make_options` and `construct(..., family='parent_single', extra_budget=2000)` exactly reproduced **all six refined memories and construction-detail objects** from the saved utility records.

## Source admission

The fixed criteria are full-source generated-answer F1 >=0.8 and full-minus-empty answer mean logprob >=0.1 nats/token. Both are required. F1 here measures **source-probe answer consistency**, not benchmark accuracy. All138 fit probes per model were scored; none were skipped for source-window capacity.

| Model | Conversation | Fit/scored | Full-answer F1 passes | Full-minus-empty passes | Both / admitted |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5 | conv-42 | 44 | 34 | 44 | 34 |
| Qwen3.5 | conv-47 | 48 | 29 | 48 | 29 |
| Qwen3.5 | conv-50 | 46 | 26 | 46 | 26 |
| **Qwen3.5 total** | | **138** | **89** | **138** | **89** |
| Gemma4 | conv-42 | 44 | 26 | 44 | 26 |
| Gemma4 | conv-47 | 48 | 26 | 48 | 26 |
| Gemma4 | conv-50 | 46 | 31 | 46 | 31 |
| **Gemma4 total** | | **138** | **83** | **138** | **83** |

In these source pools, the likelihood threshold excludes no probe; the answer-consistency condition reduces admission to89/138 (64.49%) and83/138 (60.14%). This describes the selected source pools and does not prove that the threshold is generally unnecessary.

## Candidate mapping and selection

Non-pair counts include full and singleton alternatives that passed the frozen source admission, positive-gain and realized-generation conditions. Raw and mapped counts are before the final budget optimizer's Pareto filtering. Multiple alternatives can belong to one probe; these counts are not counts of unique factual claims.

| Model | Conversation | Raw non-pair -> mapped | Pareto non-pair alternatives | Selected (full / single) |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5 | conv-42 | 72 -> 72 | 60 | 14 (3 / 11) |
| Qwen3.5 | conv-47 | 58 -> 58 | 46 | 12 (1 / 11) |
| Qwen3.5 | conv-50 | 54 -> 54 | 40 | 10 (1 / 9) |
| **Qwen3.5 total** | | **184 -> 184** | **146** | **36 (5 / 31)** |
| Gemma4 | conv-42 | 52 -> 52 | 44 | 11 (0 / 11) |
| Gemma4 | conv-47 | 53 -> 53 | 39 | 11 (0 / 11) |
| Gemma4 | conv-50 | 64 -> 64 | 49 | 10 (0 / 10) |
| **Gemma4 total** | | **169 -> 169** | **132** | **32 (0 / 32)** |

Every admitted probe retains at least one mapped non-pair candidate:89 groups for Qwen3.5 and83 for Gemma. Every mapped non-pair option individually costs <=2000 stored tokens. Mapping therefore removes none of these non-pair candidates. The optimizer selects at most one representation per probe under the shared conversation budget. Full/single describes the required source subset; final stored payloads retain complete covering parent units.

## Added storage

Each conversation has an independent2000-token extra-storage cap. Actual cost includes payload plus distinct index text. The optimizer rounds each option's cost upward to a multiple of8; its rounded total can exceed actual usage. All totals below were independently recounted with the corresponding pinned native tokenizer and equal the saved optimizer and memory-delta fields.

| Model | Conversation | Actual added tokens | Rounded added tokens | Actual unused budget |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5 | conv-42 | 1950 | 2000 | 50 |
| Qwen3.5 | conv-47 | 1941 | 1984 | 59 |
| Qwen3.5 | conv-50 | 1930 | 1976 | 70 |
| **Qwen3.5 total** | | **5821 / 6000** | **5960 / 6000** | **179** |
| Gemma4 | conv-42 | 1973 | 2000 | 27 |
| Gemma4 | conv-47 | 1840 | 1872 | 160 |
| Gemma4 | conv-50 | 1921 | 1960 | 79 |
| **Gemma4 total** | | **5734 / 6000** | **5832 / 6000** | **266** |

Actual utilization is97.02% for Qwen3.5 and95.57% for Gemma. This establishes that both constructions add a substantial fraction of the allowed storage; it does not establish matched semantic information or matched tokenization across models. Gemma conv-47 leaves160 actual tokens unused; the fixed optimizer is not required to spend every token.

## What these observations establish

Both model-specific source pipelines are active, yield admitted candidates, map those candidates successfully and use most of their storage allowance. A missing candidate pool, wholesale mapping failure or broadly unused storage budget is therefore not an explanation supported by these artifacts. Qwen selects36 supplemental units and Gemma32; this is descriptive evidence about what the frozen constructor did.

The source-history files are identical, but the model-specific parent artifacts differ and the probes are generated independently. Only **one of138 fit-probe signatures** matches exactly across models when comparing conversation, session, literal question, literal answer and ordered candidate-source IDs. Strict text mismatch does not imply different underlying facts. In particular,89 versus83 admissions is not a paired model-capability comparison, and a difference in selected full/single counts does not identify the cause of downstream benchmark outcomes.

This audit does not measure benchmark retrieval effects, question coverage, usefulness of added information, or causal mediation. A token budget nearly being filled is not proof of useful memory. Added storage, question-key augmentation and payload choice remain combined. No rule, threshold, budget, prompt or candidate-selection policy was changed in response to these counts.

## Saved-record versus last-cache distinction

The audit uses immutable `utility/items` records, as the actual constructor does. The prior Gemma audit found two unused pair diagnostics whose shared cache keys were later overwritten by answer_removed evaluations. No empty/full/single score or control generation consumed by parent_single differs; see [GEMMA_ROLE_CACHE_AUDIT.md](../../../../generalization_20260908/modern/GEMMA_ROLE_CACHE_AUDIT.md). Replaying all six constructions exactly confirms that this memo uses the values that produced the saved memories. It does not claim all diagnostic cache records equal all historical item values or that different GPU batches are numerically identical.

## Provenance

Remote run prefix: `/workspace/generalization_20260908/modern/runs/`; run names `qwen35_locomo3_fullrole` and `gemma4_locomo3_fullrole`.

| Artifact | Qwen3.5 SHA256 | Gemma SHA256 |
| --- | --- | --- |
| `protocol.json` | `54eb61046be0f4a6f67755fbb6250ad3f81dc2edd6ccc396f92c23447988dc26` | `bf956995dcf7ef05afac315c44e1734e0c2727604c5882954bed7c6dd8d5f74e` |
| `source_probes.json` | `552b1b00b2c4e5057d99ffc79a481a92f7bfadab0b29706fe61c6ae52b78c726` | `cc02c94b5967d6a4b51c718b834d45e02d140dfa0a60f60fc9793b101f0012d0` |
| `utility_selection.json` | `14892a55d45684635d03e41edab0ea7d9254d6a1093494bd20b3acbc4fc4fad4` | `2b0e5d671d996905592a1c0ffcd63465b41639c5fa8047ed6e855380ef1b7902` |
| `memory_lock.json` | `31ec9fb949ee7402353b7972112eb7b597b5d36f9af172902a717507d2333996` | `6ac2da48d6bf0d6179682efcc21a98755f4151f2a1c691cff53b044a767e27da` |

Both `source_sessions.json` files have SHA256 `ea6d36256faa8dfc3fabaef2b1bd2d4f4eccb81f35c359202fc1d50e5797b60a`. Model assets and source methods were read only; this local Markdown document is the sole new artifact.
