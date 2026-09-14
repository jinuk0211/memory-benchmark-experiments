# Five-case Qwen3.5 LME12 diagnostic error analysis

The negative seed-relative pilot diagnostic is not entirely a wording artifact. Among the five changed cases, one F1 increase comes solely from removing unmatched words, one changes the requested temporal quantity, and three replace the reference quantity with a different number or an `unknown` response. The relevant numeric operands remain present in the logged reader contexts of all three losses.

This is a post-completion descriptive inspection of exactly the five cases selected by unequal seed/refined token F1, not an official LongMemEval evaluation or a representative error sample. It uses completed local rows and relevant passages in the already verified local source histories. No external judge, inference, remote reads, candidate fixes, or changes to prompts, methods, thresholds, selection, or runtime were made. No official accuracy or per-case official verdict is assigned.

## Observed outputs

F1 is the frozen `generic_f1` diagnostic, scaled here from 0 to 100. `refined` means `s_parent_single_2000`. In every row below, the refined answer is byte-identical to the r40 answer.

| Question ID / requested fact | Saved reference | Seed answer | r40 = refined answer | Seed → refined F1 |
|---|---|---|---|---:|
| `157a136e` / grandma–user age difference | 43 | 43 | 53 | 100 → 0 |
| `e4e14d04` / membership duration at meetup | Two weeks | one week | two weeks | 0 → 100 |
| `d6062bb9` / combined YouTube and TikTok views | 1,998 | 1,998 | unknown | 100 → 0 |
| `27016adc` / renovation cost as a percentage of property price | 10% | 10% | Unknown | 100 → 0 |
| `41275add` / previously recommended Mayo Clinic video | Title and URL | Title + “by the Mayo Clinic” | Same title | 57.142857 → 62.5 |

All 15 inspected rows have `status=ok` and `finish_reason=stop`; their read contexts contain 2,019–2,048 tokens. Thus these are completed nonempty responses, including the two literal `unknown` responses. The five differences contribute `(-1 + 1 - 1 - 1 + 0.0535714286) / 12 = -0.1622023810` to the 12-question mean F1. This arithmetic does not convert the diagnostic into accuracy.

## Case evidence and bounded interpretation

**Age difference (`157a136e`).** Source `D38:1` identifies the grandma's 75th birthday; `D45:3` discusses the user's age as 32. Both seed and refined contexts explicitly retain grandma age 75 and the memory statement that the user is 32. The required subtraction, 75 − 32, yields the saved reference 43; the changed output 53 is a different quantity, not a spelling or formatting variant. The logged refined context does not lack these operands. This supports an output-level arithmetic/fact-use discrepancy, without identifying why the reader produced 53.

**Membership duration (`e4e14d04`).** Source `D12:1` says the user joined Book Lovers Unite three weeks ago; `D41:1` says the meetup occurred last week. Both sessions are dated 2023/05/28, and both relative-time facts appear in both seed and refined contexts. The reference's two-week interval is consistent with the coarse three-weeks-minus-one-week calculation. Changing one week to two weeks changes the requested duration; it is not merely a paraphrase gain. The frozen lexical metric additionally treats `week` and `weeks` as different tokens, but that does not remove the substantive number change. Exact calendar boundaries or a causal explanation for the changed answer are not established here.

**Combined views (`d6062bb9`).** Source `D15:1` supplies the TikTok count 1,456; `D20:1` supplies the YouTube count 542, while `D20:5` describes the YouTube tutorial as the user's most popular video. Seed, r40, and refined reader contexts retain both numeric counts and the most-popular YouTube description. The sum is 1,998, matching the saved reference. Refined's `unknown` omits the requested total rather than offering alternate wording for it. The source's TikTok passage describes a successful video; this audit does not independently establish an exhaustive ranking of all TikTok videos. Presence of the relevant counts rules out their simple absence from this logged context, but does not determine why the model declined to supply a total.

**Renovation percentage (`27016adc`).** Source `D12:1` states the countryside property's listed price is $200,000; `D32:3` states the planned renovations cost about $20,000. Both values remain explicitly available in seed and refined contexts. The ratio 20,000 / 200,000 × 100 is 10%, matching the reference. `Unknown` omits that requested quantity; this is not punctuation or percent-sign normalization. The refined context also contains property-tax discussion, but no claim that those passages caused the output is supported by these artifacts.

**Mayo Clinic video (`41275add`).** Source `D26:2` and all three logged reader contexts contain the title “How to Sit Properly at a Desk to Avoid Back Pain” and the reference URL, whose video ID is `UfOvNlX9Hh0`. Both seed and refined return that same title and omit the URL. Seed additionally says “by the Mayo Clinic”; refined removes those words. The frozen normalizer yields 13 prediction tokens for seed and 10 for refined, with the same 10 overlapping reference tokens out of 22: F1 changes from 20/35 to 20/32. Thus the increase is entirely a lexical-precision effect, with no added requested fact. Whether the title alone adequately answers the user's reminder request is an official-judging question left unresolved, rather than answered from token F1.

## What this permits and leaves unresolved

The five seed-relative answer changes already occur in r40. For the age, membership, and renovation cases, r40 and refined contexts are also byte-identical. For views and the video reminder, their contexts differ but their answers remain identical. These five cases therefore do not show an incremental answer benefit or loss from the +2,000-token parent-single addition; they contextualize the seed-to-r40/refined bundle comparison. They do not establish the addition's general ineffectiveness.

The inspected contexts retain the essential operands in the three losses and the temporal operands in the substantive win. This distinguishes these observations from a simple story in which retrieval omitted a necessary number. It does not isolate retrieval, context ordering, memory wording, distraction, or reader computation as a cause. No ranking traces, counterfactual context interventions, alternate generations, or complete memory-stage comparison were used. The frozen method and current hypotheses remain unchanged. These five selected cases and the nonrepresentative 12-question pilot support no conclusion about full500 transfer.

## Reproducible inputs

All paths below are relative to this document unless specified. SHA256 values were verified locally during this inspection; question/reference metadata had already passed the completed-result audit in `PILOT_RESULTS_AUDIT.md`.

```text
3e8fdeefbe074f4e290ca22c418df72486c9268b5c79a652b63682bf049c03d8  seed.jsonl
d2ae5e2525ec1114fb65a7e39ceb48923efdabe0673b8d686598fc5dbe086a75  r40_fused_four_turn.jsonl
42800f44714e11d3c7c18427367988ba51cfb8576253ce6c798af68a4c28073f  s_parent_single_2000.jsonl
80b41d750cedd6365292fcadb828c1f9c7e5fbd401a88fc3c0a9b5bcc07399c3  ../verified_qwen35_lme12_prepare/source_sessions.json
```

Metric implementation: `../source/refine.py:98`, unchanged lowercase/alphanumeric token overlap with `a`, `an`, and `the` removed. The one wording-only case's overlap and denominator were recomputed locally. The full canonical dataset was not downloaded or reread for this analysis; source facts came from the existing verified source-only copy, and references came from the previously verified completed rows.
