# Final LoCoMo GPT-4o-mini judge protocol

Status: the final selected 1,540-answer set and its new judgments are pending. Preparation makes no API calls. The parent task verified model access with HTTP 200 from `GET /v1/models/gpt-4o-mini-2024-07-18`; that check did not create a completion or judgment. Credentials remain outside these files.

## Fixed comparison

Use the existing runner `D:\MemoryData\outputs\locomo_gpt4omini_judge_20260911\run_judge.py`. Its requested model is `gpt-4o-mini`; every final exported receipt must return **`gpt-4o-mini-2024-07-18`**, matching the old table. A different returned version must be reported and resolved before filling the comparable Accuracy cell.

The rubric is the unchanged LightMem official-distribution LoCoMo judge at [commit 8449d574df6bae1bdf3314a1564da65e2f37e046](https://github.com/zjunlp/LightMem/blob/8449d574df6bae1bdf3314a1564da65e2f37e046/experiments/locomo/llm_judge.py). Exact prompt SHA256: `62395dd312a631dfd9355026a0b69cc936018274c3198b6365b5c2a5c9bca9e0`. The local upstream copy adds optional client/model arguments; the recorded upstream verification confirms identical default prompt and API payload.

- Send only the canonical question, raw `str(answer)` gold, and unchanged saved prediction. No method, category, F1, or memory context enters the judge prompt.
- Include all categories 1--4: 282 Multi-hop, 321 Temporal, 96 Open-domain, 841 Single-hop. Exclude category 5. Do not apply category-specific F1 preprocessing to gold, including semicolon truncation.
- Preserve empty, `unknown`, and length-ended reader answers if present in the validated population. Do not drop difficult rows or modify their text.
- The rubric grades generously when the answer covers the same topic and accepts equivalent dates/time periods despite formatting or relative-time differences. Preserve its original wording, including its explanation/JSON instructions.
- Use temperature 0, `response_format={"type":"json_object"}`, and no explicit max_tokens or seed, matching upstream. Accept only the raw JSON labels `CORRECT` or `WRONG`.
- Accuracy is 100 times the number of CORRECT labels divided by **1,540**, published only after every row has a valid receipt. It is separate from token F1.

Current pricing was verified by the parent task on 2026-09-14 at the [official model page](https://developers.openai.com/api/docs/models/gpt-4o-mini): input USD 0.15, cached input USD 0.075, output USD 0.60 per million tokens. Existing runner defaults match these prices. Its default USD 5 ceiling is a spending/reservation bound, not a prediction of the final bill.

## Prepare without API use

`prepare_new_judge.py` accepts either the validated revised round1 or marginal-utility round2 report. It checks:

1. Pinned canonical dataset SHA256 `cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993`, all 1,540 unique IDs, ten histories, exact question/gold/category/QA index.
2. `VALIDATION.json` PASS, agreement with `RESULTS.json`, the native protocol seal, full1540 population, selected arm, Qwen3.5-9B pinned revision, and reported F1 aggregate.
3. Selected-arm original prediction file hashes and equality to scored rows, then the exact reader request key, native receipt hashes, saved answer and token metadata. Round1 and round2 native receipt layouts are supported.
4. Existing mini runner and exact official prompt; preparation refuses a nonempty output directory.

Example for revised round1, **only after its full report passes**:

```powershell
& 'C:\Users\tgc04\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe' `
  'D:\MemoryData\experiments\locomo_ablation1540_revised_20260914\prepare_new_judge.py' `
  --results 'D:\MemoryData\experiments\locomo_ablation1540_revised_20260914\results' `
  --run 'D:\MemoryData\experiments\locomo_ablation1540_revised_20260914\collected\full_grounded' `
  --arm ours `
  --output-dir 'D:\MemoryData\experiments\locomo_ablation1540_revised_20260914\judge_final1540'
```

For marginal round2, use `locomo_marginal_utility1540_20260914\results` and `locomo_marginal_utility1540_20260914\collected\full1540`, then choose a new output directory. Do not overwrite either prior experiment or judge run.

The output contains `input.jsonl`, `input_manifest.json` with exact source/native provenance and required returned model, and `official_accuracy_prompt.txt`. Preparation does not start judging or fabricate pending scores.

## Execute and verify later

Pass the prepared `input.jsonl` and its directory to the existing runner using `--input`, `--output-dir`, `--api-key-stdin`, `--workers 8`, and `--budget-usd 5`. Omit `--limit` for the full pending population. The already authorized key is supplied transiently through the reviewed stdin/getpass path, never a command argument, environment file, or artifact.

The runner deduplicates identical complete payloads, durably journals usage and responses, and resumes the same input/prompt/model contract. It retries transient connection/timeout, 429, and 5xx failures up to five attempts; quota/billing failures stop. Invalid labels, missing usage, or exhausted retries remain incomplete rather than becoming WRONG. Paid invalid responses remain in the cost journal. Pending answers are not removed from the denominator.

Before table export, verify all 1,540 input hashes and row mappings, raw JSON labels, unique receipt identities, token usage, `finish_reason=stop` for every selected judge response, and **every receipt's returned model equals `gpt-4o-mini-2024-07-18`**. Compare the count directly with the complete summary; preserve the raw answers and F1 values. The existing runner records the model but does not itself enforce this final version gate. Identical payload reuse may reduce API calls below 1,540 while retaining all 1,540 logical rows.

## Historical provenance

The old `s_parent_single_2000` row was F1 **55.5278135**, mini Accuracy **1,120/1,540 = 72.7272727%**. This task rechecked all canonical identities, raw gold/question/category, unchanged predictions, API payload hashes, native labels and usage. All 1,540 linked receipts returned `gpt-4o-mini-2024-07-18` with stop; there were 1,529 distinct Ours payloads. Artifacts: `D:\MemoryData\outputs\locomo_gpt4omini_judge_20260911`.

A separate historical no-binding evaluation exists at `D:\MemoryData\outputs\locomo_no_binding_gpt4o_20260913`: F1 **58.7143023**, Accuracy **1,157/1,540 = 75.1298701%**, returned model **gpt-4o-2024-08-06**. That is GPT-4o, not GPT-4o-mini, and cannot populate the mini column or replace the pending new final-answer judgment.
## Final offline export

Launch the frozen runner under Python `-W 'error::getpass.GetPassWarning'` with a real tool PTY (`tty: true`). Wait until the actual `API key:` prompt is observed before sending the transient key through stdin. Promoting `GetPassWarning` to an error makes the process stop if secure no-echo input is unavailable. The parent verified that warning filter offline; the frozen runner is unchanged.

After all judgments complete, run:

```powershell
& 'C:\Users\tgc04\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe' `
  'D:\MemoryData\experiments\locomo_ablation1540_revised_20260914\report_new_judge.py' `
  --judge-dir 'D:\MemoryData\experiments\locomo_ablation1540_revised_20260914\judge_final1540' `
  --output-dir 'D:\MemoryData\experiments\locomo_ablation1540_revised_20260914\judge_final1540_report'
```

The exporter makes no API calls and requires a new empty report directory. It rechecks the unchanged canonical 1,540-item input and native reader provenance, exact input/prompt/run configuration, every judge payload mapping, raw labels, response IDs, usage and the pinned returned mini snapshot. It independently reproduces overall/category counts and compares them with the runner's complete summary. Partial results, conflicting native labels, changed source F1, GPT-4o judgments, or a different mini snapshot fail before output.

Only valid selected labels with positive usage and `finish_reason=stop` score rows. Earlier paid malformed or length-ended responses may remain in the journal; they retain their native content and cost, never score a row, and must also return the pinned mini snapshot with consistent known usage. Every final logical question must still map to a valid selected receipt. Unknown-cost retry bounds are retained separately.

Outputs are `JUDGE_VALIDATION.json` (source and judge file hashes), `RESULTS.json` (unchanged selected source metrics plus overall/category mini accuracy and billing), `mapped_judgments.jsonl` (all 1,540 logical items and receipt hashes), `native_judgments.jsonl` (all unique billed response records), and `table_main_ours.tex` (the requested eight-column row: blank group cell, Ours, four category F1 values, Overall F1, mini Accuracy). F1 values are preserved at full precision in JSON and displayed to two decimals in LaTeX; no unverified comparative bolding is added.
