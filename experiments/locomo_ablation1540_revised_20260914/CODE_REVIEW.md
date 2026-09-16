# Independent Python review — 2026-09-14

**Decision: APPROVE. No blocking security, correctness, data-leakage, or result-integrity findings remain in the reviewed files.**

Review performed by the independent Python reviewer agent. Only this review document was written by the reviewer; implementation changes were made by their owning agents.

## Verified checks

- Ruff check passed for all six reviewed Python files after the final validation changes.
- All 6 runner behavioral tests and all 9 reporting/statistical tests passed independently under Python 3.14.
- All six reviewed source files compile.
- Frozen input and code fingerprints verified against the manifest.
- All 160 prepared memory files verified against ten locks; all ten Ours memories are byte-identical to the archived no_binding memories.
- The fixed 300 pilot records contain no gold/answer/prediction fields and match the archived full1540 no_binding question/context metadata.
- Git diff was attempted; D:\MemoryData is not a Git repository, so current new source files and pinned vendor dependencies were reviewed directly.

## Reviewed behavior

- The no-binding baseline is reproduced before all controlled variants are constructed. Temporal removal preserves normalized date headers, unit identities/order, filtering, and deduplication; only relative-time body expansion is disabled.
- Utility, no-cues, payload-key, and ten predeclared random controls preserve the intended common parent memory. Stored tokens are decomposed into parent payload, cue payload, and distinct retrieval keys, with the 2,000 extra-token cap checked.
- Native receipts require matching generated text, request keys, valid integer token metadata, and stop/length completion reasons. Runner/pilot verify rendered chat input-token counts. Empty and length-ended outputs remain in the fixed evaluation population.
- Report validation reconstructs contexts from locked payload indices, independently recounts storage/read tokens, verifies all16 full1540 populations and native receipts, and re-scores both pilot candidates before validating the reader-selection rule.
- Statistics correctly use question-weighted F1/read means and equally weighted history storage means. Paired conversation bootstrap, all1,024 cluster sign flips, Holm correction over three primary comparisons, and averaging allten random seeds are implemented and behaviorally tested.

## Limits

- This is pre-execution code and prepared-artifact approval; successful full GPU evaluation and final result validation are still required.
- The reporter treats the before-full1540 selection flag as a declared record, not an independently verified timestamp. Root must freeze and retain the selection artifact before launching full evaluation.
- The random-control confidence interval is conditional on the mean of the ten predeclared seeds; between-seed variation is separately reported.
- The 300 development questions and full1540 set were previously exposed; outputs are explicitly exploratory.
- No GPU or paid API calls were run by this reviewer. Static type checking and LaTeX compilation were not run.

## Fingerprints

Input manifest SHA256: `1b3a1357e4a893eeb2fad407fbc41ec34a56a6bd174001cedcf22ab69148bfb0`

| File | SHA256 |
|---|---|
| prepare_inputs.py | `074c34d41234ddb37445082328812c70ebdcdef899650503571bd7c4df02dac5` |
| package/code/run_revised.py | `5b2f1e456755a837c9b9ccdf1e61452d20bca99c9570d3513175057763b3c5ed` |
| package/code/test_revised.py | `acf65010fb7a12994cf3d535dcda9d6f0787fc69420537f2426ff49c55dd1666` |
| package/code/reader_pilot.py | `bd4916f43dd2ca090ed0bd2350f58149ba7f5b1153f99ceba86ebbcaaa0e1839` |
| report_results.py | `07f3821d34685d2149414e3f97058f726991aed1d7c91210c6c8b023808b0d85` |
| test_report.py | `96894e056177e143367f24a905447e9bf8c4e4251d55f7637f6c4a99ac895025` |

## Addendum: local-only reader selector and reporting runtime

Decision: APPROVE for local pilot scoring and immutable reader selection.

- Reviewed score_reader_pilot.py after its final Callable annotation and loaded-Porter-source fingerprint changes; Ruff passed independently.
- The selector verifies frozen inputs/code and canonical local gold, retains exactly the fixed 300 questions for each reader, validates both reader prompts, context/script/runtime fingerprints and all 600 native receipts before writing results.
- The winner is determined solely by overall official F1; exact ties select legacy. Category scores and empty/length/unknown counts are diagnostic only.
- Existing READER_SELECTION.json is immutable via frozen_save; reruns preserve its timestamp and must reproduce the entire record. Root must create and retain this record before launching the full1540 comparison.
- Independent behavioral checks confirmed tie handling, strict improvement, retention of empty/length/unknown answers, and rejection of missing/duplicate/non-string responses.
- Independent re-scoring of all 3,080 archived full1540 predictions exactly reproduced every prior per-question F1 and the overall values 55.37328293266584 (with binding) and 58.71430233556697 (no binding).
- Actual local scoring runtime is C:\Users\tgc04\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe, CPython 3.12.11, with nltk 3.9.1 from report_deps. Its compiled regex dependency is ABI-specific; Python 3.14 is not the supported runtime for these local scoring tools. The earlier pure behavioral tests passed on 3.14, but actual NLTK scoring was verified on the required Hermes runtime.
- No frozen package files were edited for this addendum. The report_results.py change uses ROOT/report_deps for local imports; the new selector fingerprints the actual imported nltk.stem.porter module.

Updated fingerprints (supersede earlier entries for the same file):

| File | SHA256 |
|---|---|
| score_reader_pilot.py | 3e1fc26d20dfdafad97e5506da0bf244e1699aab239e921f063a84c1f6bd6745 |
| report_results.py | 7a0236618f405952937888a1c3784d2b8f6cd2c6c4a91f0b0b0819d6b6ee248f |

## Addendum: final offline GPT-4o-mini export

Decision: APPROVE for report_new_judge.py, test_report_new_judge.py and the final offline export additions to JUDGE_PROTOCOL.md. No blocking correctness, provenance or secret-handling findings.

- All ten synthetic offline tests and Ruff passed independently under Hermes Python 3.12. The reviewer made no API calls and did not access or print credential values.
- The exporter reuses the previously reviewed preparation validation to bind all canonical 1,540 rows to their unchanged original reader predictions, native receipts and frozen source metrics. Exact input bytes, historical runner SHA, prompt, requested model, source dataset and run settings are checked again.
- Every recorded native judge response must identify the pinned gpt-4o-mini-2024-07-18 snapshot. Selected receipts additionally require a valid raw CORRECT/WRONG label, positive prompt/completion usage and stop. Missing judgments, changed models, contradictory labels, duplicate response identities, partial journals and altered original F1/input are rejected before writing outputs.
- Exact API payload hashes deduplicate requests without dropping logical rows. All 1,540 mapped judgments remain in the denominator; overall and category counts are independently reproduced and checked against the runner summary.
- Token counts must be exact nonnegative integers with consistent totals and cached-token bounds. Known response cost is recalculated with pinned prices; paid invalid responses and separately recorded unknown retry-cost bounds are retained. Malformed paid responses never score a logical row.
- Source F1, category F1, storage and read metrics remain unchanged. The eight-column LaTeX row contains the four F1 categories, Overall F1 and independently validated mini Accuracy. It introduces no unsupported comparative bolding.
- Export requires a new empty report directory and confirms all input/judge file fingerprints remain unchanged during validation. The protocol retains the secure real-PTY/getpass warning gate without modifying the historical runner.
- This is approval of offline preparation/export logic, not confirmation that final new API judgments already exist. Actual complete native artifacts must pass the exporter before publishing Accuracy.

| File | SHA256 |
|---|---|
| report_new_judge.py | 9658f1afae019da3252e0ab6103b3677f209332db2dec3468a89afb16a5dec64 |
| test_report_new_judge.py | afbf298ccb4ba310b5ea312b014a1418f22ee379f20e3513604a34688b2df544 |
| JUDGE_PROTOCOL.md | 48dd72c51a4ff5547b5653d110a318292e52b925af3e337c7eb669d7f5d7dca9 |
