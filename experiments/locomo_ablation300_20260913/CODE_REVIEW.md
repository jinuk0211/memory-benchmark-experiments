# LoCoMo300 ablation code review

**Decision: Approve measurement. No CRITICAL or HIGH issues found in the reviewed final code.**

Reviewed `run_ablation.py`, `prepare_inputs.py`, `test_ablation.py`, `run_gpu.sh`, the packaged mirrors, and the relevant frozen construction/retrieval/runtime functions. This approval covers implementation readiness; it does not certify scores before execution and result validation.

## Experimental controls

- Canonical selection contains 300 unique, deterministic IDs across all ten histories. Category 1/2/3/4 counts are 55/62/19/164; selection is invariant to input-record order.
- All seven arms share the same questions and reader configuration. Initial extraction and audit are generated once per history and shared. The omission arm starts from the pre-audit extraction.
- The temporal arm disables both explicit anchoring and anchoring inside fusion. The binding arm retains the same four-turn/two-overlap dialogue blocks and facts as separate entries.
- Random selection uses admitted, mapped, realized single/full candidates before utility Pareto pruning, with one option per probe and the same rounded 2,000-token cap.
- Payload retrieval keys preserve the complete method's selected entries, payloads, order, and multiplicity. No post-replacement deduplication occurs.
- The evaluator matches the frozen indexed BM25/dense RRF and complete-payload packing, with 2,048 evidence tokens and 96 output tokens. Payload and distinct index text are both counted for storage.
- Historical source records are checked against pinned import lineage. The package contains 416 source-utility rows and no benchmark gold file. Source constructors do not receive benchmark questions or answers.
- The global run protocol locks the manifest, environment/vendor code, shard count, arms, and budgets. A changed manifest is rejected before runtime loading. Both preparation workers finish before either evaluation worker starts.

## Verification

- Packaged unittest suite: **3 passed**.
- Ruff on all three new Python files: **passed**.
- Package verification: **5 input files and 19 code files passed their fingerprints**.
- Additional read-only checks: canonical sampling, global protocol mismatch rejection before runtime loading, and `bash -n run_gpu.sh` all passed.
- `git diff -- '*.py'` was attempted; the workspace is not a Git repository, so complete new files were reviewed. Mypy, Pylint, and Black were unavailable. No GPU job or external transfer was performed by this reviewer.

The earlier resume-protocol and lint/type findings were resolved. Final reporting must still verify 300 predictions per arm, retain empty/truncated responses in the denominator, average storage once per history, and compare against the newly measured 300-question complete method, as specified in `PROTOCOL.md`.

## Reviewed SHA-256 fingerprints

| File | SHA-256 |
|---|---|
| `run_ablation.py` | `338fd9721cabb88f1820094d7556088051a7eacecbea90ba4bc69ce623cb7869` |
| `prepare_inputs.py` | `a32248956b2119513e39379de1c99df52bdddce1aa52472432fbd0782cdc5f2d` |
| `test_ablation.py` | `4ad531cea294dd83497ae7b7ab64ea609e4023c1f7e0d3843cdbcce7ec884723` |
| `run_gpu.sh` | `ec7003f43ceebe1b60fc9acc62e67a1d05060ec514fd1e649a1a2671e611809a` |


## Local result reporter addendum

**Decision: Approve `report_results.py`. No CRITICAL or HIGH findings remain.**

- Each arm must contain exactly the same 300 unique question IDs, with matching question text, category, conversation, and QA index. Canonical labels and the local scorer/selection files are verified against frozen provenance before scoring.
- Category-specific F1 uses the pinned `core.f1`. F1 and read-token means include all 300 responses, including empty and output-limited responses. Storage is averaged once over each of the ten histories, with inconsistent within-history values rejected.
- Both reported deltas and confidence intervals use variant minus complete method in percentage points. The 5,000-draw percentile bootstrap resamples whole conversations while preserving paired per-question differences; flattening sampled clusters matches the question-weighted score.
- CSV, Markdown, and LaTeX use the same summary values. The requested seven rows, complete-method baseline, signs, four-decimal F1/delta formatting, column structure, and LaTeX escaping are consistent.
- The earlier local gold/scorer provenance gap and caption spacing were corrected. Independent Ruff analysis passed. The implementing agent additionally reported passing synthetic accounting checks; this reviewer did not generate result tables or run GPU work.

Reviewed reporter SHA-256: `a62f4ba17908fbd905f318aae6db1425aef5a661e943f12e0004d0850064156c`.

No measured outputs were available during this addendum. Actual prediction completion and resulting scores still require the authorized measurement and final validation.

