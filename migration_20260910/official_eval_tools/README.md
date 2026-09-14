# Official LongMemEval-S export and audit bridge

This separate bridge makes no judge/API calls and does not change the vendored evaluator. It preserves the exact completed model hypothesis (including whitespace and empty outputs), validates a separately declared population, and summarizes only the unchanged upstream Boolean labels. Run every method/model arm separately.

## Inputs

- `--predictions`: one completed `run_transfer.py` METHOD.jsonl, with `question_id`, `method`, `hypothesis`, `prediction`, and consistent `status` fields.
- `--protocol`: that run's frozen protocol.json. Its `selection.selected_ids` must exactly match the expected population. Model, embedding, and budgets are recorded from this file; no embedding is substituted.
- `--reference`: the full canonical cleaned-S JSON, pinned SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- `--expected-ids`: a predeclared JSON string list, an object containing `question_ids` or `selected_ids`, or the frozen `generalization_protocol/LONGMEMEVAL_SPLIT_MANIFEST.json` plus `--partition dev|validation|final_test`. Never generate this manifest from available predictions. Full evaluation requires an explicitly declared list of all 500 IDs.
- `--vendor`: the unchanged `official_longmemeval` directory at commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`. Both evaluator and metric script SHA256 values are checked at export and verification.

Canonical-input pilot protocols can use the canonical SHA with a declared subset selection. Future derived partition datasets must record both their own `dataset_sha256` and the canonical `canonical_parent_sha256` in their new frozen generation protocol. Do not edit an already frozen protocol merely to make an audit pass.

## Commands

Use Python 3.11+ for the bridge. Below, shell variables are non-secret absolute paths to an existing judge environment, bridge file, vendor folder, prediction file, protocol file, canonical reference, declared manifest, and a **fresh** hypothesis filename. Use `-X utf8` on Windows as shown: the unchanged upstream scripts use the process's default text encoding.

```bash
"$PY" -X utf8 "$BRIDGE" export \
  --predictions "$PREDICTIONS" --protocol "$PROTOCOL" \
  --reference "$REFERENCE" --expected-ids "$EXPECTED_IDS" \
  --vendor "$VENDOR" --method s_parent_single_2000 \
  --hypotheses "$HYP"
```

Add `--partition dev` (or the selected partition) for a split manifest. Export writes HYP and HYP.receipt.json without overwriting existing files. It emits canonical-order JSONL with only `question_id` and `hypothesis`.

With the separately authorized judge credentials configured, run the actual unchanged author scripts. Preserve the exact commands, process status, stdout, and stderr as separate attempt evidence:

```bash
"$PY" -X utf8 "$VENDOR/src/evaluation/evaluate_qa.py" \
  gpt-4o "$HYP" "$REFERENCE"

"$PY" -X utf8 "$BRIDGE" verify \
  --predictions "$PREDICTIONS" --protocol "$PROTOCOL" \
  --reference "$REFERENCE" --expected-ids "$EXPECTED_IDS" \
  --vendor "$VENDOR" --method s_parent_single_2000 \
  --hypotheses "$HYP" --results "$HYP.eval-results-gpt-4o" \
  --report "$HYP.audit.json"

"$PY" -X utf8 "$VENDOR/src/evaluation/print_qa_metrics.py" \
  "$HYP.eval-results-gpt-4o" "$REFERENCE"
```

Pass the same partition option to `verify` as to `export`. The upstream CLI key is `gpt-4o`; it resolves to `gpt-4o-2024-08-06`. The metrics script takes two arguments, not a model argument. Upstream opens its result file in write mode; never rerun an old attempt path. Export and verification do not execute these commands on your behalf.

## Meaning of the audit

Coverage must match the explicit population exactly. Duplicates, missing/extra IDs, failed generation statuses, inconsistent hypotheses, altered protocol/input hashes, modified official scripts, non-Boolean labels, and wrong requested judge snapshot all fail before an audit report is written. A genuinely completed empty hypothesis is retained even if the official judge labels it correct; no custom answer rule overrides the upstream label.

Reports give exact correct/count and fractional accuracy overall and per type. Six-type macro accuracy is null unless all six types are represented. Abstention accuracy is null when no abstention rows exist. Abstention follows the upstream `_abs` substring test and remains in its original question type. Any proper subset is explicitly `diagnostic_subset`, including a fully completed DEV/validation/final partition; it is never presented as the full 500-question score.

Input/output and upstream hashes bind the audit to its artifacts. A valid result schema alone cannot prove that the official script actually ran: retained execution commands and stdout/stderr are separate required evidence. Upstream does not retain raw judge text, actual returned model ID, or API token usage, so those are unavailable rather than zero. Paired bootstrap and cost reporting are deliberately outside this bridge.

## Validation

```powershell
& 'D:\MemoryData\migration_20260910\official_eval_env\Scripts\python.exe' -m unittest discover -s 'D:\MemoryData\migration_20260910\official_eval_tools' -p 'test_*.py' -v
```

21 synthetic tests pass. They exercise preservation, coverage, protocol and source changes, label types/models, partial groups, overwrite protection, concurrent input/reference mutation, and numeric agreement with the actual unchanged `print_qa_metrics.py`. The tests do not request judge judgments or inspect real target answers.
