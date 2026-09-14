# Frozen paper method on LongMemEval: isolated Qwen runtime adapter

This clone evaluates exactly `seed`, `r40_fused_four_turn`, and `s_parent_single_2000` from the completed Qwen3.5-9B/MiniLM baseline. It does not route through recursive v1/v2 or their baseline-win gate. Eleven transitive frozen algorithm modules and all four frozen plans are copied byte-for-byte. No existing experiment, dataset, result, model cache, or environment file was edited or copied into this clone.

Only the execution capacity is adapted: the `prepare` writer uses 65,536 total context tokens and explicit chunked prefill; the `evaluate` reader and `score` likelihood scorer remain at 8,192. Both the writer engine limit and the frozen generation preflight guard are addressed. Overflow raises before inference; there is no truncation or source-history filtering. The original generator cache key, temperature zero, generation batch size 24, writer/reader seed 20260907, scorer seed 20260908, native usage instrumentation, full singleton/pair utility scoring, and source-only construction are preserved. Reader engine options retain the original settings. The scorer retains its 512-token chunked prefill setting.

The method budgets are unchanged: 2,048 reader evidence tokens, 96 answer tokens, 2,000 extra storage tokens, and MiniLM's native 256-token embedding window. Runtime precision remains Qwen FP16 without quantization and MiniLM CPU FP32. Model pins are Qwen/Qwen3.5-9B `c202236235762e1c871ad0ccb60c8ee5ba337b9a` and sentence-transformers/all-MiniLM-L6-v2 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

`protocol.json` records explicit writer/reader/scorer capacities, execution profile, the new adapter's actual SHA-256, every copied algorithm module's actual SHA-256, model metadata, packages, and dataset hash. Its locked comparison prevents resuming an old run under the new adapter. Use a **fresh output directory and fresh caches**, and never import old generation/embedding caches: capacity deliberately remains absent from the original cache key. Native usage records include each active role's context capacity; shared cache reuse between the three arms remains observable rather than being interpreted as cold-cache savings.

## Invocation after deployment

The root task owns deployment and GPU verification. The intended clone location is `/workspace/paper_longmemeval_20260910`. An explicit `--environment` JSON must map the two model names to their verified new-host paths and pinned revisions. The script has no hardcoded old-project output or model path: dataset/output arguments are mandatory.

For example, using the installed project runtime and the new-host environment JSON:

```bash
cd /workspace/paper_longmemeval_20260910
py=/workspace/generalization_20260908/modern/.venv/bin/python
for stage in plan prepare score construct evaluate; do
  "$py" run_transfer.py "$stage" \
    --dataset longmemeval \
    --data /workspace/generalization_20260908/data/longmemeval_s_cleaned.json \
    --environment /workspace/paper_longmemeval_20260910/environment.json \
    --out /workspace/paper_longmemeval_20260910/runs/paper_frozen_lme500_qwen35_minilm_r1 || exit "$?"
done
```

Use the actual installed Python path if the migration's environment location differs. Do not run the loop in parallel with another GPU job. Omission of `--limit` includes all 500 canonical LongMemEval S questions; `--refined-only` is intentionally omitted so both paired controls are retained. `plan` is CPU-only and is an appropriate first command before loading an engine. The full run should use the root task's Supervisor queue and wall-time launcher. This clone itself does not schedule persistent jobs and contains only native token/inference-time logging. The evaluation-function timing addon remains a separately reviewed launcher.

## Capacity evidence and limits

`PREVIOUS_CAPACITY_EVIDENCE_CHECK.json` verifies that the existing complete full-500 source-only capacity audit applies to the clone's frozen request builders: five relevant source/plan hashes, the canonical 277,383,467-byte dataset hash, and the locally available pinned tokenizer files covered by the original audit match. No target answers or prior error outcomes were inspected.

The original Qwen maxima for rendered prompt plus reserved output were 19,737 for session summaries, 20,752 for source-QA probes, and 39,120 for audit requests using raw-history fallback parents. All three audited scenarios have zero overflows at 65,536. The last scenario is a concrete fallback, not a universal bound on all generated parent memories. The actual runtime continues checking every new uncached prompt and can fail rather than silently truncate.

The old `full_transfer/audit_target_capacity.py` is **not directly runnable as a preflight for this clone**: it imports an old two-model context policy and enforces an old virtualenv receipt. Do not bypass those checks or copy its runtime into this clone. The verified existing full-500 audit supplies the source-request CPU evidence; the new host still requires an actual writer engine smoke test and an uncached real request. Passing these CPU checks does not establish that the 65,536-token engine fits RTX 5090 memory, or measure GPU performance.

## Validation and provenance

Run from this clone with an ordinary CPU Python containing numpy and pytest:

```powershell
C:\Python314\python.exe -m pytest -q
```

53 tests passed locally after the final code change. They cover both writer/reader capacity boundaries including equality, pre-inference overflow rejection, unchanged scorer capacity, preserved cache keys/batching/sampling, native receipt accounting and exception identity, actual protocol/hash fields, source-copy integrity, source/target data isolation, and chat-tokenizer/NLL compatibility. The inherited generator is text-identical except for the method name, namespaced frozen helpers, and context-capacity guard.

`COPIED_SOURCE_PROVENANCE.json` lists exact original and clone hashes, distinguishing the three modified copies (`transfer_runtime.py`, `run_transfer.py`, and the overlong-prompt regression fixture in `test_refined_runtime.py`) from 20 untouched copies. `test_capacity_adapter.py` is new. `ADAPTER_FILES.sha256` pins deployable source, tests, and these local documentation/validation records, excluding caches and itself. The original roots are verified unchanged. No SSH, GPU inference, model/data download, or remote mutation was performed by this subtask.

LongMemEval token F1 remains diagnostic. A pinned official-judge evaluation and paired uncertainty analysis are still needed before making a generalization claim or choosing a paper revision.
