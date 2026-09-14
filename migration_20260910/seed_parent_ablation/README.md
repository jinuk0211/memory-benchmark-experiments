# Seed-parent source-utility ablation

This is a parent-choice ablation, not a new contribution or a validated replacement method. It applies the unchanged `portable_parent.augment` policy to the original Seed memory instead of the r40 fused parent. All Seed units, their order, text, keys, and source IDs remain unchanged; new routed units are appended. The frozen Qwen3.5-9B FP16 reader, MiniLM CPU FP32 native 256-token embedding, BM25+dense RRF (60, top 120), stable packing, 2,048-token read cap, 96-token answer cap, and 2,000 additional key-plus-payload storage cap remain intact.

Only two explicitly pinned source attempts can be imported: the completed LoCoMo Qwen3.5-9B/MiniLM baseline and the completed LongMemEval-S 12-example development pilot. These examples have already been exposed; results cannot establish untouched-test generalization. The candidate does not access validation/final partition outcomes. LongMemEval token F1 is diagnostic only; the unchanged official LongMemEval judge and metric scripts must supply the primary accuracy separately.

The planning stage checks the entire selected source history against the pinned canonical dataset. Construction is a separate invocation that rejects `--data` and receives only imported source artifacts. The source probe pool is reproduced from source generations, and the selected IDs must equal all and only `probe_fit` IDs. Every utility item must retain the original source-probe identity. All three original memory arms are copied solely so that the original complete memory lock can be reverified; construction consumes Seed only. No original target predictions, evaluation items, generation caches, embedding caches, or native meter journals are imported.

Each new run contains `imported_source/`, `import_lineage.json`, `protocol.json`, `history_verification.json`, `expected_ids.json`, `source_sessions.json`, `memories/seed_parent_single_2000/`, `construction/`, and a new `memory_lock.json`. After reading, it additionally contains fresh `cache/`, `runtime/evaluate/`, `evaluation_timing/`, `items/seed_parent_single_2000/`, and `seed_parent_single_2000.jsonl`. Protocol `source_hashes` includes the existing source files and runtime wrappers plus `import_source.py` and `approved_run_transfer.py`. `import_lineage_sha256` binds the exact lineage file. Imported file names are relative to `imported_source/` and map to exact byte counts and SHA-256 values.

Imported writer, source-probe, and source-likelihood costs must be reported separately. New native runtime costs are incremental only. Zero new inference tokens during CPU routing do not mean free memory construction. Do not feed this single-arm attempt to the existing three-arm fresh-work reporter. The separate incremental reporter must validate lineage and label inherited costs explicitly. Native reader costs still depend on cache reuse within the new attempt and are not an independent cold-cache comparison against historical arms.

`SOURCE_PROVENANCE.json` records byte-identical copied dependencies, including the existing timing launcher and recorder. The evaluator function is copied verbatim into the new `run_transfer.py` and checked against `approved_run_transfer.py`, because the existing timing launcher requires the evaluator's source location to be `run_transfer.py`. The original paper/date/partition adapters were not modified.

Use the pinned CPU tokenizer for construction. `--runtime-out` is the future absolute GPU run path, allowing the CPU artifacts to move without rewriting their protocol. Evaluation refuses a different actual output path, model metadata, tokenizer files, or installed package versions.

```text
python run_transfer.py plan --dataset locomo --data <canonical-locomo10.json> --import-run <completed-qwen35-baseline> --out <new-local-or-remote-run> --runtime-out /workspace/seed_parent_ablation_20260910/runs/seed_parent_locomo_qwen35_minilm_r2
python run_transfer.py construct --out <same-run> --tokenizer <pinned-Qwen3.5-9B-tokenizer-directory>
python metrics/timed_evaluate.py --runner run_transfer.py -- evaluate --out /workspace/seed_parent_ablation_20260910/runs/seed_parent_locomo_qwen35_minilm_r2 --data <canonical-locomo10.json> --environment <frozen-5090-environment.json>
```

For the LongMemEval-S pilot, use `--dataset longmemeval`, the complete canonical 500-example S JSON, and the pinned completed 12-example source run. The imported protocol selects the exact 12 histories; the canonical dataset is never truncated. Use a distinct future runtime output path for that attempt.

The first CPU-only run `runs/seed_parent_locomo_qwen35_minilm_r1` completed before annotation/docstring review changes. Preserve it as a superseded CPU construction record; its old new-source files are in `source_versions/r1/`. It has no new reader answers or official judge results. The final deployable code and fresh r2 CPU receipt are subject to the independent Python review.
