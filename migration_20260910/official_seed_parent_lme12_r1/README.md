# Seed-parent DEV12 official-evaluation input

The `seed_parent_single_2000` Qwen3.5-9B candidate has 12 completed hypotheses exported through the unchanged approved bridge. Independent ordinal string comparisons confirm that all 12 preserve both source `hypothesis` and `prediction` exactly, including whitespace. All 12 also match the earlier completed Seed answers.

This is the previously exposed LongMemEval-S DEV12 subset, not the full 500-question benchmark. Diagnostic token F1 is 70.7872%; official GPT-4o accuracy is **pending**. No judge/API calls were made. `EXPORT_STATUS.json` and the bridge receipt bind the actual protocol, predictions, expected IDs, canonical cleaned-S data, and unchanged official evaluator hashes. Actual export argv, stdout, stderr, PID and exit code 0 are retained alongside them.

The new reader-stage measurements are 0.026305M LLM tokens, 1.703430M native embedding tokens, and 7.511066 seconds per question. These are incremental reader costs: source construction and utility scoring were imported and are not free. The inherited costs remain in the original pilot's cost evidence.

Collected source artifacts and reports: `D:/MemoryData/migration_20260910/pilot_runs/seed_parent_lme12_qwen35_minilm_r1/collected`. Its adjacent `collection_r1/COLLECTION_VERIFIED.json` verifies all ten files (525,738 bytes) against the remote manifest.

The unchanged official CLI must use the `gpt-4o` alias (which resolves to `gpt-4o-2024-08-06`) and the full canonical cleaned-S reference. Any actual judge execution must retain its own stdout, stderr and exit code; export is not evaluation. `PLANNED_OFFICIAL_EVAL_COMMANDS.json` contains commands that have **not** been executed.