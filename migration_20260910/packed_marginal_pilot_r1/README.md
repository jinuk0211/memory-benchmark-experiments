# Packed marginal source-only feasibility pilot

This isolated runner prepares one source history each from historical LoCoMo and already-exposed LongMemEval DEV12. It does not load benchmark QA, run an official benchmark judge, estimate generalization, or establish novelty. The original paper methods, source scores, runtime implementations, and completed results remain unchanged.

The prior recursive-v2 code already separates original source question Q0, independent fit view QA, and diagnostic view QB. Its routing utility was a retrieval-rank proxy. This pilot instead measures actual answer-token log likelihood after the unchanged retrieval and packing procedure. It reuses saved source construction and utility scores; it does not rebuild or rescore that historical source pool.

## Pre-inference clarification of the root preregistration

The binding parent preregistration is `../PACKED_MARGINAL_FEASIBILITY_PREREG.md`, SHA256 `eeb9ad58b563d27735989bbedda8bebb21b37beebc128ca74e27d3ee485f06ff`. These implementation details are declared before this pilot's GPU inference:

- Select the first viable history under SHA256(`packed-marginal-pilot-v1|dataset|history|id`). Within it, hash-order source-admitted fit IDs (maximum 16), original source-audit IDs (maximum four), and QB IDs (maximum four of the chosen fit IDs). No target answers or outcomes determine this choice.
- Q0 remains the stored candidate index. QA and QB are generated from original source only, with the original source answer held fixed. Both undergo raw-source equivalence, answerability and fidelity checks before construction, solely for paired-view eligibility. QB retrieval, packed-memory likelihood and generated-answer evaluation occur only after **both histories' final memories** are locked.
- There were no verified saved paired-view bundles in the scoped recovered source runs. Generate them once through the pinned Runtime; an identical existing bundle in this new attempt is verified and reused. There is no Q0 fallback. If either history has fewer than eight accepted pairs, preserve the bundle and rejection/quality/count information, emit `CALIBRATION_STATUS.json: insufficient_evidence`, and stop before scoring or memory construction. No `COMPLETE.json` is written. No substitute history or query is chosen on this outcome.
- Retain original source-fit admission: full-source F1 at least 0.8, full-versus-empty mean answer logprob advantage at least 0.1, source-dependent status, and a realized positive singleton/full option. Parent mapping preserves whole Seed payload units and existing option identity; pair candidates are excluded. A changed-context candidate must also retain mapped Q0 payload-versus-empty advantage at least 0.1.
- Each candidate value is its own independent QA mean answer-logprob change on `Seed + option` versus `Seed`, using actual packed contexts. Higher logprob is better; values are token means per answer, then macro means across fixed query IDs. Equal context bytes imply exactly zero value, without additional scoring. Nonpositive values are excluded. The unchanged multiple-choice allocator chooses at most one option per Q0 under 2,000 extra storage tokens, including key and payload.
- Re-evaluate the complete selected set over every accepted QA. Adoption requires strictly positive whole-set QA macro mean logprob change **and** nondecreasing original source-audit macro mean logprob **and F1**. QA F1 is recorded but is not an additional adoption gate. Empty selection, failed whole-set utility, or source-audit rejection returns the exact Seed memory. No second selection round is allowed.
- Effective packing exposure in this minimal pilot means **changed packed-context bytes**, with actual context, source IDs and token count retained. The unchanged retriever does not expose packed unit indices. A changed context alone does not prove that the alias itself was packed; adding an item can also alter BM25 IDF/ranks. No alias-inclusion claim is permitted from this receipt.
- The globally locked memory is evaluated on the preselected QB IDs that survived source-only eligibility; invalid IDs are not replaced. Zero QB gives count zero and null likelihood/F1. Negative or unchanged QB is reported as lack of support to expand this hypothesis; it cannot change the already locked memory or trigger retuning against these QB answers.

The fixed runtime is Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, FP16, MiniLM-L6-v2 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU FP32, native maximum 256 embedding tokens, RRF60/top120, 2,048 evidence tokens and 96 answer tokens. The source view writer Runtime uses seed **20260907** and 8,192 context capacity. The source Scorer uses seed **20260908** and 8,192 capacity. These distinct existing seeds are not unified.

## Files and interfaces

- `pilot.py plan`: metadata-only, verifies exact imported protocol pins, complete original three-arm memory lock, source-generation-to-probe reconstruction, all utility IDs/rows, six pinned tokenizer files, and DEV12 identity. It freezes derived candidates and `PLAN_LOCK.json`. No GPU engine is initialized.
- `pilot.py calibrate`: a separate process initializes only the approved Runtime. It produces paired source views and `CALIBRATION_STATUS.json`.
- `pilot.py score`: a separate process initializes one approved Scorer plus a CPU-only MiniLM Index that inherits the unchanged Runtime.encode implementation. It scores options, applies whole-set/source-audit gates, freezes both memories, and then evaluates QB.
- `import_source.py` is an exact byte copy of the approved helper from `../seed_parent_ablation`; canonical source files are not edited.
- `source_views_v2_pilot.py` copies the approved source-view helper, with only the two-line zero-record exception removed so an empty result still preserves rejection/quality evidence. Its admission/generation logic is otherwise identical. The original helper is untouched. Runtime view validation remains the approved original validator for nonempty bundles, plus pilot-specific count/cue/eligibility checks.
- All imported helper files and source artifact bytes are recorded in protocol provenance. The local copies' actual imported `__file__` paths are checked. Protocol, views, construction receipts and memory files are bound before QB. Changed files abort rather than being silently overwritten.

`Execution` holds the four runtime dependencies; `score_options`, `score_history`, and `postlock_qb` expose the phase boundaries for independent CPU testing. There is no benchmark reader endpoint in this runner.

## Launch

Use the existing approved remote Python environment. Launch each stage as a separate process so the writer GPU engine is released before the Scorer is initialized. Substitute verified paths into the following command; do not transplant the CPU-preview protocol to GPU execution.

```bash
python pilot.py plan --out /workspace/packed_marginal_pilot_r1/run_r1 \
  --adapter-root /workspace/paper_longmemeval_20260910_partition_v3 \
  --v2-root /workspace/local_workspace_20260910/experiments/recursive_minilm_20260909 \
  --environment /workspace/migration_20260910/paper_environment.json \
  --locomo-source /workspace/recursive_minilm_20260909/runs/qwen35_baseline \
  --lme-source /workspace/paper_longmemeval_20260910_date_v2/runs/paper_frozen_lme12_qwen35_minilm_date_v2_r1 \
  --lme-expected-ids /workspace/migration_20260910/PILOT_EXPECTED_IDS.json \
  --prereg /workspace/migration_20260910/PACKED_MARGINAL_FEASIBILITY_PREREG.md
python pilot.py calibrate --out /workspace/packed_marginal_pilot_r1/run_r1
python pilot.py score --out /workspace/packed_marginal_pilot_r1/run_r1
```

`calibrate`/`score` exit normally with `insufficient_evidence` if paired views are too few; a zero process exit alone is not proof of a completed mechanism evaluation. `COMPLETE.json` means the source-only pipeline finished, not that it improved or passed an official benchmark. Preserve stage stdout/stderr and exit codes. A repeated stage uses new-attempt caches and verifies frozen artifacts; do not edit a frozen protocol to resume with changed code.

A separate `cpu_preview_r1` can hold local tokenizer-only metadata validation. It is not a runnable GPU protocol: its model path has tokenizer files without weights. On deployment, create a fresh remote plan using real model/source paths and compare history, fit/audit/QB IDs and candidate semantic digests with the preview. This path-only relocation does not authorize candidate changes.

## Native cost scope and expected size

Report incremental native costs from these three unmodified receipt roots:

- `<out>/calibration/runtime/calibrate`: view generation and raw-source calibration; seed 20260907.
- `<out>/scoring/runtime/score`: mapped-control/packed-context NLL and source-answer generation, including postlock QB; seed 20260908.
- `<out>/indexing/runtime/index`: actual CPU MiniLM encode invocations; no LLM engine.

Wrapper requests and native token consumption are different quantities. Imported Seed/source-probe/source-utility preparation is inherited and must be reported separately, never as zero total construction cost. No historical runtime cache is imported. `timings/<stage>_<start_ns>.json` gives successful invocation elapsed seconds, including cache replay; it is not benchmark seconds per question.

For 32 attempted fit probes, view calibration has at most 224 generation requests and 20,480 completion tokens (384 writer + two 16-token equivalence + two 96-token source answers + two 16-token fidelity checks per probe). Before mapped filtering, there are at most 69 singleton/full options in the selected sources. A conservative logical NLL request count is 303 before cache reuse, with at most one generated stub token per NLL request. QA/audit/QB answer generation has at most 96 requests and 9,216 completion tokens. Prompt tokens dominate and depend on the actual source windows; roughly 0.6–1.5 million new LLM tokens is an estimate, not a measurement or enforced cap. Native receipts determine actual consumption.

The inherited embedding cache keys the entire ordered text list. Each new Seed-plus-option list can re-encode the full Seed, even if almost all entries overlap. Several million native CPU embedding tokens may therefore be consumed by this two-history pilot. Do not label embedding cost zero or infer it from the 69 short aliases alone. No reliable elapsed/GPU-time estimate is claimed before execution.

CPU validation: `python -B -m unittest discover -s . -p test_pilot.py -v`. The independent reviewer owns `independent_*` and `INDEPENDENT_REVIEW.*`. Author tests cover source admission, hash-stable selection, unchanged/negative utility, finite values, fixed macro denominator, generation length/count, input tampering, and insufficient-evidence behavior. They are not measurements of model performance.
