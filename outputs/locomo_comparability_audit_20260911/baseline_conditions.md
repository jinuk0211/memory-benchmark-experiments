# Baseline reading and native-method comparability audit — 2026-09-11

Scope: final LoCoMo table rows Mem0 36.50, A-MEM 36.85, LangMem 29.93, SimpleMem 38.19, LightMem official 40.33, E-Mem 56.96, and full-context reference55.17. Read-only investigation; no inference rerun, no score edits. This file is a new audit artifact.

## Main finding

The table is not a fully controlled common-reader comparison. Model identity and final-reader temperature/thinking are aligned in the preserved configuration evidence, but answer prompts, effective output budgets, evidence volume, and native-to-adapter changes differ. Consequently15.20 points cannot presently be attributed exclusively to memory quality. This audit does not quantify how much of the gap these differences cause.

## Compact matrix

All rows use Qwen/Qwen3.5-9B; preservation summaries identify FP16 and no quantization. Relevant embedding baselines use all-MiniLM-L6-v2. Reader temperature0 and thinking disabled are supported by archived configurations/proxy code or native runtime. “Cap omitted” below means no explicit per-request output cap after the comparison proxy; context/server limits still apply.

| Row | Final reader and output cap | Retrieval and native-operation status | Material limitation |
|---|---|---|---|
| Mem0 | Shared LoCoMo concise/single-phrase template; requested1024 but comparison proxy removes cap | Mem0 inference/add and update machinery retained; Qdrant top100; benchmark-specific custom fact-extraction prompt replaces native default | r3 plus r5 recovery composite; one invalid update plan rejected and one repaired before mutation |
| A-MEM | Adapter-specific strict-retrieved-memory/brief-insufficiency/concise prompt; requested1000 QA cap removed by proxy | Dense top10 root notes plus linked neighbors; metadata/evolution machinery retained | Native writer budget1000 restored, with3,569 evolution JSON skips and3,575 capped memory outputs; all QA stop |
| LangMem | Shared LoCoMo concise/single-phrase template; requested1024 removed by proxy | Native create_memory_store_manager/Trustcall writes; InMemoryStore vector search top10; custom benchmark reader |1,705 native Trustcall patch-error log events; not a unique lost-fact count |
| SimpleMem | Shared benchmark reader replaces native AnswerGenerator; requested1024 removed by proxy | Planning and reflection retained inside HybridRetriever; semantic25, keyword5, structured5; adapter keeps first10 final retrieved entries; window40/overlap2 | Modified reading stage; final answers composite848 r13 +689 r17 +3 r21; historical semantic-discard instrumentation incomplete |
| LightMem official | Native speaker/date/concise-answer prompt, temp0, no explicit native reader cap | Native add_locomo/search_locomo; LLMLingua2 rate0.6, short-term768, offline update0.9, post-update combined top60, summary retrieval off | Qwen system-only compatibility adds empty user; nine completed conversations retained plus152 QA retry after Qdrant lock; not the older adapted46.60 run |
| E-Mem | Shared benchmark final reader after native querying/aggregation; requested1024 removed by proxy | Text-mode blocks, hybrid router, max5 blocks, parallel native block QA and aggregation retained | Native intermediate query8192/aggregate2048 are requested values, not final-reader cap and ordinarily stripped by proxy;2 failed QA recovered with XML guidance/repetition penalty only on retry |
| Full-context55.17 | Native reader explicitly requests shortest exact answer, digits/date format, no sentence/explanation, unknown if absent; actual32 output tokens | Raw full history, no retrieval | Different prompt/output limit from common baseline adapter;885 r1 +655 r2 continuation |

## Evidence: output budgets must distinguish writer, intermediate, and final reader

The archived common proxy sets temperature0, enable_thinking false, and chat_template_enable_thinking false, then removes max_tokens/max_completion_tokens. Reading a YAML generation_max_length1024 as the effective baseline answer cap would be wrong.

- `Archived common proxy, lines72–96` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r13/scripts/metered_openai_proxy.py:72`).
- `Archived queue launches proxy with --comparison-policy, line405` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r3/scripts/locomo_server_queue.py:405`).
- `Dataset requested generation_max_length1024, line6` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r3/comparison_config/locomo.yaml:6`).
- `A-MEM native-stage detection requires phase memory_add, lines153–156` (historical source: `amem_r26_deployment/source/scripts/repairing_openai_proxy.py:153`); `only native memory stage restores1000 after common policy, lines191–198` (historical source: `amem_r26_deployment/source/scripts/repairing_openai_proxy.py:191`).
- `Actual full-context reader call max_tokens32, line325` (historical source: `certmem_primary_deploy_r2/scripts/run_system_v15.py:325`); `native SamplingParams preserves cap and temp0, line53` (historical source: `certmem_primary_deploy_r2/certmem/llm.py:53`). The earlier archived r1 call is `line309` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/certmem_r1/scripts/run_system_v15.py:309`).
- `Full-context pinned metadata: model, revision,49152 context, reader32, lines86–88` (historical source: `certmem_primary_deploy_r2/scripts/run_certmem_full.py:86`).

These are preserved execution-source/configuration observations, not a new complete per-question request-ledger attestation. A-MEM native writer failures are additionally run-attested by its report/summary; LightMem configuration and recovery are run-manifest/receipt supported.

## Evidence: prompt differences

- `Shared LoCoMo template97–100` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r13/benchmark/memoryagentbench/prompts/benchmark_templates.py:97`): answer as concisely as possible, single phrase if possible.
- `Shared reader763–785` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r13/utils/agent.py:763`).
- `A-MEM reader273–307` (historical source: `amem_r26_deployment/source/methods/a_mem/a_mem_adapter.py:273`): strict retrieved memory, brief insufficiency and concise answer; this is a separate prompt, not the shared one.
- `Full-context reader44–48` (historical source: `certmem_primary_deploy_r2/scripts/run_read_v9.py:44`): shortest exact answer; explicit date format, digits, no sentence/explanation, missing→unknown.
- `LightMem native reader protocol10` (historical source: `lightmem_official_20260909/PROTOCOL.md:10`), `search manifest reader fields26 onward` (historical source: `upload_preparation_20260910/extracted_baseline_results/lightmem_official/manifest_search_1788918676498357933.json:26`).

Potential timestamp confound: shared reader appends Current Time from metadata.question_date or actual execution time. The fallback is source-confirmed in `759–780` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r13/utils/agent.py:759`), and Mem0 uses it at `3214` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r3/utils/agent.py:3214`). This bounded audit did not establish whether every selected LoCoMo QA metadata lacks question_date, or count the actual timestamps sent. Thus label as unresolved pending selected-request/raw-metadata inspection, not a proven cause of temporal-score loss.

## Evidence: method identity and distortions

### Mem0

- `Archived config: model/temp/top100, lines1–9` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r3/comparison_config/mem0.yaml:1`).
- `Uses inferred add, custom benchmark fact prompt, lines1565–1585` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r3/utils/agent.py:1565`). YAML null falls through to get_template, so it does not restore native extraction default.
- `Native add/search plus custom reader3142–3225` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r3/utils/agent.py:3142`).
- `Run report repaired/rejected plan counts133` (historical source: `upload_preparation_20260910/extracted_baseline_results/mem0/report.json:133`), `documented changed source174` (historical source: `upload_preparation_20260910/extracted_baseline_results/mem0/report.json:174`), `recovery protocol439` (historical source: `upload_preparation_20260910/extracted_baseline_results/mem0/report.json:439`).

### A-MEM

- `Pinned config model/temp/retrieve10` (historical source: `amem_r26_deployment/source/comparison_config/a_mem.yaml:1`); thinking disabled `38` (historical source: `amem_r26_deployment/source/comparison_config/a_mem.yaml:38`).
- `Root retrieval plus linked neighbors226–268` (historical source: `amem_r26_deployment/source/methods/a_mem/a_mem_adapter.py:226`).
- `Final run summary7–8` (historical source: `amem_final_results_r26/results_summary.md:7`), `actual metadata/evolution/length outcomes20` (historical source: `amem_final_results_r26/results_summary.md:20`).3,569/5,882 evolution calls skipped ≈60.68%. The native1000 writer cap and failure policy were deliberately restored; this is an observed compatibility/failure bottleneck, not proof that the algorithm was removed.
- `Original report counts/QA use` (historical source: `upload_preparation_20260910/extracted_baseline_results/amem/original_policy_report.json:48`). Avoid loading the entire38MB report; it includes extensive request joins.

### LangMem

- `Native memory manager setup/temperature74–82` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData/methods/langmem/langmem_adapter.py:74`).
- `Native manager.invoke and strict failure path111–126` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData/methods/langmem/langmem_adapter.py:111`).
- `Vector query with retrieve_num137–149` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData/methods/langmem/langmem_adapter.py:137`).
- `Run report269–272` (historical source: `upload_preparation_20260910/extracted_baseline_results/langmem/report.json:269`):1,705 Trustcall patch-error log events; explicitly not uniquely lost facts. Composite584 existing +956 recovered QA.

### SimpleMem

- `Archived config13–30` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r13/comparison_config/simplemem.yaml:13`).
- `Native retrieve routes through planning/reflection59–132` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r13/methods/simplemem/source/SimpleMem/core/hybrid_retriever.py:59`). Therefore claiming planning/reflection were removed would be false.
- `Default final retrieve_limit10 at1655–1657` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r13/utils/agent.py:1655`); `adapter slices final results at163–166` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r13/methods/simplemem/simplemem_adapter.py:163`).
- `Adapter benchmark QA uses common reader3250–3282` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r13/utils/agent.py:3250`), while `native system.ask calls AnswerGenerator160–163` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r13/methods/simplemem/source/SimpleMem/main.py:160`). This is a confirmed reading-stage adaptation.
- `Final selected counts/provenance1–38` (historical source: `results/locomo/simplemem/comparison_provenance.json:1`).
- `Final aggregate report513–521` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/reports/simplemem_report.json:513`) states historical semantic-discard instrumentation incomplete despite benchmark_complete true.

### LightMem official

- `Protocol3–15` (historical source: `lightmem_official_20260909/PROTOCOL.md:3`) gives upstream8449d574..., full model revision, FP16, thinking off, native add/search,0.6/768, extraction temp0.1/top_p0.1/max16000, offline update0.9, top60 reader.
- `Final status6–10` (historical source: `lightmem_official_20260909/RUN_STATUS.md:6`): Qwen system-only compatibility empty-user append; Qdrant lock recovery; no selection by score.
- `Final source selection1–14` (historical source: `upload_preparation_20260910/extracted_baseline_results/lightmem_official/selection_sources.json:1`).
- `Protocol21` (historical source: `lightmem_official_20260909/PROTOCOL.md:21`): earlier46.60 was an adapted pipeline and38.20 direct raw storage. They are not repeated trials of the same official40.33 configuration and must not be used to estimate an isolated prompt effect.

### E-Mem

- `Archived r4 adapter model/native router/query setup35–59` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r4/methods/e_mem/e_mem_adapter.py:35`).
- `Native memory block query and aggregation82–85` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r4/methods/e_mem/e_mem_adapter.py:82`).
- `Shared wrapper final reader4731–4744` (historical source: `github_upload_20260909/reproducibility_20260910/execution_archive_20260909/snapshots/server_snapshot/workspace/MemoryData-finish-r13/utils/agent.py:4731`) is additionally downstream of that native aggregation. r13 is a preserved later common-source witness; final E-Mem itself uses r4/r11, so exact byte equivalence of this region to r4 must be confirmed before calling this a full byte-level execution attestation.
- `Final report recovery364` (historical source: `upload_preparation_20260910/extracted_baseline_results/emem/report.json:364`): two saved-memory failed QA retried, with native first request unchanged, retry-only XML guidance and default repetition_penalty1.1.
- `Final report accounting limits366 onward` (historical source: `upload_preparation_20260910/extracted_baseline_results/emem/report.json:366`): an original request has unknown usage/response-delivery proof; predictions complete does not imply exact total cost/audit complete.

## Interpretation and next controlled comparison

Report these rows as completed adapted/native-configuration systems with explicit reader conditions. Do not call all rows untouched native reproductions or identical-reader memory-only comparisons. Preserve SimpleMem planning/reflection, A-MEM evolution and linked-note retrieval, and E-Mem block querying/aggregation when making controlled comparisons.

A shared final-reader rerun using saved evidence, same prompt, output cap, model settings and evaluator would isolate some reading effects without rebuilding memories. That would still not resolve writer failures or retrieval-budget differences; these require separately disclosed diagnostics or matched-budget variants. Existing failure/recovery reports justify investigation but do not establish the numerical contribution to the15.20 gap.

