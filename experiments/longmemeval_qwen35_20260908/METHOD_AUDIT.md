# LongMemEval five-baseline method audit

The user excluded LightMem direct storage and HiGMem no-profile/no-link before any baseline run on2026-09-08. The remaining methods use the existing implementation under a controlled final reader. This is not an exact reproduction of every original end-to-end paper setting.

| Method | Memory construction retained | Retrieval retained | Existing implementation detail to report |
|---|---|---|---|
| E-Mem | Text-memory block construction and native memory handler | Native block query and result aggregation | Internal block reasoning and aggregation consume additional model calls; count them separately from the final reader. |
| SimpleMem | Native MemoryBuilder and final flush | Native retrieval planning and reflection | Final AnswerGenerator is replaced by the common reader. |
| Mem0 | infer=True fact extraction with native ADD/UPDATE/DELETE | Native memory.search | Existing strict mode can request at most two corrections for invalid update/delete IDs; no non-strict two-fact truncation. |
| LangMem | Native memory store manager | Native store search | Requires native tool calling support; verify that in the actual service smoke. |
| A-MEM | Note analysis, linking and evolution | Native linked-neighbor retrieval, including original order and duplicates | Existing strict evolution validation can request at most two semantic corrections. |

## Conditions changed for the comparison

- Generative model: pinned Qwen3.5-9B BF16; embedding: pinned Qwen3-Embedding-0.6B,1024 dimensions. These are experimental backbone choices.
- All generative calls: temperature0 and thinking disabled. Native internal output caps are preserved.
- Final reader: exact common prompt,2048 Qwen-token memory budget including separators, whole-unit packing that skips oversized units and continues,96 output-token cap. Save retrieved candidates and the selected context so packing effects can be audited.
- The question and question date become available after memory construction/finalization. Gold answers, question types and evidence annotations are excluded from construction.
- Count all500 questions per method, including failures and abstention cases. A common final-reader budget does not mean equal total compute; report native generation, retrieval and final-reader costs separately.

## Evidence boundary

Read-only Python review found no additional removal of the five methods' core memory construction/retrieval paths. Local CPU tests and Linux boundary tests passed. Full-history live integration, full500 inference and the common official judge remain outstanding. Do not describe these code tests as completed benchmark evidence or claim performance improvements before the paired scores exist.

Code anchors: native_six.py, run_history.py, metered_lme_proxy.py, source_snapshot/MemoryData/methods/e_mem/e_mem_adapter.py, methods/simplemem/source/SimpleMem/core/hybrid_retriever.py, methods/mem0/source/mem0/memory/main.py, methods/langmem/langmem_adapter.py and methods/a_mem/source/a_mem/memory_layer.py.
