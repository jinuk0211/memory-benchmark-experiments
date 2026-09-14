# A-MEM paper pipeline recovery v2

This is a separate versioned candidate for the user-authorized A-MEM baseline. It preserves the failed v1 smoke, its code and its protocol. No server, service or previous result has been changed by this local candidate. Deployment requires a new runtime/source receipt and a fresh `runs/a_mem_paper_v2` path.

The v1 smoke encountered the paper's own handled evolution JSON fallback after a native 1000-token completion. Our v1 observer then rejected that history. Version 2 corrects only that added observer policy: it preserves the original returned values and records handled metadata/evolution fallbacks, while transport failures and unhandled native errors still fail. The native source, generation budget, temperature and all algorithm code remain byte-identical. See `autonomy_r1/amem_diagnosis/` for hash-bound evidence and actual-native CPU proof.

## Which original implementation

The authoritative paper-reproduction repository is [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem/tree/0c8039f28fdcc08189a23c07a3437d9d2482f9c2), pinned to commit `0c8039f28fdcc08189a23c07a3437d9d2482f9c2`. Its README identifies `test_advanced.py` as the original JSON-schema evaluation, separately from the later robust evaluation. We use the original path. `source_manifest.json` lists exact immutable URLs and SHA-256 hashes for the unchanged files under `upstream/`, including the license.

The earlier audit compared the deployed integration with **agiresearch/A-mem**, an official agent-building library. That library directs paper reproduction to WujiangXu's repository. Its Chroma retrieval, default metadata, string links and first-note behavior are not the paper pipeline. Those library differences must not be described as deviations from the paper. The paper uses cosine similarity, content plus metadata indexing, automatic metadata generation, integer links and evolution even for the first note; this candidate retains all of those behaviors.

The old vendored source first appears in our Git commit `c63391c128e33eedb91115edf689f12acf4bbc63` with the 384/640 budgets already present. This establishes when our repository recorded them, not an independently verified upstream ancestry chain.

## Actual paper differences repaired

| Area | Previous deployed integration | This candidate / pinned paper |
|---|---|---|
| Memory input | 4096-character complete-turn bundles | One call per complete turn, exact `Speaker ` + role + `says : ` + content formatting from `test_advanced.py:301-305` |
| Metadata / evolution output | 384 / 640 tokens | Original OpenAI controller's 1000 for every call |
| Query | Raw question retrieval | Original `generate_query_llm` keyword generation before retrieval |
| Retrieval | Adapter k=2 and its formatting | Original `find_related_memories_raw`, k=10; original order, links, duplicates and neighbor-loop behavior retained |
| Answer | Adapter's custom prompt, temperature 0, 60000-character cap | Original `answer_question` prompt and JSON answer, temperature 0.7, 1000 tokens; no extra context cap |
| Evolution errors | Added exact semantic checks and up to two corrective requests | Original parser and algorithm; no corrective requests. Native handled fallbacks continue with separately sealed degradation counts |

The paper's OpenAI path is selected, with the requested common Qwen3.5-9B FP16 backbone. The native `.7` temperature, 1000-token output limit, evolution k=5 and threshold=100 remain original. No location normalization, arbitrary chunk truncation, prompt rewrite, link cleanup or parameter search is introduced.

## Thin integration and explicit dataset adaptation

`runner.py` compiles unchanged AST class nodes from the exact archived source: `BaseLLMController`, `OpenAIController`, `LLMController`, `MemoryNote`, `SimpleEmbeddingRetriever`, `AgenticMemorySystem`, and `advancedMemAgent`. It does not execute the demonstration/evaluator module bodies, NLTK downloads, extra metric model loading, grading or API judge. The original class methods and prompts run directly; they are not reimplemented.

There is one explicit missing-global compatibility fix: the pinned `MemoryNote.analyze_content` uses `re.sub` at line 380 without importing `re`. The loader provides Python's standard `re` module in its globals. The archived source bytes remain unchanged. This prevents a NameError-driven default-metadata fallback; it is a disclosed compatibility correction, not an upstream claim.

LongMemEval sessions are mapped to the original per-turn interface. Canonical user/assistant roles become the paper's speaker string. Dates are converted to its native `YYYYMMDDHHmm` timestamp format. Session IDs and original dates remain in `source.json`; content is never shortened or merged. Source turns allow only `role` and `content`, excluding `has_answer` and every evaluation annotation. All 500 histories are selected from the canonical byte hash `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.

Every question uses the same generic category-1 branch with `answer=''`, irrespective of the benchmark's question type. This is necessary because the original LoCoMo category-5 branch consumes a gold answer as a multiple-choice option. No gold or question-type fields are sent to a worker. The original public question argument receives `Question date: ...\nQuestion: ...`; the original prompt template and keyword-generation method remain unchanged. This is an explicit cross-dataset adaptation, not a claim to reproduce category-specific LoCoMo scoring unchanged.

Both original LLM controllers must bind to Qwen/Qwen3.5-9B and the supplied loopback OpenAI endpoint before use. `OPENAI_API_KEY=EMPTY` and `OPENAI_BASE_URL` are set before their constructors. OpenRouter credentials are unset. All native calls pass through the metered local endpoint. MiniLM uses its original SentenceTransformer encode/cosine path, pinned local snapshot `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU FP32, 384 dimensions, native 256-token embedding window. The global embedding constructor remains pinned when original consolidation reconstructs the retriever. There is no external embedding endpoint or embedding fallback.

**Deployment gate remains external:** before any real run, the root launcher must bind these new source hashes to a newly verified runtime receipt covering Qwen/MiniLM weight hashes, FP16 serving and `enable_thinking=false`, local-only model routing, and service/GPU state. This standalone runner verifies source identity, local endpoint, configured model, MiniLM path/configuration and embedding shape; it cannot prove which weights a live endpoint serves. Existing old runner receipts do not authorize this version automatically.

## Failure and resume behavior

Each history runs in a fresh subprocess. The worker only sees source/query files. Logical LLM requests, responses, usage and finish reasons are preserved in `llm_calls.jsonl`; the physical proxy journal remains authoritative for SDK retry accounting. A non-stop finish reason is recorded as a warning and the original parser decides acceptance. Native fenced-JSON and brace-cleanup acceptance is preserved. The runner never pre-validates JSON more strictly than the original before passing it back.

A line observer matches the pinned native filename, function and branch at metadata lines 385/396 and evolution line 827. Handled fallbacks remain native outcomes and do not become harness errors. Events retain kind, native call ordinal, phase, function and line in mandatory `native_degradation.json`. Separate metadata/evolution counts are included in build, usage, prediction and failure artifacts. The successful-attempt seal and verifier require the degradation artifact. Diagnostics are saved on failure too.

Transport errors remain independently recorded by `Calls` and fatal even if native code tries to handle an exception. Propagated algorithm exceptions and malformed/blank final QA answers remain fatal. A generated result means every source turn was processed and valid QA was generated; it does not claim error-free metadata generation or evolution. No history is excluded because of a handled fallback, and no semantic repair, extra request, generation-budget increase or fabricated answer is added. The existing upstream metadata exception formatter's unbound exception variable is not repaired; its existing outer fallback remains original.

Failure attempts are immutable and are never resumed from partial memory. A later retry creates `attempt_0002` and reconstructs that history from the beginning. Successful results are reused only when the complete receipt verifies protocol/source/query identity plus all sealed artifact hashes, including memory JSON, original retriever pickle/NumPy embedding cache, native QA, calls and prediction. Never reuse caches from the previous vendor runner.

`status.json` and `predictions.json` use the existing `a_mem` native-five schema. The existing run-all completion gate and official exporter can read a complete 500-history result. A smoke subset uses the same full-population protocol identity, so a later full run can reuse its verified success. Any failure stops the candidate runner immediately and preserves an incomplete status. Official judging remains pending.

## Local verification and eventual command

CPU tests use fake clients and embeddings while executing the actual pinned native classes. They cover two complete turns (including a turn longer than 4096 characters), all model/endpoint bindings, original query rewrite and k=10 retrieval, original QA, fenced-JSON acceptance, counted native metadata/evolution fallbacks, all 550 synthetic turns followed by valid native QA, exact native request/memory parity with and without the observer, transport/unhandled errors, nonempty answers, source-label exclusion, trace restoration, failure stopping and mandatory degradation/cache tamper rejection. No real model or paid API is called by tests.

```text
python -m unittest discover -s official_recovery/a_mem_paper_v2 -p test_runner.py -v
```

After root's new runtime gate, an eventual smoke command has this form (not executed here):

```text
.venv-client/bin/python official_recovery/a_mem_paper_v2/runner.py --method a_mem --dataset longmemeval_s_cleaned.json --run-dir runs/a_mem_paper_v2 --api-base http://127.0.0.1:18083/v1 --model Qwen/Qwen3.5-9B --embedding-model /workspace/.hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --ids-file queue/smoke_ids.json
```

The full500 run is already authorized. Omit `--ids-file` after this method's full-history smoke and the matching new v2 runtime/source verification gates pass; the separate three-method GPU allocation supersedes the older all-seven gate. Keep the identical run directory and protocol to reuse a successful smoke. Native source imports need numpy, scikit-learn, sentence-transformers and openai in the client environment; their own transformer/torch dependencies are already required by that environment. The sibling native harness and frozen `source/MemoryData/utils/request_metering.py` are reused and included in protocol hashes. Exclude `__pycache__` from a deployment bundle.