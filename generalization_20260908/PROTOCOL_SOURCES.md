# Frozen transfer protocol: sources and audit

Audited 2026-09-08. Source-method discovery identified Qwen3-8B, while the user's requested source description named Qwen3.5 9B. Preserve that distinction: current-source frozen transfer is a provisional assumption until the intended source is confirmed.

## Official data and leakage boundary

- [Official LongMemEval repository](https://github.com/xiaowu0162/LongMemEval): use September 2025 cleaned S, all 500 questions, full timestamped histories and both speaker roles. Oracle and capped histories are diagnostics.
- [Official S file and SHA256](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/blob/main/longmemeval_s_cleaned.json): local file D:\MemoryData\MemoryData\datasets\LongMemEval\longmemeval_s_cleaned.json matches SHA256 d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442, 277,383,467 bytes. No acquisition needed.
- Local metadata audit: 500 unique questions; user 70, assistant 56, preference 30, multi-session 133, temporal 133, knowledge-update 78; 30 abstention questions; 23,867 sessions, 38–62 per question. There are 896 annotated evidence turns.
- All 948 gold session IDs start answer_. Give the model neutral session numbers; original IDs belong only in evaluation metadata. The adapter owner also found duplicate raw session IDs in 13 questions: preserve each occurrence and never key the history by raw ID.
- 29 abstention IDs share a base question ID with an answerable item. For paired intervals, cluster by question_id.removesuffix('_abs'): 471 clusters.

The [paper, section 3.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf) defines chronological session ingestion followed by the question/date. Ingest must not receive the question, gold, question type, abstention marker, has_answer or evidence IDs. Retrieval/QA may receive the question/date. Preserve assistant information and full history; the source-frozen retrieval output budget remains permitted.

## Crossed experiment and statistics

Freeze method code, prompts, thresholds, normalization, budgets, model revisions and seeds before target scores. Minimum design: baseline/refined × source/other-family model × LoCoMo/LongMemEval-S (eight arms; reuse an existing compatible source pair). Full model transfer replaces all LLM roles; changing only the reader with Qwen-built memory is a separate reader-transfer test. Match QA prompt, decoding, evidence budget and judge within comparisons. Target-score-informed changes form a new method version and consume the test set as development data.

Report paired deltas, wins/losses/ties, paired cluster bootstrap 95% intervals, per-type/abstention outcomes, failure/coverage counts, token cost, latency and memory footprint. An interval spanning zero leaves transfer inconclusive. Use conversation clusters for LoCoMo. Prespecify multiplicity correction if claiming significance across several transfer cells.

## Official judging

- [Official evaluator](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py): one user-message rubric, gpt-4o-2024-08-06, temperature 0, n=1, max_tokens=10. Temporal permits off-by-one time-unit errors. Preference, updates and abstention have distinct rubrics. The wrapper reproduces the exact strings with stricter complete yes/no parsing.
- [Official aggregation](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/print_qa_metrics.py): original six-type scores, six-type macro, micro, separate abstention. Abstention retains its original type in six-type aggregation.
- [Official retrieval definitions](https://github.com/xiaowu0162/LongMemEval/blob/main/src/retrieval/eval_utils.py): distinguish recall-all@k from recall-any@k. For turn recall use actual has_answer; labelling every turn of a gold session inflates evidence coverage.
- Missing predictions and judge infrastructure failures must be visible. Judge failures are pending, never automatic wrong answers. The wrapper emits official overall/macro only with complete dataset coverage; conditional partial accuracy is explicitly named accuracy_on_judged.

## Local hazards and wrapper runtime

certmem/certmem/longmemeval.py:26 drops abstention; line 39 invents turn evidence from whole gold sessions. MemoryData/benchmark/longmemeval/loader.py:86 exposes raw session IDs and optionally caps max_context_chunks. certmem/scripts/get_longmemeval.sh:8 uses the old HuggingFace repository and may rename old data as cleaned. Existing MemoryData/evaluation/longmemeval/longmemeval_judge.py adds a JSON system prompt, skips generation failures and reclassifies abstention, so it is not exact official rubric/aggregation.

judge_longmemeval.py uses standard Python only. Configure OPENAI_BASE_URL (API root ending /v1) and OPENAI_API_KEY; credentials and endpoint are never printed. The requested and returned model identities must match the pinned judge. No API calls were made during implementation. Per-item cache keys bind prediction, question, gold, original type, exact prompt, fixed model/sampling settings and endpoint identity. Invalid or mismatched cache stops spending and leaves pending items for retry.

Run:
    python judge_longmemeval.py --predictions ARM.jsonl --dataset longmemeval_s_cleaned.json --output ARM.judged.jsonl

Outputs include ARM.judged.jsonl.summary.json and atomic item cache in ARM.judged.jsonl.cache/. --cache-only recomputes coverage without model calls; --limit N bounds new calls while reading existing cache. Pilot results remain incomplete against the full dataset even when all provided predictions are judged. Never present pilot conditional scores as official 500-item scores.

Verification: all five local prompt template strings were compared byte-for-byte with string constants extracted from the current official evaluator AST. Offline checks passed for invalid verdict rejection, cache-key changes for prediction/gold/endpoint, invalid cache contents, original-type abstention grouping, incomplete denominators, judge failure remaining pending, and successful resume without a repeated API call. Judge calls were mocked; only the public source file was fetched.

2026-09-08 judge availability check: the [official GPT-4o API model page](https://developers.openai.com/api/docs/models/gpt-4o) still lists the pinned gpt-4o-2024-08-06 snapshot and Chat Completions endpoint. This verifies the published catalog, not account access or a successful paid request. No judge API calls have been made; scoped local and remote OPENAI_API_KEY/OPENAI_BASE_URL presence checks remained false at07:11UTC. The exact benchmark judge/rubric pin remains unchanged.