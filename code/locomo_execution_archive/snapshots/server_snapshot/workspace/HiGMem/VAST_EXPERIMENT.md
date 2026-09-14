# Vast.ai HiGMem LoCoMo experiment

## Target and current settings

- Instance: root@ssh2.vast.ai, SSH port 24241.
- GPU: NVIDIA RTX 5000 Ada Generation, 32 GB.
- Model: Qwen/Qwen3.5-9B, revision c202236235762e1c871ad0ccb60c8ee5ba337b9a.
- Precision: FP16 (vLLM dtype float16), no quantization. User explicitly requires FP16 for comparison.
- Thinking: disabled, temperature 0. Existing comparison code in MemoryData/utils/agent.py and A-Mem/SimpleMem adapters disables thinking too; user was asked to clarify if a different comparison is intended.
- Model output is not artificially capped by the experiment client; incomplete server generations are rejected.
- Server context limit: 32768; inputs are not silently truncated.
- Dataset: included data/locomo10.json, all 10 conversations / 5882 turns; categories 1-4 = exactly 1540 QA.
- HiGMem paper settings: no profiles, event metadata mode, no immediate links, k_event=10, k_turn=10, MiniLM embeddings.
- Embeddings run on CPU; GPU reserved for inference.
- Runtime: vLLM 0.28.0 with FlashAttention 2, CUDA graphs, chunked prefill, prefix caching, up to 48 sequences.
- Parallelism: 10 independent conversation builders; each conversation's turn order preserved. 8 QA workers sharing finalized memory snapshots; 4 internal retrieval workers per QA.
- Official LoCoMo category-aware F1 scorer downloaded from snap-research/locomo/task_eval/evaluation.py. Hash recorded in experiment manifest.

## Access and services

Local SSH identity already available: C:/Users/tgc04/.ssh/vast_codex_ed25519.

```powershell
ssh -i C:\Users\tgc04\.ssh\vast_codex_ed25519 -p 24241 root@ssh2.vast.ai -L 8080:localhost:8080 -L 18080:localhost:18080
```

8080 is the existing Jupyter service. vLLM binds privately at 127.0.0.1:18080.
Remote code: /workspace/HiGMem.

```bash
supervisorctl status higmem-vllm higmem-eval
cd /workspace/HiGMem
/venv/main/bin/python status_vast.py
tail -20 eval.log
tail -20 vllm.log
```

After an actual terminal failure, inspect logs and checkpoint compatibility before restarting:
```bash
supervisorctl start higmem-eval
```

Do not start a second evaluator while the first is alive. Do not change run_vast.py/core/prompts during a run; hashes protect against incompatible resume.

## Outputs

Remote /workspace/HiGMem/vast_run/:
- manifest.json: dataset, source, scorer hashes, server settings, installed versions, model revision.
- conv-*.pkl: memory snapshots every 20 turns and upon construction completion.
- usage.jsonl: every server-returned input/output/total token count, phase, question identity, status and timing.
- predictions.jsonl: completed answers and retrieved evidence IDs, skipped on resume.
- logs/: upstream memory/retrieval traces.
- report.json: full-count completeness check, official overall/category F1, construction and QA token totals.
- scored_predictions.json: per-question official scores.

The final report is written when the evaluation process finishes. For partial score snapshots, run report-only only when no writer is active: journal recovery may repair an incomplete trailing record.

Token counts include measured retries and repeated construction after resume. Network failures with no response usage are explicitly unmetered. GPU rental dollars require the instance billing rate; token counts are not dollar costs.

## Validation and tuning history

- Five unit tests cover concurrent journal writes, partial-tail recovery, corrupt-middle rejection, missing newline handling, and token accounting for truncated responses.
- Synthetic two-turn memory/retrieval/answer test uses a Toyota question and the official scorer.
- BF16 short-request benchmark: 31.79 output tokens/s at concurrency 1, 215.22 at 8, 375.92 at 16.
- FP8 comparison: 48.69, 313.53, 540.30 respectively. Its synthetic relevance filter dropped the answer evidence; this is one observation, not a statistical quality claim.
- BF16/FP8 measurements are tuning-only and do not count as LoCoMo results. FP8 is NOT used for the experiment.
- FP16 measured throughput: 31.69 output tokens/s at concurrency 1, 215.72 at 8, 376.25 at 16 (16 short requests per setting).
- The FP16 synthetic QA also failed: upstream final relevance filtering omits the speaker, then rejected the first-person Toyota evidence. This disproves attribution of that failure specifically to FP8. The original HiGMem filtering algorithm is preserved for comparison; this semantic limitation is recorded rather than silently repaired.
- Full evaluation started 2026-09-06 14:08 UTC (23:08 KST), Supervisor program higmem-eval, initial PID 3597. The original 10 conversations are being ingested; QA follows each completed conversation.
- First live check: 58 model requests, 24,177 input tokens, 7,151 output tokens, no request errors or truncations. This is progress, not a completed benchmark score.

Do not declare this goal complete until all 1540 distinct expected QA IDs are scored, token logs are audited, report and artifacts are copied locally, and final measured results are reported.
