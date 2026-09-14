# Intended Qwen3.5 baseline and Gemma generalization

User correction: Qwen3.5-9B was the intended baseline throughout; the earlier Qwen3-8B execution was a mistake. Qwen3-8B artifacts remain historical development provenance only. No further 8B inference or backend bridge is scheduled. Gemma is the independent-family generalization model. Mistral is excluded.

| Role | Model | Locked revision |
|---|---|---|
| Intended primary model | Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| Model generalization | google/gemma-4-E4B-it | ee0ef6023621cff504d758262d4e04895a5af4a2 |

The method was selected during the accidental 8B development run and is now frozen for revalidation. The primary Qwen3.5 experiment constructs its own seed/r40/refined memories and uses Qwen3.5 for writer/scorer/reader; 8B-generated contexts cannot establish the intended full-method baseline.

Queue: installed-runtime verification and GPU smoke → Qwen3.5 full-role LoCoMo507 → Gemma full-role LoCoMo507 → each model fixed LongMemEval12 feasibility. Predeclared LoCoMo primary population377 and secondary audit100 are saved in ../locomo_transfer/evaluation_populations.json. The three conversation histories were previously exposed. Full500 evaluation remains required and is prepared separately in ../full_transfer; no pilot result completes that requirement.

Weights/config/tokenizers are pinned and verified before publishing per-model environment files. Isolated runtime pins vLLM0.19.1, torch2.10.0+cu129, Transformers5.5.3, tokenizers0.22.2, huggingface-hub1.6.0, sentence-transformers5.2.0. CPU imports do not establish CUDA compatibility: actual fresh generation and answer-token likelihood GPU smoke gates inference. The original environment stays intact.

Runtime initialization adds language_model_only=True; frozen generate/encode/NLL implementations remain inherited. Construction rules, source probes, utility thresholds, knapsack, Qwen3-Embedding-0.6B retrieval, read2048/add2000/answer96/temperature0/nonthinking stay fixed. Actual code/package/model provenance is locked by each run.

LongMemEval official judge still needs authorized API configuration. F1 is diagnostic only. Qwen3.5 LoCoMo is complete and audited: primary377 F1 seed54.095/r4055.687/refined56.305, refined−r40+0.618pp CI[-0.091,+1.664]. Gemma LoCoMo is also complete: primary377 r4050.558/refined50.741,+0.183pp CI[-2.197,+1.499]; audit100−1.473pp/full507−0.118pp. See LOCOMO_MODEL_TRANSFER_RESULTS.md. Qwen3.5 LME12 preparation is active. Full500queue is registered and waiting for modernqueue successful completion; no full500plan/inference yet.

Official sources: https://huggingface.co/Qwen/Qwen3.5-9B ; https://huggingface.co/google/gemma-4-E4B-it ; https://github.com/vllm-project/vllm/releases/tag/v0.19.1 .

Compatibility correction: Transformers5.5.3 defaults chat-template output to BatchEncoding. chat_tokenizer_compat.py restores the Transformers4 default return_dict=False at the adapter boundary, preserving rendered prompts, integer token IDs and answer-token spans. Protocols record this bridge and hash its code; original source methods remain unchanged. The initialQwen35GPU smoke caught this before benchmark inference.
