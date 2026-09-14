# LangMem sessions v3: bounded local memory serving

This version is a labelled LongMemEval integration of authentic LangMem and Trustcall with a fixed local-serving adaptation. It is not the uncapped v2 protocol or an official LangMem LongMemEval paper reproduction. It preserves all 38 pinned upstream files and the v2 handled-PatchDoc-loss policy. All prior candidates and failed attempts remain separate.

## Declared serving change

Memory ChatOpenAI uses a fixed 8192-token output limit. Its owned synchronous and asynchronous OpenAI clients use max_retries=0. The ordinary 600-second transport timeout is unchanged. This operational budget was declared before this smoke and is neither an upstream memory-extraction default nor a budget selected from evaluation accuracy.

The benchmark's memory temperature0, tool choice, tools/schemas, prompts, original-role full-session inputs, inserts/updates, deletes-disabled policy, query_limit5, max_steps1 and native Trustcall max_attempts3 are unchanged. The entire native memory manager shares one ChatOpenAI object; the runner verifies object identity before configuration, and the actual pinned invoke/ainvoke functions construct their extractor after this configuration. The observer asserts the actual outgoing memory cap and SDK retry count.

QA and embedding clients and request parameters are unchanged. SDK transport retries are distinct from native Trustcall's original semantic validation/repair attempts. The runner adds no repair prompt, decoder change, source truncation, alternate algorithm, model request, or guessed answer.

## Retained responses and failure gate

Memory request JSON and response JSON are stored under each attempt's memory_calls directory with hashes in llm_calls.jsonl. When LangChain requests a raw SDK response wrapper, the observer additionally retains the exact underlying HTTP request-body and response-body bytes. It never serializes HTTP headers, client credentials or URLs containing credentials. The original raw wrapper is returned unchanged; its native parse method is not called, replaced or reimplemented by the observer.

The observer rejects length/content_filter completions and empty required-tool responses before LangChain/Trustcall parse them. Malformed or missing response objects and transport failures are fatal too. A separate failure ledger and per-session check prevent native caught exceptions from becoming successful histories, and subsequent memory requests are refused after a gate failure. Every returned bounded response is retained before its gate, including malformed raw JSON. No streaming memory request is accepted.

Original handled _ExtractUpdates._teardown PatchDoc drops at pinned log lines788/797/823/829 are still counted once and retained in mandatory sealed native_degradation.json. They preserve the original native returned results. Other terminal failures and blank QA remain fatal. Successful processing does not establish lossless memory construction.

## Provenance and evaluation population

LangMem commit9d033b47d9ce53e37e92c92241b0496c0278932e/version0.0.30; Trustcall0.0.39. Source_manifest.json and all38 upstream files are byte-identical to v2. Qwen/Qwen3.5-9B and MiniLM are unchanged. The external runtime receipt must verify actual FP16 Qwen weights, thinking off, client versions, CPU FP32 MiniLM and native256-token embedding behavior.

The canonical LongMemEval-S dataset SHA256 is d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442. Construction excludes question, gold answer, question_type and has_answer. Full500 retains all500 IDs; smoke must process all53 original sessions and550 turns of e47becba plus its final QA. Warning-bearing histories are not dropped. Any unresolved failure is incomplete generation, not a reduced denominator.

Use a fresh runs/langmem_native_sessions_v3 directory and newly bound runtime receipt. The first bounded smoke should run once. If it hits the gate, diagnose the retained response before any new version or retry; do not silently raise the cap or repeat failed histories. A successful smoke can be reused only under the identical source/protocol seal when expanding to full500.

Official LongMemEval judging remains a separate required stage with the pinned official evaluator. CPU checks produce no model accuracy or claim of live completion.

## Local checks

Run python -B -X utf8 -m unittest discover -s official_recovery/langmem_native_sessions_v3 -p test_runner.py -v.

Tests cover unchanged full-session mapping and gold-label exclusion, all native handled-loss branches and original repair behavior, 500-ID cache expansion, transport/fatal/blank QA, sync/async output gates, exact request snapshots without headers, raw SDK wrapper byte retention and original parse identity, fatal malformed JSON, unchanged QA clients/kwargs, and actual pinned manager constructors/invoke methods retaining and binding the configured model. No test performs a model API request; HTTPX MockTransport is used for the real SDK wrapper proof. The local SDK differs from the pinned server package, so the parent must repeat the wrapper/runtime proof in the pinned client environment before launch.
