# LangMem native sessions v5

This isolated version restores native handling of length-stopped or empty required-tool responses. It returns the original SDK response unchanged to LangChain and Trustcall, records the warning, and lets the unchanged native implementation decide what to return. It does not repair responses or add retries. A completed session may have zero new writes; these warnings are not evidence of recovered, complete, or lossless memory construction.

## Why the response gate changed

The v4 gate raised on the sixth physical response after five completed sessions. It then marked all subsequent memory calls fatal before HTTP dispatch. Pinned Trustcall `_ExtractUpdates.invoke` catches ordinary exceptions and produces a HumanMessage; `validate_or_retry` routes that directly back to extraction. Its three-attempt bound is checked only after validation, so this exception route can reach the unchanged graph recursion limit without another HTTP request. A CPU regression reproduces 25 native update invocations with one physical SDK call.

The actual response was separately parsed offline under the pinned SDK stack: it yielded an AIMessage with empty content, zero valid tool calls, zero invalid tool calls, and finish_reason=length. The recorded raw response SHA256 is `687a5ab13be8cbf91d581d563a1b603eba0573b89cd95a51adda26760cbea1f7` (server receipt: `autonomy_langmem_v5/offline_sdk_parse.json`). No model request was used for that parse. Native teardown accepts an empty AIMessage; validation has no tool work and no retry work. LangMem retains existing memories and can finish with zero puts. The native three-attempt patch path for actual schema validation errors remains unchanged.

## Fixed conditions

- Pinned LangMem 0.0.30 / commit `9d033b47d9ce53e37e92c92241b0496c0278932e`; Trustcall 0.0.39. All upstream files and source manifest are byte-identical to v4.
- Qwen3.5-9B, original tools/prompts, one full original-role session per invocation, query_limit=5, inserts enabled, deletes disabled, max_steps=1, native graph defaults and native validation retries unchanged.
- The already disclosed memory-only local-serving adaptation remains fixed: output cap8192; SDK transport retries0; temperature0.7, top_p0.8, top_k20, min_p0, presence_penalty1.5, repetition_penalty1.0. These are not native LangMem extraction defaults or score-selected settings.
- QA, embedding behavior, canonical LongMemEval-S population500, source-only field filtering, and official judging requirements are unchanged.
- Transport errors, content-filtered responses, malformed envelopes, request policy violations, unhandled/terminal native failures, and blank QA remain fatal. Native handled PatchDoc drops remain recorded.

## Evidence and interpretation

Each response warning records its call ordinal, session index, reason flags, and original parsed/raw HTTP response artifact hashes. A response with both length and empty tools counts as one warned response with two reason flags. `native_degradation.json` records warned-response counts, length/empty counts, affected-session indices/count, and original handled PatchDoc drops. Per-session journals retain actual native put counts and warning counts. Artifacts are mandatory in the existing completion seal and native cache verification; missing warning evidence, inconsistent accounting, or unsealed referenced responses prevents reuse. Success and failure attempts both retain the warning ledger.

This version has a separate protocol/cache identity. Existing v4 outcomes remain v4 and must not be silently relabelled or combined with v5. The intended next step is one fixed complete-source smoke under this policy, followed by canonical full500 only if valid; no repeated score-based selection or recursion-limit increase is part of this implementation. Generation completion remains distinct from official LongMemEval judging.

CPU tests include actual native empty-AI handling, the historical exception-loop regression, unchanged native validation bounds, continued next-session calls after a warned response, sync/async raw-response preservation, fatal transport/content-filter behavior, QA invariance, source-only canonical inputs, complete500 bookkeeping, and mandatory sealed warning evidence. The local environment uses OpenAI1.35.1 for mocked raw transport tests; deployment must rerun under pinned OpenAI2.54.0. No model/GPU/network calls were made during implementation.
