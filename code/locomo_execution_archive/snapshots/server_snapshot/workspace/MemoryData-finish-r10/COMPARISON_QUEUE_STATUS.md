# LoCoMo server queue handoff

## Current state: LightMem complete; E-Mem retry running (2026-09-07 08:04 UTC)

Latest verified wait **08:06:12 UTC**: retry workers 96528-96531 remain live, no
completed conversation shards yet. Saved QA by contexts 0-3: 0/4/5/5. All **211**
observed requests have valid usage and zero failures/truncations: 161 LLM calls,
869307 generation tokens; 50 embedding calls, 115191 embedding tokens. These are
partial observations, not final benchmark totals or a throughput ETA. Full E-Mem
QA uses multiple native subqueries/tool rounds; do not shorten them for speed.

**The original queue and finalizer are EXITED. Do not restart either.** LightMem
completed all **1540 QA**, and the finalizer's unchanged full-data, exact-template,
metadata, coverage, usage, quality and auxiliary gates all passed. Official F1:
**0.4659538136054212**. Report `/workspace/locomo-comparison-finalized-20260907/lightmem/report.json`,
PC `lightmem-final-report-20260907.json`, SHA256
`67376b549c0a63ffd4a640588dc4e4db40267b2ca6723a6d4549d1e0d268ec43`.
LLM **5582404** tokens / 2098 calls, embedding **384570** / 17371 calls; all 19469
requests have valid usage, zero failures/truncation/duplicates. Wall **3262.0738 s**,
GPU sampled energy **376.3574 Wh**, mean utilization **79.5484%**, peak **29562 MiB**,
QA mean **0.4500 s**, p50 **0.4194 s**, p95 **0.6420 s**, rental estimate **$0.358761**
excluding setup/gaps/bandwidth. LLMLingua CPU FP32: 6264 forwards, 397056 attention
tokens, 3184572 padded slots. Full raw/results/meter/score archive BOTH hosts:
`lightmem-complete-20260907.tar.zst`, SHA256
`49d73a587f191d6568b3f3825b6464173c852c33f066c11439bbd630b9ad6dc5`, verified equal.

Original E-Mem failed almost immediately: vLLM rejects `tools=[]` (HTTP 400).
Its initial report `e_mem/attempt_status.json` and PC
`emem-initial-failed-attempt-20260907.json` preserve 4 failed requests WITHOUT
reported usage (unknown, not zero), 11.4143 s wall, 0.2661 Wh, $0.0012553 rental
estimate. Initial archive BOTH hosts `emem-initial-failed-20260907.tar.zst`, SHA256
`2361a8b098946222a4a61aa6681e64696920bae1b219ca631c78948e996d95d6` verified.
The existing five-attempt aggregate does NOT yet include this sixth attempt;
include its unknown usage explicitly in the final accounting deliverable.

Transport-only repair: `methods/e-mem/src/agent/base.py` now imports `NOT_GIVEN`
and sends `tools=tools if tools else NOT_GIVEN`. Only two lines changed; real
tools/prompts/generation parameters/tool rounds/error propagation are preserved.
New tests `tests/test_emem_transport.py` exercise the actual OpenAI serializer
through offline HTTPX: 2 failures before patch, all 41 focused tests passed after
patch on PC AND server. Existing Python reviewer approved source/tests, and
separately reviewed shell retry wrapper; `bash -n` passed. No vLLM restart.

BEFORE deployment, all **454** original pinned source/config/data files were
archived and independently rehashed inside the archive. BOTH-host archive
`comparison-primary-pinned-source-20260907.tar.zst`, SHA256
`4e0f03e518410ae85021080d41e277baa0847d2704f80eedbe91f9fbccaadc94` verified equal.
Proof BOTH hosts `comparison-primary-source-backup-audit-20260907.json`.
Original base.py SHA256 `5927d3a0b36233f4d6d650a6cfbe0dc8846bfb72fee874b705131260763c7cd6`;
patched SHA256 `78f795c809eec077fe9457fac404533e1db825b54219c1677f4c82e3c267528e`.
IMPORTANT: the working source now intentionally differs from the OLD primary
plan at this one file. Reproduce that old plan using its archived source; do NOT
rewrite the old plan or restore base.py while the retry is running.

NEW active Supervisor **locomo-emem-transport-retry**, shell PID **96491**, child
queue **96492**, vLLM still **8551**. Wrapper `scripts/run_emem_transport_retry.sh`
runs E-Mem-only fresh plan then existing exact-template finalizer; it returns zero
only if the final summary actually marks E-Mem complete. It rejects reused output.
Plan `/workspace/MemoryData/comparison_config/plan-emem-tools.json` has **457** pins
and passed actual validation. Run ID `locomo-qwen35-fp16-20260907-emem-tools`.
Output `/workspace/locomo-comparison-qwen35-fp16-emem-tools`; final output
`/workspace/locomo-emem-tools-finalized-20260907`. Log
`/workspace/MemoryData/emem_transport_retry.log`. Config
`/etc/supervisor/conf.d/locomo-emem-transport-retry.conf`.
Qwen/MiniLM revisions, FP16, non-thinking policy, native E-Mem text settings, client
and server environments, four-context protocol and pricing remain unchanged.

At **08:03:38 UTC**, E-Mem contexts 0-3 were live, workers **96528-96531**.
Its first **40** generation calls all returned HTTP 200 with valid usage and no
truncation (230784 reported tokens); QA had started. This proves the transport
repair is effective, NOT that the full E-Mem benchmark has completed. Continue
to its actual final outcome, audit/back up, then assemble all-seven-method results
with explicitly excluded methods and failed-attempt overhead. Goal remains active.
SimpleMem retry remains STOPPED/excluded; never restart it under the old plan.

## Current state: LightMem 1226 QA audited (2026-09-07 07:47 UTC)

This section supersedes runtime counts below. Live queue **13095**, vLLM **8551**,
and waiting finalizer **37693** remain unchanged. LightMem contexts 0-7 completed
**1226/1540 QA**; only contexts 8 and 9 remain, live workers **85779** and **86358**.
At 07:47:30 UTC their input progress was 485/509 and 429/568 turns. No restart,
precision reduction or source modification was made. E-Mem is still next.

All 1226 completed QA passed exact-template normalization, original metadata,
unique/nonempty/full-subset predictions and metered generation coverage. Their
15736 requests have valid usage, zero failed/truncated calls: **4433520** LLM
tokens and **308636** embedding tokens. Auxiliary validation passed for all eight
complete conversations: 5109 CPU FP32 forwards, 317636 attention-mask tokens,
2597899 padded token slots. This remains a partial audit, NOT a full score.
Proof PC `lightmem-contexts-00-07-audit-20260907.json`; server directory
`/workspace/lightmem-contexts-00-07-audit-20260907`. Self-contained archive on BOTH
hosts `lightmem-contexts-00-07-20260907.tar.zst`, SHA256
`38e5007129c8e1f61b8486496a423c19a8877843e5377803482cdd1d9a517499` verified equal.

New standalone Mem0 full-facts failed-attempt report exists on both hosts:
`mem0-fullfacts-failed-attempt-20260907.json`. It was regenerated from the drained
raw journals/telemetry and revalidated against the unchanged raw archive hash.
The initial archive itself was not changed. Note there is no Mem0
`attempt_status.json` inside that original attempt directory; use this sidecar.

`failed-attempt-overhead-20260907.json` on both hosts aggregates exactly five
completed failed/invalidated attempts (A-MEM serial, Mem0 capped, Mem0 full-facts,
LangMem, SimpleMem initial): **16023148** LLM tokens, **1395120** embedding tokens,
**17418268** total recorded tokens. It excludes successful methods, preflights,
setup and gaps; source-report SHA256 values are recorded. Known rental estimates
sum to $0.6284522, but full cost is UNKNOWN because two attempts lack wall time.
Known energy sums to 720.5351 Wh, but full energy is UNKNOWN for three attempts.
HiGMem accounting sidecar is now also copied to PC `higmem_accounting.json`; its
10 unmetered failed calls and full-run embedding/energy gaps remain explicit.

## Current state: native SimpleMem retry excluded (2026-09-07 07:33 UTC)

Latest read-only snapshot **07:35 UTC**: LightMem contexts 0-5 completed, **885/1540
QA**, contexts 6-9 active. All 14536 observed requests had reported valid usage,
zero failed/truncated calls. Snapshot token totals (not final): 3412106 LLM and
286058 embedding. The separate offline audit/backups below cover contexts 0-3.
Machine-readable retry exclusion record: both hosts
`simplemem-retry-exclusion-20260907.json`. Original finalizer remains waiting for
the original queue, and will independently score fully completed methods.

**Do not restart the SimpleMem retry.** A further zero-API-call native two-batch
storage test proved a separate source-level problem: after a second `add_entries`,
the row exists and semantic/symbolic searches find it, but native Tantivy keyword
search does not. `_fts_initialized` prevents rebuilding the index after later
batches. No source/index changes were made to hide or repair this native behavior.
Proof on server and PC: `simplemem-native-append-fts-check-20260907.json`; scratch
`/workspace/simplemem-append-fts-en4ledhx`. This is separate from the repaired
dependency import/API failure and prevents claiming a clean three-view retrieval
comparison with this pinned baseline. User explicitly allowed excluding broken
baselines; this retry is therefore excluded, not counted as completed or zero-cost.

The exact waiting retry PID 80604 was verified to have no children and no retry
output, while primary LightMem was still running. Only Supervisor
`locomo-simplemem-compat-retry` was stopped. It is now **STOPPED**, with
`autostart=false` and `autorestart=false`; inference retry never began. Queue
**13095**, vLLM **8551**, and original finalizer **37693** remain running unchanged.
LightMem then E-Mem are still permitted live work. The isolated environment, plan,
tests and stopped config are retained as reproducible diagnostics, not a pending
reservation. The historical reservation paragraph below is superseded.

LightMem contexts 0-3 (584 QA) passed exact question template, original metadata,
nonempty/unique/full-subset predictions and metered generation coverage. All 6920
associated requests have valid usage, no failures or truncation: **2108507** LLM
tokens and **137586** embedding tokens. Unchanged auxiliary validation on the exact
four-conversation subset also passed: **2216** LLMLingua CPU FP32 forward calls,
**142412** attention-mask tokens, **1126737** padded slots. This remains partial,
not a full LightMem score. PC audit `lightmem-contexts-00-03-audit-20260907.json`;
server audit directory `/workspace/lightmem-contexts-00-03-audit-20260907`.
Self-contained archive on BOTH hosts `lightmem-contexts-00-03-20260907.tar.zst`,
SHA256 `d270f7cbe76303e9f069fac09dc5024e5b5e57fa7a1a14b3b2f1f885d98878eb` verified.

Initial SimpleMem failed attempt audited and backed up: 4 LLM calls / **44836**
tokens, 1 embedding call / **1234** tokens, all 5 usages reported, 3 interrupted
client requests and zero model truncations; no completed QA. Wall **85.1197 s**,
sampled energy **8.3879 Wh**, rental estimate **$0.00936141** excluding setup/gaps/
bandwidth. `simplemem/attempt_status.json` plus PC
`simplemem-initial-failed-attempt-20260907.json` record this failed-attempt overhead.
Archive on BOTH hosts `simplemem-initial-failed-20260907.tar.zst`, SHA256
`8cb715aa580dc8090c2432139fbb2bdd2caf5a8d4286bed4fbae74d1240260e5` verified.

## Current state: SimpleMem retry registered (2026-09-07 07:27 UTC)

Supervisor `locomo-simplemem-compat-retry` PID **80604** is RUNNING and explicitly
waiting for the original queue and its artifact writers to exit. It will then run
ONLY SimpleMem in `/workspace/simplemem-compat-venv` and independently finalize its
score/accounting. This is a real server-side reservation, not benchmark completion.
Original queue **13095**, vLLM **8551**, finalizer **37693** are unchanged.
LightMem contexts 0-3 have exited 0 (584 QA expected; offline audit pending), with
contexts 4-7 active. E-Mem remains next in the original queue.

Retry plan: `/workspace/MemoryData/comparison_config/plan-simplemem-compatfts.json`.
Retry output: `/workspace/locomo-comparison-qwen35-fp16-simplemem-compatfts`.
Retry final output: `/workspace/locomo-simplemem-compatfts-finalized-20260907`.
Neither output existed at registration. Supervisor config:
`/etc/supervisor/conf.d/locomo-simplemem-compat-retry.conf`; log:
`/workspace/MemoryData/simplemem_compat_retry.log`.

New script `scripts/retry_simplemem_after_queue.py` SHA256
`2298816dff2767af5d363df7036158f8618aaff019fbf8436b695c3a01be591a`.
Existing read-only Python reviewer found no blockers; all 40 wrapper/finalizer
tests passed locally and on the server. Actual retry `--check-only` passed, as did
the unchanged primary queue plan. Wrapper checks fresh/nonoverlapping outputs,
identical experiment settings, unchanged plans after waiting, and free ports.
It returns success only for a genuinely complete SimpleMem final report.

IMPORTANT: the original plan contains **454** file pins, not 364. An earlier
tool-output truncation corrupted a derived draft; validation caught it before
registration. The draft was replaced from the complete local original (SHA256
`8e4274832016fa973fa35afe062c794fbeb41316d8f1490a521a559084b0d424`, verified equal
to the server original). The final retry plan preserves all 454 original pins and
adds 5 new pins, **459 total**. No primary plan or active inference files changed.

Next: audit/archive the initial SimpleMem failed-attempt overhead; audit/back up
completed LightMem contexts; continue to actual original/retry final outcomes.
Do not stop at reservation success or count failed/partial attempts as full scores.

## Current state: LightMem running (2026-09-07 07:11 UTC)

Queue **13095**, vLLM **8551** and waiting finalizer **37693** remain unchanged.
LightMem context 1 (conv-30) completed 81 QA and exited 0; this partial result has
not yet received the additional offline audit. Active contexts are 0, 2, 3, 4,
worker PIDs **57498**, **57500**, **57501**, **66489**, helper **57497**. Verify live
PIDs before intervention. The earlier 07:03 snapshot had 20 generation and 1511
embedding calls; those are not final/current token totals.
E-Mem remains next. A-MEM, Mem0 and LangMem have no valid complete benchmark score.

SimpleMem's initial attempt failed after inserting its first 40 entries because
installed LanceDB **0.38.0** removed the local Tantivy FTS API used by the original
method (`use_tantivy=True`, `tokenizer_name="en_stem"`). Strict mode correctly
raised instead of silently dropping lexical search. Tantivy is **0.26.0**, PyArrow
**25.0.1**, all imported from `/venv/main`. The queue automatically continued to
LightMem. This is a dependency-compatibility failure, not a demonstrated model
failure. Investigate a compatible LanceDB/Tantivy version in a NEW isolated client
environment, preserving native Tantivy behavior; do not change the active shared
venv or pinned source. A compatible environment is now installed and verified at
`/workspace/simplemem-compat-venv`, but NO retry is registered yet. It owns original
upstream pins: LanceDB 0.25.3, PyArrow 22.0.0, pylance 0.39.0, lance-namespace and
lance-namespace-urllib3-client 0.0.21. Its `comparison_base.pth` adds the existing
client then main site-packages, preserving Transformers 4.57.6 / tokenizers 0.22.2
/ sentence-transformers 5.7.0 / Torch 2.13.0. Tantivy 0.26.0 is reused.

The actual vendored `VectorStore` passed strict-mode Tantivy creation, lexical,
semantic and symbolic search using synthetic data and stub embeddings, zero model
API calls. Proof is `/workspace/simplemem-fts-compat-smoke-20260907.json` and the
same-named PC file; scratch `/workspace/simplemem-fts-smoke-3_iio2xx` remains.
The shared client still imports LanceDB 0.38.0 / PyArrow 25.0.1 from main, and the
original hash-pinned queue `--check-only` still passes. No inference source changed.

Next: prepare a NEW dedicated SimpleMem retry plan/output, using this isolated
client, same Qwen/MiniLM revisions and four-context protocol. Register a separate
Supervisor job that waits for the ORIGINAL queue to EXIT and all its writers to
stop, runs the retry, then scores it with the exact-template finalizer in a separate
output. Do not let two queues contend for proxy/embedding ports. Preserve and count
the initial SimpleMem failed-attempt overhead separately. Any new Python wrapper
must receive the existing Python review workflow. Original upstream requirements
are under local
`methods/simplemem/upstream/requirements.txt`, not the vendored `source/SimpleMem`.

## Completed controlled LangMem skip (2026-09-07 06:57 UTC)

LangMem context 4 / conv-43 stalled at turn 484, then its request returned
`finish_reasons=["length"]`, `comparison_incomplete_output`: 3549 prompt + 29219
completion = the full 32768-token context, after 389.4125 seconds. The existing
strict quality gate cannot pass this attempt. Under the user's permission to skip
broken baselines, preserve it as a failed attempt and let the queue continue to
SimpleMem; do not reduce output/context limits or alter answers to make it pass.

The four clients were paused to prevent further retries while outstanding requests
finished. BOTH meters were then verified `status=ok`, `in_flight=0`, `drained=true`.
After exact PID/parent/path/state revalidation, SIGINT then SIGCONT ended the four
clients. All old client PIDs **27788, 35040, 43126, 49431** and helper **14103** are
gone; DO NOT SIGNAL THEM AGAIN. The unchanged queue preserved merged journals and
telemetry and started SimpleMem, then LightMem after SimpleMem's separate failure.

The full LangMem failed attempt contains 14708 metered requests, no missing/invalid
usage or duplicate IDs, two failed/truncated requests. Observed generation tokens:
**14887552** (13475573 prompt, 1411979 completion) across 4246 calls; embedding:
**1370893** tokens across 10462 calls. Wall time **5491.1547 s**, sampled GPU energy
**712.1472 Wh**, 1075 samples / zero missing, mean utilization **96.7553%**, peak
VRAM **28482 MiB**. Estimated instance cost **$0.603914** at the saved hourly rate,
excluding bandwidth/setup/gaps; this is failed-attempt overhead, not invoice cost.
`langmem/attempt_status.json` records the reason, control sequence and accounting;
local copy: `D:\MemoryData\langmem-failed-attempt-20260907.json`.

Full archive on PC and server: `langmem-failed-20260907.tar.zst`, SHA256
`6bde7bb7367bf288f722c8312343eb52dbd879a26ce3d204c1a3f742f2154841`, verified equal.
All 584 completed QA remain valid partial artifacts, NOT a completed 1540-QA score.

## Historical pre-skip state (2026-09-07 06:37 UTC, contexts 0-3 complete)

This section supersedes older runtime snapshots below. Do not repeat the local
PC shutdown or attempt to operate on the already-destroyed old instance.

- Fresh Vast inventory contains only running RTX 5090 instance `50118771`.
  Old RTX 5000 Ada `50064241` is already destroyed. Both the migration archive
  and completed HiGMem archive were rehashed on the PC and replacement server;
  their SHA256 values still match the recorded values below. No further instance
  deletion is needed or authorized for the replacement while results depend on it.
- Queue PID **13095** and vLLM PID **8551** are unchanged and healthy. LangMem
  contexts 0, 1, 2 and 3 have completed with 152, 81, 152 and 199 QA respectively
  (584/1540 generated, not a final LangMem score). Active contexts are 4, 5, 6
  and 7, worker PIDs **27788**, **35040**, **43126**, **49431**, helper PID
  **14103**. All five were revalidated live after the new worker started.
- An offline audit of all 584 completed QA passed original metadata, exact pinned
  query-template rendering, prediction uniqueness/completeness and generation
  coverage checks; no duplicate, missing or empty predictions. All 8889 associated
  model/embedding requests have valid reported usage, with no duplicates, API
  failures or truncation. This subset used 9049410 generation tokens and 796245
  embedding tokens. Native Trustcall logs contain 528 patch-error events across
  these four contexts; these are not uniquely lost facts or API failures. Proof:
  `D:\MemoryData\langmem-contexts-00-03-audit-20260907.json`, server directory
  `/workspace/locomo-audit-contexts-00-03-20260907`. Raw result hashes are unchanged.
- A self-contained backup of contexts 0-3, their audit, filtered completed-context
  meter rows and pinned plan exists locally and remotely as
  `langmem-contexts-00-03-complete-20260907.tar.zst`, SHA256
  `018bfbbc44077f5f5e438a185195fe29f818dbd4b825dd8bd5aed924da9bc2a7`.
  Both hashes match; neither raw answers nor shared live journals were modified.
  Earlier 0-2 audit/archive remains unchanged; its archive SHA256 is
  `fdd9c4269dc766314f903fc713a8efb5b60f56474e23512dd197cfe9bd3871c8`.
  At 06:37 UTC GPU utilization was 99%, VRAM 28482 MiB and free disk still 19 GB.
- At 06:15 UTC, all four active worker PIDs were revalidated live. Completed
  contexts 0/1 (233 QA), their worker artifacts, the conv-30 audit and pinned plan
  are backed up on the server and PC as
  `langmem-contexts-00-01-complete-20260907.tar.zst`, SHA256
  `5614fd9dabe5e9a20af612876d0c1149d9ea92be338b90480bc6631faa358a78`.
  Both archive hashes match. Shared live usage journals were deliberately not
  included in this partial archive; preserve them for the final metering report.
- The benchmark saves `query` as the exact template-wrapped question, while the
  existing queue's canonical metadata validator expects the raw question. Thus
  completed raw QA would be falsely rejected by that final queue gate. No running
  pinned source, inference setting, or raw answer was changed to address this.
- A separate postprocessor, `scripts/finalize_locomo_comparison.py`, accepts only
  an exact rendering of the pinned benchmark template, writes a separate normalized
  copy, and applies the original metadata/coverage/token/truncation/auxiliary gates.
  It records raw source hashes and native LangMem Trustcall patch-error log counts.
  All 88 focused tests passed locally and remotely; independent Python review found
  no blockers (all 15 finalizer-specific tests passed). A real completed conv-30
  audit validated all 81 original QA and their metering without changing raw data:
  `D:\MemoryData\langmem-conv30-audit-20260907.json` and
  `/workspace/locomo-audit-conv30-20260907/audit.json`. These are partial audits,
  not a claim that LangMem has completed all 1540 QA.
- Supervisor **locomo-comparison-finalizer**, PID **37693**, is registered and
  RUNNING, waiting for the existing queue to exit and all artifact writers to stop.
  Config: `/etc/supervisor/conf.d/locomo-comparison-finalizer.conf`; log:
  `/workspace/MemoryData/comparison_finalizer.log`. Its future output is
  `/workspace/locomo-comparison-finalized-20260907`; the directory correctly does
  not exist while the queue is live. A live check detected all seven current
  helper/worker/proxy writers. No queue or inference process was restarted.
- Allow the existing queue to run the remaining methods even if its final metadata
  gate labels a fully generated method failed due to the wrapper. The separate
  finalizer will re-audit those artifacts after queue termination, without repeating
  inference. Genuine failed/skipped methods remain explicitly excluded. Neither
  finalizer nor queue completion alone authorizes destruction of the replacement.

## Active comparison on RTX 5090 (2026-09-07, updated 05:33 UTC)

This section supersedes every older runtime/PID, server address, and shutdown
instruction below. Do not repeat the already-executed local PC shutdown.

- Old RTX 5000 Ada instance `50064241` is destroyed. Fresh Vast inventory shows
  only replacement RTX 5090 instance `50118771`. Verified result backups remain
  both on this PC and the replacement; never destroy the replacement while the
  comparisons/results depend on it. SSH: `ssh4.vast.ai:38771`, identity
  `C:\Users\tgc04\.ssh\vast_codex_ed25519`. Free container space: 19 GB.
- HiGMem has completed 1540/1540 QA, official F1 `0.4033310246213628`. Its completed
  archive and observed token/accounting limitations are recorded below.
  At 05:41 UTC, running the comparison scorer against the same saved HiGMem
  predictions reproduced F1 `0.40333102462136283` (floating-point rounding only),
  with all 1540 original QA validated and zero missing/empty predictions. Category
  counts are 282/321/96/841. This read-only cross-check added no model calls; proof
  is saved locally as `D:\MemoryData\higmem-cross-scorer-audit-20260907.json`.
- Active Supervisor `locomo-comparison-queue` PID **13095**, plan
  `/workspace/MemoryData/comparison_config/plan-parallel4-fullfacts.json`, output
  `/workspace/locomo-comparison-qwen35-fp16-parallel4-fullfacts`. Run ID:
  `locomo-qwen35-fp16-20260907-parallel4-fullfacts`. It is running **LangMem**, followed
  by SimpleMem, LightMem and E-Mem, with explicit failures retained.
  A-MEM is explicitly skipped under the user's permission to skip broken methods.
- `comparison-vllm` remains PID **8551**; do not restart this healthy server.
  Qwen3.5-9B FP16 and the pinned Qwen/MiniLM revisions are unchanged. Four isolated
  conversation workers use native continuous batching; context turns and QA stay
  in their original within-conversation order. The 10 original contexts contain
  exactly 1540 selected QA. At 05:24 UTC four requests were active, generation
  throughput was 292 tokens/s, and prefix-cache hit rate was 54.3%. These are live
  server observations, not a controlled end-to-end speedup benchmark.
- `scripts/run_locomo_parallel.py` writes separate worker state/journals and only
  merges completed original-context results. The queue validates original QA
  coverage and scoring/accounting gates. Parallel phase-duration sums can exceed
  elapsed wall time; cost uses wall time, not summed overlapping phase durations.
- Mem0's pre-existing two-fact cap was observed discarding extracted facts. Strict
  mode now retains all facts; live logs confirm 4-7 facts retained in early turns.
  Strict Mem0/SimpleMem/E-Mem parsing, storage and retrieval failures now propagate
  rather than becoming empty/degraded outputs. Native successful-method settings
  and non-comparison behavior are preserved. All 73 focused regression tests pass
  both locally and on the server; independent Python review found no blockers.
  New hash-pinned plan verification and a real zero-API Mem0 initialization passed.
- The corrected full-fact Mem0 attempt then failed at context 2, turn 11: Qwen
  emitted an UPDATE for nonexistent memory ID `19`. Strict mode propagated this
  rather than dropping that update; the queue recorded the failure and continued
  to LangMem as authorized. It produced no valid QA score. Its 104 model requests
  contain 158836 observed tokens (121533 prompt, 37303 completion); three requests
  were unsuccessful during the fail-fast shutdown. Raw usage and telemetry remain
  in the fullfacts run's `mem0/` directory. Wall time was 137.996 seconds; 25 GPU
  samples include one missing sample, so full energy and phase durations are null,
  not zero. A subsequent audit found all 418 model/embedding requests have reported
  usage, including the three unsuccessful client-side requests. All 104 model
  requests finished with `stop`; 314 embedding calls used 4027 tokens. Thus the
  observed failed-attempt total is 158836 model + 4027 embedding tokens, not a valid
  completed benchmark. Its archive exists both locally and on the replacement as
  `mem0-fullfacts-failed-20260907.tar.zst`, SHA256
  `9d9e721aeee3dc12a454102f55542e8445b60c3bd4ce74019493360791bb92aa`.
- At 05:29-05:33 UTC, LangMem helper PID **14103** and its four worker PIDs
  **14104-14107** are live under queue PID 13095. A 191-call snapshot showed all
  model requests successful with reported usage and `tool_calls` finish reason,
  606841 model tokens; generation throughput remained 280-292 tokens/s with GPU
  utilization 100%. These are partial snapshots, not final totals or a QA score.
- LangMem's native Trustcall dependency logs invalid `PatchDoc` operations; code
  inspection of `/venv/main/lib/python3.12/site-packages/trustcall/_base.py` found
  paths that omit failed patches while continuing. At 05:32 UTC the four worker
  logs contained 89 `Could not apply patch` events. This is native model/method
  behavior, not an added speed/precision shortcut, and is not by itself a stopped
  benchmark. Preserve/count these error events alongside any final score; they
  are not a measured number of uniquely lost facts. Do not claim zero semantic
  memory-update failures from HTTP/token gates. The explicit strict adapter guard
  still propagates terminal graph recursion errors. No running LangMem source or
  native retry/memory settings were changed after this observation.
- The superseded capped Mem0 attempt is preserved at
  `/workspace/locomo-comparison-qwen35-fp16-parallel4/mem0`, with attempt status JSON.
  Its 527 generation calls used 640126 tokens; 622 embedding calls used 7770 tokens.
  There were no truncated/failed/unmetered requests, but this is NOT a valid scored
  run because of the fact cap. Both proxies were drained before stopping the old
  queue and terminating exact frozen worker PIDs. The archive exists locally and
  remotely as `mem0-capped-attempt-20260907.tar.zst`, SHA256
  `473ff3b1026635e2b69112bc57fb4d5e86c78d01b416b814fe7f2cb81c211988`.
- A-MEM's excluded serial attempt remains at
  `/workspace/locomo-comparison-qwen35-fp16/a_mem`: 291798 generation tokens,
  11196 embedding tokens, four output truncations, seven unsuccessful requests,
  no valid QA score. Its archive SHA256 is
  `7a0ad96f4007b564ad36751a08e4106fdcff19ab9ab49a7c064832862c286014`.
  Full GPU energy for both interrupted attempts is unavailable because the old
  queues held samples in memory until normal method completion. Count their
  observed tokens separately as attempt overhead, not successful-method totals.

## Recovery and HiGMem completion (historical runtime as of 04:42 UTC)

This section supersedes all older server addresses, deletion restrictions and
historical runtime state below. Do not operate on the destroyed old instance.

- Replacement instance `50118771` is reachable at `ssh4.vast.ai:38771` and
  provides one RTX 5090 (32 GB), CUDA toolkit 12.8, driver 580.159.03 and a
  50 GB container disk. The measured Vast total rate is
  `0.3959259259259259 USD/hour` before usage-priced bandwidth.
- Old instance `50064241` was destroyed through the authenticated Vast console
  after the user's conditional authorization was satisfied. A fresh Vast CLI
  listing confirms only replacement `50118771` remains. The old instance disk
  is permanently deleted; the verified migration archive remains both locally
  at `D:\MemoryData\locomo-recovery-50064241.tar.zst` and on the new server at
  `/workspace/locomo-recovery-50064241.tar.zst`.
- A source-side zstd archive copied `HiGMem`, `MemoryData`, checkpoints,
  predictions, usage journals and queue artifacts to the replacement. SHA256
  matched on old server, local relay and new server:
  `3298989dfa3c7a61a824771326f75f15e34872e02876ef883e459a074f4c3a25`.
- Replacement verification found exactly 1417 prediction rows, 38632 usage
  rows and all ten checkpoint pickle files. The active recovery runner SHA256
  is `3e1bf7c7e027d0964efc36c54d9b8302accf0c74387390afcd752101d77532bf`.
- Qwen revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, MiniLM revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, and LLMLingua revision
  `5f0c82792b7ea14c6484e015b6a072009496b7f2` were downloaded afresh on the
  replacement. vLLM 0.28.0 / Torch 2.13.0+cu130 serves the model in float16,
  without weight or KV quantization. FlashAttention v2, prefix caching,
  chunked prefill and continuous batching are active. Blackwell sampler
  compatibility uses `VLLM_USE_FLASHINFER_SAMPLER=0`; attention stays enabled.
- Recovery passed the previously stuck affiliation calls. The schema no longer
  uses xgrammar's unsupported `uniqueItems` keyword; existing maxItems/string
  bounds and the 4096-token affiliation-only limit remain. This is a recorded
  recovery change, not a claim of bit-identical continuation.
- The migrated client environment retained interpreter symlinks but lacked
  `pyvenv.cfg`, causing its installer to target `/venv/main`. The incomplete
  directory is preserved as `/workspace/comparison-venv.incomplete-20260907`.
  A real client venv now owns Transformers 4.57.6, tokenizers 0.22.2,
  sentence-transformers 5.7.0 and huggingface-hub 0.36.2; `/venv/main` was
  restored to 5.16.1 / 0.23.2 / 6.0.1 / 1.29.0 respectively. Actual module
  paths and vLLM CLI import passed. The installer now checks venv identity.
- All six client initializers passed in isolated scratch directories under
  `/workspace/preflight-5090` with native `Qwen2TokenizerFast` and zero model
  API calls. The preflight check now unwraps the known HF adapter instead of
  falsely rejecting it; the Python review found no blocking issues.
- `locomo-comparison-queue` is running under Supervisor as PID 8543 and
  observed RUNNING in a separate SSH connection. Its valid, hash-pinned plan
  contains all six methods, each with the exact 1540 selected QA. It waits for
  full HiGMem answer/accounting verification, then switches serving configs
  and runs each baseline while recording tokens, timing and GPU metrics.
- HiGMem completed all 1540 selected QA at 04:34 UTC. The final scoring import
  exposed missing `bert-score==0.3.13` and `matplotlib`; both were installed,
  then `run_vast.py --report-only` scored existing answers without generation.
  Official F1 is **0.4033310246213628**. Recorded construction tokens are
  23,985,831 and QA tokens are 14,528,999 (combined 38,514,830). Ten failed
  requests lack usage; zero successful requests were truncated. These counts
  are observed totals, not exact all-attempt totals. Full-run HiGMem embedding
  tokens, energy and cumulative wall time were not captured from its start.
- The live queue's full original-QA, metadata and accounting audit passed.
  Its stale status was preserved as `queue_status.before-scorer-install-20260907.json`
  before restarting. It automatically stopped HiGMem vLLM and started
  `comparison-vllm` PID 8551 with the exact Qwen revision/FP16 configuration.
  At 04:41 UTC warmup finished, the queue started A-MEM's first of ten
  conversations, and real model calls returned HTTP 200. The queue status is
  `running` / `a_mem`, with the other five methods queued in the agreed order.
  Individual baseline failures are recorded and the remaining methods continue.
- The completed HiGMem artifact is `/workspace/higmem-complete-20260907.tar.zst`,
  SHA256 `11fa68d1d57260d7411c930969d2974920db395ffbc022e3e989de359b36590c`.
  It contains all saved answers, checkpoint/log/usage state, final reports,
  runner, serving config and official evaluator. The earlier migration backup
  remains separate. The completed archive was also downloaded to
  `D:\MemoryData\higmem-complete-20260907.tar.zst` and its hash matched;
  `D:\MemoryData\higmem-final-report-20260907.json` contains the final scores.
  The queue log contains historical copied tracebacks:
  use current Supervisor state and new output files to determine liveness.

## Current authorization and shutdown condition

The user wants HiGMem followed by A-MEM, Mem0, LangMem, SimpleMem,
LightMem and E-Mem, then clarified that this Windows PC should shut down
**as soon as the independent server queue is registered and confirmed**.
Do not wait for all experiment results before shutting down the PC.
Do not stop or destroy the Vast instance. Do not force-close unsaved apps.
The user also authorized skipping any failed baseline and continuing with
the remaining methods; failed methods must not be reported as completed.

The server is `ssh2.vast.ai:24241`, instance `50064241`.
SSH identity file: `C:\Users\tgc04\.ssh\vast_codex_ed25519`.
Read `/etc/vast-agents-guide.md` before operating it.

## Source deployment approved and completed

After the source archive SCP was rejected twice, the user explicitly approved
the source/config transfer to `ssh2.vast.ai:24241`. The approved v2 archive
was uploaded and extracted into `/workspace/MemoryData`.

Deployed archive:
`D:\MemoryData\comparison_staging\source-20260907-queue-v2.tar.gz`
(443 source/config files, 703212 bytes, SHA256
`099321d4cc2359e1a04310bc6c98758b64eeddac6fff51291d77409b323096c5`).
It includes the reviewed venv-symlink, recovered-connection accounting,
LightMem source-alignment and executor error-propagation fixes. The older
v1 archive is stale and must not be deployed.

The separate earlier approval blocker for downloading remote checkpoints,
conversation-derived results and usage logs to this PC remains in force.
Source deployment approval does not authorize result export.

## Verified server state / prepared environment

### Recovery observed on 2026-09-07 at 01:17 UTC

- The overnight run actually exited at 19:47 UTC with 1417/1540 QA and
  5767/5882 checkpointed turns. `conv-44` stopped at turn 560/675 after
  three construction requests each timed out at 600 seconds. Its 123 QA
  were not run. The partial official F1 was 0.3979963461551367, NOT final.
- The queue exited at 19:48 UTC on its HiGMem completeness gate; none of
  the six baselines had started. A stale `waiting_for_higmem` JSON was not
  proof of a live process. Supervisor terminal status and traceback proved it.
- The original vLLM PID 3004 remained healthy with zero running/waiting
  requests. Local and deployed HiGMem runner SHA256 matched. The runner
  restores checkpoints and skips saved QA while appending usage; a plain
  supervised restart preserves experiment settings and prior 1417 answers.
- At 01:20 UTC, resumed HiGMem as PID 12749 and queue as PID 12754. The old
  partial report and queue state were preserved on the server as
  `report.before-resume-20260907T0120Z.json` and
  `queue_status.before-resume-20260907T0120Z.json` respectively. No result
  download, model restart or precision/generation change was performed.
- Fresh SSH at 01:21 UTC confirmed both recovery processes RUNNING and
  actual new model usage (38605 cumulative attempts, four failed attempts,
  zero observed truncations). Completion remains unproven.
- HiGMem `invocation_wall_seconds` is per invocation, not cumulative across
  resume. Preserve the prior report for accounting. Four failed attempts
  lack upstream usage and must remain unknown, never zero.
- The previous normal local PC shutdown was already requested. Do not
  repeatedly shut down a PC that has subsequently been turned back on.
  The old shutdown heartbeat remains PAUSED.
- At 01:26 UTC, HiGMem PID 12749 and queue PID 12754 remained RUNNING.
  Cumulative attempts increased to 38629 from 38595, with no new failed
  attempts or observed truncations. QA count remained 1417 while the
  incomplete conversation's memory was being constructed.
- A fresh no-model-API initializer ran for each of all six clients under
  `/workspace/locomo-preflight`. No method-specific import/init failure was
  observed; LightMem loaded the pinned CPU compressor. All six diagnostic
  commands nevertheless exited 1 because `preflight_locomo_clients.py:61`
  incorrectly checks the wrapper as a `PreTrainedTokenizerBase`. A separate
  deployed shared-loader probe confirmed `HuggingFaceTokenizerAdapter`
  wraps a genuine `Qwen2TokenizerFast` and the underlying instance passes
  that invariant. This is a diagnostic false positive, not a proven tokenizer
  fallback. Do not exclude these baselines based on that exit code. The
  preflight script is not called by the live queue. Full inference remains
  unverified. No source/model/service changes were made for this check.

### GPU terminal failure observed on 2026-09-07 at 01:47 UTC

- The resumed run repeated the same `conv-44` `_decide_event_affiliation`
  request. Attempts 0 and 1 timed out after 600 seconds. Attempt 2 ended in
  an HTTP 500 when vLLM's EngineCore failed with `CUDA error: unspecified
  launch failure`; the scheduler dump showed 5012 generated tokens for the
  pathological structured response. HiGMem and the queue then exited.
- The host-injected GPU is currently unavailable inside the container:
  `vast-capabilities` reports `No GPU detected`, and `nvidia-smi` reports
  `No devices were found` / `Unable to determine the device handle`.
  Do not restart the model process in this state.
- Current preserved result is still 1417/1540 QA and 5767/5882 checkpointed
  turns. Failed attempts increased from four to seven; their token use is
  unknown, not zero. The six baselines have still not begun.
- The narrow recovery candidate is to bound only event-affiliation JSON:
  disallow extra properties, bound the brief reasoning and affiliation list
  strings/count, and add a generous 4096-token safety ceiling for this call.
  Previous valid responses were 513-961 characters, so these constraints
  preserve intended semantics while stopping runaway output. Any protocol
  change must be recorded with the original manifest preserved.
- That recovery patch is now prepared locally in
  `D:\MemoryData\HiGMem\run_vast.py`: only `_decide_event_affiliation` gets
  bounded JSON fields plus a 4096-token safety ceiling. Other construction
  and QA calls are unchanged. SHA256 is
  `2b5497f2239c5f8773f2dda5966856e6865473fb89b47a7884d5537af91093e3`.
  `test_vast_runner.py` passed all 6 tests under the existing Python 3.12
  comparison environment, followed by a successful `py_compile`. It has
  not been uploaded or applied to the server pending explicit approval.
- Vast CLI documents `reboot instance` as stop/start without loss of GPU
  priority, and the instance guide says stop/start preserves the container
  filesystem. A reboot is nevertheless a material external action and was
  not executed. User approval is required because the earlier instruction
  was not to stop or destroy the Vast instance. After approval, reboot,
  verify GPU health, start vLLM and wait for health, then resume HiGMem and
  the queue in that order. Never destroy or recycle the instance.

### Historical deployment snapshot (superseded by recovery above)

- HiGMem is independently supervised as `higmem-eval` (PID 3597) and
  `higmem-vllm` (PID 3004). Do not restart while healthy.
- Latest service observation: HiGMem remained RUNNING at 4:32 runtime.
  The preceding 4:26 aggregate snapshot had 389/1540 QA, 5137 checkpointed
  turns, 21991 requests, 21157872 reported prompt tokens and 4052210
  completion tokens; 1 failed request and 0 truncated responses.
- One 4ms `Connection error` for `conv-49`, QA index 61 was followed by
  metered successful retries and a saved nonempty answer. Its failed-attempt
  tokens remain unknown; do not replace them with zero or claim exact totals.
- `/workspace/comparison-venv` was created separately. Its `.pth` shares the
  existing Torch/CUDA packages read-only; 47 non-GPU client packages were
  installed locally to that new environment. The live HiGMem environment
  was not modified. Import probe succeeded with Torch 2.13.0+cu130,
  Transformers 4.57.6, HTTPX 0.28.1, LangChain, LanceDB, Qdrant and LLMLingua.
- Pinned LLMLingua auxiliary CPU model is downloaded. Revision
  `5f0c82792b7ea14c6484e015b6a072009496b7f2`; weight SHA256
  `22b9ecde52fec5c97e8c54a293be768727df95a81c6c8dccb03f262a50c58324` verified.
- Disk remaining after environment/model preparation: approximately 4.2 GB.
- Actual read-only Vast price query: **0.3422222222222222 USD/hour total**,
  base GPU price 0.3333333333333333 USD/hour; disk component unavailable.
- The six-method source tree and plan are deployed. Plan preparation and
  `--check-only` passed for all six methods, each with the full 1540 QA.
- Supervisor registered both new programs. `locomo-comparison-queue` is
  RUNNING as PID 8357 since approximately 2026-09-06 18:40 UTC.
  `comparison-vllm` remains STOPPED intentionally while the original
  `higmem-eval` and `higmem-vllm` services remain RUNNING.
- Fresh SSH independently confirmed queue PID 8357 RUNNING (32 seconds
  uptime), parent PID 628 (Supervisor), and `queue_status.json` state
  `waiting_for_higmem`. HiGMem PID 3597 remained RUNNING at 4:32:42 runtime
  and its original vLLM PID 3004 remained RUNNING.
- Local heartbeat `automation` was changed to PAUSED and verified. The
  reservation condition for the user's normal local PC shutdown is met.

## Deployment and shutdown handoff

1. Completed: approved v2 source deployment, manifest validation, plan
   preparation and check-only validation. Plan is at
   `/workspace/MemoryData/comparison_config/plan.json`; output root is
   `/workspace/locomo-comparison-qwen35-fp16`; run ID is
   `locomo-qwen35-fp16-20260907`, hourly rate is 0.3422222222222222 USD.
2. Completed: both Supervisor programs registered, only the queue started.
   The queue waits for HiGMem to exit and pass output/data gates before
   replacing its vLLM service with `comparison-vllm`. Individual baseline
   failures are recorded in `failures.json` and the next method is attempted.
3. Completed: fresh SSH confirmed the supervised queue is RUNNING,
   waiting for HiGMem, while the current HiGMem remains healthy.
4. Completed: local shutdown heartbeat paused. Leave server paths/state for
   the user, then perform a normal Windows shutdown without `/f`.
   Registration is not a claim that all six experiments have succeeded.

## Fixed comparison and accounting

- Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
  FP16 weights and nonquantized KV cache, 32768 context, vLLM prefix caching.
- Explicit opt-in proxy policy: temperature 0, thinking disabled, request output
  caps removed, `--generation-config vllm`. Messages, tools and schemas are
  preserved. Truncated output is rejected only after recording actual usage.
- Same MiniLM CPU encoder as HiGMem: revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 dimensions and native
  256-token encoder truncation accounted separately.
- Original public LoCoMo category 1–4: 1540 QA, 5882 complete turns.
  Per-turn ingestion preserves original speaker, date, caption and source ID.
- A-MEM retrieval 10; SimpleMem original 40-turn window / overlap 2 with
  planning/reflection; LightMem full compression/topic/extraction/offline
  update pipeline, retrieval 60; E-Mem explicitly text mode.
- Actual API usage is recorded per attempt, including internal operations.
  LightMem local auxiliary model forwards/tokens/time are separate. Official
  category-aware F1 checks exact IDs and original metadata. GPU energy is
  sampled/integrated, not a hardware-counter reading; missing values stay null.
- A recovered HiGMem transport error may permit subsequent methods after all
  original questions complete, but its accounting sidecar explicitly marks
  unknown failed-attempt tokens. This does not rewrite HiGMem's original logs.

Final combined local verification before v2 packaging: **148 passed,
1 skipped, 2 warnings** in 79.28 seconds. The skipped test requires creation
of a real venv symlink, unavailable with the Windows test permissions.
The final Python review reported no remaining blockers.
