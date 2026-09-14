# Native-three Python and launch review

**APPROVE — no CRITICAL or HIGH issues found in the reviewed new launcher/verifier code.**

Reviewed at: 2026-09-10T15:07:37.847012+00:00

Scope: `run_three.py`, `verify_native3.py`, their tests, fixed shell/Supervisor launch configuration, and copied-source provenance. Existing LangMem, Mem0 and A-MEM method implementations were not changed. This is code review approval, not a claim that remote runtime checks, inference, or official evaluation have completed.

Checks completed:

- Attempted `git diff -- '*.py'`; the workspace is not a Git repository. Reviewed new files directly and compared frozen copied files against the source package.
- Independently checked all 663 copied source records against both the original native7 package and this package: 1,326 byte-length/SHA256 checks, zero mismatches.
- Ruff passed for the four new Python launcher/verifier/test files.
- Independently reran all 17 queue/verifier unit tests; all passed. Synthetic subprocess and service fixtures only, no network or GPU use.
- Eight additional reviewer edge checks passed: exact flags, changed flags, missing required flags, thinking-on rejection, old-receipt preservation, integrity failure before CUDA, live-process change rejection, and host-specific unjudged success receipt.
- Git Bash `bash -n` passed for `serve_qwen.sh`, `run_one.sh`, `launch_three.sh`, and `environment/install.sh`.
- Mypy, Black, Pylint and Bandit are not installed in the selected local review environment; no tools were installed for this review.

The queue checks canonical S500 identity, exact three-method allocation, distinct run directories, a pinned verified-runtime receipt, and fresh complete per-method statuses. A failed smoke excludes only that method's full run; failures are retained while independent methods continue. Overall completion requires all 1,500 answers across the three completed full runs; any failed step leaves the queue incomplete and returns nonzero. Official judging stays pending with zero judged answers.

The service startup binds loopback and pins Qwen3.5-9B FP16, context, thinking, tool parser and seed. Shared `flock` prevents overlap between the supported manual and automated launchers; Supervisor starts only on explicit request and stops process groups. `queue/smoke_ids.json` contains the shared first history for manual smoke use.

Launch prerequisites already owned by the root agent: complete installation/import checks for both client environments, run the fresh host runtime verifier, bind its resulting SHA256 in the launch plan, and then start the single supported queue launcher. The verifier explicitly records that client-environment evidence is separate. Source transfer and service-only startup can precede those inference-launch gates.

Reviewed file hashes:

| File | SHA256 |
|---|---|
| `run_three.py` | `fc00049aae676c76492ff3ff61da158395796bc1c379b24e0e8d2ff972070d31` |
| `test_run_three.py` | `e656a6f2c5901412c52ba3360eea2d3af13756b7c3840346017ab9c7356bdf57` |
| `verify_native3.py` | `ad4efd002ec2b29ff2ec8fb51d434c8902dc1daa2a6b842fc5839ecc71f08c71` |
| `test_verify_native3.py` | `8428b1f83cf055d1fe63d109e5f3a420ca3ce4e004c929eeaa4e1cbd1c530b5b` |
| `verify_runtime_reference.py` | `6bcc247f0b9a6625bdc92914d401319d127a44a06fc7c49454c9acbacacfb662` |
| `serve_qwen.sh` | `ed5955e2c69da8d6c99f8514882242f41bc02ed0e5f284b2c83539b46d950217` |
| `serve_minilm.py` | `fd3f5be84c442982a724b72d349bc233e47fd8872ba442973c0f92bed3f38f1c` |
| `metered_lme_proxy.py` | `9f08975c8032d2359fc597686d6c03f1fe42a2dd883c2d8a36964deb4dcf1436` |
| `native3-services.conf` | `4c2c146e16bc1645a9754d9c9b034050ddd35065148437dd8e69c2802ffee3f6` |
| `native3-queue.conf` | `d56e59c406bd62245f57aacad6f03ff71aa1a8313f187ad7b2959ff6ddb24597` |
| `run_one.sh` | `3e16addd8f4fc03ff72b95b7b9a78eafacf8da8577f9ff73916e54ed0f6322ef` |
| `launch_three.sh` | `f631193641c87917c4b9dd406694b93c36499c7f4e5725e867a4f2eb336ff64c` |
| `launch_plan.template.json` | `4a823ce873d50a588b176087c2089d87fbbef7fc2f1bf2de8c15a78a27b767d1` |
| `queue/smoke_ids.json` | `f10a45544b8977c3faf22470dc554adcbdd5a596eb91d8a002bbb95c8df8dbf7` |
| `model_integrity_native3.json` | `95ca4b260a4e7a973ec65428f2b4dd88f0ab2b3a13e35f99809f2a4c0c0be219` |
| `COPIED_SOURCE_PROVENANCE.json` | `c4e4cab4be75734c9499bbde1e526633c33b8b42343efce2c4246fb49ce31de1` |
| `environment/install.sh` | `8773e1601ab665dea33f9152fdad48ff88f246d385a24d4ea81ac4ca657af59c` |
| `environment/client.freeze.txt` | `218ee42ff131e155743d5b2c3ccf98a884a7409f8362cba183609dd99d730011` |
| `environment/mem0-py311.lock` | `54ca0f16442a7cdd6616b94d30294959c2f291f88710e8e70727876a2fd31fd5` |
