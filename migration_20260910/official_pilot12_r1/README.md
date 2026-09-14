# Frozen Qwen3.5-9B / MiniLM pilot: official evaluation handoff

All three export artifacts now exist and pass a fresh, read-only audit through the unchanged approved bridge: **12/12 hypotheses per arm, 36/36 total**. The audit recomputed each receipt from the frozen protocol, declared expected IDs, canonical reference, and completed predictions. Every exported hypothesis matches the generated text exactly, including ordering and whitespace. The original seed files were preserved.

See [EXPORT_STATUS.json](EXPORT_STATUS.json), each `.receipt.json`, and `export_completion_audit.*`. The completion audit exited 0. Both newly completed exports (`r40_fused_four_turn` and `s_parent_single_2000`) exited 0, with stdout, stderr, and exit-code files retained. The earlier seed exporter process exit code was not retained; it remains unknown. Its existing hypothesis, receipt, and stdout passed the same fresh artifact audit. Empty stderr is not used to infer an unobserved process exit code.

This is a **diagnostic 12-question subset**, covering four of the six question types and no abstention questions. **Official accuracy is not available.** No judge API request or official evaluation was performed by this export. The actual unchanged official judge and metrics scripts still need to run with the user's configured API credentials. Diagnostic token F1 is not official accuracy.

- Generation model: `Qwen/Qwen3.5-9B`, embedding: `sentence-transformers/all-MiniLM-L6-v2`.
- Frozen generation protocol SHA256: `e95bda1e33a0241fb53f7ca8c1fefeb5661e3f2c13a17990a05aa2894454aac0`.
- Approved bridge SHA256: `9659834d6cb7a51ac26171f79b434c6cb028592a1a0efb03c15c0bb67b2c5e6e`.
- Official upstream commit: `9e0b455f4ef0e2ab8f2e582289761153549043fc`; script hashes are retained in every export receipt.

## Exact later commands (PowerShell; not yet executed)

Run in a process where the user's authorized `OPENAI_API_KEY` is already configured. Do not paste the secret into these files or command logs. These commands retain each command's argument list, stdout, stderr, and actual exit code, audit the official result artifacts, and execute the actual upstream metrics script. `-X utf8` is necessary for the unchanged scripts' default file decoding on Windows.

```powershell
$OfficialPy = 'D:\MemoryData\migration_20260910\official_eval_env\Scripts\python.exe'
$OfficialVendor = 'D:\MemoryData\migration_20260910\official_longmemeval'
$OfficialBridge = 'D:\MemoryData\migration_20260910\official_eval_tools\official_eval_bridge.py'
$OfficialReference = 'D:\MemoryData\MemoryData\datasets\LongMemEval\longmemeval_s_cleaned.json'
$OfficialExpected = 'D:\MemoryData\migration_20260910\PILOT_EXPECTED_IDS.json'
$CollectedRun = 'D:\MemoryData\migration_20260910\pilot_runs\paper_frozen_lme12_qwen35_minilm_date_v2_r1\collected'
$OfficialAttempt = 'D:\MemoryData\migration_20260910\official_pilot12_r1'
if (-not $env:OPENAI_API_KEY) { throw 'Authorized OPENAI_API_KEY must be configured before judging.' }

foreach ($OfficialArm in @('seed', 'r40_fused_four_turn', 's_parent_single_2000')) {
    $OfficialHyp = Join-Path $OfficialAttempt "$OfficialArm.hypotheses.jsonl"
    $OfficialResults = "$OfficialHyp.eval-results-gpt-4o"
    $OfficialAudit = "$OfficialHyp.audit.json"
    foreach ($OfficialTarget in @($OfficialResults, $OfficialAudit)) {
        if (Test-Path -LiteralPath $OfficialTarget) { throw "Preserve previous attempt: $OfficialTarget" }
    }
    $OfficialCommands = @(
        @{ Name='judge'; Argv=@('-X', 'utf8', "$OfficialVendor\src\evaluation\evaluate_qa.py", 'gpt-4o', $OfficialHyp, $OfficialReference) },
        @{ Name='verify'; Argv=@('-X', 'utf8', $OfficialBridge, 'verify', '--predictions', "$CollectedRun\$OfficialArm.jsonl", '--protocol', "$CollectedRun\protocol.json", '--reference', $OfficialReference, '--expected-ids', $OfficialExpected, '--vendor', $OfficialVendor, '--method', $OfficialArm, '--hypotheses', $OfficialHyp, '--results', $OfficialResults, '--report', $OfficialAudit) },
        @{ Name='metrics'; Argv=@('-X', 'utf8', "$OfficialVendor\src\evaluation\print_qa_metrics.py", $OfficialResults, $OfficialReference) }
    )
    foreach ($OfficialCommand in $OfficialCommands) {
        $OfficialPrefix = Join-Path $OfficialAttempt "$OfficialArm.$($OfficialCommand.Name)"
        foreach ($OfficialSuffix in @('argv.json', 'stdout.txt', 'stderr.txt', 'exitcode.txt')) {
            $OfficialTarget = "$OfficialPrefix.$OfficialSuffix"
            if (Test-Path -LiteralPath $OfficialTarget) { throw "Preserve previous attempt: $OfficialTarget" }
        }
    }
    foreach ($OfficialCommand in $OfficialCommands) {
        $OfficialPrefix = Join-Path $OfficialAttempt "$OfficialArm.$($OfficialCommand.Name)"
        $OfficialArguments = $OfficialCommand.Argv
        @($OfficialPy) + $OfficialArguments | ConvertTo-Json | Set-Content -LiteralPath "$OfficialPrefix.argv.json" -Encoding utf8
        & $OfficialPy @OfficialArguments 1> "$OfficialPrefix.stdout.txt" 2> "$OfficialPrefix.stderr.txt"
        $OfficialExitCode = $LASTEXITCODE
        Set-Content -LiteralPath "$OfficialPrefix.exitcode.txt" -Value $OfficialExitCode -Encoding utf8
        if ($OfficialExitCode -ne 0) { throw "Official $($OfficialCommand.Name) failed for $OfficialArm; preserve all attempt files." }
    }
}
```

The evaluator CLI lookup key `gpt-4o` resolves to requested model `gpt-4o-2024-08-06`. Do not pass the snapshot name as the first CLI argument. The official metrics script takes only the result file and canonical reference file.

If interrupted, preserve the existing attempt: the upstream evaluator writes its output in overwrite mode and has no native resume cache. A new attempt or a separately validated disjoint remainder is required. Do not choose among duplicate judgments. A finished 12-question run remains a pilot; two missing question types and no abstention questions cannot establish generalization across all LongMemEval-S categories. The unchanged metrics script may print NaN for absent groups; the bridge reports those groups as null.

The official script does not retain raw judge responses, actual returned model identity, or API token usage. Those values remain unavailable rather than zero. Method-native usage and evaluation-function timing are tracked separately. No invented judgments or accuracy values are present in this handoff.
