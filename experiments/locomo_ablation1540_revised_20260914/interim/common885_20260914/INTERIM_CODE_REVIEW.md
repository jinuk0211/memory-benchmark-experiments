# Common-885 interim snapshot code review

Decision: **APPROVE** for reporting this fixed descriptive snapshot. No blocking correctness findings. This is not approval of a completed full-1,540 study or of subsequent marginal-utility experiments.

Reviewed 2026-09-14. Scope: summarize_snapshot.py, snapshot.json.gz, INTERIM_RESULTS.json and INTERIM_RESULTS_KO.md in this directory. No frozen study code was changed; this review record is the only reviewer-owned output. No GPU, API, remote calls, or LME work was performed.

## Exact reviewed artifacts

| Artifact | SHA-256 |
| --- | --- |
| summarize_snapshot.py | 437587805aaba5f242508a827fae84f598eff2b3f49a9080548806588a0125ad |
| snapshot.json.gz | 847ec0ddab9918b9ba5a066147832aae8b20e0cbec733820680477b4c35d1843 |
| INTERIM_RESULTS.json | ba1a184f79774eeccba2bfc6070b39a47268636b7c9a65321df3e2d1598a77d5 |

## Independent checks

- Ruff passed for the script. Git diff is unavailable because this workspace is not a Git repository; direct source inspection and hashes were used.
- ROOT and dataset/tokenizer/memory paths resolve to the intended existing ROOT1 study and local pinned resources.
- The complete embedded evaluation protocol was independently reconstructed with the frozen runner and compared for exact equality, including manifest, code/input hashes, all memory locks, all 16 arms, fixed random seeds, reader prompt, model environment, packing/retrieval, and full-run population configuration. Grounded reader selection passed the frozen selection validator.
- Every snapshot row has the arm indicated by its enclosing arm, and each of 16 arms contains exactly the same canonical 885 IDs in the same order. The six histories are conv-26, conv-30, conv-41, conv-42, conv-43 and conv-44. Category counts are 172 multi-hop, 180 temporal, 53 open-domain and 480 single-hop.
- All 14,160 reported prediction F1 values were independently recomputed with the frozen official scorer and canonical gold. Every value and each question-weighted arm mean matched exactly. The seed mean, sample standard deviation, and control-minus-Ours delta signs also passed independent arithmetic checks.
- The script checks the canonical full-source hash and local full-gold/question equivalence before filtering. It then validates locked memory payloads, selected context indices/hashes, exact evidence token counts, reconstructed native generation keys, and copied native cache text/finish/token metadata. No partial results are written before all arms pass these gates. Root reported the full script run completed with exit 0; that full context-tokenization run was not redundantly repeated by the reviewer.
- The independent arithmetic command completed all assertions before a console-only cp949 encoding error while printing the Korean document. The document was subsequently read successfully with explicit UTF-8. This did not affect scoring or files.

## Interpretation and limits

The table correctly labels these as first-round, descriptive interim results, explicitly distinct from the full 1,540-question results and from subsequent source marginal utility changes. It makes no confidence-interval or significance claim. Random cue selection reports the mean over all ten seeds, not the best seed. The comparison sign is clearly labeled setting minus Ours. The baseline is without evidence binding, so the component row correctly reads '+ evidence binding'.

Current Ours F1 is 59.2127%; base memory is 59.4980%; random mean is 59.0114%. Thus this snapshot does not show an improvement over base memory, and its advantage over random selection is only about 0.2013 percentage points. Neither difference is a final or statistically significant claim.

This approval covers the explicitly pinned snapshot and the checks above. The script is a narrow snapshot summarizer, not a general full-run validator. Snapshot collection records state that the six histories are the completion intersection at capture time; this local review did not independently reconstruct remote progress for omitted histories or audit full raw GPU meter files. The script's native checks cover the embedded generation cache records. Full study validation and final-population reporting remain separate.