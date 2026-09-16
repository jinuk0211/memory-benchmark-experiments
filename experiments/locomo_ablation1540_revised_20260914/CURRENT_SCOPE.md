# Current scope — 2026-09-14 user correction

User explicitly removed LongMemEval: "longmemeval에서는 할필요없어".

Continue LoCoMo only, with the same complete 1,540 category 1–4 questions for every final ablation arm. Establish without-evidence-binding baseline, fix controlled token accounting, evaluate utility-vs-random (ten seeds), cues-vs-base and question-key-vs-payload-key, paired confidence intervals/tests and full tables. Retain every exploratory round; do not claim significance before the full native results support it.

LongMemEval source construction, utility, reader, judge and reporting are canceled. ROOT4 artifacts are inactive drafts; no LME GPU writer or judgment inference has run. Do not resume them from the older goal wording. Existing LoCoMo supervisor jobs remain active.
## Added user constraint — 2026-09-14

Use the fixed300-question development population for iterative detail tuning while preserving the method. Final acceptance still requires all1,540 questions and F1 at least58.71430233556697% (display58.7143). Never adopt a lower candidate; retain unsuccessful experiments. The current whole-population jobs continue unchanged. New development artifacts are under locomo_dev300_refinement_20260914/SEARCH_SCOPE.md. LongMemEval remains canceled.