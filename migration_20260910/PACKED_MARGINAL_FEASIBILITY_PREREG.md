# Predeclared source-only packed-utility feasibility test

Recorded before new view generation/scoring or construction. The frozen paper and completed Seed-parent benchmark artifacts remain unchanged.

Hypothesis: an alias chosen by utility of its original source may not improve a bounded reader because it does not reach packed context or displaces other evidence. Scoring the change in answer likelihood on the actually retrieved and packed context for an independent source-generated question may reject ineffective aliases and select useful ones.

Design:

- One hash-stable source history from LoCoMo and one from exposed LongMemEval DEV12; deterministic selection must not use target QA, answer scores, or observed failure examples.
- Seed parent, the same source-grounded candidate pool and 2,000-token additional storage cap; unchanged Qwen3.5-9B FP16, MiniLM256, dense+BM25 RRF60/top120, 2,048-token evidence packing and 96-token answer limit.
- Preserve all Seed entries exactly. Original source probe q0 supplies a candidate key. Independently generated QA is used for option scoring; independent QB is excluded from construction and inspected only after memory lock. Use original source-audit probes as a separate acceptance check.
- At most 16 source-fit probes and four audit probes per history; independent QB diagnostics may be bounded to four by a stable source-ID hash before outcomes. Retain actual counts and any missing/invalid view cases. No replacement based on benchmark answers.
- Retain source answerability checks. Measure answer likelihood on Seed's actual packed context and on Seed plus each candidate's actual packed context for QA. Zero context change has zero marginal value. Nonpositive candidates are excluded; empty selection legitimately returns Seed.
- Allocate the unchanged storage budget using these source-derived values, then re-evaluate the complete selected set because individual marginal values need not add. Source-audit rejection rolls the tentative memory back to Seed. Lock the final memory before QB diagnostics.
- This bounded run measures feasibility, source likelihood/F1, effective packing exposure, native construction/scoring token use and elapsed time. It is not a benchmark run, an official LongMemEval score, a generalization result, or a novelty claim.

Decision rule:

An implementation is valid only if source/view separation, immutable Seed preservation, exact fixed-budget retrieval replay, explicit input hashes, and complete native accounting pass. A useful pilot must create an observable packed-evidence change and positive source-fit utility after whole-set selection while passing the source-audit gate. Independent QB determines whether there is support to expand the experiment; a negative or unchanged QB result must be reported rather than retuned against those same diagnostic answers. A null result motivates a new separately declared hypothesis, not a claimed improvement.

Broader evaluation remains necessary on the frozen DEV/validation/final split and Gemma. Final choices and paper claims require unchanged official LongMemEval judging; development F1 is auxiliary.
