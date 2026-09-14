# HiGMem LoCoMo 40.33 comparability audit

Audit date: 2026-09-11. This note records read-only examination of existing source, final archived execution traces, and saved predictions. No model rerun and no source modification were performed. Only this audit note was created. Companion comparison reports in this directory are owned by other agents.

## Finding

The reported HiGMem score is a completed **1,540-question, 40.3331024621% official F1 run**. The evidence contradicts the suggestion that this run replaced HiGMem with vector-only retrieval: the executed pipeline retained event organization, event-to-turn LLM selection, and final LLM relevance filtering.

The settings with profiles and immediate turn links disabled are explicitly the official repository's main paper setting, despite the flags being named `ablation`. This does not establish identical reader conditions across all methods. The reader uses HiGMem's native short-answer prompt family and has no explicit answer output-token cap. Any claim of controlled memory-quality superiority must account for those reading conditions and other methods' differing retrieval budgets.

## Final result and provenance

Use these files:

- `Final completed report` (historical source: `upload_preparation_20260910/extracted_baseline_results/higmem/report.json:2`): expected 1,540, evaluated 1,540, complete true, F1 0.4033310246213628.
- `Final manifest` (historical source: `upload_preparation_20260910/extracted_baseline_results/higmem/manifest.json:2`).
- `Final scored predictions` (historical source: `upload_preparation_20260910/extracted_baseline_results/higmem/scored_predictions.json`).
- `Final raw predictions` (historical source: `upload_preparation_20260910/extracted_baseline_results/higmem/predictions.jsonl`).
- `Final raw archive` (historical source: `results/locomo/higmem/higmem-complete-20260907.tar.zst`): contains `HiGMem/vast_run/` logs, usage, predictions, scored predictions, manifest, report, and final memory checkpoints; also archived runner, serving configuration, and scorer.
- `Separate final report copy` (historical source: `results/locomo/higmem/higmem-final-report-20260907.json`).

Do not substitute the older `D:/MemoryData/results/locomo/higmem/higmem result.zip`: its `orkspace/HiGMem/vast_run/report.json` reports only 1,417/1,540 questions, complete false, F1 0.3979963461551367. `D:/MemoryData/HiGMem/remote_results/` also contains earlier snapshots/smoke runs, not the final completed run.

The final report's category scores reproduce the table: Multi-hop 27.9480%, Temporal 26.3972%, Open-domain 12.7026%, Single-hop 52.9592%.

### Source hash evidence

The local tracked upstream checkout is commit `f275072f25323a01a8bff3680edbb34ed97d33be` from `https://github.com/ZeroLoss-Lab/HiGMem.git`; `git diff --stat` was empty. The local experiment runner is untracked relative to that upstream checkout. Final manifest hashes are at `manifest.json:26` (historical source: `upload_preparation_20260910/extracted_baseline_results/higmem/manifest.json:26`).

| File | Final manifest SHA256 | Verification |
|---|---|---|
| run_vast.py | `3e1bf7c7e027d0964efc36c54d9b8302accf0c74387390afcd752101d77532bf` | Exact raw bytes match both local runner and final archive `HiGMem/run_vast.py` |
| fphm_core.py | `27482d2f43d5ae4ab2570c397cfd5e884e457e1240f3ef76f198ec5bcc7a87bb` | Local bytes match after CRLF-to-LF normalization |
| prompts.py | `161d2223545620bd1c2f23fe69f6f916ed23cf08a22e2d85e9be8d2c59c07aff` | Local bytes match after CRLF-to-LF normalization |
| memory_layer.py | `0485bd7da21d800e875222662381448a1b92fca4642be004795a24987e7e7c1b` | Local bytes match after CRLF-to-LF normalization |
| official_locomo_evaluation.py | `8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd` | Exact local raw bytes match final manifest |

A raw-byte mismatch for the three Windows checkout files is therefore explained by line endings, not an algorithm difference. The final archive's manifest and report were read directly and agree with the extracted copies.

## Official method and local implementation

The official repository was browsed on 2026-09-11: [official HiGMem reproduction setting](https://github.com/ZeroLoss-Lab/HiGMem#reproducing-the-paper-setting). It documents query rewriting enabled, profiles disabled, event metadata enabled, immediate turn links disabled, and k_event=10 as its main setting. See the matching local `README.md:110` (historical source: `HiGMem/README.md:110`). The same README explains that no explicit `max_tokens` is set for OpenAI-compatible endpoints `README.md:132` (historical source: `HiGMem/README.md:132`).

Implementation map:

| Stage | Evidence |
|---|---|
| Paper-setting system initialization | `run_vast.py:180` (historical source: `HiGMem/run_vast.py:180`): profiles false, metadata true, no-link true, affiliation k=10 |
| Embedding model | `fphm_core.py:95` (historical source: `HiGMem/fphm_core.py:95`): all-MiniLM-L6-v2 unless MPNet ablation enabled |
| Query rewrite + event/turn retrieval | `run_vast.py:147` (historical source: `HiGMem/run_vast.py:147`), `fphm_core.py:986` (historical source: `HiGMem/fphm_core.py:986`): turn k=10 and event k=10 |
| LLM chooses turns within each retrieved event | `fphm_core.py:1055` (historical source: `HiGMem/fphm_core.py:1055`): supplies event title plus constituent turn contents; calls `predict_turns_from_event` |
| Union of vector turn hits and event-selected turns | `fphm_core.py:1092` (historical source: `HiGMem/fphm_core.py:1092`) |
| Final LLM filtering | `fphm_core.py:1116` (historical source: `HiGMem/fphm_core.py:1116`), `fphm_core.py:886` (historical source: `HiGMem/fphm_core.py:886`): candidate turns judged in chunks of 10 |
| Final reader context | `fphm_core.py:1131` (historical source: `HiGMem/fphm_core.py:1131`): timestamp, speaker, original content, context summary, sorted by timestamp |

The no-event and no-filter flags both default to false `fphm_core.py:53` (historical source: `HiGMem/fphm_core.py:53`). More directly, final archive log init rows record `ablation_no_event=false` and `ablation_no_filter=false`, alongside the intended no-profile/no-link settings.

## Executed trace evidence

Counting `step` in all 40 final archive `HiGMem/vast_run/logs/*.jsonl` members gives 79,920 total rows. The logs span original execution and resumes; do not interpret the 40 init/index entries as 40 separate benchmark configurations.

| Logged action | Count | Meaning |
|---|---:|---|
| initial_recall | 1,540 | One initial turn/event retrieval per completed question |
| predict_turns_from_event, aggregate step | 1,540 | One aggregation of event-selected turns per question |
| predict_turns_from_event, LLM caller | 15,400 | Ten event-level LLM selection calls per question |
| judge_relevance_turn, aggregate step | 1,540 | One final candidate-filter aggregation per question |
| _judge_relevance_parallel_turn, LLM caller | 3,743 | Chunk-level filtering calls, not 3,743 distinct questions |
| generate_keyword_query, LLM caller | 1,540 | Query rewriting |
| final_answer_generation, LLM caller | 1,540 | Final reader answers |
| final_context_construction | 1,540 | Reader context construction |
| llm_call | 40,769 | Successful parsed calls across construction and QA |
| call_error | 10 | Recorded errors, retried by strict runner |
| decide_event_affiliation_fallback | 76 | Upstream fallback when no valid final event affiliation remains; not 76 failed questions |

The report records 18,546 construction calls and 22,223 QA calls, totaling 40,769 successful calls, consistent with the trace count. It records missing usage=0 and truncated requests=0. See `report.json:24` (historical source: `upload_preparation_20260910/extracted_baseline_results/higmem/report.json:24`). The wall-time field is approximately zero because the report came from a report-only invocation; it must not be cited as experiment duration.

Exact archive-member line examples (line numbers are 1-based within the member, not within the compressed archive):

| Archive member | Line | Evidence |
|---|---:|---|
| HiGMem/vast_run/logs/run_conv-26_20260906_140853.jsonl | 1 | Init: no-event false, no-filter false, profiles false, metadata true, no-link true |
| HiGMem/vast_run/logs/run_conv-26_20260906_140853.jsonl | 3686 | Final index: 419 turns and 48 events |
| HiGMem/vast_run/logs/run_conv-26_20260906_140853.jsonl | 3689 | Initial retrieval for `What is Caroline's identity?`: 10 events and 10 turns |
| HiGMem/vast_run/logs/run_conv-26_20260906_140853.jsonl | 3758 | Event-based turn selection for Caroline relationship-status question |
| HiGMem/vast_run/logs/run_conv-26_20260906_140853.jsonl | 3779 | Final LLM filtering for that relationship-status question; selected=[] |
| HiGMem/vast_run/logs/run_conv-26_20260906_140853.jsonl | 3780 | That question's empty final context |
| HiGMem/vast_run/logs/run_conv-30_20260906_140853.jsonl | 3947 | Favorite-dance question: gold turn D1:8 in candidates, absent from selected |

The initial-recall example at line 3689 is a different question from lines 3758/3779/3780 because QA executes concurrently. Do not present those four entries as one contiguous question trace.

## Evidence-retrieval diagnostic

This is an observational diagnostic of annotated evidence retention, not an estimate of how many F1 points the filter costs. Gold evidence can be incomplete, and alternative retrieved turns can contain a correct answer. Comparing stages identifies where annotated evidence is retained or lost, but only a rerun/ablation can establish the effect on answer F1.

Matching procedure:

1. Use final `scored_predictions.json` rows, identified by `(sample,index)` for outcome statistics.
2. Trace records do not carry question index. Derive sample from log member `run_(conv-\d+)_...`; join trace records by `(sample,query)` to prediction `(sample,question)`.
3. There are 1,529 distinct `(sample,question)` keys over 1,540 rows. Eleven keys occur twice: **22 ambiguous rows**, excluded from stage comparison instead of assigning an arbitrary trace.
4. Four additional questions have no annotated evidence. Exclude them from evidence recall denominators. Thus **1,514 questions** are used for the before/after stage comparison.
5. Initial set = `initial_recall.data.recalled_turns`. Event-expanded candidate set = `judge_relevance_turn.data.candidates`; in this no-link run it is the union of vector turn hits and event-selected turns. Final set = the question's saved `retrieved` array in predictions. Deduplicate IDs with sets.
6. For each question with gold set G and stage set R, compute `|G intersect R| / |G|`. Report the arithmetic mean over 1,514 questions. This is **macro recall over questions**, not pooled/micro recall over gold turns.

| Stage (n=1,514) | Macro annotated-evidence recall | Any gold retained | All gold retained |
|---|---:|---:|---:|
| Initial vector turn hits | 45.6736929590% | 787 | 617 |
| Event-expanded candidate set | 68.2173752804% | 1,132 | 945 |
| Final LLM-filtered set | 60.7856105979% | 1,011 | 845 |

The event path increases annotated-evidence recall by 22.54 percentage points, followed by a 7.43-point decrease at the final filter. On 121 of these 1,514 questions, some annotated evidence was present in candidates and all annotated evidence was absent from the final retrieved set. These are recall differences, **not F1 differences**.

Prediction-only statistics, which do not require the ambiguous trace join:

- All 1,540 questions: average final retrieved set size **6.7915584416 turns**; **67** have an empty final set.
- The 67 empty-set questions have mean answer F1 **6.3999536947%**. A correct answer with empty context is possible; empty retrieval does not force the output to be empty or incorrect.
- Among 1,536 questions with annotated evidence, final macro evidence recall is **61.0760076683%**. This denominator differs from the 1,514-question stage comparison and the two values must not be conflated.
- Final retrieved set includes all annotated evidence in 857 questions (mean answer F1 58.0945563265%), some evidence in 1,033 questions (mean F1 53.6097281864%), and no annotated evidence in 503 questions (mean F1 13.1394206265%). The `all` group is a subset of `some`, not an additional disjoint group.

### Concrete failures

**Selection failure:** `conv-30`, index 40, `What is Jon's favorite style of dance?` Gold answer `Contemporary`, gold evidence D1:8. The stage before final filtering includes D1:8, but the filter removes it. The model answers `Not mentioned in the context`, F1=0. `Saved question/prediction` (historical source: `upload_preparation_20260910/extracted_baseline_results/higmem/scored_predictions.json:1120`); exact selection trace at archive member `HiGMem/vast_run/logs/run_conv-30_20260906_140853.jsonl`, line 3947.

**Reader/date failure with retrieval success:** `conv-30`, index 1, `When Gina has lost her job at Door Dash?` Gold `January, 2023`; retrieved IDs include gold D1:3, but answer is `this month`, F1=0. `Saved question/prediction` (historical source: `upload_preparation_20260910/extracted_baseline_results/higmem/scored_predictions.json:5`). This shows that having the gold turn does not guarantee that the native reader resolves relative dates into the official gold wording.

## Reader and scoring conditions

| Condition | HiGMem final run |
|---|---|
| Model | Qwen/Qwen3.5-9B |
| Model revision | c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| Thinking | Disabled |
| Temperature | 0 |
| Serving precision | float16 |
| Server max model length | 32,768 |
| Final answer output cap | No explicit max_tokens |
| Answer format | JSON schema requiring string field `answer`; system message requires a JSON object |
| Temporal answer instruction | Use date of conversation to provide approximate date; shortest possible answer |
| Other categories | Short phrase; exact words from context whenever possible |
| Embedding | all-MiniLM-L6-v2 |
| Retrieval | 10 turn candidates + 10 event candidates before hierarchical expansion and filtering |

Evidence: `manifest.json:2` (historical source: `upload_preparation_20260910/extracted_baseline_results/higmem/manifest.json:2`), `request construction` (historical source: `HiGMem/run_vast.py:56`), `native-category reader prompt` (historical source: `HiGMem/run_vast.py:158`). The native upstream prompt family is preserved semantically; compare `run_fphm_evaluation.py:50` (historical source: `HiGMem/run_fphm_evaluation.py:50`).

The adaptation adds strict retry and schema validation `run_vast.py:86` (historical source: `HiGMem/run_vast.py:86`). It bounds **event affiliation only** to 4,096 output tokens, a 2,048-character reasoning field, at most 11 affiliation entries, and 128 characters per entry `run_vast.py:21` (historical source: `HiGMem/run_vast.py:21`). This cap is not an answer-generation cap and should be disclosed as an implementation safeguard. Its score impact has not been isolated.

The runner imports and calls `official_locomo_evaluation.eval_question_answering` directly `run_vast.py:220` (historical source: `HiGMem/run_vast.py:220`). It verifies uniqueness of `(sample,index)` and exact selected-question coverage before marking complete. The saved scorer includes:

- Token F1 with normalization and Porter stemming `official_locomo_evaluation.py:126` (historical source: `HiGMem/official_locomo_evaluation.py:126`).
- Multi-hop splitting on commas and average best-match F1 for each gold component `official_locomo_evaluation.py:141` (historical source: `HiGMem/official_locomo_evaluation.py:141`).
- Open-domain gold-answer preprocessing using text before the first semicolon `official_locomo_evaluation.py:203` (historical source: `HiGMem/official_locomo_evaluation.py:203`).

This verifies HiGMem's own scoring path. Establishing that every table row uses the same evaluator requires the companion cross-method rescoring audit; HiGMem's findings alone do not establish it.

## Compact machine-readable audit record

```json
{
  "method": "HiGMem",
  "evaluated": 1540,
  "official_f1_fraction": 0.4033310246213628,
  "final_archive": "D:/MemoryData/results/locomo/higmem/higmem-complete-20260907.tar.zst",
  "trace_rows": 79920,
  "initial_recall_questions": 1540,
  "event_expansion_questions": 1540,
  "event_turn_selection_llm_calls": 15400,
  "final_filter_questions": 1540,
  "final_filter_llm_calls": 3743,
  "final_answer_llm_calls": 1540,
  "empty_final_retrieval_questions": 67,
  "mean_final_retrieved_turns": 6.791558441558442,
  "trace_join_key": ["sample", "question_text"],
  "duplicate_question_keys": 11,
  "excluded_ambiguous_rows": 22,
  "excluded_no_gold_rows": 4,
  "stage_comparison_questions": 1514,
  "macro_recall_fraction": {
    "initial_vector_turns": 0.45673692959029816,
    "event_expanded_candidates": 0.6821737528040107,
    "final_filtered_turns": 0.6078561059788898
  },
  "questions_losing_all_candidate_gold_in_filter": 121,
  "all_question_gold_recall_denominator": 1536,
  "all_question_final_macro_gold_recall": 0.6107600766831852,
  "limitations": [
    "Evidence retention is observational and is not an F1 attribution.",
    "Gold evidence annotations may omit alternative sufficient evidence.",
    "Native reader/retrieval conditions require cross-method disclosure.",
    "One completed configuration does not estimate seed sensitivity."
  ]
}
```

## Reproducible read-only trace procedure

The Windows tar executable could not find its external zstd executable. The installed `C:/Python314/python.exe` supports `tarfile.open(..., 'r|zst')`, so the following standard-library code reads members directly without unpacking files or deserializing checkpoints. Run via Python 3.14 from a temporary interpreter input if reproducing; it does not write files.

```python
import collections
import json
import pathlib
import re
import statistics
import tarfile

archive_path = pathlib.Path(
    'D:/MemoryData/results/locomo/higmem/higmem-complete-20260907.tar.zst'
)
scored_path = pathlib.Path(
    'D:/MemoryData/upload_preparation_20260910/'
    'extracted_baseline_results/higmem/scored_predictions.json'
)
rows = json.loads(scored_path.read_text(encoding='utf-8'))
key_counts = collections.Counter((r['sample'], r['question']) for r in rows)
steps = collections.Counter()
callers = collections.Counter()
traces = collections.defaultdict(dict)
locations = collections.defaultdict(dict)

with tarfile.open(archive_path, 'r|zst') as archive:
    for member in archive:
        if '/logs/' not in member.name or not member.isfile():
            continue
        sample = re.search(r'run_(conv-\d+)_', member.name).group(1)
        for line_number, line in enumerate(archive.extractfile(member), 1):
            record = json.loads(line)
            step, data = record['step'], record.get('data', {})
            steps[step] += 1
            if step == 'llm_call':
                callers[data.get('caller_function')] += 1
            if step in ('initial_recall', 'judge_relevance_turn'):
                key = sample, data['query']
                traces[key][step] = data
                locations[key][step] = member.name, line_number

usable = [
    r for r in rows
    if key_counts[(r['sample'], r['question'])] == 1 and r['evidence']
]
assert len(rows) == 1540
assert len(usable) == 1514
assert sum(v for v in key_counts.values() if v > 1) == 22
print('step_counts', dict(steps))
print('llm_caller_counts', dict(callers))

for stage in ('initial', 'candidates', 'final'):
    recalls, any_gold, all_gold = [], 0, 0
    for row in usable:
        trace = traces[row['sample'], row['question']]
        gold = set(row['evidence'])
        selected = {
            'initial': trace['initial_recall']['recalled_turns'],
            'candidates': trace['judge_relevance_turn']['candidates'],
            'final': row['retrieved'],
        }[stage]
        retrieved = set(selected)
        recalls.append(len(gold & retrieved) / len(gold))
        any_gold += bool(gold & retrieved)
        all_gold += gold <= retrieved
    print(stage, len(recalls), statistics.mean(recalls), any_gold, all_gold)

lost_all = sum(
    bool(set(r['evidence']) & set(
        traces[r['sample'], r['question']]['judge_relevance_turn']['candidates']
    )) and not bool(set(r['evidence']) & set(r['retrieved']))
    for r in usable
)
print('lost_all_gold_in_filter', lost_all)
print('empty_final', sum(not set(r['retrieved']) for r in rows))
print('mean_final_turns', statistics.mean(len(set(r['retrieved'])) for r in rows))
example = ('conv-30', "What is Jon's favorite style of dance?")
print('example_filter_location', locations[example]['judge_relevance_turn'])
print('example_filter_record', traces[example]['judge_relevance_turn'])
```

## Defensible interpretation

The current evidence supports: **Under its official native algorithm settings and the Qwen3.5-9B run conditions, HiGMem preserves hierarchical evidence search and LLM selection but obtains 40.33 F1; its traces expose both retrieval-selection losses and final-reader errors.** It does not support attributing the entire difference from Our method to memory design alone, nor asserting that HiGMem was reduced to vector-only search. A common-reader evaluation over saved retrieved contexts, plus targeted selection/reader ablations, would test the remaining confounding factors.
