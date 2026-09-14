"""Recursive source rehearsal with a fixed additional-memory budget.

All questions here were generated from source sessions, never benchmark QA.
Reader/retrieval stay fixed. Rehearsal changes evidence allocation and index cues.
"""
from collections import defaultdict
import copy

import refine as core
from budgeted_evidence import multiple_choice_budget, probe_options, storage_cost
from evidence_utility import POLICY, context_for, reader_user
from parent_evidence import make_options

RECIPE = {'name': 'recursive_source_rehearsal_v1', 'rounds': 3, 'extra_tokens': 2000,
          'read_tokens': 2048, 'answer_tokens': 96, 'query_rewrite_tokens': 64,
          'selection': 'strict source-fit improvement with nondecreasing source-audit mean',
          'allocation': 'cumulative residual times verified likelihood utility times retrieved reciprocal rank',
          'benchmark_qa_used_for_construction': False}
REINDEX = (
    'Write one concise search question that retrieves the given SOURCE evidence. '
    'The previous question retrieved the DISTRACTOR memory instead. Keep the original '
    'information need, but make the correct people, event, object and time explicit '
    'using only SOURCE. Do not copy unsupported information from the distractor. '
    'Do not include the answer or any explanation. Output only the revised question.'
)


def retrieve(rt, units: list[dict], questions: list[str]) -> list[dict]:
    """Same MiniLM/BM25 RRF60/top120 and2048-token packing as the frozen reader."""
    import numpy as np
    from rank_bm25 import BM25Okapi
    if not units:
        raise ValueError('Empty memory')
    if not questions:
        return []
    texts = [unit.get('index_text', unit['text']) for unit in units]
    vectors = rt.encode(texts)
    sparse = BM25Okapi([core.lexical(text) or ['_empty'] for text in texts])
    queries = rt.encode(questions, query=True)
    def ranks(scores):
        positions = np.empty(len(scores), dtype=np.int64)
        positions[np.argsort(-scores, kind='stable')] = np.arange(len(scores))
        return positions
    results = []
    for question, query in zip(questions, queries):
        scores = 1 / (60 + ranks(vectors @ query)) + 1 / (60 + ranks(np.asarray(sparse.get_scores(core.lexical(question)))))
        order = np.argsort(-scores, kind='stable')[:min(len(units), 120)]
        context, hits, tokens = core.pack(units, order, rt.ntok, RECIPE['read_tokens'])
        results.append({'context': context, 'read_tokens': tokens,
                        'source_ids': sorted({sid for index in hits for sid in units[index]['sources']})})
    return results


def rehearse(rt, units: list[dict], probes: list[dict]) -> dict[str, dict]:
    contexts = retrieve(rt, units, [probe['question'] for probe in probes])
    prompts = [reader_user(probe['question'], context['context']) for probe, context in zip(probes, contexts)]
    answers = rt.generate(core.READER, prompts, max_tokens=RECIPE['answer_tokens']) if prompts else []
    if len(answers) != len(probes):
        raise ValueError('Missing rehearsal predictions')
    return {probe['id']: {**context, 'prediction': answer,
                         'score': core.generic_f1(answer, probe['answer']), 'split': probe['split']}
            for probe, context, answer in zip(probes, contexts, answers)}


def means(results: dict[str, dict]) -> dict[str, float | None]:
    values = defaultdict(list)
    for row in results.values():
        values[row['split']].append(row['score'])
    return {split: sum(values[split]) / len(values[split]) if values[split] else None
            for split in ('probe_fit', 'probe_audit')}


def accept(previous: dict, proposed: dict) -> bool:
    """No acceptance on empty checks, fit ties, or audit regressions."""
    old, new = means(previous), means(proposed)
    return (all(value is not None for value in list(old.values()) + list(new.values()))
            and new['probe_fit'] > old['probe_fit']
            and new['probe_audit'] >= old['probe_audit'])


def option_pool(rt, sessions: list[dict], parent: list[dict], rows: list[dict]) -> list[list[dict]]:
    """Keep original likelihood admission and verify each literal/parent payload."""
    if any(row['split'] != 'probe_fit' for row in rows):
        raise ValueError('Only source-fit probes may create memory options')
    raw, _ = probe_options(sessions, rows, rt.ntok, 'single', realized_only=True)
    _, mapped = make_options(rt, sessions, parent, rows)
    by_id = {row['id']: row for row in rows}
    groups = []
    for group in raw:
        options = []
        seen = set()
        for original in group:
            for family, item in (('literal', original), ('parent', mapped.get(original['id']))):
                if item is None:
                    continue
                option = copy.deepcopy(item)
                identity = core.digest(option['unit'])
                if identity in seen or option['cost'] > RECIPE['extra_tokens']:
                    continue
                seen.add(identity)
                option['id'] += ':' + family
                options.append(option)
        groups.append(options)
    flattened = [option for group in groups for option in group]
    prompts = [reader_user(by_id[option['unit']['source_probe_id']]['question'], option['unit']['text'])
               for option in flattened]
    predictions = rt.generate(core.READER, prompts, max_tokens=RECIPE['answer_tokens']) if prompts else []
    if len(predictions) != len(flattened):
        raise ValueError('Missing evidence-verification predictions')
    for option, prediction in zip(flattened, predictions):
        option['verification_prediction'] = prediction
        option['verification_f1'] = core.generic_f1(prediction, by_id[option['unit']['source_probe_id']]['answer'])
    return [[option for option in group if option['verification_f1'] >= POLICY['minimum_full_answer_f1']]
            for group in groups]


def score_index_routes(rt, parent: list[dict], groups: list[list[dict]], probes: list[dict]) -> None:
    """Measure each cue's actual fixed-retriever rank before comparing its cost.

    A longer rewritten cue can survive the budget frontier when it retrieves its
    evidence earlier. Final reader rehearsal, not rank alone, decides acceptance.
    """
    import numpy as np
    from rank_bm25 import BM25Okapi
    options = [option for group in groups for option in group]
    if not options:
        return
    base_texts = [unit.get('index_text', unit['text']) for unit in parent]
    base_tokens = [core.lexical(text) or ['_empty'] for text in base_texts]
    base_vectors = rt.encode(base_texts)
    option_texts = [option['unit'].get('index_text', option['unit']['text']) for option in options]
    option_vectors = rt.encode(option_texts)
    query_vectors = rt.encode([probe['question'] for probe in probes], query=True)
    by_id = {probe['id']: (probe['question'], vector) for probe, vector in zip(probes, query_vectors)}
    def ranks(scores):
        positions = np.empty(len(scores), dtype=np.int64)
        positions[np.argsort(-scores, kind='stable')] = np.arange(len(scores))
        return positions
    for option, text, vector in zip(options, option_texts, option_vectors):
        question, query = by_id[option['unit']['source_probe_id']]
        dense = np.append(base_vectors @ query, vector @ query)
        sparse = BM25Okapi(base_tokens + [core.lexical(text) or ['_empty']])
        fused = 1 / (60 + ranks(dense)) + 1 / (60 + ranks(np.asarray(sparse.get_scores(core.lexical(question)))))
        order = np.argsort(-fused, kind='stable')[:min(len(parent) + 1, 120)]
        _, hits, _ = core.pack(parent + [option['unit']], order, rt.ntok, RECIPE['read_tokens'])
        reciprocal_rank = 1 / (1 + hits.index(len(parent))) if len(parent) in hits else 0.0
        option.setdefault('likelihood_gain', option['gain'])
        option['retrieved_reciprocal_rank'] = reciprocal_rank
        option['gain'] = option['likelihood_gain'] * reciprocal_rank


def allocation(groups: list[list[dict]], residual: dict[str, float], ntok) -> tuple[list[dict], dict]:
    weighted = copy.deepcopy(groups)
    for group in weighted:
        for option in group:
            option['gain'] *= residual[option['unit']['source_probe_id']]
            option['cost'] = storage_cost(option['unit'], ntok)
    selected, stats = multiple_choice_budget(weighted, RECIPE['extra_tokens'], quantum=8)
    return [option['unit'] for option in selected], {**stats, 'selected_options': [option['id'] for option in selected]}


def refine_memory(rt, sessions: list[dict], parent: list[dict], initial: list[dict],
                  utility_rows: list[dict], audit_probes: list[dict], save_round) -> tuple[list[dict], list[dict]]:
    """Recursively check current memory, revise retrieval cues, reallocate, and verify."""
    if any(probe['split'] != 'probe_audit' for probe in audit_probes):
        raise ValueError('Audit partition mismatch')
    groups = option_pool(rt, sessions, parent, utility_rows)
    admitted = {option['unit']['source_probe_id'] for group in groups for option in group}
    fit = [row for row in utility_rows if row['id'] in admitted]
    score_index_routes(rt, parent, groups, fit)
    probes = fit + audit_probes
    current = copy.deepcopy(initial)
    baseline_storage = sum(storage_cost(unit, rt.ntok) for unit in parent)
    if sum(storage_cost(unit, rt.ntok) for unit in current) > baseline_storage + RECIPE['extra_tokens']:
        raise ValueError('Initial memory exceeds the fixed comparison storage cap')
    performance = rehearse(rt, current, probes)
    residual = {row['id']: 0.0 for row in fit}
    turns = {turn['id']: turn for session in sessions for turn in session['turns']}
    histories = []
    for round_id in range(1, RECIPE['rounds'] + 1):
        for row in fit:
            residual[row['id']] += 1.0 - performance[row['id']]['score']
        if round_id > 1:
            failed = [row for row in fit if performance[row['id']]['score'] < 1.0]
            prompts = []
            for row in failed:
                source = context_for(row, turns, row['candidate_context_ids'])
                prompts.append(f"SOURCE:\n{source}\n\nPrevious question: {row['question']}\n\n"
                               f"DISTRACTOR memory:\n{performance[row['id']]['context']}\n\nRevised question:")
            rewrites = rt.generate(REINDEX, prompts, max_tokens=RECIPE['query_rewrite_tokens']) if prompts else []
            if len(rewrites) != len(failed):
                raise ValueError('Missing retrieval-cue rewrites')
            lookup = {row['id']: rewrite.strip() for row, rewrite in zip(failed, rewrites) if rewrite.strip()}
            for group in groups:
                originals = [option for option in group if ':rewrite:' not in option['id']]
                for original in originals:
                    pid = original['unit']['source_probe_id']
                    if pid not in lookup:
                        continue
                    option = copy.deepcopy(original)
                    option['id'] += f':rewrite:{round_id}'
                    option['unit']['index_text'] = lookup[pid]
                    option['cost'] = storage_cost(option['unit'], rt.ntok)
                    group.append(option)
            score_index_routes(rt, parent, groups, fit)
        added, stats = allocation(groups, residual, rt.ntok)
        proposal = parent + added
        tested = rehearse(rt, proposal, probes)
        accepted = accept(performance, tested)
        record = {'round': round_id, 'accepted': accepted, 'before': means(performance), 'after': means(tested),
                  'allocation': stats, 'cumulative_residual': dict(residual),
                  'actual_extra_tokens': sum(storage_cost(unit, rt.ntok) for unit in added),
                  'fit_probe_count': len(fit), 'audit_probe_count': len(audit_probes),
                  'proposal_memory_sha256': core.digest(proposal),
                  'rehearsal': tested, 'candidate_pool': copy.deepcopy(groups)}
        if accepted:
            current, performance = proposal, tested
        record['accepted_memory_sha256'] = core.digest(current)
        histories.append(record)
        save_round(round_id, current, record)
    return current, histories
