"""Source rehearsal v2: original cue Q0 is never the fit retrieval query QA."""
import copy
import math

import refine as core
import recursive_memory as base
from crossview_probes import QUESTION_CHECK
from evidence_fidelity import accepted as equivalent
from evidence_utility import context_for
from source_views_v2 import copies_view, fit_rows, forbidden_views, generate

RECIPE = {**base.RECIPE, 'name': 'recursive_source_rehearsal_v2',
          'allocation_query': 'frozen source-only QA; Q0 remains the stored cue',
          'rewrite_input': 'Q0 and source; QA/QB never supplied to cue writer',
          'diagnostic_query': 'QB only after final memory lock'}


def contract(probes: list[dict]) -> dict[str, str]:
    if len({p['id'] for p in probes}) != len(probes):
        raise ValueError('Duplicate rehearsal probe ID')
    return {p['id']: core.digest({key: p[key] for key in ('id', 'question', 'answer', 'split')})
            for p in probes}


def rehearse(rt, memory: list[dict], probes: list[dict], expected: dict) -> dict:
    if contract(probes) != expected:
        raise ValueError('Fixed rehearsal denominator, wording or answer changed')
    result = base.rehearse(rt, memory, probes)
    if set(result) != set(expected):
        raise ValueError('Missing rehearsal result')
    splits = {p['id']: p['split'] for p in probes}
    for identity, row in result.items():
        if (row['split'] != splits[identity] or not math.isfinite(row['score'])
                or not 0 <= row['score'] <= 1):
            raise ValueError('Invalid rehearsal score or split')
        row['task_sha256'] = expected[identity]
    return result


def accept(previous: dict, proposed: dict, expected: dict) -> bool:
    for result in (previous, proposed):
        if set(result) != set(expected) or any(result[k].get('task_sha256') != v for k, v in expected.items()):
            raise ValueError('Cannot compare different source rehearsal tasks')
    if any(previous[k]['split'] != proposed[k]['split'] for k in expected):
        raise ValueError('Rehearsal partition changed')
    return base.accept(previous, proposed)


def refine_memory(rt, sessions: list[dict], parent: list[dict], initial: list[dict],
                  utility_rows: list[dict], audit_probes: list[dict], views: dict,
                  save_round) -> tuple[list[dict], list[dict]]:
    fit = fit_rows(views, utility_rows)
    if not audit_probes or any(p['split'] != 'probe_audit' for p in audit_probes):
        raise ValueError('Nonempty original source-audit partition required')
    valid_ids = {p['id'] for p in fit}
    # Keep Q0 for likelihood admission, literal verification and indexed options.
    originals = [p for p in utility_rows if p['id'] in valid_ids]
    if any(p.get('view') or p.get('original_question') for p in originals):
        raise ValueError('Question views cannot be passed to option creation')
    groups = base.option_pool(rt, sessions, parent, originals)
    admitted = {o['unit']['source_probe_id'] for g in groups for o in g}
    fit = [p for p in fit if p['id'] in admitted]
    if not fit:
        raise ValueError('No admitted paired source-fit probes')
    probes = fit + copy.deepcopy(audit_probes)
    expected = contract(probes)
    forbidden = forbidden_views(views)
    for unit in parent + initial + [o['unit'] for g in groups for o in g]:
        if copies_view(unit.get('index_text', unit['text']), forbidden):
            raise ValueError('An evaluation view occurs in a stored index')
    base.score_index_routes(rt, parent, groups, fit)
    current = copy.deepcopy(initial)
    parent_cost = sum(base.storage_cost(u, rt.ntok) for u in parent)
    if sum(base.storage_cost(u, rt.ntok) for u in current) > parent_cost + RECIPE['extra_tokens']:
        raise ValueError('Initial memory exceeds the original storage cap')
    performance = rehearse(rt, current, probes, expected)
    residual = {p['id']: 0.0 for p in fit}
    by_id = {p['id']: p for p in originals}
    turns = {t['id']: t for s in sessions for t in s['turns']}
    histories = []
    for round_id in range(1, RECIPE['rounds'] + 1):
        for row in fit:
            residual[row['id']] += 1 - performance[row['id']]['score']
        rewrite_checks = []
        if round_id > 1:
            failed = [by_id[p['id']] for p in fit if performance[p['id']]['score'] < 1]
            contexts = {r['id']: context_for(r, turns, r['candidate_context_ids']) for r in failed}
            prompts = [f"SOURCE:\n{contexts[r['id']]}\n\nPrevious question: {r['question']}\n\n"
                       f"DISTRACTOR memory:\n{performance[r['id']]['context']}\n\nRevised question:" for r in failed]
            rewrites = generate(rt, base.REINDEX, prompts, RECIPE['query_rewrite_tokens'])
            checks = [f"Original dialogue:\n{contexts[r['id']]}\nOriginal question: {r['question']}"
                      f"\nReference answer: {r['answer']}\nNew question: {q}\nVerdict:" for r, q in zip(failed, rewrites)]
            verdicts = generate(rt, QUESTION_CHECK, checks, 16)
            lookup = {}
            for row, text, verdict in zip(failed, rewrites, verdicts):
                allowed = (equivalent(verdict) and rt.ntok(text) <= RECIPE['query_rewrite_tokens']
                           and not copies_view(text, forbidden))
                rewrite_checks.append({'id': row['id'], 'text': text, 'verdict': verdict, 'allowed': allowed})
                if allowed:
                    lookup[row['id']] = text.strip()
            for group in groups:
                originals_in_group = [o for o in group if ':rewrite:' not in o['id']]
                known = {o['unit'].get('index_text', o['unit']['text']) for o in group}
                for original in originals_in_group:
                    pid = original['unit']['source_probe_id']
                    if pid not in lookup or lookup[pid] in known:
                        continue
                    option = copy.deepcopy(original)
                    option['id'] += f':rewrite:{round_id}'
                    option['unit']['index_text'] = lookup[pid]
                    option['cost'] = base.storage_cost(option['unit'], rt.ntok)
                    group.append(option)
            base.score_index_routes(rt, parent, groups, fit)
        added, stats = base.allocation(groups, residual, rt.ntok)
        proposal = parent + added
        tested = rehearse(rt, proposal, probes, expected)
        adopted = accept(performance, tested, expected)
        record = {'round': round_id, 'accepted': adopted, 'before': base.means(performance),
                  'after': base.means(tested), 'allocation': stats, 'cumulative_residual': dict(residual),
                  'actual_extra_tokens': sum(base.storage_cost(u, rt.ntok) for u in added),
                  'fit_probe_count': len(fit), 'audit_probe_count': len(audit_probes),
                  'view_manifest_sha256': views['manifest_sha256'], 'rehearsal_contract': expected,
                  'proposal_memory_sha256': core.digest(proposal), 'rehearsal': tested,
                  'candidate_pool': copy.deepcopy(groups), 'rewrite_checks': rewrite_checks}
        if adopted:
            current, performance = proposal, tested
        record['accepted_memory_sha256'] = core.digest(current)
        histories.append(record)
        save_round(round_id, current, record)
    return current, histories
