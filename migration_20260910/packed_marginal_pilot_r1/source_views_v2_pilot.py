"""Fixed source-only query variants; no candidate-memory outcome selects views."""
import copy
import json

import refine as core
from crossview_probes import QUESTION_CHECK, WRITER, assign_views, fingerprint
from evidence_fidelity import VERIFIER, accepted
from evidence_utility import POLICY, context_for, reader_user

VIEW_POLICY = {'name': 'source_query_variants_v2', 'generation_tokens': 384,
               'question_tokens': 64, 'reference': 'original source probe answer',
               'fit': 'QA', 'post_lock_diagnostic': 'QB',
               'selection': 'source equivalence and support before memory fitness'}


def generate(rt, system: str, prompts: list[str], tokens: int) -> list[str]:
    outputs = rt.generate(system, prompts, max_tokens=tokens) if prompts else []
    if len(outputs) != len(prompts) or any(not isinstance(x, str) or not x.strip() for x in outputs):
        raise ValueError('Missing source-view generation or verification output')
    return outputs


def _object(pairs: list[tuple]) -> dict:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError('Duplicate JSON field')
    return value


def copies_view(text: str, forbidden: set[tuple]) -> bool:
    words = fingerprint(text)
    return any(len(key) <= len(words) and any(words[i:i + len(key)] == key
               for i in range(len(words) - len(key) + 1)) for key in forbidden if key)


def prepare_views(rt, sessions: list[dict], rows: list[dict], stored: list[dict]) -> dict:
    if not rows or len({r['id'] for r in rows}) != len(rows) or any(r['split'] != 'probe_fit' for r in rows):
        raise ValueError('Expected unique original source-fit probes')
    turns = {t['id']: t for s in sessions for t in s['turns']}
    contexts = {r['id']: context_for(r, turns, r['candidate_context_ids']) for r in rows}
    cues = [u.get('index_text', u['text']) for u in stored] + [r['question'] for r in rows]
    forbidden = {fingerprint(cue) for cue in cues}
    prompts = [f"Original dialogue:\n{contexts[r['id']]}\n\nOriginal question: {r['question']}"
               f"\nReference answer (do not reveal): {r['answer']}\n\nJSON:" for r in rows]
    raw = generate(rt, WRITER, prompts, VIEW_POLICY['generation_tokens'])
    planned, rejected = [], []
    for row, text in zip(rows, raw):
        try:
            value = json.loads(text, object_pairs_hook=_object)
            if not isinstance(value, dict):
                raise ValueError('Question-view JSON must be an object')
            views = assign_views(row['id'], value, forbidden)
            if any(copies_view(q, forbidden) for q in views.values()):
                raise ValueError('Source query view wraps an existing question or cue')
            if any(rt.ntok(q) > VIEW_POLICY['question_tokens'] for q in views.values()):
                raise ValueError('Source query view exceeds question token cap')
            if any(copies_view(cue, {fingerprint(q)}) for cue in cues for q in views.values()):
                raise ValueError('Source query view already occurs inside a stored cue')
            forbidden.update(fingerprint(q) for q in views.values())
            cues.extend(views.values())
            planned.append((row, views))
        except (ValueError, TypeError) as error:
            rejected.append({'id': row['id'], 'reason': str(error), 'generated': text})
    tasks = [(r, role, q) for r, views in planned for role, q in views.items()]
    questions = [f"Original dialogue:\n{contexts[r['id']]}\nOriginal question: {r['question']}"
                 f"\nReference answer: {r['answer']}\nNew question: {q}\nVerdict:" for r, role, q in tasks]
    checks = generate(rt, QUESTION_CHECK, questions, 16)
    answers = generate(rt, core.READER, [reader_user(q, contexts[r['id']]) for r, role, q in tasks], 96)
    prompts = [f"ORIGINAL source:\n{contexts[r['id']]}\n\nQuestion: {q}\nReference answer: {r['answer']}"
               f"\n\nCANDIDATE fragment:\n{contexts[r['id']]}\nCandidate answer: {answer}\n\nVerdict:"
               for (r, role, q), answer in zip(tasks, answers)]
    verdicts = generate(rt, VERIFIER, prompts, 16)
    quality = {}
    for (row, role, question), answer, check, verdict in zip(tasks, answers, checks, verdicts):
        score = core.generic_f1(answer, row['answer'])
        quality.setdefault(row['id'], {})[role] = {'question': question, 'source_prediction': answer,
            'question_verdict': check, 'answer_verdict': verdict, 'f1': score,
            'accepted': accepted(check) and accepted(verdict) and score >= POLICY['minimum_full_answer_f1']}
    records = []
    for row, views in planned:
        if all(quality[row['id']][role]['accepted'] for role in ('fit', 'audit')):
            records.append({'id': row['id'], 'original': copy.deepcopy(row),
                            'fit_question': views['fit'], 'diagnostic_question': views['audit']})
    manifest = {'policy': VIEW_POLICY, 'input_sha256': core.digest(rows),
                'stored_cues_sha256': core.digest(cues[:len(stored) + len(rows)]), 'records': records}
    return {'manifest': manifest, 'manifest_sha256': core.digest(manifest),
            'quality': quality, 'rejected_structure': rejected, 'input_count': len(rows),
            'accepted_count': len(records)}


def validate_bundle(bundle: dict, rows: list[dict]) -> None:
    manifest = bundle['manifest']
    if (bundle['manifest_sha256'] != core.digest(manifest) or manifest['policy'] != VIEW_POLICY
            or manifest['input_sha256'] != core.digest(rows)):
        raise ValueError('Frozen source-view manifest or original probes changed')
    originals = {row['id']: row for row in rows}
    records = manifest['records']
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Missing or duplicate source view IDs')
    for record in records:
        if record['original'] != originals.get(record['id']):
            raise ValueError('View changes original question, answer, split or provenance')


def fit_rows(bundle: dict, rows: list[dict]) -> list[dict]:
    validate_bundle(bundle, rows)
    return [{**copy.deepcopy(r['original']), 'question': r['fit_question'],
             'view': 'fit', 'original_question': r['original']['question']}
            for r in bundle['manifest']['records']]


def forbidden_views(bundle: dict) -> set[tuple]:
    return {fingerprint(r[field]) for r in bundle['manifest']['records']
            for field in ('fit_question', 'diagnostic_question')}
