"""Pure old-first composition of caller-validated canonical SimpleMem predictions.

The outer finalizer must verify raw prompts, original/global/context identities,
paths, hashes and closed sources, then audit coverage, numeric usage, delivery and
semantic discards. This module neither scores nor proves raw-artifact integrity.
"""
from copy import deepcopy
from typing import Any

from scripts.score_locomo_comparison import validate_predictions


def _scope(expected: dict, old_run_id: str, new_run_id: str,
           repair_run_id: str | None = None) -> dict[str, str]:
    runs = {'old': old_run_id, 'new': new_run_id}
    if repair_run_id is not None:
        runs['repair'] = repair_run_id
    if (any(not isinstance(run, str) or not run.strip() for run in runs.values())
            or len(set(runs.values())) != len(runs)):
        raise ValueError('Run IDs must be nonempty and distinct')
    if len(expected) != 1540:
        raise ValueError('The expected original scope must contain exactly 1540 QA')
    return runs


def compose_predictions(
    old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], expected: dict,
    old_run_id: str, new_run_id: str, *, repair_rows: list[dict[str, Any]] | None = None,
    repair_run_id: str | None = None,
) -> dict[str, Any]:
    """Preserve every old success before scoring; fill only absent keys from new."""
    if (repair_rows is None) != (repair_run_id is None):
        raise ValueError('Third-run rows and run ID must both be explicit')
    runs = _scope(expected, old_run_id, new_run_id, repair_run_id)
    candidates = {}
    supplied = {'old': old_rows, 'new': new_rows}
    if repair_rows is not None:
        supplied['repair'] = repair_rows
    for source, rows in supplied.items():
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError('Prediction candidates must be objects')
        _, issues, _ = validate_predictions(rows, expected)
        if issues:
            raise ValueError(f'Invalid {source} prediction candidates: {issues}')
        if any(not row['prediction'].strip() for row in rows):
            raise ValueError(f'Empty {source} prediction candidate')
        candidates[source] = {(row['sample'], row['index']): (number, row)
                              for number, row in enumerate(rows)}
    if 'repair' in candidates and (len(candidates['repair']) != 3
            or set(candidates['repair']) & (set(candidates['old']) | set(candidates['new']))):
        raise ValueError('The explicit QA-only repair must fill exactly three absent keys')
    if set().union(*candidates.values()) != set(expected):
        raise ValueError('Missing predictions in the original 1540 QA union')
    predictions, provenance = [], []
    for sample, index in expected:
        key = (sample, index)
        source = next(name for name in runs if key in candidates[name])
        number, row = candidates[source][key]
        predictions.append(deepcopy(row))
        provenance.append({'sample': sample, 'index': index, 'source': source,
                           'run_id': runs[source], 'source_row_index': number})
    return {'predictions': predictions, 'provenance': provenance,
            'policy': 'Old successful predictions take precedence before any scoring; '
                      'the outer finalizer must independently audit raw artifacts.'}


def partition_usage(
    usage_rows: list[dict[str, Any]], provenance: list[dict[str, Any]], expected: dict,
    old_run_id: str, new_run_id: str, *, repair_run_id: str | None = None,
) -> dict[str, Any]:
    """Partition existing calls, retaining every attempt and mixed construction.

    No token values are inferred, normalized, added or proportionally allocated.
    Attribution of every row is validated, including rows that will be excluded.
    """
    runs = _scope(expected, old_run_id, new_run_id, repair_run_id)
    chosen = {}
    for item in provenance:
        if (not isinstance(item, dict) or not isinstance(item.get('sample'), str)
                or type(item.get('index')) is not int):
            raise ValueError('Invalid chosen-source identity')
        key = (item['sample'], item['index'])
        source = item.get('source')
        if (key not in expected or key in chosen or source not in runs
                or item.get('run_id') != runs[source]):
            raise ValueError('Ambiguous chosen-source provenance')
        chosen[key] = item['run_id']
    if set(chosen) != set(expected):
        raise ValueError('Chosen-source provenance must cover the original 1540 QA')
    construction = {(run, sample) for (sample, _), run in chosen.items()}
    repair_keys = {key for key, run in chosen.items() if run == repair_run_id}
    if repair_run_id is not None:
        if len(repair_keys) != 3:
            raise ValueError('Expected exactly three chosen QA-only repairs')
        construction.update((new_run_id, sample) for sample, _ in repair_keys)
    samples = {sample for sample, _ in expected}
    request_ids, selections = set(), []
    for row in usage_rows:
        if not isinstance(row, dict):
            raise ValueError('Usage rows must be objects')  # noqa: TRY004 - invalid journal
        request_id = row.get('request_id')
        if (not isinstance(request_id, str) or not request_id.strip()
                or request_id in request_ids):
            raise ValueError('Missing or duplicate request ID')
        request_ids.add(request_id)
        run, sample, phase = row.get('run_id'), row.get('sample_id'), row.get('phase')
        if (row.get('method') != 'simplemem' or run not in runs.values()
                or not isinstance(sample, str) or sample not in samples
                or phase not in ('qa', 'initialize', 'memory_add', 'memory_finalize')
                or row.get('request_kind') not in ('chat_completion', 'embedding')):
            raise ValueError('Foreign or invalid usage attribution')
        if run == repair_run_id and (phase not in ('qa', 'initialize')
                or sample not in {key[0] for key in repair_keys}):
            raise ValueError('QA-only repair may reuse, never rebuild, r17 construction')
        if phase == 'qa':
            question = row.get('question_id')
            if type(question) is int:
                index = question
            elif (isinstance(question, str) and question.isascii()
                  and question.isdecimal() and str(int(question)) == question):
                index = int(question)
            else:
                raise ValueError('QA usage requires an original integer question index')
            key = (sample, index)
            if key not in expected:
                raise ValueError('QA usage is outside the original evaluation set')
            if run == repair_run_id and key not in repair_keys:
                raise ValueError('Third-run usage contains an unselected QA')
            selections.append(chosen[key] == run)
        else:
            selections.append((run, sample) in construction)
    return {
        'selected': [deepcopy(row) for row, selected in zip(usage_rows, selections) if selected],
        'excluded': [deepcopy(row) for row, selected in zip(usage_rows, selections) if not selected],
        'note': 'Gross usage remains all supplied rows. The outer finalizer must audit '
                'numeric usage, coverage, delivery, semantic discards and closed raw '
                'sources. Mixed contexts retain both chosen runs construction calls; '
                'no proportional cost allocation is performed.',
    }
