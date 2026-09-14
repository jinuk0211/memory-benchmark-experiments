"""CPU-only review artifact: exact retained-output parity for parent_single."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def read(path: Path):
    return json.loads(path.read_text())


def reduce_row(row: dict, utility) -> dict:
    if 'analysis' not in row:
        return copy.deepcopy(row)
    jobs = [copy.deepcopy(s) for s in row['subsets'] if s['kind'] in ('empty', 'full', 'single')]
    assert [s['kind'] for s in jobs[:2]] == ['empty', 'full']
    empty, full = jobs[:2]
    singles = jobs[2:]
    assert all(s['kind'] == 'single' for s in singles)
    def lp(s: dict) -> float:
        return s['score']['mean_logprob']
    best = max(singles, key=lp)
    eligible = [s for s in singles + [full] if lp(s) >= lp(full) - utility.POLICY['preservation_tolerance_nats']]
    minimum = min(eligible, key=lambda s: (s['context_tokens'], -lp(s), s['source_ids']))
    dependent = lp(full) - lp(empty) >= utility.POLICY['minimum_full_advantage_nats']
    assert row['analysis']['source_dependent'] == dependent
    assert row['generations']['best_single']['source_ids'] == best['source_ids']
    assert row['generations']['single_policy']['source_ids'] == minimum['source_ids']
    generations = {key: copy.deepcopy(row['generations'][key]) for key in ('full', 'best_single', 'single_policy', 'answer_removed')}
    return {**copy.deepcopy(row), 'subsets': jobs, 'analysis': {'source_dependent': dependent}, 'generations': generations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code))
    import refine as core
    import evidence_utility as utility
    import parent_evidence as parent_code
    from transformers import AutoTokenizer

    run = args.source_run
    protocol = read(run / 'protocol.json')
    module_names = ('refine.py', 'evidence_utility.py', 'budgeted_evidence.py', 'parent_evidence.py', 'portable_parent.py')
    source_hashes = {name: hashlib.sha256((args.code / name).read_bytes()).hexdigest() for name in module_names}
    for name, digest in source_hashes.items():
        assert protocol['source_sha256'][name] == digest, name
    model = protocol['args']['model']
    metadata = protocol['environment']['models'][model]
    tokenizer = AutoTokenizer.from_pretrained(metadata['path'], local_files_only=True, trust_remote_code=False)

    class TokenRuntime:
        @staticmethod
        def ntok(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

    runtime = TokenRuntime()
    sessions = read(run / 'source_sessions.json')['sessions_by_id']
    selected = read(run / 'utility_selection_locked.json')['selected_ids']
    assert len(selected) == len(set(selected))
    by_id = {}
    files = []
    for path in sorted((run / 'utility/items').glob('*.json')):
        row = read(path)
        assert row['id'] not in by_id
        by_id[row['id']] = row
        files.append(path)
    assert set(selected) <= by_id.keys()
    rows = [by_id[qid] for qid in selected]
    assert all(row['split'] == 'probe_fit' for row in rows)
    reduced = [reduce_row(row, utility) for row in rows]
    result = {'status': 'passed', 'condition': 'Exact retained NLL and generation outputs reused; no GPU batch-invariance claim', 'source_run': str(run), 'source_code_sha256': source_hashes, 'tokenizer': metadata, 'selected_source_rows': len(rows), 'scored_source_rows': sum('analysis' in row for row in rows), 'conversations': {}}
    for cid, history in sorted(sessions.items()):
        parent = read(run / 'memories/r40_fused_four_turn' / (cid + '.json'))
        old_rows = [row for row in rows if row['conv_id'] == cid]
        new_rows = [row for row in reduced if row['conv_id'] == cid]
        old_groups, old_mapped = parent_code.make_options(runtime, history, parent, old_rows)
        new_groups, new_mapped = parent_code.make_options(runtime, history, parent, new_rows)
        assert [[o for o in group if o['kind'] != 'pair'] for group in old_groups] == new_groups
        assert {key: value for key, value in old_mapped.items() if value['kind'] != 'pair'} == new_mapped
        checked = []
        for budget in (0, 8, 64, 512, 2000):
            old = parent_code.construct(parent, old_groups, old_mapped, {}, {}, runtime.ntok, 'parent_single', budget)
            new = parent_code.construct(parent, new_groups, new_mapped, {}, {}, runtime.ntok, 'parent_single', budget)
            assert old == new, (cid, budget)
            checked.append(budget)
            if budget == 2000:
                assert old[0] == read(run / 'memories/s_parent_single_2000' / (cid + '.json'))
                recorded_details = read(run / 'construction/s_parent_single_2000' / (cid + '.json'))
                assert old[1] == recorded_details
                result['conversations'][cid] = {'rows': len(old_rows), 'budgets_checked': checked, 'memory_sha256': core.digest(old[0]), 'details_sha256': core.digest(old[1]), 'selected_options': old[1]['selected_options'], 'matches_original_recorded_memory_and_details': True}
        print(json.dumps({'checked_conversation': cid, 'source_rows': len(old_rows)}), flush=True)
    result['inputs_sha256'] = {str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files + [run / 'source_sessions.json', run / 'utility_selection_locked.json', run / 'protocol.json']}
    core.save(args.out, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'inputs_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()

