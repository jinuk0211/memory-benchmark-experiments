import copy
import pytest
from candidate_gate import decision, METHOD


def fixture():
    return ({'methods': {METHOD: {'n': 1540, 'official_f1': .60},
                        's_parent_single_2000': {'n': 1540, 'official_f1': .57}},
             'paired': {'development7': {'effect_pp': 1.0}}},
            {'n': 1540, 'official_f1': .5695671313240028,
             'predictions_complete': True, 'coverage_complete': True})


def test_transfer_requires_actual_baseline_win_and_development_gain():
    report, reference = fixture()
    assert decision(report, reference)['ready_for_transfer']
    for score in (.56, .57):
        changed = copy.deepcopy(report)
        changed['methods'][METHOD]['official_f1'] = score
        assert not decision(changed, reference)['ready_for_transfer']
    report['paired']['development7']['effect_pp'] = 0
    assert not decision(report, reference)['ready_for_transfer']


def test_incomplete_or_nonfinite_results_cannot_launch():
    report, reference = fixture()
    reference['n'] = 100
    with pytest.raises(ValueError, match='Incomplete'):
        decision(report, reference)
    reference['n'] = 1540
    report['methods'][METHOD]['official_f1'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        decision(report, reference)
