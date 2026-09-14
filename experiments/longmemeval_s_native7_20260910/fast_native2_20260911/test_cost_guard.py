from cost_guard import decide


def sample():
    return {'free_bytes': 9 * 1024**3, 'all_terminal': False, 'finish': None,
            'generated': {'simplemem': 1, 'lightmem': 1}}


def test_does_not_stop_progressing_method_when_other_fails():
    assert decide(sample(), {'started_at': 0, 'last_progress': 2000}, 2100) is None


def test_stall_stops_after_bound():
    assert '30_minutes' in decide(sample(), {'started_at': 0, 'last_progress': 0}, 1801)


def test_terminal_grace_and_incomplete():
    obs = sample(); obs['all_terminal'] = True
    state = {'started_at': 0, 'last_progress': 2000, 'terminal_since': 2000}
    assert decide(obs, state, 2599) is None
    assert decide(obs, state, 2601) == 'both_queues_terminal_without_verified_completion'


def test_low_disk_even_during_startup():
    obs = sample(); obs['free_bytes'] = 1
    assert decide(obs, {'started_at': 0, 'last_progress': 0}, 1).startswith('low_disk')


def test_completion_verification_has_one_hour():
    obs = sample(); obs['generated'] = {'simplemem': 500, 'lightmem': 500}
    assert decide(obs, {'started_at': 0, 'last_progress': 0}, 2000) is None
    assert decide(obs, {'started_at': 0, 'last_progress': 0}, 3601)


def test_absolute_deadline_despite_response_activity():
    assert decide(sample(), {'started_at': 0, 'last_progress': 300000}, 300000) == 'maximum_72_hour_budget'
