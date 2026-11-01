"""Prepared structures must not freeze dynamic values or leak across runs."""
import numpy as np
import pytest
import pyheor as ph


def model():
    m = ph.MarkovModel(states=['Alive', 'Dead'], strategies=['A'],
                       cycle=ph.Cycle(1, 'month'), n_cycles=4,
                       dr_cost=0, dr_qaly=0, method='end')
    m.set_transitions('A', [[1, 0], [0, 1]])
    return m


def test_unhashable_callback_remains_time_varying_and_is_inspected_once(monkeypatch):
    import inspect
    class Reward:
        __hash__ = None
        def __call__(self, params, cycle):
            return params['amount'] * cycle
    reward = Reward()
    m = model().add_param('amount', 2)
    m.set_state_cost('care', {'Alive': reward})
    original = inspect.signature
    calls = []
    def signature(obj, *args, **kwargs):
        if obj is reward:
            calls.append(obj)
        return original(obj, *args, **kwargs)
    monkeypatch.setattr(inspect, 'signature', signature)
    result = m.run_base_case().results['A']
    np.testing.assert_array_equal(result['costs_by_cycle']['care'], [2, 4, 6, 8])
    assert len(calls) == 1
    m.params['amount'].base = 3
    assert m.run_base_case().results['A']['total_costs']['care'] == 30
    assert len(calls) == 2


def test_mapping_edits_between_runs_are_not_cached():
    m = model()
    values = {'A': {'Alive': 2}}
    m.set_state_cost('care', values)
    assert m.run_base_case().results['A']['total_costs']['care'] == 8
    values['A']['Alive'] = 5
    assert m.run_base_case().results['A']['total_costs']['care'] == 20
    values['A']['Typo'] = 1
    with pytest.raises(ValueError, match='unknown states'):
        m.run_base_case()


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_dynamic_scalar_is_checked_after_first_cycle(bad):
    m = model()
    m.set_state_cost('care', {'Alive': lambda p, t: 2 if t == 1 else bad})
    with pytest.raises(ValueError, match='finite'):
        m.run_base_case()


def test_dynamic_mapping_keys_are_checked_every_cycle():
    m = model()
    m.set_state_cost('care', lambda p, t: {'Alive' if t == 1 else 'Typo': 2})
    with pytest.raises(ValueError, match='unknown state'):
        m.run_base_case()


def test_dynamic_mapping_can_change_states_and_callback_arity():
    m = model()
    m.set_state_cost('care', lambda p, t: {'Alive': (lambda p: 2)} if t == 1
                     else {'Alive': lambda p, t, attrs: t, 'Dead': 9})
    np.testing.assert_array_equal(
        m.run_base_case().results['A']['costs_by_cycle']['care'], [2, 2, 3, 4])


@pytest.mark.parametrize('container', [list, np.array])
def test_fixed_matrix_matches_dynamic_and_refreshes_between_runs(container):
    m = model()
    matrix = container([[.8, .2], [0, 1]])
    m.set_transitions('A', matrix)
    fixed = m.run_base_case().results['A']['trace']
    m.set_transitions('A', lambda p, t: matrix)
    np.testing.assert_array_equal(m.run_base_case().results['A']['trace'], fixed)
    m.set_transitions('A', matrix)
    matrix[0][0], matrix[0][1] = .5, .5
    np.testing.assert_allclose(m.run_base_case().results['A']['trace'][:, 0], .5 ** np.arange(5))
    matrix[0][0] = -.1
    with pytest.raises(ValueError, match='Invalid transition matrix'):
        m.run_base_case()


@pytest.mark.parametrize('whole_matrix', [False, True])
def test_dynamic_matrix_is_still_checked_each_cycle(whole_matrix):
    m = model()
    if whole_matrix:
        matrix = lambda p, t: [[1, 0], [0, 1]] if t <= 1 else [[-1, 2], [0, 1]]
    else:
        matrix = [[lambda p, t: 1 if t <= 1 else -1, ph.C], [0, 1]]
    m.set_transitions('A', matrix)
    with pytest.raises(ValueError, match='interval 2'):
        m.run_base_case()
