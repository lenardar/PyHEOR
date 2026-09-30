import pyheor as _ph
from tests.input_helpers import _method, _cycle_values, _starting_values
import numpy as np
import pytest
from openpyxl import load_workbook
from pyheor import MarkovModel, MicroSimModel, PSMModel, Exponential, Beta, Gamma, export_excel_model

def make_model(kind):
    kwargs = dict(states=['Alive', 'Dead'], strategies=['S'], n_cycles=2, cycle=_ph.Cycle(1, 'year'))
    if kind == 'markov':
        model = MarkovModel(**kwargs)
        model.set_transitions('S', [[1, 0], [0, 1]])
    else:
        model = PSMModel(**kwargs, survival_endpoints=['OS'])
        model.set_survival('S', 'OS', Exponential(rate=0.1))
    return model

@pytest.mark.parametrize('kind', ['markov', 'psm'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('reward', ['cost', 'utility'])
def test_nonfinite_rewards_fail(kind, value, reward):
    model = make_model(kind)
    if reward == 'cost':
        model.set_state_cost('care', _cycle_values(model, {'Alive': value}))
    else:
        model.set_state_qaly('health', _cycle_values(model, {'Alive': value}))
    with pytest.raises(ValueError, match='finite'):
        model.run_base_case()

@pytest.mark.parametrize('factory,mean', [(Beta, 0.5), (Gamma, 100)])
@pytest.mark.parametrize('sd', [0, -1, np.nan, np.inf])
def test_invalid_uncertainty_is_not_reinterpreted(factory, mean, sd):
    with pytest.raises(ValueError, match='sd must be finite and positive'):
        factory(mean=mean, sd=sd)

def test_dominated_classification_agrees_across_analyses():
    model = MarkovModel(states=['Alive', 'Dead'], strategies=['SOC', 'Bad'], n_cycles=1, cycle=_ph.Cycle(1, 'year'))
    model.set_transitions('SOC', [[1, 0], [0, 1]])
    model.set_transitions('Bad', [[0, 1], [0, 1]])
    model.add_param('cost', 100, low=80, high=120)
    model.set_starting_cost('init', _starting_values(model, {'Bad': {'Alive': 'cost'}}))
    base = model.run_base_case().icer().iloc[0]
    assert base['ICER Classification'] == 'Dominated'
    assert np.isnan(base['ICER'])
    psa = model.run_psa(n_sim=2, seed=1, progress=False).icer().iloc[0]
    assert np.isnan(psa['ICER'])
    assert psa['ICER Classification'] == 'Dominated'
    owsa = model.run_owsa().summary(outcome='icer').iloc[0]
    for case in ['Low', 'High', 'Base']:
        assert np.isnan(owsa[f'ICER ({case})'])
        assert owsa[f'ICER Classification ({case})'] == 'Dominated'

def test_dominated_classification_agrees_across_microsim_analyses():
    model = MicroSimModel(states=['Alive', 'Dead'], strategies=['SOC', 'Bad'], n_cycles=1, n_patients=20, cycle=_ph.Cycle(1, 'year'))
    model.set_transitions('SOC', [[1, 0], [0, 1]])
    model.set_transitions('Bad', [[0, 1], [0, 1]])
    model.add_param('cost', 100, low=80, high=120)
    model.set_starting_cost('init', _starting_values(model, {'Bad': {'Alive': 'cost'}}))
    base = model.run_base_case(seed=1, verbose=False).icer().iloc[0]
    assert base['ICER Classification'] == 'Dominated'
    assert np.isnan(base['ICER'])
    psa = model.run_psa(n_outer=2, seed=1, verbose=False).icer().iloc[0]
    assert np.isnan(psa['ICER'])
    assert psa['ICER Classification'] == 'Dominated'
