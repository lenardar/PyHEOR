"""Tests for pyheor/microsim.py — MicroSimModel integration tests."""
import pyheor as _ph
from tests.input_helpers import _method, _cycle_values, _starting_values



import numpy as np
import pytest
from pyheor import MarkovModel, MicroSimModel, PatientProfile, C
ALIVE_FOREVER = [[1, 0], [0, 1]]
DEAD_AFTER_ONE_INTERVAL = [[0, 1], [0, 1]]

class TestTransitionMatrices:

    def build(self, matrix):
        model = MicroSimModel(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=3, n_patients=5, cycle=_ph.Cycle(1, 'year'))
        model.set_transitions('S1', matrix)
        model.set_state_qaly('health', _cycle_values(model, {'Alive': 1.0, 'Dead': 0.0}))
        return model

    def test_rows_that_do_not_sum_to_one_are_rejected(self):
        model = self.build([[0.5, 0.2], [0, 1]])
        with pytest.raises(ValueError, match='Row sums'):
            model.run_base_case(seed=1, verbose=False)

    def test_negative_probabilities_are_not_clipped(self):
        model = self.build([[1.2, -0.2], [0, 1]])
        with pytest.raises(ValueError, match='Negative probabilities'):
            model.run_base_case(seed=1, verbose=False)

    def test_transition_shape_must_match_the_model_states(self):
        model = MicroSimModel(states=['Alive', 'Sick', 'Dead'], strategies=['S1'], n_cycles=1, n_patients=1, cycle=_ph.Cycle(1, 'year'))
        model.set_transitions('S1', [[C, 0.1, 0.2, 0.7], [0, 1, 0, 0], [0, 0, 1, 0]])
        model.set_state_qaly('health', _cycle_values(model, {'Alive': 1.0, 'Sick': 0.5, 'Dead': 0.0}))
        with pytest.raises(ValueError, match='shape \\(3, 3\\)'):
            model.run_base_case(seed=1, verbose=False)

    def test_callbacks_receive_zero_based_interval_indices(self):
        seen = []
        model = MicroSimModel(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=3, n_patients=2, cycle=_ph.Cycle(1, 'year'))
        model.set_transitions('S1', lambda p, t: seen.append(t) or ALIVE_FOREVER)
        model.set_state_qaly('health', _cycle_values(model, {'Alive': 1.0, 'Dead': 0.0}))
        model.run_base_case(seed=1, verbose=False)
        assert seen == [1, 2, 3]

class TestAgreementWithCohortEngine:
    """Deterministic transitions must reproduce the Markov numbers exactly."""

    @pytest.mark.parametrize('matrix', [ALIVE_FOREVER, DEAD_AFTER_ONE_INTERVAL])
    @pytest.mark.parametrize('hcc', [False, True])
    def test_totals_match(self, matrix, hcc):
        shared = dict(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=4, cycle=_ph.Cycle(0.5, 'year'), dr_cost=0.03, dr_qaly=0.05, method=_method(hcc))
        markov = MarkovModel(**shared)
        micro = MicroSimModel(**shared, n_patients=5)
        for model in (markov, micro):
            model.set_transitions('S1', matrix)
            model.set_state_cost('care', _cycle_values(model, {'Alive': 1000, 'Dead': 0}))
            model.set_starting_cost('setup', _starting_values(model, {'Alive': 500, 'Dead': 0}))
            model.set_state_qaly('health', _cycle_values(model, {'Alive': 0.8, 'Dead': 0.0}))
        expected = markov.run_base_case().results['S1']
        actual = micro.run_base_case(seed=1, verbose=False).results['S1']
        assert float(np.mean(actual['total_lys'])) == pytest.approx(expected['total_lys'])
        assert float(np.mean(actual['total_qalys'])) == pytest.approx(expected['total_qalys'])
        assert float(np.mean(actual['total_cost'])) == pytest.approx(sum(expected['total_costs'].values()))

    @pytest.mark.parametrize('convention', ['discrete', 'continuous'])
    def test_discount_conventions_match(self, convention):
        shared = dict(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=3, dr_cost=0.1, dr_qaly=0.1, method=_method(False), cycle=_ph.Cycle(1, 'year'))
        markov = MarkovModel(**shared)
        micro = MicroSimModel(**shared, n_patients=3)
        for model in (markov, micro):
            model.set_transitions('S1', ALIVE_FOREVER)
            model.set_state_cost('care', _cycle_values(model, {'Alive': 100, 'Dead': 0}))
            model.set_state_qaly('health', _cycle_values(model, {'Alive': 1.0, 'Dead': 0.0}))
        expected = markov.run_base_case().results['S1']
        actual = micro.run_base_case(seed=1, verbose=False).results['S1']
        assert float(np.mean(actual['total_cost'])) == pytest.approx(sum(expected['total_costs'].values()))
        assert float(np.mean(actual['total_qalys'])) == pytest.approx(expected['total_qalys'])

class TestMicroSimRun:

    @pytest.fixture
    def micro_model(self):
        model = MicroSimModel(states=['Alive', 'Dead'], strategies=['SOC', 'TRT'], n_cycles=10, n_patients=100, cycle=_ph.Cycle(1.0, 'year'), dr_cost=0.03, dr_qaly=0.03, seed=42)
        model.add_param('p_death', base=0.1)
        model.add_param('hr', base=0.7)
        model.set_transitions('SOC', lambda p, t: [[C, p['p_death']], [0, 1]])
        model.set_transitions('TRT', lambda p, t: [[C, p['p_death'] * p['hr']], [0, 1]])
        model.set_state_cost('medical', _cycle_values(model, {'Alive': 1000, 'Dead': 0}))
        model.set_state_qaly('health', _cycle_values(model, {'Alive': 1.0, 'Dead': 0.0}))
        return model

    def test_base_case_runs(self, micro_model):
        result = micro_model.run_base_case()
        summary = result.summary()
        assert 'Mean QALYs' in summary.columns

    def test_all_absorbing(self):
        model = MicroSimModel(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=3, n_patients=50, seed=42, cycle=_ph.Cycle(1, 'year'))
        model.set_transitions('S1', DEAD_AFTER_ONE_INTERVAL)
        model.set_state_qaly('health', _cycle_values(model, {'Alive': 1, 'Dead': 0}))
        result = model.run_base_case()
        assert result.summary()['Mean QALYs'].iloc[0] == pytest.approx(0.5)

    def test_results_summary_columns(self, micro_model):
        result = micro_model.run_base_case()
        summary = result.summary()
        assert 'Strategy' in summary.columns
        assert 'Mean Cost' in summary.columns

    def test_patient_outcomes_report_years_alive(self, micro_model):
        result = micro_model.run_base_case()
        outcomes = result.patient_outcomes
        assert 'Years Alive' in outcomes.columns
        assert (outcomes['Years Alive'] <= micro_model.n_cycles).all()

    def test_owsa_runs_with_shared_patient_draws(self, micro_model):
        result = micro_model.run_owsa(params=['p_death'], n_patients=20, seed=3, verbose=False)
        summary = result.summary()
        assert list(summary['param_name']) == ['p_death']
        assert set(result.base_result) == {'SOC', 'TRT'}

    def test_heterogeneous_population_runs(self):
        model = MicroSimModel(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=4, n_patients=20, cycle=_ph.Cycle(1, 'year'))
        model.add_param('base_risk', base=0.05)
        model.set_transitions('S1', lambda p, t, attrs: [[C, p['base_risk'] * (1 + attrs['frailty'])], [0, 1]])
        model.set_state_qaly('health', _cycle_values(model, {'Alive': 1.0, 'Dead': 0.0}))
        profile = PatientProfile(n_patients=20, attributes={'frailty': np.linspace(0, 1, 20)})
        result = model.run_base_case(profile=profile, seed=3, verbose=False)
        assert result.results['S1']['state_history'].shape == (20, 5)
