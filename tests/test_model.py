"""Tests for pyheor/model.py — MarkovModel full workflow."""
import pyheor as _ph
from tests.input_helpers import _method, _cycle_values, _starting_values



import numpy as np
import pytest
from pyheor import MarkovModel, C, Beta, Distribution, Gamma
from pyheor.analysis.results import BaseResult, OWSAResult, PSAResult

class _LegacyDistribution(Distribution):
    """Third-party style distribution implementing the historical API."""

    def sample(self, n=1):
        return np.random.uniform(0.1, 0.9, size=n)

    def __repr__(self):
        return 'LegacyDistribution()'

class TestModelConstruction:

    def test_basic(self):
        model = MarkovModel(states=['A', 'B'], strategies=['S1'], n_cycles=5, cycle=_ph.Cycle(1, 'year'))
        assert model.states == ['A', 'B']
        assert len(model.strategy_names) == 1
        assert model.n_cycles == 5

    def test_add_param(self):
        model = MarkovModel(states=['A', 'B'], strategies=['S1'], n_cycles=5, cycle=_ph.Cycle(1, 'year'))
        model.add_param('p', base=0.5)
        assert 'p' in model.params

    def test_owsa_defaults(self):
        model = MarkovModel(states=['A', 'B'], strategies=['S1'], n_cycles=5, cycle=_ph.Cycle(1, 'year'))
        model.add_param('p', base=0.5)
        param = model.params['p']
        assert param.low == pytest.approx(0.4)
        assert param.high == pytest.approx(0.6)

class TestTraceInvariants:

    def test_rows_sum_to_one(self, simple_markov_model):
        result = simple_markov_model.run_base_case()
        for strat in simple_markov_model.strategy_names:
            trace = result.results[strat]['trace']
            row_sums = trace.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_initial_state(self, simple_markov_model):
        result = simple_markov_model.run_base_case()
        for strat in simple_markov_model.strategy_names:
            trace = result.results[strat]['trace']
            np.testing.assert_allclose(trace[0], [1, 0, 0])

    def test_all_dead_model(self):
        model = MarkovModel(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=5, method=_method(False), cycle=_ph.Cycle(1, 'year'))
        model.set_transitions('S1', lambda p, t: [[0, 1], [0, 1]])
        model.set_state_qaly('health', _cycle_values(model, {'Alive': 1, 'Dead': 0}))
        result = model.run_base_case()
        trace = result.results['S1']['trace']
        np.testing.assert_allclose(trace[1], [0, 1])
        np.testing.assert_allclose(trace[-1], [0, 1])

    def test_identity_transitions(self):
        model = MarkovModel(states=['A', 'B'], strategies=['S1'], n_cycles=5, method=_method(False), cycle=_ph.Cycle(1, 'year'))
        model.set_transitions('S1', lambda p, t: [[1, 0], [0, 1]])
        model.set_state_qaly('health', _cycle_values(model, {'A': 1, 'B': 0}))
        result = model.run_base_case()
        trace = result.results['S1']['trace']
        for k in range(trace.shape[0]):
            np.testing.assert_allclose(trace[k], [1, 0])

class TestResults:

    def test_base_case_returns_base_result(self, simple_markov_model):
        result = simple_markov_model.run_base_case()
        assert isinstance(result, BaseResult)

    def test_summary_columns(self, simple_markov_model):
        result = simple_markov_model.run_base_case()
        summary = result.summary()
        assert 'Strategy' in summary.columns
        assert 'QALYs' in summary.columns
        assert 'Total Cost' in summary.columns

    def test_icer(self, simple_markov_model):
        result = simple_markov_model.run_base_case()
        icer_df = result.icer()
        assert 'ICER' in icer_df.columns

    def test_hcc_effect(self):
        """HCC on vs off should produce different QALYs for non-trivial model."""

        def make_model(hcc):
            model = MarkovModel(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=10, method=_method(hcc), cycle=_ph.Cycle(1, 'year'))
            model.set_transitions('S1', lambda p, t: [[0.9, 0.1], [0, 1]])
            model.set_state_qaly('health', _cycle_values(model, {'Alive': 1.0, 'Dead': 0.0}))
            return model
        r_on = make_model(True).run_base_case()
        r_off = make_model(False).run_base_case()
        q_on = r_on.summary()['QALYs'].iloc[0]
        q_off = r_off.summary()['QALYs'].iloc[0]
        assert q_on != q_off

    def test_discount_effect(self):
        """Higher discount -> lower QALYs."""

        def make_model(dr):
            model = MarkovModel(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=20, dr_cost=dr, dr_qaly=dr, method=_method(False), cycle=_ph.Cycle(1, 'year'))
            model.set_transitions('S1', lambda p, t: [[0.95, 0.05], [0, 1]])
            model.set_state_qaly('health', _cycle_values(model, {'Alive': 1.0, 'Dead': 0.0}))
            return model
        q_low = make_model(0.0).run_base_case().summary()['QALYs'].iloc[0]
        q_high = make_model(0.1).run_base_case().summary()['QALYs'].iloc[0]
        assert q_high < q_low

class TestCosts:

    def test_first_cycle_only(self):
        model = MarkovModel(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=5, method=_method(False), cycle=_ph.Cycle(1, 'year'))
        model.set_transitions('S1', lambda p, t: [[1, 0], [0, 1]])
        model.set_state_cost('init', _cycle_values(model, {'Alive': 10000, 'Dead': 0}), cycles=1)
        model.set_state_cost('ongoing', _cycle_values(model, {'Alive': 1000, 'Dead': 0}))
        model.set_state_qaly('health', _cycle_values(model, {'Alive': 1.0, 'Dead': 0.0}))
        result = model.run_base_case()
        total = result.summary()['Total Cost'].iloc[0]
        np.testing.assert_allclose(total, 15000, rtol=0.01)

class TestSensitivityAnalysis:

    def test_owsa_returns_result(self, simple_markov_model):
        owsa = simple_markov_model.run_owsa()
        assert isinstance(owsa, OWSAResult)

    def test_psa_returns_result(self, simple_markov_model):
        psa = simple_markov_model.run_psa(n_sim=5, seed=42, progress=False)
        assert isinstance(psa, PSAResult)

    def test_psa_deterministic_with_seed(self, simple_markov_model):
        r1 = simple_markov_model.run_psa(n_sim=5, seed=123, progress=False)
        r2 = simple_markov_model.run_psa(n_sim=5, seed=123, progress=False)
        s1 = r1.summary()['Mean QALYs'].values
        s2 = r2.summary()['Mean QALYs'].values
        np.testing.assert_allclose(s1, s2)

    def test_psa_does_not_change_global_numpy_rng(self, simple_markov_model):
        np.random.seed(2026)
        expected = np.random.random()
        np.random.seed(2026)
        simple_markov_model.run_psa(n_sim=2, seed=123, progress=False)
        assert np.random.random() == pytest.approx(expected)

    def test_psa_rejects_empty_simulation(self, simple_markov_model):
        with pytest.raises(ValueError, match='positive integer'):
            simple_markov_model.run_psa(n_sim=0, progress=False)

    def test_psa_supports_legacy_distribution_without_global_rng_leak(self):
        model = MarkovModel(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=1, cycle=_ph.Cycle(1, 'year'))
        model.add_param('p', base=0.5, dist=_LegacyDistribution())
        model.set_transitions('S1', lambda p, t: [[1 - p['p'], p['p']], [0, 1]])
        np.random.seed(2026)
        expected = np.random.random()
        np.random.seed(2026)
        first = model.run_psa(n_sim=3, seed=7, progress=False)
        observed = np.random.random()
        second = model.run_psa(n_sim=3, seed=7, progress=False)
        assert observed == pytest.approx(expected)
        assert first.sampled_params == second.sampled_params

    def test_owsa_icer_ranking(self, simple_markov_model):
        """ICER-based ranking may differ from NMB-based ranking."""
        owsa = simple_markov_model.run_owsa()
        nmb_summary = owsa.summary(outcome='nmb')
        icer_summary = owsa.summary(outcome='icer')
        assert set(nmb_summary['param_name']) == set(icer_summary['param_name'])
        assert 'ICER (Low)' in icer_summary.columns
        assert 'ICER (High)' in icer_summary.columns
        assert 'ICER (Base)' in icer_summary.columns

    def test_owsa_discount_rate_param(self):
        """Discount rate can be varied in OWSA via Param."""
        from pyheor import Param
        model = MarkovModel(states=['Alive', 'Dead'], strategies=['S1', 'S2'], n_cycles=10, dr_cost=Param(0.05, low=0.0, high=0.08), dr_qaly=Param(0.05, low=0.0, high=0.08), method=_method(False), cycle=_ph.Cycle(1, 'year'))
        model.add_param('p_death', base=0.1, low=0.05, high=0.15)
        model.set_transitions('S1', lambda p, t: [[1 - p['p_death'], p['p_death']], [0, 1]])
        model.set_transitions('S2', lambda p, t: [[1 - p['p_death'] * 0.8, p['p_death'] * 0.8], [0, 1]])
        model.set_state_cost('drug', _cycle_values(model, {'S1': {'Alive': 1000, 'Dead': 0}, 'S2': {'Alive': 5000, 'Dead': 0}}))
        model.set_state_qaly('health', _cycle_values(model, {'Alive': 1.0, 'Dead': 0.0}))
        owsa = model.run_owsa(params=['dr_cost', 'p_death'])
        summary = owsa.summary()
        assert 'dr_cost' in summary['param_name'].values
        dr_row = summary[summary['param_name'] == 'dr_cost'].iloc[0]
        assert dr_row['INMB (Low)'] != dr_row['INMB (High)']
        assert model.dr_cost == 0.05
        assert model.dr_qaly == 0.05

class TestTrapezoidalHCC:

    @staticmethod
    def _make_model(hcc, n_cycles=10, dr_cost=0, dr_qaly=0):
        m = MarkovModel(states=['Alive', 'Dead'], strategies=['S1'], n_cycles=n_cycles, method=_method(hcc), dr_cost=dr_cost, dr_qaly=dr_qaly, cycle=_ph.Cycle(1, 'year'))
        m.set_transitions('S1', lambda p, t: [[0.9, 0.1], [0, 1]])
        m.set_state_qaly('health', _cycle_values(m, {'Alive': 1.0, 'Dead': 0.0}))
        m.set_state_cost('drug', _cycle_values(m, {'Alive': 1000, 'Dead': 0}))
        return m

    def test_trapezoidal_differs_from_no_hcc(self):
        r_lt = self._make_model('trapezoidal').run_base_case()
        r_none = self._make_model(None).run_base_case()
        q_lt = r_lt.summary()['QALYs'].iloc[0]
        q_none = r_none.summary()['QALYs'].iloc[0]
        assert q_lt != q_none

    def test_trapezoidal_manual_verification(self):
        """Verify against hand-computed corrected trace."""
        model = self._make_model('trapezoidal')
        result = model.run_base_case()
        trace = result.results['S1']['trace']
        qalys_hcc = result.results['S1']['qalys_hcc']
        assert len(qalys_hcc) == model.n_cycles
        for t in range(model.n_cycles):
            corrected_alive = (trace[t, 0] + trace[t + 1, 0]) / 2.0
            expected_qaly = corrected_alive * model.cycle_length
            np.testing.assert_allclose(qalys_hcc[t], expected_qaly, atol=1e-10)

    def test_trapezoidal_costs_manual(self):
        """Verify trapezoidal corrected costs."""
        model = self._make_model('trapezoidal')
        result = model.run_base_case()
        trace = result.results['S1']['trace']
        costs_hcc = result.results['S1']['costs_hcc']['drug']
        n = model.n_cycles
        for t in range(n):
            corrected_alive = (trace[t, 0] + trace[t + 1, 0]) / 2.0
            expected_cost = corrected_alive * 1000 * model.cycle_length
            np.testing.assert_allclose(costs_hcc[t], expected_cost, atol=1e-10)

    def test_trapezoidal_with_discount(self):
        """Trapezoidal correction + discounting runs without error."""
        r = self._make_model('trapezoidal', dr_cost=0.05, dr_qaly=0.05).run_base_case()
        q = r.summary()['QALYs'].iloc[0]
        assert q > 0
