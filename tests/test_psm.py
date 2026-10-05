"""Tests for pyheor/psm.py — PSMModel."""
import pyheor as _ph
from tests.input_helpers import _method, _cycle_values, _starting_values
import numpy as np
import pytest
from pyheor import PSMModel
from pyheor.analysis.results import PSMBaseResult, PSAResult
from pyheor.survival import Exponential, SurvivalDistribution

class TestPSMConstruction:

    def test_basic(self, simple_psm_model):
        assert len(simple_psm_model.states) == 3

class TestPSMTraceInvariants:

    def test_state_probs_sum_to_one(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        for strat in simple_psm_model.strategy_names:
            trace = result.results[strat]['trace']
            row_sums = trace.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-06)

    def test_state_probs_nonnegative(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        for strat in simple_psm_model.strategy_names:
            trace = result.results[strat]['trace']
            assert np.all(trace >= -1e-10)

    def test_dead_state_nondecreasing(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        for strat in simple_psm_model.strategy_names:
            trace = result.results[strat]['trace']
            dead = trace[:, -1]
            diffs = np.diff(dead)
            assert np.all(diffs >= -1e-10)

class TestPSMResults:

    def test_base_case_returns_result(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        assert isinstance(result, PSMBaseResult)

    def test_summary(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        summary = result.summary()
        assert 'QALYs' in summary.columns
        assert 'Total Cost' in summary.columns

    def test_costs_and_qalys_positive(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        summary = result.summary()
        assert (summary['QALYs'] > 0).all()
        assert (summary['Total Cost'] > 0).all()

    def test_psa_runs(self, simple_psm_model):
        simple_psm_model.add_param('dummy', base=1.0)
        psa = simple_psm_model.run_psa(n_sim=3, seed=42, progress=False)
        assert isinstance(psa, PSAResult)

class _FlatSurvival(SurvivalDistribution):

    def survival(self, t):
        values = np.asarray(t, dtype=float)
        result = np.ones_like(values)
        return float(result) if result.ndim == 0 else result

    def hazard(self, t):
        values = np.asarray(t, dtype=float)
        result = np.zeros_like(values)
        return float(result) if result.ndim == 0 else result

    def __repr__(self):
        return 'FlatSurvival()'

class TestPSMGoldenCalculations:

    @staticmethod
    def _flat_model(**kwargs):
        model = PSMModel(states=['Alive', 'Dead'], survival_endpoints=['OS'], strategies=['S1'], n_cycles=kwargs.pop('n_cycles', 10), method=_method(kwargs.pop('half_cycle_correction', 'trapezoidal')), **kwargs, cycle=kwargs.pop('cycle', _ph.Cycle(kwargs.pop('cycle_length', 1), 'year')))
        model.set_survival('S1', 'OS', _FlatSurvival())
        model.set_state_qaly('health', _cycle_values(model, {'Alive': 1.0, 'Dead': 0.0}))
        return model

    def test_ten_intervals_equal_ten_life_years(self):
        result = self._flat_model().run_base_case().results['S1']
        assert result['trace'].shape == (11, 2)
        assert result['qalys_by_cycle'].shape == (10,)
        assert result['total_lys'] == pytest.approx(10)
        assert result['total_qalys'] == pytest.approx(10)

    def test_callback_receives_zero_based_interval(self):
        seen = []
        model = self._flat_model(n_cycles=3)

        def costs(params, interval):
            seen.append(interval)
            return {'Alive': 1, 'Dead': 0}
        model.set_state_cost('care', _cycle_values(model, costs))
        model.run_base_case()
        assert seen == [1, 2, 3]

    def test_curve_crossing_raises_instead_of_clamping(self):
        model = PSMModel(states=['PFS', 'Progressed', 'Dead'], survival_endpoints=['PFS', 'OS'], strategies=['S1'], n_cycles=3, cycle=_ph.Cycle(1, 'year'))
        model.set_survival('S1', 'PFS', Exponential(rate=0.1))
        model.set_survival('S1', 'OS', Exponential(rate=0.2))
        with pytest.raises(ValueError, match='curve crossing'):
            model.run_base_case()

    def test_missing_curve_has_context(self):
        model = PSMModel(states=['Alive', 'Dead'], survival_endpoints=['OS'], strategies=['S1'], n_cycles=1, cycle=_ph.Cycle(1, 'year'))
        with pytest.raises(ValueError, match="strategy 'S1'.*endpoint 'OS'"):
            model.run_base_case()

    def test_psa_rejects_empty_simulation(self):
        model = self._flat_model(n_cycles=1)
        with pytest.raises(ValueError, match='positive integer'):
            model.run_psa(n_sim=0, progress=False)


class TestPSACrossingResampling:
    @staticmethod
    def _model(multi_state=False):
        endpoints = ['PFS', 'PFS2', 'OS'] if multi_state else ['PFS', 'OS']
        states = ['PF', 'PD1', 'PD2', 'Dead'] if multi_state else ['PF', 'PD', 'Dead']
        model = PSMModel(states=states, survival_endpoints=endpoints,
                         strategies=['A', 'B'], n_cycles=3,
                         cycle=_ph.Cycle(1, 'year'))
        model.add_param('rate', base=0.3, dist=object())
        for strategy in model.strategy_names:
            model.set_survival(strategy, 'PFS', Exponential(rate=0.4))
            if multi_state:
                model.set_survival(strategy, 'PFS2',
                                   lambda p: Exponential(rate=p['rate']))
                model.set_survival(strategy, 'OS', Exponential(rate=0.1))
            else:
                model.set_survival(strategy, 'OS',
                                   Exponential(rate=0.1) if strategy == 'A'
                                   else lambda p: Exponential(rate=p['rate']))
        return model

    @pytest.mark.parametrize('multi_state', [False, True])
    def test_rejects_whole_draw_and_returns_valid_results(self, monkeypatch, multi_state):
        model = self._model(multi_state)
        draws = iter([0.5, 0.2, 0.3])
        monkeypatch.setattr('pyheor.models.psm.sample_distribution',
                            lambda *args: np.array([next(draws)]))
        with pytest.warns(UserWarning, match='rejected 1 draws'):
            result = model.run_psa(n_sim=2, seed=42, progress=False)
        assert [p['rate'] for p in result.sampled_params] == [0.2, 0.3]
        assert len(result.psa_results) == 2
        for draw in result.psa_results:
            for strategy in model.strategy_names:
                trace = draw[strategy]['trace']
                assert np.all(trace >= 0)
                np.testing.assert_allclose(trace.sum(axis=1), 1)
        assert model.params['rate'].base == 0.3

    def test_stops_at_attempt_limit(self, monkeypatch):
        model = self._model()
        calls = []
        def draw(*args):
            calls.append(1)
            return np.array([0.5])
        monkeypatch.setattr('pyheor.models.psm.sample_distribution', draw)
        with pytest.raises(RuntimeError, match='accepted 0/2.*rejected 3'):
            model.run_psa(n_sim=2, max_attempts=3, progress=False)
        assert len(calls) == 3

    def test_other_errors_are_not_retried(self, monkeypatch):
        model = self._model()
        calls = []
        def draw(*args):
            calls.append(1)
            return np.array([0.2])
        monkeypatch.setattr('pyheor.models.psm.sample_distribution', draw)
        model.set_survival('B', 'OS', lambda p: (_ for _ in ()).throw(
            ValueError('invalid custom curve')))
        with pytest.raises(ValueError, match='invalid custom curve'):
            model.run_psa(n_sim=2, progress=False)
        assert len(calls) == 1

    def test_seed_reproduces_accepted_draws(self):
        from pyheor.distributions import Uniform
        model = self._model()
        model.params['rate'].dist = Uniform(0.1, 0.6)
        with pytest.warns(UserWarning):
            first = model.run_psa(n_sim=10, seed=42, progress=False)
        with pytest.warns(UserWarning):
            second = model.run_psa(n_sim=10, seed=42, progress=False)
        assert first.sampled_params == second.sampled_params

    @pytest.mark.parametrize('limit', [0, 1, True, 2.5])
    def test_invalid_attempt_limit(self, limit):
        with pytest.raises(ValueError, match='max_attempts'):
            self._model().run_psa(n_sim=2, max_attempts=limit, progress=False)
