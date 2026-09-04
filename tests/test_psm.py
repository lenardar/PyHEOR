"""Tests for pyheor/psm.py — PSMModel."""

import numpy as np
import pytest
from pyheor import PSMModel
from pyheor.analysis.results import PSMBaseResult, PSAResult
from pyheor.survival import Exponential, SurvivalDistribution


# =========================================================================
# Construction
# =========================================================================

class TestPSMConstruction:
    def test_basic(self, simple_psm_model):
        assert len(simple_psm_model.states) == 3


# =========================================================================
# Trace Invariants
# =========================================================================

class TestPSMTraceInvariants:
    def test_state_probs_sum_to_one(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        for strat in simple_psm_model.strategy_names:
            trace = result.results[strat]["trace"]
            row_sums = trace.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_state_probs_nonnegative(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        for strat in simple_psm_model.strategy_names:
            trace = result.results[strat]["trace"]
            assert np.all(trace >= -1e-10)

    def test_dead_state_nondecreasing(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        for strat in simple_psm_model.strategy_names:
            trace = result.results[strat]["trace"]
            dead = trace[:, -1]
            diffs = np.diff(dead)
            assert np.all(diffs >= -1e-10)


# =========================================================================
# Results
# =========================================================================

class TestPSMResults:
    def test_base_case_returns_result(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        assert isinstance(result, PSMBaseResult)

    def test_summary(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        summary = result.summary()
        assert "QALYs" in summary.columns
        assert "Total Cost" in summary.columns

    def test_costs_and_qalys_positive(self, simple_psm_model):
        result = simple_psm_model.run_base_case()
        summary = result.summary()
        assert (summary["QALYs"] > 0).all()
        assert (summary["Total Cost"] > 0).all()

    def test_psa_runs(self, simple_psm_model):
        simple_psm_model.add_param("dummy", base=1.0)
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
        return "FlatSurvival()"


class TestPSMGoldenCalculations:
    @staticmethod
    def _flat_model(**kwargs):
        model = PSMModel(
            states=["Alive", "Dead"],
            survival_endpoints=["OS"],
            strategies=["S1"],
            n_cycles=kwargs.pop("n_cycles", 10),
            half_cycle_correction=kwargs.pop(
                "half_cycle_correction", "trapezoidal"
            ),
            **kwargs,
        )
        model.set_survival("S1", "OS", _FlatSurvival())
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        return model

    def test_ten_intervals_equal_ten_life_years(self):
        result = self._flat_model().run_base_case().results["S1"]
        assert result["trace"].shape == (11, 2)
        assert result["qalys_by_cycle"].shape == (10,)
        assert result["total_lys"] == pytest.approx(10)
        assert result["total_qalys"] == pytest.approx(10)

    def test_callback_receives_zero_based_interval(self):
        seen = []
        model = self._flat_model(n_cycles=3)

        def costs(params, interval):
            seen.append(interval)
            return {"Alive": 1, "Dead": 0}

        model.set_state_cost("care", costs)
        model.run_base_case()
        assert seen == [0, 1, 2]

    def test_curve_crossing_raises_instead_of_clamping(self):
        model = PSMModel(
            states=["PFS", "Progressed", "Dead"],
            survival_endpoints=["PFS", "OS"],
            strategies=["S1"],
            n_cycles=3,
        )
        model.set_survival("S1", "PFS", Exponential(rate=0.1))
        model.set_survival("S1", "OS", Exponential(rate=0.2))
        with pytest.raises(ValueError, match="curve crossing"):
            model.run_base_case()

    def test_missing_curve_has_context(self):
        model = PSMModel(
            states=["Alive", "Dead"], survival_endpoints=["OS"],
            strategies=["S1"], n_cycles=1,
        )
        with pytest.raises(ValueError, match="strategy 'S1'.*endpoint 'OS'"):
            model.run_base_case()

    def test_continuous_discounting_uses_interval_midpoint(self):
        model = self._flat_model(
            n_cycles=1, dr_qaly=0.1, discount_convention="continuous"
        )
        qaly = model.run_base_case().summary()["QALYs"].iloc[0]
        assert qaly == pytest.approx(np.exp(-0.1 * 0.5))

    def test_starting_cost_rejects_ignored_cycle_filter(self):
        model = self._flat_model(n_cycles=2)
        with pytest.raises(ValueError, match="apply_cycles cannot be combined"):
            model.set_state_cost(
                "init", {"Alive": 100}, method="starting", apply_cycles=[1],
            )

    def test_psa_rejects_empty_simulation(self):
        model = self._flat_model(n_cycles=1)
        with pytest.raises(ValueError, match="positive integer"):
            model.run_psa(n_sim=0, progress=False)
