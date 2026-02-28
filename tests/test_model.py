"""Tests for pyheor/model.py — MarkovModel full workflow."""

import numpy as np
import pytest
from pyheor import MarkovModel, C, Beta, Gamma
from pyheor.results import BaseResult, OWSAResult, PSAResult


# =========================================================================
# Construction & Parameters
# =========================================================================

class TestModelConstruction:
    def test_basic(self):
        model = MarkovModel(
            states=["A", "B"], strategies=["S1"], n_cycles=5,
        )
        assert model.states == ["A", "B"]
        assert len(model.strategy_names) == 1
        assert model.n_cycles == 5

    def test_add_param(self):
        model = MarkovModel(
            states=["A", "B"], strategies=["S1"], n_cycles=5,
        )
        model.add_param("p", base=0.5)
        assert "p" in model.params

    def test_owsa_defaults(self):
        model = MarkovModel(
            states=["A", "B"], strategies=["S1"], n_cycles=5,
        )
        model.add_param("p", base=0.5)
        param = model.params["p"]
        # Default OWSA range: base * 0.8 to base * 1.2
        assert param.low == pytest.approx(0.4)
        assert param.high == pytest.approx(0.6)


# =========================================================================
# Trace Invariants
# =========================================================================

class TestTraceInvariants:
    def test_rows_sum_to_one(self, simple_markov_model):
        result = simple_markov_model.run_base_case()
        for strat in simple_markov_model.strategy_names:
            trace = result.results[strat]["trace"]
            row_sums = trace.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_initial_state(self, simple_markov_model):
        result = simple_markov_model.run_base_case()
        for strat in simple_markov_model.strategy_names:
            trace = result.results[strat]["trace"]
            np.testing.assert_allclose(trace[0], [1, 0, 0])

    def test_all_dead_model(self):
        model = MarkovModel(
            states=["Alive", "Dead"],
            strategies=["S1"],
            n_cycles=5,
            half_cycle_correction=False,
        )
        model.set_transitions("S1", lambda p, t: [
            [0, 1],
            [0, 1],
        ])
        model.set_utility({"Alive": 1, "Dead": 0})
        result = model.run_base_case()
        trace = result.results["S1"]["trace"]
        # After cycle 1, everyone is dead
        np.testing.assert_allclose(trace[1], [0, 1])
        np.testing.assert_allclose(trace[-1], [0, 1])

    def test_identity_transitions(self):
        model = MarkovModel(
            states=["A", "B"],
            strategies=["S1"],
            n_cycles=5,
            half_cycle_correction=False,
        )
        model.set_transitions("S1", lambda p, t: [
            [1, 0],
            [0, 1],
        ])
        model.set_utility({"A": 1, "B": 0})
        result = model.run_base_case()
        trace = result.results["S1"]["trace"]
        # Trace should be constant
        for k in range(trace.shape[0]):
            np.testing.assert_allclose(trace[k], [1, 0])


# =========================================================================
# Results & Analysis
# =========================================================================

class TestResults:
    def test_base_case_returns_base_result(self, simple_markov_model):
        result = simple_markov_model.run_base_case()
        assert isinstance(result, BaseResult)

    def test_summary_columns(self, simple_markov_model):
        result = simple_markov_model.run_base_case()
        summary = result.summary()
        assert "Strategy" in summary.columns
        assert "QALYs" in summary.columns
        assert "Total Cost" in summary.columns

    def test_icer(self, simple_markov_model):
        result = simple_markov_model.run_base_case()
        icer_df = result.icer()
        assert "ICER" in icer_df.columns

    def test_hcc_effect(self):
        """HCC on vs off should produce different QALYs for non-trivial model."""
        def make_model(hcc):
            model = MarkovModel(
                states=["Alive", "Dead"],
                strategies=["S1"],
                n_cycles=10,
                half_cycle_correction=hcc,
                discount_rate=0,
            )
            model.set_transitions("S1", lambda p, t: [
                [0.9, 0.1],
                [0, 1],
            ])
            model.set_utility({"Alive": 1.0, "Dead": 0.0})
            return model

        r_on = make_model(True).run_base_case()
        r_off = make_model(False).run_base_case()
        q_on = r_on.summary()["QALYs"].iloc[0]
        q_off = r_off.summary()["QALYs"].iloc[0]
        assert q_on != q_off

    def test_discount_effect(self):
        """Higher discount -> lower QALYs."""
        def make_model(dr):
            model = MarkovModel(
                states=["Alive", "Dead"],
                strategies=["S1"],
                n_cycles=20,
                discount_rate=dr,
                half_cycle_correction=False,
            )
            model.set_transitions("S1", lambda p, t: [
                [0.95, 0.05],
                [0, 1],
            ])
            model.set_utility({"Alive": 1.0, "Dead": 0.0})
            return model

        q_low = make_model(0.0).run_base_case().summary()["QALYs"].iloc[0]
        q_high = make_model(0.10).run_base_case().summary()["QALYs"].iloc[0]
        assert q_high < q_low


# =========================================================================
# Cost Features
# =========================================================================

class TestCosts:
    def test_first_cycle_only(self):
        model = MarkovModel(
            states=["Alive", "Dead"],
            strategies=["S1"],
            n_cycles=5,
            half_cycle_correction=False,
            discount_rate=0,
        )
        model.set_transitions("S1", lambda p, t: [[1, 0], [0, 1]])
        model.set_state_cost("init", {"Alive": 10000, "Dead": 0},
                             first_cycle_only=True)
        model.set_state_cost("ongoing", {"Alive": 1000, "Dead": 0})
        model.set_utility({"Alive": 1.0, "Dead": 0.0})

        result = model.run_base_case()
        total = result.summary()["Total Cost"].iloc[0]
        # 10000 (first cycle) + 1000 * 6 cycles (0..5) = 16000
        np.testing.assert_allclose(total, 16000, rtol=0.01)

    def test_custom_cost(self, simple_markov_model):
        """Custom cost function runs and contributes to total."""
        def my_cost(strategy, params, t, state_prev, state_curr, P, states):
            return 100.0  # flat 100 per cycle

        simple_markov_model.set_custom_cost("extra", my_cost)
        result = simple_markov_model.run_base_case()
        total = result.summary()["Total Cost"].iloc[0]
        assert total > 0


# =========================================================================
# OWSA & PSA
# =========================================================================

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
        s1 = r1.summary()["Mean QALYs"].values
        s2 = r2.summary()["Mean QALYs"].values
        np.testing.assert_allclose(s1, s2)
