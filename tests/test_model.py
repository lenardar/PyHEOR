"""Tests for pyheor/model.py — MarkovModel full workflow."""

import numpy as np
import pytest
from pyheor import MarkovModel, C, Beta, Distribution, Gamma
from pyheor.analysis.results import BaseResult, OWSAResult, PSAResult


class _LegacyDistribution(Distribution):
    """Third-party style distribution implementing the historical API."""

    def sample(self, n=1):
        return np.random.uniform(0.1, 0.9, size=n)

    def __repr__(self):
        return "LegacyDistribution()"


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
                dr_cost=dr,
                dr_qaly=dr,
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
        )
        model.set_transitions("S1", lambda p, t: [[1, 0], [0, 1]])
        model.set_state_cost("init", {"Alive": 10000, "Dead": 0},
                             first_cycle_only=True)
        model.set_state_cost("ongoing", {"Alive": 1000, "Dead": 0})
        model.set_utility({"Alive": 1.0, "Dead": 0.0})

        result = model.run_base_case()
        total = result.summary()["Total Cost"].iloc[0]
        # Five intervals: 10000 in interval 0 + 1000 * 5 intervals.
        np.testing.assert_allclose(total, 15000, rtol=0.01)

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

    def test_psa_does_not_change_global_numpy_rng(self, simple_markov_model):
        np.random.seed(2026)
        expected = np.random.random()
        np.random.seed(2026)
        simple_markov_model.run_psa(n_sim=2, seed=123, progress=False)
        assert np.random.random() == pytest.approx(expected)

    def test_psa_rejects_empty_simulation(self, simple_markov_model):
        with pytest.raises(ValueError, match="positive integer"):
            simple_markov_model.run_psa(n_sim=0, progress=False)

    def test_psa_supports_legacy_distribution_without_global_rng_leak(self):
        model = MarkovModel(
            states=["Alive", "Dead"], strategies=["S1"], n_cycles=1,
        )
        model.add_param("p", base=0.5, dist=_LegacyDistribution())
        model.set_transitions("S1", lambda p, t: [[1 - p["p"], p["p"]], [0, 1]])

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
        nmb_summary = owsa.summary(outcome="nmb")
        icer_summary = owsa.summary(outcome="icer")
        # Both should have same parameters
        assert set(nmb_summary['param_name']) == set(icer_summary['param_name'])
        # ICER summary should have ICER columns
        assert "ICER (Low)" in icer_summary.columns
        assert "ICER (High)" in icer_summary.columns
        assert "ICER (Base)" in icer_summary.columns

    def test_owsa_discount_rate_param(self):
        """Discount rate can be varied in OWSA via Param."""
        from pyheor.models.markov import Param
        model = MarkovModel(
            states=["Alive", "Dead"],
            strategies=["S1", "S2"],
            n_cycles=10,
            dr_cost=Param(0.05, low=0.0, high=0.08),
            dr_qaly=Param(0.05, low=0.0, high=0.08),
            half_cycle_correction=False,
        )
        model.add_param("p_death", base=0.1, low=0.05, high=0.15)
        model.set_transitions("S1", lambda p, t: [
            [1 - p["p_death"], p["p_death"]],
            [0, 1],
        ])
        model.set_transitions("S2", lambda p, t: [
            [1 - p["p_death"] * 0.8, p["p_death"] * 0.8],
            [0, 1],
        ])
        model.set_state_cost("drug", {
            "S1": {"Alive": 1000, "Dead": 0},
            "S2": {"Alive": 5000, "Dead": 0},
        })
        model.set_utility({"Alive": 1.0, "Dead": 0.0})

        owsa = model.run_owsa(params=["dr_cost", "p_death"])
        summary = owsa.summary()
        assert "dr_cost" in summary["param_name"].values

        # Verify discount rate was actually varied (different results)
        dr_row = summary[summary["param_name"] == "dr_cost"].iloc[0]
        assert dr_row["INMB (Low)"] != dr_row["INMB (High)"]

        # Verify model discount rate was restored
        assert model.dr_cost == 0.05
        assert model.dr_qaly == 0.05


# =========================================================================
# Half-Cycle Correction Methods
# =========================================================================

class TestHCCNormalization:
    def test_true_to_trapezoidal(self):
        m = MarkovModel(
            states=["A", "B"], strategies=["S1"], n_cycles=5,
            half_cycle_correction=True,
        )
        assert m._hcc_method == "trapezoidal"

    def test_false_to_none(self):
        m = MarkovModel(
            states=["A", "B"], strategies=["S1"], n_cycles=5,
            half_cycle_correction=False,
        )
        assert m._hcc_method is None

    def test_none_to_none(self):
        m = MarkovModel(
            states=["A", "B"], strategies=["S1"], n_cycles=5,
            half_cycle_correction=None,
        )
        assert m._hcc_method is None

    def test_string_trapezoidal(self):
        m = MarkovModel(
            states=["A", "B"], strategies=["S1"], n_cycles=5,
            half_cycle_correction="trapezoidal",
        )
        assert m._hcc_method == "trapezoidal"

    def test_string_life_table(self):
        m = MarkovModel(
            states=["A", "B"], strategies=["S1"], n_cycles=5,
            half_cycle_correction="life-table",
        )
        assert m._hcc_method == "trapezoidal"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            MarkovModel(
                states=["A", "B"], strategies=["S1"], n_cycles=5,
                half_cycle_correction="invalid",
            )

    def test_setter(self):
        m = MarkovModel(
            states=["A", "B"], strategies=["S1"], n_cycles=5,
            half_cycle_correction=True,
        )
        m.half_cycle_correction = "life-table"
        assert m._hcc_method == "trapezoidal"
        m.half_cycle_correction = False
        assert m._hcc_method is None

    def test_backward_compat_bool_true(self):
        """True and 'trapezoidal' produce identical results."""
        def make(hcc):
            m = MarkovModel(
                states=["Alive", "Dead"], strategies=["S1"],
                n_cycles=10, half_cycle_correction=hcc, dr_cost=0.05, dr_qaly=0.05,
            )
            m.set_transitions("S1", [[0.9, 0.1], [0, 1]])
            m.set_utility({"Alive": 1.0, "Dead": 0.0})
            m.set_state_cost("drug", {"Alive": 1000, "Dead": 0})
            return m.run_base_case()

        r_bool = make(True)
        r_str = make("trapezoidal")
        np.testing.assert_allclose(
            r_bool.summary()["QALYs"].values,
            r_str.summary()["QALYs"].values,
        )
        np.testing.assert_allclose(
            r_bool.summary()["Total Cost"].values,
            r_str.summary()["Total Cost"].values,
        )


class TestLifeTableHCC:
    @staticmethod
    def _make_model(hcc, n_cycles=10, dr_cost=0, dr_qaly=0):
        m = MarkovModel(
            states=["Alive", "Dead"], strategies=["S1"],
            n_cycles=n_cycles, half_cycle_correction=hcc,
            dr_cost=dr_cost, dr_qaly=dr_qaly,
        )
        m.set_transitions("S1", lambda p, t: [[0.9, 0.1], [0, 1]])
        m.set_utility({"Alive": 1.0, "Dead": 0.0})
        m.set_state_cost("drug", {"Alive": 1000, "Dead": 0})
        return m

    def test_life_table_differs_from_no_hcc(self):
        r_lt = self._make_model("life-table").run_base_case()
        r_none = self._make_model(None).run_base_case()
        q_lt = r_lt.summary()["QALYs"].iloc[0]
        q_none = r_none.summary()["QALYs"].iloc[0]
        assert q_lt != q_none

    def test_life_table_equals_trapezoidal(self):
        """life-table is a compatibility alias for trapezoidal."""
        r_lt = self._make_model("life-table").run_base_case()
        r_trap = self._make_model("trapezoidal").run_base_case()
        q_lt = r_lt.summary()["QALYs"].iloc[0]
        q_trap = r_trap.summary()["QALYs"].iloc[0]
        np.testing.assert_allclose(q_lt, q_trap, rtol=0, atol=0)

    def test_life_table_manual_verification(self):
        """Verify against hand-computed corrected trace."""
        model = self._make_model("life-table")
        result = model.run_base_case()
        trace = result.results["S1"]["trace"]
        qalys_hcc = result.results["S1"]["qalys_hcc"]

        assert len(qalys_hcc) == model.n_cycles
        for t in range(model.n_cycles):
            corrected_alive = (trace[t, 0] + trace[t + 1, 0]) / 2.0
            expected_qaly = corrected_alive * model.cycle_length  # u=1.0
            np.testing.assert_allclose(
                qalys_hcc[t], expected_qaly, atol=1e-10,
            )

    def test_life_table_costs_manual(self):
        """Verify life-table corrected costs."""
        model = self._make_model("life-table")
        result = model.run_base_case()
        trace = result.results["S1"]["trace"]
        costs_hcc = result.results["S1"]["costs_hcc"]["drug"]

        n = model.n_cycles
        for t in range(n):
            corrected_alive = (trace[t, 0] + trace[t + 1, 0]) / 2.0
            expected_cost = corrected_alive * 1000 * model.cycle_length
            np.testing.assert_allclose(
                costs_hcc[t], expected_cost, atol=1e-10,
            )

    def test_life_table_with_discount(self):
        """Life-table + discounting runs without error."""
        r = self._make_model("life-table", dr_cost=0.05, dr_qaly=0.05).run_base_case()
        q = r.summary()["QALYs"].iloc[0]
        assert q > 0


# =========================================================================
# Golden calculation contract
# =========================================================================

class TestGoldenCalculations:
    @staticmethod
    def _identity_model(**kwargs):
        model = MarkovModel(
            states=["Alive", "Dead"],
            strategies=["S1"],
            n_cycles=kwargs.pop("n_cycles", 10),
            half_cycle_correction=kwargs.pop("half_cycle_correction", False),
            **kwargs,
        )
        model.set_transitions("S1", [[1, 0], [0, 1]])
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        return model

    def test_ten_intervals_equal_ten_life_years(self):
        result = self._identity_model(n_cycles=10).run_base_case()
        strategy = result.results["S1"]
        assert strategy["trace"].shape == (11, 2)
        assert strategy["qalys_by_cycle"].shape == (10,)
        assert strategy["discounted_qalys"].shape == (10,)
        assert strategy["total_lys"] == pytest.approx(10.0)
        assert strategy["total_qalys"] == pytest.approx(10.0)

    @pytest.mark.parametrize(
        ("hcc", "expected"),
        [(False, 1.0), ("trapezoidal", 0.5), ("life-table", 0.5)],
    )
    def test_death_at_first_interval_end(self, hcc, expected):
        model = MarkovModel(
            states=["Alive", "Dead"], strategies=["S1"],
            n_cycles=1, half_cycle_correction=hcc,
        )
        model.set_transitions("S1", [[0, 1], [0, 1]])
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        result = model.run_base_case().results["S1"]
        assert result["total_lys"] == pytest.approx(expected)
        assert result["total_qalys"] == pytest.approx(expected)

    def test_half_year_cost_rate(self):
        model = self._identity_model(n_cycles=1, cycle_length=0.5)
        model.set_state_cost("care", {"Alive": 100, "Dead": 0})
        assert model.run_base_case().summary()["Total Cost"].iloc[0] == pytest.approx(50)

    def test_starting_cost_is_a_lump_sum(self):
        model = self._identity_model(
            n_cycles=1, cycle_length=0.5,
            half_cycle_correction="trapezoidal", dr_cost=0.20,
        )
        model.set_state_cost(
            "init", {"Alive": 100, "Dead": 0}, method="starting"
        )
        assert model.run_base_case().summary()["Total Cost"].iloc[0] == pytest.approx(100)

    def test_starting_cost_rejects_ignored_cycle_filter(self):
        model = self._identity_model(n_cycles=2)
        with pytest.raises(ValueError, match="apply_cycles cannot be combined"):
            model.set_state_cost(
                "init", {"Alive": 100}, method="starting", apply_cycles=[1],
            )

    def test_transition_cost_uses_event_timing_even_with_same_category(self):
        model = MarkovModel(
            states=["Alive", "Dead"], strategies=["S1"], n_cycles=1,
            half_cycle_correction="trapezoidal", dr_cost=0.10,
        )
        model.set_transitions("S1", [[0, 1], [0, 1]])
        model.set_state_cost("medical", {"Alive": 0, "Dead": 0})
        model.set_transition_cost("medical", "Alive", "Dead", 100)
        total = model.run_base_case().summary()["Total Cost"].iloc[0]
        assert total == pytest.approx(100 / 1.10)

    @pytest.mark.parametrize(
        ("convention", "expected"),
        [
            ("discrete", 1 / np.sqrt(1.10)),
            ("continuous", np.exp(-0.10 * 0.5)),
        ],
    )
    def test_flow_discounted_at_interval_midpoint(self, convention, expected):
        model = self._identity_model(
            n_cycles=1, dr_qaly=0.10, discount_convention=convention,
        )
        assert model.run_base_case().summary()["QALYs"].iloc[0] == pytest.approx(expected)

    def test_callbacks_receive_zero_based_interval(self):
        seen = []
        model = MarkovModel(
            states=["Alive", "Dead"], strategies=["S1"], n_cycles=3,
            half_cycle_correction=False,
        )

        def transitions(params, interval):
            seen.append(interval)
            return [[1, 0], [0, 1]]

        model.set_transitions("S1", transitions)
        model.run_base_case()
        assert seen == [0, 1, 2]

    def test_invalid_transition_matrix_is_not_repaired(self):
        model = MarkovModel(
            states=["Alive", "Dead"], strategies=["S1"], n_cycles=1,
        )
        with pytest.raises(ValueError, match="Invalid transition matrix"):
            model.set_transitions("S1", [[-0.2, 1.2], [0, 1]])

    def test_unknown_state_key_is_not_silently_ignored(self):
        model = self._identity_model(n_cycles=1)
        with pytest.raises(ValueError, match="unknown state or strategy"):
            model.set_state_cost("care", {"Alvie": 100})

    def test_strongly_dominated_strategy_is_classified(self):
        model = MarkovModel(
            states=["Alive", "Dead"], strategies=["SOC", "New"],
            n_cycles=1, half_cycle_correction=True,
        )
        model.set_transitions("SOC", [[1, 0], [0, 1]])
        model.set_transitions("New", [[0, 1], [0, 1]])
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        model.set_state_cost("init", {
            "SOC": {"Alive": 0},
            "New": {"Alive": 100},
        }, method="starting")

        row = model.run_base_case().icer().iloc[0]
        assert row["ICER"] == "Dominated"
        assert np.isnan(row["ICER ($/QALY)"])
