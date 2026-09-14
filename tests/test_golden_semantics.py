"""Golden tests pinning the calculation semantics shared by all engines.

Each class corresponds to one rule of the agreed calculation conventions:

1. ``n_cycles=N`` means N reward intervals, not N+1.
2. Half-cycle correction averages the two interval endpoints.
3. Per-year state costs scale with ``cycle_length``.
4. ``method="starting"`` costs are lump sums at time zero.
5. Transition costs are events, unaffected by half-cycle correction.
6. Discrete and continuous discounting are distinct, hand-checkable conventions.
7. Seeded runs reproduce (covered in ``test_reproducibility.py``).
8. Incremental quadrants are classified before dividing.
"""

import numpy as np
import pytest

from pyheor import (
    C,
    DESModel,
    MarkovModel,
    MicroSimModel,
    PSMModel,
)
from pyheor.survival import Exponential, KaplanMeier

ALIVE_FOREVER = [[1, 0], [0, 1]]
DEAD_AFTER_ONE_INTERVAL = [[0, 1], [0, 1]]

# Step curves let PSM reproduce the cohort scenarios exactly rather than
# approximately: S(t)=1 never leaves the alive state, S(0)=1/S(1)=0 empties it
# at the end of the first interval.
NEVER_DIES = KaplanMeier(times=[0.0], survival_probs=[1.0])
DIES_AT_ONE = KaplanMeier(times=[0.0, 1.0], survival_probs=[1.0, 0.0])


# =============================================================================
# Builders
# =============================================================================

def build_markov(transitions, n_cycles=10, cycle_length=1.0, hcc=False, **kw):
    model = MarkovModel(
        states=["Alive", "Dead"],
        strategies=["S1"],
        n_cycles=n_cycles,
        cycle_length=cycle_length,
        half_cycle_correction=hcc,
        **kw,
    )
    model.set_transitions("S1", lambda p, t: transitions)
    model.set_utility({"Alive": 1.0, "Dead": 0.0})
    return model


def build_psm(curve, n_cycles=10, cycle_length=1.0, hcc=False, **kw):
    model = PSMModel(
        states=["Alive", "Dead"],
        survival_endpoints=["OS"],
        strategies=["S1"],
        n_cycles=n_cycles,
        cycle_length=cycle_length,
        half_cycle_correction=hcc,
        **kw,
    )
    model.set_survival("S1", "OS", curve)
    model.set_utility({"Alive": 1.0, "Dead": 0.0})
    return model


def build_microsim(transitions, n_cycles=10, cycle_length=1.0, hcc=False, **kw):
    model = MicroSimModel(
        states=["Alive", "Dead"],
        strategies=["S1"],
        n_cycles=n_cycles,
        cycle_length=cycle_length,
        n_patients=5,
        half_cycle_correction=hcc,
        **kw,
    )
    model.set_transitions("S1", lambda p, t: transitions)
    model.set_utility({"Alive": 1.0, "Dead": 0.0})
    return model


def total_lys(model):
    if isinstance(model, MicroSimModel):
        result = model.run_base_case(seed=1, verbose=False)
        return float(np.mean(result.results["S1"]["total_lys"]))
    return model.run_base_case().results["S1"]["total_lys"]


def total_cost(model, category="care"):
    if isinstance(model, MicroSimModel):
        result = model.run_base_case(seed=1, verbose=False)
        return float(np.mean(result.results["S1"]["total_cost"]))
    return model.run_base_case().results["S1"]["total_costs"][category]


def xfail_microsim(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


# =============================================================================
# 1. N intervals, not N+1
# =============================================================================

class TestTenIntervalsGiveTenLifeYears:
    """Ten one-year intervals alive accrue 10 LY, regardless of correction."""

    @pytest.mark.parametrize("hcc", [False, True])
    def test_markov(self, hcc):
        assert total_lys(build_markov(ALIVE_FOREVER, hcc=hcc)) == pytest.approx(10.0)

    @pytest.mark.parametrize("hcc", [False, True])
    def test_psm(self, hcc):
        assert total_lys(build_psm(NEVER_DIES, hcc=hcc)) == pytest.approx(10.0)

    @pytest.mark.parametrize("hcc", [
        pytest.param(False, marks=xfail_microsim(
            "microsim.py:692 accrues over N+1 observation points, giving 11 LY")),
        True,
    ])
    def test_microsim(self, hcc):
        assert total_lys(build_microsim(ALIVE_FOREVER, hcc=hcc)) == pytest.approx(10.0)

    def test_des(self):
        model = DESModel(
            states=["Alive", "Dead"], strategies=["S1"], time_horizon=10.0
        )
        model.set_utility({"Alive": 1.0})
        result = model.run(n_patients=1, seed=1, progress=False)
        assert float(result.results["S1"]["total_lys"][0]) == pytest.approx(10.0)


# =============================================================================
# 2. Half-cycle correction averages interval endpoints
# =============================================================================

class TestDeathAtFirstIntervalEnd:
    """Without correction the interval yields 1 LY; trapezoidal yields 0.5."""

    @pytest.mark.parametrize("hcc,expected", [(False, 1.0), (True, 0.5)])
    def test_markov(self, hcc, expected):
        model = build_markov(DEAD_AFTER_ONE_INTERVAL, hcc=hcc)
        assert total_lys(model) == pytest.approx(expected)

    @pytest.mark.parametrize("hcc,expected", [(False, 1.0), (True, 0.5)])
    def test_psm(self, hcc, expected):
        assert total_lys(build_psm(DIES_AT_ONE, hcc=hcc)) == pytest.approx(expected)

    @pytest.mark.parametrize("hcc,expected", [(False, 1.0), (True, 0.5)])
    def test_microsim(self, hcc, expected):
        model = build_microsim(DEAD_AFTER_ONE_INTERVAL, hcc=hcc)
        assert total_lys(model) == pytest.approx(expected)


# =============================================================================
# 3. Per-year state costs scale with cycle length
# =============================================================================

class TestHalfYearCycleHalvesAnnualCost:
    """A 100/year rate over one half-year interval costs 50."""

    @pytest.mark.parametrize("hcc", [False, True])
    def test_markov(self, hcc):
        model = build_markov(ALIVE_FOREVER, n_cycles=1, cycle_length=0.5, hcc=hcc)
        model.set_state_cost("care", {"Alive": 100, "Dead": 0})
        assert total_cost(model) == pytest.approx(50.0)

    @pytest.mark.parametrize("hcc", [False, True])
    def test_psm(self, hcc):
        model = build_psm(NEVER_DIES, n_cycles=1, cycle_length=0.5, hcc=hcc)
        model.set_state_cost("care", {"Alive": 100, "Dead": 0})
        assert total_cost(model) == pytest.approx(50.0)

    @pytest.mark.parametrize("hcc", [
        pytest.param(False, marks=xfail_microsim(
            "microsim.py:692 accrues two observation points for one interval")),
        True,
    ])
    def test_microsim(self, hcc):
        model = build_microsim(ALIVE_FOREVER, n_cycles=1, cycle_length=0.5, hcc=hcc)
        model.set_state_cost("care", {"Alive": 100, "Dead": 0})
        assert total_cost(model) == pytest.approx(50.0)


# =============================================================================
# 4. Starting costs are lump sums
# =============================================================================

class TestStartingCostIsInvariant:
    """A 100 starting cost stays 100 across cycle lengths and corrections."""

    @pytest.mark.parametrize("cycle_length", [0.5, 1.0])
    @pytest.mark.parametrize("hcc", [False, True])
    def test_markov(self, cycle_length, hcc):
        model = build_markov(
            ALIVE_FOREVER, n_cycles=4, cycle_length=cycle_length, hcc=hcc
        )
        model.set_state_cost(
            "care", {"Alive": 100, "Dead": 0}, method="starting"
        )
        assert total_cost(model) == pytest.approx(100.0)

    @pytest.mark.parametrize("cycle_length", [0.5, 1.0])
    @pytest.mark.parametrize("hcc", [False, True])
    def test_psm(self, cycle_length, hcc):
        model = build_psm(
            NEVER_DIES, n_cycles=4, cycle_length=cycle_length, hcc=hcc
        )
        model.set_state_cost(
            "care", {"Alive": 100, "Dead": 0}, method="starting"
        )
        assert total_cost(model) == pytest.approx(100.0)

    @pytest.mark.parametrize("cycle_length", [0.5, 1.0])
    @pytest.mark.parametrize("hcc", [False, True])
    def test_microsim(self, cycle_length, hcc):
        model = build_microsim(
            ALIVE_FOREVER, n_cycles=4, cycle_length=cycle_length, hcc=hcc
        )
        model.set_state_cost(
            "care", {"Alive": 100, "Dead": 0}, method="starting"
        )
        assert total_cost(model) == pytest.approx(100.0)


# =============================================================================
# 5. Transition costs are events
# =============================================================================

class TestTransitionCostIsAnEvent:
    @pytest.mark.parametrize("hcc", [False, True])
    def test_unaffected_by_half_cycle_correction(self, hcc):
        model = build_markov(DEAD_AFTER_ONE_INTERVAL, n_cycles=1, hcc=hcc)
        model.set_transition_cost("surgery", "Alive", "Dead", 100)
        assert total_cost(model, "surgery") == pytest.approx(100.0)

    @pytest.mark.parametrize("hcc,state_part", [(False, 100.0), (True, 50.0)])
    def test_sharing_a_category_with_a_state_cost_keeps_both_timings(
        self, hcc, state_part
    ):
        model = build_markov(DEAD_AFTER_ONE_INTERVAL, n_cycles=1, hcc=hcc)
        model.set_state_cost("surgery", {"Alive": 100, "Dead": 0})
        model.set_transition_cost("surgery", "Alive", "Dead", 100)
        assert total_cost(model, "surgery") == pytest.approx(state_part + 100.0)


# =============================================================================
# 6. Discrete vs continuous discounting
# =============================================================================

class TestDiscountingConventions:
    """State flows discount at interval midpoints, events at interval ends."""

    @pytest.mark.parametrize("convention,factor", [
        ("discrete", 1.1 ** -0.5),
        ("continuous", np.exp(-0.1 * 0.5)),
    ])
    def test_markov_state_cost_uses_interval_midpoint(self, convention, factor):
        model = build_markov(
            ALIVE_FOREVER, n_cycles=1, dr_cost=0.1,
            discount_convention=convention,
        )
        model.set_state_cost("care", {"Alive": 100, "Dead": 0})
        assert total_cost(model) == pytest.approx(100.0 * factor)

    @pytest.mark.parametrize("convention,factor", [
        ("discrete", 1.1 ** -1.0),
        ("continuous", np.exp(-0.1)),
    ])
    def test_markov_transition_cost_uses_interval_end(self, convention, factor):
        model = build_markov(
            DEAD_AFTER_ONE_INTERVAL, n_cycles=1, dr_cost=0.1,
            discount_convention=convention,
        )
        model.set_transition_cost("surgery", "Alive", "Dead", 100)
        assert total_cost(model, "surgery") == pytest.approx(100.0 * factor)

    @pytest.mark.parametrize("convention,factor", [
        ("discrete", 1.1 ** -0.5),
        ("continuous", np.exp(-0.1 * 0.5)),
    ])
    def test_psm_state_cost_uses_interval_midpoint(self, convention, factor):
        model = build_psm(
            NEVER_DIES, n_cycles=1, dr_cost=0.1,
            discount_convention=convention,
        )
        model.set_state_cost("care", {"Alive": 100, "Dead": 0})
        assert total_cost(model) == pytest.approx(100.0 * factor)

    @pytest.mark.parametrize("convention,expected", [
        ("discrete", 100.0 * (1 - 1.1 ** -1) / np.log(1.1)),
        ("continuous", 100.0 * (1 - np.exp(-0.1)) / 0.1),
    ])
    def test_des_integrates_the_flow_over_real_time(self, convention, expected):
        model = DESModel(
            states=["Alive", "Dead"], strategies=["S1"], time_horizon=1.0,
            dr_cost=0.1, discount_convention=convention,
        )
        model.set_state_cost("care", {"Alive": 100})
        model.set_utility({"Alive": 1.0})
        result = model.run(n_patients=1, seed=1, progress=False)
        assert float(result.results["S1"]["total_cost"][0]) == pytest.approx(expected)


# =============================================================================
# 8. Quadrant before division
# =============================================================================

def dominance_label(frame):
    row = frame.iloc[0]
    if "ICER Classification" in frame.columns:
        return row["ICER Classification"]
    return row["ICER"]


class TestCostlierAndLessEffectiveIsDominated:
    """+100 cost and -0.5 QALY is Dominated, never a negative ratio."""

    def test_markov(self):
        model = MarkovModel(
            states=["Alive", "Dead"], strategies=["SOC", "TRT"],
            n_cycles=1, half_cycle_correction=False,
        )
        for strategy in ("SOC", "TRT"):
            model.set_transitions(strategy, lambda p, t: ALIVE_FOREVER)
        model.set_state_cost("care", {"SOC": {"Alive": 0}, "TRT": {"Alive": 100}})
        model.set_utility({"SOC": {"Alive": 1.0}, "TRT": {"Alive": 0.5}})
        assert dominance_label(model.run_base_case().icer()) == "Dominated"

    def test_psm(self):
        model = PSMModel(
            states=["Alive", "Dead"], survival_endpoints=["OS"],
            strategies=["SOC", "TRT"], n_cycles=1, half_cycle_correction=False,
        )
        for strategy in ("SOC", "TRT"):
            model.set_survival(strategy, "OS", NEVER_DIES)
        model.set_state_cost("care", {"SOC": {"Alive": 0}, "TRT": {"Alive": 100}})
        model.set_utility({"SOC": {"Alive": 1.0}, "TRT": {"Alive": 0.5}})
        assert dominance_label(model.run_base_case().icer()) == "Dominated"

    def test_des(self):
        model = DESModel(
            states=["Alive", "Dead"], strategies=["SOC", "TRT"], time_horizon=1.0,
        )
        model.set_state_cost("care", {"SOC": {"Alive": 0}, "TRT": {"Alive": 100}})
        model.set_utility({"SOC": {"Alive": 1.0}, "TRT": {"Alive": 0.5}})
        result = model.run(n_patients=1, seed=1, progress=False)
        assert dominance_label(result.icer()) == "Dominated"

    @pytest.mark.xfail(
        strict=True,
        reason="results.py:779-787 divides instead of classifying the quadrant, "
               "reporting a negative ICER for a dominated strategy",
    )
    def test_microsim(self):
        model = MicroSimModel(
            states=["Alive", "Dead"], strategies=["SOC", "TRT"],
            n_cycles=1, n_patients=5, half_cycle_correction=False,
        )
        for strategy in ("SOC", "TRT"):
            model.set_transitions(strategy, lambda p, t: ALIVE_FOREVER)
        model.set_state_cost("care", {"SOC": {"Alive": 0}, "TRT": {"Alive": 100}})
        model.set_utility({"SOC": {"Alive": 1.0}, "TRT": {"Alive": 0.5}})
        result = model.run_base_case(seed=1, verbose=False)
        assert dominance_label(result.icer()) == "Dominated"
