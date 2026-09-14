"""One-way sensitivity analysis behaviour."""

import numpy as np
import pytest

from pyheor import MarkovModel

ALIVE_FOREVER = [[1, 0], [0, 1]]


def owsa_model():
    """Two strategies over one interval, so outcomes are exact.

    ``SOC`` yields 0.5 QALY at no cost. ``TRT`` yields ``u_trt`` QALY at
    ``c_trt``. At the low bound of ``u_trt`` the treatment becomes both
    costlier and less effective, i.e. dominated.
    """
    model = MarkovModel(
        states=["Alive", "Dead"],
        strategies=["SOC", "TRT"],
        n_cycles=1,
        half_cycle_correction=False,
    )
    model.add_param("u_trt", base=0.6, low=0.4, high=0.8)
    model.add_param("c_trt", base=100, low=90, high=110)
    for strategy in ("SOC", "TRT"):
        model.set_transitions(strategy, lambda p, t: ALIVE_FOREVER)
    model.set_utility({"SOC": {"Alive": 0.5}, "TRT": {"Alive": "u_trt"}})
    model.set_state_cost("care", {"SOC": {"Alive": 0}, "TRT": {"Alive": "c_trt"}})
    return model


class TestIcerRanking:
    def test_scenario_setup_is_exact(self):
        """Guard the arithmetic the ranking assertions depend on."""
        frame = owsa_model().run_owsa(wtp=50000).summary(outcome="icer")
        rows = frame.set_index("param_name")

        assert rows.loc["c_trt", "ICER (Low)"] == pytest.approx(900.0)
        assert rows.loc["c_trt", "ICER (High)"] == pytest.approx(1100.0)
        assert rows.loc["u_trt", "ICER (High)"] == pytest.approx(1000.0 / 3.0)

    def test_dominated_bound_is_reported_as_such(self):
        frame = owsa_model().run_owsa(wtp=50000).summary(outcome="icer")
        rows = frame.set_index("param_name")

        assert np.isnan(rows.loc["u_trt", "ICER (Low)"])
        assert rows.loc["u_trt", "ICER Classification (Low)"] == "Dominated"

    def test_a_dominated_bound_ranks_first(self):
        """A parameter that can flip the conclusion is the most sensitive one.

        Its ICER range is not a finite number, so it must not sort behind
        parameters whose range is merely large.
        """
        frame = owsa_model().run_owsa(wtp=50000).summary(outcome="icer")

        assert frame.iloc[0]["param_name"] == "u_trt"
        assert frame.iloc[0]["Range"] == float("inf")

    def test_finite_ranges_keep_descending_order(self):
        frame = owsa_model().run_owsa(wtp=50000).summary(outcome="icer")
        finite = frame[np.isfinite(frame["Range"])]["Range"].to_numpy()

        assert np.all(np.diff(finite) <= 0)
        assert not np.isnan(frame["Range"]).any()


class TestNmbRanking:
    def test_ranks_by_inmb_span(self):
        frame = owsa_model().run_owsa(wtp=50000).summary(outcome="nmb")
        spans = (frame["INMB (High)"] - frame["INMB (Low)"]).abs().to_numpy()

        np.testing.assert_allclose(frame["Range"].to_numpy(), spans)
        assert np.all(np.diff(frame["Range"].to_numpy()) <= 0)
