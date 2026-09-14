"""Cross-engine consistency of the result objects' public tables.

Every engine answers the same questions, so the tables it returns should
carry the same column names and dtypes. Downstream code reads these tables
by column name, so a divergence here is a silent trap.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyheor import DESModel, Gamma, MarkovModel, MicroSimModel, PSMModel
from pyheor.survival import KaplanMeier

ALIVE_FOREVER = [[1, 0], [0, 1]]
NEVER_DIES = KaplanMeier(times=[0.0], survival_probs=[1.0])

COSTS = {"SOC": {"Alive": 0}, "TRT": {"Alive": 100}}
UTILITIES = {"SOC": {"Alive": 1.0}, "TRT": {"Alive": 0.5}}

ICER_COLUMNS = [
    "Strategy", "vs", "Incremental Cost", "Incremental QALYs",
    "Incremental LYs", "ICER", "ICER Classification",
]


def markov_result():
    model = MarkovModel(
        states=["Alive", "Dead"], strategies=["SOC", "TRT"], n_cycles=1
    )
    for strategy in ("SOC", "TRT"):
        model.set_transitions(strategy, lambda p, t: ALIVE_FOREVER)
    model.set_state_cost("care", COSTS)
    model.set_utility(UTILITIES)
    return model.run_base_case()


def psm_result():
    model = PSMModel(
        states=["Alive", "Dead"], survival_endpoints=["OS"],
        strategies=["SOC", "TRT"], n_cycles=1,
    )
    for strategy in ("SOC", "TRT"):
        model.set_survival(strategy, "OS", NEVER_DIES)
    model.set_state_cost("care", COSTS)
    model.set_utility(UTILITIES)
    return model.run_base_case()


def microsim_result():
    model = MicroSimModel(
        states=["Alive", "Dead"], strategies=["SOC", "TRT"],
        n_cycles=1, n_patients=3,
    )
    for strategy in ("SOC", "TRT"):
        model.set_transitions(strategy, lambda p, t: ALIVE_FOREVER)
    model.set_state_cost("care", COSTS)
    model.set_utility(UTILITIES)
    return model.run_base_case(seed=1, verbose=False)


def des_result():
    model = DESModel(
        states=["Alive", "Dead"], strategies=["SOC", "TRT"], time_horizon=1.0
    )
    model.set_state_cost("care", COSTS)
    model.set_utility(UTILITIES)
    return model.run(n_patients=2, seed=1, progress=False)


ENGINES = pytest.mark.parametrize(
    "factory",
    [markov_result, psm_result, microsim_result, des_result],
    ids=["markov", "psm", "microsim", "des"],
)


class TestIcerTable:
    @ENGINES
    def test_columns_are_identical(self, factory):
        assert list(factory().icer().columns) == ICER_COLUMNS

    @ENGINES
    def test_icer_is_numeric_and_classification_is_text(self, factory):
        frame = factory().icer()
        assert pd.api.types.is_numeric_dtype(frame["ICER"])
        assert isinstance(frame.iloc[0]["ICER Classification"], str)

    @ENGINES
    def test_dominated_comparison_has_no_ratio(self, factory):
        # TRT costs 100 more for 0.5 fewer QALYs in every engine.
        row = factory().icer().iloc[0]
        assert row["ICER Classification"] == "Dominated"
        assert np.isnan(row["ICER"])

    @ENGINES
    def test_unknown_comparator_is_rejected(self, factory):
        with pytest.raises(ValueError, match="Unknown comparator"):
            factory().icer(comparator="Missing")


class TestNmbTable:
    @ENGINES
    def test_reports_absolute_and_incremental_benefit(self, factory):
        frame = factory().nmb(wtp=50000)
        assert "NMB" in frame.columns
        assert "Incremental NMB" in frame.columns
        assert len(frame) == 2

    @ENGINES
    def test_comparator_row_has_zero_increment(self, factory):
        frame = factory().nmb(wtp=50000).set_index("Strategy")
        assert frame.loc["SOC", "Incremental NMB"] == 0.0

    @ENGINES
    def test_unknown_comparator_is_rejected(self, factory):
        with pytest.raises(ValueError, match="Unknown comparator"):
            factory().nmb(comparator="Missing")


class TestPsaPlotShortcuts:
    """Smoke-test the plot helpers, which only run end to end.

    These call through to ``ceac_data`` and ``ce_table``, so a signature
    change that the unit tests miss shows up here rather than in an example.
    """

    @pytest.fixture
    def psa(self):
        model = MarkovModel(
            states=["Alive", "Dead"], strategies=["SOC", "TRT"], n_cycles=2
        )
        model.add_param("c_trt", base=100, dist=Gamma(mean=100, sd=10))
        for strategy in ("SOC", "TRT"):
            model.set_transitions(strategy, lambda p, t: ALIVE_FOREVER)
        model.set_state_cost(
            "care", {"SOC": {"Alive": 0}, "TRT": {"Alive": "c_trt"}}
        )
        model.set_utility({"SOC": {"Alive": 0.5}, "TRT": {"Alive": 0.9}})
        return model.run_psa(n_sim=5, seed=1, progress=False)

    @pytest.mark.parametrize(
        "method", ["plot_ceac", "plot_scatter", "plot_convergence"]
    )
    def test_renders(self, psa, method):
        figure = getattr(psa, method)()
        try:
            assert figure is not None
        finally:
            plt.close(figure)

    def test_ceac_probabilities_sum_to_one_per_threshold(self, psa):
        # The curve is a probability of being optimal, so at each threshold
        # the strategies partition all simulations.
        ceac = psa.ceac_data(wtp_range=(0, 100000), n_wtp=5)
        totals = ceac.groupby("WTP")["Prob CE"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 1.0)
