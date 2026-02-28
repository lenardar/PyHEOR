"""Tests for pyheor/comparison.py — CEAnalysis, ICER, NMB."""

import numpy as np
import pytest
from pyheor.comparison import CEAnalysis, calculate_icers


# =========================================================================
# calculate_icers
# =========================================================================

class TestCalculateICERs:
    def test_two_strategies(self):
        df = calculate_icers(
            strategies=["A", "B"],
            costs=[10000, 20000],
            qalys=[5.0, 7.0],
        )
        assert len(df) == 2
        assert df.iloc[0]["Status"] == "Ref"
        # ICER = (20000-10000)/(7-5) = 5000
        icer_b = df[df["Strategy"] == "B"]["ICER"].values[0]
        np.testing.assert_allclose(icer_b, 5000)

    def test_strong_dominance(self):
        # C costs more and has fewer QALYs than B -> dominated
        df = calculate_icers(
            strategies=["A", "B", "C"],
            costs=[10000, 20000, 25000],
            qalys=[5.0, 8.0, 6.0],
        )
        status_c = df[df["Strategy"] == "C"]["Status"].values[0]
        assert status_c == "D"

    def test_single_strategy(self):
        df = calculate_icers(
            strategies=["A"],
            costs=[10000],
            qalys=[5.0],
        )
        assert df.iloc[0]["Status"] == "Ref"

    def test_three_on_frontier(self):
        # All non-dominated with increasing ICERs
        df = calculate_icers(
            strategies=["A", "B", "C"],
            costs=[10000, 20000, 50000],
            qalys=[5.0, 8.0, 15.0],
        )
        on_frontier = df[df["Status"].isin(["Ref", "ND"])]
        assert len(on_frontier) == 3


# =========================================================================
# CEAnalysis construction
# =========================================================================

class TestCEAnalysis:
    def test_construction(self):
        cea = CEAnalysis(
            strategies=["A", "B"],
            costs=[10000, 20000],
            qalys=[5.0, 7.0],
        )
        assert cea is not None

    def test_frontier(self):
        cea = CEAnalysis(
            strategies=["A", "B", "C"],
            costs=[10000, 20000, 25000],
            qalys=[5.0, 8.0, 6.0],
        )
        f = cea.frontier()
        assert "Status" in f.columns

    def test_frontier_strategies(self):
        cea = CEAnalysis(
            strategies=["A", "B", "C"],
            costs=[10000, 20000, 25000],
            qalys=[5.0, 8.0, 6.0],
        )
        fs = cea.frontier_strategies()
        assert "C" not in fs  # C is dominated

    def test_is_dominated(self):
        cea = CEAnalysis(
            strategies=["A", "B", "C"],
            costs=[10000, 20000, 25000],
            qalys=[5.0, 8.0, 6.0],
        )
        assert cea.is_dominated("C")
        assert not cea.is_dominated("A")


# =========================================================================
# NMB
# =========================================================================

class TestNMB:
    def test_formula(self):
        cea = CEAnalysis(
            strategies=["A", "B"],
            costs=[10000, 20000],
            qalys=[5.0, 7.0],
        )
        nmb = cea.nmb(wtp=50000)
        # NMB_A = 5*50000 - 10000 = 240000
        # NMB_B = 7*50000 - 20000 = 330000
        nmb_a = nmb[nmb["Strategy"] == "A"]["NMB"].values[0]
        nmb_b = nmb[nmb["Strategy"] == "B"]["NMB"].values[0]
        np.testing.assert_allclose(nmb_a, 240000)
        np.testing.assert_allclose(nmb_b, 330000)

    def test_optimal_wtp_zero(self):
        """At WTP=0, cheapest strategy is optimal."""
        cea = CEAnalysis(
            strategies=["Cheap", "Expensive"],
            costs=[5000, 50000],
            qalys=[5.0, 7.0],
        )
        assert cea.optimal_strategy(wtp=0) == "Cheap"

    def test_optimal_wtp_high(self):
        """At very high WTP, highest-QALY strategy is optimal."""
        cea = CEAnalysis(
            strategies=["A", "B"],
            costs=[10000, 20000],
            qalys=[5.0, 10.0],
        )
        assert cea.optimal_strategy(wtp=1e8) == "B"


# =========================================================================
# CEAF & EVPI (require PSA data)
# =========================================================================

class TestPSAAnalysis:
    @pytest.fixture
    def cea_with_psa(self):
        rng = np.random.default_rng(42)
        n_sim = 100
        psa_costs = rng.normal(
            loc=[[10000, 20000]], scale=[[1000, 2000]], size=(n_sim, 2)
        )
        psa_qalys = rng.normal(
            loc=[[5.0, 7.0]], scale=[[0.5, 0.7]], size=(n_sim, 2)
        )
        return CEAnalysis(
            strategies=["A", "B"],
            costs=[10000, 20000],
            qalys=[5.0, 7.0],
            psa_costs=psa_costs,
            psa_qalys=psa_qalys,
        )

    def test_ceaf(self, cea_with_psa):
        ceaf = cea_with_psa.ceaf(wtp_range=(0, 100000), n_wtp=10)
        assert len(ceaf) > 0

    def test_evpi_nonnegative(self, cea_with_psa):
        evpi = cea_with_psa.evpi(wtp_range=(0, 100000), n_wtp=10)
        assert np.all(evpi["EVPI"].values >= -1e-10)

    def test_evpi_single(self, cea_with_psa):
        val = cea_with_psa.evpi_single(50000)
        assert val >= 0
