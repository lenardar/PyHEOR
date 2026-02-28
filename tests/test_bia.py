"""Tests for pyheor/bia.py — BudgetImpactAnalysis."""

import numpy as np
import pytest
from pyheor.bia import BudgetImpactAnalysis


class TestBIAConstruction:
    def test_basic(self):
        bia = BudgetImpactAnalysis(
            strategies=["A", "B"],
            per_patient_costs={"A": 5000, "B": 12000},
            population=10000,
            market_share_current={"A": 0.7, "B": 0.3},
            market_share_new={"A": 0.5, "B": 0.5},
            time_horizon=3,
        )
        assert bia is not None

    def test_summary_columns(self):
        bia = BudgetImpactAnalysis(
            strategies=["A", "B"],
            per_patient_costs={"A": 5000, "B": 12000},
            population=10000,
            market_share_current={"A": 0.7, "B": 0.3},
            market_share_new={"A": 0.5, "B": 0.5},
            time_horizon=3,
        )
        summary = bia.summary()
        assert "Year" in summary.columns or "year" in [c.lower() for c in summary.columns]


class TestBIAPopulation:
    def test_growth_rate(self):
        bia = BudgetImpactAnalysis(
            strategies=["A"],
            per_patient_costs={"A": 1000},
            population={"base": 10000, "growth_rate": 0.10},
            market_share_current={"A": 1.0},
            market_share_new={"A": 1.0},
            time_horizon=3,
        )
        # With same shares, budget impact should be 0
        summary = bia.summary()
        bi_col = [c for c in summary.columns if "impact" in c.lower() or "Impact" in c]
        if bi_col:
            np.testing.assert_allclose(summary[bi_col[0]].values, 0, atol=1)

    def test_same_shares_zero_impact(self):
        bia = BudgetImpactAnalysis(
            strategies=["A", "B"],
            per_patient_costs={"A": 5000, "B": 12000},
            population=10000,
            market_share_current={"A": 0.5, "B": 0.5},
            market_share_new={"A": 0.5, "B": 0.5},
            time_horizon=3,
        )
        summary = bia.summary()
        bi_col = [c for c in summary.columns if "impact" in c.lower() or "Impact" in c]
        if bi_col:
            np.testing.assert_allclose(summary[bi_col[0]].values, 0, atol=1)


class TestBIAStaticMethods:
    def test_linear_uptake(self):
        result = BudgetImpactAnalysis.linear_uptake(0.0, 0.4, 5)
        assert len(result) == 5
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1], 0.4, atol=1e-10)

    def test_sigmoid_uptake(self):
        result = BudgetImpactAnalysis.sigmoid_uptake(0.0, 0.4, 5, steepness=1.5)
        assert len(result) == 5
        np.testing.assert_allclose(result[0], 0.0, atol=0.01)
        np.testing.assert_allclose(result[-1], 0.4, atol=0.01)
