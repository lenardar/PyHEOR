"""Tests for pyheor/des.py — DESModel integration tests."""

import numpy as np
import pytest
from pyheor import DESModel
from pyheor.survival import Exponential, Weibull


class TestDESConstruction:
    def test_basic(self):
        model = DESModel(
            states=["Alive", "Dead"],
            strategies=["S1"],
            time_horizon=20,
        )
        assert model is not None


class TestDESRun:
    @pytest.fixture
    def des_model(self):
        model = DESModel(
            states=["PFS", "Progressed", "Dead"],
            strategies=["SOC", "TRT"],
            time_horizon=20,
            discount_rate=0.03,
        )
        model.set_event("SOC", "PFS", "Progressed", Weibull(shape=1.2, scale=5.0))
        model.set_event("SOC", "PFS", "Dead", Exponential(rate=0.01))
        model.set_event("SOC", "Progressed", "Dead", Weibull(shape=1.5, scale=3.0))
        model.set_event("TRT", "PFS", "Progressed", Weibull(shape=1.2, scale=7.0))
        model.set_event("TRT", "PFS", "Dead", Exponential(rate=0.01))
        model.set_event("TRT", "Progressed", "Dead", Weibull(shape=1.5, scale=3.0))

        model.set_state_cost("drug", {
            "SOC": {"PFS": 1000, "Progressed": 500, "Dead": 0},
            "TRT": {"PFS": 3000, "Progressed": 500, "Dead": 0},
        })
        model.set_utility({"PFS": 0.85, "Progressed": 0.50, "Dead": 0})
        return model

    def test_run_basic(self, des_model):
        result = des_model.run(n_patients=100, seed=42)
        summary = result.summary()
        assert "Mean QALYs" in summary.columns

    def test_time_horizon_respected(self, des_model):
        result = des_model.run(n_patients=100, seed=42)
        summary = result.summary()
        # DES results exist and have expected columns
        assert "Strategy" in summary.columns

    def test_results_summary(self, des_model):
        result = des_model.run(n_patients=100, seed=42)
        summary = result.summary()
        assert "Strategy" in summary.columns
        assert "Mean Cost" in summary.columns

    def test_known_exponential(self):
        """With pure exponential rate=0.1, mean survival ~ 10."""
        model = DESModel(
            states=["Alive", "Dead"],
            strategies=["S1"],
            time_horizon=100,
            discount_rate=0,
        )
        model.set_event("S1", "Alive", "Dead", Exponential(rate=0.1))
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        result = model.run(n_patients=2000, seed=42)
        mean_lys = result.summary()["Mean LYs"].iloc[0]
        np.testing.assert_allclose(mean_lys, 10.0, rtol=0.20)
