"""Tests for pyheor/microsim.py — MicroSimModel integration tests."""

import numpy as np
import pytest
from pyheor import MicroSimModel, PatientProfile, C


class TestMicroSimConstruction:
    def test_basic(self):
        model = MicroSimModel(
            states=["Alive", "Dead"],
            strategies=["S1"],
            n_cycles=5,
            n_patients=50,
            seed=42,
        )
        assert model is not None


class TestMicroSimRun:
    @pytest.fixture
    def micro_model(self):
        model = MicroSimModel(
            states=["Alive", "Dead"],
            strategies=["SOC", "TRT"],
            n_cycles=10,
            n_patients=100,
            cycle_length=1.0,
            dr_cost=0.03,
            dr_qaly=0.03,
            seed=42,
        )
        model.add_param("p_death", base=0.1)
        model.add_param("hr", base=0.7)
        model.set_transitions("SOC", lambda p, t: [
            [C, p["p_death"]],
            [0, 1],
        ])
        model.set_transitions("TRT", lambda p, t: [
            [C, p["p_death"] * p["hr"]],
            [0, 1],
        ])
        model.set_state_cost("medical", {"Alive": 1000, "Dead": 0})
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        return model

    def test_base_case_runs(self, micro_model):
        result = micro_model.run_base_case()
        summary = result.summary()
        assert "Mean QALYs" in summary.columns

    def test_all_absorbing(self):
        model = MicroSimModel(
            states=["Alive", "Dead"],
            strategies=["S1"],
            n_cycles=3,
            n_patients=50,
            seed=42,
        )
        model.set_transitions("S1", lambda p, t: [
            [0, 1],
            [0, 1],
        ])
        model.set_utility({"Alive": 1, "Dead": 0})
        result = model.run_base_case()
        # After 1 cycle all should be dead -> low QALYs
        q = result.summary()["Mean QALYs"].iloc[0]
        assert q < 1.5  # at most ~1 cycle alive

    def test_results_summary_columns(self, micro_model):
        result = micro_model.run_base_case()
        summary = result.summary()
        assert "Strategy" in summary.columns
        assert "Mean Cost" in summary.columns
