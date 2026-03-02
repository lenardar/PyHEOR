"""Tests for pyheor/psm.py — PSMModel."""

import numpy as np
import pytest
from pyheor.analysis.results import PSMBaseResult, PSAResult


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
