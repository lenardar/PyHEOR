"""Tests for pyheor/fitting.py — SurvivalFitter, kaplan_meier."""

import numpy as np
import pytest
from pyheor.evidence.fitting import SurvivalFitter, kaplan_meier


# =========================================================================
# kaplan_meier
# =========================================================================

class TestKaplanMeier:
    def test_returns_dataframe(self):
        time = np.array([1, 2, 3, 4, 5])
        event = np.array([1, 1, 0, 1, 1])
        km = kaplan_meier(time, event)
        for col in ["time", "n_risk", "n_event", "survival"]:
            assert col in km.columns

    def test_starts_at_one(self):
        km = kaplan_meier(np.array([1, 2, 3]), np.array([1, 1, 1]))
        assert km.iloc[0]["survival"] == 1.0

    def test_monotone_decreasing(self):
        rng = np.random.default_rng(42)
        time = rng.exponential(5, 50)
        event = rng.binomial(1, 0.8, 50)
        km = kaplan_meier(time, event)
        diffs = np.diff(km["survival"].values)
        assert np.all(diffs <= 1e-10)

    def test_ci_bounds(self):
        rng = np.random.default_rng(42)
        time = rng.exponential(5, 100)
        event = np.ones(100, dtype=int)
        km = kaplan_meier(time, event)
        # Where SE > 0, lower <= survival <= upper
        valid = km["se"] > 0
        assert np.all(km.loc[valid, "lower"] <= km.loc[valid, "survival"] + 1e-10)
        assert np.all(km.loc[valid, "survival"] <= km.loc[valid, "upper"] + 1e-10)


# =========================================================================
# SurvivalFitter
# =========================================================================

class TestSurvivalFitter:
    def test_fit_all_distributions(self, weibull_ipd_data):
        time, event, _, _ = weibull_ipd_data
        fitter = SurvivalFitter(time, event, label="Test")
        fitter.fit(verbose=False)
        summary = fitter.summary()
        assert len(summary) == 6  # 6 distributions

    def test_summary_columns(self, weibull_ipd_data):
        time, event, _, _ = weibull_ipd_data
        fitter = SurvivalFitter(time, event)
        fitter.fit(verbose=False)
        summary = fitter.summary()
        for col in ["Distribution", "AIC", "BIC"]:
            assert col in summary.columns

    def test_best_model_has_lowest_aic(self, weibull_ipd_data):
        time, event, _, _ = weibull_ipd_data
        fitter = SurvivalFitter(time, event)
        fitter.fit(verbose=False)
        best = fitter.best_model()
        summary = fitter.summary(sort_by="aic")
        assert best.aic == pytest.approx(summary["AIC"].iloc[0])

    def test_recover_weibull_params(self, weibull_ipd_data):
        time, event, true_shape, true_scale = weibull_ipd_data
        fitter = SurvivalFitter(time, event)
        fitter.fit(verbose=False)
        result = fitter.get_result("Weibull")
        fitted_shape = result.params.get(
            "shape", result.params.get("k", None)
        )
        fitted_scale = result.params.get(
            "scale", result.params.get("lambda", None)
        )
        if fitted_shape is not None:
            np.testing.assert_allclose(
                fitted_shape, true_shape, rtol=0.2
            )
        if fitted_scale is not None:
            np.testing.assert_allclose(
                fitted_scale, true_scale, rtol=0.2
            )

    def test_get_distribution(self, weibull_ipd_data):
        time, event, _, _ = weibull_ipd_data
        fitter = SurvivalFitter(time, event)
        fitter.fit(verbose=False)
        dist = fitter.get_distribution("Weibull")
        assert hasattr(dist, "survival")
        np.testing.assert_allclose(dist.survival(0), 1.0)
