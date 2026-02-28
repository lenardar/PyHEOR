"""Tests for pyheor/survival.py — parametric survival distributions."""

import numpy as np
import pytest
from pyheor.survival import (
    SurvivalDistribution, Exponential, Weibull, LogLogistic,
    SurvLogNormal, Gompertz, GeneralizedGamma,
    ProportionalHazards, AcceleratedFailureTime,
    KaplanMeier, PiecewiseExponential,
)

# Test points
T_POINTS = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
P_POINTS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

# All 6 base distributions for parametrized tests
ALL_BASE_DISTS = [
    Exponential(rate=0.1),
    Weibull(shape=1.5, scale=10),
    LogLogistic(shape=2.0, scale=8.0),
    SurvLogNormal(meanlog=2.0, sdlog=0.5),
    Gompertz(shape=0.05, rate=0.02),
    GeneralizedGamma(mu=2.0, sigma=0.5, Q=1.0),
]


# =========================================================================
# Universal invariants (parametrized across all distributions)
# =========================================================================

@pytest.mark.parametrize("dist", ALL_BASE_DISTS, ids=lambda d: type(d).__name__)
class TestSurvivalInvariants:
    def test_survival_at_zero(self, dist):
        np.testing.assert_allclose(dist.survival(0), 1.0, atol=1e-10)

    def test_survival_monotone_decreasing(self, dist):
        t = np.linspace(0.01, 50, 200)
        s = dist.survival(t)
        assert np.all(np.diff(s) <= 1e-10), "S(t) must be non-increasing"

    def test_pdf_equals_hazard_times_survival(self, dist):
        for t in T_POINTS:
            if t == 0:
                continue
            expected = dist.hazard(t) * dist.survival(t)
            actual = dist.pdf(t)
            np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_cumhaz_equals_neg_log_survival(self, dist):
        for t in T_POINTS:
            s = dist.survival(t)
            if s > 1e-15:
                expected = -np.log(s)
                actual = dist.cumulative_hazard(t)
                np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_quantile_inverts_survival(self, dist):
        for p in [0.1, 0.5, 0.9]:
            t = dist.quantile(p)
            if np.isfinite(t) and t > 0:
                s_at_t = dist.survival(t)
                np.testing.assert_allclose(s_at_t, 1 - p, atol=1e-3)

    def test_array_input(self, dist):
        t = np.array([1.0, 5.0, 10.0])
        assert dist.survival(t).shape == (3,)
        assert dist.hazard(t).shape == (3,)


# =========================================================================
# Exponential
# =========================================================================

class TestExponential:
    def test_known_values(self):
        d = Exponential(rate=0.1)
        np.testing.assert_allclose(d.survival(10), np.exp(-1), rtol=1e-10)

    def test_constant_hazard(self):
        d = Exponential(rate=0.1)
        for t in [1, 5, 10, 50]:
            np.testing.assert_allclose(d.hazard(t), 0.1, rtol=1e-10)

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            Exponential(rate=0)
        with pytest.raises(ValueError):
            Exponential(rate=-1)

    def test_quantile_median(self):
        d = Exponential(rate=0.1)
        expected_median = -np.log(0.5) / 0.1
        np.testing.assert_allclose(d.quantile(0.5), expected_median, rtol=1e-10)

    def test_restricted_mean(self):
        d = Exponential(rate=0.1)
        T = 20.0
        analytic = (1 - np.exp(-0.1 * T)) / 0.1
        np.testing.assert_allclose(d.restricted_mean(T), analytic, rtol=1e-3)


# =========================================================================
# Weibull
# =========================================================================

class TestWeibull:
    def test_shape_one_is_exponential(self):
        rate = 0.1
        w = Weibull(shape=1, scale=1 / rate)
        e = Exponential(rate=rate)
        for t in T_POINTS:
            np.testing.assert_allclose(
                w.survival(t), e.survival(t), rtol=1e-10
            )

    def test_known_median(self):
        w = Weibull(shape=1.5, scale=10)
        expected = 10 * np.log(2) ** (1 / 1.5)
        np.testing.assert_allclose(w.quantile(0.5), expected, rtol=1e-10)

    def test_from_ph(self):
        w_ph = Weibull.from_ph(shape=1.5, scale=0.01)
        # Verify it's a valid Weibull with correct survival values
        assert isinstance(w_ph, Weibull)
        np.testing.assert_allclose(w_ph.survival(0), 1.0)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            Weibull(shape=0, scale=10)
        with pytest.raises(ValueError):
            Weibull(shape=1, scale=-1)


# =========================================================================
# LogLogistic
# =========================================================================

class TestLogLogistic:
    def test_median_equals_scale(self):
        d = LogLogistic(shape=2.0, scale=8.0)
        np.testing.assert_allclose(d.quantile(0.5), 8.0, rtol=1e-10)

    def test_survival_formula(self):
        d = LogLogistic(shape=2.0, scale=8.0)
        t = 4.0
        expected = 1.0 / (1.0 + (4.0 / 8.0) ** 2)
        np.testing.assert_allclose(d.survival(t), expected, rtol=1e-10)


# =========================================================================
# SurvLogNormal
# =========================================================================

class TestSurvLogNormal:
    def test_median(self):
        d = SurvLogNormal(meanlog=2.0, sdlog=0.5)
        np.testing.assert_allclose(d.survival(np.exp(2.0)), 0.5, atol=1e-6)

    def test_invalid_sdlog(self):
        with pytest.raises(ValueError):
            SurvLogNormal(meanlog=0, sdlog=0)


# =========================================================================
# Gompertz
# =========================================================================

class TestGompertz:
    def test_shape_near_zero_is_exponential(self):
        g = Gompertz(shape=1e-13, rate=0.1)
        e = Exponential(rate=0.1)
        for t in [1, 5, 10]:
            np.testing.assert_allclose(
                g.survival(t), e.survival(t), rtol=1e-3
            )

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            Gompertz(shape=0.05, rate=0)


# =========================================================================
# GeneralizedGamma
# =========================================================================

class TestGeneralizedGamma:
    def test_q_one_approximates_weibull(self):
        gg = GeneralizedGamma(mu=np.log(10), sigma=1 / 1.5, Q=1.0)
        w = Weibull(shape=1.5, scale=10)
        for t in [2, 5, 10]:
            np.testing.assert_allclose(
                gg.survival(t), w.survival(t), rtol=0.05
            )

    def test_q_zero_approximates_lognormal(self):
        gg = GeneralizedGamma(mu=2.0, sigma=0.5, Q=0.0)
        ln = SurvLogNormal(meanlog=2.0, sdlog=0.5)
        for t in [2, 5, 10]:
            np.testing.assert_allclose(
                gg.survival(t), ln.survival(t), rtol=0.01
            )


# =========================================================================
# ProportionalHazards
# =========================================================================

class TestProportionalHazards:
    def test_survival(self):
        base = Weibull(shape=1.5, scale=10)
        ph = ProportionalHazards(base, hr=0.7)
        for t in T_POINTS:
            expected = base.survival(t) ** 0.7
            np.testing.assert_allclose(ph.survival(t), expected, rtol=1e-6)

    def test_hazard(self):
        base = Weibull(shape=1.5, scale=10)
        ph = ProportionalHazards(base, hr=0.7)
        for t in T_POINTS:
            if t > 0:
                expected = base.hazard(t) * 0.7
                np.testing.assert_allclose(ph.hazard(t), expected, rtol=1e-6)

    def test_hr_one_equals_baseline(self):
        base = Weibull(shape=1.5, scale=10)
        ph = ProportionalHazards(base, hr=1.0)
        for t in T_POINTS:
            np.testing.assert_allclose(
                ph.survival(t), base.survival(t), rtol=1e-10
            )


# =========================================================================
# AcceleratedFailureTime
# =========================================================================

class TestAcceleratedFailureTime:
    def test_survival(self):
        base = Weibull(shape=1.5, scale=10)
        aft = AcceleratedFailureTime(base, af=1.3)
        for t in T_POINTS:
            expected = base.survival(t / 1.3)
            np.testing.assert_allclose(aft.survival(t), expected, rtol=1e-6)

    def test_af_one_equals_baseline(self):
        base = Weibull(shape=1.5, scale=10)
        aft = AcceleratedFailureTime(base, af=1.0)
        for t in T_POINTS:
            np.testing.assert_allclose(
                aft.survival(t), base.survival(t), rtol=1e-10
            )


# =========================================================================
# KaplanMeier
# =========================================================================

class TestKaplanMeier:
    def test_survival_at_zero(self):
        km = KaplanMeier(times=[1, 3, 5], survival_probs=[0.9, 0.7, 0.5])
        np.testing.assert_allclose(km.survival(0), 1.0)

    def test_step_function(self):
        km = KaplanMeier(
            times=[0, 2, 5, 10],
            survival_probs=[1.0, 0.8, 0.6, 0.3],
        )
        # Between t=2 and t=5, should be 0.8
        np.testing.assert_allclose(km.survival(3), 0.8)

    def test_prepends_zero(self):
        km = KaplanMeier(times=[5, 10], survival_probs=[0.8, 0.5])
        np.testing.assert_allclose(km.survival(0), 1.0)


# =========================================================================
# PiecewiseExponential
# =========================================================================

class TestPiecewiseExponential:
    def test_single_piece_is_exponential(self):
        pe = PiecewiseExponential(breakpoints=[], rates=[0.1])
        e = Exponential(rate=0.1)
        for t in T_POINTS:
            np.testing.assert_allclose(
                pe.survival(t), e.survival(t), rtol=1e-6
            )

    def test_continuity_at_breakpoint(self):
        pe = PiecewiseExponential(breakpoints=[5], rates=[0.1, 0.2])
        eps = 1e-8
        np.testing.assert_allclose(
            pe.survival(5 - eps), pe.survival(5 + eps), rtol=1e-4
        )
