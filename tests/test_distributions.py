"""Tests for pyheor/distributions.py — PSA probability distributions."""

import numpy as np
import pytest
from pyheor.distributions import (
    Distribution, Beta, Gamma, Normal, LogNormal,
    Uniform, Triangular, Dirichlet, Fixed,
)


# =========================================================================
# Beta
# =========================================================================

class TestBeta:
    def test_sample_shape(self):
        d = Beta(mean=0.3, sd=0.1)
        assert d.sample(100).shape == (100,)

    def test_domain(self):
        samples = Beta(mean=0.3, sd=0.1).sample(5000)
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_mean_convergence(self):
        d = Beta(mean=0.3, sd=0.1)
        np.testing.assert_allclose(d.sample(50000).mean(), d.mean, atol=0.01)

    def test_from_alpha_beta(self):
        d = Beta(alpha=2, beta=5)
        np.testing.assert_allclose(d.mean, 2 / 7, rtol=1e-10)

    def test_invalid_params_raises(self):
        with pytest.raises(ValueError):
            Beta(mean=0.5, sd=10)

    def test_missing_params_raises(self):
        with pytest.raises(ValueError):
            Beta()


# =========================================================================
# Gamma
# =========================================================================

class TestGamma:
    def test_sample_shape(self):
        assert Gamma(mean=100, sd=20).sample(100).shape == (100,)

    def test_positive(self):
        samples = Gamma(mean=100, sd=20).sample(5000)
        assert np.all(samples > 0)

    def test_mean_convergence(self):
        d = Gamma(mean=100, sd=20)
        np.testing.assert_allclose(d.sample(50000).mean(), d.mean, atol=1.0)

    def test_from_shape_rate(self):
        d = Gamma(shape=2, rate=0.5)
        np.testing.assert_allclose(d.mean, 4.0, rtol=1e-10)

    def test_missing_params_raises(self):
        with pytest.raises(ValueError):
            Gamma()


# =========================================================================
# Normal
# =========================================================================

class TestNormal:
    def test_sample_shape(self):
        assert Normal(mean=0, sd=1).sample(100).shape == (100,)

    def test_mean_convergence(self):
        d = Normal(mean=5.0, sd=2.0)
        np.testing.assert_allclose(d.sample(50000).mean(), d.mean, atol=0.05)


# =========================================================================
# LogNormal
# =========================================================================

class TestLogNormal:
    def test_positive(self):
        samples = LogNormal(meanlog=0, sdlog=1).sample(5000)
        assert np.all(samples > 0)

    def test_from_meanlog_sdlog(self):
        d = LogNormal(meanlog=0, sdlog=1)
        np.testing.assert_allclose(d.mean, np.exp(0.5), rtol=1e-10)

    def test_from_mean_sd(self):
        d = LogNormal(mean=5.0, sd=1.0)
        np.testing.assert_allclose(d.mean, 5.0, rtol=0.01)

    def test_missing_params_raises(self):
        with pytest.raises(ValueError):
            LogNormal()


# =========================================================================
# Uniform
# =========================================================================

class TestUniform:
    def test_bounds(self):
        samples = Uniform(low=2, high=8).sample(5000)
        assert np.all(samples >= 2) and np.all(samples <= 8)

    def test_mean(self):
        d = Uniform(low=2, high=8)
        np.testing.assert_allclose(d.mean, 5.0)


# =========================================================================
# Triangular
# =========================================================================

class TestTriangular:
    def test_bounds(self):
        samples = Triangular(low=1, mode=3, high=5).sample(5000)
        assert np.all(samples >= 1) and np.all(samples <= 5)

    def test_mean(self):
        d = Triangular(low=1, mode=3, high=5)
        np.testing.assert_allclose(d.mean, 3.0)


# =========================================================================
# Dirichlet
# =========================================================================

class TestDirichlet:
    def test_sample_shape(self):
        d = Dirichlet([1, 2, 3])
        assert d.sample(100).shape == (100, 3)

    def test_rows_sum_to_one(self):
        samples = Dirichlet([1, 2, 3]).sample(100)
        np.testing.assert_allclose(samples.sum(axis=1), 1.0, atol=1e-10)

    def test_mean(self):
        d = Dirichlet([1, 2, 3])
        np.testing.assert_allclose(d.mean, [1/6, 2/6, 3/6])


# =========================================================================
# Fixed
# =========================================================================

class TestFixed:
    def test_constant(self):
        samples = Fixed(42).sample(10)
        assert np.all(samples == 42)

    def test_mean(self):
        assert Fixed(42).mean == 42

    def test_shape(self):
        assert Fixed(0).sample(5).shape == (5,)


# =========================================================================
# General
# =========================================================================

class TestGeneral:
    @pytest.mark.parametrize("dist", [
        Beta(mean=0.3, sd=0.1),
        Gamma(mean=100, sd=20),
        Normal(mean=0, sd=1),
        LogNormal(meanlog=0, sdlog=0.5),
        Uniform(low=0, high=1),
        Triangular(low=0, mode=0.5, high=1),
        Fixed(1.0),
    ], ids=lambda d: type(d).__name__)
    def test_repr_is_string(self, dist):
        assert isinstance(repr(dist), str)

    @pytest.mark.parametrize("dist", [
        Beta(mean=0.3, sd=0.1),
        Gamma(mean=100, sd=20),
        Normal(mean=0, sd=1),
        LogNormal(meanlog=0, sdlog=0.5),
        Uniform(low=0, high=1),
        Fixed(1.0),
    ], ids=lambda d: type(d).__name__)
    def test_sample_n_one(self, dist):
        result = dist.sample(1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
