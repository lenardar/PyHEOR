"""Tests for pyheor/nma.py — NMAPosterior, make_ph_curves, make_aft_curves."""

import numpy as np
import pytest
from pyheor.nma import NMAPosterior, PosteriorDist, make_ph_curves, make_aft_curves
from pyheor.survival import Weibull, ProportionalHazards, AcceleratedFailureTime


@pytest.fixture
def nma_posterior():
    """Create a simple NMAPosterior from random data."""
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=[[0.7, 0.6]], scale=0.1, size=(200, 2))
    return NMAPosterior(
        samples=samples,
        treatment_names=["Drug_A", "Drug_B"],
    )


class TestNMAPosterior:
    def test_construction(self, nma_posterior):
        assert nma_posterior.n_treatments == 2
        assert nma_posterior.n_iter == 200

    def test_summary(self, nma_posterior):
        summary = nma_posterior.summary()
        assert "mean" in summary.columns or "Mean" in summary.columns

    def test_indexing(self, nma_posterior):
        col = nma_posterior["Drug_A"]
        assert len(col) == 200

    def test_dist(self, nma_posterior):
        d = nma_posterior.dist("Drug_A")
        assert isinstance(d, PosteriorDist)
        samples = d.sample(50)
        assert samples.shape == (50,)

    def test_posterior_dist_mean(self, nma_posterior):
        d = nma_posterior.dist("Drug_A")
        col = nma_posterior["Drug_A"]
        np.testing.assert_allclose(d.mean, np.mean(col), rtol=1e-10)


class TestMakeCurves:
    def test_make_ph_curves(self, nma_posterior):
        baseline = Weibull(shape=1.5, scale=10)
        curves = make_ph_curves(baseline, nma_posterior)
        assert "Drug_A" in curves
        assert "Drug_B" in curves
        assert isinstance(curves["Drug_A"], ProportionalHazards)

    def test_make_aft_curves(self, nma_posterior):
        baseline = Weibull(shape=1.5, scale=10)
        curves = make_aft_curves(baseline, nma_posterior)
        assert "Drug_A" in curves
        assert isinstance(curves["Drug_A"], AcceleratedFailureTime)
