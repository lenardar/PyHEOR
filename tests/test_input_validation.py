"""Constructor validation for distributions, curves and analyses.

Inputs that cannot describe what the caller meant are rejected where they are
supplied, rather than producing a silently different model or failing later
in the middle of a simulation.
"""
import pyheor as _ph
from tests.input_helpers import _method, _cycle_values, _starting_values



import numpy as np
import pytest
from pyheor import Dirichlet, Fixed, LogNormal, Normal, Triangular, Uniform
from pyheor.analysis.comparison import CEAnalysis
from pyheor.distributions import sample_distribution
from pyheor.survival import Gompertz, KaplanMeier, PiecewiseExponential

class TestDistributionBounds:

    def test_lognormal_rejects_non_positive_mean(self):
        with pytest.raises(ValueError, match='mean must be finite and positive'):
            LogNormal(mean=-1, sd=0.5)

    @pytest.mark.parametrize('sd', [0, -1, np.nan, np.inf])
    def test_lognormal_rejects_degenerate_sd(self, sd):
        with pytest.raises(ValueError, match='sd must be finite and positive'):
            LogNormal(mean=1.0, sd=sd)

    @pytest.mark.parametrize('sd', [0, -1, np.nan, np.inf])
    def test_normal_rejects_degenerate_sd(self, sd):
        with pytest.raises(ValueError, match='sd must be finite and positive'):
            Normal(mean=1.0, sd=sd)

    def test_normal_suggests_fixed_for_zero_uncertainty(self):
        with pytest.raises(ValueError, match='Fixed'):
            Normal(mean=1.0, sd=0)
        assert Fixed(1.0).sample(3).tolist() == [1.0, 1.0, 1.0]

    def test_uniform_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match='low must be less than high'):
            Uniform(low=5, high=1)

    def test_triangular_rejects_mode_outside_bounds(self):
        with pytest.raises(ValueError, match='mode must lie within'):
            Triangular(low=0, mode=5, high=1)

    @pytest.mark.parametrize('alpha', [[1, 0, 2], [1, -1], [1, np.nan]])
    def test_dirichlet_rejects_non_positive_alpha(self, alpha):
        with pytest.raises(ValueError, match='finite and positive'):
            Dirichlet(alpha)

    def test_dirichlet_cannot_fill_a_scalar_parameter(self):
        with pytest.raises(TypeError, match='scalar parameter'):
            sample_distribution(Dirichlet([1, 2, 3]), 1, np.random.default_rng(0))

    def test_dirichlet_still_samples_directly(self):
        draws = Dirichlet([1, 2, 3]).sample(4, rng=np.random.default_rng(0))
        assert draws.shape == (4, 3)
        np.testing.assert_allclose(draws.sum(axis=1), 1.0)

class TestKaplanMeierInputs:

    def test_rejects_empty_times(self):
        with pytest.raises(ValueError, match='non-empty'):
            KaplanMeier(times=[], survival_probs=[])

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match='same length'):
            KaplanMeier(times=[0, 1, 2], survival_probs=[1.0, 0.5])

    @pytest.mark.parametrize('probs', [[1.0, 1.2], [1.0, -0.1]])
    def test_rejects_probabilities_outside_unit_interval(self, probs):
        with pytest.raises(ValueError, match='\\[0, 1\\]'):
            KaplanMeier(times=[0, 1], survival_probs=probs)

    def test_rejects_increasing_survival(self):
        with pytest.raises(ValueError, match='non-increasing'):
            KaplanMeier(times=[0, 1, 2], survival_probs=[1.0, 0.4, 0.8])

    def test_rejects_misspelled_extrapolation(self):
        with pytest.raises(ValueError, match='extrapolation'):
            KaplanMeier(times=[0, 1], survival_probs=[1.0, 0.5], extrapolation='exponetial')

class TestPiecewiseExponentialInputs:

    def test_rejects_negative_rates(self):
        with pytest.raises(ValueError, match='non-negative'):
            PiecewiseExponential(breakpoints=[3.0], rates=[0.1, -0.2])

    def test_rejects_unsorted_breakpoints(self):
        with pytest.raises(ValueError, match='strictly increasing'):
            PiecewiseExponential(breakpoints=[5.0, 3.0], rates=[0.1, 0.2, 0.3])

    def test_rejects_non_positive_breakpoints(self):
        with pytest.raises(ValueError, match='positive'):
            PiecewiseExponential(breakpoints=[0.0], rates=[0.1, 0.2])

class TestQuantileInversion:

    def test_unreachable_quantile_is_infinite_not_nan(self):
        """A cure fraction means the quantile does not exist, not NaN.

        NaN would propagate silently into any downstream sum.
        """
        curve = Gompertz(shape=-0.5, rate=0.01)
        plateau = float(curve.survival(1000000.0))
        assert plateau > 0.5
        assert np.isinf(curve.quantile(0.99))

    def test_reachable_quantile_inverts_correctly(self):
        curve = Gompertz(shape=0.1, rate=0.05)
        t = curve.quantile(0.5)
        assert np.isfinite(t)
        assert float(curve.survival(t)) == pytest.approx(0.5, abs=1e-08)

    def test_large_time_scales_are_supported(self):
        curve = Gompertz(shape=1e-07, rate=1e-07)
        t = curve.quantile(0.5)
        assert np.isfinite(t)
        assert float(curve.survival(t)) == pytest.approx(0.5, abs=1e-08)

class TestCEAnalysisInputs:

    def test_rejects_mismatched_column_lengths(self):
        with pytest.raises(ValueError, match='one entry per strategy'):
            CEAnalysis(strategies=['A', 'B'], costs=[1.0], qalys=[1.0, 2.0])

    def test_rejects_psa_matrix_with_wrong_width(self):
        with pytest.raises(ValueError, match='n_strategies'):
            CEAnalysis(strategies=['A', 'B'], costs=[1.0, 2.0], qalys=[1.0, 2.0], psa_costs=np.zeros((10, 3)), psa_qalys=np.zeros((10, 3)))

    def test_accepts_consistent_inputs(self):
        cea = CEAnalysis(strategies=['A', 'B'], costs=[1.0, 2.0], qalys=[1.0, 2.0], psa_costs=np.zeros((10, 2)), psa_qalys=np.zeros((10, 2)))
        assert cea.n_sim == 10
