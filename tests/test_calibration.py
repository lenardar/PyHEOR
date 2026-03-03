"""Tests for pyheor/calibration.py — model calibration module."""

import numpy as np
import pytest
from pyheor import (
    MarkovModel,
    C,
    calibrate,
    CalibrationTarget,
    CalibrationParam,
    CalibrationResult,
    latin_hypercube,
    gof_sse,
    gof_wsse,
    gof_loglik_normal,
)


# =========================================================================
# CalibrationTarget & CalibrationParam
# =========================================================================

class TestCalibrationTarget:
    def test_construction(self):
        t = CalibrationTarget(
            name="prev",
            observed=0.5,
            extract_fn=lambda sim: 1.0,
        )
        assert t.name == "prev"
        assert t.observed == 0.5
        assert t.se is None

    def test_extract_fn_called(self):
        t = CalibrationTarget(
            name="x",
            observed=1.0,
            extract_fn=lambda sim: sim["val"],
        )
        assert t.extract_fn({"val": 42}) == 42


class TestCalibrationParam:
    def test_construction(self):
        cp = CalibrationParam("p", lower=0.0, upper=1.0)
        assert cp.initial == 0.5

    def test_custom_initial(self):
        cp = CalibrationParam("p", lower=0.0, upper=1.0, initial=0.2)
        assert cp.initial == 0.2

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError, match="lower"):
            CalibrationParam("p", lower=1.0, upper=0.5)


# =========================================================================
# GoF Functions
# =========================================================================

class TestGoF:
    def test_sse_known(self):
        val = gof_sse([1.0, 2.0], [1.1, 1.8])
        expected = 0.01 + 0.04
        np.testing.assert_allclose(val, expected, rtol=1e-10)

    def test_sse_perfect(self):
        val = gof_sse([3.0, 4.0], [3.0, 4.0])
        assert val == 0.0

    def test_wsse_weights(self):
        # w = 1/se^2 = [1/0.01, 1/0.04] = [100, 25]
        val = gof_wsse([1.0, 2.0], [1.1, 1.8], se=[0.1, 0.2])
        expected = 100 * 0.01 + 25 * 0.04
        np.testing.assert_allclose(val, expected, rtol=1e-10)

    def test_wsse_requires_se(self):
        with pytest.raises(ValueError, match="standard errors"):
            gof_wsse([1.0], [1.0])

    def test_loglik_positive(self):
        # Negative log-likelihood is always positive (for non-degenerate data)
        val = gof_loglik_normal([1.0], [1.5], se=[0.5])
        assert val > 0

    def test_loglik_requires_se(self):
        with pytest.raises(ValueError, match="standard errors"):
            gof_loglik_normal([1.0], [1.0])

    def test_loglik_minimum_at_match(self):
        # NLL should be smaller when predicted = observed
        nll_match = gof_loglik_normal([1.0], [1.0], se=[0.5])
        nll_off = gof_loglik_normal([1.0], [2.0], se=[0.5])
        assert nll_match < nll_off


# =========================================================================
# Latin Hypercube Sampling
# =========================================================================

class TestLHS:
    def test_shape(self):
        samples = latin_hypercube(50, [(0, 1), (10, 20)], seed=42)
        assert samples.shape == (50, 2)

    def test_within_bounds(self):
        bounds = [(0.1, 0.9), (100, 200)]
        samples = latin_hypercube(100, bounds, seed=42)
        assert np.all(samples[:, 0] >= 0.1)
        assert np.all(samples[:, 0] <= 0.9)
        assert np.all(samples[:, 1] >= 100)
        assert np.all(samples[:, 1] <= 200)

    def test_coverage(self):
        """LHS should cover the space reasonably well."""
        samples = latin_hypercube(100, [(0, 1)], seed=42)
        # Check that all 10 deciles are represented
        for i in range(10):
            lo, hi = i * 0.1, (i + 1) * 0.1
            assert np.any((samples[:, 0] >= lo) & (samples[:, 0] < hi))


# =========================================================================
# Calibration (Nelder-Mead)
# =========================================================================

class TestNelderMead:
    @pytest.fixture
    def calibration_setup(self):
        """Create a simple model with known parameter to recover."""
        model = MarkovModel(
            states=["Alive", "Dead"],
            strategies=["S1"],
            n_cycles=10,
            half_cycle_correction=False,
        )
        # True parameter: p_death = 0.10
        model.add_param("p_death", base=0.05)  # intentionally wrong
        model.set_transitions(
            "S1",
            lambda p, t: [[1 - p["p_death"], p["p_death"]], [0, 1]],
        )
        model.set_utility({"Alive": 1.0, "Dead": 0.0})

        # Generate "observed" data from true parameter
        true_p = 0.10
        # Alive at cycle 5: (1-0.10)^5 = 0.9^5 = 0.59049
        # Alive at cycle 10: (1-0.10)^10 = 0.9^10 = 0.34868
        targets = [
            CalibrationTarget(
                name="alive_5yr",
                observed=(1 - true_p) ** 5,
                extract_fn=lambda sim: sim["S1"]["trace"][5, 0],
            ),
            CalibrationTarget(
                name="alive_10yr",
                observed=(1 - true_p) ** 10,
                extract_fn=lambda sim: sim["S1"]["trace"][10, 0],
            ),
        ]

        calib_params = [
            CalibrationParam("p_death", lower=0.01, upper=0.50),
        ]

        return model, targets, calib_params, true_p

    def test_recovers_parameter(self, calibration_setup):
        model, targets, calib_params, true_p = calibration_setup
        result = calibrate(
            model,
            targets,
            calib_params,
            gof="sse",
            method="nelder_mead",
            n_restarts=3,
            seed=42,
            progress=False,
        )
        assert isinstance(result, CalibrationResult)
        np.testing.assert_allclose(
            result.best_params["p_death"],
            true_p,
            rtol=0.05,
        )

    def test_gof_near_zero(self, calibration_setup):
        model, targets, calib_params, _ = calibration_setup
        result = calibrate(
            model,
            targets,
            calib_params,
            gof="sse",
            method="nelder_mead",
            n_restarts=3,
            seed=42,
            progress=False,
        )
        assert result.best_gof < 1e-6

    def test_result_summary(self, calibration_setup):
        model, targets, calib_params, _ = calibration_setup
        result = calibrate(
            model,
            targets,
            calib_params,
            method="nelder_mead",
            n_restarts=2,
            seed=42,
            progress=False,
        )
        df = result.summary()
        assert "Parameter" in df.columns
        assert "Best Value" in df.columns
        assert len(df) == 1

    def test_target_comparison(self, calibration_setup):
        model, targets, calib_params, _ = calibration_setup
        result = calibrate(
            model,
            targets,
            calib_params,
            method="nelder_mead",
            n_restarts=2,
            seed=42,
            progress=False,
        )
        tc = result.target_comparison()
        assert "Target" in tc.columns
        assert "Predicted" in tc.columns
        assert len(tc) == 2

    def test_apply_to_model(self, calibration_setup):
        model, targets, calib_params, true_p = calibration_setup
        result = calibrate(
            model,
            targets,
            calib_params,
            method="nelder_mead",
            n_restarts=3,
            seed=42,
            progress=False,
        )
        result.apply_to_model(model)
        np.testing.assert_allclose(
            model.params["p_death"].base,
            true_p,
            rtol=0.05,
        )


# =========================================================================
# Calibration (Random Search)
# =========================================================================

class TestRandomSearch:
    def test_recovers_parameter(self):
        model = MarkovModel(
            states=["Alive", "Dead"],
            strategies=["S1"],
            n_cycles=10,
            half_cycle_correction=False,
        )
        true_p = 0.10
        model.add_param("p_death", base=0.05)
        model.set_transitions(
            "S1",
            lambda p, t: [[1 - p["p_death"], p["p_death"]], [0, 1]],
        )
        model.set_utility({"Alive": 1.0, "Dead": 0.0})

        targets = [
            CalibrationTarget(
                name="alive_5yr",
                observed=(1 - true_p) ** 5,
                extract_fn=lambda sim: sim["S1"]["trace"][5, 0],
            ),
        ]
        calib_params = [
            CalibrationParam("p_death", lower=0.01, upper=0.30),
        ]

        result = calibrate(
            model,
            targets,
            calib_params,
            method="random_search",
            n_samples=500,
            seed=42,
            progress=False,
        )
        # Random search is less precise, allow 20% tolerance
        np.testing.assert_allclose(
            result.best_params["p_death"],
            true_p,
            rtol=0.20,
        )
        assert result.method == "random_search"
        assert result.all_params.shape == (500, 1)


# =========================================================================
# Validation
# =========================================================================

class TestValidation:
    def test_unknown_gof_raises(self, simple_markov_model):
        with pytest.raises(ValueError, match="Unknown GoF"):
            calibrate(
                simple_markov_model,
                targets=[],
                calib_params=[],
                gof="bad_gof",
                progress=False,
            )

    def test_unknown_method_raises(self, simple_markov_model):
        with pytest.raises(ValueError, match="Unknown method"):
            calibrate(
                simple_markov_model,
                targets=[],
                calib_params=[],
                method="bad_method",
                progress=False,
            )

    def test_missing_param_raises(self, simple_markov_model):
        with pytest.raises(ValueError, match="not found"):
            calibrate(
                simple_markov_model,
                targets=[
                    CalibrationTarget(
                        "x",
                        1.0,
                        extract_fn=lambda s: 0,
                    ),
                ],
                calib_params=[
                    CalibrationParam("nonexistent", 0, 1),
                ],
                progress=False,
            )
