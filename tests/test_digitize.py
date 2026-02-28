"""Tests for pyheor/digitize.py — clean_digitized_km, guyot_reconstruct."""

import numpy as np
import pytest
from pyheor.digitize import clean_digitized_km, guyot_reconstruct


# =========================================================================
# clean_digitized_km
# =========================================================================

class TestCleanDigitizedKM:
    def test_prepends_zero_one(self):
        t, s = clean_digitized_km([2, 5, 10], [0.9, 0.7, 0.4])
        assert t[0] == 0.0
        assert s[0] == 1.0

    def test_fixes_start_survival(self):
        t, s = clean_digitized_km([0, 5, 10], [0.95, 0.7, 0.4])
        assert s[0] == 1.0

    def test_enforces_monotone(self):
        # Non-monotone input: 0.8 then 0.85 (increase!)
        t, s = clean_digitized_km(
            [0, 2, 4, 6, 8],
            [1.0, 0.8, 0.85, 0.6, 0.4],
        )
        diffs = np.diff(s)
        assert np.all(diffs <= 1e-10), "Output must be non-increasing"

    def test_removes_out_of_bounds(self):
        t, s = clean_digitized_km(
            [-1, 0, 5, 10],
            [1.5, 1.0, 0.7, 0.4],
        )
        assert np.all(t >= 0)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            clean_digitized_km([0, 1, 2], [1.0, 0.8])


# =========================================================================
# guyot_reconstruct
# =========================================================================

class TestGuyotReconstruct:
    def test_returns_arrays(self):
        t = [0, 3, 6, 9, 12]
        s = [1.0, 0.85, 0.70, 0.55, 0.40]
        t_risk = [0, 6, 12]
        n_risk = [100, 70, 40]
        ipd_time, ipd_event = guyot_reconstruct(t, s, t_risk, n_risk)
        assert isinstance(ipd_time, np.ndarray)
        assert isinstance(ipd_event, np.ndarray)

    def test_event_indicator_binary(self):
        t = [0, 3, 6, 9, 12]
        s = [1.0, 0.85, 0.70, 0.55, 0.40]
        t_risk = [0, 6, 12]
        n_risk = [100, 70, 40]
        _, ipd_event = guyot_reconstruct(t, s, t_risk, n_risk)
        assert set(np.unique(ipd_event)).issubset({0, 1})

    def test_total_n_close_to_nrisk0(self):
        t = [0, 3, 6, 9, 12]
        s = [1.0, 0.85, 0.70, 0.55, 0.40]
        t_risk = [0, 6, 12]
        n_risk = [100, 70, 40]
        ipd_time, ipd_event = guyot_reconstruct(t, s, t_risk, n_risk)
        assert abs(len(ipd_time) - 100) < 15

    def test_roundtrip(self):
        """Generate IPD from known Weibull, compute KM, reconstruct, refit."""
        from pyheor.fitting import SurvivalFitter, kaplan_meier

        rng = np.random.default_rng(42)
        n = 200
        true_shape, true_scale = 1.5, 10.0
        t_event = true_scale * rng.weibull(true_shape, size=n)
        t_censor = rng.uniform(0, 30, size=n)
        time = np.minimum(t_event, t_censor)
        event = (t_event <= t_censor).astype(int)

        # Compute KM and extract nrisk table
        km = kaplan_meier(time, event)
        km_times = km["time"].values
        km_surv = km["survival"].values

        # Create nrisk table at evenly-spaced points
        t_risk_pts = np.array([0, 5, 10, 15, 20, 25])
        n_risk_vals = []
        for tr in t_risk_pts:
            n_risk_vals.append(int(km.loc[km["time"] <= tr].iloc[-1]["n_risk"]))
        n_risk_vals = np.array(n_risk_vals)

        # Reconstruct
        ipd_t, ipd_e = guyot_reconstruct(
            km_times, km_surv, t_risk_pts, n_risk_vals,
            tot_events=int(event.sum()),
        )

        assert len(ipd_t) > 0
        assert ipd_e.sum() > 0  # some events reconstructed

        # Refit and check parameter recovery
        fitter = SurvivalFitter(ipd_t, ipd_e, label="Reconstructed")
        fitter.fit(verbose=False)
        result = fitter.get_result("Weibull")
        fitted_shape = result.params.get(
            "shape", result.params.get("k", None)
        )
        fitted_scale = result.params.get(
            "scale", result.params.get("lambda", None)
        )
        if fitted_shape is not None:
            assert abs(fitted_shape - true_shape) / true_shape < 0.15
        if fitted_scale is not None:
            assert abs(fitted_scale - true_scale) / true_scale < 0.15
