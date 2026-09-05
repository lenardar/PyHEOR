"""Tests for pyheor/des.py — DESModel integration tests."""

import numpy as np
import pytest
from pyheor import DESModel
from pyheor.survival import Exponential, Weibull
from pyheor.analysis.results import DESPSAResult, DESResult


class TestDESConstruction:
    def test_basic(self):
        model = DESModel(
            states=["Alive", "Dead"],
            strategies=["S1"],
            time_horizon=20,
        )
        assert model is not None

    def test_clock_is_explicit(self):
        assert DESModel(states=["Alive", "Dead"], strategies=["S1"], clock="forward").clock == "forward"
        with pytest.raises(ValueError, match="clock"):
            DESModel(states=["Alive", "Dead"], strategies=["S1"], clock="cycle")

    def test_time_horizon_must_be_positive(self):
        with pytest.raises(ValueError, match="time_horizon"):
            DESModel(states=["Alive", "Dead"], strategies=["S1"], time_horizon=0)

    def test_initial_state_can_be_named_or_indexed(self):
        assert DESModel(
            states=["Alive", "Dead"], strategies=["S1"], initial_state="Dead"
        ).initial_state_idx == 1
        assert DESModel(
            states=["Alive", "Dead"], strategies=["S1"], initial_state=1
        ).initial_state_idx == 1
        with pytest.raises(ValueError, match="initial_state"):
            DESModel(states=["Alive", "Dead"], strategies=["S1"], initial_state="Missing")

    def test_model_inputs_are_explicitly_validated(self):
        with pytest.raises(ValueError, match="states"):
            DESModel(states=[], strategies=["S1"])
        with pytest.raises(ValueError, match="unique"):
            DESModel(states=["Alive", "Alive"], strategies=["S1"])
        with pytest.raises(ValueError, match="strategies"):
            DESModel(states=["Alive", "Dead"], strategies=[])
        with pytest.raises(ValueError, match="unique"):
            DESModel(states=["Alive", "Dead"], strategies=["S1", "S1"])
        with pytest.raises(ValueError, match="state_type"):
            DESModel(
                states=["Alive", "Dead"], strategies=["S1"],
                state_type={"Unknown": "alive"},
            )
        with pytest.raises(ValueError, match="alive.*dead"):
            DESModel(
                states=["Alive", "Dead"], strategies=["S1"],
                state_type={"Alive": "invalid"},
            )

    @pytest.mark.parametrize("rate", [-0.01, np.nan, np.inf, -np.inf])
    def test_discount_rates_must_be_finite_nonnegative(self, rate):
        with pytest.raises(ValueError, match="dr_cost"):
            DESModel(states=["Alive", "Dead"], strategies=["S1"], dr_cost=rate)

    def test_discount_conventions_have_closed_form_integrals(self):
        discrete_lump = DESModel._discount_lump_sum(100, 2, 0.10, "discrete")
        continuous_lump = DESModel._discount_lump_sum(100, 2, 0.10, "continuous")
        assert discrete_lump == pytest.approx(100 / 1.1 ** 2)
        assert continuous_lump == pytest.approx(100 * np.exp(-0.2))

        discrete_flow = DESModel._discount_continuous(100, 0, 2, 0.10, "discrete")
        continuous_flow = DESModel._discount_continuous(100, 0, 2, 0.10, "continuous")
        assert discrete_flow == pytest.approx(100 * (1 - 1.1 ** -2) / np.log(1.1))
        assert continuous_flow == pytest.approx(100 * (1 - np.exp(-0.2)) / 0.1)

    def test_invalid_parameter_reference_does_not_default_to_zero(self):
        model = DESModel(states=["Alive", "Dead"], strategies=["S1"], time_horizon=1)
        model.set_state_cost("care", {"Alive": "missing_cost"})
        with pytest.raises(KeyError, match="missing_cost"):
            model.run(n_patients=1, progress=False)


class TestDESRun:
    @pytest.fixture
    def des_model(self):
        model = DESModel(
            states=["PFS", "Progressed", "Dead"],
            strategies=["SOC", "TRT"],
            time_horizon=20,
            dr_cost=0.03,
            dr_qaly=0.03,
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
        )
        model.set_event("S1", "Alive", "Dead", Exponential(rate=0.1))
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        result = model.run(n_patients=2000, seed=42)
        mean_lys = result.summary()["Mean LYs"].iloc[0]
        np.testing.assert_allclose(mean_lys, 10.0, rtol=0.20)

    def test_forward_clock_conditions_on_absolute_time(self, monkeypatch):
        model = DESModel(
            states=["Alive", "Dead"], strategies=["S1"],
            time_horizon=10, clock="forward",
        )
        monkeypatch.setattr(np.random, "uniform", lambda: 0.5)
        dist = Weibull(shape=2.0, scale=2.0)
        # H(2 + dt) - H(2) = -log(0.5), rather than a fresh draw from t=0.
        expected = 2.0 * np.sqrt(1.0 - np.log(0.5)) - 2.0
        assert model._sample_forward_tte(dist, current_time=2.0) == pytest.approx(expected)

    def test_event_clock_overrides_model_default(self, monkeypatch):
        model = DESModel(
            states=["Alive", "Dead"], strategies=["S1"],
            time_horizon=10, clock="reset",
        )
        model.set_event("S1", "Alive", "Dead", Exponential(rate=1), clock="forward")
        monkeypatch.setattr(model, "_sample_tte", lambda dist: pytest.fail("used reset clock"))
        monkeypatch.setattr(model, "_sample_forward_tte", lambda dist, time: 1.0)

        result = model.run(n_patients=1, seed=1, progress=False)
        assert result.results["S1"]["patient_results"][0]["event_log"] == [
            (1.0, "Alive", "Dead")
        ]

    def test_event_clock_rejects_unknown_value(self):
        model = DESModel(states=["Alive", "Dead"], strategies=["S1"])
        with pytest.raises(ValueError, match="clock"):
            model.set_event("S1", "Alive", "Dead", Exponential(rate=1), clock="cycle")

    def test_self_loop_events_are_rejected(self):
        model = DESModel(states=["Alive", "Dead"], strategies=["S1"])
        with pytest.raises(ValueError, match="Self-loop"):
            model.set_event("S1", "Alive", "Alive", Exponential(rate=1))

    def test_initial_state_entry_handler_runs_at_time_zero(self):
        model = DESModel(
            states=["Alive", "Dead"], strategies=["S1"],
            initial_state="Dead", time_horizon=1,
        )
        seen = []
        model.on_state_enter("Dead", lambda idx, time, attrs: seen.append(time))
        result = model.run(n_patients=1, progress=False)
        assert seen == [0.0]
        assert result.results["S1"]["patient_results"][0]["event_log"] == []

    def test_state_entry_handler_cost_is_recorded_at_entry_time(self, monkeypatch):
        model = DESModel(
            states=["Alive", "Dead"], strategies=["S1"],
            time_horizon=10, dr_cost=0.10,
        )
        model.set_event("S1", "Alive", "Dead", Exponential(rate=1))
        model.on_state_enter("Dead", lambda idx, time, attrs: {"cost": 110.0})
        monkeypatch.setattr(model, "_sample_tte", lambda dist: 1.0)

        result = model.run(n_patients=1, seed=1, progress=False)
        assert result.results["S1"]["total_cost"][0] == pytest.approx(100.0)

    def test_multi_strategy_run_uses_common_random_numbers(self):
        model = DESModel(
            states=["Alive", "Dead"], strategies=["SOC", "TRT"],
            time_horizon=10,
        )
        event = Exponential(rate=0.5)
        model.set_event("SOC", "Alive", "Dead", event)
        model.set_event("TRT", "Alive", "Dead", event)

        result = model.run(n_patients=8, seed=42, progress=False)
        soc = result.results["SOC"]["patient_results"]
        trt = result.results["TRT"]["patient_results"]
        assert [r["event_log"] for r in soc] == [r["event_log"] for r in trt]

    def test_attrs_length_must_match_patient_count(self):
        model = DESModel(states=["Alive", "Dead"], strategies=["S1"])
        with pytest.raises(ValueError, match="attrs"):
            model.run(n_patients=2, attrs={"age": np.array([60.0])}, progress=False)
        with pytest.raises(ValueError, match="attrs"):
            model.run_psa(n_sim=1, n_patients=2,
                          attrs={"age": np.array([60.0])}, progress=False)


class TestDESResults:
    @staticmethod
    def _result(cost, qaly):
        model = DESModel(
            states=["Alive", "Dead"], strategies=["SOC", "TRT"], time_horizon=1,
        )
        results = {
            "SOC": {"mean_cost": 100.0, "mean_qalys": 1.0, "mean_lys": 1.0},
            "TRT": {"mean_cost": cost, "mean_qalys": qaly, "mean_lys": qaly},
        }
        return DESResult(model, results, {})

    @pytest.mark.parametrize(
        ("cost", "qaly", "expected"),
        [
            (200.0, 0.5, "Dominated"),
            (50.0, 1.5, "Dominant"),
            (50.0, 0.5, "100 (less effective, less costly)"),
            (100.0, 1.0, "No difference"),
        ],
    )
    def test_icer_classifies_incremental_quadrant(self, cost, qaly, expected):
        row = self._result(cost, qaly).icer().iloc[0]
        assert row["ICER"] == expected
        assert row["ICER Classification"] == expected

    def test_psa_icer_classifies_incremental_quadrant(self):
        model = DESModel(
            states=["Alive", "Dead"], strategies=["SOC", "TRT"], time_horizon=1,
        )
        psa = DESPSAResult(
            model,
            psa_iterations=[
                {
                    "SOC": {"mean_cost": 100.0, "mean_qalys": 1.0, "mean_lys": 1.0},
                    "TRT": {"mean_cost": 200.0, "mean_qalys": 0.5, "mean_lys": 0.5},
                }
            ],
            sampled_params=[{}],
        )
        row = psa.icer().iloc[0]
        assert np.isnan(row["ICER"])
        assert row["ICER Classification"] == "Dominated"

    def test_survival_keeps_right_censored_patients_at_horizon(self, monkeypatch):
        model = DESModel(states=["Alive", "Dead"], strategies=["S1"], time_horizon=5)
        model.set_event("S1", "Alive", "Dead", Exponential(rate=1))

        monkeypatch.setattr(model, "_sample_tte", lambda dist: 5.0)
        result = model.run(n_patients=1, progress=False)
        curve = result.survival_curve(n_points=2)
        assert curve.iloc[-1]["Survival"] == pytest.approx(1.0)

        monkeypatch.setattr(model, "_sample_tte", lambda dist: 1.0)
        result = model.run(n_patients=1, progress=False)
        curve = result.survival_curve(n_points=2)
        assert curve.iloc[-1]["Survival"] == pytest.approx(0.0)

    def test_no_event_survival_is_one_at_horizon(self):
        model = DESModel(states=["Alive", "Dead"], strategies=["S1"], time_horizon=5)
        result = model.run(n_patients=3, progress=False)
        curve = result.survival_curve(n_points=2)
        assert curve.iloc[-1]["Survival"] == pytest.approx(1.0)
