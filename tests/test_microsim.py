"""Tests for pyheor/microsim.py — MicroSimModel integration tests."""

import numpy as np
import pytest
from pyheor import MarkovModel, MicroSimModel, PatientProfile, C

ALIVE_FOREVER = [[1, 0], [0, 1]]
DEAD_AFTER_ONE_INTERVAL = [[0, 1], [0, 1]]


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

    def test_model_inputs_are_explicitly_validated(self):
        base = dict(states=["Alive", "Dead"], strategies=["S1"], n_cycles=5)

        with pytest.raises(ValueError, match="states"):
            MicroSimModel(states=[], strategies=["S1"], n_cycles=5)
        with pytest.raises(ValueError, match="unique"):
            MicroSimModel(states=["A", "A"], strategies=["S1"], n_cycles=5)
        with pytest.raises(ValueError, match="unique"):
            MicroSimModel(states=["Alive", "Dead"], strategies=["S", "S"], n_cycles=5)
        with pytest.raises(ValueError, match="n_cycles"):
            MicroSimModel(states=["Alive", "Dead"], strategies=["S1"], n_cycles=0)
        with pytest.raises(ValueError, match="cycle_length"):
            MicroSimModel(**base, cycle_length=0)
        with pytest.raises(ValueError, match="n_patients"):
            MicroSimModel(**base, n_patients=0)
        with pytest.raises(ValueError, match="initial_state"):
            MicroSimModel(**base, initial_state="Missing")
        with pytest.raises(ValueError, match="initial_state"):
            MicroSimModel(**base, initial_state=7)
        with pytest.raises(ValueError, match="state_type"):
            MicroSimModel(**base, state_type={"Unknown": "alive"})
        with pytest.raises(ValueError, match="alive.*dead"):
            MicroSimModel(**base, state_type={"Alive": "invalid"})
        with pytest.raises(ValueError, match="discount_convention"):
            MicroSimModel(**base, discount_convention="compound")


class TestMicroSimSetters:
    @pytest.fixture
    def model(self):
        return MicroSimModel(
            states=["Alive", "Dead"], strategies=["S1"], n_cycles=5,
        )

    def test_unknown_state_key_is_rejected(self, model):
        with pytest.raises(ValueError, match="unknown state or strategy"):
            model.set_state_cost("care", {"Typo": 100})
        with pytest.raises(ValueError, match="unknown state or strategy"):
            model.set_utility({"Typo": 0.5})

    def test_unknown_handler_state_is_rejected(self, model):
        with pytest.raises(ValueError, match="Unknown state"):
            model.on_state_enter("Typo", lambda i, t, a: None)
        with pytest.raises(ValueError, match="Unknown state"):
            model.on_state_exit("Typo", lambda i, t, a: None)

    def test_apply_cycles_must_be_interval_indices(self, model):
        with pytest.raises(ValueError, match="interval range"):
            model.set_state_cost("care", {"Alive": 100}, apply_cycles=[0, 5])

    def test_starting_costs_cannot_be_scheduled(self, model):
        with pytest.raises(ValueError, match="one-off charge"):
            model.set_state_cost(
                "care", {"Alive": 100}, method="starting", first_cycle_only=True
            )

    def test_unknown_cost_method_is_rejected(self, model):
        with pytest.raises(ValueError, match="method"):
            model.set_state_cost("care", {"Alive": 100}, method="lump")


class TestTransitionMatrices:
    def build(self, matrix):
        model = MicroSimModel(
            states=["Alive", "Dead"], strategies=["S1"],
            n_cycles=3, n_patients=5,
        )
        model.set_transitions("S1", matrix)
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        return model

    def test_rows_that_do_not_sum_to_one_are_rejected(self):
        model = self.build([[0.5, 0.2], [0, 1]])
        with pytest.raises(ValueError, match="Row sums"):
            model.run_base_case(seed=1, verbose=False)

    def test_negative_probabilities_are_not_clipped(self):
        model = self.build([[1.2, -0.2], [0, 1]])
        with pytest.raises(ValueError, match="Negative probabilities"):
            model.run_base_case(seed=1, verbose=False)

    def test_transition_shape_must_match_the_model_states(self):
        model = MicroSimModel(
            states=["Alive", "Sick", "Dead"], strategies=["S1"],
            n_cycles=1, n_patients=1,
        )
        model.set_transitions("S1", [[C, 0.1, 0.2, 0.7], [0, 1, 0, 0], [0, 0, 1, 0]])
        model.set_utility({"Alive": 1.0, "Sick": 0.5, "Dead": 0.0})

        with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
            model.run_base_case(seed=1, verbose=False)

    def test_callbacks_receive_zero_based_interval_indices(self):
        seen = []
        model = MicroSimModel(
            states=["Alive", "Dead"], strategies=["S1"],
            n_cycles=3, n_patients=2,
        )
        model.set_transitions(
            "S1", lambda p, t: seen.append(t) or ALIVE_FOREVER
        )
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        model.run_base_case(seed=1, verbose=False)

        assert seen == [0, 1, 2]


class TestAgreementWithCohortEngine:
    """Deterministic transitions must reproduce the Markov numbers exactly."""

    @pytest.mark.parametrize("matrix", [ALIVE_FOREVER, DEAD_AFTER_ONE_INTERVAL])
    @pytest.mark.parametrize("hcc", [False, True])
    def test_totals_match(self, matrix, hcc):
        shared = dict(
            states=["Alive", "Dead"], strategies=["S1"], n_cycles=4,
            cycle_length=0.5, dr_cost=0.03, dr_qaly=0.05,
            half_cycle_correction=hcc,
        )
        markov = MarkovModel(**shared)
        micro = MicroSimModel(**shared, n_patients=5)
        for model in (markov, micro):
            model.set_transitions("S1", matrix)
            model.set_state_cost("care", {"Alive": 1000, "Dead": 0})
            model.set_state_cost(
                "setup", {"Alive": 500, "Dead": 0}, method="starting"
            )
            model.set_utility({"Alive": 0.8, "Dead": 0.0})

        expected = markov.run_base_case().results["S1"]
        actual = micro.run_base_case(seed=1, verbose=False).results["S1"]

        assert float(np.mean(actual["total_lys"])) == pytest.approx(
            expected["total_lys"]
        )
        assert float(np.mean(actual["total_qalys"])) == pytest.approx(
            expected["total_qalys"]
        )
        assert float(np.mean(actual["total_cost"])) == pytest.approx(
            sum(expected["total_costs"].values())
        )

    @pytest.mark.parametrize("convention", ["discrete", "continuous"])
    def test_discount_conventions_match(self, convention):
        shared = dict(
            states=["Alive", "Dead"], strategies=["S1"], n_cycles=3,
            dr_cost=0.1, dr_qaly=0.1, discount_convention=convention,
            half_cycle_correction=False,
        )
        markov = MarkovModel(**shared)
        micro = MicroSimModel(**shared, n_patients=3)
        for model in (markov, micro):
            model.set_transitions("S1", ALIVE_FOREVER)
            model.set_state_cost("care", {"Alive": 100, "Dead": 0})
            model.set_utility({"Alive": 1.0, "Dead": 0.0})

        expected = markov.run_base_case().results["S1"]
        actual = micro.run_base_case(seed=1, verbose=False).results["S1"]

        assert float(np.mean(actual["total_cost"])) == pytest.approx(
            sum(expected["total_costs"].values())
        )
        assert float(np.mean(actual["total_qalys"])) == pytest.approx(
            expected["total_qalys"]
        )


class TestEventCosts:
    def test_entry_cost_is_discounted_at_the_interval_end(self):
        model = MicroSimModel(
            states=["Alive", "Dead"], strategies=["S1"],
            n_cycles=1, n_patients=4, dr_cost=0.1,
        )
        model.set_transitions("S1", DEAD_AFTER_ONE_INTERVAL)
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        model.on_state_enter("Dead", lambda i, t, a: {"cost": 100.0})

        result = model.run_base_case(seed=1, verbose=False)
        # The transition lands at the end of interval 0, i.e. time 1.
        assert float(np.mean(result.results["S1"]["total_cost"])) == pytest.approx(
            100.0 / 1.1
        )

    def test_handlers_receive_zero_based_interval_indices(self):
        seen = []
        model = MicroSimModel(
            states=["Alive", "Dead"], strategies=["S1"],
            n_cycles=2, n_patients=1,
        )
        model.set_transitions("S1", DEAD_AFTER_ONE_INTERVAL)
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        model.on_state_enter("Dead", lambda i, t, a: seen.append(t))

        model.run_base_case(seed=1, verbose=False)
        assert seen == [0]


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
        model.set_transitions("S1", DEAD_AFTER_ONE_INTERVAL)
        model.set_utility({"Alive": 1, "Dead": 0})
        result = model.run_base_case()
        # Half of the first interval is spent alive under the default
        # trapezoidal correction.
        assert result.summary()["Mean QALYs"].iloc[0] == pytest.approx(0.5)

    def test_results_summary_columns(self, micro_model):
        result = micro_model.run_base_case()
        summary = result.summary()
        assert "Strategy" in summary.columns
        assert "Mean Cost" in summary.columns

    def test_patient_outcomes_report_years_alive(self, micro_model):
        result = micro_model.run_base_case()
        outcomes = result.patient_outcomes
        assert "Years Alive" in outcomes.columns
        assert (outcomes["Years Alive"] <= micro_model.n_cycles).all()

    def test_owsa_runs_with_shared_patient_draws(self, micro_model):
        result = micro_model.run_owsa(
            params=["p_death"], n_patients=20, seed=3, verbose=False
        )
        summary = result.summary()

        assert list(summary["param_name"]) == ["p_death"]
        assert set(result.base_result) == {"SOC", "TRT"}

    def test_heterogeneous_population_runs(self):
        model = MicroSimModel(
            states=["Alive", "Dead"], strategies=["S1"],
            n_cycles=4, n_patients=20,
        )
        model.add_param("base_risk", base=0.05)
        model.set_transitions("S1", lambda p, t, attrs: [
            [C, p["base_risk"] * (1 + attrs["frailty"])],
            [0, 1],
        ])
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        profile = PatientProfile(
            n_patients=20,
            attributes={"frailty": np.linspace(0, 1, 20)},
        )

        result = model.run_base_case(profile=profile, seed=3, verbose=False)
        assert result.results["S1"]["state_history"].shape == (20, 5)
