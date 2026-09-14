"""Reproducibility guarantees for seeded runs.

A public run method that accepts ``seed`` must control every random source it
uses, and must not disturb the caller's global numpy RNG.
"""

import numpy as np
import pytest

from pyheor import (
    C,
    Beta,
    DESModel,
    Gamma,
    MarkovModel,
    MicroSimModel,
    PSMModel,
)
from pyheor.distributions import sample_distribution
from pyheor.distributions import sample_distribution
from pyheor.survival import Exponential, Weibull


class KwargsSwallowingDistribution:
    """A third-party distribution whose ``sample`` absorbs unknown keywords."""

    def sample(self, n=1, **kwargs):
        return np.random.norm# =============================================================================
# Model factories
# =============================================================================

def markov_model():
    model = MarkovModel(
        states=["Alive", "Dead"],
        strategies=["SOC", "TRT"],
        n_cycles=5,
    )
    model.add_param("p_death", base=0.1, dist=Beta(mean=0.1, sd=0.02))
    model.add_param("hr", base=0.8, dist=Gamma(mean=0.8, sd=0.1))
    model.set_transitions("SOC", lambda p, t: [[C, p["p_death"]], [0, 1]])
    model.set_transitions(
        "TRT", lambda p, t: [[C, p["p_death"] * p["hr"]], [0, 1]]
    )
    model.set_state_cost("care", {"Alive": 1000, "Dead": 0})
    model.set_utility({"Alive": 1.0, "Dead": 0.0})
    return model


def psm_model():
    model = PSMModel(
        states=["PFS", "Progressed", "Dead"],
        survival_endpoints=["PFS", "OS"],
        strategies=["SOC", "TRT"],
        n_cycles=5,
    )
    model.add_param("c_drug", base=1000, dist=Gamma(mean=1000, sd=100))
    for strategy in ("SOC", "TRT"):
        model.set_survival_all(strategy, {
            "PFS": Weibull(shape=1.0, scale=5.0),
            "OS": Weibull(shape=1.0, scale=10.0),
        })
    model.set_state_cost(
        "drug", {"PFS": "c_drug", "Progressed": 500, "Dead": 0}
    )
    model.set_utility({"PFS": 0.8, "Progressed": 0.5, "Dead": 0.0})
    return model


def microsim_model():
    model = MicroSimModel(
        states=["Alive", "Dead"],
        strategies=["SOC", "TRT"],
        n_cycles=5,
        n_patients=40,
    )
    model.add_param("p_death", base=0.1, dist=Beta(mean=0.1, sd=0.02))
    model.add_param("hr", base=0.8, dist=Gamma(mean=0.8, sd=0.1))
    model.set_transitions("SOC", lambda p, t: [[C, p["p_death"]], [0, 1]])
    model.set_transitions(
        "TRT", lambda p, t: [[C, p["p_death"] * p["hr"]], [0, 1]]
    )
    model.set_state_cost("care", {"Alive": 1000, "Dead": 0})
    model.set_utility({"Alive": 1.0, "Dead": 0.0})
    return model


def des_model():
    model = DESModel(
        states=["Alive", "Dead"],
        strategies=["SOC", "TRT"],
        time_horizon=10.0,
    )
    model.add_param("rate", base=0.1, dist=Gamma(mean=0.1, sd=0.02))
    model.set_event("SOC", "Alive", "Dead", Exponential(rate=0.1))
    model.set_event(
        "TRT", "Alive", "Dead", lambda p: Exponential(rate=p["rate"])
    )
    model.set_state_cost("care", {"Alive": 1000})
    model.set_utility({"Alive": 1.0})
    return model


# =============================================================================
# Helpers
# =============================================================================

def params_matrix(sampled_params):
    """Flatten a list of parameter dicts into a comparable array."""
    keys = sorted(sampled_params[0])
    return np.array([[draw[key] for key in keys] for draw in sampled_params])


def global_rng_state():
    """Return the comparable parts of the global numpy RNG state."""
    state = np.random.get_state()
    return state[1].copy(), state[2], state[3], state[4]


def assert_same_global_rng_state(before, after):
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1:] == after[1:]


def run_psa(model, seed):
    """Call run_psa with the small-sample keyword each engine expects."""
    if isinstance(model, MicroSimModel):
        return model.run_psa(n_outer=4, seed=seed, verbose=False)
    if isinstance(model, DESModel):
        return model.run_psa(n_sim=4, n_patients=20, seed=seed, progress=False)
    return model.run_psa(n_sim=4, seed=seed, progress=False)


# =============================================================================
# Same seed reproduces the same draws
# =============================================================================

class TestSeededPsaIsReproducible:
    @pytest.mark.parametrize("factory", [markov_model, psm_model])
    def test_cohort_engines(self, factory):
        first = run_psa(factory(), seed=7)
        second = run_psa(factory(), seed=7)
        np.testing.assert_array_equal(
            params_matrix(first.sampled_params),
            params_matrix(second.sampled_params),
        )

    def test_des(self):
        first = run_psa(des_model(), seed=7)
        second = run_psa(des_model(), seed=7)
        np.testing.assert_array_equal(
            params_matrix(first.sampled_params),
            params_matrix(second.sampled_params),
        )

    def test_microsim(self):
        first = run_psa(microsim_model(), seed=7)
        second = run_psa(microsim_model(), seed=7)
        np.testing.assert_array_equal(
            params_matrix(first.sampled_params),
            params_matrix(second.sampled_params),
        )


class TestSeededRunIsReproducible:
    def test_microsim_base_case(self):
        first = microsim_model().run_base_case(seed=11, verbose=False)
        second = microsim_model().run_base_case(seed=11, verbose=False)
        for strategy in ("SOC", "TRT"):
            np.testing.assert_array_equal(
                first.results[strategy]["total_cost"],
                second.results[strategy]["total_cost"],
            )
            np.testing.assert_array_equal(
                first.results[strategy]["total_qalys"],
                second.results[strategy]["total_qalys"],
            )

    def test_des_run(self):
        first = des_model().run(n_patients=20, seed=11, progress=False)
        second = des_model().run(n_patients=20, seed=11, progress=False)
        for strategy in ("SOC", "TRT"):
            np.testing.assert_array_equal(
                first.results[strategy]["total_cost"],
                second.results[strategy]["total_cost"],
            )
            np.testing.assert_array_equal(
                first.results[strategy]["total_qalys"],
                second.results[strategy]["total_qalys"],
            )


# =============================================================================
# Common random numbers across strategies
# =============================================================================

def identical_strategy_model(engine):
    """Two strategies with identical inputs, so any difference is noise."""
    if engine == "microsim":
        model = MicroSimModel(
            states=["Alive", "Dead"],
            strategies=["A", "B"],
            n_cycles=10,
            n_patients=200,
        )
        model.add_param("p_death", base=0.1)
        for strategy in ("A", "B"):
            model.set_transitions(
                strategy, lambda p, t: [[C, p["p_death"]], [0, 1]]
            )
        model.set_utility({"Alive": 1.0, "Dead": 0.0})
        return model

    model = DESModel(
        states=["Alive", "Dead"],
        strategies=["A", "B"],
        time_horizon=10.0,
    )
    for strategy in ("A", "B"):
        model.set_event(strategy, "Alive", "Dead", Exponential(rate=0.1))
    model.set_utility({"Alive": 1.0})
    return model


class TestCommonRandomNumbers:
    def test_des_aligns_patients_across_strategies(self):
        result = identical_strategy_model("des").run(
            n_patients=200, seed=5, progress=False
        )
        np.testing.assert_array_equal(
            result.results["A"]["total_qalys"],
            result.results["B"]["total_qalys"],
        )

    @pytest.mark.xfail(
        strict=True,
        reason="microsim.py:831-835 reuses one rng sequentially across "
               "strategies instead of per-patient common random numbers",
    )
    def test_microsim_aligns_patients_across_strategies(self):
        result = identical_strategy_model("microsim").run_base_case(
            seed=5, verbose=False
        )
        np.testing.assert_array_equal(
            result.results["A"]["total_qalys"],
            result.results["B"]["total_qalys"],
        )


# =============================================================================
# The sampling shim
# =============================================================================

class KwargsSwallowingDistribution:
    """A third-party distribution whose sample absorbs unknown keywords."""

    def sample(self, n=1, **kwargs):
        return np.random.normal(size=n)


class TestSampleDistributionShim:
    """A **kwargs signature must not be mistaken for rng support."""

    def test_draws_are_reproducible(self):
        distribution = KwargsSwallowingDistribution()
        first = sample_distribution(distribution, 3, np.random.default_rng(5))
        second = sample_distribution(distribution, 3, np.random.default_rng(5))
        np.testing.assert_array_equal(first, second)

    def test_caller_global_state_is_restored(self):
        np.random.seed(0)
        before = global_rng_state()
        sample_distribution(
            KwargsSwallowingDistribution(), 3, np.random.default_rng(5)
        )
        assert_same_global_rng_state(before, global_rng_state())


# =============================================================================
# Seeded runs must not disturb the caller's global RNG
# =============================================================================

class TestGlobalRngIsNotDisturbed:
    @pytest.mark.parametrize("factory", [markov_model, psm_model])
    def test_cohort_psa(self, factory):
        np.random.seed(0)
        before = global_rng_state()
        run_psa(factory(), seed=3)
        assert_same_global_rng_state(before, global_rng_state())

    def test_microsim_base_case(self):
        np.random.seed(0)
        before = global_rng_state()
        microsim_model().run_base_case(seed=3, verbose=False)
        assert_same_global_rng_state(before, global_rng_state())

    def test_microsim_psa(self):
        np.random.seed(0)
        before = global_rng_state()
        run_psa(microsim_model(), seed=3)
        assert_same_global_rng_state(before, global_rng_state())

    def test_des_run(self):
        np.random.seed(0)
        before = global_rng_state()
        des_model().run(n_patients=20, seed=3, progress=False)
        assert_same_global_rng_state(before, global_rng_state())

    def test_des_psa(self):
        np.random.seed(0)
        before = global_rng_state()
        run_psa(des_model(), seed=3)
        assert_same_global_rng_state(before, global_rng_state())
