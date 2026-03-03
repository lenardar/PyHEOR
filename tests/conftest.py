"""Shared fixtures for PyHEOR test suite."""

import sys
import os
import pytest
import numpy as np

# Ensure pyheor is importable without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def seed_rng():
    """Seed numpy RNG before every test for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def simple_markov_model():
    """A minimal 3-state, 2-strategy Markov model ready for run_base_case()."""
    from pyheor import MarkovModel, C, Beta, Gamma

    model = MarkovModel(
        states=["Healthy", "Sick", "Dead"],
        strategies=["SOC", "New"],
        n_cycles=10,
        cycle_length=1.0,
        dr_cost=0.03,
        dr_qaly=0.03,
        half_cycle_correction=True,
    )
    model.add_param("p_HS", base=0.15, dist=Beta(mean=0.15, sd=0.03))
    model.add_param("p_HD", base=0.02)
    model.add_param("p_SD", base=0.10, dist=Beta(mean=0.10, sd=0.02))
    model.add_param("hr", base=0.7, dist=Gamma(mean=0.7, sd=0.1))

    model.set_transitions("SOC", lambda p, t: [
        [C, p["p_HS"], p["p_HD"]],
        [0, C,         p["p_SD"]],
        [0, 0,         1],
    ])
    model.set_transitions("New", lambda p, t: [
        [C, p["p_HS"] * p["hr"], p["p_HD"]],
        [0, C,                    p["p_SD"]],
        [0, 0,                    1],
    ])

    model.set_state_cost("drug", {
        "SOC": {"Healthy": 2000, "Sick": 2000, "Dead": 0},
        "New": {"Healthy": 8000, "Sick": 5000, "Dead": 0},
    })
    model.set_state_cost("medical", {"Healthy": 500, "Sick": 3000, "Dead": 0})
    model.set_utility({"Healthy": 0.95, "Sick": 0.60, "Dead": 0.0})

    return model


@pytest.fixture
def simple_psm_model():
    """A minimal 3-state PSM with Weibull curves."""
    from pyheor import PSMModel
    from pyheor.survival import Weibull, ProportionalHazards

    model = PSMModel(
        states=["PFS", "Progressed", "Dead"],
        survival_endpoints=["PFS", "OS"],
        strategies=["SOC", "TRT"],
        n_cycles=20,
        cycle_length=1.0,
        dr_cost=0.03,
        dr_qaly=0.03,
    )
    model.set_survival_all("SOC", {
        "PFS": Weibull(shape=1.0, scale=5.0),
        "OS": Weibull(shape=1.0, scale=10.0),
    })
    model.set_survival_all("TRT", {
        "PFS": ProportionalHazards(Weibull(shape=1.0, scale=5.0), hr=0.8),
        "OS": ProportionalHazards(Weibull(shape=1.0, scale=10.0), hr=0.7),
    })

    model.set_state_cost("drug", {
        "SOC": {"PFS": 5000, "Progressed": 2000, "Dead": 0},
        "TRT": {"PFS": 12000, "Progressed": 2000, "Dead": 0},
    })
    model.set_utility({"PFS": 0.85, "Progressed": 0.5, "Dead": 0.0})

    return model


@pytest.fixture
def weibull_ipd_data():
    """Synthetic IPD from known Weibull(shape=1.5, scale=10) with ~20% censoring."""
    rng = np.random.default_rng(123)
    n = 200
    true_shape, true_scale = 1.5, 10.0
    t_event = true_scale * rng.weibull(true_shape, size=n)
    t_censor = rng.uniform(0, 25, size=n)
    time = np.minimum(t_event, t_censor)
    event = (t_event <= t_censor).astype(int)
    return time, event, true_shape, true_scale
