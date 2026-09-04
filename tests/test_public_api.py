"""Tests for canonical model names and their concise public aliases."""

import pyheor as ph


def test_model_aliases_reference_canonical_classes():
    assert ph.MarkovModel is ph.CohortStateTransitionModel
    assert ph.PSMModel is ph.PartitionedSurvivalModel
    assert ph.MicroSimModel is ph.IndividualStateTransitionModel
    assert ph.DESModel is ph.DiscreteEventSimulationModel


def test_canonical_model_names_are_exported():
    expected = {
        "CohortStateTransitionModel",
        "PartitionedSurvivalModel",
        "IndividualStateTransitionModel",
        "DiscreteEventSimulationModel",
    }
    assert expected.issubset(set(ph.__all__))
