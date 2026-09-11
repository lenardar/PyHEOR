"""Tests for canonical model names and their concise public aliases."""

import pyheor as ph
from pyheor.models import Param as ModelsParam
from pyheor.models.common import Param as CommonParam
from pyheor.models import des, markov, microsim, psm


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


def test_param_has_one_shared_class_identity():
    assert ph.Param is ModelsParam
    assert ph.Param is CommonParam


def test_model_modules_do_not_expose_param_as_their_own_api():
    for module in (markov, psm, microsim, des):
        assert not hasattr(module, "Param")
