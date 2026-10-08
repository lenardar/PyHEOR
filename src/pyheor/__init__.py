"""PyHEOR: health economic models with explicit time and reward units."""

__version__ = "0.4.1"
__author__ = "PyHEOR Team"

# Core sentinel
from .utils import C
from .time import Cycle, qaly, rescale_discount_rate
from .survival_tools import from_flexsurv, rescale_survival, ScaledSurvival
from .models.rewards import RewardContext

# Distributions
from .distributions import (
    Distribution,
    Beta,
    Gamma,
    Normal,
    LogNormal,
    Uniform,
    Triangular,
    Dirichlet,
    Fixed,
)

# Survival distributions
from .survival import (
    SurvivalDistribution,
    Exponential,
    Weibull,
    LogLogistic,
    SurvLogNormal,
    Gompertz,
    GeneralizedGamma,
    ProportionalHazards,
    AcceleratedFailureTime,
    KaplanMeier,
    PiecewiseExponential,
)

# Models
from .models.common import Param
from .models.markov import CohortStateTransitionModel, MarkovModel
from .models.psm import PartitionedSurvivalModel, PSMModel
from .models.microsim import (
    IndividualStateTransitionModel,
    MicroSimModel,
    PatientProfile,
)
from .models.des import DiscreteEventSimulationModel, DESModel

# Results
from .analysis.results import (
    BaseResult, OWSAResult, PSAResult, PSMBaseResult,
    MicroSimResult, MicroSimPSAResult,
    DESResult, DESPSAResult,
)

# Excel export
from .export.excel import export_to_excel, export_comparison_excel
from .export.excel_model import export_excel_model

# Report
from .export.report import generate_report

# Comparison / CEA
from .analysis.comparison import CEAnalysis, calculate_icers

__all__ = [
    "Cycle", "qaly", "rescale_discount_rate", "from_flexsurv",
    "rescale_survival", "ScaledSurvival", "RewardContext",
    # Sentinel
    "C",
    # Distributions
    "Distribution",
    "Beta",
    "Gamma", 
    "Normal",
    "LogNormal",
    "Uniform",
    "Triangular",
    "Dirichlet",
    "Fixed",
    # Survival
    "SurvivalDistribution",
    "Exponential",
    "Weibull",
    "LogLogistic",
    "SurvLogNormal",
    "Gompertz",
    "GeneralizedGamma",
    "ProportionalHazards",
    "AcceleratedFailureTime",
    "KaplanMeier",
    "PiecewiseExponential",
    # Models
    "CohortStateTransitionModel",
    "PartitionedSurvivalModel",
    "IndividualStateTransitionModel",
    "DiscreteEventSimulationModel",
    "MarkovModel",
    "PSMModel",
    "MicroSimModel",
    "PatientProfile",
    "DESModel",
    "Param",
    # Results
    "BaseResult",
    "PSMBaseResult",
    "OWSAResult", 
    "PSAResult",
    "MicroSimResult",
    "MicroSimPSAResult",
    "DESResult",
    "DESPSAResult",
    # Excel / Report
    "export_to_excel",
    "export_comparison_excel",
    "export_excel_model",
    "generate_report",
    # Comparison / CEA
    "CEAnalysis",
    "calculate_icers",
]
