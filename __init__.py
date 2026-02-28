"""
PyHEOR - Python Health Economics and Outcome Research
====================================================

A Python framework for health economic modeling and cost-effectiveness analysis.
Inspired by R's hesim package but with enhanced features:

- **Base case analysis**: Deterministic analysis with point estimates
- **One-way sensitivity analysis (OWSA)**: Tornado diagrams and parameter exploration
- **Probabilistic sensitivity analysis (PSA)**: Full uncertainty quantification
- **Flexible cost definitions**: First-cycle-only costs, time-dependent functions, 
  one-time costs, and more
- **Beautiful visualizations**: State transition diagrams, TreeAge-style model diagrams,
  Markov traces, tornado plots, CE planes, and CEACs

Quick Start
-----------
>>> import pyheor as ph
>>> model = ph.MarkovModel(
...     states=["Healthy", "Sick", "Dead"],
...     strategies=["Standard", "New Treatment"],
...     n_cycles=20,
... )
>>> model.add_param("p_HS", base=0.15, dist=ph.Beta(mean=0.15, sd=0.03))
>>> model.set_transitions("Standard", lambda p, t: [
...     [ph.C,  p["p_HS"], 0.02],
...     [0,     ph.C,      0.10],
...     [0,     0,         1   ],
... ])
>>> result = model.run_base_case()
>>> result.summary()
"""

__version__ = "0.1.0"
__author__ = "PyHEOR Team"

# Core sentinel
from .utils import C

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
from .model import MarkovModel, Param
from .psm import PSMModel
from .microsim import MicroSimModel, PatientProfile
from .des import DESModel

# Results
from .results import (
    BaseResult, OWSAResult, PSAResult, PSMBaseResult,
    MicroSimResult, MicroSimPSAResult,
    DESResult, DESPSAResult,
)

# Excel export
from .excel_export import export_to_excel, export_comparison_excel
from .excel_model import export_excel_model

# Comparison / CEA
from .comparison import CEAnalysis, calculate_icers

# Budget Impact Analysis
from .bia import BudgetImpactAnalysis

# IPD Fitting
from .fitting import SurvivalFitter, FitResult, kaplan_meier

# Digitize / IPD Reconstruction
from .digitize import clean_digitized_km, guyot_reconstruct

# NMA Integration
from .nma import (
    NMAPosterior,
    PosteriorDist,
    CorrelatedPosterior,
    load_nma_samples,
    make_ph_curves,
    make_aft_curves,
)

__all__ = [
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
    # Excel
    "export_to_excel",
    "export_comparison_excel",
    "export_excel_model",
    # Comparison / CEA
    "CEAnalysis",
    "calculate_icers",
    # Budget Impact Analysis
    "BudgetImpactAnalysis",
    # IPD Fitting
    "SurvivalFitter",
    "FitResult",
    "kaplan_meier",
    # Digitize / IPD Reconstruction
    "clean_digitized_km",
    "guyot_reconstruct",
    # NMA Integration
    "NMAPosterior",
    "PosteriorDist",
    "CorrelatedPosterior",
    "load_nma_samples",
    "make_ph_curves",
    "make_aft_curves",
]
