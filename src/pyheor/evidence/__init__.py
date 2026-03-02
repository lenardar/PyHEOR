"""Evidence synthesis: IPD fitting, digitization, and NMA integration."""

from .fitting import SurvivalFitter, FitResult, kaplan_meier
from .digitize import clean_digitized_km, guyot_reconstruct
from .nma import (
    NMAPosterior,
    PosteriorDist,
    CorrelatedPosterior,
    load_nma_samples,
    make_ph_curves,
    make_aft_curves,
)

__all__ = [
    "SurvivalFitter",
    "FitResult",
    "kaplan_meier",
    "clean_digitized_km",
    "guyot_reconstruct",
    "NMAPosterior",
    "PosteriorDist",
    "CorrelatedPosterior",
    "load_nma_samples",
    "make_ph_curves",
    "make_aft_curves",
]
