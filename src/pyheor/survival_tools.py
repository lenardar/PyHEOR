"""Explicit time scaling and flexsurv parameter adapters."""

import numpy as np
from .time import Cycle
from .survival import (
    SurvivalDistribution, Exponential, Weibull, LogLogistic, SurvLogNormal,
    Gompertz, GeneralizedGamma,
)


class ScaledSurvival(SurvivalDistribution):
    """S_target(t) = S_source(t × factor), with hazards and quantiles scaled too."""

    def __init__(self, baseline, factor):
        if not isinstance(baseline, SurvivalDistribution):
            raise TypeError("baseline must be a SurvivalDistribution")
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Time scaling factor must be positive and finite")
        self.baseline, self.factor = baseline, float(factor)

    def survival(self, t):
        return self.baseline.survival(np.asarray(t) * self.factor)

    def hazard(self, t):
        return self.baseline.hazard(np.asarray(t) * self.factor) * self.factor

    def quantile(self, p):
        return self.baseline.quantile(p) / self.factor

    def __repr__(self):
        return f"ScaledSurvival({self.baseline!r}, factor={self.factor})"


def rescale_survival(curve, *, from_unit, to_period):
    """Adapt a fitted curve to model cycles (or one unit of DES time).

    Use inside a parameter callback so every PSA/OWSA draw rebuilds the curve.
    """
    if not isinstance(to_period, Cycle):
        raise TypeError("to_period must be an explicit Cycle")
    return ScaledSurvival(curve, to_period.in_unit(from_unit))


def from_flexsurv(distribution, **parameters):
    """Construct from natural-scale flexsurv parameters, not optimizer coefficients.

    Supported R names: exp, weibull, weibullPH, llogis, lnorm, gompertz,
    gengamma (Prentice), gengamma.orig (original Stacy).
    The fitted time unit is preserved; rescale_survival changes time units.
    """
    if not all(np.isfinite(v) for v in parameters.values()):
        raise ValueError("Distribution parameters must be finite")
    specs = {
        "exp": (Exponential, ("rate",)),
        "weibull": (Weibull, ("shape", "scale")),
        "llogis": (LogLogistic, ("shape", "scale")),
        "lnorm": (SurvLogNormal, ("meanlog", "sdlog")),
        "gompertz": (Gompertz, ("shape", "rate")),
        "gengamma": (GeneralizedGamma, ("mu", "sigma", "Q")),
    }
    if distribution == "weibullPH":
        if set(parameters) != {"shape", "scale"}:
            raise ValueError("weibullPH requires shape and scale")
        shape, rate = parameters["shape"], parameters["scale"]
        if shape <= 0 or rate <= 0:
            raise ValueError("weibullPH shape and scale must be positive")
        return Weibull(shape=shape, scale=rate ** (-1 / shape))
    if distribution == "gengamma.orig":
        if set(parameters) != {"shape", "scale", "k"}:
            raise ValueError("gengamma.orig requires shape, scale and k")
        shape, scale, k = (parameters[n] for n in ("shape", "scale", "k"))
        if shape <= 0 or scale <= 0 or k <= 0:
            raise ValueError("gengamma.orig parameters must be positive")
        return GeneralizedGamma(mu=np.log(scale) + np.log(k) / shape,
                                sigma=1 / (shape * np.sqrt(k)), Q=1 / np.sqrt(k))
    if distribution not in specs:
        raise ValueError(f"Unsupported flexsurv distribution {distribution!r}")
    constructor, names = specs[distribution]
    if set(parameters) != set(names):
        raise ValueError(f"{distribution} requires exactly {names}")
    if not all(np.isfinite(v) for v in parameters.values()):
        raise ValueError("Distribution parameters must be finite")
    return constructor(**parameters)
