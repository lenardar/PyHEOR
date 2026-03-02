"""
Parametric survival distributions for Partitioned Survival Models.

Supports common parametric families used in health economics:
- Exponential
- Weibull (AFT & PH parameterizations)
- Log-logistic
- Log-normal
- Gompertz
- Generalized Gamma

Each distribution provides:
- survival(t): S(t) = P(T > t)
- hazard(t): h(t) = f(t) / S(t)
- cumulative_hazard(t): H(t) = -log(S(t))
- pdf(t): probability density function
- quantile(p): inverse survival function
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union
from scipy import stats as sp_stats


class SurvivalDistribution(ABC):
    """Base class for parametric survival distributions."""

    @abstractmethod
    def survival(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Survival function S(t) = P(T > t)."""
        pass

    @abstractmethod
    def hazard(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Hazard function h(t) = f(t) / S(t)."""
        pass

    def cumulative_hazard(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative hazard H(t) = -log(S(t))."""
        s = self.survival(t)
        return -np.log(np.clip(s, 1e-300, None))

    def pdf(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function f(t) = h(t) * S(t)."""
        return self.hazard(t) * self.survival(t)

    def quantile(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function (inverse CDF). Returns t such that F(t) = p.

        Default uses numerical inversion; subclasses may override with
        closed-form solutions.
        """
        from scipy.optimize import brentq

        p = np.asarray(p, dtype=float)
        scalar = p.ndim == 0
        p = np.atleast_1d(p)

        results = np.empty_like(p)
        for i, pi in enumerate(p):
            if pi <= 0:
                results[i] = 0.0
            elif pi >= 1:
                results[i] = np.inf
            else:
                try:
                    results[i] = brentq(lambda t: 1 - self.survival(t) - pi, 0, 1e6)
                except ValueError:
                    results[i] = np.nan

        return float(results[0]) if scalar else results

    def restricted_mean(self, t_max: float, n_points: int = 1000) -> float:
        """Restricted mean survival time (RMST) up to t_max.

        RMST = integral from 0 to t_max of S(t) dt.
        """
        t = np.linspace(0, t_max, n_points)
        s = self.survival(t)
        return float(np.trapezoid(s, t))

    @abstractmethod
    def __repr__(self) -> str:
        pass


# =============================================================================
# Exponential Distribution
# =============================================================================

class Exponential(SurvivalDistribution):
    """Exponential survival distribution.

    S(t) = exp(-λt)
    h(t) = λ

    Parameters
    ----------
    rate : float
        Hazard rate λ > 0.
    """

    def __init__(self, rate: float):
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        self.rate = float(rate)

    def survival(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-self.rate * t)

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        return np.full_like(t, self.rate) if t.ndim > 0 else self.rate

    def cumulative_hazard(self, t):
        t = np.asarray(t, dtype=float)
        return self.rate * t

    def quantile(self, p):
        p = np.asarray(p, dtype=float)
        return -np.log(1 - p) / self.rate

    def __repr__(self):
        return f"Exponential(rate={self.rate:.6f})"


# =============================================================================
# Weibull Distribution
# =============================================================================

class Weibull(SurvivalDistribution):
    """Weibull survival distribution (AFT parameterization).

    S(t) = exp(-(t/scale)^shape)
    h(t) = (shape/scale) * (t/scale)^(shape-1)

    Parameters
    ----------
    shape : float
        Shape parameter (k > 0). shape < 1: decreasing hazard,
        shape = 1: constant (exponential), shape > 1: increasing hazard.
    scale : float
        Scale parameter (λ > 0).
    """

    def __init__(self, shape: float, scale: float):
        if shape <= 0 or scale <= 0:
            raise ValueError(f"Shape and scale must be positive, got shape={shape}, scale={scale}")
        self.shape = float(shape)
        self.scale = float(scale)

    def survival(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-((t / self.scale) ** self.shape))

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        t_safe = np.maximum(t, 1e-300)
        return (self.shape / self.scale) * (t_safe / self.scale) ** (self.shape - 1)

    def cumulative_hazard(self, t):
        t = np.asarray(t, dtype=float)
        return (t / self.scale) ** self.shape

    def quantile(self, p):
        p = np.asarray(p, dtype=float)
        return self.scale * (-np.log(1 - p)) ** (1 / self.shape)

    @classmethod
    def from_ph(cls, shape: float, scale: float) -> "Weibull":
        """Create from proportional hazards (PH) parameterization.

        In PH form: h(t) = scale * shape * t^(shape-1)
        Convert to AFT: scale_aft = scale^(-1/shape)
        """
        aft_scale = scale ** (-1 / shape)
        return cls(shape=shape, scale=aft_scale)

    def __repr__(self):
        return f"Weibull(shape={self.shape:.4f}, scale={self.scale:.4f})"


# =============================================================================
# Log-logistic Distribution
# =============================================================================

class LogLogistic(SurvivalDistribution):
    """Log-logistic survival distribution.

    S(t) = 1 / (1 + (t/scale)^shape)
    h(t) = (shape/scale)(t/scale)^(shape-1) / (1 + (t/scale)^shape)

    Parameters
    ----------
    shape : float
        Shape parameter (> 0).
    scale : float
        Scale parameter (> 0).
    """

    def __init__(self, shape: float, scale: float):
        if shape <= 0 or scale <= 0:
            raise ValueError(f"Shape and scale must be positive")
        self.shape = float(shape)
        self.scale = float(scale)

    def survival(self, t):
        t = np.asarray(t, dtype=float)
        return 1.0 / (1.0 + (t / self.scale) ** self.shape)

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        t_safe = np.maximum(t, 1e-300)
        num = (self.shape / self.scale) * (t_safe / self.scale) ** (self.shape - 1)
        den = 1.0 + (t_safe / self.scale) ** self.shape
        return num / den

    def quantile(self, p):
        p = np.asarray(p, dtype=float)
        return self.scale * (p / (1 - p)) ** (1 / self.shape)

    def __repr__(self):
        return f"LogLogistic(shape={self.shape:.4f}, scale={self.scale:.4f})"


# =============================================================================
# Log-normal Distribution
# =============================================================================

class SurvLogNormal(SurvivalDistribution):
    """Log-normal survival distribution.

    S(t) = 1 - Φ((log(t) - μ) / σ)

    where Φ is the standard normal CDF.

    Parameters
    ----------
    meanlog : float
        Mean of log(T).
    sdlog : float
        Standard deviation of log(T).
    """

    def __init__(self, meanlog: float, sdlog: float):
        if sdlog <= 0:
            raise ValueError(f"sdlog must be positive, got {sdlog}")
        self.meanlog = float(meanlog)
        self.sdlog = float(sdlog)

    def survival(self, t):
        t = np.asarray(t, dtype=float)
        t_safe = np.maximum(t, 1e-300)
        z = (np.log(t_safe) - self.meanlog) / self.sdlog
        return 1.0 - sp_stats.norm.cdf(z)

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        t_safe = np.maximum(t, 1e-300)
        z = (np.log(t_safe) - self.meanlog) / self.sdlog
        f = sp_stats.norm.pdf(z) / (t_safe * self.sdlog)
        s = 1.0 - sp_stats.norm.cdf(z)
        return f / np.maximum(s, 1e-300)

    def quantile(self, p):
        p = np.asarray(p, dtype=float)
        return np.exp(self.meanlog + self.sdlog * sp_stats.norm.ppf(p))

    def __repr__(self):
        return f"SurvLogNormal(meanlog={self.meanlog:.4f}, sdlog={self.sdlog:.4f})"


# =============================================================================
# Gompertz Distribution
# =============================================================================

class Gompertz(SurvivalDistribution):
    """Gompertz survival distribution.

    S(t) = exp(-b/a * (exp(a*t) - 1))
    h(t) = b * exp(a*t)

    Parameters
    ----------
    shape : float
        Shape parameter (a). Can be positive (increasing hazard) or
        negative (decreasing hazard).
    rate : float
        Rate parameter (b > 0). Baseline hazard.
    """

    def __init__(self, shape: float, rate: float):
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        self.shape = float(shape)
        self.rate = float(rate)

    def survival(self, t):
        t = np.asarray(t, dtype=float)
        if abs(self.shape) < 1e-12:
            return np.exp(-self.rate * t)
        return np.exp(-self.rate / self.shape * (np.exp(self.shape * t) - 1))

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        return self.rate * np.exp(self.shape * t)

    def cumulative_hazard(self, t):
        t = np.asarray(t, dtype=float)
        if abs(self.shape) < 1e-12:
            return self.rate * t
        return self.rate / self.shape * (np.exp(self.shape * t) - 1)

    def __repr__(self):
        return f"Gompertz(shape={self.shape:.6f}, rate={self.rate:.6f})"


# =============================================================================
# Generalized Gamma Distribution
# =============================================================================

class GeneralizedGamma(SurvivalDistribution):
    """Generalized Gamma survival distribution (Stacy parameterization).

    Uses scipy's implementation via the gamma distribution.

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (> 0).
    Q : float
        Shape parameter. Special cases:
        - Q = 1: Weibull
        - Q = 0: Log-normal
        - sigma = Q: Gamma
    """

    def __init__(self, mu: float, sigma: float, Q: float):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.Q = float(Q)

    def _params(self):
        """Convert to scipy-compatible parameters."""
        if abs(self.Q) < 1e-10:
            # Limiting case: log-normal
            return None
        gamma_shape = self.Q ** (-2)
        gamma_scale = np.exp(self.mu + self.sigma * np.log(self.Q ** 2) / self.Q)
        gamma_pow = self.Q / self.sigma
        return gamma_shape, gamma_scale, gamma_pow

    def survival(self, t):
        t = np.asarray(t, dtype=float)
        t_safe = np.maximum(t, 1e-300)

        if abs(self.Q) < 1e-10:
            # Log-normal limit
            z = (np.log(t_safe) - self.mu) / self.sigma
            return 1.0 - sp_stats.norm.cdf(z)

        gamma_shape, gamma_scale, gamma_pow = self._params()
        u = (t_safe / gamma_scale) ** gamma_pow
        if self.Q > 0:
            return 1.0 - sp_stats.gamma.cdf(u, gamma_shape)
        else:
            return sp_stats.gamma.cdf(u, gamma_shape)

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        t_safe = np.maximum(t, 1e-300)
        s = self.survival(t_safe)
        f = self.pdf(t_safe)
        return f / np.maximum(s, 1e-300)

    def pdf(self, t):
        t = np.asarray(t, dtype=float)
        t_safe = np.maximum(t, 1e-300)

        if abs(self.Q) < 1e-10:
            z = (np.log(t_safe) - self.mu) / self.sigma
            return sp_stats.norm.pdf(z) / (t_safe * self.sigma)

        gamma_shape, gamma_scale, gamma_pow = self._params()
        u = (t_safe / gamma_scale) ** gamma_pow
        du_dt = gamma_pow * t_safe ** (gamma_pow - 1) / gamma_scale ** gamma_pow
        return sp_stats.gamma.pdf(u, gamma_shape) * np.abs(du_dt)

    def __repr__(self):
        return f"GeneralizedGamma(mu={self.mu:.4f}, sigma={self.sigma:.4f}, Q={self.Q:.4f})"


# =============================================================================
# Survival Curve with Treatment Effect
# =============================================================================

class ProportionalHazards(SurvivalDistribution):
    """Apply a proportional hazards treatment effect to a baseline curve.

    h_trt(t) = h0(t) * HR
    S_trt(t) = S0(t) ^ HR

    Parameters
    ----------
    baseline : SurvivalDistribution
        Baseline survival curve.
    hr : float
        Hazard ratio (HR < 1 means treatment benefit).
    """

    def __init__(self, baseline: SurvivalDistribution, hr: float):
        if hr <= 0:
            raise ValueError(f"Hazard ratio must be positive, got {hr}")
        self.baseline = baseline
        self.hr = float(hr)

    def survival(self, t):
        return self.baseline.survival(t) ** self.hr

    def hazard(self, t):
        return self.baseline.hazard(t) * self.hr

    def cumulative_hazard(self, t):
        return self.baseline.cumulative_hazard(t) * self.hr

    def __repr__(self):
        return f"PH({self.baseline}, HR={self.hr:.4f})"


class AcceleratedFailureTime(SurvivalDistribution):
    """Apply an accelerated failure time (AFT) treatment effect.

    S_trt(t) = S0(t / AF)

    Parameters
    ----------
    baseline : SurvivalDistribution
        Baseline survival curve.
    af : float
        Acceleration factor (AF > 1 means treatment extends survival).
    """

    def __init__(self, baseline: SurvivalDistribution, af: float):
        if af <= 0:
            raise ValueError(f"Acceleration factor must be positive, got {af}")
        self.baseline = baseline
        self.af = float(af)

    def survival(self, t):
        t = np.asarray(t, dtype=float)
        return self.baseline.survival(t / self.af)

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        return self.baseline.hazard(t / self.af) / self.af

    def __repr__(self):
        return f"AFT({self.baseline}, AF={self.af:.4f})"


# =============================================================================
# Kaplan-Meier (empirical) Survival Curve
# =============================================================================

class KaplanMeier(SurvivalDistribution):
    """Empirical survival curve from Kaplan-Meier data.

    Supports step-function interpolation of digitized/fitted KM curves.

    Parameters
    ----------
    times : array-like
        Time points.
    survival_probs : array-like
        Survival probabilities at each time point.
    extrapolation : str
        How to extrapolate beyond observed data:
        - "constant": last observed S(t)
        - "exponential": fit exponential tail from last point
    """

    def __init__(self, times, survival_probs, extrapolation: str = "constant"):
        self.times = np.asarray(times, dtype=float)
        self.surv = np.asarray(survival_probs, dtype=float)
        self.extrapolation = extrapolation

        # Sort by time
        order = np.argsort(self.times)
        self.times = self.times[order]
        self.surv = self.surv[order]

        # Prepend t=0, S=1 if not present
        if self.times[0] > 0:
            self.times = np.concatenate([[0], self.times])
            self.surv = np.concatenate([[1.0], self.surv])

        # Compute tail rate for exponential extrapolation
        if extrapolation == "exponential" and len(self.times) >= 2:
            last_s = max(self.surv[-1], 1e-10)
            t_last = self.times[-1]
            self._tail_rate = -np.log(last_s) / t_last if t_last > 0 else 0
        else:
            self._tail_rate = 0

    def survival(self, t):
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        # Step-function interpolation
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = np.clip(idx, 0, len(self.surv) - 1)
        result = self.surv[idx]

        # Handle extrapolation beyond last observed time
        beyond = t > self.times[-1]
        if np.any(beyond):
            if self.extrapolation == "exponential":
                result[beyond] = np.exp(-self._tail_rate * t[beyond])
            # "constant" keeps the last value (already set)

        return float(result[0]) if scalar else result

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        t_safe = np.maximum(t, 1e-300)
        # Numerical differentiation of cumulative hazard
        dt = 0.001
        H1 = self.cumulative_hazard(t_safe)
        H2 = self.cumulative_hazard(t_safe + dt)
        return (H2 - H1) / dt

    def __repr__(self):
        return (
            f"KaplanMeier(n_points={len(self.times)}, "
            f"max_t={self.times[-1]:.1f}, "
            f"extrapolation='{self.extrapolation}')"
        )


# =============================================================================
# Piecewise Constant Hazard
# =============================================================================

class PiecewiseExponential(SurvivalDistribution):
    """Piecewise constant hazard (piecewise exponential) model.

    Useful for modeling different phases (e.g., treatment-on vs treatment-off).

    Parameters
    ----------
    breakpoints : array-like
        Time points where hazard changes (not including 0).
    rates : array-like
        Hazard rates for each interval. Length = len(breakpoints) + 1.
        rates[0] is the rate from 0 to breakpoints[0], etc.
    """

    def __init__(self, breakpoints, rates):
        self.breakpoints = np.asarray(breakpoints, dtype=float)
        self.rates = np.asarray(rates, dtype=float)
        if len(self.rates) != len(self.breakpoints) + 1:
            raise ValueError(
                f"rates length ({len(self.rates)}) must be "
                f"breakpoints length + 1 ({len(self.breakpoints) + 1})"
            )

    def survival(self, t):
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        H = self._cumhaz(t)
        result = np.exp(-H)
        return float(result[0]) if scalar else result

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        result = np.full_like(t, self.rates[-1])
        for i, bp in enumerate(self.breakpoints):
            result[t <= bp] = self.rates[i]
        # first interval
        result[t <= (self.breakpoints[0] if len(self.breakpoints) > 0 else np.inf)] = self.rates[0]

        return float(result[0]) if scalar else result

    def _cumhaz(self, t):
        H = np.zeros_like(t)
        prev = 0.0
        for i, rate in enumerate(self.rates):
            if i < len(self.breakpoints):
                bp = self.breakpoints[i]
                duration = np.clip(t - prev, 0, bp - prev)
            else:
                duration = np.clip(t - prev, 0, None)
            H += rate * duration
            if i < len(self.breakpoints):
                prev = bp
        return H

    def cumulative_hazard(self, t):
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        result = self._cumhaz(t)
        return float(result[0]) if scalar else result

    def __repr__(self):
        return f"PiecewiseExponential(breakpoints={self.breakpoints}, rates={self.rates})"
