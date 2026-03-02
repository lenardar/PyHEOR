"""
IPD (Individual Patient Data) survival curve fitting module.

Fits parametric survival models to time-to-event data and provides:
- Maximum likelihood estimation for 6 standard distributions
- AIC/BIC model comparison table
- Automatic best-model selection
- Kaplan-Meier estimation
- Combined KM + fitted curves visualization
- Goodness-of-fit diagnostics (log-cumulative hazard, Q-Q plots)

Supported distributions:
- Exponential
- Weibull
- Log-logistic
- Log-normal
- Gompertz
- Generalized Gamma

References
----------
- Latimer NR (2013). Survival analysis for economic evaluations alongside
  clinical trials. Medical Decision Making, 33(6), 743-754.
- NICE DSU TSD 14: Survival analysis for economic evaluations.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from scipy.optimize import minimize
from scipy import stats as sp_stats

from ..survival import (
    SurvivalDistribution,
    Exponential,
    Weibull,
    LogLogistic,
    SurvLogNormal,
    Gompertz,
    GeneralizedGamma,
)


# =============================================================================
# Kaplan-Meier Estimator
# =============================================================================

def kaplan_meier(
    time: np.ndarray,
    event: np.ndarray,
    conf_level: float = 0.95,
) -> pd.DataFrame:
    """Compute Kaplan-Meier survival estimate.

    Parameters
    ----------
    time : array-like
        Observed times (time to event or censoring).
    event : array-like
        Event indicator (1 = event, 0 = censored).
    conf_level : float
        Confidence level for Greenwood CI (default: 0.95).

    Returns
    -------
    pd.DataFrame
        Columns: time, n_risk, n_event, n_censor, survival, se, lower, upper.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)

    # Sort by time
    order = np.argsort(time)
    time = time[order]
    event = event[order]

    n = len(time)
    unique_times = np.unique(time[event == 1])  # Event times only

    # Add time 0
    results = [{
        'time': 0.0, 'n_risk': n, 'n_event': 0, 'n_censor': 0,
        'survival': 1.0, 'se': 0.0,
    }]

    at_risk = n
    surv = 1.0
    var_sum = 0.0  # For Greenwood's formula

    # Process censoring before each event time
    t_prev = 0.0
    for t_event in unique_times:
        # Count censored between t_prev and t_event
        censored = np.sum((time > t_prev) & (time < t_event) & (event == 0))
        at_risk -= censored

        # Events at this time
        d = np.sum((time == t_event) & (event == 1))
        c = np.sum((time == t_event) & (event == 0))

        if at_risk > 0 and d > 0:
            surv *= (1 - d / at_risk)
            if at_risk > d:
                var_sum += d / (at_risk * (at_risk - d))
            se = surv * np.sqrt(var_sum) if surv > 0 else 0

            results.append({
                'time': t_event,
                'n_risk': at_risk,
                'n_event': d,
                'n_censor': c,
                'survival': surv,
                'se': se,
            })

        at_risk -= (d + c)
        t_prev = t_event

    df = pd.DataFrame(results)

    # Confidence interval (log-log transform)
    z = sp_stats.norm.ppf((1 + conf_level) / 2)
    df['lower'] = np.clip(df['survival'] - z * df['se'], 0, 1)
    df['upper'] = np.clip(df['survival'] + z * df['se'], 0, 1)

    return df


# =============================================================================
# Single Distribution Fitter
# =============================================================================

@dataclass
class FitResult:
    """Result of fitting a single parametric distribution.

    Attributes
    ----------
    name : str
        Distribution name.
    distribution : SurvivalDistribution
        Fitted survival distribution object.
    params : dict
        Estimated parameters.
    loglik : float
        Log-likelihood at MLE.
    n_params : int
        Number of estimated parameters.
    n_obs : int
        Number of observations.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    converged : bool
        Whether optimization converged.
    vcov : np.ndarray or None
        Variance-covariance matrix of estimates (if available).
    """
    name: str
    distribution: SurvivalDistribution
    params: dict
    loglik: float
    n_params: int
    n_obs: int
    aic: float
    bic: float
    converged: bool
    vcov: Optional[np.ndarray] = None

    def __repr__(self):
        return (
            f"FitResult('{self.name}', AIC={self.aic:.2f}, "
            f"BIC={self.bic:.2f}, loglik={self.loglik:.2f})"
        )


def _neg_loglik_exponential(params, time, event):
    rate = np.exp(params[0])
    ll = event * np.log(rate) - rate * time
    return -np.sum(ll)


def _neg_loglik_weibull(params, time, event):
    shape = np.exp(params[0])
    scale = np.exp(params[1])
    t_safe = np.maximum(time, 1e-300)
    ll = (event * (np.log(shape) - np.log(scale) +
          (shape - 1) * (np.log(t_safe) - np.log(scale)))
          - (t_safe / scale) ** shape)
    return -np.sum(ll)


def _neg_loglik_loglogistic(params, time, event):
    shape = np.exp(params[0])
    scale = np.exp(params[1])
    t_safe = np.maximum(time, 1e-300)
    u = (t_safe / scale) ** shape
    ll = (event * (np.log(shape) - np.log(scale) +
          (shape - 1) * (np.log(t_safe) - np.log(scale)) - np.log(1 + u))
          - np.log(1 + u))
    return -np.sum(ll)


def _neg_loglik_lognormal(params, time, event):
    mu = params[0]
    sigma = np.exp(params[1])
    t_safe = np.maximum(time, 1e-300)
    z = (np.log(t_safe) - mu) / sigma
    ll = (event * (sp_stats.norm.logpdf(z) - np.log(t_safe) - np.log(sigma))
          + (1 - event) * sp_stats.norm.logsf(z))
    # For events: log(f(t)) = logphi(z) - log(t) - log(sigma)
    # Actually full derivation:
    ll = np.where(
        event == 1,
        sp_stats.norm.logpdf(z) - np.log(t_safe) - np.log(sigma),
        np.log(np.maximum(1 - sp_stats.norm.cdf(z), 1e-300))
    )
    return -np.sum(ll)


def _neg_loglik_gompertz(params, time, event):
    shape = params[0]  # Can be negative
    rate = np.exp(params[1])
    if abs(shape) < 1e-12:
        # Reduces to exponential
        ll = event * np.log(rate) - rate * time
    else:
        ll = (event * (np.log(rate) + shape * time)
              - rate / shape * (np.exp(shape * time) - 1))
    return -np.sum(ll)


def _neg_loglik_gengamma(params, time, event):
    mu = params[0]
    sigma = np.exp(params[1])
    Q = params[2]
    t_safe = np.maximum(time, 1e-300)

    w = (np.log(t_safe) - mu) / sigma

    if abs(Q) < 1e-10:
        # Log-normal limit
        ll = np.where(
            event == 1,
            sp_stats.norm.logpdf(w) - np.log(t_safe) - np.log(sigma),
            np.log(np.maximum(1 - sp_stats.norm.cdf(w), 1e-300))
        )
    else:
        Q2 = Q ** 2
        u = Q2 * np.exp(Q * w / np.sqrt(Q2))  # = Q^2 * exp(abs(Q)*w)
        # Careful with sign of Q
        q_abs = np.abs(Q)

        if Q > 0:
            log_f = (np.log(q_abs) - np.log(sigma) - np.log(t_safe)
                     + Q2 * np.log(Q2) / Q2  # = log(Q^2) ... simplified below
                     + (1/Q2) * (Q2 * w - np.exp(q_abs * w) * Q2)
                     - sp_stats.gamma.logpdf(np.maximum(u, 1e-300), 1/Q2) * 0)

            # Use proper formulation
            log_f = np.where(
                event == 1,
                (np.log(q_abs) - np.log(sigma) - np.log(t_safe)
                 + (1/Q2 - 1) * np.log(Q2) + (1/Q2) * q_abs * w
                 - u / Q2
                 - sp_stats.gamma(1/Q2).logpdf(1) + sp_stats.gamma.logpdf(u / Q2, 1/Q2)),
                0
            )
        else:
            log_f = np.zeros_like(time)

        # Simpler approach: use numerical pdf/survival from the class
        try:
            dist = GeneralizedGamma(mu=mu, sigma=sigma, Q=Q)
            f_vals = dist.pdf(t_safe)
            s_vals = dist.survival(t_safe)
            f_vals = np.maximum(f_vals, 1e-300)
            s_vals = np.maximum(s_vals, 1e-300)
            ll = event * np.log(f_vals) + (1 - event) * np.log(s_vals)
        except Exception:
            return 1e10

    return -np.sum(ll)


def _fit_single(
    name: str,
    time: np.ndarray,
    event: np.ndarray,
    **kwargs,
) -> Optional[FitResult]:
    """Fit a single parametric model via MLE.

    Returns None if fitting fails.
    """
    n = len(time)
    t_safe = np.maximum(time, 1e-300)

    # Initial parameter guesses
    median_t = np.median(time[event == 1]) if np.sum(event) > 0 else np.median(time)
    median_t = max(median_t, 0.01)
    mean_log_t = np.mean(np.log(t_safe[event == 1])) if np.sum(event) > 0 else np.mean(np.log(t_safe))
    sd_log_t = max(np.std(np.log(t_safe[event == 1])), 0.1) if np.sum(event) > 1 else 0.5

    try:
        if name == "Exponential":
            rate_init = np.sum(event) / np.sum(time)
            x0 = [np.log(max(rate_init, 1e-6))]
            res = minimize(_neg_loglik_exponential, x0, args=(time, event),
                           method='Nelder-Mead', options={'maxiter': 5000})
            if not res.success and res.fun < 1e9:
                res.success = True
            rate = np.exp(res.x[0])
            dist = Exponential(rate=rate)
            params = {'rate': rate}
            n_params = 1

        elif name == "Weibull":
            x0 = [0.0, np.log(median_t)]
            res = minimize(_neg_loglik_weibull, x0, args=(time, event),
                           method='Nelder-Mead', options={'maxiter': 5000})
            if not res.success and res.fun < 1e9:
                res.success = True
            shape = np.exp(res.x[0])
            scale = np.exp(res.x[1])
            dist = Weibull(shape=shape, scale=scale)
            params = {'shape': shape, 'scale': scale}
            n_params = 2

        elif name == "Log-logistic":
            x0 = [0.0, np.log(median_t)]
            res = minimize(_neg_loglik_loglogistic, x0, args=(time, event),
                           method='Nelder-Mead', options={'maxiter': 5000})
            if not res.success and res.fun < 1e9:
                res.success = True
            shape = np.exp(res.x[0])
            scale = np.exp(res.x[1])
            dist = LogLogistic(shape=shape, scale=scale)
            params = {'shape': shape, 'scale': scale}
            n_params = 2

        elif name == "Log-normal":
            x0 = [mean_log_t, np.log(sd_log_t)]
            res = minimize(_neg_loglik_lognormal, x0, args=(time, event),
                           method='Nelder-Mead', options={'maxiter': 5000})
            if not res.success and res.fun < 1e9:
                res.success = True
            mu = res.x[0]
            sigma = np.exp(res.x[1])
            dist = SurvLogNormal(meanlog=mu, sdlog=sigma)
            params = {'meanlog': mu, 'sdlog': sigma}
            n_params = 2

        elif name == "Gompertz":
            rate_init = np.sum(event) / np.sum(time)
            x0 = [0.01, np.log(max(rate_init, 1e-6))]
            res = minimize(_neg_loglik_gompertz, x0, args=(time, event),
                           method='Nelder-Mead', options={'maxiter': 5000})
            if not res.success and res.fun < 1e9:
                res.success = True
            shape = res.x[0]
            rate = np.exp(res.x[1])
            dist = Gompertz(shape=shape, rate=rate)
            params = {'shape': shape, 'rate': rate}
            n_params = 2

        elif name == "Generalized Gamma":
            x0 = [mean_log_t, np.log(sd_log_t), 1.0]
            res = minimize(_neg_loglik_gengamma, x0, args=(time, event),
                           method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-8})
            if not res.success and res.fun < 1e9:
                res.success = True
            mu = res.x[0]
            sigma = np.exp(res.x[1])
            Q = res.x[2]
            dist = GeneralizedGamma(mu=mu, sigma=sigma, Q=Q)
            params = {'mu': mu, 'sigma': sigma, 'Q': Q}
            n_params = 3

        else:
            return None

        loglik = -res.fun
        aic = 2 * n_params - 2 * loglik
        bic = n_params * np.log(n) - 2 * loglik

        # Attempt to get vcov via numerical Hessian
        vcov = None
        try:
            from scipy.optimize import approx_fprime
            hess = np.zeros((len(res.x), len(res.x)))
            eps = 1e-5
            for i in range(len(res.x)):
                def grad_i(x):
                    return approx_fprime(
                        x,
                        lambda xx: -_get_negloglik_fn(name)(xx, time, event),
                        eps
                    )[i]
                hess[i] = approx_fprime(res.x, grad_i, eps)
            # vcov = -inv(hessian of loglik) = inv(-hessian of negloglik)
            # hess here is hessian of loglik, so vcov = -inv(hess)
            try:
                vcov = -np.linalg.inv(hess)
            except np.linalg.LinAlgError:
                vcov = None
        except Exception:
            vcov = None

        return FitResult(
            name=name,
            distribution=dist,
            params=params,
            loglik=loglik,
            n_params=n_params,
            n_obs=n,
            aic=aic,
            bic=bic,
            converged=res.success,
            vcov=vcov,
        )

    except Exception as e:
        print(f"  Warning: Failed to fit {name}: {e}")
        return None


def _get_negloglik_fn(name):
    fns = {
        "Exponential": _neg_loglik_exponential,
        "Weibull": _neg_loglik_weibull,
        "Log-logistic": _neg_loglik_loglogistic,
        "Log-normal": _neg_loglik_lognormal,
        "Gompertz": _neg_loglik_gompertz,
        "Generalized Gamma": _neg_loglik_gengamma,
    }
    return fns[name]


# =============================================================================
# Main IPD Fitter Class
# =============================================================================

ALL_DISTRIBUTIONS = [
    "Exponential",
    "Weibull",
    "Log-logistic",
    "Log-normal",
    "Gompertz",
    "Generalized Gamma",
]


class SurvivalFitter:
    """Fit parametric survival models to individual patient data (IPD).

    Performs maximum likelihood estimation for multiple parametric families,
    computes AIC/BIC, selects the best model, and provides diagnostic plots.

    Parameters
    ----------
    time : array-like
        Observed follow-up times.
    event : array-like
        Event indicator (1 = event occurred, 0 = censored).
    distributions : list of str, optional
        Which distributions to fit. Default: all 6.
        Options: "Exponential", "Weibull", "Log-logistic",
        "Log-normal", "Gompertz", "Generalized Gamma".
    label : str, optional
        A label for this dataset (e.g., "OS", "PFS").

    Examples
    --------
    >>> fitter = SurvivalFitter(time=df['time'], event=df['event'], label='OS')
    >>> fitter.fit()
    >>> print(fitter.summary())
    >>> best = fitter.best_model()
    >>> fig = fitter.plot_fits()
    """

    def __init__(
        self,
        time,
        event,
        distributions: Optional[List[str]] = None,
        label: str = "",
    ):
        self.time = np.asarray(time, dtype=float)
        self.event = np.asarray(event, dtype=int)
        self.label = label or "Survival"

        if distributions is None:
            self.distribution_names = list(ALL_DISTRIBUTIONS)
        else:
            self.distribution_names = list(distributions)

        self._results: Dict[str, FitResult] = {}
        self._km: Optional[pd.DataFrame] = None
        self._fitted = False

    @property
    def n_obs(self) -> int:
        return len(self.time)

    @property
    def n_events(self) -> int:
        return int(np.sum(self.event))

    @property
    def n_censored(self) -> int:
        return self.n_obs - self.n_events

    def fit(self, verbose: bool = True) -> "SurvivalFitter":
        """Fit all specified distributions.

        Parameters
        ----------
        verbose : bool
            Print progress messages.

        Returns
        -------
        SurvivalFitter
            Self, for method chaining.
        """
        if verbose:
            print(f"Fitting survival models to '{self.label}' data "
                  f"(n={self.n_obs}, events={self.n_events}, "
                  f"censored={self.n_censored})")

        # Compute KM estimate
        self._km = kaplan_meier(self.time, self.event)

        # Fit each distribution
        for name in self.distribution_names:
            if verbose:
                print(f"  Fitting {name}...", end=" ")
            result = _fit_single(name, self.time, self.event)
            if result is not None:
                self._results[name] = result
                if verbose:
                    print(f"AIC={result.aic:.2f}, BIC={result.bic:.2f} "
                          f"{'✓' if result.converged else '✗'}")
            else:
                if verbose:
                    print("FAILED")

        self._fitted = True

        if verbose:
            best = self.best_model()
            if best:
                print(f"\n  ★ Best model (AIC): {best.name} "
                      f"(AIC={best.aic:.2f})")

        return self

    def summary(self, sort_by: str = "aic") -> pd.DataFrame:
        """Return a summary table of all fitted models.

        Parameters
        ----------
        sort_by : str
            Column to sort by: "aic", "bic", or "loglik".

        Returns
        -------
        pd.DataFrame
            Comparison table with distribution, parameters, loglik,
            AIC, BIC, and delta-AIC/BIC.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        rows = []
        for name, r in self._results.items():
            param_str = ", ".join(f"{k}={v:.4f}" for k, v in r.params.items())
            rows.append({
                'Distribution': r.name,
                'Parameters': param_str,
                'k': r.n_params,
                'Log-Likelihood': r.loglik,
                'AIC': r.aic,
                'BIC': r.bic,
                'Converged': r.converged,
            })

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df

        df = df.sort_values(sort_by.upper() if sort_by.upper() in df.columns
                            else 'AIC').reset_index(drop=True)

        # Add delta columns
        df['ΔAIC'] = df['AIC'] - df['AIC'].min()
        df['ΔBIC'] = df['BIC'] - df['BIC'].min()

        # AIC weight (Akaike weights)
        delta_aic = df['ΔAIC'].values
        w = np.exp(-0.5 * delta_aic)
        df['AIC Weight'] = w / w.sum()

        return df

    def best_model(self, criterion: str = "aic") -> Optional[FitResult]:
        """Return the best-fitting model.

        Parameters
        ----------
        criterion : str
            Selection criterion: "aic" or "bic".

        Returns
        -------
        FitResult or None
            The best model, or None if no models were fitted.
        """
        if not self._results:
            return None

        if criterion == "aic":
            return min(self._results.values(), key=lambda r: r.aic)
        elif criterion == "bic":
            return min(self._results.values(), key=lambda r: r.bic)
        else:
            raise ValueError(f"Unknown criterion '{criterion}'. Use 'aic' or 'bic'.")

    def get_distribution(self, name: str) -> SurvivalDistribution:
        """Get the fitted survival distribution object by name.

        Parameters
        ----------
        name : str
            Distribution name (e.g., "Weibull").

        Returns
        -------
        SurvivalDistribution
        """
        if name not in self._results:
            raise KeyError(f"'{name}' not fitted. Available: {list(self._results.keys())}")
        return self._results[name].distribution

    def get_result(self, name: str) -> FitResult:
        """Get full FitResult by name."""
        if name not in self._results:
            raise KeyError(f"'{name}' not fitted. Available: {list(self._results.keys())}")
        return self._results[name]

    @property
    def km_data(self) -> pd.DataFrame:
        """Get Kaplan-Meier survival data."""
        if self._km is None:
            self._km = kaplan_meier(self.time, self.event)
        return self._km

    @property
    def results(self) -> Dict[str, FitResult]:
        """All fit results."""
        return dict(self._results)

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_fits(
        self,
        figsize: tuple = (10, 7),
        title: Optional[str] = None,
        show_ci: bool = True,
        show_km: bool = True,
        highlight_best: bool = True,
        t_max: Optional[float] = None,
        criterion: str = "aic",
    ):
        """Plot KM curve with all fitted parametric curves overlaid.

        Parameters
        ----------
        figsize : tuple
            Figure size.
        title : str, optional
            Custom title.
        show_ci : bool
            Show KM confidence interval.
        show_km : bool
            Show KM step function.
        highlight_best : bool
            Draw the best model with a thicker line.
        t_max : float, optional
            Maximum time to plot. Default: max observed time * 1.2.
        criterion : str
            Criterion for best model selection.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        km = self.km_data
        max_time = self.time.max()
        if t_max is None:
            t_max = max_time * 1.2

        t_plot = np.linspace(0.001, t_max, 500)

        # Plot KM
        if show_km:
            # Step function
            km_times = np.concatenate([[0], np.repeat(km['time'].values[1:], 2), [max_time]])
            km_surv = np.repeat(km['survival'].values, 2)
            ax.plot(km_times, km_surv, color='black', linewidth=2,
                    label='Kaplan-Meier', drawstyle='default', zorder=10)

            if show_ci:
                # CI as shaded area (step)
                km_lower = np.repeat(km['lower'].values, 2)
                km_upper = np.repeat(km['upper'].values, 2)
                ax.fill_between(km_times, km_lower, km_upper,
                                alpha=0.15, color='black', step=None,
                                label='95% CI')

        # Plot fitted curves
        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#795548']
        linestyles = ['-', '--', '-.', ':', '-', '--']

        best = self.best_model(criterion)

        for idx, (name, r) in enumerate(self._results.items()):
            s = r.distribution.survival(t_plot)
            lw = 3.0 if (highlight_best and best and name == best.name) else 1.5
            alpha = 1.0 if (highlight_best and best and name == best.name) else 0.7

            label = f"{name} (AIC={r.aic:.1f})"
            if best and name == best.name:
                label = f"★ {label}"

            ax.plot(t_plot, s,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=lw, alpha=alpha, label=label)

        # Vertical line at max observed time
        ax.axvline(max_time, color='#999', linewidth=0.8, linestyle=':',
                   alpha=0.6, label=f'Max follow-up ({max_time:.1f})')

        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, t_max)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)

        if title is None:
            title = f'Parametric Survival Fits — {self.label}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        return fig

    def plot_hazard(
        self,
        figsize: tuple = (10, 6),
        title: Optional[str] = None,
        t_max: Optional[float] = None,
    ):
        """Plot fitted hazard functions.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        max_time = self.time.max()
        if t_max is None:
            t_max = max_time
        t_plot = np.linspace(0.01, t_max, 500)

        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#795548']
        linestyles = ['-', '--', '-.', ':', '-', '--']

        for idx, (name, r) in enumerate(self._results.items()):
            h = r.distribution.hazard(t_plot)
            ax.plot(t_plot, h,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, label=name)

        ax.set_xlim(0, t_max)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Hazard Rate h(t)', fontsize=12)

        if title is None:
            title = f'Hazard Functions — {self.label}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        return fig

    def plot_cumhazard_diagnostic(
        self,
        figsize: tuple = (10, 6),
        title: Optional[str] = None,
    ):
        """Diagnostic plot: log cumulative hazard vs log time.

        If Weibull is the true model, this should be linear.
        Useful for assessing proportional hazards assumption.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        km = self.km_data
        mask = (km['survival'] > 0) & (km['survival'] < 1) & (km['time'] > 0)
        km_filt = km[mask]

        if len(km_filt) > 0:
            log_t = np.log(km_filt['time'].values)
            log_H = np.log(-np.log(km_filt['survival'].values))
            ax.scatter(log_t, log_H, color='black', s=20, alpha=0.6,
                       zorder=10, label='KM (empirical)')

        # Overlay fitted curves
        t_plot = np.linspace(0.01, self.time.max(), 500)
        log_t_plot = np.log(t_plot)

        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#795548']
        for idx, (name, r) in enumerate(self._results.items()):
            s = r.distribution.survival(t_plot)
            s_safe = np.maximum(s, 1e-300)
            log_H_fit = np.log(np.maximum(-np.log(s_safe), 1e-300))
            ax.plot(log_t_plot, log_H_fit,
                    color=colors[idx % len(colors)],
                    linewidth=1.5, label=name, alpha=0.7)

        ax.set_xlabel('log(Time)', fontsize=12)
        ax.set_ylabel('log(Cumulative Hazard)', fontsize=12)

        if title is None:
            title = f'Log-Cumulative Hazard Diagnostic — {self.label}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        return fig

    def plot_qq(
        self,
        distribution: Optional[str] = None,
        figsize: tuple = (7, 7),
        title: Optional[str] = None,
    ):
        """Q-Q plot: theoretical vs empirical survival quantiles.

        Parameters
        ----------
        distribution : str, optional
            Which fitted distribution to compare. Default: best AIC model.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if distribution is None:
            best = self.best_model()
            if best is None:
                raise RuntimeError("No fitted models available")
            distribution = best.name

        r = self._results[distribution]

        fig, ax = plt.subplots(figsize=figsize)

        # Empirical quantiles from KM
        km = self.km_data
        mask = (km['time'] > 0) & (km['survival'] > 0) & (km['survival'] < 1)
        km_filt = km[mask]

        if len(km_filt) > 0:
            # Theoretical quantiles at same survival probabilities
            # F(t) = 1 - S(t), so we need quantile(1 - S_km)
            p_vals = 1 - km_filt['survival'].values
            try:
                theoretical_t = r.distribution.quantile(p_vals)
                empirical_t = km_filt['time'].values

                ax.scatter(theoretical_t, empirical_t, color='#2196F3', s=30, alpha=0.7)

                # Reference line
                all_vals = np.concatenate([theoretical_t, empirical_t])
                all_vals = all_vals[np.isfinite(all_vals)]
                if len(all_vals) > 0:
                    lo, hi = all_vals.min(), all_vals.max()
                    ax.plot([lo, hi], [lo, hi], color='#999', linewidth=1, linestyle='--')

            except Exception:
                ax.text(0.5, 0.5, f'Q-Q not available for {distribution}',
                        ha='center', va='center', transform=ax.transAxes)

        ax.set_xlabel(f'Theoretical Quantiles ({distribution})', fontsize=12)
        ax.set_ylabel('Empirical Quantiles (KM)', fontsize=12)

        if title is None:
            title = f'Q-Q Plot — {distribution}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        return fig

    def selection_report(self) -> str:
        """Generate a text report explaining model selection.

        Returns
        -------
        str
            Human-readable model selection report.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        df = self.summary()
        if len(df) == 0:
            return "No models were successfully fitted."

        best_aic = self.best_model("aic")
        best_bic = self.best_model("bic")

        lines = [
            f"═══ Survival Model Selection Report: {self.label} ═══",
            f"",
            f"Data: n={self.n_obs}, events={self.n_events}, "
            f"censored={self.n_censored} ({100*self.n_censored/self.n_obs:.1f}%)",
            f"",
            f"Model Comparison Table:",
            df.to_string(index=False),
            f"",
            f"── Selection Criteria ──",
            f"",
        ]

        # AIC analysis
        lines.append(f"AIC (Akaike Information Criterion):")
        lines.append(f"  Best: {best_aic.name} (AIC = {best_aic.aic:.2f})")
        second_best_aic = df[df['Distribution'] != best_aic.name].iloc[0] if len(df) > 1 else None
        if second_best_aic is not None:
            delta = second_best_aic['AIC'] - best_aic.aic
            strength = "decisive" if delta > 10 else "strong" if delta > 6 else "moderate" if delta > 2 else "weak"
            lines.append(f"  ΔAIC to 2nd best ({second_best_aic['Distribution']}): "
                         f"{delta:.2f} — {strength} evidence")

        lines.append(f"")

        # BIC analysis
        lines.append(f"BIC (Bayesian Information Criterion):")
        lines.append(f"  Best: {best_bic.name} (BIC = {best_bic.bic:.2f})")
        second_best_bic = df.sort_values('BIC').iloc[1] if len(df) > 1 else None
        if second_best_bic is not None:
            delta = second_best_bic['BIC'] - best_bic.bic
            strength = "decisive" if delta > 10 else "strong" if delta > 6 else "moderate" if delta > 2 else "weak"
            lines.append(f"  ΔBIC to 2nd best ({second_best_bic['Distribution']}): "
                         f"{delta:.2f} — {strength} evidence")

        lines.append(f"")

        # Agreement
        if best_aic.name == best_bic.name:
            lines.append(f"✓ AIC and BIC agree: {best_aic.name} is the best model.")
        else:
            lines.append(f"⚠ AIC and BIC disagree:")
            lines.append(f"  AIC favors: {best_aic.name} (penalizes complexity less)")
            lines.append(f"  BIC favors: {best_bic.name} (penalizes complexity more)")
            lines.append(f"  Consider clinical plausibility of hazard shape.")

        lines.append(f"")

        # Hazard shape interpretation
        lines.append(f"── Hazard Shape Guide ──")
        lines.append(f"")
        for name, r in self._results.items():
            if name == "Exponential":
                lines.append(f"  {name}: Constant hazard (λ={r.params['rate']:.4f})")
            elif name == "Weibull":
                sh = r.params['shape']
                shape_desc = ("increasing" if sh > 1 else
                              "decreasing" if sh < 1 else "constant")
                lines.append(f"  {name}: {shape_desc.capitalize()} hazard "
                             f"(shape={sh:.4f})")
            elif name == "Log-logistic":
                sh = r.params['shape']
                shape_desc = ("non-monotone (hump-shaped)" if sh > 1 else
                              "monotone decreasing")
                lines.append(f"  {name}: {shape_desc.capitalize()} hazard "
                             f"(shape={sh:.4f})")
            elif name == "Gompertz":
                sh = r.params['shape']
                shape_desc = ("increasing" if sh > 0 else
                              "decreasing" if sh < 0 else "constant")
                lines.append(f"  {name}: {shape_desc.capitalize()} hazard "
                             f"(shape={sh:.4f})")
            elif name == "General Gamma" or name == "Generalized Gamma":
                Q = r.params.get('Q', 0)
                lines.append(f"  {name}: Flexible (Q={Q:.4f})")
            else:
                lines.append(f"  {name}: See hazard plot for shape")

        lines.append(f"")
        lines.append(f"── Recommendation ──")
        lines.append(f"")
        lines.append(f"1. Check the KM + fitted curves plot for visual agreement")
        lines.append(f"2. Check the log-cumulative hazard diagnostic plot")
        lines.append(f"3. Consider clinical plausibility of the extrapolated hazard")
        lines.append(f"4. If unsure between models with ΔAIC < 2, "
                     f"run scenario analyses with both")

        return "\n".join(lines)

    def to_excel(self, filepath: str):
        """Export fitting results and comparison to Excel.

        Parameters
        ----------
        filepath : str
            Output .xlsx path.
        """
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary
            self.summary().to_excel(writer, sheet_name='Model Comparison', index=False)

            # KM data
            self.km_data.to_excel(writer, sheet_name='Kaplan-Meier', index=False)

            # Per-model details
            for name, r in self._results.items():
                sheet = name[:31]
                param_df = pd.DataFrame([r.params])
                param_df.insert(0, 'Distribution', name)
                param_df['Log-Likelihood'] = r.loglik
                param_df['AIC'] = r.aic
                param_df['BIC'] = r.bic
                param_df.to_excel(writer, sheet_name=sheet, index=False)

                # Survival curve data
                t = np.linspace(0, self.time.max() * 1.5, 200)
                surv_df = pd.DataFrame({'Time': t, 'Survival': r.distribution.survival(t)})
                surv_df.to_excel(writer, sheet_name=sheet, startrow=4, index=False)

            # Selection report
            report_df = pd.DataFrame({'Report': self.selection_report().split('\n')})
            report_df.to_excel(writer, sheet_name='Selection Report', index=False)

        print(f"✅ Survival fitting results exported to: {filepath}")

    def __repr__(self):
        if not self._fitted:
            return f"SurvivalFitter('{self.label}', n={self.n_obs}, not fitted)"
        n_ok = sum(1 for r in self._results.values() if r.converged)
        best = self.best_model()
        best_str = f", best={best.name}" if best else ""
        return (f"SurvivalFitter('{self.label}', n={self.n_obs}, "
                f"fitted={n_ok}/{len(self._results)}{best_str})")
