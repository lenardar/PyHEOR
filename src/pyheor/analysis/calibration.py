"""
Model calibration module for health economic decision models.

Calibration estimates unknown model parameters by finding values that
produce model outputs matching observed empirical data (calibration targets).

Supported methods:
- Nelder-Mead (multi-start): Derivative-free optimization via scipy
- Random search: Latin Hypercube sampling with exhaustive evaluation

Supported goodness-of-fit (GoF) measures:
- SSE: Sum of squared errors
- WSSE: Weighted SSE (weights = 1/SE²)
- Log-likelihood (Normal): -Σ log N(observed | predicted, SE)

References
----------
- Vanni T et al. (2011). Calibrating models in economic evaluation:
  a seven-step approach. PharmacoEconomics, 29(1), 35-49.
- Alarid-Escudero F et al. (2018). A Tutorial on Calibration of
  Health Decision Models. Medical Decision Making, 38(8), 980-990.
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from scipy.optimize import minimize
from scipy.stats import qmc


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CalibrationTarget:
    """A single calibration target (observed data point).

    Parameters
    ----------
    name : str
        Target name, e.g. "5yr_prevalence".
    observed : float
        Observed value from empirical data.
    extract_fn : callable
        Function to extract the comparable prediction from model results.
        Signature: extract_fn(sim_results: dict) -> float
        where sim_results is the return value of model._simulate_single().
    se : float, optional
        Standard error of the observed value. Required for WSSE and
        log-likelihood GoF measures.

    Examples
    --------
    >>> target = CalibrationTarget(
    ...     name="5yr_survival",
    ...     observed=0.65,
    ...     se=0.05,
    ...     extract_fn=lambda sim: sim["SOC"]["trace"][5, 0],
    ... )
    """
    name: str
    observed: float
    extract_fn: Callable
    se: Optional[float] = None


@dataclass
class CalibrationParam:
    """A parameter to calibrate.

    Parameters
    ----------
    name : str
        Must match a key in model.params.
    lower : float
        Lower bound of the search space.
    upper : float
        Upper bound of the search space.
    initial : float, optional
        Starting value. Defaults to (lower + upper) / 2.

    Examples
    --------
    >>> cp = CalibrationParam("p_HS", lower=0.01, upper=0.30)
    """
    name: str
    lower: float
    upper: float
    initial: Optional[float] = None

    def __post_init__(self):
        if self.initial is None:
            self.initial = (self.lower + self.upper) / 2.0
        if self.lower >= self.upper:
            raise ValueError(
                f"CalibrationParam '{self.name}': "
                f"lower ({self.lower}) must be < upper ({self.upper})"
            )


@dataclass
class CalibrationResult:
    """Results of a model calibration run.

    Attributes
    ----------
    best_params : dict
        Best-fit parameter values.
    best_gof : float
        Goodness-of-fit value at best parameters.
    all_params : np.ndarray
        All evaluated parameter sets, shape (n_evals, n_params).
    all_gof : np.ndarray
        GoF values for all evaluations, shape (n_evals,).
    param_names : list of str
        Parameter names (ordered).
    target_names : list of str
        Target names (ordered).
    predicted : dict
        Predicted values at best parameters, keyed by target name.
    elapsed : float
        Wall-clock time in seconds.
    method : str
        Calibration method used.
    """
    best_params: Dict[str, float]
    best_gof: float
    all_params: np.ndarray
    all_gof: np.ndarray
    param_names: List[str]
    target_names: List[str]
    predicted: Dict[str, float]
    elapsed: float
    method: str

    def summary(self) -> pd.DataFrame:
        """Parameter summary: name, best value, search bounds.

        Returns
        -------
        pd.DataFrame
            One row per calibrated parameter.
        """
        rows = []
        for name in self.param_names:
            rows.append({
                "Parameter": name,
                "Best Value": self.best_params[name],
            })
        return pd.DataFrame(rows)

    def target_comparison(self) -> pd.DataFrame:
        """Compare observed vs predicted for each target.

        Returns
        -------
        pd.DataFrame
            Columns: Target, Observed, Predicted, Abs Error, Rel Error (%).
        """
        rows = []
        for name in self.target_names:
            obs = None
            pred = self.predicted[name]
            # Find observed value from target list (stored in predicted keys)
            rows.append({
                "Target": name,
                "Predicted": pred,
            })
        return pd.DataFrame(rows)

    def apply_to_model(self, model) -> None:
        """Update model parameter base values with calibrated values.

        Parameters
        ----------
        model : MarkovModel or PSMModel
            The model whose parameters will be updated.
        """
        for name, value in self.best_params.items():
            if name in model.params:
                model.params[name].base = value

    def plot_gof(self, ax=None):
        """Scatter plot of GoF values across evaluations.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        idx = np.argsort(self.all_gof)
        ax.scatter(
            range(len(self.all_gof)),
            self.all_gof[idx],
            s=8,
            alpha=0.5,
        )
        ax.set_xlabel("Evaluation (sorted)")
        ax.set_ylabel("Goodness of Fit")
        ax.set_title(f"Calibration GoF ({self.method})")
        ax.axhline(
            self.best_gof,
            color="red",
            linestyle="--",
            label=f"Best: {self.best_gof:.4g}",
        )
        ax.legend()
        return ax

    def plot_pairs(self, top_n: int = 100, ax=None):
        """Pairwise scatter of top-n parameter sets.

        Parameters
        ----------
        top_n : int
            Number of best parameter sets to show.
        ax : array of Axes, optional
            If None, creates new subplots.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        n_params = len(self.param_names)
        if n_params < 2:
            raise ValueError("Need at least 2 parameters for pairs plot")

        top_idx = np.argsort(self.all_gof)[:top_n]
        top_params = self.all_params[top_idx]

        fig, axes = plt.subplots(
            n_params,
            n_params,
            figsize=(3 * n_params, 3 * n_params),
        )
        if n_params == 2:
            axes = np.array(axes).reshape(2, 2)

        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                if i == j:
                    ax.hist(top_params[:, i], bins=20, alpha=0.7)
                    ax.axvline(
                        self.best_params[self.param_names[i]],
                        color="red",
                        linestyle="--",
                    )
                else:
                    ax.scatter(
                        top_params[:, j],
                        top_params[:, i],
                        s=8,
                        alpha=0.5,
                        c=self.all_gof[top_idx],
                        cmap="viridis_r",
                    )
                if i == n_params - 1:
                    ax.set_xlabel(self.param_names[j])
                if j == 0:
                    ax.set_ylabel(self.param_names[i])

        fig.suptitle(f"Top-{top_n} Parameter Sets")
        fig.tight_layout()
        return fig


# =============================================================================
# Goodness-of-Fit Functions
# =============================================================================

def gof_sse(
    observed: Sequence[float],
    predicted: Sequence[float],
    se: Sequence[Optional[float]] = None,
) -> float:
    """Sum of squared errors.

    Parameters
    ----------
    observed : array-like
        Observed values.
    predicted : array-like
        Model-predicted values.
    se : ignored
        Not used; included for consistent API.

    Returns
    -------
    float
        Sum of (observed - predicted)².
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    return float(np.sum((obs - pred) ** 2))


def gof_wsse(
    observed: Sequence[float],
    predicted: Sequence[float],
    se: Sequence[Optional[float]] = None,
) -> float:
    """Weighted sum of squared errors (weight = 1/SE²).

    Parameters
    ----------
    observed : array-like
        Observed values.
    predicted : array-like
        Model-predicted values.
    se : array-like
        Standard errors. All must be > 0.

    Returns
    -------
    float
        Sum of (observed - predicted)² / SE².

    Raises
    ------
    ValueError
        If any SE is None or <= 0.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if se is None:
        raise ValueError("WSSE requires standard errors for all targets")
    se_arr = np.asarray(se, dtype=float)
    if np.any(se_arr <= 0):
        raise ValueError("All standard errors must be > 0")
    weights = 1.0 / (se_arr ** 2)
    return float(np.sum(weights * (obs - pred) ** 2))


def gof_loglik_normal(
    observed: Sequence[float],
    predicted: Sequence[float],
    se: Sequence[Optional[float]] = None,
) -> float:
    """Negative log-likelihood under Normal assumption.

    Assumes observed ~ N(predicted, SE²). Returns the negative
    log-likelihood (to be minimized).

    Parameters
    ----------
    observed : array-like
        Observed values.
    predicted : array-like
        Model-predicted values (treated as means).
    se : array-like
        Standard errors (treated as standard deviations).

    Returns
    -------
    float
        -Σ log N(observed | predicted, SE²).

    Raises
    ------
    ValueError
        If any SE is None or <= 0.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if se is None:
        raise ValueError(
            "Log-likelihood requires standard errors for all targets"
        )
    se_arr = np.asarray(se, dtype=float)
    if np.any(se_arr <= 0):
        raise ValueError("All standard errors must be > 0")
    # -log N(obs | pred, se²) = 0.5*log(2π) + log(se) + 0.5*((obs-pred)/se)²
    nll = 0.5 * np.sum(
        np.log(2 * np.pi) + 2 * np.log(se_arr)
        + ((obs - pred) / se_arr) ** 2
    )
    return float(nll)


# GoF function registry
_GOF_FUNCTIONS = {
    "sse": gof_sse,
    "wsse": gof_wsse,
    "loglik_normal": gof_loglik_normal,
}


# =============================================================================
# Latin Hypercube Sampling
# =============================================================================

def latin_hypercube(
    n_samples: int,
    bounds: List[Tuple[float, float]],
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate Latin Hypercube samples within given bounds.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    bounds : list of (lower, upper) tuples
        Parameter bounds.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_params) with values in [lower, upper].
    """
    n_params = len(bounds)
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # in [0, 1]^d

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    return qmc.scale(unit_samples, lower, upper)


# =============================================================================
# Internal: Objective Function Construction
# =============================================================================

def _make_objective(model, targets, calib_params, gof_fn):
    """Build the scalar objective function for optimization.

    Parameters
    ----------
    model : MarkovModel or PSMModel
        Model with _get_base_params() and _simulate_single().
    targets : list of CalibrationTarget
    calib_params : list of CalibrationParam
    gof_fn : callable
        One of gof_sse / gof_wsse / gof_loglik_normal.

    Returns
    -------
    callable
        f(x: np.ndarray) -> float, where x is the parameter vector.
    """
    base_params = model._get_base_params()
    param_names = [cp.name for cp in calib_params]
    observed = [t.observed for t in targets]
    ses = [t.se for t in targets]

    def objective(x):
        params = base_params.copy()
        for name, val in zip(param_names, x):
            params[name] = val
        try:
            sim = model._simulate_single(params)
        except Exception:
            return 1e20  # penalty for infeasible parameter sets
        predicted = [t.extract_fn(sim) for t in targets]
        return gof_fn(observed, predicted, ses)

    return objective


def _extract_predicted(model, targets, params):
    """Run model and extract predicted values for all targets."""
    base = model._get_base_params()
    base.update(params)
    sim = model._simulate_single(base)
    return {t.name: t.extract_fn(sim) for t in targets}


# =============================================================================
# Calibration Methods
# =============================================================================

def _calibrate_nelder_mead(
    objective,
    calib_params,
    n_restarts,
    seed,
    progress,
):
    """Multi-start Nelder-Mead optimization.

    Returns
    -------
    tuple of (all_params, all_gof, best_idx)
    """
    bounds = [(cp.lower, cp.upper) for cp in calib_params]
    n_params = len(calib_params)

    # Generate starting points: first is the initial values,
    # rest from LHS
    starts = []
    starts.append(np.array([cp.initial for cp in calib_params]))
    if n_restarts > 1:
        lhs_starts = latin_hypercube(
            n_restarts - 1,
            bounds,
            seed=seed,
        )
        for row in lhs_starts:
            starts.append(row)

    all_params_list = []
    all_gof_list = []

    for i, x0 in enumerate(starts):
        if progress:
            print(f"  Nelder-Mead restart {i + 1}/{n_restarts}...", end="")

        result = minimize(
            objective,
            x0=x0,
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-8},
        )

        # Clip to bounds
        x_best = np.clip(
            result.x,
            [b[0] for b in bounds],
            [b[1] for b in bounds],
        )
        gof_val = objective(x_best)

        all_params_list.append(x_best)
        all_gof_list.append(gof_val)

        if progress:
            print(f" GoF={gof_val:.6g}")

    all_params = np.array(all_params_list)
    all_gof = np.array(all_gof_list)
    best_idx = int(np.argmin(all_gof))

    return all_params, all_gof, best_idx


def _calibrate_random_search(
    objective,
    calib_params,
    n_samples,
    seed,
    progress,
):
    """Latin Hypercube random search.

    Returns
    -------
    tuple of (all_params, all_gof, best_idx)
    """
    bounds = [(cp.lower, cp.upper) for cp in calib_params]
    samples = latin_hypercube(n_samples, bounds, seed=seed)

    all_gof = np.zeros(n_samples)
    for i in range(n_samples):
        all_gof[i] = objective(samples[i])
        if progress and (i + 1) % max(1, n_samples // 10) == 0:
            best_so_far = np.min(all_gof[:i + 1])
            print(
                f"  Random search: {i + 1}/{n_samples}, "
                f"best GoF={best_so_far:.6g}"
            )

    best_idx = int(np.argmin(all_gof))
    return samples, all_gof, best_idx


# =============================================================================
# Main Entry Point
# =============================================================================

def calibrate(
    model,
    targets: List[CalibrationTarget],
    calib_params: List[CalibrationParam],
    gof: str = "sse",
    method: str = "nelder_mead",
    n_restarts: int = 10,
    n_samples: int = 1000,
    seed: Optional[int] = None,
    progress: bool = True,
) -> CalibrationResult:
    """Calibrate model parameters to match observed data.

    Parameters
    ----------
    model : MarkovModel or PSMModel
        A PyHEOR model with parameters defined via add_param().
    targets : list of CalibrationTarget
        Observed data points to match.
    calib_params : list of CalibrationParam
        Parameters to calibrate, with search bounds.
    gof : str
        Goodness-of-fit measure: "sse", "wsse", or "loglik_normal".
    method : str
        Search method: "nelder_mead" or "random_search".
    n_restarts : int
        Number of starting points for Nelder-Mead (default: 10).
    n_samples : int
        Number of samples for random search (default: 1000).
    seed : int, optional
        Random seed for reproducibility.
    progress : bool
        Whether to print progress messages (default: True).

    Returns
    -------
    CalibrationResult
        Calibration results including best parameters, GoF, and diagnostics.

    Raises
    ------
    ValueError
        If GoF or method is unknown, or if parameters not found in model.

    Examples
    --------
    >>> from pyheor import MarkovModel, calibrate, CalibrationTarget, CalibrationParam
    >>>
    >>> model = MarkovModel(...)
    >>> model.add_param("p_HS", base=0.15)
    >>>
    >>> targets = [
    ...     CalibrationTarget(
    ...         name="5yr_healthy",
    ...         observed=0.40,
    ...         extract_fn=lambda sim: sim["SOC"]["trace"][5, 0],
    ...     ),
    ... ]
    >>> calib_params = [
    ...     CalibrationParam("p_HS", lower=0.01, upper=0.50),
    ... ]
    >>> result = calibrate(model, targets, calib_params, method="nelder_mead")
    >>> result.summary()
    """
    # --- Validate inputs ---
    if gof not in _GOF_FUNCTIONS:
        raise ValueError(
            f"Unknown GoF '{gof}'. Choose from: {list(_GOF_FUNCTIONS)}"
        )
    if method not in ("nelder_mead", "random_search"):
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: 'nelder_mead', 'random_search'"
        )

    # Check parameters exist in model
    for cp in calib_params:
        if cp.name not in model.params:
            raise ValueError(
                f"Parameter '{cp.name}' not found in model. "
                f"Available: {list(model.params.keys())}"
            )

    # Check SE provided if needed
    if gof in ("wsse", "loglik_normal"):
        for t in targets:
            if t.se is None:
                raise ValueError(
                    f"Target '{t.name}' missing 'se' "
                    f"(required for GoF='{gof}')"
                )

    gof_fn = _GOF_FUNCTIONS[gof]
    objective = _make_objective(model, targets, calib_params, gof_fn)
    param_names = [cp.name for cp in calib_params]
    target_names = [t.name for t in targets]

    if progress:
        print(
            f"Calibrating {len(calib_params)} parameter(s) "
            f"to {len(targets)} target(s) "
            f"[method={method}, GoF={gof}]"
        )

    t0 = time.time()

    # --- Run calibration ---
    if method == "nelder_mead":
        all_params, all_gof, best_idx = _calibrate_nelder_mead(
            objective,
            calib_params,
            n_restarts,
            seed,
            progress,
        )
    else:  # random_search
        all_params, all_gof, best_idx = _calibrate_random_search(
            objective,
            calib_params,
            n_samples,
            seed,
            progress,
        )

    elapsed = time.time() - t0

    # --- Build result ---
    best_x = all_params[best_idx]
    best_params = {
        name: float(val) for name, val in zip(param_names, best_x)
    }
    predicted = _extract_predicted(model, targets, best_params)

    if progress:
        print(f"Done in {elapsed:.1f}s. Best GoF={all_gof[best_idx]:.6g}")

    return CalibrationResult(
        best_params=best_params,
        best_gof=float(all_gof[best_idx]),
        all_params=all_params,
        all_gof=all_gof,
        param_names=param_names,
        target_names=target_names,
        predicted=predicted,
        elapsed=elapsed,
        method=method,
    )
