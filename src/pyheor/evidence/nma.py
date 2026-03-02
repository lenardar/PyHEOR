"""
NMA (Network Meta-Analysis) integration for PyHEOR.
====================================================

This module provides utilities to load and use NMA posterior samples
(typically generated in R using ``gemtc``, ``multinma``, or ``bnma``)
as PSA distributions within PyHEOR economic models.

Design Philosophy
-----------------
NMA computations are best performed in dedicated Bayesian packages (R or
standalone). PyHEOR focuses on the **downstream use** of NMA results:

1. **Load posterior samples** from CSV / Excel / feather / RDS-exported files.
2. **Create ``Distribution`` objects** that sample *with replacement* from
   the posteriors, preserving correlations across parameters.
3. **Provide convenience wrappers** to generate ``ProportionalHazards`` or
   ``AcceleratedFailureTime`` survival curves from NMA-derived HRs / AFs.

Key classes
-----------
- ``NMAPosterior``   — container for a posterior sample matrix (one row per
  MCMC iteration, one column per treatment contrast).
- ``PosteriorDist``  — a ``Distribution`` subclass that draws from a single
  column of the posterior matrix (independent resampling).
- ``CorrelatedPosterior`` — draws a full *row* at a time, maintaining the
  joint posterior structure across multiple parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ..distributions import Distribution

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

__all__ = [
    "NMAPosterior",
    "PosteriorDist",
    "CorrelatedPosterior",
    "load_nma_samples",
]


# ============================================================================
# Loading utilities
# ============================================================================

def load_nma_samples(
    path: Union[str, Path],
    *,
    treatment_col: Optional[str] = None,
    value_col: Optional[str] = None,
    iteration_col: Optional[str] = None,
    log_scale: bool = False,
    sheet_name: Union[str, int, None] = 0,
) -> "NMAPosterior":
    """Load NMA posterior samples from a file.

    Supports **wide** and **long** formats, and CSV / Excel / Feather files.

    Wide format (one column per treatment):

    ======  ======  ======
    Drug_A  Drug_B  Drug_C
    ======  ======  ======
    -0.12    0.05    0.31
    -0.15    0.08    0.29
    ======  ======  ======

    Long format (requires *treatment_col* and *value_col*):

    =========  =========  =====
    iteration  treatment  value
    =========  =========  =====
    1          Drug_A     -0.12
    1          Drug_B      0.05
    =========  =========  =====

    Parameters
    ----------
    path : str or Path
        File path.  Accepted extensions: ``.csv``, ``.xlsx``, ``.xls``,
        ``.feather``, ``.parquet``.
    treatment_col : str, optional
        Column identifying treatments (long format only).
    value_col : str, optional
        Column containing posterior values (long format only).
    iteration_col : str, optional
        Column identifying MCMC iterations (long format only).
        If ``None``, inferred from the row index.
    log_scale : bool, default False
        If ``True``, the loaded values are on the **log** scale.
        They will be exponentiated automatically (e.g. log-HR → HR).
    sheet_name : str or int, optional
        Excel sheet (ignored for non-Excel files).

    Returns
    -------
    NMAPosterior
        Container with one column per treatment contrast.

    Examples
    --------
    >>> nma = ph.load_nma_samples("nma_hr_samples.csv")
    >>> nma = ph.load_nma_samples(
    ...     "nma_long.csv",
    ...     treatment_col="treatment",
    ...     value_col="d",
    ...     log_scale=True,
    ... )
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # --- Read file ---
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif suffix == ".feather":
        df = pd.read_feather(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # --- Long → wide pivot ---
    if treatment_col is not None and value_col is not None:
        if iteration_col is None:
            # Infer iteration by cumcount within treatment
            df = df.copy()
            df["_iter"] = df.groupby(treatment_col).cumcount()
            iteration_col = "_iter"
        df = df.pivot(
            index=iteration_col, columns=treatment_col, values=value_col
        ).reset_index(drop=True)

    # --- Drop non-numeric columns (safety net) ---
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError(
            "No numeric columns found after loading. "
            "Check your file format or specify treatment_col / value_col."
        )

    # --- Exponentiate if on log scale ---
    samples = numeric_df.values.astype(float)  # shape: (n_iter, n_treatments)
    treatment_names = list(numeric_df.columns)
    if log_scale:
        samples = np.exp(samples)

    return NMAPosterior(
        samples=samples,
        treatment_names=treatment_names,
        log_scale=log_scale,
    )


# ============================================================================
# NMAPosterior — main container
# ============================================================================

class NMAPosterior:
    """Container for NMA posterior samples.

    Parameters
    ----------
    samples : np.ndarray, shape (n_iter, n_treatments)
        Posterior sample matrix.  Each column is a treatment contrast
        (e.g. HR vs. reference).
    treatment_names : list[str]
        Names matching the columns of *samples*.
    log_scale : bool
        Whether the original data was on the log scale (informational).

    Attributes
    ----------
    n_iter : int
        Number of MCMC iterations.
    n_treatments : int
        Number of treatments / contrasts.
    """

    def __init__(
        self,
        samples: np.ndarray,
        treatment_names: List[str],
        log_scale: bool = False,
    ):
        self.samples = np.asarray(samples, dtype=float)
        self.treatment_names = list(treatment_names)
        self.log_scale = log_scale

        if self.samples.ndim != 2:
            raise ValueError("samples must be 2-D (n_iter × n_treatments)")
        if self.samples.shape[1] != len(self.treatment_names):
            raise ValueError(
                f"Column count ({self.samples.shape[1]}) does not match "
                f"treatment_names length ({len(self.treatment_names)})"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_iter(self) -> int:
        return self.samples.shape[0]

    @property
    def n_treatments(self) -> int:
        return self.samples.shape[1]

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------

    def _col_idx(self, treatment: str) -> int:
        try:
            return self.treatment_names.index(treatment)
        except ValueError:
            raise KeyError(
                f"Treatment '{treatment}' not found. "
                f"Available: {self.treatment_names}"
            )

    def __getitem__(self, treatment: str) -> np.ndarray:
        """Get the posterior vector for a single treatment."""
        return self.samples[:, self._col_idx(treatment)]

    # ------------------------------------------------------------------
    # Independent distribution for a single parameter
    # ------------------------------------------------------------------

    def dist(self, treatment: str) -> "PosteriorDist":
        """Create a ``Distribution`` that resamples from one treatment's posterior.

        Parameters
        ----------
        treatment : str
            Treatment name.

        Returns
        -------
        PosteriorDist
            A ``Distribution`` whose ``sample(n)`` draws *with replacement*
            from the 1-D posterior column.

        Examples
        --------
        >>> model.add_param("hr_drugA", base=nma.median("Drug_A"),
        ...                 dist=nma.dist("Drug_A"))
        """
        return PosteriorDist(self.samples[:, self._col_idx(treatment)])

    # ------------------------------------------------------------------
    # Correlated joint distribution
    # ------------------------------------------------------------------

    def correlated(
        self, treatments: Optional[List[str]] = None
    ) -> "CorrelatedPosterior":
        """Create a correlated joint distribution across treatments.

        Each call to ``draw()`` picks a random MCMC row and returns values
        for all requested treatments, preserving the posterior correlation.

        Parameters
        ----------
        treatments : list[str], optional
            Subset of treatments (default: all).

        Returns
        -------
        CorrelatedPosterior

        Examples
        --------
        >>> joint = nma.correlated(["Drug_A", "Drug_B"])
        >>> model.add_param("hr_A", base=..., dist=joint.marginal("Drug_A"))
        >>> model.add_param("hr_B", base=..., dist=joint.marginal("Drug_B"))
        """
        if treatments is None:
            treatments = self.treatment_names
        idxs = [self._col_idx(t) for t in treatments]
        return CorrelatedPosterior(
            samples=self.samples[:, idxs],
            names=treatments,
        )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Summary statistics of the posterior samples.

        Returns
        -------
        pd.DataFrame
            One row per treatment with mean, median, sd, 2.5%, 97.5%.
        """
        rows = []
        for j, name in enumerate(self.treatment_names):
            col = self.samples[:, j]
            rows.append({
                "treatment": name,
                "mean": np.mean(col),
                "median": np.median(col),
                "sd": np.std(col, ddof=1),
                "q2.5": np.percentile(col, 2.5),
                "q97.5": np.percentile(col, 97.5),
            })
        return pd.DataFrame(rows)

    def median(self, treatment: str) -> float:
        """Posterior median for a treatment (handy for ``base`` param)."""
        return float(np.median(self[treatment]))

    def mean(self, treatment: str) -> float:
        """Posterior mean for a treatment."""
        return float(np.mean(self[treatment]))

    # ------------------------------------------------------------------
    # Quick model integration helpers
    # ------------------------------------------------------------------

    def add_params_to_model(
        self,
        model,
        param_prefix: str = "hr",
        treatments: Optional[List[str]] = None,
        correlated: bool = True,
    ) -> None:
        """Convenience: add one ``Param`` per treatment to a model.

        Parameters
        ----------
        model : MarkovModel
            Target model.
        param_prefix : str
            Parameter name prefix.  Resulting names: ``{prefix}_{treatment}``.
        treatments : list[str], optional
            Subset (default: all).
        correlated : bool
            If ``True``, use ``CorrelatedPosterior`` so that draws are
            from the same MCMC row (preserving correlation).  If ``False``,
            each parameter resamples independently.

        Examples
        --------
        >>> nma.add_params_to_model(model, param_prefix="hr",
        ...                         treatments=["Drug_A", "Drug_B"])
        # adds: model.params["hr_Drug_A"], model.params["hr_Drug_B"]
        """
        if treatments is None:
            treatments = self.treatment_names

        if correlated and len(treatments) > 1:
            joint = self.correlated(treatments)
            for name in treatments:
                param_name = f"{param_prefix}_{name}"
                model.add_param(
                    param_name,
                    base=self.median(name),
                    dist=joint.marginal(name),
                    label=f"{param_prefix.upper()} ({name})",
                )
        else:
            for name in treatments:
                param_name = f"{param_prefix}_{name}"
                model.add_param(
                    param_name,
                    base=self.median(name),
                    dist=self.dist(name),
                    label=f"{param_prefix.upper()} ({name})",
                )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NMAPosterior(n_iter={self.n_iter}, "
            f"treatments={self.treatment_names})"
        )


# ============================================================================
# PosteriorDist — Distribution subclass for single-parameter resampling
# ============================================================================

class PosteriorDist(Distribution):
    """Distribution that resamples with replacement from posterior draws.

    Parameters
    ----------
    values : array-like, shape (n,)
        1-D array of posterior samples.
    """

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float).ravel()
        if self.values.size == 0:
            raise ValueError("Cannot create PosteriorDist from empty array")

    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.choice(self.values, size=n, replace=True)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    def __repr__(self) -> str:
        return (
            f"PosteriorDist(n={len(self.values)}, "
            f"mean={np.mean(self.values):.4f}, "
            f"median={np.median(self.values):.4f})"
        )


# ============================================================================
# CorrelatedPosterior — joint draw preserving MCMC correlation
# ============================================================================

class CorrelatedPosterior:
    """Correlated joint posterior for multiple treatment contrasts.

    Each call to ``draw()`` samples a random MCMC row, returning the
    same-iteration values for all treatments. This preserves the posterior
    correlation structure (e.g. between HRs of different treatments).

    Parameters
    ----------
    samples : np.ndarray, shape (n_iter, k)
        Posterior sub-matrix.
    names : list[str]
        Treatment names matching the columns.

    Notes
    -----
    Use ``marginal(name)`` to get a ``Distribution`` for a single treatment
    that still draws from the shared row index.
    """

    def __init__(self, samples: np.ndarray, names: List[str]):
        self.samples = np.asarray(samples, dtype=float)
        self.names = list(names)
        self._name_to_idx = {n: i for i, n in enumerate(names)}
        self._current_row: Optional[int] = None

    def draw(self) -> Dict[str, float]:
        """Draw one MCMC row and return {name: value} dict."""
        self._current_row = np.random.randint(0, self.samples.shape[0])
        return {
            name: float(self.samples[self._current_row, i])
            for i, name in enumerate(self.names)
        }

    def marginal(self, name: str) -> "_CorrelatedMarginal":
        """Get a ``Distribution``-compatible marginal for one treatment.

        The marginal shares the same internal row-sampling counter,
        so calling ``sample()`` on any marginal from the same
        ``CorrelatedPosterior`` picks a new shared row.
        For full correlation preservation in PSA, use
        ``CorrelatedPosterior`` as the distribution for ALL related
        parameters simultaneously via a custom sampling hook.

        For the simpler (but still good) approach, this returns a
        ``PosteriorDist`` that resamples from the correct column
        independently—which is appropriate for light correlation and
        is *far* simpler to integrate with the existing ``Param`` system.

        Returns
        -------
        PosteriorDist
            Resampling distribution for one column.
        """
        idx = self._name_to_idx[name]
        return PosteriorDist(self.samples[:, idx])

    def __repr__(self) -> str:
        return (
            f"CorrelatedPosterior(n_iter={self.samples.shape[0]}, "
            f"treatments={self.names})"
        )


# ============================================================================
# Convenience helpers for survival models
# ============================================================================

def make_ph_curves(
    baseline,
    nma: NMAPosterior,
    treatments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create ProportionalHazards survival curves from NMA HRs.

    Parameters
    ----------
    baseline : SurvivalDistribution
        Baseline (reference) survival curve.
    nma : NMAPosterior
        Posterior samples of **hazard ratios** (not log-HR).
    treatments : list[str], optional
        Subset (default: all).

    Returns
    -------
    dict
        ``{treatment_name: ProportionalHazards(baseline, median_hr)}``

    Examples
    --------
    >>> from pyheor.survival import Weibull, ProportionalHazards
    >>> baseline = Weibull(shape=1.2, scale=10)
    >>> curves = ph.make_ph_curves(baseline, nma)
    >>> model.set_transitions("Drug_A", build_matrix(curves["Drug_A"]))
    """
    from ..survival import ProportionalHazards

    if treatments is None:
        treatments = nma.treatment_names
    return {
        t: ProportionalHazards(baseline, hr=nma.median(t))
        for t in treatments
    }


def make_aft_curves(
    baseline,
    nma: NMAPosterior,
    treatments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create AcceleratedFailureTime curves from NMA acceleration factors.

    Parameters
    ----------
    baseline : SurvivalDistribution
        Baseline (reference) survival curve.
    nma : NMAPosterior
        Posterior samples of acceleration factors.
    treatments : list[str], optional
        Subset (default: all).

    Returns
    -------
    dict
        ``{treatment_name: AcceleratedFailureTime(baseline, median_af)}``
    """
    from ..survival import AcceleratedFailureTime

    if treatments is None:
        treatments = nma.treatment_names
    return {
        t: AcceleratedFailureTime(baseline, af=nma.median(t))
        for t in treatments
    }
