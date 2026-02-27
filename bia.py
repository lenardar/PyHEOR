"""
bia.py — Budget Impact Analysis (BIA)
======================================

Provides:
  - BudgetImpactAnalysis: estimate the financial consequences of
    adopting a new health technology on a healthcare budget.

Follows ISPOR BIA Good Practices guidelines:
  - Sullivan et al., "Budget Impact Analysis — Principles of Good Practice" (2014)
  - Mauskopf et al., "Principles of Good Practice for BIA" (2007)

Supports:
  - Flexible population models (constant, compound growth, linear growth, explicit)
  - Time-varying market shares with linear / sigmoid uptake curves
  - Time-varying per-patient costs with optional inflation
  - Scenario analysis and one-way sensitivity
  - Publication-ready visualizations
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Sequence, Tuple, Union


# ───────────────────────────────────────────────────────────────────────────
# Budget Impact Analysis
# ───────────────────────────────────────────────────────────────────────────

class BudgetImpactAnalysis:
    """
    Budget Impact Analysis (BIA).

    Estimates the financial impact of adopting a new health technology
    on a healthcare payer's budget over a short-term horizon
    (typically 1–5 years).

    Parameters
    ----------
    strategies : sequence of str
        Treatment strategy names (all strategies in the market).
    per_patient_costs : dict
        Annual per-patient costs by strategy.

        - ``{strategy: float}`` — constant across all years
        - ``{strategy: list[float]}`` — one value per year (length = *time_horizon*)
    population : int, float, list, or dict
        Eligible patient population each year.

        - ``int / float`` — constant every year
        - ``list`` — explicit values (length = *time_horizon*)
        - ``{"base": N, "growth_rate": r}`` — compound growth  N × (1 + r)^t
        - ``{"base": N, "annual_increase": k}`` — linear growth  N + k × t
    market_share_current : dict
        Market shares under the **current** (without new technology) scenario.

        - ``{strategy: float}`` — constant
        - ``{strategy: list[float]}`` — per-year (length = *time_horizon*)
    market_share_new : dict
        Market shares under the **new** (with new technology adoption) scenario.
        Same format as *market_share_current*.
    time_horizon : int, default 5
        Number of years.
    strategy_labels : dict, optional
        ``{strategy: display_label}``.  Defaults to strategy names.
    cost_inflation : float, default 0.0
        Annual cost inflation rate (applied cumulatively on top of specified costs).
    discount_rate : float, default 0.0
        Annual discount rate for budget values (BIA often uses 0).

    Examples
    --------
    >>> import pyheor as ph
    >>> bia = ph.BudgetImpactAnalysis(
    ...     strategies=["Drug A", "Drug B", "Drug C"],
    ...     per_patient_costs={"Drug A": 5000, "Drug B": 12000, "Drug C": 8000},
    ...     population=10000,
    ...     market_share_current={"Drug A": 0.6, "Drug B": 0.1, "Drug C": 0.3},
    ...     market_share_new={"Drug A": 0.4, "Drug B": 0.3, "Drug C": 0.3},
    ...     time_horizon=5,
    ... )
    >>> print(bia.summary())
    >>> bia.plot_budget_impact()
    """

    # -------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------

    def __init__(
        self,
        strategies: Sequence[str],
        per_patient_costs: Dict[str, Union[float, List[float]]],
        population: Union[int, float, List[float], Dict],
        market_share_current: Dict[str, Union[float, List[float]]],
        market_share_new: Dict[str, Union[float, List[float]]],
        time_horizon: int = 5,
        *,
        strategy_labels: Optional[Dict[str, str]] = None,
        cost_inflation: float = 0.0,
        discount_rate: float = 0.0,
    ):
        self.strategies = list(strategies)
        self.n_strategies = len(self.strategies)
        self.time_horizon = int(time_horizon)
        self.years = np.arange(1, self.time_horizon + 1)

        # Labels
        if strategy_labels is not None:
            self.strategy_labels = {
                s: strategy_labels.get(s, s) for s in self.strategies
            }
        else:
            self.strategy_labels = {s: s for s in self.strategies}

        self.cost_inflation = float(cost_inflation)
        self.discount_rate = float(discount_rate)

        # --- Resolve inputs to arrays ---
        self.population = self._resolve_population(population)
        self.per_patient_costs_matrix = self._resolve_costs(per_patient_costs)
        self.market_share_current_matrix = self._resolve_market_shares(
            market_share_current, "market_share_current"
        )
        self.market_share_new_matrix = self._resolve_market_shares(
            market_share_new, "market_share_new"
        )

        # Store raw inputs for scenario/sensitivity rebuilds
        self._raw_inputs = {
            "per_patient_costs": per_patient_costs,
            "population": population,
            "market_share_current": market_share_current,
            "market_share_new": market_share_new,
        }

        # Compute
        self._compute()

    # -------------------------------------------------------------------
    # Resolution helpers
    # -------------------------------------------------------------------

    def _resolve_population(self, population) -> np.ndarray:
        """Resolve population specification to 1-D array of length *time_horizon*."""
        T = self.time_horizon
        if isinstance(population, (int, float)):
            return np.full(T, float(population))
        if isinstance(population, (list, np.ndarray)):
            arr = np.asarray(population, dtype=float)
            if len(arr) != T:
                raise ValueError(
                    f"Population list length ({len(arr)}) != time_horizon ({T})"
                )
            return arr
        if isinstance(population, dict):
            base = float(population["base"])
            if "growth_rate" in population:
                rate = float(population["growth_rate"])
                return np.array([base * (1 + rate) ** t for t in range(T)])
            if "annual_increase" in population:
                inc = float(population["annual_increase"])
                return np.array([base + inc * t for t in range(T)])
            raise ValueError(
                "Population dict must contain 'growth_rate' or 'annual_increase'"
            )
        raise TypeError(f"Unsupported population type: {type(population)}")

    def _resolve_costs(self, costs_dict) -> np.ndarray:
        """Resolve per-patient costs → 2-D array  (n_strategies × time_horizon)."""
        T = self.time_horizon
        matrix = np.zeros((self.n_strategies, T))
        for i, strategy in enumerate(self.strategies):
            if strategy not in costs_dict:
                raise ValueError(
                    f"Missing per-patient cost for strategy '{strategy}'"
                )
            val = costs_dict[strategy]
            if isinstance(val, (int, float)):
                matrix[i, :] = float(val)
            elif isinstance(val, (list, np.ndarray)):
                arr = np.asarray(val, dtype=float)
                if len(arr) != T:
                    raise ValueError(
                        f"Cost list for '{strategy}' length ({len(arr)}) "
                        f"!= time_horizon ({T})"
                    )
                matrix[i, :] = arr
            else:
                raise TypeError(
                    f"Unsupported cost type for '{strategy}': {type(val)}"
                )

        # Apply cumulative inflation
        if self.cost_inflation > 0:
            inflation = np.array(
                [(1 + self.cost_inflation) ** t for t in range(T)]
            )
            matrix = matrix * inflation[np.newaxis, :]

        return matrix

    def _resolve_market_shares(self, shares_dict, name: str) -> np.ndarray:
        """Resolve market shares → 2-D array  (n_strategies × time_horizon)."""
        T = self.time_horizon
        matrix = np.zeros((self.n_strategies, T))
        for i, strategy in enumerate(self.strategies):
            if strategy not in shares_dict:
                raise ValueError(
                    f"Missing market share for '{strategy}' in {name}"
                )
            val = shares_dict[strategy]
            if isinstance(val, (int, float)):
                matrix[i, :] = float(val)
            elif isinstance(val, (list, np.ndarray)):
                arr = np.asarray(val, dtype=float)
                if len(arr) != T:
                    raise ValueError(
                        f"Market share list for '{strategy}' in {name} "
                        f"length ({len(arr)}) != time_horizon ({T})"
                    )
                matrix[i, :] = arr
            else:
                raise TypeError(
                    f"Unsupported market share type for '{strategy}': {type(val)}"
                )

        # Warn if shares don't sum to ~1
        sums = matrix.sum(axis=0)
        for t in range(T):
            if abs(sums[t] - 1.0) > 0.01:
                warnings.warn(
                    f"{name}: market shares in year {t + 1} sum to "
                    f"{sums[t]:.4f} (expected ≈ 1.0). Proceeding without "
                    f"normalization."
                )
                break  # warn once

        return matrix

    # -------------------------------------------------------------------
    # Core computation
    # -------------------------------------------------------------------

    def _compute(self):
        """Compute budget impact for both scenarios."""
        T = self.time_horizon

        # Discount factors
        if self.discount_rate > 0:
            discount = np.array(
                [1 / (1 + self.discount_rate) ** t for t in range(T)]
            )
        else:
            discount = np.ones(T)

        # Patient counts: population × market_share  →  (n_strategies, T)
        self.patients_current = (
            self.population[np.newaxis, :] * self.market_share_current_matrix
        )
        self.patients_new = (
            self.population[np.newaxis, :] * self.market_share_new_matrix
        )

        # Per-strategy costs: patients × per-patient cost  →  (n_strategies, T)
        self.costs_current = (
            self.patients_current * self.per_patient_costs_matrix
        )
        self.costs_new = (
            self.patients_new * self.per_patient_costs_matrix
        )

        # Discounted costs
        self.costs_current_disc = self.costs_current * discount[np.newaxis, :]
        self.costs_new_disc = self.costs_new * discount[np.newaxis, :]

        # Totals per year (sum across strategies)
        self.total_current = self.costs_current_disc.sum(axis=0)
        self.total_new = self.costs_new_disc.sum(axis=0)

        # Budget impact
        self.impact = self.total_new - self.total_current
        self.cumulative_impact = np.cumsum(self.impact)

    # -------------------------------------------------------------------
    # Summary tables
    # -------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Annual budget impact summary.

        Returns
        -------
        pd.DataFrame
            Columns: Year, Population, Current Scenario, New Scenario,
            Budget Impact, Cumulative Impact.
        """
        df = pd.DataFrame({
            "Year": self.years,
            "Population": self.population.astype(int),
            "Current Scenario": self.total_current,
            "New Scenario": self.total_new,
            "Budget Impact": self.impact,
            "Cumulative Impact": self.cumulative_impact,
        })
        return df

    def detail(self, scenario: str = "both") -> pd.DataFrame:
        """
        Detailed cost breakdown by strategy and year.

        Parameters
        ----------
        scenario : str
            ``"current"``, ``"new"``, or ``"both"`` (default).

        Returns
        -------
        pd.DataFrame
            Columns: Scenario, Year, Strategy, Market Share, Patients,
            Per-Patient Cost, Total Cost.
        """
        rows: list[dict] = []

        def _add(name, share_mat, patients, costs):
            for t in range(self.time_horizon):
                for i, strategy in enumerate(self.strategies):
                    rows.append({
                        "Scenario": name,
                        "Year": int(self.years[t]),
                        "Strategy": self.strategy_labels[strategy],
                        "Market Share": share_mat[i, t],
                        "Patients": patients[i, t],
                        "Per-Patient Cost": self.per_patient_costs_matrix[i, t],
                        "Total Cost": costs[i, t],
                    })

        if scenario in ("current", "both"):
            _add(
                "Current",
                self.market_share_current_matrix,
                self.patients_current,
                self.costs_current_disc,
            )
        if scenario in ("new", "both"):
            _add(
                "New",
                self.market_share_new_matrix,
                self.patients_new,
                self.costs_new_disc,
            )

        return pd.DataFrame(rows)

    def cost_by_strategy(self) -> pd.DataFrame:
        """
        Cost breakdown by strategy summed over the entire time horizon.

        Returns
        -------
        pd.DataFrame
            Columns: Strategy, Current Total, New Total, Difference.
        """
        rows = []
        for i, strategy in enumerate(self.strategies):
            cur = self.costs_current_disc[i, :].sum()
            new = self.costs_new_disc[i, :].sum()
            rows.append({
                "Strategy": self.strategy_labels[strategy],
                "Current Total": cur,
                "New Total": new,
                "Difference": new - cur,
            })

        # Total row
        rows.append({
            "Strategy": "Total",
            "Current Total": self.total_current.sum(),
            "New Total": self.total_new.sum(),
            "Difference": self.impact.sum(),
        })

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------
    # Uptake-curve utilities
    # -------------------------------------------------------------------

    @staticmethod
    def linear_uptake(
        start: float, end: float, n_years: int
    ) -> List[float]:
        """
        Generate a linear market-share uptake curve.

        Parameters
        ----------
        start : float
            Starting market share.
        end : float
            Final market share.
        n_years : int
            Number of years.

        Returns
        -------
        list of float
        """
        return np.linspace(start, end, n_years).tolist()

    @staticmethod
    def sigmoid_uptake(
        start: float,
        end: float,
        n_years: int,
        steepness: float = 1.0,
    ) -> List[float]:
        """
        Generate an S-shaped (sigmoid) market-share uptake curve.

        Parameters
        ----------
        start : float
            Starting market share.
        end : float
            Final (asymptotic) market share.
        n_years : int
            Number of years.
        steepness : float, default 1.0
            Controls transition speed.  Higher → steeper.

        Returns
        -------
        list of float
        """
        t = np.linspace(-3, 3, n_years)
        raw = 1 / (1 + np.exp(-steepness * t))
        # normalise to exactly [start, end]
        normed = (raw - raw[0]) / (raw[-1] - raw[0])
        return (start + (end - start) * normed).tolist()

    # -------------------------------------------------------------------
    # Factory helpers
    # -------------------------------------------------------------------

    @classmethod
    def from_result(
        cls,
        result,
        population: Union[int, float, List[float], Dict],
        market_share_current: Dict[str, Union[float, List[float]]],
        market_share_new: Dict[str, Union[float, List[float]]],
        time_horizon: int = 5,
        *,
        annualize_years: Optional[float] = None,
        cost_inflation: float = 0.0,
        discount_rate: float = 0.0,
    ) -> "BudgetImpactAnalysis":
        """
        Create BIA from a deterministic model result.

        Extracts total costs per strategy from ``result.summary()`` and
        annualises them.  The annualised cost is used as the constant
        per-patient cost for BIA.

        Parameters
        ----------
        result : BaseResult, PSMBaseResult, MicroSimResult, or DESResult
            A deterministic model result.
        population, market_share_current, market_share_new, time_horizon
            See :class:`BudgetImpactAnalysis`.
        annualize_years : float, optional
            Number of years over which to annualise the model's total cost.
            For a Markov model with ``n_cycles=20`` annual cycles, set to 20.
            If *None*, the total cost is used as-is (assumes the model
            already represents one year of cost).
        cost_inflation : float
            Annual cost inflation rate.
        discount_rate : float
            Annual discount rate.

        Returns
        -------
        BudgetImpactAnalysis
        """
        summary = result.summary()

        # Identify strategy name/label columns
        strategies = summary["Strategy"].tolist()

        # Find cost column
        if "Total Cost" in summary.columns:
            costs = summary["Total Cost"].values.astype(float)
        elif "Mean Cost" in summary.columns:
            costs = summary["Mean Cost"].values.astype(float)
        else:
            raise ValueError(
                "Cannot find cost column in summary. "
                f"Columns: {list(summary.columns)}"
            )

        if annualize_years is not None:
            costs = costs / float(annualize_years)

        per_patient_costs = {s: float(c) for s, c in zip(strategies, costs)}

        # Recover internal strategy IDs if labels differ
        # (use labels since summary() returns labels)
        return cls(
            strategies=strategies,
            per_patient_costs=per_patient_costs,
            population=population,
            market_share_current=market_share_current,
            market_share_new=market_share_new,
            time_horizon=time_horizon,
            cost_inflation=cost_inflation,
            discount_rate=discount_rate,
        )

    # -------------------------------------------------------------------
    # Scenario / Sensitivity analysis
    # -------------------------------------------------------------------

    def _rebuild(self, **overrides) -> "BudgetImpactAnalysis":
        """Return a new BIA with selectively overridden parameters."""
        kwargs = dict(
            strategies=self.strategies,
            per_patient_costs=overrides.get(
                "per_patient_costs", self._raw_inputs["per_patient_costs"]
            ),
            population=overrides.get(
                "population", self._raw_inputs["population"]
            ),
            market_share_current=overrides.get(
                "market_share_current",
                self._raw_inputs["market_share_current"],
            ),
            market_share_new=overrides.get(
                "market_share_new", self._raw_inputs["market_share_new"]
            ),
            time_horizon=self.time_horizon,
            strategy_labels=self.strategy_labels,
            cost_inflation=overrides.get(
                "cost_inflation", self.cost_inflation
            ),
            discount_rate=overrides.get(
                "discount_rate", self.discount_rate
            ),
        )
        return BudgetImpactAnalysis(**kwargs)

    def scenario_analysis(
        self,
        scenarios: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Run multiple BIA scenarios and compare total budget impact.

        Parameters
        ----------
        scenarios : dict
            ``{scenario_name: {param: value, ...}}``.  Recognised keys:

            - ``"population"``
            - ``"per_patient_costs"``
            - ``"market_share_current"``
            - ``"market_share_new"``
            - ``"cost_inflation"``
            - ``"discount_rate"``

            An empty dict ``{}`` uses the base-case settings.

        Returns
        -------
        pd.DataFrame
            Columns: Scenario, Year 1 … Year T, Total.
        """
        rows = []
        for name, overrides in scenarios.items():
            bia = self._rebuild(**overrides)
            row: dict = {"Scenario": name}
            for t in range(self.time_horizon):
                row[f"Year {t + 1}"] = bia.impact[t]
            row["Total"] = float(bia.impact.sum())
            rows.append(row)

        return pd.DataFrame(rows)

    def one_way_sensitivity(
        self,
        param: str,
        values: Sequence,
        *,
        labels: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        One-way sensitivity analysis on a single parameter.

        Parameters
        ----------
        param : str
            One of ``"population"``, ``"cost_inflation"``,
            ``"discount_rate"``, or a strategy name (to vary that
            strategy's per-patient cost).
        values : sequence
            Values to evaluate.
        labels : sequence of str, optional
            Display labels (defaults to ``str(v)``).

        Returns
        -------
        pd.DataFrame
            Columns: Parameter, Value, Year 1 … Year T, Total Budget Impact.
        """
        if labels is None:
            labels = [str(v) for v in values]

        rows = []
        for lbl, val in zip(labels, values):
            overrides: dict = {}
            if param == "population":
                overrides["population"] = val
            elif param == "cost_inflation":
                overrides["cost_inflation"] = val
            elif param == "discount_rate":
                overrides["discount_rate"] = val
            elif param in self.strategies:
                # Vary per-patient cost for one strategy
                new_costs = dict(self._raw_inputs["per_patient_costs"])
                new_costs[param] = val
                overrides["per_patient_costs"] = new_costs
            else:
                raise ValueError(
                    f"Unknown parameter: '{param}'. Must be 'population', "
                    f"'cost_inflation', 'discount_rate', or a strategy name."
                )

            bia = self._rebuild(**overrides)
            row: dict = {
                "Parameter": param,
                "Value": lbl,
            }
            for t in range(self.time_horizon):
                row[f"Year {t + 1}"] = bia.impact[t]
            row["Total Budget Impact"] = float(bia.impact.sum())
            rows.append(row)

        return pd.DataFrame(rows)

    def tornado(
        self,
        sensitivities: Dict[str, Tuple[object, object]],
    ) -> pd.DataFrame:
        """
        Compute tornado data for multiple parameters (low / high).

        Parameters
        ----------
        sensitivities : dict
            ``{param_label: (low_value, high_value)}``.  *param_label* is
            one of ``"population"``, ``"cost_inflation"``,
            ``"discount_rate"``, or a strategy name.

        Returns
        -------
        pd.DataFrame
            Columns: Parameter, Low Value, High Value, Impact (Low),
            Impact (High), Impact (Base), Range.  Sorted by Range desc.
        """
        base_total = float(self.impact.sum())
        rows = []
        for param, (lo, hi) in sensitivities.items():
            bia_lo = self._rebuild(**self._override_single(param, lo))
            bia_hi = self._rebuild(**self._override_single(param, hi))
            impact_lo = float(bia_lo.impact.sum())
            impact_hi = float(bia_hi.impact.sum())
            rows.append({
                "Parameter": param,
                "Low Value": lo,
                "High Value": hi,
                "Impact (Low)": impact_lo,
                "Impact (High)": impact_hi,
                "Impact (Base)": base_total,
                "Range": abs(impact_hi - impact_lo),
            })

        df = pd.DataFrame(rows).sort_values(
            "Range", ascending=False
        ).reset_index(drop=True)
        return df

    def _override_single(self, param: str, value) -> dict:
        """Build a single-parameter override dict."""
        if param == "population":
            return {"population": value}
        if param == "cost_inflation":
            return {"cost_inflation": value}
        if param == "discount_rate":
            return {"discount_rate": value}
        if param in self.strategies:
            new_costs = dict(self._raw_inputs["per_patient_costs"])
            new_costs[param] = value
            return {"per_patient_costs": new_costs}
        raise ValueError(f"Unknown parameter: '{param}'")

    # -------------------------------------------------------------------
    # Plotting shortcuts
    # -------------------------------------------------------------------

    def plot_budget_impact(self, **kwargs):
        """Bar chart of annual budget impact (new − current)."""
        from .plotting import plot_budget_impact
        return plot_budget_impact(self, **kwargs)

    def plot_budget_comparison(self, **kwargs):
        """Grouped bar chart: total costs current vs new per year."""
        from .plotting import plot_budget_comparison
        return plot_budget_comparison(self, **kwargs)

    def plot_market_share(self, **kwargs):
        """Market-share evolution over time (current & new)."""
        from .plotting import plot_market_share
        return plot_market_share(self, **kwargs)

    def plot_detail(self, **kwargs):
        """Stacked-bar cost breakdown by strategy per year."""
        from .plotting import plot_bia_detail
        return plot_bia_detail(self, **kwargs)

    def plot_tornado(self, sensitivities: Dict[str, Tuple], **kwargs):
        """
        Tornado diagram for BIA one-way sensitivity.

        Parameters
        ----------
        sensitivities : dict
            ``{param_label: (low_value, high_value)}``.
        """
        from .plotting import plot_bia_tornado
        return plot_bia_tornado(self, sensitivities, **kwargs)

    # -------------------------------------------------------------------
    # repr
    # -------------------------------------------------------------------

    def __repr__(self) -> str:
        total = self.impact.sum()
        sign = "+" if total >= 0 else ""
        return (
            f"BudgetImpactAnalysis("
            f"strategies={self.n_strategies}, "
            f"horizon={self.time_horizon}y, "
            f"total_impact={sign}{total:,.0f})"
        )
