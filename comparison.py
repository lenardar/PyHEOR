"""
comparison.py — Multi-Strategy Cost-Effectiveness Comparison
============================================================

Provides:
  - CEAnalysis: comprehensive CE analysis with frontier detection,
    NMB analysis, CEAF, and EVPI calculation.
  - calculate_icers(): standalone sequential ICER with dominance detection.

Algorithms follow the standards described in:
  - Drummond et al., "Methods for the Economic Evaluation of Health Care Programmes"
  - Fenwick et al., "Cost-effectiveness acceptability curves" (2001)
  - Barton et al., "Optimal Cost-Effectiveness Decisions" (2008)
  - dampack R package (Alarid-Escudero et al.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Standalone ICER calculation with dominance detection
# ---------------------------------------------------------------------------

def calculate_icers(
    strategies: Sequence[str],
    costs: Sequence[float],
    qalys: Sequence[float],
) -> pd.DataFrame:
    """
    Calculate sequential ICERs with strong and extended dominance detection.

    This implements the standard algorithm:
    1. Sort strategies by cost (ascending)
    2. Eliminate strongly dominated (higher cost, lower/equal QALYs)
    3. Compute sequential ICERs for remaining
    4. Eliminate extendedly dominated (ICER > next strategy's ICER)
    5. Repeat until stable

    Parameters
    ----------
    strategies : sequence of str
        Strategy names.
    costs : sequence of float
        Mean total costs per strategy.
    qalys : sequence of float
        Mean total QALYs per strategy.

    Returns
    -------
    pd.DataFrame
        Columns: Strategy, Cost, QALYs, Status, ICER, Inc_Cost, Inc_QALYs, Ref
        Status: "Ref" (reference/cheapest), "ND" (non-dominated on frontier),
                "D" (strongly dominated), "ED" (extendedly dominated)
    """
    strategies = list(strategies)
    costs = np.asarray(costs, dtype=float)
    qalys = np.asarray(qalys, dtype=float)
    n = len(strategies)

    if n < 2:
        return pd.DataFrame({
            "Strategy": strategies,
            "Cost": costs,
            "QALYs": qalys,
            "Status": ["Ref"] if n == 1 else [],
            "ICER": [np.nan] if n == 1 else [],
            "Inc_Cost": [0.0] if n == 1 else [],
            "Inc_QALYs": [0.0] if n == 1 else [],
            "Ref": [""] if n == 1 else [],
        })

    # Sort by cost, then by QALYs descending (prefer higher QALYs at same cost)
    order = np.lexsort((-qalys, costs))
    s_names = [strategies[i] for i in order]
    s_costs = costs[order].copy()
    s_qalys = qalys[order].copy()

    status = [""] * n
    icers = np.full(n, np.nan)
    inc_costs = np.zeros(n)
    inc_qalys = np.zeros(n)
    refs = [""] * n

    # Step 1: Strong dominance detection
    # Strategy j is strongly dominated if there exists strategy i with
    # cost_i <= cost_j AND qaly_i >= qaly_j (and at least one strict)
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            if (s_costs[i] <= s_costs[j] and s_qalys[i] >= s_qalys[j]
                    and (s_costs[i] < s_costs[j] or s_qalys[i] > s_qalys[j])):
                status[j] = "D"
                break

    # Step 2: Sequential ICER + extended dominance (iterative)
    converged = False
    while not converged:
        converged = True
        # Get non-dominated indices
        nd_idx = [i for i in range(n) if status[i] != "D" and status[i] != "ED"]

        if len(nd_idx) < 2:
            if nd_idx:
                status[nd_idx[0]] = "Ref"
            break

        # Cheapest is reference
        status[nd_idx[0]] = "Ref"
        icers[nd_idx[0]] = np.nan
        inc_costs[nd_idx[0]] = 0.0
        inc_qalys[nd_idx[0]] = 0.0
        refs[nd_idx[0]] = ""

        # Compute sequential ICERs
        for k in range(1, len(nd_idx)):
            i_curr = nd_idx[k]
            i_prev = nd_idx[k - 1]
            dc = s_costs[i_curr] - s_costs[i_prev]
            dq = s_qalys[i_curr] - s_qalys[i_prev]
            inc_costs[i_curr] = dc
            inc_qalys[i_curr] = dq
            refs[i_curr] = s_names[i_prev]

            if dq <= 0:
                # Should have been caught by strong dominance, but safety net
                status[i_curr] = "D"
                converged = False
                break
            else:
                icers[i_curr] = dc / dq
                if status[i_curr] != "Ref":
                    status[i_curr] = "ND"

        if not converged:
            continue

        # Check extended dominance: ICER should be monotonically increasing
        # If ICER[k] < ICER[k-1], then strategy k-1 is extendedly dominated
        nd_idx2 = [i for i in range(n) if status[i] in ("Ref", "ND")]
        for k in range(2, len(nd_idx2)):
            i_curr = nd_idx2[k]
            i_prev = nd_idx2[k - 1]
            if icers[i_curr] < icers[i_prev]:
                status[i_prev] = "ED"
                converged = False
                break

    # Fill non-dominated labels
    for i in range(n):
        if status[i] == "":
            status[i] = "ND"

    result = pd.DataFrame({
        "Strategy": s_names,
        "Cost": s_costs,
        "QALYs": s_qalys,
        "Inc_Cost": inc_costs,
        "Inc_QALYs": inc_qalys,
        "ICER": icers,
        "Status": status,
        "Ref": refs,
    })

    # Format ICER column for display
    return result


# ---------------------------------------------------------------------------
# CEAnalysis — full comparison class
# ---------------------------------------------------------------------------

class CEAnalysis:
    """
    Comprehensive multi-strategy cost-effectiveness analysis.

    Supports:
    - Deterministic analysis: efficiency frontier, sequential ICER, NMB
    - PSA analysis: CEAF, EVPI (requires PSA data)

    Examples
    --------
    >>> cea = CEAnalysis.from_result(base_case_result)
    >>> print(cea.frontier())       # Efficiency frontier with dominance
    >>> print(cea.nmb(wtp=50000))   # NMB ranking

    >>> cea = CEAnalysis.from_psa(psa_result)
    >>> print(cea.evpi())           # EVPI curve
    >>> cea.plot_ceaf()             # CEAF plot
    """

    def __init__(
        self,
        strategies: Sequence[str],
        costs: Sequence[float],
        qalys: Sequence[float],
        *,
        lys: Optional[Sequence[float]] = None,
        psa_costs: Optional[np.ndarray] = None,
        psa_qalys: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        strategies : sequence of str
            Strategy names.
        costs : sequence of float
            Mean total costs per strategy (deterministic or PSA mean).
        qalys : sequence of float
            Mean total QALYs per strategy.
        lys : sequence of float, optional
            Mean life years per strategy.
        psa_costs : ndarray of shape (n_sim, n_strategies), optional
            Per-simulation costs (for CEAF/EVPI). Columns align with strategies.
        psa_qalys : ndarray of shape (n_sim, n_strategies), optional
            Per-simulation QALYs.
        """
        self.strategies = list(strategies)
        self.costs = np.asarray(costs, dtype=float)
        self.qalys = np.asarray(qalys, dtype=float)
        self.lys = np.asarray(lys, dtype=float) if lys is not None else None
        self.n_strategies = len(self.strategies)

        # PSA data
        self.has_psa = psa_costs is not None and psa_qalys is not None
        if self.has_psa:
            self.psa_costs = np.asarray(psa_costs, dtype=float)
            self.psa_qalys = np.asarray(psa_qalys, dtype=float)
            self.n_sim = self.psa_costs.shape[0]
        else:
            self.psa_costs = None
            self.psa_qalys = None
            self.n_sim = 0

        # Cache
        self._frontier_cache = None

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_result(cls, result) -> "CEAnalysis":
        """
        Create CEAnalysis from a deterministic result object.

        Accepts: BaseResult, PSMBaseResult, MicroSimResult.
        """
        summary = result.summary()

        # Extract strategy names
        strategies = summary["Strategy"].tolist()

        # Extract costs — different column names across result types
        if "Total Cost" in summary.columns:
            costs = summary["Total Cost"].values
        elif "Mean Cost" in summary.columns:
            costs = summary["Mean Cost"].values
        else:
            raise ValueError("Cannot find cost column in summary")

        # Extract QALYs
        if "QALYs" in summary.columns:
            qalys = summary["QALYs"].values
        elif "Mean QALYs" in summary.columns:
            qalys = summary["Mean QALYs"].values
        else:
            raise ValueError("Cannot find QALYs column in summary")

        # Extract LYs
        lys = None
        if "LYs" in summary.columns:
            lys = summary["LYs"].values
        elif "Mean LYs" in summary.columns:
            lys = summary["Mean LYs"].values

        return cls(strategies, costs, qalys, lys=lys)

    @classmethod
    def from_psa(cls, psa_result) -> "CEAnalysis":
        """
        Create CEAnalysis from a PSA result object.

        Accepts: PSAResult, MicroSimPSAResult.
        Extracts per-simulation cost/QALY data for CEAF and EVPI.
        """
        ce = psa_result.ce_table

        # Normalize column names — handle both PSAResult (lowercase)
        # and MicroSimPSAResult (capitalized) conventions
        col_map = {}
        for col in ce.columns:
            col_map[col.lower()] = col

        strat_col = col_map.get("strategy", col_map.get("strategy_label", None))
        sim_col = col_map.get("simulation", col_map.get("sim", None))
        cost_col = col_map.get("cost", col_map.get("total_cost", None))
        qaly_col = col_map.get("qalys", None)
        ly_col = col_map.get("lys", None)

        if strat_col is None or sim_col is None or cost_col is None or qaly_col is None:
            raise ValueError(
                f"Cannot parse ce_table columns: {list(ce.columns)}. "
                "Expected columns for strategy, simulation, cost, qalys."
            )

        strategies = ce[strat_col].unique().tolist()

        # Build per-simulation matrices
        n_sim = int(ce[sim_col].max())
        psa_costs = np.zeros((n_sim, len(strategies)))
        psa_qalys = np.zeros((n_sim, len(strategies)))

        for j, strat in enumerate(strategies):
            mask = ce[strat_col] == strat
            subset = ce.loc[mask].sort_values(sim_col)
            psa_costs[:, j] = subset[cost_col].values[:n_sim]
            psa_qalys[:, j] = subset[qaly_col].values[:n_sim]

        mean_costs = psa_costs.mean(axis=0)
        mean_qalys = psa_qalys.mean(axis=0)

        # Try to get LYs
        lys = None
        if ly_col is not None:
            psa_lys = np.zeros((n_sim, len(strategies)))
            for j, strat in enumerate(strategies):
                mask = ce[strat_col] == strat
                subset = ce.loc[mask].sort_values(sim_col)
                psa_lys[:, j] = subset[ly_col].values[:n_sim]
            lys = psa_lys.mean(axis=0)

        return cls(
            strategies, mean_costs, mean_qalys,
            lys=lys, psa_costs=psa_costs, psa_qalys=psa_qalys,
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Basic cost/QALY summary table."""
        data = {
            "Strategy": self.strategies,
            "Cost": self.costs,
            "QALYs": self.qalys,
        }
        if self.lys is not None:
            data["LYs"] = self.lys
        return pd.DataFrame(data)

    # -----------------------------------------------------------------------
    # Efficiency frontier & sequential ICER
    # -----------------------------------------------------------------------

    def frontier(self) -> pd.DataFrame:
        """
        Efficiency frontier analysis with sequential ICER and dominance detection.

        Returns
        -------
        pd.DataFrame
            Sorted by cost. Columns: Strategy, Cost, QALYs, Inc_Cost,
            Inc_QALYs, ICER, Status, Ref.
            Status values:
              "Ref"  — reference (cheapest non-dominated)
              "ND"   — non-dominated (on frontier)
              "D"    — strongly dominated
              "ED"   — extendedly dominated
        """
        if self._frontier_cache is None:
            self._frontier_cache = calculate_icers(
                self.strategies, self.costs, self.qalys
            )
        return self._frontier_cache.copy()

    def frontier_strategies(self) -> list:
        """Return names of strategies on the efficiency frontier."""
        f = self.frontier()
        return f.loc[f["Status"].isin(["Ref", "ND"]), "Strategy"].tolist()

    def is_dominated(self, strategy: str) -> bool:
        """Check if a strategy is dominated or extendedly dominated."""
        f = self.frontier()
        row = f.loc[f["Strategy"] == strategy]
        if row.empty:
            raise ValueError(f"Strategy '{strategy}' not found")
        return row["Status"].iloc[0] in ("D", "ED")

    # -----------------------------------------------------------------------
    # NMB analysis
    # -----------------------------------------------------------------------

    def nmb(self, wtp: float = 50000) -> pd.DataFrame:
        """
        Net Monetary Benefit ranking at a given WTP threshold.

        NMB = QALYs × WTP − Cost

        Parameters
        ----------
        wtp : float
            Willingness-to-pay threshold ($/QALY).

        Returns
        -------
        pd.DataFrame
            Columns: Strategy, Cost, QALYs, NMB, Rank, Optimal.
            Sorted by NMB descending.
        """
        nmb_vals = self.qalys * wtp - self.costs
        df = pd.DataFrame({
            "Strategy": self.strategies,
            "Cost": self.costs,
            "QALYs": self.qalys,
            "NMB": nmb_vals,
        })
        df = df.sort_values("NMB", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)
        df["Optimal"] = df["Rank"] == 1
        return df

    def nmb_curve(
        self,
        wtp_range: Tuple[float, float] = (0, 150000),
        n_wtp: int = 301,
    ) -> pd.DataFrame:
        """
        NMB across a range of WTP thresholds.

        Parameters
        ----------
        wtp_range : tuple of (min, max)
        n_wtp : int
            Number of WTP points.

        Returns
        -------
        pd.DataFrame
            Columns: WTP, then one column per strategy with NMB values.
        """
        wtp_vals = np.linspace(wtp_range[0], wtp_range[1], n_wtp)
        data = {"WTP": wtp_vals}
        for j, strat in enumerate(self.strategies):
            data[strat] = self.qalys[j] * wtp_vals - self.costs[j]
        return pd.DataFrame(data)

    def optimal_strategy(self, wtp: float = 50000) -> str:
        """Return the strategy with highest NMB at given WTP."""
        nmb_vals = self.qalys * wtp - self.costs
        return self.strategies[np.argmax(nmb_vals)]

    def optimal_curve(
        self,
        wtp_range: Tuple[float, float] = (0, 150000),
        n_wtp: int = 301,
    ) -> pd.DataFrame:
        """
        Optimal strategy at each WTP threshold (deterministic).

        Returns
        -------
        pd.DataFrame
            Columns: WTP, Optimal_Strategy, Max_NMB.
        """
        wtp_vals = np.linspace(wtp_range[0], wtp_range[1], n_wtp)
        optimal = []
        max_nmb = []
        for w in wtp_vals:
            nmb_vals = self.qalys * w - self.costs
            idx = np.argmax(nmb_vals)
            optimal.append(self.strategies[idx])
            max_nmb.append(nmb_vals[idx])
        return pd.DataFrame({
            "WTP": wtp_vals,
            "Optimal_Strategy": optimal,
            "Max_NMB": max_nmb,
        })

    # -----------------------------------------------------------------------
    # Incremental NMB analysis (pairwise)
    # -----------------------------------------------------------------------

    def inmb(
        self,
        wtp: float = 50000,
        comparator: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Incremental NMB vs a comparator at given WTP.

        Parameters
        ----------
        wtp : float
        comparator : str, optional
            Reference strategy. Defaults to cheapest.

        Returns
        -------
        pd.DataFrame
            Columns: Strategy, INMB, vs.
        """
        comp = comparator or self.strategies[np.argmin(self.costs)]
        comp_idx = self.strategies.index(comp)
        comp_nmb = self.qalys[comp_idx] * wtp - self.costs[comp_idx]

        rows = []
        for j, strat in enumerate(self.strategies):
            if strat == comp:
                continue
            nmb_j = self.qalys[j] * wtp - self.costs[j]
            rows.append({
                "Strategy": strat,
                "vs": comp,
                "INMB": nmb_j - comp_nmb,
            })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # PSA-based: CEAF
    # -----------------------------------------------------------------------

    def ceaf(
        self,
        wtp_range: Tuple[float, float] = (0, 150000),
        n_wtp: int = 301,
    ) -> pd.DataFrame:
        """
        Cost-Effectiveness Acceptability Frontier (CEAF).

        At each WTP:
        1. Identify the strategy with highest mean NMB (deterministically optimal)
        2. CEAF = P(that strategy has max NMB across simulations)

        This is the probability that the strategy recommended by the frontier
        is actually cost-effective.

        Parameters
        ----------
        wtp_range : tuple of (min, max)
        n_wtp : int

        Returns
        -------
        pd.DataFrame
            Columns: WTP, Optimal_Strategy, CEAF, then per-strategy CEAC columns.

        Raises
        ------
        ValueError
            If no PSA data available.
        """
        if not self.has_psa:
            raise ValueError("CEAF requires PSA data. Use CEAnalysis.from_psa().")

        wtp_vals = np.linspace(wtp_range[0], wtp_range[1], n_wtp)
        rows = []

        for w in wtp_vals:
            # Per-simulation NMB: (n_sim, n_strategies)
            nmb_matrix = self.psa_qalys * w - self.psa_costs

            # Deterministic optimal: strategy with highest expected NMB
            mean_nmb = nmb_matrix.mean(axis=0)
            det_optimal_idx = np.argmax(mean_nmb)
            det_optimal = self.strategies[det_optimal_idx]

            # CEAC: proportion of sims where each strategy has max NMB
            best_per_sim = np.argmax(nmb_matrix, axis=1)
            ceac = {}
            for j, strat in enumerate(self.strategies):
                ceac[strat] = np.mean(best_per_sim == j)

            # CEAF: probability that the deterministic optimal is best
            ceaf_val = ceac[det_optimal]

            row = {"WTP": w, "Optimal_Strategy": det_optimal, "CEAF": ceaf_val}
            for strat in self.strategies:
                row[f"CEAC_{strat}"] = ceac[strat]
            rows.append(row)

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # PSA-based: EVPI
    # -----------------------------------------------------------------------

    def evpi(
        self,
        wtp_range: Tuple[float, float] = (0, 150000),
        n_wtp: int = 301,
    ) -> pd.DataFrame:
        """
        Expected Value of Perfect Information (EVPI).

        EVPI(WTP) = E[max_s NMB_s] − max_s E[NMB_s]

        This represents the maximum amount a decision-maker should pay
        to eliminate all parameter uncertainty.

        Parameters
        ----------
        wtp_range : tuple of (min, max)
        n_wtp : int

        Returns
        -------
        pd.DataFrame
            Columns: WTP, EVPI, Max_Expected_NMB, Expected_Max_NMB.
        """
        if not self.has_psa:
            raise ValueError("EVPI requires PSA data. Use CEAnalysis.from_psa().")

        wtp_vals = np.linspace(wtp_range[0], wtp_range[1], n_wtp)
        rows = []

        for w in wtp_vals:
            nmb_matrix = self.psa_qalys * w - self.psa_costs  # (n_sim, n_strat)

            # max_s E[NMB_s] — best expected NMB under current info
            max_expected_nmb = nmb_matrix.mean(axis=0).max()

            # E[max_s NMB_s] — expected NMB if we knew which was best per sim
            expected_max_nmb = nmb_matrix.max(axis=1).mean()

            evpi_val = expected_max_nmb - max_expected_nmb

            rows.append({
                "WTP": w,
                "EVPI": evpi_val,
                "Max_Expected_NMB": max_expected_nmb,
                "Expected_Max_NMB": expected_max_nmb,
            })

        return pd.DataFrame(rows)

    def evpi_single(self, wtp: float = 50000) -> float:
        """EVPI at a single WTP threshold."""
        if not self.has_psa:
            raise ValueError("EVPI requires PSA data.")
        nmb_matrix = self.psa_qalys * wtp - self.psa_costs
        return nmb_matrix.max(axis=1).mean() - nmb_matrix.mean(axis=0).max()

    # -----------------------------------------------------------------------
    # Plotting helpers (delegate to plotting.py)
    # -----------------------------------------------------------------------

    def plot_frontier(self, **kwargs):
        """Plot CE plane with efficiency frontier."""
        from .plotting import plot_ce_frontier
        return plot_ce_frontier(self, **kwargs)

    def plot_nmb_curve(self, **kwargs):
        """Plot NMB curves across WTP thresholds."""
        from .plotting import plot_nmb_curve
        return plot_nmb_curve(self, **kwargs)

    def plot_ceaf(self, **kwargs):
        """Plot CEAF (requires PSA data)."""
        from .plotting import plot_ceaf
        return plot_ceaf(self, **kwargs)

    def plot_evpi(self, **kwargs):
        """Plot EVPI curve (requires PSA data)."""
        from .plotting import plot_evpi
        return plot_evpi(self, **kwargs)

    # -----------------------------------------------------------------------
    # Info / repr
    # -----------------------------------------------------------------------

    def info(self) -> str:
        lines = [
            "CEAnalysis",
            f"  Strategies ({self.n_strategies}): {self.strategies}",
            f"  Costs: {[f'{c:,.0f}' for c in self.costs]}",
            f"  QALYs: {[f'{q:.4f}' for q in self.qalys]}",
            f"  PSA data: {'Yes (' + str(self.n_sim) + ' sims)' if self.has_psa else 'No'}",
            f"  Frontier strategies: {self.frontier_strategies()}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"CEAnalysis(strategies={self.strategies}, "
                f"has_psa={self.has_psa})")
