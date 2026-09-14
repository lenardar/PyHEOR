"""
Result classes for base case, OWSA, and PSA analyses.

Each result class provides summary tables, ICER computation, and
convenient access to plotting methods.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


def classify_incremental(
    delta_cost: float, delta_effect: float,
    cost_tol: float = 1e-10, effect_tol: Optional[float] = None,
):
    """Return a numeric ICER when meaningful and a quadrant label.

    Cost and effect live on unrelated scales (currency vs. QALYs), so one
    tolerance for both is either too loose for money or too tight for
    effect: ``classify_incremental(100.0, 1e-9)`` would otherwise report a
    1e11 ICER instead of "No difference". ``effect_tol`` defaults to
    ``cost_tol`` for callers that have not been updated to pass a
    QALY-scale tolerance explicitly.
    """
    if effect_tol is None:
        effect_tol = cost_tol

    if abs(delta_effect) <= effect_tol:
        if delta_cost > cost_tol:
            return np.nan, "Dominated"
        if delta_cost < -cost_tol:
            return np.nan, "Dominant"
        return np.nan, "No difference"

    if delta_effect > 0:
        if delta_cost <= cost_tol:
            return np.nan, "Dominant"
        value = delta_cost / delta_effect
        return value, f"{value:,.0f}"

    if delta_cost >= -cost_tol:
        return np.nan, "Dominated"
    value = delta_cost / delta_effect
    return value, f"{value:,.0f} (less effective, less costly)"


def _scale_tol(*magnitudes: float, rtol: float = 1e-9, atol: float = 1e-10) -> float:
    """A tolerance proportional to the largest magnitude in play.

    Mirrors the relative tolerance used for frontier dominance, so a real
    difference at the scale of Monte Carlo noise is not read as one.
    """
    scale = max((abs(m) for m in magnitudes if np.isfinite(m)), default=0.0)
    return max(rtol * scale, atol)


def _paired_psa_icer(ce: pd.DataFrame, strategy: str, comparator: str):
    """Per-simulation incremental cost/QALYs, paired by simulation id.

    Selecting each strategy's rows independently and subtracting by position
    only pairs simulations correctly if both selections share one row order.
    Sorting by ``sim`` makes that an explicit guarantee instead of an
    accident of how ``ce_table`` was built.
    """
    int_df = ce.loc[ce['strategy'] == strategy].sort_values('sim')
    comp_df = ce.loc[ce['strategy'] == comparator].sort_values('sim')
    inc_cost = int_df['total_cost'].to_numpy() - comp_df['total_cost'].to_numpy()
    inc_qaly = int_df['qalys'].to_numpy() - comp_df['qalys'].to_numpy()
    return int_df, comp_df, inc_cost, inc_qaly


class StrategyOutcomeResult:
    """Shared incremental analysis for deterministic result objects.

    Subclasses provide per-strategy totals through :meth:`_totals` and name
    the columns their tables use, so ICER and NMB are derived in one place
    for every engine.
    """

    #: Column labels for the effect and cost totals reported by ``nmb``.
    _EFFECT_LABEL = "QALYs"
    _COST_LABEL = "Total Cost"

    def _totals(self, strategy: str):
        """Return ``(cost, qalys, lys)`` for one strategy."""
        raise NotImplementedError

    def _resolve_comparator(self, comparator: Optional[str]) -> str:
        if comparator is None:
            return self.model.strategy_names[0]
        if comparator not in self.model.strategy_names:
            raise ValueError(
                f"Unknown comparator {comparator!r}; available strategies "
                f"are {self.model.strategy_names!r}"
            )
        return comparator

    def icer(self, comparator: Optional[str] = None) -> pd.DataFrame:
        """Pairwise ICERs against a comparator, classified before dividing.

        ``ICER`` is numeric and is NaN whenever the quadrant admits no ratio;
        ``ICER Classification`` always carries the readable verdict.
        """
        comparator = self._resolve_comparator(comparator)
        base_cost, base_qaly, base_ly = self._totals(comparator)

        rows = []
        for strategy in self.model.strategy_names:
            if strategy == comparator:
                continue
            cost, qaly, ly = self._totals(strategy)
            inc_cost = cost - base_cost
            inc_qaly = qaly - base_qaly
            value, classification = classify_incremental(
                inc_cost, inc_qaly,
                cost_tol=_scale_tol(cost, base_cost),
                effect_tol=_scale_tol(qaly, base_qaly),
            )
            rows.append({
                'Strategy': self.model.strategy_labels[strategy],
                'vs': self.model.strategy_labels[comparator],
                'Incremental Cost': inc_cost,
                'Incremental QALYs': inc_qaly,
                'Incremental LYs': ly - base_ly,
                'ICER': value,
                'ICER Classification': classification,
            })

        return pd.DataFrame(rows)

    def nmb(self, wtp: float = 50000,
            comparator: Optional[str] = None) -> pd.DataFrame:
        """Net monetary benefit at a willingness-to-pay threshold."""
        comparator = self._resolve_comparator(comparator)
        base_cost, base_qaly, _ = self._totals(comparator)

        rows = []
        for strategy in self.model.strategy_names:
            cost, qaly, _ = self._totals(strategy)
            incremental = (
                0.0 if strategy == comparator
                else (qaly - base_qaly) * wtp - (cost - base_cost)
            )
            rows.append({
                'Strategy': self.model.strategy_labels[strategy],
                self._EFFECT_LABEL: qaly,
                self._COST_LABEL: cost,
                'NMB': qaly * wtp - cost,
                'Incremental NMB': incremental,
            })

        return pd.DataFrame(rows)


class BaseResult(StrategyOutcomeResult):
    """Results from a deterministic base case analysis.
    
    Attributes
    ----------
    model : MarkovModel
        The source model.
    results : dict
        Raw simulation results keyed by strategy.
    params : dict
        Parameter values used.
    """
    
    def __init__(self, model, results: dict, params: dict):
        self.model = model
        self.results = results
        self.params = params
    
    def summary(self) -> pd.DataFrame:
        """Summarize total costs and QALYs per strategy.
        
        Returns
        -------
        pd.DataFrame
            Summary table with costs by category, total cost, QALYs, and LYs.
        """
        rows = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            row = {
                'Strategy': self.model.strategy_labels[strategy],
                'LYs': r['total_lys'],
                'QALYs': r['total_qalys'],
            }
            total_cost = 0.0
            for cat in r['total_costs']:
                row[f'Cost ({cat})'] = r['total_costs'][cat]
                total_cost += r['total_costs'][cat]
            row['Total Cost'] = total_cost
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _totals(self, strategy: str):
        r = self.results[strategy]
        return (
            sum(r['total_costs'].values()), r['total_qalys'], r['total_lys']
        )

    @property
    def markov_trace(self) -> pd.DataFrame:
        """Get Markov trace (state occupancy over time) as DataFrame."""
        dfs = []
        for strategy in self.model.strategy_names:
            trace = self.results[strategy]['trace']
            df = pd.DataFrame(trace, columns=self.model.states)
            df.insert(0, 'Cycle', np.arange(self.model.n_cycles + 1))
            df.insert(1, 'Strategy', self.model.strategy_labels[strategy])
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    # --- Plotting Shortcuts ---
    
    def plot_trace(self, **kwargs):
        """Plot Markov trace (state occupancy over time)."""
        from ..plotting import plot_trace
        return plot_trace(self, **kwargs)
    
    def plot_transition_diagram(self, strategy=None, **kwargs):
        """Plot state transition diagram."""
        from ..plotting import plot_transition_diagram
        return plot_transition_diagram(self.model, self.params, strategy=strategy, **kwargs)
    
    def plot_model_diagram(self, **kwargs):
        """Plot TreeAge-style model structure diagram."""
        from ..plotting import plot_model_diagram
        return plot_model_diagram(self.model, **kwargs)


class OWSAResult:
    """Results from one-way sensitivity analysis.
    
    Attributes
    ----------
    model : MarkovModel
        The source model.
    base_result : dict
        Base case simulation results.
    base_params : dict
        Base case parameter values.
    owsa_data : list
        List of dicts with varied parameter results.
    wtp : float
        Willingness-to-pay threshold.
    """
    
    def __init__(self, model, base_result: dict, base_params: dict,
                 owsa_data: list, wtp: float = 50000):
        self.model = model
        self.base_result = base_result
        self.base_params = base_params
        self.owsa_data = owsa_data
        self.wtp = wtp
    
    @staticmethod
    def _compute_icer(cost_int, cost_comp, qaly_int, qaly_comp):
        """Return a ratio only when the incremental quadrant permits one."""
        d_cost = cost_int - cost_comp
        d_qaly = qaly_int - qaly_comp
        return classify_incremental(
            d_cost, d_qaly,
            cost_tol=_scale_tol(cost_int, cost_comp),
            effect_tol=_scale_tol(qaly_int, qaly_comp),
        )[0]

    def summary(self, comparator: Optional[str] = None,
                intervention: Optional[str] = None,
                outcome: str = "nmb") -> pd.DataFrame:
        """Summarize OWSA results.

        OWSA is a pairwise comparison, so with more than two strategies the
        one being evaluated must be named explicitly.

        Parameters
        ----------
        comparator : str, optional
            Comparator strategy (default: first strategy).
        intervention : str, optional
            Strategy being evaluated against ``comparator``. Required when
            the model has more than two strategies; defaults to the other
            one otherwise.
        outcome : str
            "nmb" — rank by INMB range (default).
            "icer" — rank by ICER range (matches R heemod tornado).

        Returns
        -------
        pd.DataFrame
            Summary with parameter, low/high values, and outcomes.
        """
        if comparator is None:
            comparator = self.model.strategy_names[0]

        others = [s for s in self.model.strategy_names if s != comparator]
        if intervention is None:
            if len(others) > 1:
                raise ValueError(
                    "intervention must be given explicitly when the model "
                    f"has more than two strategies; choose from {others!r}"
                )
            intervention = others[0]
        elif intervention not in self.model.strategy_names:
            raise ValueError(
                f"Unknown intervention {intervention!r}; available strategies "
                f"are {self.model.strategy_names!r}"
            )
        elif intervention == comparator:
            raise ValueError("intervention must differ from comparator")

        # Base case values
        base_cost_comp = sum(self.base_result[comparator]['total_costs'].values())
        base_cost_int = sum(self.base_result[intervention]['total_costs'].values())
        base_qaly_comp = self.base_result[comparator]['total_qalys']
        base_qaly_int = self.base_result[intervention]['total_qalys']
        base_inmb = (base_qaly_int - base_qaly_comp) * self.wtp - (base_cost_int - base_cost_comp)
        base_icer = self._compute_icer(
            base_cost_int, base_cost_comp, base_qaly_int, base_qaly_comp
        )

        rows = []
        param_names = list(dict.fromkeys(d['param'] for d in self.owsa_data))

        for param_name in param_names:
            entries = {d['bound']: d for d in self.owsa_data if d['param'] == param_name}
            missing = {'low', 'high'} - set(entries)
            if missing:
                raise ValueError(
                    f"OWSA data for {param_name!r} is missing bound(s) "
                    f"{sorted(missing)!r}"
                )
            low_entry = entries['low']
            high_entry = entries['high']

            low_result = low_entry['result']
            high_result = high_entry['result']

            # Compute INMB for low and high
            low_cost_int = sum(low_result[intervention]['total_costs'].values())
            low_cost_comp = sum(low_result[comparator]['total_costs'].values())
            low_qaly_int = low_result[intervention]['total_qalys']
            low_qaly_comp = low_result[comparator]['total_qalys']
            low_inmb = (low_qaly_int - low_qaly_comp) * self.wtp - (low_cost_int - low_cost_comp)

            high_cost_int = sum(high_result[intervention]['total_costs'].values())
            high_cost_comp = sum(high_result[comparator]['total_costs'].values())
            high_qaly_int = high_result[intervention]['total_qalys']
            high_qaly_comp = high_result[comparator]['total_qalys']
            high_inmb = (high_qaly_int - high_qaly_comp) * self.wtp - (high_cost_int - high_cost_comp)

            # Compute ICER for low and high
            low_icer = self._compute_icer(
                low_cost_int, low_cost_comp, low_qaly_int, low_qaly_comp
            )
            high_icer = self._compute_icer(
                high_cost_int, high_cost_comp, high_qaly_int, high_qaly_comp
            )

            row = {
                'Parameter': low_entry['label'] or param_name,
                'param_name': param_name,
                'Base Value': low_entry['base_value'],
                'Low Value': low_entry['value'],
                'High Value': high_entry['value'],
                'INMB (Low)': low_inmb,
                'INMB (High)': high_inmb,
                'INMB (Base)': base_inmb,
                'ICER (Low)': low_icer,
                'ICER (High)': high_icer,
                'ICER (Base)': base_icer,
                'ICER Classification (Low)': classify_incremental(
                    low_cost_int - low_cost_comp, low_qaly_int - low_qaly_comp,
                    cost_tol=_scale_tol(low_cost_int, low_cost_comp),
                    effect_tol=_scale_tol(low_qaly_int, low_qaly_comp),
                )[1],
                'ICER Classification (High)': classify_incremental(
                    high_cost_int - high_cost_comp, high_qaly_int - high_qaly_comp,
                    cost_tol=_scale_tol(high_cost_int, high_cost_comp),
                    effect_tol=_scale_tol(high_qaly_int, high_qaly_comp),
                )[1],
                'ICER Classification (Base)': classify_incremental(
                    base_cost_int - base_cost_comp, base_qaly_int - base_qaly_comp,
                    cost_tol=_scale_tol(base_cost_int, base_cost_comp),
                    effect_tol=_scale_tol(base_qaly_int, base_qaly_comp),
                )[1],
            }

            if outcome == "icer":
                # A dominant or dominated bound has no ratio, so its range is
                # not a finite number. Rank it first: such a parameter can flip
                # the conclusion, unlike one with a merely wide range.
                if np.isnan(low_icer) or np.isnan(high_icer):
                    row['Range'] = float('inf')
                else:
                    row['Range'] = abs(high_icer - low_icer)
            else:
                row['Range'] = abs(high_inmb - low_inmb)

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('Range', ascending=False).reset_index(drop=True)
        return df
    
    # --- Plotting Shortcuts ---
    
    def plot_tornado(self, comparator=None, intervention=None, outcome="nmb", **kwargs):
        """Plot tornado diagram."""
        from ..plotting import plot_tornado
        return plot_tornado(
            self, comparator=comparator, intervention=intervention,
            outcome=outcome, **kwargs,
        )
    
    def plot_owsa(self, param_name: str, comparator=None, intervention=None,
                  **kwargs):
        """Plot one-way sensitivity for a specific parameter."""
        from ..plotting import plot_owsa_param
        return plot_owsa_param(
            self, param_name, comparator=comparator, intervention=intervention,
            **kwargs,
        )


class PSAResult:
    """Results from probabilistic sensitivity analysis.
    
    Attributes
    ----------
    model : MarkovModel
        The source model.
    psa_results : list
        List of simulation result dicts (one per PSA iteration).
    sampled_params : list
        List of parameter dicts used in each iteration.
    """
    
    def __init__(self, model, psa_results: list, sampled_params: list):
        self.model = model
        self.psa_results = psa_results
        self.sampled_params = sampled_params
        self._ce_table = None  # Cached
    
    @property
    def n_sim(self) -> int:
        return len(self.psa_results)
    
    @property
    def ce_table(self) -> pd.DataFrame:
        """Cost-effectiveness table with costs and QALYs for each simulation."""
        if self._ce_table is not None:
            return self._ce_table
        
        rows = []
        for i, result in enumerate(self.psa_results):
            for strategy in self.model.strategy_names:
                r = result[strategy]
                total_cost = sum(r['total_costs'].values())
                rows.append({
                    'sim': i + 1,
                    'strategy': strategy,
                    'strategy_label': self.model.strategy_labels[strategy],
                    'qalys': r['total_qalys'],
                    'lys': r['total_lys'],
                    'total_cost': total_cost,
                    **{f'cost_{cat}': r['total_costs'][cat] 
                       for cat in r['total_costs']},
                })
        
        self._ce_table = pd.DataFrame(rows)
        return self._ce_table
    
    def summary(self) -> pd.DataFrame:
        """Summarize PSA results with mean, SD, and credible intervals.

        Returns
        -------
        pd.DataFrame
            Summary statistics for each strategy.
        """
        ce = self.ce_table
        
        rows = []
        for strategy in self.model.strategy_names:
            df_s = ce[ce['strategy'] == strategy]
            row = {
                'Strategy': self.model.strategy_labels[strategy],
                'Mean QALYs': df_s['qalys'].mean(),
                'SD QALYs': df_s['qalys'].std(),
                'QALYs (2.5%)': df_s['qalys'].quantile(0.025),
                'QALYs (97.5%)': df_s['qalys'].quantile(0.975),
                'Mean Cost': df_s['total_cost'].mean(),
                'SD Cost': df_s['total_cost'].std(),
                'Cost (2.5%)': df_s['total_cost'].quantile(0.025),
                'Cost (97.5%)': df_s['total_cost'].quantile(0.975),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def icer(self, comparator: Optional[str] = None) -> pd.DataFrame:
        """Compute ICER with uncertainty from PSA.
        
        Parameters
        ----------
        comparator : str, optional
            Comparator strategy (default: first).
        
        Returns
        -------
        pd.DataFrame
            ICER table with mean incremental costs/QALYs and CI.
        """
        if comparator is None:
            comparator = self.model.strategy_names[0]
        
        ce = self.ce_table
        
        rows = []
        for strategy in self.model.strategy_names:
            if strategy == comparator:
                continue
            
            int_df, comp_df, inc_cost, inc_qaly = _paired_psa_icer(
                ce, strategy, comparator
            )
            
            mean_ic = inc_cost.mean()
            mean_iq = inc_qaly.mean()
            icer_val, classification = classify_incremental(
                mean_ic, mean_iq,
                cost_tol=_scale_tol(
                    int_df['total_cost'].mean(), comp_df['total_cost'].mean()
                ),
                effect_tol=_scale_tol(
                    int_df['qalys'].mean(), comp_df['qalys'].mean()
                ),
            )
            
            rows.append({
                'Strategy': self.model.strategy_labels[strategy],
                'vs': self.model.strategy_labels[comparator],
                'Mean Inc. Cost': mean_ic,
                'Inc. Cost (2.5%)': np.percentile(inc_cost, 2.5),
                'Inc. Cost (97.5%)': np.percentile(inc_cost, 97.5),
                'Mean Inc. QALYs': mean_iq,
                'Inc. QALYs (2.5%)': np.percentile(inc_qaly, 2.5),
                'Inc. QALYs (97.5%)': np.percentile(inc_qaly, 97.5),
                'ICER': icer_val,
                'ICER Classification': classification,
            })
        
        return pd.DataFrame(rows)
    
    def ceac_data(self, wtp_range: tuple = (0, 100000),
                  n_wtp: int = 200) -> pd.DataFrame:
        """Compute CEAC (cost-effectiveness acceptability curve) data.
        
        Parameters
        ----------
        wtp_range : tuple
            (min, max) WTP values.
        n_wtp : int
            Number of WTP points.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with WTP values and probability cost-effective per strategy.
        """
        wtp_values = np.linspace(wtp_range[0], wtp_range[1], n_wtp)
        ce = self.ce_table
        
        strategies = self.model.strategy_names
        n_sim = self.n_sim
        
        # Get costs and QALYs per sim per strategy
        cost_matrix = np.zeros((n_sim, len(strategies)))
        qaly_matrix = np.zeros((n_sim, len(strategies)))
        
        for j, strategy in enumerate(strategies):
            df_s = ce[ce['strategy'] == strategy].sort_values('sim')
            cost_matrix[:, j] = df_s['total_cost'].values
            qaly_matrix[:, j] = df_s['qalys'].values
        
        rows = []
        for wtp in wtp_values:
            # NMB for each strategy per simulation
            nmb_matrix = qaly_matrix * wtp - cost_matrix
            # Which strategy has max NMB in each sim?
            best = nmb_matrix.argmax(axis=1)
            
            for j, strategy in enumerate(strategies):
                prob = (best == j).mean()
                rows.append({
                    'WTP': wtp,
                    'Strategy': self.model.strategy_labels[strategy],
                    'strategy': strategy,
                    'Prob CE': prob,
                })
        
        return pd.DataFrame(rows)
    
    # --- Plotting Shortcuts ---
    
    def plot_ceac(self, wtp_range=(0, 100000), **kwargs):
        """Plot cost-effectiveness acceptability curve."""
        from ..plotting import plot_ceac
        return plot_ceac(self, wtp_range=wtp_range, **kwargs)
    
    def plot_scatter(self, comparator=None, wtp=None, **kwargs):
        """Plot CE scatter (incremental cost-effectiveness plane)."""
        from ..plotting import plot_scatter
        return plot_scatter(self, comparator=comparator, wtp=wtp, **kwargs)
    
    def plot_convergence(self, comparator=None, wtp=50000, **kwargs):
        """Plot PSA convergence (running mean of incremental NMB)."""
        from ..plotting import plot_convergence
        return plot_convergence(self, comparator=comparator, wtp=wtp, **kwargs)


# =============================================================================
# PSM Base Case Result
# =============================================================================

class PSMBaseResult(StrategyOutcomeResult):
    """Results from a PSM deterministic base case analysis.

    Extends BaseResult with PSM-specific features:
    - Survival curve data
    - Area-between-curves visualization
    - State occupancy from partitioned survival

    Attributes
    ----------
    model : PSMModel
        The source PSM model.
    results : dict
        Raw simulation results keyed by strategy.
    params : dict
        Parameter values used.
    """

    def __init__(self, model, results: dict, params: dict):
        self.model = model
        self.results = results
        self.params = params

    def summary(self) -> pd.DataFrame:
        """Summarize total costs and QALYs per strategy."""
        rows = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            row = {
                'Strategy': self.model.strategy_labels[strategy],
                'LYs': r['total_lys'],
                'QALYs': r['total_qalys'],
            }
            total_cost = 0.0
            for cat in r['total_costs']:
                row[f'Cost ({cat})'] = r['total_costs'][cat]
                total_cost += r['total_costs'][cat]
            row['Total Cost'] = total_cost
            rows.append(row)
        return pd.DataFrame(rows)

    def _totals(self, strategy: str):
        r = self.results[strategy]
        return (
            sum(r['total_costs'].values()), r['total_qalys'], r['total_lys']
        )

    @property
    def state_trace(self) -> pd.DataFrame:
        """Get state occupancy over time as DataFrame."""
        dfs = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            trace = r['trace']
            df = pd.DataFrame(trace, columns=self.model.states)
            df.insert(0, 'Cycle', np.arange(self.model.n_cycles + 1))
            df.insert(1, 'Time', r['times'])
            df.insert(2, 'Strategy', self.model.strategy_labels[strategy])
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    @property
    def survival_data(self) -> pd.DataFrame:
        """Get survival curves data as DataFrame."""
        rows = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            times = r['times']
            for endpoint in self.model.survival_endpoints:
                s = r['survival_curves'][endpoint]
                for t_idx in range(len(times)):
                    rows.append({
                        'Time': times[t_idx],
                        'Cycle': t_idx,
                        'Strategy': self.model.strategy_labels[strategy],
                        'strategy': strategy,
                        'Endpoint': endpoint,
                        'Survival': s[t_idx],
                    })
        return pd.DataFrame(rows)

    # --- Plotting Shortcuts ---

    def plot_survival(self, **kwargs):
        """Plot survival curves."""
        from ..plotting import plot_survival_curves
        return plot_survival_curves(self, **kwargs)

    def plot_state_area(self, **kwargs):
        """Plot area-between-curves (state occupancy)."""
        from ..plotting import plot_state_area
        return plot_state_area(self, **kwargs)

    def plot_trace(self, **kwargs):
        """Plot state occupancy as line plot."""
        from ..plotting import plot_psm_trace
        return plot_psm_trace(self, **kwargs)


# =============================================================================
# Microsimulation Results
# =============================================================================

class MicroSimResult(StrategyOutcomeResult):
    """Results from a microsimulation base case analysis.

    Stores per-patient outcomes and provides summary statistics,
    ICER, survival curves, and state traces.

    Attributes
    ----------
    model : MicroSimModel
        The source model.
    results : dict
        Per-strategy simulation results containing individual-level data.
    params : dict
        Parameter values used.
    """

    _EFFECT_LABEL = "Mean QALYs"
    _COST_LABEL = "Mean Cost"

    def __init__(self, model, results: dict, params: dict):
        self.model = model
        self.results = results
        self.params = params

    def summary(self) -> pd.DataFrame:
        """Summary table with mean costs, QALYs, and confidence intervals."""
        from scipy import stats as sp_stats
        rows = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            costs = r['total_cost']
            qalys = r['total_qalys']
            lys = r['total_lys']
            n = len(costs)

            # 95% CI via t-distribution
            ci_mult = sp_stats.t.ppf(0.975, n - 1)

            row = {
                'Strategy': self.model.strategy_labels[strategy],
                'N': n,
                'Mean Cost': costs.mean(),
                'SD Cost': costs.std(ddof=1),
                'Cost (2.5%)': np.percentile(costs, 2.5),
                'Cost (97.5%)': np.percentile(costs, 97.5),
                'Mean QALYs': qalys.mean(),
                'SD QALYs': qalys.std(ddof=1),
                'QALYs (2.5%)': np.percentile(qalys, 2.5),
                'QALYs (97.5%)': np.percentile(qalys, 97.5),
                'Mean LYs': lys.mean(),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _totals(self, strategy: str):
        r = self.results[strategy]
        return r['mean_cost'], r['mean_qalys'], r['mean_lys']

    @property
    def patient_outcomes(self) -> pd.DataFrame:
        """Per-patient outcomes for all strategies."""
        dfs = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            n = len(r['total_cost'])
            df = pd.DataFrame({
                'Patient': np.arange(1, n + 1),
                'Strategy': self.model.strategy_labels[strategy],
                'Total Cost': r['total_cost'],
                'Total QALYs': r['total_qalys'],
                'Total LYs': r['total_lys'],
                'Years Alive': r['time_alive'],
            })
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    @property
    def markov_trace(self) -> pd.DataFrame:
        """Mean state occupancy trace (like cohort trace)."""
        dfs = []
        for strategy in self.model.strategy_names:
            trace = self.results[strategy]['trace']
            df = pd.DataFrame(trace, columns=self.model.states)
            df.insert(0, 'Cycle', np.arange(self.model.n_cycles + 1))
            df.insert(1, 'Strategy', self.model.strategy_labels[strategy])
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def survival_curve(self, strategy: Optional[str] = None) -> pd.DataFrame:
        """Compute empirical survival curve (proportion alive over time).

        Parameters
        ----------
        strategy : str, optional
            Specific strategy. Default: all.
        """
        strategies = [strategy] if strategy else self.model.strategy_names
        rows = []
        for strat in strategies:
            trace = self.results[strat]['trace']
            # Sum alive state columns
            alive_cols = [i for i in self.model._alive_states]
            surv = trace[:, alive_cols].sum(axis=1)
            for t in range(len(surv)):
                rows.append({
                    'Cycle': t,
                    'Time': t * self.model.cycle_length,
                    'Strategy': self.model.strategy_labels[strat],
                    'Survival': surv[t],
                })
        return pd.DataFrame(rows)

    # --- Plotting Shortcuts ---

    def plot_trace(self, **kwargs):
        """Plot state occupancy trace."""
        from ..plotting import plot_microsim_trace
        return plot_microsim_trace(self, **kwargs)

    def plot_survival(self, **kwargs):
        """Plot empirical survival curves."""
        from ..plotting import plot_microsim_survival
        return plot_microsim_survival(self, **kwargs)

    def plot_outcomes_histogram(self, **kwargs):
        """Plot distribution of patient-level outcomes."""
        from ..plotting import plot_microsim_outcomes
        return plot_microsim_outcomes(self, **kwargs)


class MicroSimPSAResult:
    """Results from microsimulation PSA (outer × inner loop).

    Attributes
    ----------
    model : MicroSimModel
        The source model.
    psa_results : list of dict
        Each element is a {strategy: sim_result} dict for one PSA iteration.
    sampled_params : list of dict
        Parameter dicts used in each PSA iteration.
    """

    def __init__(self, model, psa_results: list, sampled_params: list):
        self.model = model
        self.psa_results = psa_results
        self.sampled_params = sampled_params
        self._ce_table = None

    @property
    def n_outer(self) -> int:
        return len(self.psa_results)

    #: Alias for n_outer, matching the cohort engines' PSAResult.n_sim.
    n_sim = n_outer

    @property
    def ce_table(self) -> pd.DataFrame:
        """Cost-effectiveness table across all PSA iterations."""
        if self._ce_table is not None:
            return self._ce_table

        rows = []
        for i, result in enumerate(self.psa_results):
            for strategy in self.model.strategy_names:
                r = result[strategy]
                rows.append({
                    'sim': i + 1,
                    'strategy': strategy,
                    'strategy_label': self.model.strategy_labels[strategy],
                    'qalys': r['mean_qalys'],
                    'lys': r['mean_lys'],
                    'total_cost': r['mean_cost'],
                })
        self._ce_table = pd.DataFrame(rows)
        return self._ce_table

    def summary(self) -> pd.DataFrame:
        """Summary statistics across PSA iterations."""
        ce = self.ce_table
        rows = []
        for strategy in self.model.strategy_names:
            df_s = ce[ce['strategy'] == strategy]
            row = {
                'Strategy': self.model.strategy_labels[strategy],
                'Mean QALYs': df_s['qalys'].mean(),
                'SD QALYs': df_s['qalys'].std(),
                'QALYs (2.5%)': df_s['qalys'].quantile(0.025),
                'QALYs (97.5%)': df_s['qalys'].quantile(0.975),
                'Mean Cost': df_s['total_cost'].mean(),
                'SD Cost': df_s['total_cost'].std(),
                'Cost (2.5%)': df_s['total_cost'].quantile(0.025),
                'Cost (97.5%)': df_s['total_cost'].quantile(0.975),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def icer(self, comparator: Optional[str] = None) -> pd.DataFrame:
        """Compute ICER from PSA results."""
        if comparator is None:
            comparator = self.model.strategy_names[0]

        ce = self.ce_table

        rows = []
        for strategy in self.model.strategy_names:
            if strategy == comparator:
                continue
            int_df, comp_df, inc_cost, inc_qaly = _paired_psa_icer(
                ce, strategy, comparator
            )

            mean_ic = inc_cost.mean()
            mean_iq = inc_qaly.mean()
            icer_val, classification = classify_incremental(
                mean_ic, mean_iq,
                cost_tol=_scale_tol(
                    int_df['total_cost'].mean(), comp_df['total_cost'].mean()
                ),
                effect_tol=_scale_tol(
                    int_df['qalys'].mean(), comp_df['qalys'].mean()
                ),
            )

            rows.append({
                'Strategy': self.model.strategy_labels[strategy],
                'vs': self.model.strategy_labels[comparator],
                'Mean Inc. Cost': mean_ic,
                'Inc. Cost (2.5%)': np.percentile(inc_cost, 2.5),
                'Inc. Cost (97.5%)': np.percentile(inc_cost, 97.5),
                'Mean Inc. QALYs': mean_iq,
                'Inc. QALYs (2.5%)': np.percentile(inc_qaly, 2.5),
                'Inc. QALYs (97.5%)': np.percentile(inc_qaly, 97.5),
                'ICER': icer_val,
                'ICER Classification': classification,
            })
        return pd.DataFrame(rows)

    def ceac_data(self, wtp_range: tuple = (0, 100000),
                  n_wtp: int = 200) -> pd.DataFrame:
        """Compute CEAC data."""
        wtp_values = np.linspace(wtp_range[0], wtp_range[1], n_wtp)
        ce = self.ce_table
        strategies = self.model.strategy_names
        n_sim = self.n_outer

        cost_matrix = np.zeros((n_sim, len(strategies)))
        qaly_matrix = np.zeros((n_sim, len(strategies)))
        for j, strategy in enumerate(strategies):
            df_s = ce[ce['strategy'] == strategy].sort_values('sim')
            cost_matrix[:, j] = df_s['total_cost'].values
            qaly_matrix[:, j] = df_s['qalys'].values

        rows = []
        for wtp in wtp_values:
            nmb_matrix = qaly_matrix * wtp - cost_matrix
            best = nmb_matrix.argmax(axis=1)
            for j, strategy in enumerate(strategies):
                prob = (best == j).mean()
                rows.append({
                    'WTP': wtp,
                    'Strategy': self.model.strategy_labels[strategy],
                    'strategy': strategy,
                    'Prob CE': prob,
                })
        return pd.DataFrame(rows)

    # --- Plotting Shortcuts ---

    def plot_ceac(self, wtp_range=(0, 100000), **kwargs):
        """Plot CEAC."""
        from ..plotting import plot_ceac
        return plot_ceac(self, wtp_range=wtp_range, **kwargs)

    def plot_scatter(self, comparator=None, wtp=None, **kwargs):
        """Plot CE scatter."""
        from ..plotting import plot_scatter
        return plot_scatter(self, comparator=comparator, wtp=wtp, **kwargs)


# =============================================================================
# DES Results
# =============================================================================

class DESResult(StrategyOutcomeResult):
    """Results from a DES base case analysis.

    Stores per-patient outcomes and provides summary statistics,
    ICER, NMB, event logs, and time-in-state information.

    Attributes
    ----------
    model : DESModel
        The source model.
    results : dict
        Per-strategy simulation results containing individual-level data.
    params : dict
        Parameter values used.
    """

    _EFFECT_LABEL = "Mean QALYs"
    _COST_LABEL = "Mean Cost"

    def __init__(self, model, results: dict, params: dict):
        self.model = model
        self.results = results
        self.params = params

    def summary(self) -> pd.DataFrame:
        """Summary table with mean costs, QALYs, and confidence intervals."""
        from scipy import stats as sp_stats
        rows = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            costs = r['total_cost']
            qalys = r['total_qalys']
            lys = r['total_lys']
            n = len(costs)

            row = {
                'Strategy': self.model.strategy_labels[strategy],
                'N': n,
                'Mean Cost': costs.mean(),
                'SD Cost': costs.std(ddof=1),
                'Cost (2.5%)': np.percentile(costs, 2.5),
                'Cost (97.5%)': np.percentile(costs, 97.5),
                'Mean QALYs': qalys.mean(),
                'SD QALYs': qalys.std(ddof=1),
                'QALYs (2.5%)': np.percentile(qalys, 2.5),
                'QALYs (97.5%)': np.percentile(qalys, 97.5),
                'Mean LYs': lys.mean(),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _totals(self, strategy: str):
        r = self.results[strategy]
        return r['mean_cost'], r['mean_qalys'], r['mean_lys']

    @property
    def patient_outcomes(self) -> pd.DataFrame:
        """Per-patient outcomes for all strategies."""
        dfs = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            n = r['n_patients']
            df = pd.DataFrame({
                'Patient': np.arange(1, n + 1),
                'Strategy': self.model.strategy_labels[strategy],
                'Total Cost': r['total_cost'],
                'Total QALYs': r['total_qalys'],
                'Total LYs': r['total_lys'],
            })
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    @property
    def event_log(self) -> pd.DataFrame:
        """Consolidated event log for all patients and strategies.

        Returns
        -------
        pd.DataFrame
            Columns: Patient, Strategy, Time, From, To
        """
        rows = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            for i, pr in enumerate(r['patient_results']):
                for time, from_s, to_s in pr['event_log']:
                    rows.append({
                        'Patient': i + 1,
                        'Strategy': self.model.strategy_labels[strategy],
                        'Time': time,
                        'From': from_s,
                        'To': to_s,
                    })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['Patient', 'Strategy', 'Time', 'From', 'To'])

    @property
    def time_in_state(self) -> pd.DataFrame:
        """Mean time in state for each strategy.

        Returns
        -------
        pd.DataFrame
            Columns: Strategy, State, Mean Time, Median Time, SD Time
        """
        rows = []
        for strategy in self.model.strategy_names:
            tis = self.results[strategy]['time_in_state']
            for state in self.model.states:
                arr = tis[state]
                rows.append({
                    'Strategy': self.model.strategy_labels[strategy],
                    'State': state,
                    'Mean Time': arr.mean(),
                    'Median Time': np.median(arr),
                    'SD Time': arr.std(ddof=1),
                })
        return pd.DataFrame(rows)

    @property
    def costs_by_category(self) -> pd.DataFrame:
        """Mean costs by category for each strategy."""
        rows = []
        for strategy in self.model.strategy_names:
            cats = self.results[strategy]['costs_by_cat']
            for cat, arr in cats.items():
                rows.append({
                    'Strategy': self.model.strategy_labels[strategy],
                    'Category': cat,
                    'Mean Cost': arr.mean(),
                    'SD Cost': arr.std(ddof=1),
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['Strategy', 'Category', 'Mean Cost', 'SD Cost'])

    def survival_curve(
        self, strategy: Optional[str] = None, n_points: int = 200,
    ) -> pd.DataFrame:
        """Compute empirical (Kaplan-Meier-like) survival curve from event logs.

        Survival is defined as proportion of patients not yet in an
        absorbing state at each time point.

        Parameters
        ----------
        strategy : str, optional
            Specific strategy. Default: all strategies.
        n_points : int
            Number of time grid points.

        Returns
        -------
        pd.DataFrame
            Columns: Time, Strategy, Survival
        """
        strategies = [strategy] if strategy else self.model.strategy_names
        time_grid = np.linspace(0, self.model.time_horizon, n_points)
        absorbing = self.model._absorbing

        rows = []
        for strat in strategies:
            pr_list = self.results[strat]['patient_results']
            n = len(pr_list)

            # For each patient, determine the time of entering an absorbing state
            absorb_times = []
            for pr in pr_list:
                t_absorb = self.model.time_horizon  # censored
                for t_ev, from_s, to_s in pr['event_log']:
                    to_idx = self.model.states.index(to_s)
                    if to_idx in absorbing:
                        t_absorb = t_ev
                        break
                absorb_times.append(t_absorb)

            absorb_times = np.array(absorb_times)

            for t in time_grid:
                # Patients censored at the horizon remain in the risk set at
                # the endpoint. Compare against the grid's own last value
                # (exactly time_horizon by construction) rather than
                # np.isclose, whose default relative tolerance would cover
                # several trailing grid points for a large time_horizon.
                if t == time_grid[-1]:
                    surv = (absorb_times >= t).mean()
                else:
                    surv = (absorb_times > t).mean()
                rows.append({
                    'Time': t,
                    'Strategy': self.model.strategy_labels[strat],
                    'Survival': surv,
                })

        return pd.DataFrame(rows)

    # --- Plotting Shortcuts ---

    def plot_survival(self, **kwargs):
        """Plot empirical survival curves from the event log."""
        from ..plotting import plot_microsim_survival
        return plot_microsim_survival(self, **kwargs)

    def plot_outcomes_histogram(self, outcome: str = "qalys", **kwargs):
        """Plot a histogram of per-patient outcomes."""
        from ..plotting import plot_microsim_outcomes
        return plot_microsim_outcomes(self, outcome=outcome, **kwargs)


class DESPSAResult:
    """Results from DES probabilistic sensitivity analysis.

    Attributes
    ----------
    model : DESModel
        The source model.
    psa_iterations : list of dict
        Each element: {strategy: {mean_cost, mean_qalys, mean_lys}}.
    sampled_params : list of dict
        Parameter dicts used in each PSA iteration.
    """

    def __init__(self, model, psa_iterations: list, sampled_params: list):
        self.model = model
        self.psa_iterations = psa_iterations
        self.sampled_params = sampled_params
        self._ce_table = None

    @property
    def n_outer(self) -> int:
        return len(self.psa_iterations)

    #: Alias for n_outer, matching the cohort engines' PSAResult.n_sim.
    n_sim = n_outer

    @property
    def ce_table(self) -> pd.DataFrame:
        """Cost-effectiveness table across all PSA iterations."""
        if self._ce_table is not None:
            return self._ce_table

        rows = []
        for i, result in enumerate(self.psa_iterations):
            for strategy in self.model.strategy_names:
                r = result[strategy]
                rows.append({
                    'sim': i + 1,
                    'strategy': strategy,
                    'strategy_label': self.model.strategy_labels[strategy],
                    'qalys': r['mean_qalys'],
                    'lys': r['mean_lys'],
                    'total_cost': r['mean_cost'],
                })
        self._ce_table = pd.DataFrame(rows)
        return self._ce_table

    def summary(self) -> pd.DataFrame:
        """Summary statistics across PSA iterations."""
        ce = self.ce_table
        rows = []
        for strategy in self.model.strategy_names:
            df_s = ce[ce['strategy'] == strategy]
            row = {
                'Strategy': self.model.strategy_labels[strategy],
                'Mean QALYs': df_s['qalys'].mean(),
                'SD QALYs': df_s['qalys'].std(),
                'QALYs (2.5%)': df_s['qalys'].quantile(0.025),
                'QALYs (97.5%)': df_s['qalys'].quantile(0.975),
                'Mean Cost': df_s['total_cost'].mean(),
                'SD Cost': df_s['total_cost'].std(),
                'Cost (2.5%)': df_s['total_cost'].quantile(0.025),
                'Cost (97.5%)': df_s['total_cost'].quantile(0.975),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def icer(self, comparator: Optional[str] = None) -> pd.DataFrame:
        """Compute ICER from PSA results."""
        if comparator is None:
            comparator = self.model.strategy_names[0]

        ce = self.ce_table

        rows = []
        for strategy in self.model.strategy_names:
            if strategy == comparator:
                continue
            int_df, comp_df, inc_cost, inc_qaly = _paired_psa_icer(
                ce, strategy, comparator
            )

            mean_ic = inc_cost.mean()
            mean_iq = inc_qaly.mean()
            icer_val, classification = classify_incremental(
                mean_ic, mean_iq,
                cost_tol=_scale_tol(
                    int_df['total_cost'].mean(), comp_df['total_cost'].mean()
                ),
                effect_tol=_scale_tol(
                    int_df['qalys'].mean(), comp_df['qalys'].mean()
                ),
            )

            rows.append({
                'Strategy': self.model.strategy_labels[strategy],
                'vs': self.model.strategy_labels[comparator],
                'Mean Inc. Cost': mean_ic,
                'Inc. Cost (2.5%)': np.percentile(inc_cost, 2.5),
                'Inc. Cost (97.5%)': np.percentile(inc_cost, 97.5),
                'Mean Inc. QALYs': mean_iq,
                'Inc. QALYs (2.5%)': np.percentile(inc_qaly, 2.5),
                'Inc. QALYs (97.5%)': np.percentile(inc_qaly, 97.5),
                'ICER': icer_val,
                'ICER Classification': classification,
            })
        return pd.DataFrame(rows)

    def ceac_data(self, wtp_range: tuple = (0, 100000),
                  n_wtp: int = 200) -> pd.DataFrame:
        """Compute CEAC data."""
        wtp_values = np.linspace(wtp_range[0], wtp_range[1], n_wtp)
        ce = self.ce_table
        strategies = self.model.strategy_names
        n_sim = self.n_outer

        cost_matrix = np.zeros((n_sim, len(strategies)))
        qaly_matrix = np.zeros((n_sim, len(strategies)))
        for j, strategy in enumerate(strategies):
            df_s = ce[ce['strategy'] == strategy].sort_values('sim')
            cost_matrix[:, j] = df_s['total_cost'].values
            qaly_matrix[:, j] = df_s['qalys'].values

        rows = []
        for wtp in wtp_values:
            nmb_matrix = qaly_matrix * wtp - cost_matrix
            best = nmb_matrix.argmax(axis=1)
            for j, strategy in enumerate(strategies):
                prob = (best == j).mean()
                rows.append({
                    'WTP': wtp,
                    'Strategy': self.model.strategy_labels[strategy],
                    'strategy': strategy,
                    'Prob CE': prob,
                })
        return pd.DataFrame(rows)

    # --- Plotting Shortcuts ---

    def plot_ceac(self, wtp_range=(0, 100000), **kwargs):
        """Plot CEAC."""
        from ..plotting import plot_ceac
        return plot_ceac(self, wtp_range=wtp_range, **kwargs)

    def plot_scatter(self, comparator=None, wtp=None, **kwargs):
        """Plot CE scatter."""
        from ..plotting import plot_scatter
        return plot_scatter(self, comparator=comparator, wtp=wtp, **kwargs)
