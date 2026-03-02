"""
Result classes for base case, OWSA, and PSA analyses.

Each result class provides summary tables, ICER computation, and
convenient access to plotting methods.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


class BaseResult:
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
    
    def icer(self, comparator: Optional[str] = None) -> pd.DataFrame:
        """Compute incremental cost-effectiveness ratio (ICER).
        
        Parameters
        ----------
        comparator : str, optional
            Comparator strategy (default: first strategy).
        
        Returns
        -------
        pd.DataFrame
            ICER table with incremental costs, QALYs, and ICER.
        """
        if comparator is None:
            comparator = self.model.strategy_names[0]
        
        comp = self.results[comparator]
        comp_cost = sum(comp['total_costs'].values())
        comp_qaly = comp['total_qalys']
        comp_ly = comp['total_lys']
        
        rows = []
        for strategy in self.model.strategy_names:
            if strategy == comparator:
                continue
            r = self.results[strategy]
            total_cost = sum(r['total_costs'].values())
            inc_cost = total_cost - comp_cost
            inc_qaly = r['total_qalys'] - comp_qaly
            inc_ly = r['total_lys'] - comp_ly
            
            if abs(inc_qaly) < 1e-10:
                icer_val = float('inf') if inc_cost > 0 else float('-inf')
                icer_str = "Dominated" if inc_cost > 0 else "Dominant"
            elif inc_cost < 0 and inc_qaly > 0:
                icer_val = inc_cost / inc_qaly
                icer_str = "Dominant"
            else:
                icer_val = inc_cost / inc_qaly
                icer_str = f"{icer_val:,.0f}"
            
            rows.append({
                'Strategy': self.model.strategy_labels[strategy],
                'vs': self.model.strategy_labels[comparator],
                'Incremental Cost': inc_cost,
                'Incremental QALYs': inc_qaly,
                'Incremental LYs': inc_ly,
                'ICER ($/QALY)': icer_val,
                'ICER': icer_str,
            })
        
        return pd.DataFrame(rows)
    
    def nmb(self, wtp: float = 50000, comparator: Optional[str] = None) -> pd.DataFrame:
        """Compute net monetary benefit (NMB).
        
        Parameters
        ----------
        wtp : float
            Willingness-to-pay per QALY.
        comparator : str, optional
            Comparator strategy.
        
        Returns
        -------
        pd.DataFrame
            NMB table.
        """
        if comparator is None:
            comparator = self.model.strategy_names[0]
        
        comp = self.results[comparator]
        comp_cost = sum(comp['total_costs'].values())
        comp_qaly = comp['total_qalys']
        
        rows = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            total_cost = sum(r['total_costs'].values())
            nmb_val = r['total_qalys'] * wtp - total_cost
            
            row = {
                'Strategy': self.model.strategy_labels[strategy],
                'QALYs': r['total_qalys'],
                'Total Cost': total_cost,
                'NMB': nmb_val,
            }
            
            if strategy != comparator:
                inc_qaly = r['total_qalys'] - comp_qaly
                inc_cost = total_cost - comp_cost
                row['Incremental NMB'] = inc_qaly * wtp - inc_cost
            else:
                row['Incremental NMB'] = 0.0
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
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
        """Compute ICER, returning inf when ΔQALYs ≈ 0."""
        d_cost = cost_int - cost_comp
        d_qaly = qaly_int - qaly_comp
        if abs(d_qaly) < 1e-12:
            return float('inf') if d_cost > 0 else float('-inf')
        return d_cost / d_qaly

    def summary(self, comparator: Optional[str] = None,
                outcome: str = "nmb") -> pd.DataFrame:
        """Summarize OWSA results.

        Parameters
        ----------
        comparator : str, optional
            Comparator strategy (default: first strategy).
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

        # Determine the intervention strategy
        intervention = [s for s in self.model.strategy_names if s != comparator][0]

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
            entries = [d for d in self.owsa_data if d['param'] == param_name]
            low_entry = next(d for d in entries if d['bound'] == 'low')
            high_entry = next(d for d in entries if d['bound'] == 'high')

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
            }

            if outcome == "icer":
                # Rank by ICER range (absolute difference)
                if float('inf') in (abs(low_icer), abs(high_icer)):
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
    
    def plot_tornado(self, comparator=None, outcome="nmb", **kwargs):
        """Plot tornado diagram."""
        from ..plotting import plot_tornado
        return plot_tornado(self, comparator=comparator, outcome=outcome, **kwargs)
    
    def plot_owsa(self, param_name: str, comparator=None, **kwargs):
        """Plot one-way sensitivity for a specific parameter."""
        from ..plotting import plot_owsa_param
        return plot_owsa_param(self, param_name, comparator=comparator, **kwargs)


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
    
    def summary(self, comparator: Optional[str] = None) -> pd.DataFrame:
        """Summarize PSA results with mean, SD, and credible intervals.
        
        Parameters
        ----------
        comparator : str, optional
            Comparator strategy for incremental analysis.
        
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
        comp_df = ce[ce['strategy'] == comparator]
        
        rows = []
        for strategy in self.model.strategy_names:
            if strategy == comparator:
                continue
            
            int_df = ce[ce['strategy'] == strategy]
            
            inc_cost = int_df['total_cost'].values - comp_df['total_cost'].values
            inc_qaly = int_df['qalys'].values - comp_df['qalys'].values
            
            mean_ic = inc_cost.mean()
            mean_iq = inc_qaly.mean()
            icer_val = mean_ic / mean_iq if abs(mean_iq) > 1e-10 else float('inf')
            
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
            })
        
        return pd.DataFrame(rows)
    
    def ceac_data(self, comparator: Optional[str] = None,
                  wtp_range: tuple = (0, 100000),
                  n_wtp: int = 200) -> pd.DataFrame:
        """Compute CEAC (cost-effectiveness acceptability curve) data.
        
        Parameters
        ----------
        comparator : str, optional
            Comparator strategy.
        wtp_range : tuple
            (min, max) WTP values.
        n_wtp : int
            Number of WTP points.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with WTP values and probability cost-effective per strategy.
        """
        if comparator is None:
            comparator = self.model.strategy_names[0]
        
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
    
    def plot_ceac(self, comparator=None, wtp_range=(0, 100000), **kwargs):
        """Plot cost-effectiveness acceptability curve."""
        from ..plotting import plot_ceac
        return plot_ceac(self, comparator=comparator, wtp_range=wtp_range, **kwargs)
    
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

class PSMBaseResult:
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

    def icer(self, comparator: Optional[str] = None) -> pd.DataFrame:
        """Compute ICER."""
        if comparator is None:
            comparator = self.model.strategy_names[0]

        comp = self.results[comparator]
        comp_cost = sum(comp['total_costs'].values())
        comp_qaly = comp['total_qalys']
        comp_ly = comp['total_lys']

        rows = []
        for strategy in self.model.strategy_names:
            if strategy == comparator:
                continue
            r = self.results[strategy]
            total_cost = sum(r['total_costs'].values())
            inc_cost = total_cost - comp_cost
            inc_qaly = r['total_qalys'] - comp_qaly
            inc_ly = r['total_lys'] - comp_ly

            if abs(inc_qaly) < 1e-10:
                icer_val = float('inf') if inc_cost > 0 else float('-inf')
                icer_str = "Dominated" if inc_cost > 0 else "Dominant"
            elif inc_cost < 0 and inc_qaly > 0:
                icer_val = inc_cost / inc_qaly
                icer_str = "Dominant"
            else:
                icer_val = inc_cost / inc_qaly
                icer_str = f"{icer_val:,.0f}"

            rows.append({
                'Strategy': self.model.strategy_labels[strategy],
                'vs': self.model.strategy_labels[comparator],
                'Incremental Cost': inc_cost,
                'Incremental QALYs': inc_qaly,
                'Incremental LYs': inc_ly,
                'ICER ($/QALY)': icer_val,
                'ICER': icer_str,
            })
        return pd.DataFrame(rows)

    def nmb(self, wtp: float = 50000, comparator: Optional[str] = None) -> pd.DataFrame:
        """Compute net monetary benefit."""
        if comparator is None:
            comparator = self.model.strategy_names[0]

        comp = self.results[comparator]
        comp_cost = sum(comp['total_costs'].values())
        comp_qaly = comp['total_qalys']

        rows = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            total_cost = sum(r['total_costs'].values())
            nmb_val = r['total_qalys'] * wtp - total_cost

            row = {
                'Strategy': self.model.strategy_labels[strategy],
                'QALYs': r['total_qalys'],
                'Total Cost': total_cost,
                'NMB': nmb_val,
            }
            if strategy != comparator:
                inc_qaly = r['total_qalys'] - comp_qaly
                inc_cost = total_cost - comp_cost
                row['Incremental NMB'] = inc_qaly * wtp - inc_cost
            else:
                row['Incremental NMB'] = 0.0
            rows.append(row)
        return pd.DataFrame(rows)

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

class MicroSimResult:
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

    def icer(self, comparator: Optional[str] = None) -> pd.DataFrame:
        """Compute ICER from mean patient-level outcomes."""
        if comparator is None:
            comparator = self.model.strategy_names[0]

        comp = self.results[comparator]
        comp_cost = comp['mean_cost']
        comp_qaly = comp['mean_qalys']
        comp_ly = comp['mean_lys']

        rows = []
        for strategy in self.model.strategy_names:
            if strategy == comparator:
                continue
            r = self.results[strategy]
            inc_cost = r['mean_cost'] - comp_cost
            inc_qaly = r['mean_qalys'] - comp_qaly
            inc_ly = r['mean_lys'] - comp_ly

            if abs(inc_qaly) < 1e-10:
                icer_val = float('inf') if inc_cost > 0 else float('-inf')
                icer_str = "Dominated" if inc_cost > 0 else "Dominant"
            elif inc_cost < 0 and inc_qaly > 0:
                icer_val = inc_cost / inc_qaly
                icer_str = "Dominant"
            else:
                icer_val = inc_cost / inc_qaly
                icer_str = f"{icer_val:,.0f}"

            rows.append({
                'Strategy': self.model.strategy_labels[strategy],
                'vs': self.model.strategy_labels[comparator],
                'Incremental Cost': inc_cost,
                'Incremental QALYs': inc_qaly,
                'Incremental LYs': inc_ly,
                'ICER ($/QALY)': icer_val,
                'ICER': icer_str,
            })

        return pd.DataFrame(rows)

    def nmb(self, wtp: float = 50000, comparator: Optional[str] = None) -> pd.DataFrame:
        """Compute net monetary benefit."""
        if comparator is None:
            comparator = self.model.strategy_names[0]

        comp = self.results[comparator]
        rows = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            nmb_val = r['mean_qalys'] * wtp - r['mean_cost']
            row = {
                'Strategy': self.model.strategy_labels[strategy],
                'Mean QALYs': r['mean_qalys'],
                'Mean Cost': r['mean_cost'],
                'NMB': nmb_val,
            }
            if strategy != comparator:
                inc_q = r['mean_qalys'] - comp['mean_qalys']
                inc_c = r['mean_cost'] - comp['mean_cost']
                row['INMB'] = inc_q * wtp - inc_c
            else:
                row['INMB'] = 0.0
            rows.append(row)
        return pd.DataFrame(rows)

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
                'Alive Cycles': r['alive_cycles'],
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

    def summary(self, comparator: Optional[str] = None) -> pd.DataFrame:
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
        comp_df = ce[ce['strategy'] == comparator]

        rows = []
        for strategy in self.model.strategy_names:
            if strategy == comparator:
                continue
            int_df = ce[ce['strategy'] == strategy]
            inc_cost = int_df['total_cost'].values - comp_df['total_cost'].values
            inc_qaly = int_df['qalys'].values - comp_df['qalys'].values

            mean_ic = inc_cost.mean()
            mean_iq = inc_qaly.mean()
            icer_val = mean_ic / mean_iq if abs(mean_iq) > 1e-10 else float('inf')

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
            })
        return pd.DataFrame(rows)

    def ceac_data(self, comparator: Optional[str] = None,
                  wtp_range: tuple = (0, 100000),
                  n_wtp: int = 200) -> pd.DataFrame:
        """Compute CEAC data."""
        if comparator is None:
            comparator = self.model.strategy_names[0]

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

    def plot_ceac(self, comparator=None, wtp_range=(0, 100000), **kwargs):
        """Plot CEAC."""
        from ..plotting import plot_ceac
        return plot_ceac(self, comparator=comparator, wtp_range=wtp_range, **kwargs)

    def plot_scatter(self, comparator=None, wtp=None, **kwargs):
        """Plot CE scatter."""
        from ..plotting import plot_scatter
        return plot_scatter(self, comparator=comparator, wtp=wtp, **kwargs)


# =============================================================================
# DES Results
# =============================================================================

class DESResult:
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

    def icer(self, comparator: Optional[str] = None) -> pd.DataFrame:
        """Compute ICER from mean patient-level outcomes."""
        if comparator is None:
            comparator = self.model.strategy_names[0]

        comp = self.results[comparator]
        comp_cost = comp['mean_cost']
        comp_qaly = comp['mean_qalys']
        comp_ly = comp['mean_lys']

        rows = []
        for strategy in self.model.strategy_names:
            if strategy == comparator:
                continue
            r = self.results[strategy]
            inc_cost = r['mean_cost'] - comp_cost
            inc_qaly = r['mean_qalys'] - comp_qaly
            inc_ly = r['mean_lys'] - comp_ly

            if abs(inc_qaly) < 1e-10:
                icer_val = float('inf') if inc_cost > 0 else float('-inf')
                icer_str = "Dominated" if inc_cost > 0 else "Dominant"
            elif inc_cost < 0 and inc_qaly > 0:
                icer_val = inc_cost / inc_qaly
                icer_str = "Dominant"
            else:
                icer_val = inc_cost / inc_qaly
                icer_str = f"{icer_val:,.0f}"

            rows.append({
                'Strategy': self.model.strategy_labels[strategy],
                'vs': self.model.strategy_labels[comparator],
                'Incremental Cost': inc_cost,
                'Incremental QALYs': inc_qaly,
                'Incremental LYs': inc_ly,
                'ICER ($/QALY)': icer_val,
                'ICER': icer_str,
            })

        return pd.DataFrame(rows)

    def nmb(self, wtp: float = 50000, comparator: Optional[str] = None) -> pd.DataFrame:
        """Compute net monetary benefit."""
        if comparator is None:
            comparator = self.model.strategy_names[0]

        comp = self.results[comparator]
        rows = []
        for strategy in self.model.strategy_names:
            r = self.results[strategy]
            nmb_val = r['mean_qalys'] * wtp - r['mean_cost']
            row = {
                'Strategy': self.model.strategy_labels[strategy],
                'Mean QALYs': r['mean_qalys'],
                'Mean Cost': r['mean_cost'],
                'NMB': nmb_val,
            }
            if strategy != comparator:
                inc_q = r['mean_qalys'] - comp['mean_qalys']
                inc_c = r['mean_cost'] - comp['mean_cost']
                row['INMB'] = inc_q * wtp - inc_c
            else:
                row['INMB'] = 0.0
            rows.append(row)
        return pd.DataFrame(rows)

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
                surv = (absorb_times > t).mean()
                rows.append({
                    'Time': t,
                    'Strategy': self.model.strategy_labels[strat],
                    'Survival': surv,
                })

        return pd.DataFrame(rows)


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

    def summary(self, comparator: Optional[str] = None) -> pd.DataFrame:
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
        comp_df = ce[ce['strategy'] == comparator]

        rows = []
        for strategy in self.model.strategy_names:
            if strategy == comparator:
                continue
            int_df = ce[ce['strategy'] == strategy]
            inc_cost = int_df['total_cost'].values - comp_df['total_cost'].values
            inc_qaly = int_df['qalys'].values - comp_df['qalys'].values

            mean_ic = inc_cost.mean()
            mean_iq = inc_qaly.mean()
            icer_val = mean_ic / mean_iq if abs(mean_iq) > 1e-10 else float('inf')

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
            })
        return pd.DataFrame(rows)

    def ceac_data(self, comparator: Optional[str] = None,
                  wtp_range: tuple = (0, 100000),
                  n_wtp: int = 200) -> pd.DataFrame:
        """Compute CEAC data."""
        if comparator is None:
            comparator = self.model.strategy_names[0]

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

    def plot_ceac(self, comparator=None, wtp_range=(0, 100000), **kwargs):
        """Plot CEAC."""
        from ..plotting import plot_ceac
        return plot_ceac(self, comparator=comparator, wtp_range=wtp_range, **kwargs)

    def plot_scatter(self, comparator=None, wtp=None, **kwargs):
        """Plot CE scatter."""
        from ..plotting import plot_scatter
        return plot_scatter(self, comparator=comparator, wtp=wtp, **kwargs)