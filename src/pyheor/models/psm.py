"""
Partitioned Survival Model (PSM) for health economic evaluation.

A PSM derives health state occupancy from overlaid survival curves:
- State 1 (e.g., PFS): S_PFS(t)
- State 2 (e.g., Progressed): S_OS(t) - S_PFS(t)
- State 3 (e.g., Dead): 1 - S_OS(t)

More generally, for N survival curves and N+1 states:
- State_1 = S_1(t)
- State_k = S_{k-1}(t) - S_k(t)  for k = 2..N
- State_{N+1} = 1 - S_N(t)

Supports:
- Multiple treatment strategies with different survival curves
- Flexible cost and utility definitions (same API as MarkovModel)
- Base case, OWSA, and PSA analysis
- Treatment effects via HR or AFT on baseline curves
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..distributions import Distribution
from .markov import Param, _CostDef
from ..survival import SurvivalDistribution, ProportionalHazards
from ..utils import (
    resolve_value, discount_factor, normalize_hcc, life_table_corrected_trace,
)


class PSMModel:
    """Partitioned Survival Model (PSM).

    Derives state probabilities from survival curves rather than
    transition matrices. Common in oncology where OS and PFS data
    are available from trials.

    Parameters
    ----------
    states : list of str
        Health state names. Typically 3 states: ["PFS", "Progressed", "Dead"].
        The number of states must be len(survival_endpoints) + 1.
    survival_endpoints : list of str
        Names of the survival endpoints that partition the states.
        E.g., ["PFS", "OS"]. Each endpoint separates two adjacent states.
        Must satisfy: S_PFS(t) <= S_OS(t) at all times.
    strategies : list of str or dict
        Treatment strategies.
    n_cycles : int
        Number of model cycles.
    cycle_length : float
        Length of each cycle in years (default: 1.0).
    discount_rate : float or dict
        Annual discount rate(s).
    half_cycle_correction : bool or str or None
        Half-cycle correction method. Options:

        - True or ``"trapezoidal"``: endpoint weighting [0.5, 1, ..., 1, 0.5]
        - ``"life-table"``: average adjacent trace rows (heemod-style)
        - False or None: no correction

        Default: True (trapezoidal).
    state_type : dict, optional
        Map state names to "alive" or "dead".

    Examples
    --------
    >>> model = PSMModel(
    ...     states=["PFS", "Progressed", "Dead"],
    ...     survival_endpoints=["PFS", "OS"],
    ...     strategies={"SOC": "Standard of Care", "TRT": "New Treatment"},
    ...     n_cycles=40,
    ...     cycle_length=1/12,
    ... )
    """

    def __init__(
        self,
        states: List[str],
        survival_endpoints: List[str],
        strategies: Union[List[str], Dict[str, str]],
        n_cycles: int,
        cycle_length: float = 1.0,
        discount_rate: Union[float, Dict[str, float]] = 0.03,
        half_cycle_correction: Union[bool, str, None] = True,
        state_type: Optional[Dict[str, str]] = None,
    ):
        # States
        self.states = list(states)
        self.n_states = len(self.states)

        # Survival endpoints
        self.survival_endpoints = list(survival_endpoints)
        self.n_endpoints = len(self.survival_endpoints)

        if self.n_states != self.n_endpoints + 1:
            raise ValueError(
                f"Number of states ({self.n_states}) must be "
                f"number of survival endpoints + 1 ({self.n_endpoints + 1})"
            )

        # Strategies
        if isinstance(strategies, dict):
            self.strategy_names = list(strategies.keys())
            self.strategy_labels = dict(strategies)
        else:
            self.strategy_names = list(strategies)
            self.strategy_labels = {s: s for s in self.strategy_names}
        self.n_strategies = len(self.strategy_names)

        # Model settings
        self.n_cycles = n_cycles
        self.cycle_length = cycle_length
        self._hcc_method = normalize_hcc(half_cycle_correction)

        # Discount rates
        if isinstance(discount_rate, (int, float)):
            self.dr_costs = float(discount_rate)
            self.dr_qalys = float(discount_rate)
        else:
            self.dr_costs = float(discount_rate.get('costs', 0.03))
            self.dr_qalys = float(discount_rate.get('qalys', 0.03))

        # State types
        if state_type is not None:
            self._alive_states = [
                i for i, s in enumerate(self.states)
                if state_type.get(s, "alive") == "alive"
            ]
        else:
            self._alive_states = list(range(self.n_states - 1))

        # Parameters
        self.params: Dict[str, Param] = {}

        # Survival curves: {strategy: {endpoint: SurvivalDistribution or callable}}
        self._survival_curves: Dict[str, Dict[str, Any]] = {}

        # Costs and utility (same structure as MarkovModel)
        self._costs: Dict[str, _CostDef] = {}
        self._utility: Any = None

        # Custom costs: list of {'category': str, 'func': callable}
        self._custom_costs: list = []

    @property
    def half_cycle_correction(self):
        """Half-cycle correction method (str or None)."""
        return self._hcc_method

    @half_cycle_correction.setter
    def half_cycle_correction(self, value):
        self._hcc_method = normalize_hcc(value)

    # =========================================================================
    # Parameter Management (same API as MarkovModel)
    # =========================================================================

    def add_param(self, name: str, base: float, dist=None, label=None,
                  low=None, high=None) -> "PSMModel":
        """Add a single parameter to the model."""
        self.params[name] = Param(
            base=base, dist=dist,
            label=label or name,
            low=low, high=high,
        )
        return self

    def add_params(self, params_dict: Dict[str, Union[Param, float]]) -> "PSMModel":
        """Add multiple parameters at once."""
        for name, param in params_dict.items():
            if isinstance(param, Param):
                if not param.label:
                    param.label = name
                self.params[name] = param
            elif isinstance(param, (int, float)):
                self.params[name] = Param(base=float(param), label=name)
            else:
                raise TypeError(
                    f"Parameter '{name}': expected Param or numeric, got {type(param)}"
                )
        return self

    # =========================================================================
    # Survival Curves
    # =========================================================================

    def set_survival(
        self,
        strategy: str,
        endpoint: str,
        curve: Union[SurvivalDistribution, Callable],
    ) -> "PSMModel":
        """Set a survival curve for a strategy and endpoint.

        Parameters
        ----------
        strategy : str
            Strategy name.
        endpoint : str
            Survival endpoint name (e.g., "PFS" or "OS").
        curve : SurvivalDistribution or callable
            A survival distribution object, or a callable:
            ``f(params_dict) -> SurvivalDistribution``
            that creates a curve from current parameter values.

        Returns
        -------
        PSMModel
            Self, for method chaining.

        Examples
        --------
        Fixed survival curve:

        >>> model.set_survival("SOC", "OS", Weibull(shape=1.2, scale=15))

        Parameter-dependent curve with HR:

        >>> model.set_survival("TRT", "OS", lambda p: ProportionalHazards(
        ...     Weibull(shape=1.2, scale=15), hr=p["hr_os"]
        ... ))
        """
        if strategy not in self.strategy_names:
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {self.strategy_names}")
        if endpoint not in self.survival_endpoints:
            raise ValueError(f"Unknown endpoint '{endpoint}'. Available: {self.survival_endpoints}")

        if strategy not in self._survival_curves:
            self._survival_curves[strategy] = {}
        self._survival_curves[strategy][endpoint] = curve
        return self

    def set_survival_all(
        self,
        strategy: str,
        curves: Dict[str, Union[SurvivalDistribution, Callable]],
    ) -> "PSMModel":
        """Set all survival curves for a strategy at once.

        Parameters
        ----------
        strategy : str
            Strategy name.
        curves : dict
            Maps endpoint names to survival distributions or callables.

        Examples
        --------
        >>> model.set_survival_all("SOC", {
        ...     "PFS": Weibull(shape=1.0, scale=8),
        ...     "OS":  Weibull(shape=1.2, scale=15),
        ... })
        """
        for endpoint, curve in curves.items():
            self.set_survival(strategy, endpoint, curve)
        return self

    # =========================================================================
    # Costs & Utility (same API as MarkovModel)
    # =========================================================================

    def set_state_cost(
        self,
        category: str,
        values: Any,
        first_cycle_only: bool = False,
        apply_cycles: Optional[List[int]] = None,
        method: str = "wlos",
    ) -> "PSMModel":
        """Define a cost category (same interface as MarkovModel).

        Parameters
        ----------
        category : str
            Cost category name.
        values : dict or callable
            Cost values per state.
        first_cycle_only : bool
            If True, cost only applies in cycle 0.
        apply_cycles : list of int, optional
            Specific cycles where cost applies.
        method : str
            "wlos" or "starting".
        """
        self._costs[category] = _CostDef(
            name=category,
            values=values,
            first_cycle_only=first_cycle_only,
            apply_cycles=apply_cycles,
            method=method,
        )
        return self

    def set_utility(self, values: Any) -> "PSMModel":
        """Define utility weights for health states."""
        self._utility = values
        return self

    def set_custom_cost(
        self,
        category: str,
        func: Callable,
    ) -> "PSMModel":
        """Define a custom cost computed from simulation state each cycle.

        The user-supplied function is called once per cycle (t = 1 … n_cycles)
        for each strategy.  Its return value is the **undiscounted cost** for
        that cycle and category.

        Parameters
        ----------
        category : str
            Cost category name.
        func : callable
            ``func(strategy, params, t, state_prev, state_curr, P, states) -> float``

            - **strategy** (str): Current strategy name.
            - **params** (dict): Parameter values ``{name: float}``.
            - **t** (int): Current cycle number (1-based).
            - **state_prev** (np.ndarray): State proportion vector at *t − 1*.
            - **state_curr** (np.ndarray): State proportion vector at *t*.
            - **P**: Always ``None`` for PSMModel (no transition matrix).
            - **states** (list[str]): State names (same order as array indices).

        Returns
        -------
        PSMModel
            Self, for method chaining.

        Examples
        --------
        Cost based on newly progressed patients:

        >>> def prog_cost(strategy, params, t, state_prev, state_curr, P, states):
        ...     i = states.index("Progressed")
        ...     new_prog = max(0, state_curr[i] - state_prev[i])
        ...     return new_prog * params['c_progression']
        >>> model.set_custom_cost("progression", prog_cost)
        """
        if not callable(func):
            raise TypeError("func must be callable")
        self._custom_costs.append({
            'category': category,
            'func': func,
        })
        return self

    # =========================================================================
    # Internal: Resolve Values
    # =========================================================================

    def _get_base_params(self) -> Dict[str, float]:
        return {name: p.base for name, p in self.params.items()}

    def _resolve_curve(self, strategy: str, endpoint: str,
                       params: Dict[str, float]) -> SurvivalDistribution:
        """Resolve a survival curve, evaluating callable if needed."""
        curve = self._survival_curves[strategy][endpoint]
        if callable(curve) and not isinstance(curve, SurvivalDistribution):
            return curve(params)
        return curve

    def _resolve_state_values(self, values: Any, strategy: str,
                              params: Dict[str, float], t: int) -> np.ndarray:
        """Resolve state-level values (same logic as MarkovModel)."""
        if callable(values):
            values = values(params, t)

        result = np.zeros(self.n_states)
        if not values:
            return result

        first_key = next(iter(values))
        if isinstance(first_key, str) and first_key in self.strategy_names:
            if strategy in values:
                state_vals = values[strategy]
                if isinstance(state_vals, dict):
                    for state_name, val in state_vals.items():
                        if state_name in self.states:
                            idx = self.states.index(state_name)
                            result[idx] = resolve_value(val, params, t)
                else:
                    v = resolve_value(state_vals, params, t)
                    result[:] = v
        else:
            for state_name, val in values.items():
                if state_name in self.states:
                    idx = self.states.index(state_name)
                    result[idx] = resolve_value(val, params, t)

        return result

    def _get_state_costs(self, category: str, strategy: str,
                         params: Dict[str, float], t: int) -> np.ndarray:
        """Get per-state costs for a category at cycle t."""
        cost_def = self._costs[category]
        if cost_def.first_cycle_only and t != 0:
            return np.zeros(self.n_states)
        if cost_def.apply_cycles is not None and t not in cost_def.apply_cycles:
            return np.zeros(self.n_states)
        if cost_def.method == "starting" and t != 0:
            return np.zeros(self.n_states)
        return self._resolve_state_values(cost_def.values, strategy, params, t)

    def _get_utilities(self, strategy: str, params: Dict[str, float],
                       t: int) -> np.ndarray:
        """Get per-state utility weights at cycle t."""
        if self._utility is None:
            u = np.zeros(self.n_states)
            for i in self._alive_states:
                u[i] = 1.0
            return u
        return self._resolve_state_values(self._utility, strategy, params, t)

    # =========================================================================
    # Simulation Engine
    # =========================================================================

    def _compute_state_probs(
        self, strategy: str, params: Dict[str, float],
    ) -> np.ndarray:
        """Compute state probabilities from survival curves.

        Returns
        -------
        np.ndarray
            Shape (n_cycles + 1, n_states). State membership at each cycle.
        """
        times = np.arange(self.n_cycles + 1) * self.cycle_length

        # Evaluate all survival curves
        surv_values = np.zeros((self.n_cycles + 1, self.n_endpoints))
        for j, endpoint in enumerate(self.survival_endpoints):
            curve = self._resolve_curve(strategy, endpoint, params)
            surv_values[:, j] = curve.survival(times)

        # Ensure monotonicity: S_1(t) <= S_2(t) <= ... <= S_N(t)
        # (e.g., PFS <= OS)
        for j in range(1, self.n_endpoints):
            surv_values[:, j] = np.maximum(surv_values[:, j], surv_values[:, j - 1])

        # Derive state probabilities
        state_probs = np.zeros((self.n_cycles + 1, self.n_states))

        # First state: S_1(t)
        state_probs[:, 0] = surv_values[:, 0]

        # Middle states: S_{k-1}(t) - S_k(t)  (note: endpoint k maps to state k)
        # Actually for the standard 3-state PSM:
        # states = [PFS, Prog, Dead], endpoints = [PFS, OS]
        # PFS_state = S_PFS(t)
        # Prog_state = S_OS(t) - S_PFS(t)
        # Dead_state = 1 - S_OS(t)
        # So: state[0] = surv[0], state[k] = surv[k] - surv[k-1] for k=1..N-1, state[N] = 1 - surv[N-1]
        for k in range(1, self.n_endpoints):
            state_probs[:, k] = surv_values[:, k] - surv_values[:, k - 1]

        # Last state: 1 - S_last(t)
        state_probs[:, -1] = 1.0 - surv_values[:, -1]

        # Clip numerical errors
        state_probs = np.clip(state_probs, 0, 1)

        return state_probs

    def _simulate_single(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Run one deterministic simulation with given parameter values."""
        results = {}

        for strategy in self.strategy_names:
            # --- State probabilities from survival curves ---
            trace = self._compute_state_probs(strategy, params)

            # --- Per-cycle rewards ---
            costs_by_cat = {cat: np.zeros(self.n_cycles + 1) for cat in self._costs}
            qalys = np.zeros(self.n_cycles + 1)
            lys = np.zeros(self.n_cycles + 1)

            alive_mask = np.zeros(self.n_states)
            for i in self._alive_states:
                alive_mask[i] = 1.0

            for t in range(self.n_cycles + 1):
                state_probs = trace[t]

                # Utilities → QALYs
                u = self._get_utilities(strategy, params, t)
                qalys[t] = np.dot(state_probs, u) * self.cycle_length

                # Life years
                lys[t] = np.dot(state_probs, alive_mask) * self.cycle_length

                # Costs per category
                for cat in self._costs:
                    cost_def = self._costs[cat]
                    c = self._get_state_costs(cat, strategy, params, t)
                    if cost_def.method == "wlos":
                        costs_by_cat[cat][t] = np.dot(state_probs, c) * self.cycle_length
                    elif cost_def.method == "starting":
                        costs_by_cat[cat][t] = np.dot(state_probs, c)
                    else:
                        costs_by_cat[cat][t] = np.dot(state_probs, c) * self.cycle_length

            # --- Custom costs (user-defined functions) ---
            if self._custom_costs:
                for cc in self._custom_costs:
                    cat = cc['category']
                    cc_costs = np.zeros(self.n_cycles + 1)
                    for t in range(1, self.n_cycles + 1):
                        cost_val = cc['func'](
                            strategy, params, t,
                            trace[t - 1], trace[t], None, self.states
                        )
                        cc_costs[t] = float(cost_val)
                    costs_by_cat[cat] = (
                        costs_by_cat.get(cat, np.zeros(self.n_cycles + 1))
                        + cc_costs
                    )

            # --- Half-cycle correction ---
            # Collect custom-cost-only categories (no HCC for these)
            cc_only_cats = set()
            if self._custom_costs:
                for cc in self._custom_costs:
                    if cc['category'] not in self._costs:
                        cc_only_cats.add(cc['category'])

            if self._hcc_method == "trapezoidal":
                hcc_weights = np.ones(self.n_cycles + 1)
                hcc_weights[0] = 0.5
                hcc_weights[-1] = 0.5

                qalys_hcc = qalys * hcc_weights
                lys_hcc = lys * hcc_weights

                costs_hcc = {}
                for cat in costs_by_cat:
                    if cat in cc_only_cats:
                        costs_hcc[cat] = costs_by_cat[cat].copy()
                    elif cat in self._costs and self._costs[cat].method in ("starting",):
                        costs_hcc[cat] = costs_by_cat[cat].copy()
                    else:
                        costs_hcc[cat] = costs_by_cat[cat] * hcc_weights

            elif self._hcc_method == "life-table":
                corrected = life_table_corrected_trace(trace)

                qalys_hcc = np.zeros(self.n_cycles + 1)
                lys_hcc = np.zeros(self.n_cycles + 1)
                for t in range(self.n_cycles + 1):
                    u = self._get_utilities(strategy, params, t)
                    qalys_hcc[t] = np.dot(corrected[t], u) * self.cycle_length
                    lys_hcc[t] = np.dot(corrected[t], alive_mask) * self.cycle_length

                costs_hcc = {}
                for cat in costs_by_cat:
                    if cat in cc_only_cats:
                        costs_hcc[cat] = costs_by_cat[cat].copy()
                    elif cat in self._costs and self._costs[cat].method in ("starting",):
                        costs_hcc[cat] = costs_by_cat[cat].copy()
                    else:
                        costs_hcc[cat] = np.zeros(self.n_cycles + 1)
                        for t in range(self.n_cycles + 1):
                            c = self._get_state_costs(cat, strategy, params, t)
                            costs_hcc[cat][t] = (
                                np.dot(corrected[t], c) * self.cycle_length
                            )

            else:
                # No correction
                qalys_hcc = qalys.copy()
                lys_hcc = lys.copy()
                costs_hcc = {cat: arr.copy() for cat, arr in costs_by_cat.items()}

            # --- Discounting ---
            cycles = np.arange(self.n_cycles + 1, dtype=float)
            df_c = discount_factor(cycles, self.dr_costs, self.cycle_length)
            df_q = discount_factor(cycles, self.dr_qalys, self.cycle_length)

            discounted_costs = {cat: costs_hcc[cat] * df_c for cat in costs_hcc}
            discounted_qalys = qalys_hcc * df_q
            discounted_lys = lys_hcc * df_q

            # --- Survival values for plotting ---
            times = np.arange(self.n_cycles + 1) * self.cycle_length
            surv_curves = {}
            for endpoint in self.survival_endpoints:
                curve = self._resolve_curve(strategy, endpoint, params)
                surv_curves[endpoint] = curve.survival(times)

            # --- Totals ---
            results[strategy] = {
                'trace': trace,
                'survival_curves': surv_curves,
                'times': times,
                'costs_by_cycle': costs_by_cat,
                'qalys_by_cycle': qalys,
                'lys_by_cycle': lys,
                'costs_hcc': costs_hcc,
                'qalys_hcc': qalys_hcc,
                'lys_hcc': lys_hcc,
                'discounted_costs': discounted_costs,
                'discounted_qalys': discounted_qalys,
                'discounted_lys': discounted_lys,
                'total_costs': {
                    cat: float(np.sum(discounted_costs[cat]))
                    for cat in discounted_costs
                },
                'total_qalys': float(np.sum(discounted_qalys)),
                'total_lys': float(np.sum(discounted_lys)),
            }

        return results

    # =========================================================================
    # Analysis Entry Points
    # =========================================================================

    def run_base_case(self) -> "PSMBaseResult":
        """Run deterministic base case analysis."""
        from ..analysis.results import PSMBaseResult
        params = self._get_base_params()
        sim = self._simulate_single(params)
        return PSMBaseResult(model=self, results=sim, params=params)

    def run_owsa(
        self,
        params: Optional[List[str]] = None,
        range_pct: float = 0.2,
        wtp: float = 50000,
    ) -> "OWSAResult":
        """Run one-way sensitivity analysis."""
        from ..analysis.results import OWSAResult

        if params is None:
            params = [
                name for name, p in self.params.items()
                if p.dist is not None
            ]
            if not params:
                params = list(self.params.keys())

        base_params = self._get_base_params()
        base_result = self._simulate_single(base_params)

        owsa_data = []
        for param_name in params:
            p = self.params[param_name]
            low = p.low if p.low is not None else p.base * (1 - range_pct)
            high = p.high if p.high is not None else p.base * (1 + range_pct)

            for bound, val in [('low', low), ('high', high)]:
                test_params = base_params.copy()
                test_params[param_name] = val
                result = self._simulate_single(test_params)
                owsa_data.append({
                    'param': param_name,
                    'label': p.label,
                    'value': val,
                    'base_value': p.base,
                    'bound': bound,
                    'result': result,
                })

        return OWSAResult(
            model=self,
            base_result=base_result,
            base_params=base_params,
            owsa_data=owsa_data,
            wtp=wtp,
        )

    def run_psa(
        self,
        n_sim: int = 1000,
        seed: Optional[int] = None,
        progress: bool = True,
    ) -> "PSAResult":
        """Run probabilistic sensitivity analysis."""
        from ..analysis.results import PSAResult

        if seed is not None:
            np.random.seed(seed)

        sampled_params = []
        for i in range(n_sim):
            p = self._get_base_params()
            for name, param in self.params.items():
                if param.dist is not None:
                    p[name] = float(param.dist.sample(1)[0])
            sampled_params.append(p)

        psa_results = []
        for i, p in enumerate(sampled_params):
            if progress and (i + 1) % max(1, n_sim // 10) == 0:
                print(f"  PSA: {i+1}/{n_sim} ({100*(i+1)/n_sim:.0f}%)")
            result = self._simulate_single(p)
            psa_results.append(result)

        if progress:
            print(f"  PSA complete: {n_sim} simulations")

        return PSAResult(
            model=self,
            psa_results=psa_results,
            sampled_params=sampled_params,
        )

    # =========================================================================
    # Convenience
    # =========================================================================

    def info(self) -> str:
        """Return a summary string describing the model."""
        lines = [
            f"PSMModel",
            f"  States ({self.n_states}): {self.states}",
            f"  Endpoints ({self.n_endpoints}): {self.survival_endpoints}",
            f"  Strategies ({self.n_strategies}): {self.strategy_names}",
            f"  Cycles: {self.n_cycles} × {self.cycle_length} year(s)",
            f"  Discount rates: costs={self.dr_costs:.1%}, QALYs={self.dr_qalys:.1%}",
            f"  Half-cycle correction: {self._hcc_method or 'None'}",
            f"  Parameters ({len(self.params)}):",
        ]
        for name, p in self.params.items():
            dist_str = repr(p.dist) if p.dist else "Fixed"
            lines.append(f"    {name}: {p.base} [{dist_str}]")

        lines.append(f"  Survival curves:")
        for strategy in self.strategy_names:
            curves = self._survival_curves.get(strategy, {})
            for ep in self.survival_endpoints:
                c = curves.get(ep, "NOT SET")
                lines.append(f"    {strategy}/{ep}: {c}")

        lines.append(f"  Cost categories ({len(self._costs)}):")
        for cat, cd in self._costs.items():
            flags = []
            if cd.first_cycle_only:
                flags.append("first-cycle")
            if cd.method == "starting":
                flags.append("one-time")
            flag_str = f" ({', '.join(flags)})" if flags else ""
            lines.append(f"    {cat}{flag_str}")

        return "\n".join(lines)

    def __repr__(self):
        return (
            f"PSMModel(states={self.states}, "
            f"endpoints={self.survival_endpoints}, "
            f"strategies={self.strategy_names}, "
            f"n_cycles={self.n_cycles})"
        )
