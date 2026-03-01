"""
MarkovModel - Core model class for cohort discrete-time state transition models.

This module implements the main MarkovModel class which provides:
- Flexible parameter definition with PSA distributions
- Transition probability matrices (constant or time-varying)
- Flexible cost/utility definitions (per-cycle, first-cycle-only, time-dependent)
- Base case, OWSA, and PSA analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .distributions import Distribution
from .utils import (
    C, _Complement, resolve_complement, resolve_value, discount_factor,
    normalize_hcc, life_table_corrected_trace,
)


# =============================================================================
# Parameter Definition
# =============================================================================

@dataclass
class Param:
    """A model parameter with point estimate and optional PSA distribution.
    
    Parameters
    ----------
    base : float
        Base case (point estimate) value.
    dist : Distribution, optional
        Probability distribution for PSA sampling.
    label : str, optional
        Human-readable label for display in plots/tables.
    low : float, optional
        Lower bound for OWSA. Default: base * 0.8.
    high : float, optional
        Upper bound for OWSA. Default: base * 1.2.
    
    Examples
    --------
    >>> p = Param(0.15, dist=Beta(mean=0.15, sd=0.03), label="P(H→S)")
    >>> p = Param(5000, dist=Gamma(mean=5000, sd=500), label="Drug cost")
    >>> p = Param(0.7, low=0.5, high=0.9)  # Custom OWSA range
    """
    base: float
    dist: Optional[Distribution] = None
    label: Optional[str] = None
    low: Optional[float] = None
    high: Optional[float] = None
    
    def __post_init__(self):
        if self.label is None:
            self.label = ""
        if self.low is None:
            self.low = self.base * 0.8
        if self.high is None:
            self.high = self.base * 1.2


# =============================================================================
# Cost Definition
# =============================================================================

@dataclass 
class _CostDef:
    """Internal cost category definition."""
    name: str
    values: Any
    first_cycle_only: bool = False
    apply_cycles: Optional[List[int]] = None
    method: str = "wlos"


# =============================================================================
# MarkovModel
# =============================================================================

class MarkovModel:
    """Cohort Discrete-Time State Transition Model (cDTSTM).
    
    A Markov cohort model for health economic evaluation. Supports:
    - Time-homogeneous and time-inhomogeneous models
    - Multiple treatment strategies
    - Multiple cost categories with flexible timing
    - Base case, one-way sensitivity analysis (OWSA), and PSA
    
    Parameters
    ----------
    states : list of str
        Health state names (e.g., ["PFS", "Progressed", "Dead"]).
    strategies : list of str or dict
        Treatment strategies. If dict, maps short names to display labels.
    n_cycles : int
        Number of model cycles to simulate.
    cycle_length : float
        Length of each cycle in years (default: 1.0).
    discount_rate : float or dict
        Annual discount rate(s). If dict, keys are 'costs' and 'qalys'.
    half_cycle_correction : bool or str or None
        Half-cycle correction method. Options:

        - True or ``"trapezoidal"``: endpoint weighting [0.5, 1, ..., 1, 0.5]
        - ``"life-table"``: average adjacent trace rows (heemod-style)
        - False or None: no correction

        Default: True (trapezoidal).
    initial_state : str or int
        Starting health state (default: 0, the first state).
    state_type : dict, optional
        Map state names to type: "alive" or "dead". Used for LY calculation.
        By default, the last state is considered "dead".
    
    Examples
    --------
    >>> model = MarkovModel(
    ...     states=["Healthy", "Sick", "Dead"],
    ...     strategies=["SOC", "New"],
    ...     n_cycles=20,
    ...     cycle_length=1.0,
    ...     discount_rate={"costs": 0.03, "qalys": 0.03},
    ... )
    """
    
    def __init__(
        self,
        states: List[str],
        strategies: Union[List[str], Dict[str, str]],
        n_cycles: int,
        cycle_length: float = 1.0,
        discount_rate: Union[float, Dict[str, float]] = 0.03,
        half_cycle_correction: Union[bool, str, None] = True,
        initial_state: Union[str, int] = 0,
        state_type: Optional[Dict[str, str]] = None,
    ):
        # States
        self.states = list(states)
        self.n_states = len(self.states)
        
        # Strategies
        if isinstance(strategies, dict):
            self.strategy_names = list(strategies.keys())
            self.strategy_labels = dict(strategies)
        else:
            self.strategy_names = list(strategies)
            self.strategy_labels = {s: s for s in self.strategy_names}
        self.n_strategies = len(self.strategy_names)
        
        # Model cycles
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
        
        # Initial state
        if isinstance(initial_state, str):
            self.initial_state_idx = self.states.index(initial_state)
        else:
            self.initial_state_idx = int(initial_state)
        
        # State types (alive vs dead) for LY calculation
        if state_type is not None:
            self._alive_states = [
                i for i, s in enumerate(self.states) 
                if state_type.get(s, "alive") == "alive"
            ]
        else:
            # Default: all states except last are "alive"
            self._alive_states = list(range(self.n_states - 1))
        
        # Parameters
        self.params: Dict[str, Param] = {}
        
        # Transitions: strategy_name -> matrix or callable
        self._transitions: Dict[str, Any] = {}
        
        # Costs: category_name -> _CostDef
        self._costs: Dict[str, _CostDef] = {}
        
        # Transition costs: list of (category, from_state, to_state, value)
        # value: float, str (param name), or callable(params, t) -> float
        self._transition_costs: list = []

        # Custom costs: list of {'category': str, 'func': callable}
        self._custom_costs: list = []

        # Utility
        self._utility: Any = None

    @property
    def half_cycle_correction(self):
        """Half-cycle correction method (str or None)."""
        return self._hcc_method

    @half_cycle_correction.setter
    def half_cycle_correction(self, value):
        self._hcc_method = normalize_hcc(value)

    # =========================================================================
    # Parameter Management
    # =========================================================================
    
    def add_param(self, name: str, base: float, dist=None, label=None,
                  low=None, high=None) -> "MarkovModel":
        """Add a single parameter to the model.
        
        Parameters
        ----------
        name : str
            Parameter name (used as key).
        base : float
            Base case value.
        dist : Distribution, optional
            PSA distribution.
        label : str, optional
            Display label.
        low, high : float, optional
            OWSA bounds.
            
        Returns
        -------
        MarkovModel
            Self, for method chaining.
        """
        self.params[name] = Param(
            base=base, dist=dist, 
            label=label or name,
            low=low, high=high,
        )
        return self
    
    def add_params(self, params_dict: Dict[str, Union[Param, float]]) -> "MarkovModel":
        """Add multiple parameters at once.
        
        Parameters
        ----------
        params_dict : dict
            Maps parameter names to Param objects or numeric values.
            
        Returns
        -------
        MarkovModel
            Self, for method chaining.
        """
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
    # Transition Probabilities
    # =========================================================================
    
    def set_transitions(self, strategy: str, transitions) -> "MarkovModel":
        """Set transition probabilities for a strategy.
        
        Parameters
        ----------
        strategy : str
            Strategy name.
        transitions : list, np.ndarray, or callable
            Transition probability matrix. Options:
            
            - **Constant matrix** (list of lists or np.ndarray):
              Use `C` for complement (1 - sum of other row entries).
              
            - **Time-varying** (callable):
              ``f(params_dict, cycle) -> matrix``
              where matrix is a list of lists (can include `C`).
        
        Returns
        -------
        MarkovModel
            Self, for method chaining.
            
        Examples
        --------
        Constant matrix:
        
        >>> model.set_transitions("SOC", [
        ...     [C,  0.15, 0.02],
        ...     [0,  C,    0.30],
        ...     [0,  0,    1   ],
        ... ])
        
        Time-varying with parameters:
        
        >>> model.set_transitions("New", lambda p, t: [
        ...     [C,  p["p_prog"] * p["hr"],  p["p_death"]],
        ...     [0,  C,                       p["p_death2"]],
        ...     [0,  0,                       1],
        ... ])
        """
        if strategy not in self.strategy_names:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {self.strategy_names}"
            )
        self._transitions[strategy] = transitions
        return self
    
    # =========================================================================
    # Costs
    # =========================================================================
    
    def set_state_cost(
        self,
        category: str,
        values: Any,
        first_cycle_only: bool = False,
        apply_cycles: Optional[List[int]] = None,
        method: str = "wlos",
    ) -> "MarkovModel":
        """Define a cost category.
        
        Parameters
        ----------
        category : str
            Cost category name (e.g., "drug", "medical", "ae").
        values : dict or callable
            Cost values per state. Formats:
            
            - ``{state: value}`` — Same for all strategies.
            - ``{strategy: {state: value}}`` — Different per strategy.
            - ``callable(params, t) -> dict`` — Dynamic costs.
            
            Each ``value`` can be: float, str (parameter name), or
            ``callable(params, t) -> float``.
            
        first_cycle_only : bool
            If True, cost only applies in cycle 0 (e.g., adverse event costs).
        apply_cycles : list of int, optional
            Specific cycles where cost applies.
        method : str
            - ``"wlos"``: Weighted by time in state, scaled by cycle_length.
            - ``"starting"``: One-time cost at model start (no cycle_length scaling).
        
        Returns
        -------
        MarkovModel
            Self, for method chaining.
            
        Examples
        --------
        Same cost for all strategies:
        
        >>> model.set_state_cost("medical", {
        ...     "PFS": 500, "Progressed": 3000, "Dead": 0
        ... })
        
        Strategy-specific costs:
        
        >>> model.set_state_cost("drug", {
        ...     "SOC": {"PFS": 2000, "Progressed": 2000},
        ...     "New": {"PFS": 8000, "Progressed": 5000},
        ... })
        
        First-cycle-only AE costs:
        
        >>> model.set_state_cost("ae", {
        ...     "New": {"PFS": 3000}
        ... }, first_cycle_only=True)
        
        Time-dependent cost using a function:
        
        >>> model.set_state_cost("monitoring", lambda p, t: {
        ...     "PFS": 500 if t < 5 else 200,
        ...     "Progressed": 1000,
        ... })
        """
        self._costs[category] = _CostDef(
            name=category,
            values=values,
            first_cycle_only=first_cycle_only,
            apply_cycles=apply_cycles,
            method=method,
        )
        return self
    
    # =========================================================================
    # Transition Costs
    # =========================================================================

    def set_transition_cost(
        self,
        category: str,
        from_state: str,
        to_state: str,
        value: Any,
    ) -> "MarkovModel":
        """Define costs triggered when patients transition between states.
        
        In a cohort model, the cost is applied to the **flow** of patients
        transitioning from one state to another each cycle:
        ``cost_t = trace[t-1, from] × P[from, to] × unit_cost``.
        
        **Cost schedule (费用计划表)**: Pass a ``list`` to define costs that
        span multiple cycles after each transition event. For example,
        ``[100, 200]`` means 100 in the cycle of transition and 200 in
        the next cycle. The engine tracks the inflow at every cycle and
        applies the schedule via convolution, so overlapping cohorts of
        new entrants are handled correctly.
        
        Parameters
        ----------
        category : str
            Cost category name for this transition cost.
        from_state : str
            Origin state name.
        to_state : str
            Destination state name.
        value : float, str, dict, list, or callable
            The unit cost per transition. Can be:
            
            - ``float`` — Fixed cost, same for all strategies.
            - ``str`` — Parameter name reference.
            - ``list`` — **Cost schedule**: ``[cost_at_transition, cost_1_cycle_later, ...]``.
              Each element can be ``float`` or ``str`` (parameter reference).
            - ``{strategy: value}`` — Strategy-specific. Each *value* can itself
              be ``float``, ``str``, ``list`` (schedule), or ``callable``.
            - ``callable(params, t) -> float`` — Time-varying cost (single cycle).
        
        Returns
        -------
        MarkovModel
            Self, for method chaining.
            
        Examples
        --------
        Fixed cost when entering "Progressed":
        
        >>> model.set_transition_cost("surgery", "PFS", "Progressed", 50000)
        
        Parameter-driven:
        
        >>> model.set_transition_cost("hospitalization", "Stable", "ICU", "c_icu")
        
        Cost schedule — 50k surgery + 10k follow-up next cycle:
        
        >>> model.set_transition_cost("surgery", "PFS", "Progressed", [50000, 10000])
        
        Parameter references in schedule:
        
        >>> model.set_transition_cost("chemo", "PFS", "Progressed",
        ...     ["c_chemo_init", "c_chemo_maint", "c_chemo_maint"])
        
        Strategy-specific with schedule:
        
        >>> model.set_transition_cost("rescue", "PFS", "Progressed", {
        ...     "SOC": [30000, 5000],
        ...     "New Drug": 15000,
        ... })
        """
        if from_state not in self.states:
            raise ValueError(f"Unknown from_state '{from_state}'. Available: {self.states}")
        if to_state not in self.states:
            raise ValueError(f"Unknown to_state '{to_state}'. Available: {self.states}")
        
        self._transition_costs.append({
            'category': category,
            'from_idx': self.states.index(from_state),
            'to_idx': self.states.index(to_state),
            'from_state': from_state,
            'to_state': to_state,
            'value': value,
        })
        return self

    def set_custom_cost(
        self,
        category: str,
        func: Callable,
    ) -> "MarkovModel":
        """Define a custom cost computed from simulation state each cycle.

        Unlike ``set_transition_cost`` which targets individual state pairs,
        this method gives full access to the transition matrix and state
        distribution, allowing arbitrary cost logic.

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
            - **P** (np.ndarray): Transition probability matrix at cycle *t*.
            - **states** (list[str]): State names (same order as array indices).

        Returns
        -------
        MarkovModel
            Self, for method chaining.

        Examples
        --------
        Compute surgery cost from PFS → Progressed flow:

        >>> def surgery_cost(strategy, params, t, state_prev, state_curr, P, states):
        ...     i_from = states.index("PFS")
        ...     i_to = states.index("Progressed")
        ...     flow = state_prev[i_from] * P[i_from, i_to]
        ...     return flow * params['c_surgery']
        >>> model.set_custom_cost("surgery", surgery_cost)
        """
        if not callable(func):
            raise TypeError("func must be callable")
        self._custom_costs.append({
            'category': category,
            'func': func,
        })
        return self

    def _get_tc_schedule(
        self, tc: dict, strategy: str, params: Dict[str, float]
    ):
        """Resolve transition cost value into a cost schedule (list of floats).
        
        Returns
        -------
        list[float] or None
            A list of per-cycle costs starting from the cycle of transition.
            Returns ``None`` when the resolved value is a callable (handled
            separately in the engine with per-cycle evaluation).
        """
        val = tc['value']
        # Strategy-specific dict → get this strategy's value
        if isinstance(val, dict):
            val = val.get(strategy, 0)
        # Already a schedule (list/tuple)
        if isinstance(val, (list, tuple)):
            resolved = []
            for v in val:
                if isinstance(v, str):
                    v = params.get(v, 0)
                resolved.append(float(v))
            return resolved
        # Callable → signal to engine to use per-cycle evaluation
        if callable(val):
            return None
        # Scalar: parameter reference or float → single-element schedule
        if isinstance(val, str):
            val = params.get(val, 0)
        return [float(val)]

    # =========================================================================
    # Utility
    # =========================================================================
    
    def set_utility(self, values: Any) -> "MarkovModel":
        """Define utility (quality-of-life) weights for health states.
        
        Parameters
        ----------
        values : dict or callable
            Utility values per state. Formats:
            
            - ``{state: value}`` — Same for all strategies.
            - ``{strategy: {state: value}}`` — Different per strategy.
            - ``callable(params, t) -> dict``
            
            Each ``value`` can be: float, str (parameter name), or callable.
        
        Returns
        -------
        MarkovModel
            Self, for method chaining.
            
        Examples
        --------
        >>> model.set_utility({
        ...     "PFS": "u_pfs",          # Parameter reference
        ...     "Progressed": "u_prog",
        ...     "Dead": 0.0,
        ... })
        """
        self._utility = values
        return self
    
    # =========================================================================
    # Internal: Parameter Resolution
    # =========================================================================
    
    def _get_base_params(self) -> Dict[str, float]:
        """Get base case parameter values as a dict."""
        return {name: p.base for name, p in self.params.items()}
    
    def _get_transition_matrix(self, strategy: str, params: Dict[str, float],
                               cycle: int) -> np.ndarray:
        """Compute the transition probability matrix for a given context."""
        trans = self._transitions[strategy]
        
        if callable(trans):
            matrix_data = trans(params, cycle)
        else:
            matrix_data = trans
        
        # Handle numpy arrays (no C sentinel)
        if isinstance(matrix_data, np.ndarray):
            P = matrix_data.copy().astype(float)
        else:
            # List of lists - resolve numbers and C sentinels
            resolved = []
            for row in matrix_data:
                resolved_row = []
                for val in row:
                    if isinstance(val, _Complement) or val is C:
                        resolved_row.append(C)
                    elif callable(val):
                        resolved_row.append(float(val(params, cycle)))
                    else:
                        resolved_row.append(float(val))
                resolved.append(resolved_row)
            P = resolve_complement(resolved)
        
        # Clip small numerical errors
        P = np.clip(P, 0.0, 1.0)
        
        # Ensure rows sum to 1 (renormalize if needed)
        row_sums = P.sum(axis=1, keepdims=True)
        mask = row_sums.flatten() > 0
        P[mask] = P[mask] / row_sums[mask]
        
        return P
    
    def _resolve_state_values(self, values: Any, strategy: str,
                              params: Dict[str, float], t: int) -> np.ndarray:
        """Resolve state-level values (costs or utilities) to a numpy array.
        
        Handles all input formats:
        - {state: val} — uniform across strategies
        - {strategy: {state: val}} — strategy-specific
        - callable(params, t) -> dict
        """
        # Evaluate callable first
        if callable(values):
            values = values(params, t)
        
        result = np.zeros(self.n_states)
        
        if not values:
            return result
        
        # Inspect structure
        first_key = next(iter(values))
        
        if isinstance(first_key, str) and first_key in self.strategy_names:
            # Format: {strategy: {state: value}}
            if strategy in values:
                state_vals = values[strategy]
                if isinstance(state_vals, dict):
                    for state_name, val in state_vals.items():
                        if state_name in self.states:
                            idx = self.states.index(state_name)
                            result[idx] = resolve_value(val, params, t)
                else:
                    # state_vals might be a single value for all states
                    v = resolve_value(state_vals, params, t)
                    result[:] = v
        else:
            # Format: {state: value} — same for all strategies
            for state_name, val in values.items():
                if state_name in self.states:
                    idx = self.states.index(state_name)
                    result[idx] = resolve_value(val, params, t)
        
        return result
    
    def _get_state_costs(self, category: str, strategy: str,
                         params: Dict[str, float], t: int) -> np.ndarray:
        """Get per-state costs for a category at cycle t."""
        cost_def = self._costs[category]
        
        # Check cycle applicability
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
            # Default: 1 for alive states, 0 for dead
            u = np.zeros(self.n_states)
            for i in self._alive_states:
                u[i] = 1.0
            return u
        return self._resolve_state_values(self._utility, strategy, params, t)
    
    # =========================================================================
    # Simulation Engine
    # =========================================================================
    
    def _simulate_single(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Run one deterministic simulation with given parameter values.
        
        Returns
        -------
        dict
            Results keyed by strategy name, each containing:
            trace, costs, qalys, lys, totals.
        """
        results = {}
        
        for strategy in self.strategy_names:
            # --- Markov trace ---
            trace = np.zeros((self.n_cycles + 1, self.n_states))
            trace[0, self.initial_state_idx] = 1.0
            
            for t in range(1, self.n_cycles + 1):
                P = self._get_transition_matrix(strategy, params, t)
                trace[t] = trace[t - 1] @ P
            
            # --- Per-cycle rewards (undiscounted) ---
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
                        # Annual cost → per-cycle cost
                        costs_by_cat[cat][t] = np.dot(state_probs, c) * self.cycle_length
                    elif cost_def.method == "starting":
                        # One-time cost, no cycle_length scaling
                        costs_by_cat[cat][t] = np.dot(state_probs, c)
                    else:
                        costs_by_cat[cat][t] = np.dot(state_probs, c) * self.cycle_length
            
            # --- Transition costs (flow-based, schedule-aware) ---
            if self._transition_costs:
                # Group by category, keeping global index for inflow tracking
                tc_cats: Dict[str, list] = {}
                for i, tc in enumerate(self._transition_costs):
                    cat = tc['category']
                    if cat not in tc_cats:
                        tc_cats[cat] = []
                    tc_cats[cat].append((i, tc))
                
                # Inflow history matrix: [tc_index, cycle] → flow
                n_tc = len(self._transition_costs)
                tc_inflows = np.zeros((n_tc, self.n_cycles + 1))
                
                # Pre-resolve schedules (None = callable, handled per-cycle)
                tc_schedules = [
                    self._get_tc_schedule(tc, strategy, params)
                    for tc in self._transition_costs
                ]
                
                for cat, tc_list in tc_cats.items():
                    tc_costs = np.zeros(self.n_cycles + 1)
                    for t in range(1, self.n_cycles + 1):
                        P = self._get_transition_matrix(strategy, params, t)
                        for idx, tc in tc_list:
                            fi, ti = tc['from_idx'], tc['to_idx']
                            flow = trace[t - 1, fi] * P[fi, ti]
                            tc_inflows[idx, t] = flow
                            
                            schedule = tc_schedules[idx]
                            if schedule is None:
                                # Callable: evaluate per-cycle (no schedule)
                                val = tc['value']
                                if isinstance(val, dict):
                                    val = val.get(strategy, 0)
                                if callable(val):
                                    tc_costs[t] += flow * float(val(params, t))
                            else:
                                # Schedule convolution:
                                # cost[t] += Σ_k inflow[t-k] × schedule[k]
                                for k, sched_val in enumerate(schedule):
                                    past_t = t - k
                                    if past_t >= 1:
                                        tc_costs[t] += tc_inflows[idx, past_t] * sched_val
                    costs_by_cat[cat] = costs_by_cat.get(cat, np.zeros(self.n_cycles + 1)) + tc_costs

            # --- Custom costs (user-defined functions) ---
            if self._custom_costs:
                for cc in self._custom_costs:
                    cat = cc['category']
                    cc_costs = np.zeros(self.n_cycles + 1)
                    for t in range(1, self.n_cycles + 1):
                        P = self._get_transition_matrix(strategy, params, t)
                        cost_val = cc['func'](
                            strategy, params, t,
                            trace[t - 1], trace[t], P, self.states
                        )
                        cc_costs[t] = float(cost_val)
                    costs_by_cat[cat] = (
                        costs_by_cat.get(cat, np.zeros(self.n_cycles + 1))
                        + cc_costs
                    )

            # --- Half-cycle correction ---
            # Collect categories excluded from HCC (event-based, not state-based)
            no_hcc_cats = set()
            if self._transition_costs:
                for tc in self._transition_costs:
                    if tc['category'] not in self._costs:
                        no_hcc_cats.add(tc['category'])
            if self._custom_costs:
                for cc in self._custom_costs:
                    if cc['category'] not in self._costs:
                        no_hcc_cats.add(cc['category'])

            if self._hcc_method == "trapezoidal":
                hcc_weights = np.ones(self.n_cycles + 1)
                hcc_weights[0] = 0.5
                hcc_weights[-1] = 0.5

                qalys_hcc = qalys * hcc_weights
                lys_hcc = lys * hcc_weights

                costs_hcc = {}
                for cat in costs_by_cat:
                    if cat in no_hcc_cats:
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
                    if cat in no_hcc_cats:
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
            
            # --- Totals ---
            results[strategy] = {
                'trace': trace,
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
    
    def run_base_case(self) -> "BaseResult":
        """Run deterministic base case analysis.
        
        Returns
        -------
        BaseResult
            Results including summary, ICER, Markov trace, and plotting methods.
        """
        from .results import BaseResult
        params = self._get_base_params()
        sim = self._simulate_single(params)
        return BaseResult(model=self, results=sim, params=params)
    
    # Model-level parameters that can be varied in OWSA.
    # Maps param name → tuple of model attributes to set.
    _MODEL_LEVEL_PARAMS = {
        'dr': ('dr_costs', 'dr_qalys'),
        'discount_rate': ('dr_costs', 'dr_qalys'),
    }

    def run_owsa(
        self,
        params: Optional[List[str]] = None,
        range_pct: float = 0.2,
        wtp: float = 50000,
    ) -> "OWSAResult":
        """Run one-way sensitivity analysis (OWSA).

        Each parameter is varied independently to its low and high values
        while all other parameters remain at base case.

        Parameters
        ----------
        params : list of str, optional
            Parameter names to vary. Default: all parameters with distributions.
            Model-level parameters like "dr" (discount rate) are also supported.
        range_pct : float
            Percentage range for variation if low/high not set (default: ±20%).
        wtp : float
            Willingness-to-pay threshold for NMB calculation.

        Returns
        -------
        OWSAResult
            Results with tornado plot and sensitivity summary.
        """
        from .results import OWSAResult

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

            is_model_level = param_name in self._MODEL_LEVEL_PARAMS

            for bound, val in [('low', low), ('high', high)]:
                test_params = base_params.copy()

                # Model-level params: temporarily modify model attributes
                saved_attrs = {}
                if is_model_level:
                    for attr in self._MODEL_LEVEL_PARAMS[param_name]:
                        saved_attrs[attr] = getattr(self, attr)
                        setattr(self, attr, val)
                else:
                    test_params[param_name] = val

                try:
                    result = self._simulate_single(test_params)
                finally:
                    # Restore model-level attributes
                    for attr, orig_val in saved_attrs.items():
                        setattr(self, attr, orig_val)

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
        """Run probabilistic sensitivity analysis (PSA).
        
        Parameters
        ----------
        n_sim : int
            Number of Monte Carlo simulations.
        seed : int, optional
            Random seed for reproducibility.
        progress : bool
            Whether to print progress updates.
        
        Returns
        -------
        PSAResult
            Results with CEAC, CE plane, and summary statistics.
        """
        from .results import PSAResult
        
        if seed is not None:
            np.random.seed(seed)
        
        # Sample parameters
        sampled_params = []
        for i in range(n_sim):
            p = self._get_base_params()
            for name, param in self.params.items():
                if param.dist is not None:
                    p[name] = float(param.dist.sample(1)[0])
            sampled_params.append(p)
        
        # Run simulations
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
    # Convenience / Info
    # =========================================================================
    
    def info(self) -> str:
        """Return a summary string describing the model."""
        lines = [
            f"MarkovModel",
            f"  States ({self.n_states}): {self.states}",
            f"  Strategies ({self.n_strategies}): {self.strategy_names}",
            f"  Cycles: {self.n_cycles} × {self.cycle_length} year(s)",
            f"  Discount rates: costs={self.dr_costs:.1%}, QALYs={self.dr_qalys:.1%}",
            f"  Half-cycle correction: {self._hcc_method or 'None'}",
            f"  Parameters ({len(self.params)}):",
        ]
        for name, p in self.params.items():
            dist_str = repr(p.dist) if p.dist else "Fixed"
            lines.append(f"    {name}: {p.base} [{dist_str}]")
        
        lines.append(f"  Cost categories ({len(self._costs)}):")
        for cat, cd in self._costs.items():
            flags = []
            if cd.first_cycle_only:
                flags.append("first-cycle")
            if cd.method == "starting":
                flags.append("one-time")
            flag_str = f" ({', '.join(flags)})" if flags else ""
            lines.append(f"    {cat}{flag_str}")
        
        if self._transition_costs:
            # Group by category for display
            tc_cats = {}
            for tc in self._transition_costs:
                cat = tc['category']
                if cat not in tc_cats:
                    tc_cats[cat] = []
                val = tc['value']
                # Detect schedule length for display
                sched_info = ""
                display_val = val
                if isinstance(val, dict):
                    # Strategy-specific: check if any value is a list
                    has_schedule = any(isinstance(v, (list, tuple)) for v in val.values())
                    if has_schedule:
                        max_len = max(
                            (len(v) for v in val.values() if isinstance(v, (list, tuple))),
                            default=1,
                        )
                        sched_info = f" [{max_len}-cycle schedule]"
                elif isinstance(val, (list, tuple)):
                    sched_info = f" [{len(val)}-cycle schedule]"
                tc_cats[cat].append(f"{tc['from_state']}→{tc['to_state']}{sched_info}")
            lines.append(f"  Transition costs ({len(self._transition_costs)}):")
            for cat, transitions in tc_cats.items():
                lines.append(f"    {cat}: {', '.join(transitions)}")
        
        return "\n".join(lines)
    
    def __repr__(self):
        return (
            f"MarkovModel(states={self.states}, "
            f"strategies={self.strategy_names}, "
            f"n_cycles={self.n_cycles})"
        )
