"""
CohortStateTransitionModel - Core model class for cohort discrete-time state transition models.

This module implements the main CohortStateTransitionModel class which provides:
- Flexible parameter definition with PSA distributions
- Transition probability matrices (constant or time-varying)
- Flexible cost/utility definitions (per-cycle, first-cycle-only, time-dependent)
- Base case, OWSA, and PSA analysis
"""

import numpy as np
import pandas as pd
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..distributions import Distribution, sample_distribution
from ..utils import (
    C, _Complement, resolve_complement, resolve_value, discount_factor,
    normalize_hcc, interval_occupancy, validate_transition_matrix,
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
# CohortStateTransitionModel
# =============================================================================

class CohortStateTransitionModel:
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
        Number of model intervals to simulate. The state trace contains
        ``n_cycles + 1`` observation points, including time zero.
    cycle_length : float
        Length of each cycle in years (default: 1.0).
    dr_cost : float or Param
        Annual discount rate for costs. Default: 0 (no discounting).
        Pass a ``Param`` object to enable sensitivity analysis on this rate.
    dr_qaly : float or Param
        Annual discount rate for QALYs. Default: 0 (no discounting).
        Pass a ``Param`` object to enable sensitivity analysis on this rate.
    half_cycle_correction : bool or str or None
        Half-cycle correction method. Options:

        - True, ``"trapezoidal"``, or ``"life-table"``: average the two
          adjacent state observations within each interval.
        - False or None: no correction

        Default: True (trapezoidal).
    initial_state : str or int
        Starting health state (default: 0, the first state).
    state_type : dict, optional
        Map state names to type: "alive" or "dead". Used for LY calculation.
        By default, the last state is considered "dead".
    discount_convention : str
        ``"discrete"`` uses ``(1 + rate) ** -time``; ``"continuous"`` uses
        ``exp(-rate * time)``. Default: ``"discrete"``.
    
    Examples
    --------
    >>> model = CohortStateTransitionModel(
    ...     states=["Healthy", "Sick", "Dead"],
    ...     strategies=["SOC", "New"],
    ...     n_cycles=20,
    ...     cycle_length=1.0,
    ...     dr_cost=0.03,
    ...     dr_qaly=0.03,
    ... )
    """
    
    def __init__(
        self,
        states: List[str],
        strategies: Union[List[str], Dict[str, str]],
        n_cycles: int,
        cycle_length: float = 1.0,
        dr_cost: Union[float, "Param"] = 0.0,
        dr_qaly: Union[float, "Param"] = 0.0,
        half_cycle_correction: Union[bool, str, None] = True,
        initial_state: Union[str, int] = 0,
        state_type: Optional[Dict[str, str]] = None,
        discount_convention: str = "discrete",
    ):
        # States
        self.states = list(states)
        if not self.states:
            raise ValueError("states must contain at least one state")
        if len(set(self.states)) != len(self.states):
            raise ValueError(f"State names must be unique, got {self.states!r}")
        self.n_states = len(self.states)
        
        # Strategies
        if isinstance(strategies, dict):
            self.strategy_names = list(strategies.keys())
            self.strategy_labels = dict(strategies)
        else:
            self.strategy_names = list(strategies)
            self.strategy_labels = {s: s for s in self.strategy_names}
        if not self.strategy_names:
            raise ValueError("strategies must contain at least one strategy")
        if len(set(self.strategy_names)) != len(self.strategy_names):
            raise ValueError(
                f"Strategy names must be unique, got {self.strategy_names!r}"
            )
        self.n_strategies = len(self.strategy_names)
        
        # Model cycles
        if isinstance(n_cycles, bool) or not isinstance(n_cycles, (int, np.integer)):
            raise TypeError(f"n_cycles must be an integer, got {type(n_cycles).__name__}")
        if n_cycles <= 0:
            raise ValueError(f"n_cycles must be positive, got {n_cycles!r}")
        if not np.isfinite(cycle_length) or cycle_length <= 0:
            raise ValueError(
                f"cycle_length must be a positive finite number, got {cycle_length!r}"
            )
        if discount_convention not in {"discrete", "continuous"}:
            raise ValueError(
                f"Unknown discount_convention {discount_convention!r}; "
                "expected 'discrete' or 'continuous'."
            )
        self.n_cycles = int(n_cycles)
        self.cycle_length = float(cycle_length)
        self.discount_convention = discount_convention
        self._hcc_method = normalize_hcc(half_cycle_correction)

        # Parameters (init early so discount rates can register into it)
        self.params: Dict[str, Param] = {}

        # Discount rates
        if isinstance(dr_cost, Param):
            self.dr_cost = dr_cost.base
            if not dr_cost.label:
                dr_cost.label = "Discount Rate (Cost)"
            self.params["dr_cost"] = dr_cost
        else:
            self.dr_cost = float(dr_cost)
        if isinstance(dr_qaly, Param):
            self.dr_qaly = dr_qaly.base
            if not dr_qaly.label:
                dr_qaly.label = "Discount Rate (QALY)"
            self.params["dr_qaly"] = dr_qaly
        else:
            self.dr_qaly = float(dr_qaly)
        discount_factor(0, self.dr_cost, convention=self.discount_convention)
        discount_factor(0, self.dr_qaly, convention=self.discount_convention)
        
        # Initial state
        if isinstance(initial_state, str):
            if initial_state not in self.states:
                raise ValueError(
                    f"Unknown initial_state {initial_state!r}; "
                    f"available states are {self.states!r}"
                )
            self.initial_state_idx = self.states.index(initial_state)
        else:
            self.initial_state_idx = int(initial_state)
            if not 0 <= self.initial_state_idx < self.n_states:
                raise ValueError(
                    f"initial_state index must be between 0 and "
                    f"{self.n_states - 1}, got {self.initial_state_idx}"
                )
        
        # State types (alive vs dead) for LY calculation
        if state_type is not None:
            unknown_states = set(state_type) - set(self.states)
            if unknown_states:
                raise ValueError(
                    f"state_type contains unknown states: {sorted(unknown_states)!r}"
                )
            invalid_types = {
                name: value for name, value in state_type.items()
                if value not in {"alive", "dead"}
            }
            if invalid_types:
                raise ValueError(
                    "state_type values must be 'alive' or 'dead'; "
                    f"got {invalid_types!r}"
                )
            self._alive_states = [
                i for i, s in enumerate(self.states) 
                if state_type.get(s, "alive") == "alive"
            ]
        else:
            # Default: all states except last are "alive"
            self._alive_states = list(range(self.n_states - 1))

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
                  low=None, high=None) -> "CohortStateTransitionModel":
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
        CohortStateTransitionModel
            Self, for method chaining.
        """
        self.params[name] = Param(
            base=base, dist=dist, 
            label=label or name,
            low=low, high=high,
        )
        return self
    
    def add_params(self, params_dict: Dict[str, Union[Param, float]]) -> "CohortStateTransitionModel":
        """Add multiple parameters at once.
        
        Parameters
        ----------
        params_dict : dict
            Maps parameter names to Param objects or numeric values.
            
        Returns
        -------
        CohortStateTransitionModel
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
    
    def set_transitions(self, strategy: str, transitions) -> "CohortStateTransitionModel":
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
        CohortStateTransitionModel
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
        if not callable(transitions):
            self._resolve_transition_data(transitions, self._get_base_params(), 0, strategy)
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
    ) -> "CohortStateTransitionModel":
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
        CohortStateTransitionModel
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
        if method not in {"wlos", "starting"}:
            raise ValueError(
                f"Unknown cost method {method!r}; expected 'wlos' or 'starting'."
            )
        if method == "starting" and first_cycle_only:
            raise ValueError(
                "first_cycle_only cannot be combined with method='starting'; "
                "a starting cost already occurs once at t=0."
            )
        if method == "starting" and apply_cycles is not None:
            raise ValueError(
                "apply_cycles cannot be combined with method='starting'; "
                "a starting cost occurs once at t=0."
            )
        if apply_cycles is not None:
            try:
                apply_cycles = tuple(apply_cycles)
            except TypeError as exc:
                raise TypeError("apply_cycles must be an iterable of interval indices") from exc
            invalid_cycles = [
                cycle for cycle in apply_cycles
                if isinstance(cycle, bool)
                or not isinstance(cycle, (int, np.integer))
                or not 0 <= int(cycle) < self.n_cycles
            ]
            if invalid_cycles:
                raise ValueError(
                    f"apply_cycles contains invalid interval indices {invalid_cycles!r}; "
                    f"expected integers from 0 to {self.n_cycles - 1}."
                )
            apply_cycles = tuple(int(cycle) for cycle in apply_cycles)
        if not callable(values):
            self._validate_state_mapping(values, "state cost")
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
    ) -> "CohortStateTransitionModel":
        """Define costs triggered when patients transition between states.
        
        In a cohort model, the cost is applied to the **flow** of patients.
        For interval ``t`` (0-based), the event cost is
        ``trace[t, from] × P[t, from, to] × unit_cost`` and is paid at
        the end of that interval.
        
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
        CohortStateTransitionModel
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
    ) -> "CohortStateTransitionModel":
        """Define a custom cost computed from simulation state each cycle.

        Unlike ``set_transition_cost`` which targets individual state pairs,
        this method gives full access to the transition matrix and state
        distribution, allowing arbitrary cost logic.

        The user-supplied function is called once per interval
        (``t = 0, ..., n_cycles - 1``)
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
            - **t** (int): Current interval index (0-based).
            - **state_prev** (np.ndarray): State proportions at interval start.
            - **state_curr** (np.ndarray): State proportions at interval end.
            - **P** (np.ndarray): Transition probability matrix at cycle *t*.
            - **states** (list[str]): State names (same order as array indices).

        Returns
        -------
        CohortStateTransitionModel
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
            unknown = set(val) - set(self.strategy_names)
            if unknown:
                raise ValueError(
                    f"Transition cost '{tc['category']}' contains unknown "
                    f"strategies: {sorted(unknown)!r}"
                )
            val = val.get(strategy, 0)
        # Already a schedule (list/tuple)
        if isinstance(val, (list, tuple)):
            resolved = []
            for v in val:
                resolved.append(resolve_value(v, params))
            return resolved
        # Callable → signal to engine to use per-cycle evaluation
        if callable(val):
            return None
        # Scalar: parameter reference or float → single-element schedule
        if isinstance(val, str):
            val = resolve_value(val, params)
        return [float(val)]

    # =========================================================================
    # Utility
    # =========================================================================
    
    def set_utility(self, values: Any) -> "CohortStateTransitionModel":
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
        CohortStateTransitionModel
            Self, for method chaining.
            
        Examples
        --------
        >>> model.set_utility({
        ...     "PFS": "u_pfs",          # Parameter reference
        ...     "Progressed": "u_prog",
        ...     "Dead": 0.0,
        ... })
        """
        if not callable(values):
            self._validate_state_mapping(values, "utility")
        self._utility = values
        return self
    
    # =========================================================================
    # Internal: Parameter Resolution
    # =========================================================================
    
    def _get_base_params(self) -> Dict[str, float]:
        """Get base case parameter values as a dict."""
        return {name: p.base for name, p in self.params.items()}

    def _resolve_transition_data(
        self, transitions: Any, params: Dict[str, float], cycle: int,
        strategy: str,
    ) -> np.ndarray:
        """Resolve and validate one transition matrix without repairing it."""
        matrix_data = transitions(params, cycle) if callable(transitions) else transitions

        if isinstance(matrix_data, np.ndarray):
            try:
                matrix = matrix_data.astype(float, copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Transition matrix for strategy {strategy!r}, interval {cycle} "
                    "must contain only numeric probabilities."
                ) from exc
        else:
            try:
                rows = list(matrix_data)
            except TypeError as exc:
                raise TypeError(
                    f"Transition matrix for strategy {strategy!r}, interval {cycle} "
                    "must be a 2D array or iterable of rows."
                ) from exc

            if len(rows) != self.n_states:
                raise ValueError(
                    f"Transition matrix for strategy {strategy!r}, interval {cycle} "
                    f"has {len(rows)} rows; expected {self.n_states}."
                )
            resolved = []
            for row_index, row in enumerate(rows):
                try:
                    values = list(row)
                except TypeError as exc:
                    raise TypeError(
                        f"Transition row {row_index} for strategy {strategy!r}, "
                        f"interval {cycle} is not iterable."
                    ) from exc
                if len(values) != self.n_states:
                    raise ValueError(
                        f"Transition row {row_index} for strategy {strategy!r}, "
                        f"interval {cycle} has {len(values)} values; "
                        f"expected {self.n_states}."
                    )
                resolved_row = []
                for value in values:
                    if isinstance(value, _Complement) or value is C:
                        resolved_row.append(C)
                    elif callable(value):
                        resolved_row.append(float(value(params, cycle)))
                    else:
                        resolved_row.append(float(value))
                resolved.append(resolved_row)
            matrix = resolve_complement(resolved)

        expected_shape = (self.n_states, self.n_states)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Transition matrix for strategy {strategy!r}, interval {cycle} "
                f"has shape {matrix.shape}; expected {expected_shape}."
            )
        try:
            validate_transition_matrix(matrix)
        except ValueError as exc:
            raise ValueError(
                f"Invalid transition matrix for strategy {strategy!r}, "
                f"interval {cycle}: {exc}"
            ) from exc
        return matrix

    def _get_transition_matrix(self, strategy: str, params: Dict[str, float],
                               cycle: int) -> np.ndarray:
        """Compute the transition probability matrix for a given context."""
        if strategy not in self._transitions:
            raise ValueError(
                f"No transition matrix configured for strategy {strategy!r}."
            )
        return self._resolve_transition_data(
            self._transitions[strategy], params, cycle, strategy
        )

    def _validate_state_mapping(self, values: Any, label: str) -> None:
        """Validate state/strategy keys while allowing omitted known states."""
        if not isinstance(values, dict):
            raise TypeError(
                f"{label} values must be a mapping or callable, "
                f"got {type(values).__name__}."
            )
        if not values:
            return

        keys = set(values)
        state_names = set(self.states)
        strategy_names = set(self.strategy_names)

        if keys <= strategy_names and all(
            isinstance(value, dict) for value in values.values()
        ):
            for strategy, state_values in values.items():
                unknown = set(state_values) - state_names
                if unknown:
                    raise ValueError(
                        f"{label} for strategy {strategy!r} contains unknown "
                        f"states: {sorted(unknown)!r}."
                    )
            return

        if keys <= state_names:
            return

        unknown = keys - state_names - strategy_names
        if unknown:
            raise ValueError(
                f"{label} contains unknown state or strategy names: "
                f"{sorted(unknown)!r}."
            )
        raise ValueError(
            f"{label} mixes state-level and strategy-level keys; "
            "use either {state: value} or {strategy: {state: value}}."
        )
    
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
        self._validate_state_mapping(values, "Resolved state value")

        result = np.zeros(self.n_states)
        
        if not values:
            return result
        
        if set(values) <= set(self.strategy_names) and all(
            isinstance(value, dict) for value in values.values()
        ):
            # Format: {strategy: {state: value}}
            if strategy in values:
                state_vals = values[strategy]
                for state_name, val in state_vals.items():
                    idx = self.states.index(state_name)
                    result[idx] = resolve_value(val, params, t)
        else:
            # Format: {state: value} — same for all strategies
            for state_name, val in values.items():
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
        missing = [
            strategy for strategy in self.strategy_names
            if strategy not in self._transitions
        ]
        if missing:
            raise ValueError(
                f"Missing transition matrices for strategies: {missing!r}"
            )

        results = {}
        n_intervals = self.n_cycles
        interval_index = np.arange(n_intervals, dtype=float)
        flow_df_cost = discount_factor(
            interval_index + 0.5, self.dr_cost, self.cycle_length,
            self.discount_convention,
        )
        flow_df_qaly = discount_factor(
            interval_index + 0.5, self.dr_qaly, self.cycle_length,
            self.discount_convention,
        )
        event_df_cost = discount_factor(
            interval_index + 1.0, self.dr_cost, self.cycle_length,
            self.discount_convention,
        )

        alive_mask = np.zeros(self.n_states)
        alive_mask[self._alive_states] = 1.0

        for strategy in self.strategy_names:
            matrices = [
                self._get_transition_matrix(strategy, params, interval)
                for interval in range(n_intervals)
            ]

            trace = np.zeros((n_intervals + 1, self.n_states))
            trace[0, self.initial_state_idx] = 1.0
            for interval, matrix in enumerate(matrices):
                trace[interval + 1] = trace[interval] @ matrix

            start_occupancy = interval_occupancy(trace, None)
            reward_occupancy = interval_occupancy(trace, self._hcc_method)

            qalys = np.zeros(n_intervals)
            qalys_hcc = np.zeros(n_intervals)
            lys = start_occupancy @ alive_mask * self.cycle_length
            lys_hcc = reward_occupancy @ alive_mask * self.cycle_length

            state_costs_raw = {
                category: np.zeros(n_intervals) for category in self._costs
            }
            state_costs_hcc = {
                category: np.zeros(n_intervals) for category in self._costs
            }
            starting_costs = {
                category: np.zeros(n_intervals) for category in self._costs
            }
            event_costs: Dict[str, np.ndarray] = {}

            for interval in range(n_intervals):
                utility = self._get_utilities(strategy, params, interval)
                qalys[interval] = (
                    np.dot(start_occupancy[interval], utility)
                    * self.cycle_length
                )
                qalys_hcc[interval] = (
                    np.dot(reward_occupancy[interval], utility)
                    * self.cycle_length
                )

                for category, cost_def in self._costs.items():
                    costs = self._get_state_costs(
                        category, strategy, params, interval
                    )
                    if cost_def.method == "starting":
                        if interval == 0:
                            amount = float(np.dot(trace[0], costs))
                            starting_costs[category][0] = amount
                    else:
                        state_costs_raw[category][interval] = (
                            np.dot(start_occupancy[interval], costs)
                            * self.cycle_length
                        )
                        state_costs_hcc[category][interval] = (
                            np.dot(reward_occupancy[interval], costs)
                            * self.cycle_length
                        )

            for transition_cost in self._transition_costs:
                category = transition_cost['category']
                category_events = event_costs.setdefault(
                    category, np.zeros(n_intervals)
                )
                from_index = transition_cost['from_idx']
                to_index = transition_cost['to_idx']
                inflows = np.array([
                    trace[interval, from_index]
                    * matrices[interval][from_index, to_index]
                    for interval in range(n_intervals)
                ])
                schedule = self._get_tc_schedule(
                    transition_cost, strategy, params
                )

                if schedule is None:
                    value = transition_cost['value']
                    if isinstance(value, dict):
                        value = value.get(strategy, 0)
                    for interval, flow in enumerate(inflows):
                        category_events[interval] += (
                            flow * resolve_value(value, params, interval)
                        )
                else:
                    for source_interval, flow in enumerate(inflows):
                        for offset, amount in enumerate(schedule):
                            target_interval = source_interval + offset
                            if target_interval < n_intervals:
                                category_events[target_interval] += flow * amount

            for custom_cost in self._custom_costs:
                category = custom_cost['category']
                category_events = event_costs.setdefault(
                    category, np.zeros(n_intervals)
                )
                for interval, matrix in enumerate(matrices):
                    amount = float(custom_cost['func'](
                        strategy, params, interval,
                        trace[interval], trace[interval + 1],
                        matrix, self.states,
                    ))
                    if not np.isfinite(amount):
                        raise ValueError(
                            f"Custom cost {category!r} returned a non-finite "
                            f"value for strategy {strategy!r}, interval {interval}."
                        )
                    category_events[interval] += amount

            categories = list(dict.fromkeys([*self._costs, *event_costs]))
            costs_by_cycle = {}
            costs_hcc = {}
            discounted_costs = {}
            for category in categories:
                state_raw = state_costs_raw.get(
                    category, np.zeros(n_intervals)
                )
                state_hcc = state_costs_hcc.get(
                    category, np.zeros(n_intervals)
                )
                at_start = starting_costs.get(
                    category, np.zeros(n_intervals)
                )
                at_event = event_costs.get(
                    category, np.zeros(n_intervals)
                )
                costs_by_cycle[category] = state_raw + at_start + at_event
                costs_hcc[category] = state_hcc + at_start + at_event
                discounted_costs[category] = (
                    state_hcc * flow_df_cost
                    + at_start
                    + at_event * event_df_cost
                )

            discounted_qalys = qalys_hcc * flow_df_qaly
            discounted_lys = lys_hcc * flow_df_qaly

            results[strategy] = {
                'trace': trace,
                'interval_times': (interval_index + 0.5) * self.cycle_length,
                'costs_by_cycle': costs_by_cycle,
                'qalys_by_cycle': qalys,
                'lys_by_cycle': lys,
                'costs_hcc': costs_hcc,
                'qalys_hcc': qalys_hcc,
                'lys_hcc': lys_hcc,
                'discounted_costs': discounted_costs,
                'discounted_qalys': discounted_qalys,
                'discounted_lys': discounted_lys,
                'total_costs': {
                    category: float(np.sum(values))
                    for category, values in discounted_costs.items()
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
        from ..analysis.results import BaseResult
        params = self._get_base_params()
        sim = self._simulate_single(params)
        return BaseResult(model=self, results=sim, params=params)
    
    # Parameters that live as model attributes rather than in the params dict
    # passed to _simulate_single. When varied in OWSA or sampled in PSA, the
    # corresponding model attribute must be temporarily overwritten -- reading
    # them out of the params dict alone has no effect on the simulation.
    _ATTR_PARAMS = {'dr_cost', 'dr_qaly'}

    @contextmanager
    def _attr_param_override(self, values: Dict[str, float]):
        """Temporarily apply any _ATTR_PARAMS present in `values`."""
        saved = {
            name: getattr(self, name)
            for name in self._ATTR_PARAMS
            if name in values
        }
        try:
            for name in saved:
                setattr(self, name, values[name])
            yield
        finally:
            for name, original in saved.items():
                setattr(self, name, original)

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
            Parameter names to vary. Default: all parameters with
            low/high bounds or distributions defined.
        range_pct : float
            Percentage range for variation if low/high not set (default: ±20%).
        wtp : float
            Willingness-to-pay threshold for NMB calculation.

        Returns
        -------
        OWSAResult
            Results with tornado plot and sensitivity summary.
        """
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

            is_attr = param_name in self._ATTR_PARAMS

            for bound, val in [('low', low), ('high', high)]:
                test_params = base_params.copy()
                test_params[param_name] = val

                if is_attr:
                    with self._attr_param_override({param_name: val}):
                        result = self._simulate_single(test_params)
                else:
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
        from ..analysis.results import PSAResult

        if isinstance(n_sim, bool) or not isinstance(n_sim, (int, np.integer)):
            raise TypeError("n_sim must be a positive integer")
        if n_sim <= 0:
            raise ValueError("n_sim must be a positive integer")
        rng = np.random.default_rng(seed)
        
        # Sample parameters
        sampled_params = []
        for i in range(n_sim):
            p = self._get_base_params()
            for name, param in self.params.items():
                if param.dist is not None:
                    p[name] = float(sample_distribution(param.dist, 1, rng)[0])
            sampled_params.append(p)
        
        # Run simulations
        psa_results = []
        for i, p in enumerate(sampled_params):
            if progress and (i + 1) % max(1, n_sim // 10) == 0:
                print(f"  PSA: {i+1}/{n_sim} ({100*(i+1)/n_sim:.0f}%)")
            with self._attr_param_override(p):
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
            f"CohortStateTransitionModel",
            f"  States ({self.n_states}): {self.states}",
            f"  Strategies ({self.n_strategies}): {self.strategy_names}",
            f"  Cycles: {self.n_cycles} × {self.cycle_length} year(s)",
            f"  Discount rates: cost={self.dr_cost:.1%}, QALY={self.dr_qaly:.1%}",
            f"  Discount convention: {self.discount_convention}",
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
            f"CohortStateTransitionModel(states={self.states}, "
            f"strategies={self.strategy_names}, "
            f"n_cycles={self.n_cycles})"
        )


# Concise public alias retained for compatibility and everyday use.
MarkovModel = CohortStateTransitionModel
