"""
MicroSimModel — Individual-level state transition microsimulation.

Unlike the cohort MarkovModel which tracks a hypothetical cohort proportion,
the microsimulation tracks individual patients through health states. Each
patient independently samples transitions, costs, and utilities each cycle.

Key features
------------
- Individual patient simulation with probabilistic state transitions
- Patient heterogeneity: individual attributes (age, sex, risk, …)
  that influence transition probabilities, costs, and utilities
- State entry/exit event handlers (e.g. one-time costs on entering a state)
- Tunnels: automatic sub-state tracking for time-in-state-dependent logic
- Per-patient outcome tracking (costs, QALYs, state history)
- Base case, OWSA, and PSA (outer-loop parameter uncertainty ×
  inner-loop patient stochasticity)
- Built-in convergence diagnostics

Typical workflow
----------------
>>> model = MicroSimModel(
...     states=["Healthy", "Sick", "Sicker", "Dead"],
...     strategies=["SOC", "New"],
...     n_cycles=40,
...     n_patients=5000,
... )
>>> model.add_param("p_HS", base=0.15, dist=ph.Beta(0.15, 0.03))
>>> model.set_transitions("SOC", lambda p, t, attrs: [...])
>>> model.set_state_cost(...)
>>> model.set_utility(...)
>>> result = model.run_base_case()
>>> print(result.summary())

References
----------
- Krijkamp EM, et al. (2018). Microsimulation modeling for health decision
  sciences using R: A tutorial. Medical Decision Making, 38(3), 400-422.
- DARTH group tutorial materials.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..distributions import Distribution
from .markov import Param, _CostDef
from ..utils import (
    C, _Complement, resolve_complement, resolve_value, discount_factor,
    normalize_hcc,
)


# =============================================================================
# Patient Population
# =============================================================================

@dataclass
class PatientProfile:
    """Defines a heterogeneous patient population.

    Attributes
    ----------
    n_patients : int
        Number of patients to simulate.
    attributes : dict
        Maps attribute name → array of length n_patients.
        Example: {"age": np.random.normal(60, 10, 5000),
                  "female": np.random.binomial(1, 0.5, 5000)}
    """
    n_patients: int
    attributes: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        for k, v in self.attributes.items():
            arr = np.asarray(v)
            if len(arr) != self.n_patients:
                raise ValueError(
                    f"Attribute '{k}' has length {len(arr)}, "
                    f"expected {self.n_patients}"
                )
            self.attributes[k] = arr

    def get(self, attr: str, idx: int) -> float:
        """Get attribute value for patient idx."""
        return float(self.attributes[attr][idx])

    def get_all(self, attr: str) -> np.ndarray:
        """Get all values for an attribute."""
        return self.attributes[attr]

    @staticmethod
    def homogeneous(n_patients: int) -> "PatientProfile":
        """Create a homogeneous population (no attributes)."""
        return PatientProfile(n_patients=n_patients)


# =============================================================================
# Microsimulation Model
# =============================================================================

class MicroSimModel:
    """Individual-level state transition microsimulation model.

    Each patient is independently simulated through health states.
    Transition probabilities can depend on model parameters, time (cycle),
    and individual patient attributes.

    Parameters
    ----------
    states : list of str
        Health state names.
    strategies : list of str or dict
        Treatment strategies.
    n_cycles : int
        Maximum number of cycles.
    n_patients : int
        Number of patients to simulate per run. Can be overridden
        by providing a PatientProfile.
    cycle_length : float
        Cycle length in years (default: 1.0).
    discount_rate : float or dict
        Annual discount rate(s). Keys: 'costs', 'qalys'.
    half_cycle_correction : bool or str or None
        Half-cycle correction method. Options:

        - True or ``"trapezoidal"``: endpoint weighting [0.5, 1, ..., 1, 0.5]
        - ``"life-table"``: average adjacent per-cycle rewards
        - False or None: no correction

        Default: True (trapezoidal).
    initial_state : str or int
        Starting state for all patients (default: 0).
    state_type : dict, optional
        Map state names to "alive" or "dead".
    seed : int, optional
        Random seed for reproducibility of base case.

    Notes
    -----
    The key difference from MarkovModel is that transitions are stochastic:
    at each cycle, each living patient independently samples their next state
    from the transition probability row for their current state.
    """

    def __init__(
        self,
        states: List[str],
        strategies: Union[List[str], Dict[str, str]],
        n_cycles: int,
        n_patients: int = 1000,
        cycle_length: float = 1.0,
        discount_rate: Union[float, Dict[str, float]] = 0.03,
        half_cycle_correction: Union[bool, str, None] = True,
        initial_state: Union[str, int] = 0,
        state_type: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
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

        # Model settings
        self.n_cycles = n_cycles
        self.n_patients = n_patients
        self.cycle_length = cycle_length
        self._hcc_method = normalize_hcc(half_cycle_correction)
        self.seed = seed

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

        # State types
        if state_type is not None:
            self._alive_states = set(
                i for i, s in enumerate(self.states)
                if state_type.get(s, "alive") == "alive"
            )
        else:
            self._alive_states = set(range(self.n_states - 1))

        # Absorbing states (dead)
        self._absorbing = set(range(self.n_states)) - self._alive_states

        # Parameters
        self.params: Dict[str, Param] = {}

        # Transitions: strategy -> callable(params, cycle, attrs_dict) -> matrix
        self._transitions: Dict[str, Any] = {}

        # Costs
        self._costs: Dict[str, _CostDef] = {}

        # Utilities
        self._utility: Any = None

        # Event handlers: {state_name: callable(patient_idx, cycle, attrs)}
        self._on_enter: Dict[str, List[Callable]] = {}
        self._on_exit: Dict[str, List[Callable]] = {}

        # Patient profile (overrides n_patients if set)
        self._profile: Optional[PatientProfile] = None

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
                  low=None, high=None) -> "MicroSimModel":
        """Add a single parameter."""
        self.params[name] = Param(
            base=base, dist=dist,
            label=label or name,
            low=low, high=high,
        )
        return self

    def add_params(self, params_dict: Dict[str, Union[Param, float]]) -> "MicroSimModel":
        """Add multiple parameters."""
        for name, param in params_dict.items():
            if isinstance(param, Param):
                if not param.label:
                    param.label = name
                self.params[name] = param
            elif isinstance(param, (int, float)):
                self.params[name] = Param(base=float(param), label=name)
            else:
                raise TypeError(f"Expected Param or numeric, got {type(param)}")
        return self

    # =========================================================================
    # Patient Population
    # =========================================================================

    def set_population(self, profile: PatientProfile) -> "MicroSimModel":
        """Set a heterogeneous patient population.

        Parameters
        ----------
        profile : PatientProfile
            Patient population with individual attributes.

        Examples
        --------
        >>> pop = PatientProfile(
        ...     n_patients=5000,
        ...     attributes={
        ...         "age": np.random.normal(60, 10, 5000),
        ...         "female": np.random.binomial(1, 0.5, 5000),
        ...     }
        ... )
        >>> model.set_population(pop)
        """
        self._profile = profile
        self.n_patients = profile.n_patients
        return self

    def _get_profile(self) -> PatientProfile:
        """Get current patient profile."""
        if self._profile is not None:
            return self._profile
        return PatientProfile.homogeneous(self.n_patients)

    def _get_patient_attrs(self, profile: PatientProfile, idx: int) -> dict:
        """Get all attributes for patient idx as a dict."""
        return {k: float(v[idx]) for k, v in profile.attributes.items()}

    # =========================================================================
    # Transitions
    # =========================================================================

    def set_transitions(self, strategy: str, transitions) -> "MicroSimModel":
        """Set transition probabilities for a strategy.

        Parameters
        ----------
        strategy : str
            Strategy name.
        transitions : callable or list
            Transition probability matrix. Options:

            - **Constant matrix** (list of lists): Use ``C`` for complement.
            - **Parameter-dependent**: ``f(params, cycle) -> matrix``
            - **Patient-dependent**: ``f(params, cycle, attrs) -> matrix``
              where ``attrs`` is a dict of patient attributes for the
              current individual.

        Notes
        -----
        Unlike the cohort model where the matrix is applied to the entire
        cohort vector, here each row is used as a probability distribution
        from which each patient's next state is sampled.

        Examples
        --------
        Age-dependent transitions:

        >>> model.set_transitions("SOC", lambda p, t, attrs: [
        ...     [C,  p["p_HS"] * (1 + attrs.get("age", 60)/100), 0.02],
        ...     [0,  C,                                           0.10],
        ...     [0,  0,                                           1   ],
        ... ])
        """
        if strategy not in self.strategy_names:
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {self.strategy_names}")
        self._transitions[strategy] = transitions
        return self

    # =========================================================================
    # Costs & Utilities
    # =========================================================================

    def set_state_cost(
        self,
        category: str,
        values: Any,
        first_cycle_only: bool = False,
        apply_cycles: Optional[List[int]] = None,
        method: str = "wlos",
    ) -> "MicroSimModel":
        """Define a cost category (same interface as MarkovModel).

        The ``values`` can also accept patient attributes:
        - ``callable(params, cycle, attrs) -> {state: cost}``
        """
        self._costs[category] = _CostDef(
            name=category,
            values=values,
            first_cycle_only=first_cycle_only,
            apply_cycles=apply_cycles,
            method=method,
        )
        return self

    def set_utility(self, values: Any) -> "MicroSimModel":
        """Define utility weights (same interface as MarkovModel).

        The ``values`` can also accept patient attributes:
        - ``callable(params, cycle, attrs) -> {state: utility}``
        """
        self._utility = values
        return self

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_state_enter(self, state: str, handler: Callable) -> "MicroSimModel":
        """Register a handler called when a patient enters a state.

        Parameters
        ----------
        state : str
            State name.
        handler : callable
            ``f(patient_idx, cycle, patient_attrs) -> dict or None``
            May return ``{"cost": float}`` to add a one-time transition cost.

        Examples
        --------
        >>> model.on_state_enter("Sick", lambda idx, t, a: {"cost": 5000})
        """
        if state not in self._on_enter:
            self._on_enter[state] = []
        self._on_enter[state].append(handler)
        return self

    def on_state_exit(self, state: str, handler: Callable) -> "MicroSimModel":
        """Register a handler called when a patient leaves a state."""
        if state not in self._on_exit:
            self._on_exit[state] = []
        self._on_exit[state].append(handler)
        return self

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _get_base_params(self) -> Dict[str, float]:
        return {name: p.base for name, p in self.params.items()}

    def _get_transition_matrix(self, strategy: str, params: dict,
                               cycle: int, attrs: dict) -> np.ndarray:
        """Compute transition matrix for given context."""
        trans = self._transitions[strategy]

        if callable(trans):
            # Try 3-arg (with attrs) first, fall back to 2-arg
            import inspect
            sig = inspect.signature(trans)
            n_args = len(sig.parameters)
            if n_args >= 3:
                matrix_data = trans(params, cycle, attrs)
            else:
                matrix_data = trans(params, cycle)
        else:
            matrix_data = trans

        if isinstance(matrix_data, np.ndarray):
            P = matrix_data.copy().astype(float)
        else:
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

        P = np.clip(P, 0.0, 1.0)
        row_sums = P.sum(axis=1, keepdims=True)
        mask = row_sums.flatten() > 0
        P[mask] = P[mask] / row_sums[mask]
        return P

    def _resolve_state_values(self, values, strategy: str, params: dict,
                              t: int, attrs: dict = None) -> np.ndarray:
        """Resolve state-level values. Supports (params,t) and (params,t,attrs)."""
        if callable(values):
            import inspect
            sig = inspect.signature(values)
            n_args = len(sig.parameters)
            if n_args >= 3 and attrs is not None:
                values = values(params, t, attrs)
            else:
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
                    result[:] = resolve_value(state_vals, params, t)
        else:
            for state_name, val in values.items():
                if state_name in self.states:
                    idx = self.states.index(state_name)
                    result[idx] = resolve_value(val, params, t)

        return result

    def _get_state_costs(self, category: str, strategy: str, params: dict,
                         t: int, attrs: dict = None) -> np.ndarray:
        cost_def = self._costs[category]
        if cost_def.first_cycle_only and t != 0:
            return np.zeros(self.n_states)
        if cost_def.apply_cycles is not None and t not in cost_def.apply_cycles:
            return np.zeros(self.n_states)
        if cost_def.method == "starting" and t != 0:
            return np.zeros(self.n_states)
        return self._resolve_state_values(cost_def.values, strategy, params, t, attrs)

    def _get_utilities(self, strategy: str, params: dict, t: int,
                       attrs: dict = None) -> np.ndarray:
        if self._utility is None:
            u = np.zeros(self.n_states)
            for i in self._alive_states:
                u[i] = 1.0
            return u
        return self._resolve_state_values(self._utility, strategy, params, t, attrs)

    # =========================================================================
    # Core Simulation Engine
    # =========================================================================

    def _simulate_patients(
        self,
        strategy: str,
        params: dict,
        profile: PatientProfile,
        rng: np.random.Generator,
    ) -> dict:
        """Simulate all patients for one strategy.

        Returns
        -------
        dict with keys:
            state_history : (n_patients, n_cycles+1) int array
            cost_history  : (n_patients, n_cycles+1) float array
            qaly_history  : (n_patients, n_cycles+1) float array
            ly_history    : (n_patients, n_cycles+1) float array
            event_costs   : (n_patients,) float — one-time event costs
            trace         : (n_cycles+1, n_states) float — mean state occupancy
        """
        N = profile.n_patients
        T = self.n_cycles

        # Histories
        state_hist = np.full((N, T + 1), -1, dtype=int)
        cost_hist = np.zeros((N, T + 1))
        qaly_hist = np.zeros((N, T + 1))
        ly_hist = np.zeros((N, T + 1))
        event_cost_total = np.zeros(N)

        # Initial state
        state_hist[:, 0] = self.initial_state_idx

        # Check if transitions depend on patient attributes
        has_attrs = bool(profile.attributes)

        # Pre-check transition callable arity
        trans_fn = self._transitions.get(strategy)
        trans_is_callable = callable(trans_fn)
        trans_needs_attrs = False
        if trans_is_callable:
            import inspect
            try:
                sig = inspect.signature(trans_fn)
                trans_needs_attrs = len(sig.parameters) >= 3
            except (ValueError, TypeError):
                trans_needs_attrs = False

        # If transitions don't depend on patient attrs, compute matrix once per cycle
        can_batch = not (trans_needs_attrs and has_attrs)

        for t in range(1, T + 1):
            cycle = t  # cycle number (1-based for transitions)

            if can_batch:
                # Compute one matrix for all patients
                P = self._get_transition_matrix(
                    strategy, params, cycle, {}
                )

                # Sample next state for all alive patients at once
                alive_mask = np.isin(state_hist[:, t - 1],
                                     list(self._alive_states))
                alive_indices = np.where(alive_mask)[0]

                if len(alive_indices) == 0:
                    state_hist[:, t] = state_hist[:, t - 1]
                    continue

                # Absorbing patients stay
                state_hist[:, t] = state_hist[:, t - 1]

                # For alive patients, sample from their current state's row
                current_states = state_hist[alive_indices, t - 1]

                # Vectorized multinomial sampling per-state-group
                for s in np.unique(current_states):
                    in_state = alive_indices[current_states == s]
                    probs = P[s]
                    # Ensure valid probability distribution
                    probs = np.clip(probs, 0, None)
                    total = probs.sum()
                    if total > 0:
                        probs = probs / total
                    else:
                        probs = np.zeros(self.n_states)
                        probs[s] = 1.0
                    new_states = rng.choice(self.n_states, size=len(in_state), p=probs)
                    state_hist[in_state, t] = new_states

            else:
                # Per-patient transitions (heterogeneous)
                state_hist[:, t] = state_hist[:, t - 1]

                for i in range(N):
                    if state_hist[i, t - 1] not in self._alive_states:
                        continue
                    attrs = self._get_patient_attrs(profile, i)
                    P = self._get_transition_matrix(strategy, params, cycle, attrs)
                    s = state_hist[i, t - 1]
                    probs = P[s]
                    probs = np.clip(probs, 0, None)
                    total = probs.sum()
                    if total > 0:
                        probs = probs / total
                    else:
                        probs = np.zeros(self.n_states)
                        probs[s] = 1.0
                    state_hist[i, t] = rng.choice(self.n_states, p=probs)

            # Event handlers for state transitions
            if self._on_enter or self._on_exit:
                prev_states = state_hist[:, t - 1]
                curr_states = state_hist[:, t]
                changed = prev_states != curr_states

                for i in np.where(changed)[0]:
                    prev_s = self.states[prev_states[i]]
                    curr_s = self.states[curr_states[i]]
                    attrs = self._get_patient_attrs(profile, i) if has_attrs else {}

                    # Exit handlers
                    if prev_s in self._on_exit:
                        for handler in self._on_exit[prev_s]:
                            result = handler(i, t, attrs)
                            if result and "cost" in result:
                                event_cost_total[i] += result["cost"]

                    # Enter handlers
                    if curr_s in self._on_enter:
                        for handler in self._on_enter[curr_s]:
                            result = handler(i, t, attrs)
                            if result and "cost" in result:
                                event_cost_total[i] += result["cost"]

        # --- Compute per-cycle costs and QALYs ---
        alive_mask_arr = np.zeros(self.n_states)
        for i in self._alive_states:
            alive_mask_arr[i] = 1.0

        # Pre-compute state-level costs and utilities per cycle
        # (assuming they don't vary by patient unless attrs-dependent)
        cost_needs_attrs = False
        util_needs_attrs = False
        for cat_name, cat_def in self._costs.items():
            if callable(cat_def.values):
                import inspect
                try:
                    sig = inspect.signature(cat_def.values)
                    if len(sig.parameters) >= 3:
                        cost_needs_attrs = True
                except (ValueError, TypeError):
                    pass
        if callable(self._utility):
            import inspect
            try:
                sig = inspect.signature(self._utility)
                if len(sig.parameters) >= 3:
                    util_needs_attrs = True
            except (ValueError, TypeError):
                pass

        needs_per_patient = (cost_needs_attrs or util_needs_attrs) and has_attrs

        for t in range(T + 1):
            if not needs_per_patient:
                # Vectorized: same costs/utilities for all patients
                total_cost_vec = np.zeros(self.n_states)
                for cat in self._costs:
                    c = self._get_state_costs(cat, strategy, params, t)
                    cost_def = self._costs[cat]
                    if cost_def.method == "wlos":
                        total_cost_vec += c * self.cycle_length
                    elif cost_def.method == "starting":
                        total_cost_vec += c
                    else:
                        total_cost_vec += c * self.cycle_length

                u_vec = self._get_utilities(strategy, params, t)
                u_vec = u_vec * self.cycle_length  # QALYs per cycle

                # Assign per patient based on their state
                states_t = state_hist[:, t]
                cost_hist[:, t] = total_cost_vec[states_t]
                qaly_hist[:, t] = u_vec[states_t]
                ly_hist[:, t] = alive_mask_arr[states_t] * self.cycle_length

            else:
                # Per-patient costs/utilities
                for i in range(N):
                    s = state_hist[i, t]
                    attrs = self._get_patient_attrs(profile, i)

                    total_cost = 0.0
                    for cat in self._costs:
                        c = self._get_state_costs(cat, strategy, params, t, attrs)
                        cost_def = self._costs[cat]
                        if cost_def.method == "wlos":
                            total_cost += c[s] * self.cycle_length
                        elif cost_def.method == "starting":
                            total_cost += c[s]
                        else:
                            total_cost += c[s] * self.cycle_length

                    cost_hist[i, t] = total_cost

                    u = self._get_utilities(strategy, params, t, attrs)
                    qaly_hist[i, t] = u[s] * self.cycle_length
                    ly_hist[i, t] = alive_mask_arr[s] * self.cycle_length

        # --- Half-cycle correction ---
        if self._hcc_method == "trapezoidal":
            hcc = np.ones(T + 1)
            hcc[0] = 0.5
            hcc[-1] = 0.5
            cost_hist *= hcc[np.newaxis, :]
            qaly_hist *= hcc[np.newaxis, :]
            ly_hist *= hcc[np.newaxis, :]

        elif self._hcc_method == "life-table":
            # Average adjacent per-cycle rewards
            for arr in (cost_hist, qaly_hist, ly_hist):
                orig = arr.copy()
                arr[:, :-1] = (orig[:, :-1] + orig[:, 1:]) / 2.0
                # last cycle unchanged

        # --- Discounting ---
        cycles = np.arange(T + 1, dtype=float)
        df_c = discount_factor(cycles, self.dr_costs, self.cycle_length)
        df_q = discount_factor(cycles, self.dr_qalys, self.cycle_length)

        cost_hist_disc = cost_hist * df_c[np.newaxis, :]
        qaly_hist_disc = qaly_hist * df_q[np.newaxis, :]
        ly_hist_disc = ly_hist * df_q[np.newaxis, :]

        # Add event costs (undiscounted for simplicity — they're one-offs)
        # Distribute across last alive cycle
        total_cost_per_patient = cost_hist_disc.sum(axis=1) + event_cost_total
        total_qaly_per_patient = qaly_hist_disc.sum(axis=1)
        total_ly_per_patient = ly_hist_disc.sum(axis=1)

        # --- State occupancy trace (proportion) ---
        trace = np.zeros((T + 1, self.n_states))
        for t in range(T + 1):
            for s in range(self.n_states):
                trace[t, s] = (state_hist[:, t] == s).mean()

        # --- Time-in-state (survival) ---
        # Compute time alive (cycles alive × cycle_length)
        alive_cycles = np.zeros(N)
        for i in range(N):
            for t in range(T + 1):
                if state_hist[i, t] in self._alive_states:
                    alive_cycles[i] = t
                else:
                    break
            else:
                alive_cycles[i] = T

        return {
            'state_history': state_hist,
            'cost_history': cost_hist_disc,
            'qaly_history': qaly_hist_disc,
            'ly_history': ly_hist_disc,
            'event_costs': event_cost_total,
            'total_cost': total_cost_per_patient,
            'total_qalys': total_qaly_per_patient,
            'total_lys': total_ly_per_patient,
            'trace': trace,
            'alive_cycles': alive_cycles,
            'mean_cost': float(total_cost_per_patient.mean()),
            'mean_qalys': float(total_qaly_per_patient.mean()),
            'mean_lys': float(total_ly_per_patient.mean()),
        }

    # =========================================================================
    # Analysis Entry Points
    # =========================================================================

    def run_base_case(
        self,
        profile: Optional[PatientProfile] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> "MicroSimResult":
        """Run deterministic base case microsimulation.

        Parameters
        ----------
        profile : PatientProfile, optional
            Patient population. If not set, uses self._profile or
            creates a homogeneous population.
        seed : int, optional
            Random seed. Overrides model-level seed.
        verbose : bool
            Print progress.

        Returns
        -------
        MicroSimResult
        """
        from ..analysis.results import MicroSimResult

        s = seed if seed is not None else self.seed
        rng = np.random.default_rng(s)
        params = self._get_base_params()
        prof = profile or self._get_profile()

        results = {}
        for strat in self.strategy_names:
            if verbose:
                print(f"  Simulating {prof.n_patients} patients: {strat}...", end=" ")
            results[strat] = self._simulate_patients(strat, params, prof, rng)
            if verbose:
                print(f"mean cost={results[strat]['mean_cost']:,.0f}, "
                      f"mean QALYs={results[strat]['mean_qalys']:.3f}")

        return MicroSimResult(model=self, results=results, params=params)

    def run_psa(
        self,
        n_outer: int = 200,
        n_inner: Optional[int] = None,
        seed: Optional[int] = None,
        profile: Optional[PatientProfile] = None,
        verbose: bool = True,
    ) -> "MicroSimPSAResult":
        """Run probabilistic sensitivity analysis.

        Two-level simulation:
        - **Outer loop** (n_outer): sample parameter values from distributions
        - **Inner loop** (n_inner patients): simulate individuals with those params

        Parameters
        ----------
        n_outer : int
            Number of parameter sets to draw (default: 200).
        n_inner : int, optional
            Patients per parameter draw. Default: self.n_patients.
        seed : int, optional
            Random seed.
        verbose : bool
            Print progress.

        Returns
        -------
        MicroSimPSAResult
        """
        from ..analysis.results import MicroSimPSAResult

        s = seed if seed is not None else self.seed
        rng = np.random.default_rng(s)

        n_inner = n_inner or self.n_patients
        prof = profile or PatientProfile.homogeneous(n_inner)

        psa_results = []
        sampled_params = []

        for i in range(n_outer):
            # Sample parameters
            p = self._get_base_params()
            for name, param in self.params.items():
                if param.dist is not None:
                    p[name] = float(param.dist.sample(1)[0])
            sampled_params.append(p)

            # Simulate all strategies
            iter_results = {}
            for strat in self.strategy_names:
                iter_results[strat] = self._simulate_patients(strat, p, prof, rng)

            psa_results.append(iter_results)

            if verbose and (i + 1) % max(1, n_outer // 10) == 0:
                print(f"  PSA: {i+1}/{n_outer} ({100*(i+1)/n_outer:.0f}%)")

        if verbose:
            print(f"  PSA complete: {n_outer} outer × {n_inner} inner")

        return MicroSimPSAResult(
            model=self,
            psa_results=psa_results,
            sampled_params=sampled_params,
        )

    def run_owsa(
        self,
        params: Optional[List[str]] = None,
        wtp: float = 50000,
        seed: Optional[int] = None,
        profile: Optional[PatientProfile] = None,
        n_patients: Optional[int] = None,
        verbose: bool = True,
    ) -> "OWSAResult":
        """Run one-way sensitivity analysis.

        Parameters
        ----------
        params : list of str, optional
            Parameters to vary. Default: all with PSA distributions.
        wtp : float
            WTP threshold for NMB.
        seed : int, optional
            Random seed (shared for all runs to reduce noise).
        profile : PatientProfile, optional
            Patient population.
        n_patients : int, optional
            Override number of patients (use more for lower noise).
        verbose : bool
            Print progress.

        Returns
        -------
        OWSAResult
        """
        from ..analysis.results import OWSAResult

        if params is None:
            params = [n for n, p in self.params.items() if p.dist is not None]
            if not params:
                params = list(self.params.keys())

        s = seed if seed is not None else self.seed
        n_p = n_patients or self.n_patients
        prof = profile or PatientProfile.homogeneous(n_p)

        base_params = self._get_base_params()

        # Base case
        rng = np.random.default_rng(s)
        base_result = {}
        for strat in self.strategy_names:
            sim = self._simulate_patients(strat, base_params, prof, rng)
            # Convert to cohort-like format for OWSAResult compatibility
            base_result[strat] = {
                'total_costs': {'total': sim['mean_cost']},
                'total_qalys': sim['mean_qalys'],
                'total_lys': sim['mean_lys'],
            }

        owsa_data = []
        for param_name in params:
            p = self.params[param_name]
            low = p.low if p.low is not None else p.base * 0.8
            high = p.high if p.high is not None else p.base * 1.2

            for bound, val in [('low', low), ('high', high)]:
                test_params = base_params.copy()
                test_params[param_name] = val

                rng = np.random.default_rng(s)  # Reset seed for comparability
                result = {}
                for strat in self.strategy_names:
                    sim = self._simulate_patients(strat, test_params, prof, rng)
                    result[strat] = {
                        'total_costs': {'total': sim['mean_cost']},
                        'total_qalys': sim['mean_qalys'],
                        'total_lys': sim['mean_lys'],
                    }

                owsa_data.append({
                    'param': param_name,
                    'label': p.label,
                    'value': val,
                    'base_value': p.base,
                    'bound': bound,
                    'result': result,
                })

            if verbose:
                print(f"  OWSA: {param_name} done")

        return OWSAResult(
            model=self,
            base_result=base_result,
            base_params=base_params,
            owsa_data=owsa_data,
            wtp=wtp,
        )

    # =========================================================================
    # Convenience
    # =========================================================================

    def info(self) -> str:
        """Summary string."""
        lines = [
            f"MicroSimModel (Individual-Level Simulation)",
            f"  States ({self.n_states}): {self.states}",
            f"  Strategies ({self.n_strategies}): {self.strategy_names}",
            f"  Cycles: {self.n_cycles} × {self.cycle_length} year(s)",
            f"  Patients: {self.n_patients}",
            f"  Discount rates: costs={self.dr_costs:.1%}, QALYs={self.dr_qalys:.1%}",
            f"  Half-cycle correction: {self._hcc_method or 'None'}",
            f"  Parameters ({len(self.params)}):",
        ]
        for name, p in self.params.items():
            dist_str = repr(p.dist) if p.dist else "Fixed"
            lines.append(f"    {name}: {p.base} [{dist_str}]")

        if self._profile and self._profile.attributes:
            lines.append(f"  Patient attributes: {list(self._profile.attributes.keys())}")

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
            f"MicroSimModel(states={self.states}, "
            f"strategies={self.strategy_names}, "
            f"n_cycles={self.n_cycles}, n_patients={self.n_patients})"
        )
