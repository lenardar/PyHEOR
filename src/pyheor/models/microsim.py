"""
IndividualStateTransitionModel — Individual-level state transition microsimulation.

Unlike the cohort MarkovModel which tracks a hypothetical cohort proportion,
the microsimulation tracks individual patients through health states. Each
patient independently samples transitions, costs, and utilities each cycle.

Key features
------------
- Individual patient simulation with probabilistic state transitions
- Patient heterogeneity: individual attributes (age, sex, risk, …)
  that influence transition probabilities, costs, and utilities
- State entry/exit event handlers (e.g. one-time costs on entering a state)
- Per-patient outcome tracking (costs, QALYs, state history)
- Base case, OWSA, and PSA (outer-loop parameter uncertainty ×
  inner-loop patient stochasticity)
- Built-in convergence diagnostics

Typical workflow
----------------
>>> model = IndividualStateTransitionModel(
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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .common import Param as _Param, StateMappingModel, _CostDef
from ..distributions import sample_distribution
from ..utils import (
    C, _Complement, resolve_complement, resolve_value, discount_factor,
    normalize_hcc, validate_transition_matrix,
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

class IndividualStateTransitionModel(StateMappingModel):
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
    dr_cost : float or Param
        Annual discount rate for costs. Default: 0 (no discounting).
        Pass a ``Param`` to enable sensitivity analysis.
    dr_qaly : float or Param
        Annual discount rate for QALYs. Default: 0 (no discounting).
        Pass a ``Param`` to enable sensitivity analysis.
    half_cycle_correction : bool or str or None
        Half-cycle correction method. Options:

        - True or ``"trapezoidal"``: endpoint weighting [0.5, 1, ..., 1, 0.5]
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
        dr_cost: Union[float, "_Param"] = 0.0,
        dr_qaly: Union[float, "_Param"] = 0.0,
        half_cycle_correction: Union[bool, str, None] = True,
        initial_state: Union[str, int] = 0,
        state_type: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
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

        # Model settings
        if isinstance(n_cycles, bool) or not isinstance(n_cycles, (int, np.integer)):
            raise TypeError(
                f"n_cycles must be an integer, got {type(n_cycles).__name__}"
            )
        if n_cycles <= 0:
            raise ValueError(f"n_cycles must be positive, got {n_cycles!r}")
        if isinstance(n_patients, bool) or not isinstance(n_patients, (int, np.integer)):
            raise TypeError(
                f"n_patients must be an integer, got {type(n_patients).__name__}"
            )
        if n_patients <= 0:
            raise ValueError(f"n_patients must be positive, got {n_patients!r}")
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
        self.n_patients = int(n_patients)
        self.cycle_length = float(cycle_length)
        self.discount_convention = discount_convention
        self._hcc_method = normalize_hcc(half_cycle_correction)
        self.seed = seed

        # Parameters (init early so discount rates can register into it)
        self.params: Dict[str, _Param] = {}

        # Discount rates
        self._register_discount_rates(
            dr_cost, dr_qaly, self.discount_convention
        )

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

        # State types
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
            self._alive_states = set(
                i for i, s in enumerate(self.states)
                if state_type.get(s, "alive") == "alive"
            )
        else:
            self._alive_states = set(range(self.n_states - 1))

        # Absorbing states (dead)
        self._absorbing = set(range(self.n_states)) - self._alive_states


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



    # =========================================================================
    # Patient Population
    # =========================================================================

    def set_population(self, profile: PatientProfile) -> "IndividualStateTransitionModel":
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

    def set_transitions(self, strategy: str, transitions) -> "IndividualStateTransitionModel":
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
    ) -> "IndividualStateTransitionModel":
        """Define a cost category (same interface as MarkovModel).

        The ``values`` can also accept patient attributes:
        - ``callable(params, cycle, attrs) -> {state: cost}``
        """
        if method not in {"wlos", "starting"}:
            raise ValueError(
                f"Unknown cost method {method!r}; expected 'wlos' or 'starting'."
            )
        if method == "starting" and (first_cycle_only or apply_cycles is not None):
            raise ValueError(
                "method='starting' is a one-off charge at time zero, so it "
                "cannot be combined with first_cycle_only or apply_cycles."
            )
        if apply_cycles is not None:
            resolved_cycles = []
            for cycle in apply_cycles:
                if isinstance(cycle, bool) or not isinstance(cycle, (int, np.integer)):
                    raise TypeError(
                        f"apply_cycles must contain integers, got {cycle!r}"
                    )
                if not 0 <= cycle < self.n_cycles:
                    raise ValueError(
                        f"apply_cycles entry {cycle} is outside the model's "
                        f"interval range 0..{self.n_cycles - 1}"
                    )
                resolved_cycles.append(int(cycle))
            apply_cycles = tuple(resolved_cycles)
        if not callable(values):
            self._validate_state_mapping(values, "cost values")

        self._costs[category] = _CostDef(
            name=category,
            values=values,
            first_cycle_only=first_cycle_only,
            apply_cycles=apply_cycles,
            method=method,
        )
        return self

    def set_utility(self, values: Any) -> "IndividualStateTransitionModel":
        """Define utility weights (same interface as MarkovModel).

        The ``values`` can also accept patient attributes:
        - ``callable(params, cycle, attrs) -> {state: utility}``
        """
        if not callable(values):
            self._validate_state_mapping(values, "utility values")
        self._utility = values
        return self

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_state_enter(self, state: str, handler: Callable) -> "IndividualStateTransitionModel":
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
        if state not in self.states:
            raise ValueError(
                f"Unknown state {state!r}; available states are {self.states!r}"
            )
        if not callable(handler):
            raise TypeError("handler must be callable")
        if state not in self._on_enter:
            self._on_enter[state] = []
        self._on_enter[state].append(handler)
        return self

    def on_state_exit(self, state: str, handler: Callable) -> "IndividualStateTransitionModel":
        """Register a handler called when a patient leaves a state."""
        if state not in self.states:
            raise ValueError(
                f"Unknown state {state!r}; available states are {self.states!r}"
            )
        if not callable(handler):
            raise TypeError("handler must be callable")
        if state not in self._on_exit:
            self._on_exit[state] = []
        self._on_exit[state].append(handler)
        return self

    # =========================================================================
    # Internal Helpers
    # =========================================================================


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
            if (
                len(matrix_data) != self.n_states
                or any(len(row) != self.n_states for row in matrix_data)
            ):
                raise ValueError(
                    "Transition matrix must have shape "
                    f"({self.n_states}, {self.n_states})."
                )
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

        if P.shape != (self.n_states, self.n_states):
            raise ValueError(
                "Transition matrix must have shape "
                f"({self.n_states}, {self.n_states}), got {P.shape}."
            )

        # Repairing an invalid matrix would quietly turn a typo into a
        # different model, so reject it the way the cohort engine does.
        validate_transition_matrix(P)
        return P



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


    def _draw_next_state(self, row: np.ndarray, draws: np.ndarray) -> np.ndarray:
        """Inverse-transform sample destinations from one transition row.

        Spending exactly one uniform per patient per interval is what keeps a
        patient's path identical across strategies.
        """
        picks = np.searchsorted(np.cumsum(row), draws, side="right")
        return np.minimum(picks, self.n_states - 1)

    @staticmethod
    def _interval_reward(values, begin, end, trapezoidal):
        """Per-year value earned over one interval.

        Trapezoidal correction averages the value at the two interval
        endpoints, mirroring the cohort engines' averaging of occupancy.
        """
        if trapezoidal:
            return (values[begin] + values[end]) / 2.0
        return values[begin]

    def _run_state_handlers(self, profile, has_attrs, interval, previous,
                            current, event_cost_by_interval) -> None:
        """Fire exit/enter callbacks and bank any one-off cost they return."""
        for i in np.where(previous != current)[0]:
            attrs = self._get_patient_attrs(profile, i) if has_attrs else {}
            for state_idx, registry in (
                (previous[i], self._on_exit), (current[i], self._on_enter),
            ):
                for handler in registry.get(self.states[state_idx], ()):
                    outcome = handler(i, interval, attrs)
                    if not outcome or "cost" not in outcome:
                        continue
                    amount = float(outcome["cost"])
                    if not np.isfinite(amount):
                        raise ValueError(
                            "State handler returned a non-finite cost at "
                            f"interval {interval} for patient {i}"
                        )
                    event_cost_by_interval[i, interval] += amount

    def _simulate_patients(
        self,
        strategy: str,
        params: dict,
        profile: PatientProfile,
        uniforms: np.ndarray,
    ) -> dict:
        """Simulate all patients for one strategy.

        ``uniforms`` is an ``(n_patients, n_cycles)`` array shared by every
        strategy, so a strategy difference reflects the strategy rather than
        unrelated Monte Carlo noise.

        Rewards accrue over the ``n_cycles`` intervals between the
        ``n_cycles + 1`` observation points, as in the cohort engines.

        Returns
        -------
        dict with keys:
            state_history : (n_patients, n_cycles+1) int array
            cost_history  : (n_patients, n_cycles) float, discounted
            qaly_history  : (n_patients, n_cycles) float, discounted
            ly_history    : (n_patients, n_cycles) float, discounted
            event_costs   : (n_patients,) float, undiscounted lump sums
            trace         : (n_cycles+1, n_states) float, mean occupancy
            time_alive    : (n_patients,) float, undiscounted years alive
        """
        N = profile.n_patients
        T = self.n_cycles

        state_hist = np.full((N, T + 1), -1, dtype=int)
        state_hist[:, 0] = self.initial_state_idx

        # Lump sums banked against the interval whose transition produced
        # them, so they discount at that interval's end.
        event_cost_by_interval = np.zeros((N, T))

        has_attrs = bool(profile.attributes)
        can_batch = not (
            has_attrs
            and self._callable_needs_attrs(self._transitions.get(strategy))
        )

        for interval in range(T):
            previous = state_hist[:, interval]
            current = previous.copy()
            draws = uniforms[:, interval]

            if can_batch:
                P = self._get_transition_matrix(strategy, params, interval, {})
                alive = np.where(np.isin(previous, list(self._alive_states)))[0]
                if len(alive):
                    origins = previous[alive]
                    for s in np.unique(origins):
                        movers = alive[origins == s]
                        current[movers] = self._draw_next_state(
                            P[s], draws[movers]
                        )
            else:
                for i in range(N):
                    if previous[i] not in self._alive_states:
                        continue
                    attrs = self._get_patient_attrs(profile, i)
                    P = self._get_transition_matrix(
                        strategy, params, interval, attrs
                    )
                    current[i] = self._draw_next_state(
                        P[previous[i]], draws[i:i + 1]
                    )[0]

            state_hist[:, interval + 1] = current

            if self._on_enter or self._on_exit:
                self._run_state_handlers(
                    profile, has_attrs, interval, previous, current,
                    event_cost_by_interval,
                )

        # --- Rewards, one row per interval ---
        alive_mask = np.zeros(self.n_states)
        for index in self._alive_states:
            alive_mask[index] = 1.0

        cost_hist = np.zeros((N, T))
        qaly_hist = np.zeros((N, T))
        ly_hist = np.zeros((N, T))
        starting_cost = np.zeros(N)

        needs_per_patient = has_attrs and (
            self._callable_needs_attrs(self._utility)
            or any(
                self._callable_needs_attrs(cost.values)
                for cost in self._costs.values()
            )
        )
        trapezoidal = self._hcc_method == "trapezoidal"

        for interval in range(T):
            begin = state_hist[:, interval]
            end = state_hist[:, interval + 1]

            if not needs_per_patient:
                rate, lump = self._interval_cost_vectors(
                    strategy, params, interval
                )
                utility = self._get_utilities(strategy, params, interval)

                cost_hist[:, interval] = self.cycle_length * self._interval_reward(
                    rate, begin, end, trapezoidal
                )
                qaly_hist[:, interval] = self.cycle_length * self._interval_reward(
                    utility, begin, end, trapezoidal
                )
                ly_hist[:, interval] = self.cycle_length * self._interval_reward(
                    alive_mask, begin, end, trapezoidal
                )
                if interval == 0:
                    starting_cost += lump[begin]
            else:
                for i in range(N):
                    attrs = self._get_patient_attrs(profile, i)
                    rate, lump = self._interval_cost_vectors(
                        strategy, params, interval, attrs
                    )
                    utility = self._get_utilities(
                        strategy, params, interval, attrs
                    )
                    cost_hist[i, interval] = self.cycle_length * self._interval_reward(
                        rate, begin[i], end[i], trapezoidal
                    )
                    qaly_hist[i, interval] = self.cycle_length * self._interval_reward(
                        utility, begin[i], end[i], trapezoidal
                    )
                    ly_hist[i, interval] = self.cycle_length * self._interval_reward(
                        alive_mask, begin[i], end[i], trapezoidal
                    )
                    if interval == 0:
                        starting_cost[i] += lump[begin[i]]

        # --- Discounting: flows at interval midpoints, events at their end ---
        intervals = np.arange(T, dtype=float)
        flow_cost_df = discount_factor(
            intervals + 0.5, self.dr_cost, self.cycle_length,
            self.discount_convention,
        )
        flow_qaly_df = discount_factor(
            intervals + 0.5, self.dr_qaly, self.cycle_length,
            self.discount_convention,
        )
        event_cost_df = discount_factor(
            intervals + 1.0, self.dr_cost, self.cycle_length,
            self.discount_convention,
        )

        cost_hist_disc = cost_hist * flow_cost_df
        qaly_hist_disc = qaly_hist * flow_qaly_df
        ly_hist_disc = ly_hist * flow_qaly_df

        # A starting cost is paid at time zero, so it is never discounted.
        total_cost_per_patient = (
            cost_hist_disc.sum(axis=1)
            + starting_cost
            + (event_cost_by_interval * event_cost_df).sum(axis=1)
        )
        total_qaly_per_patient = qaly_hist_disc.sum(axis=1)
        total_ly_per_patient = ly_hist_disc.sum(axis=1)

        trace = np.zeros((T + 1, self.n_states))
        for s in range(self.n_states):
            trace[:, s] = (state_hist == s).mean(axis=0)

        # Undiscounted years alive, on the same footing as the LY accrual.
        time_alive = ly_hist.sum(axis=1)

        return {
            'state_history': state_hist,
            'cost_history': cost_hist_disc,
            'qaly_history': qaly_hist_disc,
            'ly_history': ly_hist_disc,
            'event_costs': event_cost_by_interval.sum(axis=1),
            'total_cost': total_cost_per_patient,
            'total_qalys': total_qaly_per_patient,
            'total_lys': total_ly_per_patient,
            'trace': trace,
            'time_alive': time_alive,
            'mean_cost': float(total_cost_per_patient.mean()),
            'mean_qalys': float(total_qaly_per_patient.mean()),
            'mean_lys': float(total_ly_per_patient.mean()),
        }

    def _interval_cost_vectors(self, strategy, params, interval, attrs=None):
        """Split this interval's costs into a per-year rate and a lump sum."""
        rate = np.zeros(self.n_states)
        lump = np.zeros(self.n_states)
        for category, definition in self._costs.items():
            values = self._get_state_costs(
                category, strategy, params, interval, attrs
            )
            if definition.method == "starting":
                lump += values
            else:
                rate += values
        return rate, lump
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
        uniforms = rng.random((prof.n_patients, self.n_cycles))

        results = {}
        for strat in self.strategy_names:
            if verbose:
                print(f"  Simulating {prof.n_patients} patients: {strat}...", end=" ")
            results[strat] = self._simulate_patients(strat, params, prof, uniforms)
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
                    p[name] = float(sample_distribution(param.dist, 1, rng)[0])
            sampled_params.append(p)

            # Shared across strategies within this draw, redrawn between draws.
            uniforms = rng.random((prof.n_patients, self.n_cycles))

            iter_results = {}
            with self._attr_param_override(p):
                for strat in self.strategy_names:
                    iter_results[strat] = self._simulate_patients(
                        strat, p, prof, uniforms
                    )

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

        uniforms = np.random.default_rng(s).random((prof.n_patients, self.n_cycles))

        def aggregate(sim):
            return {
                'total_costs': {'total': sim['mean_cost']},
                'total_qalys': sim['mean_qalys'],
                'total_lys': sim['mean_lys'],
            }

        base_result = {}
        for strat in self.strategy_names:
            sim = self._simulate_patients(strat, base_params, prof, uniforms)
            base_result[strat] = aggregate(sim)

        owsa_data = []
        for param_name in params:
            p = self.params[param_name]
            low = p.low if p.low is not None else p.base * 0.8
            high = p.high if p.high is not None else p.base * 1.2

            is_attr = param_name in self._ATTR_PARAMS

            for bound, val in [('low', low), ('high', high)]:
                test_params = base_params.copy()
                test_params[param_name] = val

                override = (
                    self._attr_param_override({param_name: val}) if is_attr
                    else nullcontext()
                )
                with override:
                    result = {}
                    for strat in self.strategy_names:
                        sim = self._simulate_patients(
                            strat, test_params, prof, uniforms
                        )
                        result[strat] = aggregate(sim)

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
            f"IndividualStateTransitionModel (Individual-Level Simulation)",
            f"  States ({self.n_states}): {self.states}",
            f"  Strategies ({self.n_strategies}): {self.strategy_names}",
            f"  Cycles: {self.n_cycles} × {self.cycle_length} year(s)",
            f"  Patients: {self.n_patients}",
            f"  Discount rates: cost={self.dr_cost:.1%}, QALY={self.dr_qaly:.1%}",
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
            f"IndividualStateTransitionModel(states={self.states}, "
            f"strategies={self.strategy_names}, "
            f"n_cycles={self.n_cycles}, n_patients={self.n_patients})"
        )


# Concise public alias retained for compatibility and everyday use.
MicroSimModel = IndividualStateTransitionModel
