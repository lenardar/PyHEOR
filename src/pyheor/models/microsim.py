"""Individual state-transition simulation with explicit cycles and per-cycle rewards. See README.md."""

import numpy as np
import pandas as pd
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .common import Param as _Param, StateMappingModel
from ..time import Cycle
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
    """Explicit-cycle state-transition model. See README.md."""

    def __init__(
        self,
        states: List[str],
        strategies: Union[List[str], Dict[str, str]],
        n_cycles: int,
        cycle: Cycle,
        n_patients: int = 1000,
        dr_cost: Union[float, "_Param"] = 0.0,
        dr_qaly: Union[float, "_Param"] = 0.0,
        method: str = "life-table",
        initial_state: Union[str, int] = 0,
        state_type: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
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
        if not isinstance(cycle, Cycle):
            raise TypeError("cycle must be an explicit Cycle(length, unit)")
        self.n_cycles = int(n_cycles)
        self.n_patients = int(n_patients)
        self.cycle = cycle
        self.cycle_length = cycle.years
        self.discount_convention = "discrete"
        self.method = normalize_hcc(method)
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

        self._init_rewards()


        # Patient profile (overrides n_patients if set)
        self._profile: Optional[PatientProfile] = None


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


    # =========================================================================
    # Event Handlers
    # =========================================================================


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

    def _simulate_patients(self, strategy, params, profile, uniforms):
        N, T = profile.n_patients, self.n_cycles
        state_hist = np.full((N, T + 1), self.initial_state_idx, dtype=int)
        has_attrs = bool(profile.attributes)
        for i in range(T):
            previous = state_hist[:, i]
            current = previous.copy()
            if not (has_attrs and self._callable_needs_attrs(self._transitions.get(strategy))):
                P = self._get_transition_matrix(strategy, params, i + 1, {})
                for source in self._alive_states:
                    movers = np.where(previous == source)[0]
                    if len(movers):
                        current[movers] = self._draw_next_state(P[source], uniforms[movers, i])
            else:
                for patient in range(N):
                    if previous[patient] in self._alive_states:
                        attrs = self._get_patient_attrs(profile, patient)
                        P = self._get_transition_matrix(strategy, params, i + 1, attrs)
                        current[patient] = self._draw_next_state(P[previous[patient]], uniforms[patient, i:i+1])[0]
            state_hist[:, i + 1] = current
        if not has_attrs and not any(self._custom_rewards.values()):
            cost_hist, qaly_hist, ly_hist, time_alive, by_cost, by_qaly, raw_cost, raw_qaly = self._batch_patient_rewards(strategy, params, state_hist)
        else:
            cost_hist = np.zeros((N, T))
            qaly_hist = np.zeros((N, T))
            ly_hist = np.zeros((N, T))
            time_alive = np.zeros(N)
            by_cost, by_qaly, raw_cost, raw_qaly = {}, {}, {}, {}
            for patient in range(N):
                attrs = self._get_patient_attrs(profile, patient)
                history = state_hist[patient]
                trace = np.eye(self.n_states)[history]
                flows = np.zeros((T, self.n_states, self.n_states))
                flows[np.arange(T), history[:-1], history[1:]] = 1
                matrices = np.array([self._get_transition_matrix(strategy, params, k, attrs) for k in range(1, T + 1)])
                result = self._cycle_rewards(strategy, params, trace, flows, matrices, attrs=attrs, patient_index=patient)
                cost_hist[patient] = sum(result['discounted_costs'].values(), np.zeros(T))
                qaly_hist[patient] = result['discounted_qalys']
                ly_hist[patient] = result['discounted_lys']
                time_alive[patient] = result['lys_hcc'].sum()
                for output, key in ((raw_cost, 'undiscounted_costs'), (raw_qaly, 'undiscounted_qalys')):
                    for category, values in result[key].items():
                        output.setdefault(category, np.zeros((N, T)))[patient] = values
                for category, values in result['discounted_costs'].items():
                    by_cost.setdefault(category, np.zeros((N, T)))[patient] = values
                for category, values in result['qaly_components'].items():
                    by_qaly.setdefault(category, np.zeros((N, T)))[patient] = values
        trace = np.array([(state_hist == s).mean(axis=0) for s in range(self.n_states)]).T
        total_cost, total_qaly, total_ly = cost_hist.sum(axis=1), qaly_hist.sum(axis=1), ly_hist.sum(axis=1)
        return dict(state_history=state_hist, cost_history=cost_hist, qaly_history=qaly_hist,
                    ly_history=ly_hist, total_cost=total_cost, total_qalys=total_qaly,
                    total_lys=total_ly, trace=trace, time_alive=time_alive,
                    costs_by_category=by_cost, qaly_components=by_qaly,
                    undiscounted_costs=raw_cost, undiscounted_qalys=raw_qaly,
                    mean_cost=float(total_cost.mean()), mean_qalys=float(total_qaly.mean()),
                    mean_lys=float(total_ly.mean()))


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
                    self._attr_param_override(test_params) if is_attr
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

    def info(self):
        return (f"{type(self).__name__}: {self.n_states} states, {self.n_strategies} strategies\n"
                f"  Cycles: {self.n_cycles} × {self.cycle.length} {self.cycle.unit}\n"
                f"  Method: {self.method}; discount rates per cycle: {self._discount_rate('dr_cost', self._get_base_params())}, {self._discount_rate('dr_qaly', self._get_base_params())}\n"
                f"  Cost components: {list(self._costs)}; QALY components: {list(self._qalys)}")


    def __repr__(self):
        return (
            f"IndividualStateTransitionModel(states={self.states}, "
            f"strategies={self.strategy_names}, "
            f"n_cycles={self.n_cycles}, n_patients={self.n_patients})"
        )


# Concise public alias retained for compatibility and everyday use.
MicroSimModel = IndividualStateTransitionModel
