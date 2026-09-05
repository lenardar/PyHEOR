"""
DiscreteEventSimulationModel — Discrete Event Simulation for health economic evaluation.
====================================================================

Unlike the cohort MarkovModel (fixed cycle, proportion-based) or the
MicroSimModel (fixed cycle, individual-based), a DES operates in
**continuous time**: each patient's event times are sampled from
survival distributions, and the simulation clock jumps from event to
event.

Key advantages over cycle-based models
---------------------------------------
- No cycle-length artefact (arbitrary precision in time)
- Natural fit for time-to-event data (directly use parametric distributions)
- Easy to model competing risks, recurrent events, and patient heterogeneity
- Straightforward integration with externally supplied HR samples

Architecture
------------
1. **Events** are defined per strategy.  Each event has a *source state*,
   a *destination state*, and a *time-to-event distribution* (``SurvivalDistribution``
   or callable returning one).
2. For each patient in each state, all eligible events race (competing risks).
   The earliest event fires; the patient transitions and the process repeats
   from the new state.
3. **Costs** accrue by state (rate × time in state) or as one-time amounts on
   state entry.
4. **Utilities** accrue by state (weight × time in state).

Typical workflow
----------------
>>> model = DiscreteEventSimulationModel(
...     states=["PFS", "Progressed", "Dead"],
...     strategies=["SOC", "Treatment"],
...     time_horizon=40,
... )
>>> model.add_param("hr_pfs", base=0.75, dist=ph.LogNormal(mean=-0.29, sd=0.15))
>>> model.set_event("SOC", "PFS", "Progressed", ph.Weibull(shape=1.2, scale=5))
>>> model.set_event("SOC", "PFS", "Dead",       ph.Weibull(shape=1.0, scale=20))
>>> model.set_event("SOC", "Progressed", "Dead", ph.Weibull(shape=1.5, scale=3))
>>> # Treatment: HR applied to PFS->Progressed
>>> model.set_event("Treatment", "PFS", "Progressed",
...     lambda p: ph.ProportionalHazards(ph.Weibull(shape=1.2, scale=5), p["hr_pfs"]))
>>> model.set_event("Treatment", "PFS", "Dead",        ph.Weibull(shape=1.0, scale=20))
>>> model.set_event("Treatment", "Progressed", "Dead",  ph.Weibull(shape=1.5, scale=3))
>>> model.set_state_cost("drug", {"PFS": 5000, "Progressed": 2000, "Dead": 0})
>>> model.set_entry_cost("surgery", "Progressed", 50000)
>>> model.set_utility({"PFS": 0.85, "Progressed": 0.5, "Dead": 0})
>>> result = model.run(n_patients=5000, seed=42)
>>> print(result.summary())

References
----------
- Karnon J, et al. (2012). Modeling using discrete event simulation:
  a report of the ISPOR-SMDM Modeling Good Research Practices Task Force.
  Medical Decision Making, 32(5), 701-711.
- Caro JJ, Möller J. (2016). Advantages and disadvantages of discrete-event
  simulation for health economic analyses. Expert Rev Pharmacoecon Outcomes Res.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..distributions import Distribution
from .markov import Param
from ..survival import SurvivalDistribution
from ..utils import resolve_value, discount_factor


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class _EventDef:
    """Internal definition of a single event (state transition)."""
    from_state: str
    to_state: str
    from_idx: int
    to_idx: int
    distribution: Any  # SurvivalDistribution, callable, or None
    clock: Optional[str] = None  # None inherits the model-level default


@dataclass
class _StateCostDef:
    """A per-unit-time cost incurred while in a state."""
    category: str
    values: Any  # dict or callable


@dataclass
class _EntryCostDef:
    """A one-time cost triggered on entering a state."""
    category: str
    state: str
    state_idx: int
    value: Any  # float, str, dict, callable


# =============================================================================
# DiscreteEventSimulationModel
# =============================================================================

class DiscreteEventSimulationModel:
    """Discrete Event Simulation model for health economic evaluation.

    Parameters
    ----------
    states : list of str
        Health state names.
    strategies : list of str or dict
        Treatment strategies. Dict maps internal name → display label.
    time_horizon : float
        Maximum simulation time in years.
    dr_cost : float or Param
        Annual discount rate for costs. Default: 0 (no discounting).
        Pass a ``Param`` to enable sensitivity analysis.
    dr_qaly : float or Param
        Annual discount rate for QALYs. Default: 0 (no discounting).
        Pass a ``Param`` to enable sensitivity analysis.
    state_type : dict, optional
        Map state names to "alive" or "dead" (default: last state is dead).
    clock : {"reset", "forward"}
        Event-time clock. ``"reset"`` (default) samples time from entry into
        the current state. ``"forward"`` conditions each event on the
        absolute study time using its cumulative hazard.
    discount_convention : {"discrete", "continuous"}
        How ``dr_cost`` and ``dr_qaly`` are defined. ``"discrete"`` (default)
        uses annual-effective rates and ``(1 + rate) ** -time``. ``"continuous"``
        uses continuously compounded rates and ``exp(-rate * time)``.

    Examples
    --------
    >>> model = DiscreteEventSimulationModel(
    ...     states=["PFS", "Progressed", "Dead"],
    ...     strategies=["SOC", "Treatment"],
    ...     time_horizon=40,
    ... )
    """

    def __init__(
        self,
        states: List[str],
        strategies: Union[List[str], Dict[str, str]],
        time_horizon: float = 40.0,
        dr_cost: Union[float, "Param"] = 0.0,
        dr_qaly: Union[float, "Param"] = 0.0,
        state_type: Optional[Dict[str, str]] = None,
        clock: str = "reset",
        discount_convention: str = "discrete",
    ):
        self.states = list(states)
        if not self.states:
            raise ValueError("states must contain at least one state")
        if len(set(self.states)) != len(self.states):
            raise ValueError(f"State names must be unique, got {self.states!r}")
        self.n_states = len(self.states)
        self.time_horizon = float(time_horizon)
        if self.time_horizon <= 0 or not np.isfinite(self.time_horizon):
            raise ValueError("time_horizon must be a finite positive number")
        if clock not in {"reset", "forward"}:
            raise ValueError("clock must be 'reset' or 'forward'")
        if discount_convention not in {"discrete", "continuous"}:
            raise ValueError(
                "discount_convention must be 'discrete' or 'continuous'"
            )
        self.clock = clock
        self.discount_convention = discount_convention

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
        self._validate_discount_rate(self.dr_cost, "dr_cost")
        self._validate_discount_rate(self.dr_qaly, "dr_qaly")

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
        self._absorbing = set(range(self.n_states)) - self._alive_states


        # Events: strategy -> list[_EventDef]
        self._events: Dict[str, List[_EventDef]] = {
            s: [] for s in self.strategy_names
        }

        # Costs
        self._state_costs: List[_StateCostDef] = []
        self._entry_costs: List[_EntryCostDef] = []

        # Utility
        self._utility: Any = None

        # Event handlers
        self._on_enter: Dict[str, List[Callable]] = {}
        self._on_event: Dict[Tuple[str, str], List[Callable]] = {}

    # =====================================================================
    # Parameters
    # =====================================================================

    def add_param(
        self, name: str, base: float, dist=None, label=None,
        low=None, high=None,
    ) -> "DiscreteEventSimulationModel":
        """Add a model parameter (same API as MarkovModel)."""
        self.params[name] = Param(
            base=base, dist=dist,
            label=label or name,
            low=low, high=high,
        )
        return self

    def add_params(self, params_dict):
        """Add multiple parameters at once."""
        for name, param in params_dict.items():
            if isinstance(param, Param):
                if not param.label:
                    param.label = name
                self.params[name] = param
            elif isinstance(param, (int, float)):
                self.params[name] = Param(base=float(param), label=name)
            else:
                raise TypeError(f"Parameter '{name}': expected Param or numeric")
        return self

    def _get_base_params(self) -> Dict[str, float]:
        """Get base-case parameter values."""
        return {name: p.base for name, p in self.params.items()}

    # Parameters that live as model attributes rather than in the params dict.
    # Discount rates are read off self during simulation, so a value sampled
    # into the params dict has no effect unless written to the attribute too.
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

    # =====================================================================
    # Events (state transitions)
    # =====================================================================

    def set_event(
        self,
        strategy: str,
        from_state: str,
        to_state: str,
        distribution: Any,
        clock: Optional[str] = None,
    ) -> "DiscreteEventSimulationModel":
        """Define a transition event with a time-to-event distribution.

        Multiple events from the same source state are treated as
        **competing risks**: the earliest event fires.

        Parameters
        ----------
        strategy : str
            Strategy name.
        from_state : str
            Source state.
        to_state : str
            Destination state.
        distribution : SurvivalDistribution or callable
            Time-to-event distribution. Can be:

            - A ``SurvivalDistribution`` — fixed distribution.
            - ``callable(params) -> SurvivalDistribution`` — parameter-dependent.
            - ``callable(params, attrs) -> SurvivalDistribution`` — also
              depends on patient attributes.
        clock : {"reset", "forward"}, optional
            Override the model's event-time clock for this event. Use
            ``"reset"`` for time since entering ``from_state`` and
            ``"forward"`` for absolute study time.

        Returns
        -------
        DiscreteEventSimulationModel
            Self, for method chaining.

        Examples
        --------
        Fixed distribution:

        >>> model.set_event("SOC", "PFS", "Progressed",
        ...                 ph.Weibull(shape=1.2, scale=5))

        Parameter-dependent (e.g. sampled HR):

        >>> model.set_event("Treatment", "PFS", "Progressed",
        ...     lambda p: ph.ProportionalHazards(
        ...         ph.Weibull(shape=1.2, scale=5), p["hr_pfs"]))

        Patient-attribute-dependent:

        >>> model.set_event("SOC", "PFS", "Dead",
        ...     lambda p, a: ph.Weibull(shape=1.0, scale=20 - 0.1 * a["age"]))
        """
        if strategy not in self.strategy_names:
            raise ValueError(f"Unknown strategy '{strategy}'")
        if from_state not in self.states:
            raise ValueError(f"Unknown state '{from_state}'")
        if to_state not in self.states:
            raise ValueError(f"Unknown state '{to_state}'")
        if clock not in {None, "reset", "forward"}:
            raise ValueError("clock must be 'reset', 'forward', or None")

        ev = _EventDef(
            from_state=from_state,
            to_state=to_state,
            from_idx=self.states.index(from_state),
            to_idx=self.states.index(to_state),
            distribution=distribution,
            clock=clock,
        )
        self._events[strategy].append(ev)
        return self

    def set_events_from(
        self,
        strategy: str,
        from_state: str,
        events: Dict[str, Any],
    ) -> "DiscreteEventSimulationModel":
        """Set multiple events from the same source state.

        Parameters
        ----------
        strategy : str
            Strategy name.
        from_state : str
            Source state.
        events : dict
            Maps destination state → distribution.

        Examples
        --------
        >>> model.set_events_from("SOC", "PFS", {
        ...     "Progressed": ph.Weibull(shape=1.2, scale=5),
        ...     "Dead": ph.Weibull(shape=1.0, scale=20),
        ... })
        """
        for to_state, dist in events.items():
            self.set_event(strategy, from_state, to_state, dist)
        return self

    # =====================================================================
    # Costs
    # =====================================================================

    def set_state_cost(self, category: str, values: Any) -> "DiscreteEventSimulationModel":
        """Define per-unit-time costs incurred while in a state.

        These are continuous-time *rate* costs: cost per year in state.
        The engine integrates ``cost_rate × time_in_state`` with discounting.

        Parameters
        ----------
        category : str
            Cost category name.
        values : dict or callable
            - ``{state: value}`` — Same for all strategies.
            - ``{strategy: {state: value}}`` — Strategy-specific.
            Each value can be float, str (param ref), or callable.

        Examples
        --------
        >>> model.set_state_cost("drug", {
        ...     "SOC": {"PFS": 500, "Progressed": 200, "Dead": 0},
        ...     "Treatment": {"PFS": 3000, "Progressed": 200, "Dead": 0},
        ... })
        """
        self._state_costs.append(_StateCostDef(category=category, values=values))
        return self

    def set_entry_cost(
        self, category: str, state: str, value: Any,
    ) -> "DiscreteEventSimulationModel":
        """Define a one-time cost triggered on entering a state.

        Parameters
        ----------
        category : str
            Cost category name.
        state : str
            The state whose entry triggers the cost.
        value : float, str, dict, or callable
            - ``float`` — Fixed cost.
            - ``str`` — Parameter reference.
            - ``{strategy: value}`` — Strategy-specific.
            - ``callable(params) -> float`` — Parameter-dependent.

        Examples
        --------
        >>> model.set_entry_cost("surgery", "Progressed", 50000)
        >>> model.set_entry_cost("rescue", "Progressed", {
        ...     "SOC": 30000, "Treatment": 15000})
        """
        if state not in self.states:
            raise ValueError(f"Unknown state '{state}'")
        self._entry_costs.append(_EntryCostDef(
            category=category,
            state=state,
            state_idx=self.states.index(state),
            value=value,
        ))
        return self

    # =====================================================================
    # Utility
    # =====================================================================

    def set_utility(self, values: Any) -> "DiscreteEventSimulationModel":
        """Define utility weights per state.

        Parameters
        ----------
        values : dict or callable
            - ``{state: value}`` — Same for all strategies.
            - ``{strategy: {state: value}}`` — Strategy-specific.

        Examples
        --------
        >>> model.set_utility({"PFS": 0.85, "Progressed": 0.5, "Dead": 0})
        """
        self._utility = values
        return self

    # =====================================================================
    # Event handlers (advanced)
    # =====================================================================

    def on_state_enter(
        self, state: str, handler: Callable,
    ) -> "DiscreteEventSimulationModel":
        """Register a handler called when a patient enters a state.

        Parameters
        ----------
        state : str
            State name.
        handler : callable
            ``handler(patient_idx, time, attrs)``. Return ``{"cost": amount}``
            to add a one-time cost at the state-entry time.
        """
        if state not in self.states:
            raise ValueError(f"Unknown state '{state}'")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._on_enter.setdefault(state, []).append(handler)
        return self

    # =====================================================================
    # Resolve helpers
    # =====================================================================

    @staticmethod
    def _validate_discount_rate(rate: float, name: str = "discount rate") -> float:
        """Validate a DES discount rate and return it as a float."""
        try:
            value = float(rate)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a finite non-negative number") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number, got {rate!r}")
        return value

    @staticmethod
    def _resolve_param_ref(value: Any, params: dict, context: str) -> Any:
        """Resolve a string parameter reference without silently defaulting."""
        if isinstance(value, str):
            if value not in params:
                raise KeyError(
                    f"Parameter '{value}' not found while resolving {context}. "
                    f"Available: {list(params.keys())}"
                )
            return params[value]
        return value

    @staticmethod
    def _validate_mapping_keys(mapping: dict, allowed: set, context: str):
        unknown = set(mapping) - allowed
        if unknown:
            raise ValueError(
                f"{context} contains unknown keys: {sorted(unknown, key=str)!r}"
            )

    def _validate_runtime_discount_rates(self, params: dict):
        """Validate fixed and PSA-sampled discount rates before simulation."""
        self._validate_discount_rate(params.get("dr_cost", self.dr_cost), "dr_cost")
        self._validate_discount_rate(params.get("dr_qaly", self.dr_qaly), "dr_qaly")

    def _resolve_cost_rate(
        self, cost_def: _StateCostDef, strategy: str, state_idx: int,
        params: dict,
    ) -> float:
        """Resolve per-unit-time cost."""
        vals = cost_def.values
        state = self.states[state_idx]

        if callable(vals):
            vals = vals(params)

        # Strategy-specific outer layer
        if isinstance(vals, dict):
            self._validate_mapping_keys(
                vals, set(self.strategy_names) | set(self.states),
                f"State cost '{cost_def.category}'",
            )
            if strategy in vals:
                inner = vals[strategy]
                if isinstance(inner, dict):
                    self._validate_mapping_keys(
                        inner, set(self.states),
                        f"State cost '{cost_def.category}' for strategy '{strategy}'",
                    )
                    v = inner.get(state, 0)
                else:
                    v = inner  # single value for all states? unlikely
            elif state in vals:
                v = vals[state]
            else:
                v = 0
        else:
            v = vals

        v = self._resolve_param_ref(v, params, f"state cost '{cost_def.category}'")
        if callable(v):
            v = v(params)
        try:
            value = float(v)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"State cost '{cost_def.category}' resolved to a non-numeric value"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"State cost '{cost_def.category}' resolved to a non-finite value"
            )
        return value

    def _resolve_entry_cost(
        self, ec: _EntryCostDef, strategy: str, params: dict,
    ) -> float:
        """Resolve one-time entry cost."""
        val = ec.value
        if isinstance(val, dict):
            self._validate_mapping_keys(
                val, set(self.strategy_names), f"Entry cost '{ec.category}'"
            )
            val = val.get(strategy, 0)
        val = self._resolve_param_ref(val, params, f"entry cost '{ec.category}'")
        if callable(val):
            val = val(params)
        try:
            value = float(val)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Entry cost '{ec.category}' resolved to a non-numeric value"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"Entry cost '{ec.category}' resolved to a non-finite value"
            )
        return value

    def _resolve_utility(
        self, strategy: str, state_idx: int, params: dict,
    ) -> float:
        """Resolve utility weight."""
        vals = self._utility
        if vals is None:
            return 1.0
        state = self.states[state_idx]

        if callable(vals):
            vals = vals(params)

        if isinstance(vals, dict):
            self._validate_mapping_keys(
                vals, set(self.strategy_names) | set(self.states), "Utility mapping"
            )
            if strategy in vals:
                inner = vals[strategy]
                if isinstance(inner, dict):
                    self._validate_mapping_keys(
                        inner, set(self.states),
                        f"Utility mapping for strategy '{strategy}'",
                    )
                    v = inner.get(state, 0)
                else:
                    v = inner
            elif state in vals:
                v = vals[state]
            else:
                v = 0
        else:
            v = vals

        v = self._resolve_param_ref(v, params, f"utility for state '{state}'")
        if callable(v):
            v = v(params)
        try:
            value = float(v)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Utility for state '{state}' resolved to a non-numeric value"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"Utility for state '{state}' resolved to a non-finite value")
        return value

    def _resolve_distribution(
        self, ev: _EventDef, params: dict, attrs: Optional[dict] = None,
    ) -> SurvivalDistribution:
        """Resolve event distribution (may be callable)."""
        d = ev.distribution
        if isinstance(d, SurvivalDistribution):
            return d
        if callable(d):
            import inspect
            sig = inspect.signature(d)
            n_args = len([
                p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
            ])
            if n_args >= 2 and attrs is not None:
                return d(params, attrs)
            return d(params)
        raise TypeError(f"Event distribution must be SurvivalDistribution or callable, got {type(d)}")

    # =====================================================================
    # Discounting helpers (continuous time)
    # =====================================================================

    @classmethod
    def _discount_lump_sum(
        cls,
        amount: float,
        time: float,
        rate: float,
        convention: str = "discrete",
    ) -> float:
        """Discount a lump-sum amount at a continuous-time event."""
        if not np.isfinite(amount) or not np.isfinite(time):
            raise ValueError("Lump-sum amount and time must be finite")
        cls._validate_discount_rate(rate)
        factor = discount_factor(time, rate, convention=convention)
        return float(amount * factor)

    @classmethod
    def _discount_continuous(
        cls,
        rate_per_year: float,
        t_start: float,
        t_end: float,
        dr: float,
        convention: str = "discrete",
    ) -> float:
        """Discounted integral of a constant rate from t_start to t_end.

        For ``discrete`` convention this integrates ``rate / (1+dr)^t``.
        For ``continuous`` convention it integrates ``rate * exp(-dr*t)``.
        """
        if not np.isfinite(rate_per_year) or not np.isfinite(t_start) or not np.isfinite(t_end):
            raise ValueError("Continuous accrual rate and times must be finite")
        cls._validate_discount_rate(dr)
        if convention not in {"discrete", "continuous"}:
            raise ValueError(
                "discount_convention must be 'discrete' or 'continuous'"
            )
        if t_end <= t_start:
            return 0.0
        if dr <= 0:
            return rate_per_year * (t_end - t_start)
        if convention == "discrete":
            log_dr = np.log1p(dr)
            return float(rate_per_year * (
                np.exp(-log_dr * t_start) - np.exp(-log_dr * t_end)
            ) / log_dr)
        return float(rate_per_year * (
            np.exp(-dr * t_start) - np.exp(-dr * t_end)
        ) / dr)

    # =====================================================================
    # Simulation engine
    # =====================================================================

    def _sample_tte(self, dist: SurvivalDistribution, rng=None) -> float:
        """Sample a relative time-to-event from a survival distribution."""
        u = (rng if rng is not None else np.random).uniform()
        tte = dist.quantile(u)
        if not np.isfinite(tte) and not np.isinf(tte):
            raise ValueError(f"Event distribution returned non-finite TTE: {tte!r}")
        if tte < 0:
            raise ValueError(f"Event distribution returned negative TTE: {tte!r}")
        return float(tte)

    def _sample_forward_tte(
        self, dist: SurvivalDistribution, current_time: float, rng=None,
    ) -> float:
        """Sample a residual TTE under a clock-forward cumulative hazard."""
        from scipy.optimize import brentq

        u = (rng if rng is not None else np.random).uniform()
        if not 0 < u < 1:
            return float("inf")
        h0 = float(dist.cumulative_hazard(current_time))
        if not np.isfinite(h0) or h0 < 0:
            raise ValueError("Event distribution returned an invalid cumulative hazard")
        target = h0 - np.log(u)
        horizon_h = float(dist.cumulative_hazard(self.time_horizon))
        if np.isnan(horizon_h) or horizon_h < target:
            return float("inf")
        if horizon_h == h0:
            return float("inf")
        event_time = brentq(
            lambda t: float(dist.cumulative_hazard(t)) - target,
            current_time, self.time_horizon,
        )
        tte = event_time - current_time
        if tte < 0 or not np.isfinite(tte):
            raise ValueError(f"Event distribution returned invalid TTE: {tte!r}")
        return float(tte)

    def _simulate_patient(
        self,
        strategy: str,
        params: dict,
        attrs: Optional[dict] = None,
        patient_idx: Optional[int] = None,
        rng=None,
    ) -> dict:
        """Simulate a single patient through the event-driven process.

        Returns
        -------
        dict with keys:
            total_cost : float
            total_qalys : float
            total_lys : float
            costs_by_cat : dict[str, float]
            event_log : list of (time, from_state, to_state)
            time_in_state : dict[str, float]
        """
        self._validate_discount_rate(self.dr_cost, "dr_cost")
        self._validate_discount_rate(self.dr_qaly, "dr_qaly")
        current_state = 0  # start in first state
        current_time = 0.0
        event_log = []
        time_in_state = {s: 0.0 for s in self.states}
        costs_by_cat: Dict[str, float] = {}
        total_qalys = 0.0
        total_lys = 0.0

        # Initial entry costs
        for ec in self._entry_costs:
            if ec.state_idx == current_state:
                c = self._resolve_entry_cost(ec, strategy, params)
                cat = ec.category
                costs_by_cat[cat] = costs_by_cat.get(cat, 0) + c  # time=0, no discount

        while current_time < self.time_horizon and current_state not in self._absorbing:
            # Collect competing events from current state
            eligible = [
                ev for ev in self._events[strategy]
                if ev.from_idx == current_state
            ]

            if not eligible:
                # No events defined: patient stays until time horizon
                remaining = self.time_horizon - current_time
                lys, qalys, _ = self._sojourn_outcomes(
                    strategy, params, current_state,
                    current_time, current_time + remaining,
                )
                total_lys += lys
                total_qalys += qalys
                self._accrue_costs(
                    strategy, params, current_state,
                    current_time, current_time + remaining,
                    costs_by_cat,
                )
                time_in_state[self.states[current_state]] += remaining
                current_time = self.time_horizon
                break

            # Sample time-to-event for each competing risk
            min_time = float('inf')
            winning_event = None

            for ev in eligible:
                dist = self._resolve_distribution(ev, params, attrs)
                clock = ev.clock or self.clock
                if clock == "forward":
                    tte = (self._sample_forward_tte(dist, current_time, rng)
                           if rng is not None
                           else self._sample_forward_tte(dist, current_time))
                else:
                    tte = (self._sample_tte(dist, rng)
                           if rng is not None
                           else self._sample_tte(dist))
                if tte < min_time:
                    min_time = tte
                    winning_event = ev

            # Event time in absolute clock
            event_time = current_time + min_time

            if event_time >= self.time_horizon:
                # Censor at time horizon
                remaining = self.time_horizon - current_time
                lys, qalys, _ = self._sojourn_outcomes(
                    strategy, params, current_state,
                    current_time, self.time_horizon,
                )
                total_lys += lys
                total_qalys += qalys
                self._accrue_costs(
                    strategy, params, current_state,
                    current_time, self.time_horizon,
                    costs_by_cat,
                )
                time_in_state[self.states[current_state]] += remaining
                current_time = self.time_horizon
                break

            # Accrue outcomes for time in current state
            sojourn = min_time
            lys, qalys, _ = self._sojourn_outcomes(
                strategy, params, current_state,
                current_time, event_time,
            )
            total_lys += lys
            total_qalys += qalys
            self._accrue_costs(
                strategy, params, current_state,
                current_time, event_time,
                costs_by_cat,
            )
            time_in_state[self.states[current_state]] += sojourn

            # Log event
            event_log.append((
                event_time,
                self.states[current_state],
                self.states[winning_event.to_idx],
            ))

            # Transition
            current_state = winning_event.to_idx
            current_time = event_time

            # Entry costs for new state
            for ec in self._entry_costs:
                if ec.state_idx == current_state:
                    c = self._resolve_entry_cost(ec, strategy, params)
                    dc = self._discount_lump_sum(
                        c, current_time, self.dr_cost, self.discount_convention
                    )
                    cat = ec.category
                    costs_by_cat[cat] = costs_by_cat.get(cat, 0) + dc

            # On-enter handlers
            for handler in self._on_enter.get(self.states[current_state], []):
                result = handler(patient_idx, current_time, attrs or {})
                if result and "cost" in result:
                    amount = float(result["cost"])
                    if not np.isfinite(amount):
                        raise ValueError("on_state_enter returned a non-finite cost")
                    costs_by_cat["event"] = costs_by_cat.get("event", 0.0) + (
                        self._discount_lump_sum(
                            amount, current_time, self.dr_cost,
                            self.discount_convention,
                        )
                    )

        total_cost = sum(costs_by_cat.values())

        return {
            'total_cost': total_cost,
            'total_qalys': total_qalys,
            'total_lys': total_lys,
            'costs_by_cat': costs_by_cat,
            'event_log': event_log,
            'time_in_state': time_in_state,
        }

    def _sojourn_outcomes(
        self, strategy, params, state_idx, t_start, t_end,
    ) -> Tuple[float, float, float]:
        """Compute discounted LYs and QALYs for a sojourn."""
        if state_idx in self._absorbing:
            return 0.0, 0.0, t_end - t_start

        duration = t_end - t_start
        # Discounted LYs
        lys = self._discount_continuous(
            1.0, t_start, t_end, self.dr_qaly, self.discount_convention
        )
        # Discounted QALYs
        u = self._resolve_utility(strategy, state_idx, params)
        qalys = lys * u

        return lys, qalys, duration

    def _accrue_costs(
        self, strategy, params, state_idx, t_start, t_end,
        costs_by_cat: dict,
    ):
        """Accrue discounted state costs for a sojourn period."""
        for sc in self._state_costs:
            rate = self._resolve_cost_rate(sc, strategy, state_idx, params)
            if rate == 0:
                continue
            dc = self._discount_continuous(
                rate, t_start, t_end, self.dr_cost, self.discount_convention
            )
            cat = sc.category
            costs_by_cat[cat] = costs_by_cat.get(cat, 0) + dc

    # =====================================================================
    # Public run methods
    # =====================================================================

    def run(
        self,
        n_patients: int = 5000,
        seed: Optional[int] = None,
        progress: bool = True,
        attrs: Optional[Dict[str, np.ndarray]] = None,
    ) -> "DESResult":
        """Run a deterministic base case (point estimate parameters).

        Parameters
        ----------
        n_patients : int
            Number of patients per strategy.
        seed : int, optional
            Random seed.
        progress : bool
            Print progress updates.
        attrs : dict, optional
            Patient attributes: ``{attr_name: array of length n_patients}``.

        Returns
        -------
        DESResult
        """
        from ..analysis.results import DESResult

        if isinstance(n_patients, bool) or not isinstance(n_patients, int) or n_patients <= 0:
            raise ValueError("n_patients must be a positive integer")
        if attrs is not None:
            for name, values in attrs.items():
                values = np.asarray(values)
                if values.ndim != 1 or len(values) != n_patients:
                    raise ValueError(
                        f"attrs[{name!r}] must be a one-dimensional array of length n_patients"
                    )

        if seed is not None:
            np.random.seed(seed)

        params = self._get_base_params()
        self._validate_runtime_discount_rates(params)
        results = {}
        common_seeds = (
            np.random.randint(0, 2**32 - 1, size=n_patients, dtype=np.uint32)
            if self.n_strategies > 1 else None
        )

        for strategy in self.strategy_names:
            if progress:
                print(f"  DES: {self.strategy_labels[strategy]}...", end="", flush=True)

            patient_results = []
            for i in range(n_patients):
                pat_attrs = None
                if attrs is not None:
                    pat_attrs = {k: float(v[i]) for k, v in attrs.items()}
                patient_rng = (
                    np.random.RandomState(int(common_seeds[i]))
                    if common_seeds is not None else None
                )
                pr = self._simulate_patient(
                    strategy, params, pat_attrs, patient_idx=i, rng=patient_rng,
                )
                patient_results.append(pr)

            # Aggregate
            costs_arr = np.array([r['total_cost'] for r in patient_results])
            qalys_arr = np.array([r['total_qalys'] for r in patient_results])
            lys_arr = np.array([r['total_lys'] for r in patient_results])

            # Per-category costs
            all_cats = set()
            for r in patient_results:
                all_cats.update(r['costs_by_cat'].keys())
            cat_arrays = {
                cat: np.array([r['costs_by_cat'].get(cat, 0) for r in patient_results])
                for cat in sorted(all_cats)
            }

            # Time in state
            tis_arrays = {
                s: np.array([r['time_in_state'][s] for r in patient_results])
                for s in self.states
            }

            results[strategy] = {
                'total_cost': costs_arr,
                'total_qalys': qalys_arr,
                'total_lys': lys_arr,
                'mean_cost': float(costs_arr.mean()),
                'mean_qalys': float(qalys_arr.mean()),
                'mean_lys': float(lys_arr.mean()),
                'costs_by_cat': cat_arrays,
                'time_in_state': tis_arrays,
                'patient_results': patient_results,
                'n_patients': n_patients,
            }

            if progress:
                print(f" mean cost={costs_arr.mean():,.0f}, "
                      f"QALYs={qalys_arr.mean():.3f}, "
                      f"LYs={lys_arr.mean():.3f}")

        return DESResult(model=self, results=results, params=params)

    def run_psa(
        self,
        n_sim: int = 200,
        n_patients: int = 1000,
        seed: Optional[int] = None,
        progress: bool = True,
        attrs: Optional[Dict[str, np.ndarray]] = None,
    ) -> "DESPSAResult":
        """Run probabilistic sensitivity analysis.

        Each outer-loop iteration samples new parameter values; each
        inner-loop simulates ``n_patients`` with those parameters.

        Parameters
        ----------
        n_sim : int
            Number of PSA iterations (outer loop).
        n_patients : int
            Patients per strategy per iteration (inner loop).
        seed : int, optional
            Random seed.
        progress : bool
            Print progress.
        attrs : dict, optional
            Patient attributes.

        Returns
        -------
        DESPSAResult
        """
        from ..analysis.results import DESPSAResult

        if isinstance(n_sim, bool) or not isinstance(n_sim, int) or n_sim <= 0:
            raise ValueError("n_sim must be a positive integer")
        if isinstance(n_patients, bool) or not isinstance(n_patients, int) or n_patients <= 0:
            raise ValueError("n_patients must be a positive integer")
        if attrs is not None:
            for name, values in attrs.items():
                values = np.asarray(values)
                if values.ndim != 1 or len(values) != n_patients:
                    raise ValueError(
                        f"attrs[{name!r}] must be a one-dimensional array of length n_patients"
                    )

        if seed is not None:
            np.random.seed(seed)

        psa_iterations = []
        sampled_params_list = []

        for sim_idx in range(n_sim):
            # Sample parameters
            params = self._get_base_params()
            for name, param in self.params.items():
                if param.dist is not None:
                    params[name] = float(param.dist.sample(1)[0])
            self._validate_runtime_discount_rates(params)
            sampled_params_list.append(params)
            common_seeds = (
                np.random.randint(0, 2**32 - 1, size=n_patients, dtype=np.uint32)
                if self.n_strategies > 1 else None
            )

            # Simulate all strategies
            sim_result = {}
            with self._attr_param_override(params):
                for strategy in self.strategy_names:
                    costs_list = []
                    qalys_list = []
                    lys_list = []
                    for i in range(n_patients):
                        pat_attrs = None
                        if attrs is not None:
                            pat_attrs = {k: float(v[i]) for k, v in attrs.items()}
                        patient_rng = (
                            np.random.RandomState(int(common_seeds[i]))
                            if common_seeds is not None else None
                        )
                        pr = self._simulate_patient(
                            strategy, params, pat_attrs, patient_idx=i,
                            rng=patient_rng,
                        )
                        costs_list.append(pr['total_cost'])
                        qalys_list.append(pr['total_qalys'])
                        lys_list.append(pr['total_lys'])

                    sim_result[strategy] = {
                        'mean_cost': float(np.mean(costs_list)),
                        'mean_qalys': float(np.mean(qalys_list)),
                        'mean_lys': float(np.mean(lys_list)),
                    }

            psa_iterations.append(sim_result)

            if progress and (sim_idx + 1) % max(1, n_sim // 10) == 0:
                print(f"  PSA: {sim_idx + 1}/{n_sim} ({100 * (sim_idx + 1) / n_sim:.0f}%)")

        if progress:
            print(f"  PSA complete: {n_sim} iterations × {n_patients} patients")

        return DESPSAResult(
            model=self,
            psa_iterations=psa_iterations,
            sampled_params=sampled_params_list,
        )

    # =====================================================================
    # Info
    # =====================================================================

    def info(self) -> str:
        """Return a summary string."""
        lines = [
            "DiscreteEventSimulationModel",
            f"  States ({self.n_states}): {self.states}",
            f"  Strategies ({self.n_strategies}): {self.strategy_names}",
            f"  Time horizon: {self.time_horizon} years",
            f"  Discount rates: cost={self.dr_cost:.1%}, QALY={self.dr_qaly:.1%}",
            f"  Discount convention: {self.discount_convention}",
            f"  Parameters ({len(self.params)}):",
        ]
        for name, p in self.params.items():
            dist_str = f", dist={p.dist}" if p.dist else ""
            lines.append(f"    {name}: base={p.base:.4f}{dist_str}")

        for strategy in self.strategy_names:
            events = self._events[strategy]
            lines.append(f"  Events ({strategy}): {len(events)}")
            for ev in events:
                lines.append(f"    {ev.from_state} → {ev.to_state}: {ev.distribution}")

        if self._state_costs:
            lines.append(f"  State cost categories ({len(self._state_costs)}):")
            for sc in self._state_costs:
                lines.append(f"    {sc.category}")

        if self._entry_costs:
            lines.append(f"  Entry costs ({len(self._entry_costs)}):")
            for ec in self._entry_costs:
                lines.append(f"    {ec.category}: → {ec.state}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DiscreteEventSimulationModel(states={self.states}, "
            f"strategies={self.strategy_names}, "
            f"time_horizon={self.time_horizon})"
        )


# Concise public alias retained for compatibility and everyday use.
DESModel = DiscreteEventSimulationModel
