"""Continuous-time discrete-event simulation with declared time units and reward rates. See README.md."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .common import Param as _Param, StateMappingModel
from ..distributions import sample_distribution
from ..survival import SurvivalDistribution
from ..time import Cycle
from types import MappingProxyType
from types import SimpleNamespace
from scipy.integrate import quad
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


# =============================================================================
# DiscreteEventSimulationModel
# =============================================================================

class DiscreteEventSimulationModel(StateMappingModel):
    """Continuous-time discrete-event simulation with declared time units and reward rates. See README.md."""

    def __init__(
        self,
        states: List[str],
        strategies: Union[List[str], Dict[str, str]],
        time_horizon: float = 40.0,
        dr_cost: Union[float, "_Param"] = 0.0,
        dr_qaly: Union[float, "_Param"] = 0.0,
        state_type: Optional[Dict[str, str]] = None,
        clock: str = "reset",
        discount_convention: str = "discrete",
        time_unit: str = "year",
        initial_state: Union[str, int] = 0,
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
        self.time_unit = time_unit
        self.time_period = Cycle(1, time_unit)

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
        self.params: Dict[str, _Param] = {}

        # Discount rates
        self._register_discount_rates(
            dr_cost, dr_qaly, self.discount_convention
        )
        self._validate_discount_rate(self.dr_cost, "dr_cost")
        self._validate_discount_rate(self.dr_qaly, "dr_qaly")

        if isinstance(initial_state, str):
            if initial_state not in self.states:
                raise ValueError(
                    f"Unknown initial_state {initial_state!r}; "
                    f"available states are {self.states!r}"
                )
            self.initial_state_idx = self.states.index(initial_state)
        else:
            if isinstance(initial_state, bool):
                raise TypeError("initial_state must be a state name or integer index")
            self.initial_state_idx = int(initial_state)
            if self.initial_state_idx != initial_state or not 0 <= self.initial_state_idx < self.n_states:
                raise ValueError(
                    f"initial_state index must be between 0 and {self.n_states - 1}, "
                    f"got {initial_state!r}"
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
        self._absorbing = set(range(self.n_states)) - self._alive_states


        # Events: strategy -> list[_EventDef]
        self._events: Dict[str, List[_EventDef]] = {
            s: [] for s in self.strategy_names
        }

        self._init_rewards()

    @property
    def alive_state_indices(self) -> tuple[int, ...]:
        """Indices of states considered alive by the model."""
        return tuple(sorted(self._alive_states))

    @property
    def absorbing_state_indices(self) -> tuple[int, ...]:
        """Indices of absorbing states."""
        return tuple(sorted(self._absorbing))

    # =====================================================================
    # Parameters
    # =====================================================================


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
        if from_state == to_state:
            raise ValueError("Self-loop events are not supported; use a recurrent-event model")
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
        clock: Optional[str] = None,
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
        clock : str, optional
            Forwarded to :meth:`set_event` for every destination; the model
            default applies when omitted.

        Examples
        --------
        >>> model.set_events_from("SOC", "PFS", {
        ...     "Progressed": ph.Weibull(shape=1.2, scale=5),
        ...     "Dead": ph.Weibull(shape=1.0, scale=20),
        ... })
        """
        for to_state, dist in events.items():
            self.set_event(strategy, from_state, to_state, dist, clock=clock)
        return self

    # =====================================================================
    # Costs
    # =====================================================================


    # =====================================================================
    # Utility
    # =====================================================================


    # =====================================================================
    # Event handlers (advanced)
    # =====================================================================


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

    def _classify_mapping(self, mapping: dict, context: str) -> bool:
        """Return True for a strategy-level mapping, False for state-level.

        Rejects a mapping whose keys mix strategy and state names: allowing
        both, with strategy names taking silent priority, means a state name
        that happens to collide with a strategy name resolves unpredictably.
        """
        if not mapping:
            return False
        keys = set(mapping)
        strategy_names = set(self.strategy_names)
        state_names = set(self.states)

        if keys <= strategy_names and all(
            isinstance(value, dict) for value in mapping.values()
        ):
            for strategy, inner in mapping.items():
                unknown = set(inner) - state_names
                if unknown:
                    raise ValueError(
                        f"{context} for strategy {strategy!r} contains "
                        f"unknown states: {sorted(unknown)!r}"
                    )
            return True

        if keys <= state_names:
            return False

        unknown = keys - state_names - strategy_names
        if unknown:
            raise ValueError(
                f"{context} contains unknown state or strategy names: "
                f"{sorted(unknown)!r}"
            )
        raise ValueError(
            f"{context} mixes state-level and strategy-level keys; use "
            "either {state: value} or {strategy: {state: value}}."
        )

    def _validate_runtime_discount_rates(self, params: dict):
        """Validate fixed and PSA-sampled discount rates before simulation."""
        self._validate_discount_rate(params.get("dr_cost", self.dr_cost), "dr_cost")
        self._validate_discount_rate(params.get("dr_qaly", self.dr_qaly), "dr_qaly")


    def _resolve_distribution(
        self, ev: _EventDef, params: dict, attrs: Optional[dict] = None,
    ) -> SurvivalDistribution:
        """Resolve event distribution (may be callable)."""
        d = ev.distribution
        if isinstance(d, SurvivalDistribution):
            return d
        if callable(d):
            import inspect
            # Count every declared parameter, not just required ones: a
            # callable written as f(params, attrs=None) does accept attrs,
            # but has only one parameter without a default.
            n_args = len(inspect.signature(d).parameters)
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
        if np.isnan(tte):
            raise ValueError(f"Event distribution returned non-finite TTE: {tte!r}")
        if tte < 0:
            raise ValueError(f"Event distribution returned negative TTE: {tte!r}")
        return float(tte)

    def _sample_forward_tte(
        self, dist: SurvivalDistribution, current_time: float, rng=None,
    ) -> float:
        """Sample a residual TTE under a clock-forward cumulative hazard."""
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
        current_state = self.initial_state_idx
        current_time = 0.0
        event_count = 0
        event_log = []
        time_in_state = {s: 0.0 for s in self.states}
        costs_by_cat: Dict[str, float] = {}
        def categories(kind):
            definitions = self._costs if kind == 'cost' else self._qalys
            return dict.fromkeys([*definitions, *self._starting_rewards[kind],
                                  *(e[0] for e in self._event_rewards[kind]),
                                  *self._custom_rewards[kind]], 0.0)
        costs_by_cat = categories('cost')
        qaly_components = categories('qaly')
        undiscounted_costs, undiscounted_qalys = categories('cost'), categories('qaly')
        total_qalys = 0.0
        total_lys = 0.0

        total_qalys += self._book_event_rewards(strategy, params, None, current_state,
                                               0.0, attrs, patient_idx, costs_by_cat, qaly_components=qaly_components, undiscounted_costs=undiscounted_costs, undiscounted_qalys=undiscounted_qalys)
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
                    current_time, current_time + remaining, attrs, qaly_components=qaly_components, undiscounted_qalys=undiscounted_qalys)
                total_lys += lys
                total_qalys += qalys
                self._accrue_costs(
                    strategy, params, current_state,
                    current_time, current_time + remaining,
                    costs_by_cat, attrs, undiscounted_costs=undiscounted_costs)
                time_in_state[self.states[current_state]] += remaining
                current_time = self.time_horizon
                break

            # Sample time-to-event for each competing risk. Ties are broken
            # by declaration order (strict `<` below): irrelevant for
            # continuous distributions, but a point mass or coincident
            # degenerate distributions would always resolve to whichever
            # event was registered first.
            min_time = float('inf')
            winning_event = None

            for ev in eligible:
                dist = self._resolve_distribution(ev, params, attrs)
                clock = ev.clock or self.clock
                if clock == "forward":
                    tte = self._sample_forward_tte(dist, current_time, rng)
                else:
                    tte = self._sample_tte(dist, rng)
                if tte < min_time:
                    min_time = tte
                    winning_event = ev

            # Event time in absolute clock
            event_time = current_time + min_time

            event_count += 1
            if event_count > 10000:
                raise RuntimeError(
                    "DES exceeded 10000 events for one patient; "
                    "check for zero-time or recurrent event cycles"
                )

            if event_time >= self.time_horizon:
                # Censor at time horizon
                remaining = self.time_horizon - current_time
                lys, qalys, _ = self._sojourn_outcomes(
                    strategy, params, current_state,
                    current_time, self.time_horizon, attrs=attrs, qaly_components=qaly_components, undiscounted_qalys=undiscounted_qalys)
                total_lys += lys
                total_qalys += qalys
                self._accrue_costs(
                    strategy, params, current_state,
                    current_time, self.time_horizon,
                    costs_by_cat, attrs, undiscounted_costs=undiscounted_costs)
                time_in_state[self.states[current_state]] += remaining
                current_time = self.time_horizon
                break

            # Accrue outcomes for time in current state
            sojourn = min_time
            lys, qalys, _ = self._sojourn_outcomes(
                strategy, params, current_state,
                current_time, event_time, attrs, qaly_components=qaly_components, undiscounted_qalys=undiscounted_qalys)
            total_lys += lys
            total_qalys += qalys
            self._accrue_costs(
                strategy, params, current_state,
                current_time, event_time,
                costs_by_cat, attrs, undiscounted_costs=undiscounted_costs)
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

            total_qalys += self._book_event_rewards(
                strategy, params, winning_event.from_idx, current_state,
                current_time, attrs, patient_idx, costs_by_cat, qaly_components=qaly_components, undiscounted_costs=undiscounted_costs, undiscounted_qalys=undiscounted_qalys)

        total_cost = sum(costs_by_cat.values())

        return {
            'total_cost': total_cost,
            'total_qalys': total_qalys,
            'total_lys': total_lys,
            'costs_by_cat': costs_by_cat,
            'qaly_components': qaly_components,
            'undiscounted_costs': undiscounted_costs,
            'undiscounted_qalys': undiscounted_qalys,
            'event_log': event_log,
            'time_in_state': time_in_state,
        }

    def _reward_cycles(self, cycles):
        if cycles is not None:
            raise ValueError("DES has continuous time; use a time-dependent callback instead of cycles")
        return None

    def _set_event_reward(self, kind, category, source, target, value):
        for state in (source, target):
            if state is not None and state not in self.states:
                raise ValueError(f"Unknown state {state!r}")
        if source == target:
            raise ValueError("A transition event must change state")
        self._validate_scalar_reward(value)
        self._event_rewards[kind].append((category, source, target, value))
        return self

    def _book_event_rewards(self, strategy, params, source_idx, target_idx,
                            time, attrs, patient_idx, costs, qaly_components=None,
                            undiscounted_costs=None, undiscounted_qalys=None):
        source = self.states[source_idx] if source_idx is not None else None
        target = self.states[target_idx]
        qaly_total = 0.0
        for kind in ("cost", "qaly"):
            rate = self._discount_rate("dr_cost" if kind == "cost" else "dr_qaly", params)
            df = self._discount_lump_sum(1.0, time, rate, self.discount_convention)
            amounts = {}
            if source is None:
                for category, value in self._starting_rewards[kind].items():
                    amounts[category] = self._scalar_reward(value, strategy, params, time, attrs)
            for category, origin, destination, value in self._event_rewards[kind]:
                if destination == target and (origin is None or origin == source):
                    amounts[category] = amounts.get(category, 0) + self._scalar_reward(value, strategy, params, time, attrs)
            ctx = SimpleNamespace(strategy=strategy, params=MappingProxyType(params), time=time,
                                  time_unit=self.time_unit, from_state=source, to_state=target,
                                  patient_index=patient_idx, attributes=MappingProxyType(attrs or {}))
            for category, callback in self._custom_rewards[kind].items():
                amount = float(callback(ctx))
                if not np.isfinite(amount):
                    raise ValueError(f"Custom {kind} must be finite")
                amounts[category] = amounts.get(category, 0) + amount
            if kind == "cost":
                for category, amount in amounts.items():
                    costs[category] = costs.get(category, 0) + amount * df
                    if undiscounted_costs is not None:
                        undiscounted_costs[category] = undiscounted_costs.get(category, 0) + amount
            else:
                qaly_total += sum(amounts.values()) * df
                for category, amount in amounts.items():
                    if qaly_components is not None:
                        qaly_components[category] = qaly_components.get(category, 0) + amount * df
                    if undiscounted_qalys is not None:
                        undiscounted_qalys[category] = undiscounted_qalys.get(category, 0) + amount
        return qaly_total

    def _integrate_reward(self, definition, strategy, params, state_idx,
                          start, end, rate, attrs):
        if end == start:
            return 0.0
        def integrand(time):
            vector = self._resolve_state_values(definition.values, strategy, params, time, attrs)
            return vector[state_idx] * self._discount_lump_sum(1.0, time, rate, self.discount_convention)
        import inspect
        def dynamic(value):
            if callable(value):
                return len(inspect.signature(value).parameters) >= 2
            if isinstance(value, dict):
                return any(dynamic(v) for v in value.values())
            return False
        if not dynamic(definition.values):
            vector = self._resolve_state_values(definition.values, strategy, params, start, attrs)
            return self._discount_continuous(vector[state_idx], start, end, rate, self.discount_convention)
        return quad(integrand, start, end, epsabs=1e-8, epsrel=1e-10)[0]

    def _sojourn_outcomes(self, strategy, params, state_idx, t_start, t_end, attrs=None,
                          qaly_components=None, undiscounted_qalys=None):
        if state_idx in self._absorbing:
            return 0.0, 0.0, t_end - t_start
        lys = self._discount_continuous(self.time_period.years, t_start, t_end,
                                       self._discount_rate("dr_qaly", params), self.discount_convention)
        qalys = 0.0
        rate = self._discount_rate("dr_qaly", params)
        for category, definition in self._qalys.items():
            amount = self._integrate_reward(definition, strategy, params, state_idx,
                                            t_start, t_end, rate, attrs)
            qalys += amount
            if qaly_components is not None:
                qaly_components[category] = qaly_components.get(category, 0) + amount
            if undiscounted_qalys is not None:
                raw = amount if rate == 0 else self._integrate_reward(
                    definition, strategy, params, state_idx, t_start, t_end, 0, attrs)
                undiscounted_qalys[category] = undiscounted_qalys.get(category, 0) + raw
        return lys, qalys, t_end - t_start


    def _accrue_costs(self, strategy, params, state_idx, t_start, t_end, costs_by_cat, attrs=None, undiscounted_costs=None):
        for category, definition in self._costs.items():
            amount = self._integrate_reward(definition, strategy, params, state_idx,
                                            t_start, t_end, self._discount_rate("dr_cost", params), attrs)
            costs_by_cat[category] = costs_by_cat.get(category, 0) + amount
            if undiscounted_costs is not None:
                rate = self._discount_rate("dr_cost", params)
                raw = amount if rate == 0 else self._integrate_reward(
                    definition, strategy, params, state_idx, t_start, t_end, 0, attrs)
                undiscounted_costs[category] = undiscounted_costs.get(category, 0) + raw


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

        if isinstance(n_patients, bool) or not isinstance(n_patients, (int, np.integer)) or n_patients <= 0:
            raise ValueError("n_patients must be a positive integer")
        if attrs is not None:
            for name, values in attrs.items():
                values = np.asarray(values)
                if values.ndim != 1 or len(values) != n_patients:
                    raise ValueError(
                        f"attrs[{name!r}] must be a one-dimensional array of length n_patients"
                    )

        # One independent stream per patient, reused across strategies, so a
        # strategy difference is not inflated by unrelated draws.
        patient_seeds = np.random.SeedSequence(seed).spawn(n_patients)

        params = self._get_base_params()
        self._validate_runtime_discount_rates(params)
        results = {}

        for strategy in self.strategy_names:
            if progress:
                print(f"  DES: {self.strategy_labels[strategy]}...", end="", flush=True)

            patient_results = []
            for i in range(n_patients):
                pat_attrs = None
                if attrs is not None:
                    pat_attrs = {k: float(v[i]) for k, v in attrs.items()}
                patient_rng = np.random.default_rng(patient_seeds[i])
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
                **{key: {category: np.array([r[key].get(category, 0) for r in patient_results])
                         for category in sorted({c for r in patient_results for c in r[key]})}
                   for key in ('qaly_components', 'undiscounted_costs', 'undiscounted_qalys')},
                'time_in_state': tis_arrays,
                'patient_results': patient_results,
                'n_patients': n_patients,
            }

            if progress:
                print(f" mean cost={costs_arr.mean():,.0f}, "
                      f"QALYs={qalys_arr.mean():.3f}, "
                      f"LYs={lys_arr.mean():.3f}")

        return DESResult(model=self, results=results, params=params)

    run_base_case = run

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

        if isinstance(n_sim, bool) or not isinstance(n_sim, (int, np.integer)) or n_sim <= 0:
            raise ValueError("n_sim must be a positive integer")
        if isinstance(n_patients, bool) or not isinstance(n_patients, (int, np.integer)) or n_patients <= 0:
            raise ValueError("n_patients must be a positive integer")
        if attrs is not None:
            for name, values in attrs.items():
                values = np.asarray(values)
                if values.ndim != 1 or len(values) != n_patients:
                    raise ValueError(
                        f"attrs[{name!r}] must be a one-dimensional array of length n_patients"
                    )

        parameter_seq, patient_seq = np.random.SeedSequence(seed).spawn(2)
        parameter_rng = np.random.default_rng(parameter_seq)

        psa_iterations = []
        sampled_params_list = []

        for sim_idx in range(n_sim):
            # Sample parameters
            params = self._get_base_params()
            for name, param in self.params.items():
                if param.dist is not None:
                    params[name] = float(
                        sample_distribution(param.dist, 1, parameter_rng)[0]
                    )
            self._validate_runtime_discount_rates(params)
            sampled_params_list.append(params)
            patient_seeds = patient_seq.spawn(n_patients)

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
                        patient_rng = np.random.default_rng(patient_seeds[i])
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

    def info(self):
        return (f"DES: {self.n_states} states, {self.n_strategies} strategies; "
                f"horizon {self.time_horizon} {self.time_unit}; clock {self.clock}")


    def __repr__(self) -> str:
        return (
            f"DiscreteEventSimulationModel(states={self.states}, "
            f"strategies={self.strategy_names}, "
            f"time_horizon={self.time_horizon})"
        )


# Concise public alias retained for compatibility and everyday use.
DESModel = DiscreteEventSimulationModel
