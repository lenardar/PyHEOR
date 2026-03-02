"""
DESModel — Discrete Event Simulation for health economic evaluation.
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
- Straightforward integration with NMA posterior HR samples

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
>>> model = DESModel(
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
# DESModel
# =============================================================================

class DESModel:
    """Discrete Event Simulation model for health economic evaluation.

    Parameters
    ----------
    states : list of str
        Health state names.
    strategies : list of str or dict
        Treatment strategies. Dict maps internal name → display label.
    time_horizon : float
        Maximum simulation time in years.
    discount_rate : float or dict
        Annual discount rate(s). If dict, keys are 'costs' and 'qalys'.
    state_type : dict, optional
        Map state names to "alive" or "dead" (default: last state is dead).

    Examples
    --------
    >>> model = DESModel(
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
        discount_rate: Union[float, Dict[str, float]] = 0.03,
        state_type: Optional[Dict[str, str]] = None,
    ):
        self.states = list(states)
        self.n_states = len(self.states)
        self.time_horizon = float(time_horizon)

        # Strategies
        if isinstance(strategies, dict):
            self.strategy_names = list(strategies.keys())
            self.strategy_labels = dict(strategies)
        else:
            self.strategy_names = list(strategies)
            self.strategy_labels = {s: s for s in self.strategy_names}
        self.n_strategies = len(self.strategy_names)

        # Discount
        if isinstance(discount_rate, (int, float)):
            self.dr_costs = float(discount_rate)
            self.dr_qalys = float(discount_rate)
        else:
            self.dr_costs = float(discount_rate.get('costs', 0.03))
            self.dr_qalys = float(discount_rate.get('qalys', 0.03))

        # State types
        if state_type is not None:
            self._alive_states = set(
                i for i, s in enumerate(self.states)
                if state_type.get(s, "alive") == "alive"
            )
        else:
            self._alive_states = set(range(self.n_states - 1))
        self._absorbing = set(range(self.n_states)) - self._alive_states

        # Parameters
        self.params: Dict[str, Param] = {}

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
    ) -> "DESModel":
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

    # =====================================================================
    # Events (state transitions)
    # =====================================================================

    def set_event(
        self,
        strategy: str,
        from_state: str,
        to_state: str,
        distribution: Any,
    ) -> "DESModel":
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

        Returns
        -------
        DESModel
            Self, for method chaining.

        Examples
        --------
        Fixed distribution:

        >>> model.set_event("SOC", "PFS", "Progressed",
        ...                 ph.Weibull(shape=1.2, scale=5))

        Parameter-dependent (e.g. NMA HR):

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

        ev = _EventDef(
            from_state=from_state,
            to_state=to_state,
            from_idx=self.states.index(from_state),
            to_idx=self.states.index(to_state),
            distribution=distribution,
        )
        self._events[strategy].append(ev)
        return self

    def set_events_from(
        self,
        strategy: str,
        from_state: str,
        events: Dict[str, Any],
    ) -> "DESModel":
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

    def set_state_cost(self, category: str, values: Any) -> "DESModel":
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
    ) -> "DESModel":
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

    def set_utility(self, values: Any) -> "DESModel":
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
    ) -> "DESModel":
        """Register a handler called when a patient enters a state.

        Parameters
        ----------
        state : str
            State name.
        handler : callable
            ``handler(patient_record)`` — can modify the record in place.
        """
        self._on_enter.setdefault(state, []).append(handler)
        return self

    # =====================================================================
    # Resolve helpers
    # =====================================================================

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
            if strategy in vals:
                inner = vals[strategy]
                if isinstance(inner, dict):
                    v = inner.get(state, 0)
                else:
                    v = inner  # single value for all states? unlikely
            elif state in vals:
                v = vals[state]
            else:
                v = 0
        else:
            v = vals

        if isinstance(v, str):
            v = params.get(v, 0)
        if callable(v):
            v = v(params)
        return float(v)

    def _resolve_entry_cost(
        self, ec: _EntryCostDef, strategy: str, params: dict,
    ) -> float:
        """Resolve one-time entry cost."""
        val = ec.value
        if isinstance(val, dict):
            val = val.get(strategy, 0)
        if isinstance(val, str):
            val = params.get(val, 0)
        if callable(val):
            val = val(params)
        return float(val)

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
            if strategy in vals:
                inner = vals[strategy]
                if isinstance(inner, dict):
                    v = inner.get(state, 0)
                else:
                    v = inner
            elif state in vals:
                v = vals[state]
            else:
                v = 0
        else:
            v = vals

        if isinstance(v, str):
            v = params.get(v, 0)
        if callable(v):
            v = v(params)
        return float(v)

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

    @staticmethod
    def _discount_lump_sum(amount: float, time: float, rate: float) -> float:
        """Discount a lump-sum amount at continuous time."""
        if rate <= 0:
            return amount
        return amount / (1 + rate) ** time

    @staticmethod
    def _discount_continuous(
        rate_per_year: float, t_start: float, t_end: float, dr: float,
    ) -> float:
        """Discounted integral of a constant rate from t_start to t_end.

        ∫_{t_start}^{t_end} rate / (1+dr)^t dt

        For dr > 0:  rate × [(1+dr)^{-t_start} - (1+dr)^{-t_end}] / ln(1+dr)
        """
        if t_end <= t_start:
            return 0.0
        if dr <= 0:
            return rate_per_year * (t_end - t_start)
        ln_dr = np.log(1 + dr)
        return rate_per_year * (
            np.exp(-ln_dr * t_start) - np.exp(-ln_dr * t_end)
        ) / ln_dr

    # =====================================================================
    # Simulation engine
    # =====================================================================

    def _sample_tte(self, dist: SurvivalDistribution) -> float:
        """Sample a time-to-event from a survival distribution using inverse CDF."""
        u = np.random.uniform()
        return dist.quantile(u)

    def _simulate_patient(
        self,
        strategy: str,
        params: dict,
        attrs: Optional[dict] = None,
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
                self._accrue_sojourn(
                    strategy, params, current_state,
                    current_time, current_time + remaining,
                    costs_by_cat, total_lys, total_qalys,
                )
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
                tte = self._sample_tte(dist)
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
                    dc = self._discount_lump_sum(c, current_time, self.dr_costs)
                    cat = ec.category
                    costs_by_cat[cat] = costs_by_cat.get(cat, 0) + dc

            # On-enter handlers
            for handler in self._on_enter.get(self.states[current_state], []):
                handler({
                    'time': current_time,
                    'state': self.states[current_state],
                    'strategy': strategy,
                    'params': params,
                    'attrs': attrs,
                })

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
        lys = self._discount_continuous(1.0, t_start, t_end, self.dr_qalys)
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
            dc = self._discount_continuous(rate, t_start, t_end, self.dr_costs)
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

        if seed is not None:
            np.random.seed(seed)

        params = self._get_base_params()
        results = {}

        for strategy in self.strategy_names:
            if progress:
                print(f"  DES: {self.strategy_labels[strategy]}...", end="", flush=True)

            patient_results = []
            for i in range(n_patients):
                pat_attrs = None
                if attrs is not None:
                    pat_attrs = {k: float(v[i]) for k, v in attrs.items()}
                pr = self._simulate_patient(strategy, params, pat_attrs)
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
            sampled_params_list.append(params)

            # Simulate all strategies
            sim_result = {}
            for strategy in self.strategy_names:
                costs_list = []
                qalys_list = []
                lys_list = []
                for i in range(n_patients):
                    pat_attrs = None
                    if attrs is not None:
                        pat_attrs = {k: float(v[i % len(v)]) for k, v in attrs.items()}
                    pr = self._simulate_patient(strategy, params, pat_attrs)
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
            "DESModel",
            f"  States ({self.n_states}): {self.states}",
            f"  Strategies ({self.n_strategies}): {self.strategy_names}",
            f"  Time horizon: {self.time_horizon} years",
            f"  Discount rates: costs={self.dr_costs:.1%}, QALYs={self.dr_qalys:.1%}",
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
            f"DESModel(states={self.states}, "
            f"strategies={self.strategy_names}, "
            f"time_horizon={self.time_horizon})"
        )
