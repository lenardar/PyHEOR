"""Partitioned survival model with curves evaluated at cycle boundaries. See README.md."""

import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..time import Cycle
from ..distributions import sample_distribution
from .common import Param as _Param, CohortSweepModel
from ..survival import SurvivalDistribution, ProportionalHazards
from ..utils import (
    resolve_value, discount_factor, normalize_hcc, interval_occupancy,
)


class PartitionedSurvivalModel(CohortSweepModel):
    """Partitioned survival model with curves evaluated at cycle boundaries. See README.md."""

    def __init__(
        self,
        states: List[str],
        survival_endpoints: List[str],
        strategies: Union[List[str], Dict[str, str]],
        n_cycles: int,
        cycle: Cycle,
        dr_cost: Union[float, "_Param"] = 0.0,
        dr_qaly: Union[float, "_Param"] = 0.0,
        method: str = "life-table",
        state_type: Optional[Dict[str, str]] = None,
        terminal_state: Optional[str] = None,
    ):
        # States
        self.states = list(states)
        if not self.states:
            raise ValueError("states must contain at least one state")
        if len(set(self.states)) != len(self.states):
            raise ValueError(f"State names must be unique, got {self.states!r}")
        self.n_states = len(self.states)

        # Survival endpoints
        self.survival_endpoints = list(survival_endpoints)
        if not self.survival_endpoints:
            raise ValueError("survival_endpoints must contain at least one endpoint")
        if len(set(self.survival_endpoints)) != len(self.survival_endpoints):
            raise ValueError(
                f"Survival endpoint names must be unique, "
                f"got {self.survival_endpoints!r}"
            )
        self.n_endpoints = len(self.survival_endpoints)

        self.terminal_state = terminal_state
        if terminal_state is not None and (self.n_endpoints != 2 or self.n_states != 4
                                          or self.states[-2] != terminal_state):
            raise ValueError("Terminal PSM requires [PFS, PD, Terminal, Dead] and two endpoints")
        if self.n_states != self.n_endpoints + 1 + (terminal_state is not None):
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
        if not self.strategy_names:
            raise ValueError("strategies must contain at least one strategy")
        if len(set(self.strategy_names)) != len(self.strategy_names):
            raise ValueError(
                f"Strategy names must be unique, got {self.strategy_names!r}"
            )
        self.n_strategies = len(self.strategy_names)

        # Model settings
        if isinstance(n_cycles, bool) or not isinstance(n_cycles, (int, np.integer)):
            raise TypeError(f"n_cycles must be an integer, got {type(n_cycles).__name__}")
        if n_cycles <= 0:
            raise ValueError(f"n_cycles must be positive, got {n_cycles!r}")
        if not isinstance(cycle, Cycle):
            raise TypeError("cycle must be an explicit Cycle(length, unit)")
        self.n_cycles = int(n_cycles)
        self.cycle = cycle
        self.cycle_length = cycle.years
        self.discount_convention = "discrete"
        self.method = normalize_hcc(method)

        # Parameters (init early so discount rates can register into it)
        self.params: Dict[str, _Param] = {}

        # Discount rates
        self._register_discount_rates(
            dr_cost, dr_qaly, self.discount_convention
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
            self._alive_states = [
                i for i, s in enumerate(self.states)
                if state_type.get(s, "alive") == "alive"
            ]
        else:
            self._alive_states = list(range(self.n_states - (2 if terminal_state is not None else 1)))


        # Survival curves: {strategy: {endpoint: SurvivalDistribution or callable}}
        self._survival_curves: Dict[str, Dict[str, Any]] = {}

        self._init_rewards()


    # =========================================================================
    # Parameter Management (same API as MarkovModel)
    # =========================================================================


    # =========================================================================
    # Survival Curves
    # =========================================================================

    def set_survival(
        self,
        strategy: str,
        endpoint: str,
        curve: Union[SurvivalDistribution, Callable],
    ) -> "PartitionedSurvivalModel":
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
        PartitionedSurvivalModel
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
    ) -> "PartitionedSurvivalModel":
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


    # =========================================================================
    # Internal: Resolve Values
    # =========================================================================


    def _resolve_curve(self, strategy: str, endpoint: str,
                       params: Dict[str, float]) -> SurvivalDistribution:
        """Resolve a survival curve, evaluating callable if needed."""
        if strategy not in self._survival_curves:
            raise ValueError(
                f"No survival curve configured for strategy {strategy!r}, "
                f"endpoint {endpoint!r}."
            )
        if endpoint not in self._survival_curves[strategy]:
            raise ValueError(
                f"No survival curve configured for strategy {strategy!r}, "
                f"endpoint {endpoint!r}."
            )
        curve = self._survival_curves[strategy][endpoint]
        if callable(curve) and not isinstance(curve, SurvivalDistribution):
            curve = curve(params)
        if not isinstance(curve, SurvivalDistribution):
            raise TypeError(
                f"Survival curve for strategy {strategy!r}, endpoint "
                f"{endpoint!r} must resolve to SurvivalDistribution, "
                f"got {type(curve).__name__}."
            )
        return curve


    # =========================================================================
    # Simulation Engine
    # =========================================================================

    def _resolve_survival_values(
        self, strategy: str, params: Dict[str, float], curves=None,
    ) -> np.ndarray:
        """Evaluate and validate every endpoint curve for one strategy.

        Returns
        -------
        np.ndarray
            Shape (n_cycles + 1, n_endpoints).
        """
        times = np.arange(self.n_cycles + 1)
        surv_values = np.zeros((self.n_cycles + 1, self.n_endpoints))
        for j, endpoint in enumerate(self.survival_endpoints):
            curve = curves[endpoint] if curves is not None else self._resolve_curve(strategy, endpoint, params)
            values = np.asarray(curve.survival(times), dtype=float)
            if values.shape != times.shape:
                raise ValueError(
                    f"Survival curve for strategy {strategy!r}, endpoint "
                    f"{endpoint!r} returned shape {values.shape}; "
                    f"expected {times.shape}."
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Survival curve for strategy {strategy!r}, endpoint "
                    f"{endpoint!r} returned non-finite values."
                )
            if np.any(values < -1e-10) or np.any(values > 1 + 1e-10):
                bad = np.where((values < -1e-10) | (values > 1 + 1e-10))[0]
                raise ValueError(
                    f"Survival curve for strategy {strategy!r}, endpoint "
                    f"{endpoint!r} is outside [0, 1] at time indices "
                    f"{bad.tolist()}."
                )
            increasing = np.where(np.diff(values) > 1e-10)[0]
            if increasing.size:
                raise ValueError(
                    f"Survival curve for strategy {strategy!r}, endpoint "
                    f"{endpoint!r} increases between time indices "
                    f"{increasing.tolist()} and "
                    f"{(increasing + 1).tolist()}."
                )
            surv_values[:, j] = values

        # Ordered endpoints must not cross: S_1(t) <= ... <= S_N(t).
        for j in range(1, self.n_endpoints):
            crossed = surv_values[:, j] < surv_values[:, j - 1] - 1e-12
            if np.any(crossed):
                indices = np.flatnonzero(crossed)
                raise ValueError(
                    f"PSM curve crossing for strategy {strategy!r}: endpoint "
                    f"{self.survival_endpoints[j]!r} falls below "
                    f"{self.survival_endpoints[j - 1]!r} at time indices "
                    f"{indices.tolist()}. Check the curve parameters."
                )
        return surv_values

    def _compute_state_probs(
        self, strategy: str, params: Dict[str, float], surv_values=None,
    ) -> np.ndarray:
        """Compute state probabilities from survival curves.

        Returns
        -------
        np.ndarray
            Shape (n_cycles + 1, n_states). State membership at each cycle.
        """
        if surv_values is None:
            surv_values = self._resolve_survival_values(strategy, params)

        # Derive state probabilities
        state_probs = np.zeros((self.n_cycles + 1, self.n_states))

        # First state: S_1(t)
        state_probs[:, 0] = surv_values[:, 0]

        # Middle states: state[k] = S_k(t) - S_{k-1}(t), e.g. for
        # states=[PFS, Progressed, Dead], endpoints=[PFS, OS]:
        # PFS = S_PFS(t), Progressed = S_OS(t) - S_PFS(t), Dead = 1 - S_OS(t)
        for k in range(1, self.n_endpoints):
            state_probs[:, k] = surv_values[:, k] - surv_values[:, k - 1]

        # Last state: 1 - S_last(t)
        death = 1.0 - surv_values[:, -1]
        if self.terminal_state is not None:
            state_probs[:, -2] = np.diff(death, prepend=0)
            state_probs[:, -1] = np.concatenate(([0.0], death[:-1]))
        else:
            state_probs[:, -1] = death

        if np.any(state_probs < -1e-10) or np.any(state_probs > 1 + 1e-10):
            bad = np.argwhere(
                (state_probs < -1e-10) | (state_probs > 1 + 1e-10)
            )
            raise ValueError(
                f"PSM state probabilities for strategy {strategy!r} are "
                f"outside [0, 1] at: {bad.tolist()}."
            )

        return state_probs

    def _simulate_single(self, params):
        results = {}
        for strategy in self.strategy_names:
            curves = {endpoint: self._resolve_curve(strategy, endpoint, params)
                      for endpoint in self.survival_endpoints}
            values = self._resolve_survival_values(strategy, params, curves)
            trace = self._compute_state_probs(strategy, params, values)
            result = self._cycle_rewards(strategy, params, trace)
            result['times'] = self.cycle.time(np.arange(self.n_cycles + 1), unit="year")
            result['survival_distributions'] = curves
            result['survival_curves'] = {e: values[:, j] for j, e in enumerate(self.survival_endpoints)}
            results[strategy] = result
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


    def run_psa(
        self,
        n_sim: int = 1000,
        seed: Optional[int] = None,
        progress: bool = True,
    ) -> "PSAResult":
        """Run probabilistic sensitivity analysis."""
        from ..analysis.results import PSAResult

        if isinstance(n_sim, bool) or not isinstance(n_sim, (int, np.integer)):
            raise TypeError("n_sim must be a positive integer")
        if n_sim <= 0:
            raise ValueError("n_sim must be a positive integer")
        rng = np.random.default_rng(seed)

        sampled_params = []
        for i in range(n_sim):
            p = self._get_base_params()
            for name, param in self.params.items():
                if param.dist is not None:
                    p[name] = float(sample_distribution(param.dist, 1, rng)[0])
            sampled_params.append(p)

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
    # Convenience
    # =========================================================================

    def info(self):
        return (f"{type(self).__name__}: {self.n_states} states, {self.n_strategies} strategies\n"
                f"  Cycles: {self.n_cycles} × {self.cycle.length} {self.cycle.unit}\n"
                f"  Method: {self.method}; discount rates per cycle: {self._discount_rate('dr_cost', self._get_base_params())}, {self._discount_rate('dr_qaly', self._get_base_params())}\n"
                f"  Cost components: {list(self._costs)}; QALY components: {list(self._qalys)}")


    def __repr__(self):
        return (
            f"PartitionedSurvivalModel(states={self.states}, "
            f"endpoints={self.survival_endpoints}, "
            f"strategies={self.strategy_names}, "
            f"n_cycles={self.n_cycles})"
        )


# Concise public alias retained for compatibility and everyday use.
PSMModel = PartitionedSurvivalModel
