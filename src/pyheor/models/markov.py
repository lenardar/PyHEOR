"""Cohort Markov model with explicit cycles and per-cycle rewards. See README.md."""

import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..time import Cycle
from ..distributions import sample_distribution
from .common import Param as _Param, CohortSweepModel
from ..utils import (
    C, _Complement, resolve_complement, resolve_value, discount_factor,
    normalize_hcc, interval_occupancy, validate_transition_matrix,
)


# =============================================================================
# CohortStateTransitionModel
# =============================================================================

class CohortStateTransitionModel(CohortSweepModel):
    """Explicit-cycle state-transition model. See README.md."""
    
    def __init__(
        self,
        states: List[str],
        strategies: Union[List[str], Dict[str, str]],
        n_cycles: int,
        cycle: Cycle,
        dr_cost: Union[float, "_Param"] = 0.0,
        dr_qaly: Union[float, "_Param"] = 0.0,
        method: str = "life-table",
        initial_state: Union[str, int] = 0,
        state_type: Optional[Dict[str, str]] = None,
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
        
        self._init_rewards()


    # =========================================================================
    # Parameter Management
    # =========================================================================
    
    
    
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
    
    
    # =========================================================================
    # Transition Costs
    # =========================================================================


    # =========================================================================
    # Utility
    # =========================================================================
    
    
    # =========================================================================
    # Internal: Parameter Resolution
    # =========================================================================
    

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

    
    
    
    
    # =========================================================================
    # Simulation Engine
    # =========================================================================
    
    def _simulate_single(self, params):
        missing = set(self.strategy_names) - set(self._transitions)
        if missing:
            raise ValueError(f"Missing transition matrices for strategies: {sorted(missing)}")
        results = {}
        for strategy in self.strategy_names:
            matrices = np.array([self._get_transition_matrix(strategy, params, k)
                                 for k in range(1, self.n_cycles + 1)])
            trace = np.zeros((self.n_cycles + 1, self.n_states))
            trace[0, self.initial_state_idx] = 1
            for i, matrix in enumerate(matrices):
                trace[i + 1] = trace[i] @ matrix
            flows = trace[:-1, :, None] * matrices
            results[strategy] = self._cycle_rewards(strategy, params, trace, flows, matrices)
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
    
    def info(self):
        return (f"{type(self).__name__}: {self.n_states} states, {self.n_strategies} strategies\n"
                f"  Cycles: {self.n_cycles} × {self.cycle.length} {self.cycle.unit}\n"
                f"  Method: {self.method}; discount rates per cycle: {self._discount_rate('dr_cost', self._get_base_params())}, {self._discount_rate('dr_qaly', self._get_base_params())}\n"
                f"  Cost components: {list(self._costs)}; QALY components: {list(self._qalys)}")

    
    def __repr__(self):
        return (
            f"CohortStateTransitionModel(states={self.states}, "
            f"strategies={self.strategy_names}, "
            f"n_cycles={self.n_cycles})"
        )


# Concise public alias retained for compatibility and everyday use.
MarkovModel = CohortStateTransitionModel
