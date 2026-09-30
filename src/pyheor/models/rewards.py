"""Cycle rewards shared by cohort and individual discrete-time engines."""

from dataclasses import dataclass
from types import MappingProxyType
import numpy as np

from ..time import Cycle
from ..utils import resolve_value, interval_occupancy, discount_factor, normalize_hcc


@dataclass(frozen=True)
class StateReward:
    values: object
    cycles: object = None


@dataclass(frozen=True)
class RewardContext:
    strategy: str
    params: object
    cycle_index: int
    cycle: Cycle
    states: tuple
    state_prev: object
    state_curr: object
    occupancy: object
    transition_matrix: object = None
    flows: object = None
    corrected_flows: object = None
    patient_index: object = None
    attributes: object = None

    def flow(self, source, target, *, corrected=False):
        """Actual/expected flow, not the net change in state prevalence."""
        values = self.corrected_flows if corrected else self.flows
        if values is None:
            raise ValueError("This model does not identify state-to-state flows")
        return float(values[self.states.index(source), self.states.index(target)])


class CycleRewards:
    """Registration and evaluation of undiscounted per-cycle rewards."""

    @property
    def method(self):
        return self._occupancy_method

    @method.setter
    def method(self, value):
        self._occupancy_method = normalize_hcc(value)

    def _init_rewards(self):
        self._costs = {}
        self._qalys = {}
        self._starting_rewards = {"cost": {}, "qaly": {}}
        self._event_rewards = {"cost": [], "qaly": []}
        self._custom_rewards = {"cost": {}, "qaly": {}}

    def _reward_cycles(self, cycles):
        if cycles is None:
            return None
        if isinstance(cycles, (int, np.integer)) and not isinstance(cycles, bool):
            cycles = [cycles]
        try:
            result = tuple(cycles)
        except TypeError:
            raise ValueError(f"cycles must be an integer or an iterable of integers from 1 to {self.n_cycles}") from None
        if any(isinstance(c, bool) or not isinstance(c, (int, np.integer))
               or not 1 <= c <= self.n_cycles for c in result):
            raise ValueError(f"cycles must contain integers from 1 to {self.n_cycles}")
        if len(set(result)) != len(result):
            raise ValueError("cycles must not contain duplicates")
        return result

    def _set_state_reward(self, kind, category, values, cycles):
        if not callable(values):
            self._validate_state_mapping(values, kind)
        definitions = self._costs if kind == "cost" else self._qalys
        definitions[category] = StateReward(values, self._reward_cycles(cycles))
        return self

    def set_state_cost(self, category, values, *, cycles=None):
        """Register costs per cycle, weighted by state occupancy."""
        return self._set_state_reward("cost", category, values, cycles)

    def set_state_qaly(self, category, values, *, cycles=None):
        """Register QALYs per cycle, including negative loss components."""
        return self._set_state_reward("qaly", category, values, cycles)

    def _set_starting(self, kind, category, value):
        self._validate_scalar_reward(value)
        self._starting_rewards[kind][category] = value
        return self

    def set_starting_cost(self, category, value):
        """An unweighted lump sum at model time zero."""
        return self._set_starting("cost", category, value)

    def set_starting_qaly(self, category, value):
        return self._set_starting("qaly", category, value)

    def _validate_scalar_reward(self, value):
        if isinstance(value, (list, tuple)):
            raise TypeError("Reward schedules were removed; use a callback and cycles")
        if isinstance(value, dict):
            unknown = set(value) - set(self.strategy_names)
            if unknown:
                raise ValueError(f"Unknown strategies in reward: {sorted(unknown)}")
            for item in value.values():
                self._validate_scalar_reward(item)

    def _set_event_reward(self, kind, category, source, target, value):
        if not hasattr(self, "_transitions"):
            raise ValueError("PSM has no identifiable entry/transition flows; use Terminal")
        for state in (source, target):
            if state is not None and state not in self.states:
                raise ValueError(f"Unknown state {state!r}")
        if source == target:
            raise ValueError("A transition event must change state")
        self._validate_scalar_reward(value)
        self._event_rewards[kind].append((category, source, target, value))
        return self

    def set_transition_cost(self, category, from_state, to_state, value):
        return self._set_event_reward("cost", category, from_state, to_state, value)

    def set_transition_qaly(self, category, from_state, to_state, value):
        return self._set_event_reward("qaly", category, from_state, to_state, value)

    def set_entry_cost(self, category, state, value):
        return self._set_event_reward("cost", category, None, state, value)

    def set_entry_qaly(self, category, state, value):
        return self._set_event_reward("qaly", category, None, state, value)

    def _set_custom(self, kind, category, func):
        if not callable(func):
            raise TypeError("Custom reward must be callable")
        self._custom_rewards[kind][category] = func
        return self

    def set_custom_cost(self, category, func):
        """Callback(ctx) returns an already weighted, undiscounted cycle cost."""
        return self._set_custom("cost", category, func)

    def set_custom_qaly(self, category, func):
        return self._set_custom("qaly", category, func)

    def _scalar_reward(self, value, strategy, params, cycle_index, attrs=None):
        if isinstance(value, dict):
            value = value.get(strategy, 0)
        if callable(value):
            import inspect
            arity = len(inspect.signature(value).parameters)
            args = (params, cycle_index, attrs)[:arity]
            value = value(*args)
        result = resolve_value(value, params, cycle_index)
        if not np.isfinite(result):
            raise ValueError("Reward must be finite")
        return result

    def _state_reward_vector(self, definition, strategy, params, cycle_index, attrs=None):
        if definition.cycles is not None and cycle_index not in definition.cycles:
            return np.zeros(self.n_states)
        return self._resolve_state_values(definition.values, strategy, params, cycle_index, attrs)

    def _get_qalys(self, strategy, params, cycle_index, attrs=None):
        return sum((self._state_reward_vector(d, strategy, params, cycle_index, attrs)
                    for d in self._qalys.values()), np.zeros(self.n_states))

    def _get_state_costs(self, category, strategy, params, cycle_index, attrs=None):
        return self._state_reward_vector(self._costs[category], strategy, params, cycle_index, attrs)

    def _cycle_rewards(self, strategy, params, trace, flows=None, matrices=None,
                       attrs=None, patient_index=None):
        """Evaluate one trace; flows are known transitions, never net prevalence."""
        raw_occupancy = interval_occupancy(trace, "beginning")
        occupancy = interval_occupancy(trace, self.method)
        corrected = None
        if flows is not None:
            if self.method == "life-table":
                corrected = (flows + np.concatenate([np.zeros_like(flows[:1]), flows[:-1]])) / 2
            elif self.method == "beginning":
                corrected = np.concatenate([np.zeros_like(flows[:1]), flows[:-1]])
            else:
                corrected = flows
        outputs, raw_outputs = {}, {}
        for kind, definitions in (("cost", self._costs), ("qaly", self._qalys)):
            categories = dict.fromkeys([*definitions, *self._starting_rewards[kind],
                                       *(e[0] for e in self._event_rewards[kind]),
                                       *self._custom_rewards[kind]])
            by_category = {c: np.zeros(self.n_cycles) for c in categories}
            raw = {c: np.zeros(self.n_cycles) for c in categories}
            for i in range(self.n_cycles):
                cycle_index = i + 1
                for category, definition in definitions.items():
                    vector = self._state_reward_vector(definition, strategy, params, cycle_index, attrs)
                    by_category[category][i] += occupancy[i] @ vector
                    raw[category][i] += raw_occupancy[i] @ vector
                for category, source, target, value in self._event_rewards[kind]:
                    j = self.states.index(target)
                    if source is None:
                        weight = trace[0, j] if i == 0 else corrected[i, :, j].sum() - corrected[i, j, j]
                    else:
                        weight = corrected[i, self.states.index(source), j]
                    amount = weight * self._scalar_reward(value, strategy, params, cycle_index, attrs)
                    by_category[category][i] += amount
                    raw[category][i] += amount
                for category, callback in self._custom_rewards[kind].items():
                    def readonly(a):
                        if a is None:
                            return None
                        a = np.array(a, copy=True)
                        a.setflags(write=False)
                        return a
                    ctx = RewardContext(strategy, MappingProxyType(params), cycle_index,
                                        self.cycle, tuple(self.states), readonly(trace[i]),
                                        readonly(trace[i + 1]), readonly(occupancy[i]),
                                        readonly(matrices[i]) if matrices is not None else None,
                                        readonly(flows[i]) if flows is not None else None,
                                        readonly(corrected[i]) if corrected is not None else None,
                                        patient_index, MappingProxyType(attrs or {}))
                    amount = float(callback(ctx))
                    if not np.isfinite(amount):
                        raise ValueError(f"Custom {kind} {category!r} must be finite")
                    by_category[category][i] += amount
                    raw[category][i] += amount
            for category, value in self._starting_rewards[kind].items():
                amount = self._scalar_reward(value, strategy, params, 0, attrs)
                by_category[category][0] += amount
                raw[category][0] += amount
            outputs[kind] = by_category
            raw_outputs[kind] = raw
        cost_df = discount_factor(np.arange(self.n_cycles), self._discount_rate("dr_cost", params))
        qaly_df = discount_factor(np.arange(self.n_cycles), self._discount_rate("dr_qaly", params))
        qaly_raw = sum(raw_outputs["qaly"].values(), np.zeros(self.n_cycles))
        qaly_occ = sum(outputs["qaly"].values(), np.zeros(self.n_cycles))
        qaly_components = {c: v * qaly_df for c, v in outputs["qaly"].items()}
        alive = np.zeros(self.n_states)
        alive[list(self._alive_states)] = self.cycle.years
        lys = raw_occupancy @ alive
        lys_occ = occupancy @ alive
        cost_disc = {c: v * cost_df for c, v in outputs["cost"].items()}
        return dict(trace=trace, interval_times=np.arange(self.n_cycles) * self.cycle.years,
                    costs_by_cycle=raw_outputs["cost"], costs_hcc=outputs["cost"],
                    qalys_by_cycle=qaly_raw, qalys_hcc=qaly_occ,
                    lys_by_cycle=lys, lys_hcc=lys_occ, discounted_costs=cost_disc,
                    discounted_qalys=qaly_occ * qaly_df, discounted_lys=lys_occ * qaly_df,
                    qaly_components=qaly_components,
                    undiscounted_costs=outputs["cost"], undiscounted_qalys=outputs["qaly"],
                    total_costs={c: float(v.sum()) for c, v in cost_disc.items()},
                    total_qalys=float((qaly_occ * qaly_df).sum()),
                    total_lys=float((lys_occ * qaly_df).sum()))

    def _batch_patient_rewards(self, strategy, params, histories):
        """Vectorized rewards for homogeneous patients without custom callbacks."""
        N, T = histories.shape[0], self.n_cycles
        cost = {c: np.zeros((N, T)) for c in self._costs}
        health = {c: np.zeros((N, T)) for c in self._qalys}
        alive = np.zeros(self.n_states)
        alive[list(self._alive_states)] = self.cycle.years
        lys = np.zeros((N, T))
        def occupied(vector, i):
            begin, end = histories[:, i], histories[:, i + 1]
            if self.method == "life-table":
                return (vector[begin] + vector[end]) / 2
            return vector[begin if self.method == "beginning" else end]
        for i in range(T):
            for definitions, outputs in ((self._costs, cost), (self._qalys, health)):
                for category, definition in definitions.items():
                    vector = self._state_reward_vector(definition, strategy, params, i + 1)
                    outputs[category][:, i] = occupied(vector, i)
            lys[:, i] = occupied(alive, i)
        for kind, outputs in (("cost", cost), ("qaly", health)):
            for category, source, target, value in self._event_rewards[kind]:
                j = self.states.index(target)
                raw = (histories[:, 1:] == j) & (histories[:, :-1] != j)
                if source is not None:
                    raw &= histories[:, :-1] == self.states.index(source)
                raw = raw.astype(float)
                previous = np.concatenate([np.zeros((N, 1)), raw[:, :-1]], axis=1)
                weight = (raw + previous) / 2 if self.method == "life-table" else previous if self.method == "beginning" else raw
                if source is None:
                    weight[:, 0] = histories[:, 0] == j
                values = np.array([self._scalar_reward(value, strategy, params, i + 1) for i in range(T)])
                outputs.setdefault(category, np.zeros((N, T)))[:] += weight * values
            for category, value in self._starting_rewards[kind].items():
                outputs.setdefault(category, np.zeros((N, T)))[:, 0] += self._scalar_reward(value, strategy, params, 0)
        time_alive = lys.sum(axis=1)
        cost_df = discount_factor(np.arange(T), self._discount_rate("dr_cost", params))
        qaly_df = discount_factor(np.arange(T), self._discount_rate("dr_qaly", params))
        raw_cost, raw_health = cost, health
        cost = {c: v * cost_df for c, v in cost.items()}
        health = {c: v * qaly_df for c, v in health.items()}
        return (sum(cost.values(), np.zeros((N, T))), sum(health.values(), np.zeros((N, T))),
                lys * qaly_df, time_alive, cost, health, raw_cost, raw_health)
