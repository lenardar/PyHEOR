"""Explicit conversions for historical annual-input research fixtures."""
import pyheor as _ph

def _method(value):
    if value is True:
        return 'life-table'
    if value is False or value is None:
        return 'beginning'
    return 'life-table' if value == 'trapezoidal' else value

def _cycle_values(model, values):
    factor = model.cycle.years if hasattr(model, 'cycle') else model.time_period.years

    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if callable(value):
            import inspect
            n = len(inspect.signature(value).parameters)
            return lambda p, t, a=None: convert(value(*(p, t, a)[:n]))
        if isinstance(value, str):
            return lambda p, t: p[value] * factor
        return value * factor
    return convert(values)

def _starting_values(model, values):
    if callable(values):
        return lambda p: _starting_values(model, values(p, 1))
    if not isinstance(values, dict):
        return values
    if set(values) <= set(model.strategy_names) and all((isinstance(v, dict) for v in values.values())):
        return {s: v.get(model.states[getattr(model, 'initial_state_idx', 0)], 0) for s, v in values.items()}
    return values.get(model.states[getattr(model, 'initial_state_idx', 0)], 0)
