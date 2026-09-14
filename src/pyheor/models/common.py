"""Shared data definitions used by the model engines."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..distributions import Distribution
from ..utils import discount_factor


@dataclass
class Param:
    """A model parameter with point estimate and optional uncertainty."""

    base: float
    dist: Optional[Distribution] = None
    label: Optional[str] = None
    low: Optional[float] = None
    high: Optional[float] = None
    #: Whether the caller supplied a bound. ``low`` and ``high`` are filled in
    #: unconditionally for reporting, which would otherwise make an explicit
    #: bound indistinguishable from a default one.
    _explicit_bounds: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        if self.label is None:
            self.label = ""
        self._explicit_bounds = self.low is not None or self.high is not None
        if self.low is None:
            self.low = self.base * 0.8
        if self.high is None:
            self.high = self.base * 1.2

    def bounds(self, range_pct: float = 0.2):
        """Return ``(low, high)`` for a one-way sensitivity sweep.

        An explicit bound always wins; otherwise the range is symmetric
        around ``base``.
        """
        if self._explicit_bounds:
            return self.low, self.high
        return self.base * (1 - range_pct), self.base * (1 + range_pct)


@dataclass
class _CostDef:
    """Internal cost definition shared by cycle-based model engines."""

    name: str
    values: Any
    first_cycle_only: bool = False
    apply_cycles: Optional[List[int]] = None
    method: str = "wlos"


class ParameterisedModel:
    """Parameter registration shared by every model engine.

    Subclasses must create ``self.params`` before registering anything.
    """

    #: Parameters stored as model attributes rather than in ``params``. A
    #: sensitivity or PSA draw must be written back to the attribute to take
    #: effect, which is what :meth:`_attr_param_override` does.
    _ATTR_PARAMS = frozenset({"dr_cost", "dr_qaly"})

    params: Dict[str, Param]

    def add_param(self, name: str, base: float, dist=None, label=None,
                  low=None, high=None):
        """Register one parameter.

        Parameters
        ----------
        name : str
            Parameter name, used as the lookup key in callbacks.
        base : float
            Point estimate used for the base case.
        dist : Distribution, optional
            Sampling distribution for PSA.
        label : str, optional
            Display label; defaults to ``name``.
        low, high : float, optional
            Bounds for one-way sensitivity analysis. When omitted the sweep
            uses a symmetric range around ``base``.

        Returns
        -------
        The model, for chaining.
        """
        self.params[name] = Param(
            base=base, dist=dist, label=label or name, low=low, high=high,
        )
        return self

    def add_params(self, params_dict: Dict[str, Union[Param, float]]):
        """Register several parameters at once.

        Values may be ``Param`` objects or plain numbers, which become
        ``Param`` objects without uncertainty.

        Returns
        -------
        The model, for chaining.
        """
        for name, param in params_dict.items():
            if isinstance(param, Param):
                if not param.label:
                    param.label = name
                self.params[name] = param
            elif isinstance(param, (int, float)) and not isinstance(param, bool):
                self.params[name] = Param(base=float(param), label=name)
            else:
                raise TypeError(
                    f"Parameter {name!r}: expected Param or numeric, got "
                    f"{type(param).__name__}"
                )
        return self

    def _get_base_params(self) -> Dict[str, float]:
        """Point estimates for every registered parameter."""
        return {name: p.base for name, p in self.params.items()}

    def _register_discount_rates(self, dr_cost, dr_qaly, convention: str) -> None:
        """Store discount rates, registering any passed as ``Param``.

        Rates given as ``Param`` also enter ``params`` so they can take part
        in sensitivity and probabilistic analysis.
        """
        for attribute, value, label in (
            ("dr_cost", dr_cost, "Discount Rate (Cost)"),
            ("dr_qaly", dr_qaly, "Discount Rate (QALY)"),
        ):
            if isinstance(value, Param):
                setattr(self, attribute, value.base)
                if not value.label:
                    value.label = label
                self.params[attribute] = value
            else:
                setattr(self, attribute, float(value))
            # Reject a rate the discount factor cannot represent before any
            # simulation starts, naming the argument at fault.
            try:
                discount_factor(
                    0, getattr(self, attribute), convention=convention
                )
            except ValueError as error:
                raise ValueError(f"{attribute}: {error}") from None

    def _owsa_parameters(self, params: Optional[List[str]]) -> List[str]:
        """Names to sweep: those with a distribution, else all of them."""
        if params is not None:
            unknown = [name for name in params if name not in self.params]
            if unknown:
                raise ValueError(
                    f"Unknown parameters for OWSA: {unknown!r}. "
                    f"Available: {list(self.params)!r}"
                )
            return list(params)
        selected = [
            name for name, p in self.params.items() if p.dist is not None
        ]
        return selected or list(self.params)

    @contextmanager
    def _attr_param_override(self, values: Dict[str, float]):
        """Temporarily apply any :attr:`_ATTR_PARAMS` present in ``values``."""
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


class CohortSweepModel(ParameterisedModel):
    """One-way sensitivity analysis for engines with ``_simulate_single``."""

    def run_owsa(self, params: Optional[List[str]] = None,
                 range_pct: float = 0.2, wtp: float = 50000):
        """Run one-way sensitivity analysis (OWSA).

        Each parameter moves to its low and high value in turn while the
        others stay at their base case.

        Parameters
        ----------
        params : list of str, optional
            Parameter names to vary. Defaults to those with a distribution,
            or to all parameters when none declares one.
        range_pct : float
            Symmetric range used for parameters without explicit bounds.
        wtp : float
            Willingness-to-pay threshold for the net benefit summary.

        Returns
        -------
        OWSAResult
        """
        from ..analysis.results import OWSAResult

        names = self._owsa_parameters(params)
        base_params = self._get_base_params()
        base_result = self._simulate_single(base_params)

        owsa_data = []
        for param_name in names:
            parameter = self.params[param_name]
            low, high = parameter.bounds(range_pct)
            is_attr = param_name in self._ATTR_PARAMS

            for bound, value in (('low', low), ('high', high)):
                test_params = dict(base_params, **{param_name: value})
                if is_attr:
                    with self._attr_param_override({param_name: value}):
                        result = self._simulate_single(test_params)
                else:
                    result = self._simulate_single(test_params)

                owsa_data.append({
                    'param': param_name,
                    'label': parameter.label,
                    'value': value,
                    'base_value': parameter.base,
                    'bound': bound,
                    'result': result,
                })

        return OWSAResult(
            model=self,
            base_result=base_result,
            base_params=base_params,
            owsa_data=owsa_data,
            wtp=wtp,
        )
