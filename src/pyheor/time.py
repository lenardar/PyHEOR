"""Explicit durations and conversions; model engines never infer input units."""

from dataclasses import dataclass
import numpy as np


_YEARS = {"year": 1.0, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365}


@dataclass(frozen=True)
class Cycle:
    """A fixed model interval. A year has 12 months, 52 weeks or 365 days.

    These are modeling conventions, not calendar date arithmetic.
    """

    length: float
    unit: str

    def __post_init__(self):
        if self.unit not in _YEARS:
            raise ValueError(f"Unknown time unit {self.unit!r}; choose {tuple(_YEARS)}")
        if isinstance(self.length, bool) or not np.isfinite(self.length) or self.length <= 0:
            raise ValueError("Cycle length must be positive and finite")

    @property
    def years(self):
        return self.length * _YEARS[self.unit]

    def in_unit(self, unit):
        if unit not in _YEARS:
            raise ValueError(f"Unknown time unit {unit!r}")
        return self.years / _YEARS[unit]

    def time(self, boundary, unit=None):
        """Elapsed time at boundary 0..N, in the requested unit."""
        return np.asarray(boundary) * self.in_unit(unit or self.unit)


def qaly(utility, duration, unit=None):
    """Convert a utility weight (or decrement) and duration to QALYs.

    Call inside a parameter callback when utility participates in OWSA/PSA.
    """
    if isinstance(duration, Cycle):
        if unit is not None:
            raise ValueError("Do not specify unit together with a Cycle")
        years = duration.years
    else:
        if unit is None:
            raise ValueError("Numeric duration requires an explicit unit")
        years = Cycle(duration, unit).years
    value = np.asarray(utility, dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError("Utility must be finite")
    result = value * years
    return float(result) if result.ndim == 0 else result


def rescale_discount_rate(rate, from_period, to_period):
    """Rescale an effective discount rate between explicitly declared periods.

    Periods may be Cycle objects or numbers expressed in the same time unit.
    Do not mix a Cycle and a numeric period.
    """
    if isinstance(from_period, Cycle) != isinstance(to_period, Cycle):
        raise TypeError("Both periods must be Cycle objects, or both numeric")
    if isinstance(from_period, Cycle):
        source, target = from_period.years, to_period.years
    else:
        source, target = float(from_period), float(to_period)
    if not np.isfinite(source) or not np.isfinite(target) or source <= 0 or target <= 0:
        raise ValueError("Periods must be positive and finite")
    if not np.isfinite(rate) or rate <= -1:
        raise ValueError("Effective rate must be finite and greater than -1")
    return float(np.expm1(np.log1p(rate) * target / source))
