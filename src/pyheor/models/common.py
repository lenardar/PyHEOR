"""Shared data definitions used by the model engines."""

from dataclasses import dataclass
from typing import Any, List, Optional

from ..distributions import Distribution


@dataclass
class Param:
    """A model parameter with point estimate and optional uncertainty."""

    base: float
    dist: Optional[Distribution] = None
    label: Optional[str] = None
    low: Optional[float] = None
    high: Optional[float] = None

    def __post_init__(self):
        if self.label is None:
            self.label = ""
        if self.low is None:
            self.low = self.base * 0.8
        if self.high is None:
            self.high = self.base * 1.2


@dataclass
class _CostDef:
    """Internal cost definition shared by cycle-based model engines."""

    name: str
    values: Any
    first_cycle_only: bool = False
    apply_cycles: Optional[List[int]] = None
    method: str = "wlos"
