"""
Utility functions for PyHEOR.

Includes the C (complement) sentinel, matrix resolution, discounting,
and value resolution helpers.
"""

import numpy as np
from typing import Any, Dict, Union


# =============================================================================
# Complement Sentinel
# =============================================================================

class _Complement:
    """Sentinel value for transition matrix complement.
    
    When used in a transition matrix row, C is replaced with
    1 minus the sum of all other elements in that row.
    
    Example
    -------
    >>> [C, 0.3, 0.1]  # C will become 0.6
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __repr__(self):
        return "C"
    
    def __str__(self):
        return "C"
    
    def __eq__(self, other):
        return isinstance(other, _Complement)
    
    def __hash__(self):
        return hash("_Complement_Singleton")


C = _Complement()


# =============================================================================
# Transition Matrix Helpers
# =============================================================================

def resolve_complement(matrix_data) -> np.ndarray:
    """Resolve C (complement) sentinel values in a transition matrix.
    
    Parameters
    ----------
    matrix_data : list of lists
        Transition matrix where C sentinels should be replaced with
        the complement (1 - sum of other entries in the row).
    
    Returns
    -------
    np.ndarray
        Full transition probability matrix with all values resolved.
        
    Raises
    ------
    ValueError
        If more than one C per row, or if complement would be negative.
    """
    n = len(matrix_data)
    result = np.zeros((n, n))
    
    for i in range(n):
        c_idx = None
        row_sum = 0.0
        for j in range(n):
            val = matrix_data[i][j]
            if isinstance(val, _Complement) or val is C:
                if c_idx is not None:
                    raise ValueError(
                        f"Row {i}: only one C (complement) allowed per row"
                    )
                c_idx = j
            else:
                result[i][j] = float(val)
                row_sum += result[i][j]
        
        if c_idx is not None:
            complement = 1.0 - row_sum
            if complement < -1e-8:
                raise ValueError(
                    f"Row {i}: complement is negative ({complement:.6f}). "
                    f"Sum of other elements ({row_sum:.6f}) exceeds 1."
                )
            result[i][c_idx] = max(0.0, complement)
    
    return result


def validate_transition_matrix(P: np.ndarray, tol: float = 1e-6) -> bool:
    """Validate that P is a proper transition probability matrix.
    
    Parameters
    ----------
    P : np.ndarray
        Square matrix to validate.
    tol : float
        Tolerance for numerical errors.
        
    Returns
    -------
    bool
        True if valid.
        
    Raises
    ------
    ValueError
        If validation fails.
    """
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"Transition matrix must be square, got shape {P.shape}")

    if not np.all(np.isfinite(P)):
        bad = np.argwhere(~np.isfinite(P))
        raise ValueError(
            f"Transition matrix contains non-finite values at: {bad.tolist()}"
        )
    
    if np.any(P < -tol):
        neg = np.argwhere(P < -tol)
        raise ValueError(f"Negative probabilities at: {neg.tolist()}")

    if np.any(P > 1.0 + tol):
        high = np.argwhere(P > 1.0 + tol)
        raise ValueError(f"Probabilities greater than 1 at: {high.tolist()}")
    
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=tol))[0]
        raise ValueError(
            f"Row sums must equal 1. Rows {bad.tolist()} sum to "
            f"{row_sums[bad].tolist()}"
        )
    
    return True


# =============================================================================
# Discounting
# =============================================================================

def discount_factor(t: Union[int, float, np.ndarray], rate: float,
                    cycle_length: float = 1.0,
                    convention: str = "discrete") -> Union[float, np.ndarray]:
    """Calculate discount factor(s).
    
    Parameters
    ----------
    t : int, float, or array
        Cycle index or indices. Fractional values such as ``0.5`` are allowed.
    rate : float
        Annual discount rate (e.g., 0.03 for 3%).
    cycle_length : float
        Length of each cycle in years.
    convention : {"discrete", "continuous"}
        ``"discrete"`` treats ``rate`` as an annual effective rate and uses
        ``(1 + rate) ** -time``. ``"continuous"`` treats it as a continuous
        discount rate and uses ``exp(-rate * time)``.
    
    Returns
    -------
    float or array
        Discount factor(s) under the selected convention.
    """
    if convention not in {"discrete", "continuous"}:
        raise ValueError(
            f"Unknown discount convention {convention!r}; "
            "expected 'discrete' or 'continuous'."
        )
    if not np.isfinite(rate):
        raise ValueError(f"Discount rate must be finite, got {rate!r}")
    if not np.isfinite(cycle_length) or cycle_length <= 0:
        raise ValueError(
            f"cycle_length must be a positive finite number, got {cycle_length!r}"
        )
    if convention == "discrete" and rate <= -1:
        raise ValueError(
            f"Discrete discount rate must be greater than -1, got {rate!r}"
        )

    time = np.asarray(t, dtype=float) * cycle_length
    if not np.all(np.isfinite(time)):
        raise ValueError("Discount times must be finite")

    if convention == "discrete":
        factors = (1.0 + rate) ** (-time)
    else:
        factors = np.exp(-rate * time)

    if np.ndim(factors) == 0:
        return float(factors)
    return factors


# =============================================================================
# Half-Cycle Correction
# =============================================================================

def normalize_hcc(value):
    """Normalize half_cycle_correction input to a canonical form.

    Parameters
    ----------
    value : bool, str, or None
        - True → "trapezoidal"
        - False or None → None (no correction)
        - "trapezoidal" → "trapezoidal"
        - "life-table" → "trapezoidal" (compatibility alias)

    Returns
    -------
    str or None
        "trapezoidal" or None.

    Raises
    ------
    ValueError
        If value is not a recognized option.
    """
    if value is True:
        return "trapezoidal"
    elif value is False or value is None:
        return None
    elif isinstance(value, str):
        v = value.lower().strip()
        if v in ("trapezoidal", "life-table"):
            return "trapezoidal"
        raise ValueError(
            f"Invalid half_cycle_correction: {value!r}. "
            f"Expected True, False, None, 'trapezoidal', or 'life-table'."
        )
    else:
        raise TypeError(
            f"half_cycle_correction must be bool, str, or None, "
            f"got {type(value).__name__}"
        )


def interval_occupancy(trace, half_cycle_correction=None):
    """Return one state-occupancy row for each model interval.

    A trace contains ``n_cycles + 1`` observation points. Rewards accrue over
    the ``n_cycles`` intervals between them. Without half-cycle correction the
    interval uses its starting occupancy; trapezoidal correction averages the
    two endpoints.

    Parameters
    ----------
    trace : array-like, shape (n_cycles + 1, n_states)
    half_cycle_correction : bool, str, or None
        Accepted values are the same as :func:`normalize_hcc`.

    Returns
    -------
    np.ndarray, shape (n_cycles, n_states)
    """
    values = np.asarray(trace, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(
            "trace must be a 2D array with at least two observation points"
        )
    method = normalize_hcc(half_cycle_correction)
    if method == "trapezoidal":
        return (values[:-1] + values[1:]) / 2.0
    return values[:-1].copy()


def life_table_corrected_trace(trace):
    """Compatibility wrapper for interval-level trapezoidal occupancy."""
    return interval_occupancy(trace, "trapezoidal")


# =============================================================================
# Value Resolution
# =============================================================================

def resolve_value(value: Any, params: Dict[str, float], t: int = 0) -> float:
    """Resolve a value that may be a constant, parameter reference, or function.
    
    Parameters
    ----------
    value : float, str, or callable
        - float: used directly
        - str: looked up in params dict
        - callable: called as value(params, t)
    params : dict
        Current parameter values.
    t : int
        Current cycle number.
    
    Returns
    -------
    float
        The resolved numeric value.
    """
    if callable(value):
        return float(value(params, t))
    elif isinstance(value, str):
        if value not in params:
            raise KeyError(
                f"Parameter '{value}' not found. "
                f"Available: {list(params.keys())}"
            )
        return float(params[value])
    else:
        return float(value)
