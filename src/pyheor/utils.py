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
    
    if np.any(P < -tol):
        neg = np.argwhere(P < -tol)
        raise ValueError(f"Negative probabilities at: {neg.tolist()}")
    
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

def discount_factor(t: Union[int, np.ndarray], rate: float,
                    cycle_length: float = 1.0) -> Union[float, np.ndarray]:
    """Calculate discount factor(s).
    
    Parameters
    ----------
    t : int or array
        Cycle number(s).
    rate : float
        Annual discount rate (e.g., 0.03 for 3%).
    cycle_length : float
        Length of each cycle in years.
    
    Returns
    -------
    float or array
        Discount factor(s): (1 + rate)^(-(t * cycle_length))
    """
    if rate == 0:
        if isinstance(t, np.ndarray):
            return np.ones_like(t, dtype=float)
        return 1.0
    return (1.0 + rate) ** (-(np.asarray(t, dtype=float) * cycle_length))


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
        - "life-table" → "life-table"

    Returns
    -------
    str or None
        "trapezoidal", "life-table", or None.

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
            return v
        raise ValueError(
            f"Invalid half_cycle_correction: {value!r}. "
            f"Expected True, False, None, 'trapezoidal', or 'life-table'."
        )
    else:
        raise TypeError(
            f"half_cycle_correction must be bool, str, or None, "
            f"got {type(value).__name__}"
        )


def life_table_corrected_trace(trace):
    """Compute heemod-style life-table corrected trace.

    For t = 0..n-1: corrected[t] = (trace[t] + trace[t+1]) / 2
    corrected[n] = trace[n]  (last cycle unchanged)

    Parameters
    ----------
    trace : np.ndarray, shape (n_cycles+1, n_states)

    Returns
    -------
    np.ndarray, same shape as input
    """
    corrected = np.empty_like(trace)
    corrected[:-1] = (trace[:-1] + trace[1:]) / 2.0
    corrected[-1] = trace[-1]
    return corrected


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
