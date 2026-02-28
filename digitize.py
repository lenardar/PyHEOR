"""Reconstruct individual patient data (IPD) from digitized Kaplan-Meier curves.

Implements the Guyot et al. (2012) algorithm for mapping digitized KM
coordinates back to pseudo-IPD that can be used with ``SurvivalFitter``.

References
----------
Guyot P, Ades AE, Ouwens MJ, Welton NJ (2012). Enhanced secondary
analysis of survival data: reconstructing the data from published
Kaplan-Meier survival curves. *BMC Med Res Methodol*, 12:9.

Liu N, Zhou Y, Lee JJ (2021). IPDfromKM: reconstruct individual patient
data from published Kaplan-Meier survival curves. *BMC Med Res Methodol*.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# =========================================================================
# Public: Preprocessing
# =========================================================================

def clean_digitized_km(
    time,
    survival,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess noisy digitized Kaplan-Meier coordinates.

    Applies the following corrections in order:

    1. Sort by time.
    2. Remove points with ``time < 0`` or ``survival`` outside [0, 1].
    3. Ensure the curve starts at (0, 1.0).
    4. Remove digitization outliers — points causing two consecutive
       large jumps (top-1% method from IPDfromKM).
    5. Collapse duplicate times (keep max and min per time for KM steps).
    6. Enforce monotonically non-increasing survival.
    7. Remove consecutive duplicate survival values.

    Parameters
    ----------
    time : array-like
        Digitized time coordinates.
    survival : array-like
        Digitized survival probabilities (values in 0–1 scale).

    Returns
    -------
    time : np.ndarray
        Cleaned time points.
    survival : np.ndarray
        Cleaned survival probabilities.
    """
    t = np.asarray(time, dtype=float).copy()
    s = np.asarray(survival, dtype=float).copy()
    if len(t) != len(s):
        raise ValueError("time and survival must have the same length")

    # 1. Sort by time
    order = np.argsort(t)
    t, s = t[order], s[order]

    # 2. Remove out-of-bound points
    mask = (t >= 0) & (s >= 0) & (s <= 1)
    t, s = t[mask], s[mask]

    if len(t) == 0:
        raise ValueError("No valid data points after filtering")

    # 3. Ensure starts at (0, 1.0)
    if t[0] > 0:
        t = np.concatenate([[0.0], t])
        s = np.concatenate([[1.0], s])
    elif s[0] != 1.0:
        s[0] = 1.0

    # 4. Outlier removal — top 1% consecutive jumps "sandwich" method
    if len(s) > 3:
        diffs = np.abs(np.diff(s))
        threshold = np.percentile(diffs, 99)
        big_jump_idx = set(np.where(diffs >= threshold)[0])
        # A point k is an outlier if both k and k-1 are big-jump boundaries,
        # i.e., the jump INTO k and OUT OF k are both in the top 1%.
        remove = set()
        for idx in big_jump_idx:
            if (idx + 1) in big_jump_idx:
                # point at index idx+1 is the "sandwich" outlier
                remove.add(idx + 1)
        if remove:
            keep = np.array(
                [i for i in range(len(t)) if i not in remove]
            )
            t, s = t[keep], s[keep]

    # 5. Collapse duplicate times — keep max and min per unique time
    unique_times = np.unique(t)
    if len(unique_times) < len(t):
        new_t, new_s = [], []
        for ut in unique_times:
            mask = t == ut
            vals = s[mask]
            s_min, s_max = vals.min(), vals.max()
            if s_min == s_max:
                new_t.append(ut)
                new_s.append(s_min)
            else:
                # KM step: top then bottom
                new_t.extend([ut, ut])
                new_s.extend([s_max, s_min])
        t = np.array(new_t)
        s = np.array(new_s)

    # 6. Enforce monotonically non-increasing survival
    for k in range(1, len(s)):
        if s[k] > s[k - 1]:
            s[k] = s[k - 1]

    # 7. Remove consecutive duplicates (survival unchanged)
    keep = [0]
    for k in range(1, len(s)):
        if s[k] != s[k - 1] or t[k] != t[k - 1]:
            keep.append(k)
    t, s = t[keep], s[keep]

    return t, s


# =========================================================================
# Internal: Interval Mapping
# =========================================================================

def _build_interval_map(
    time: np.ndarray,
    t_risk: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map digitized time indices to at-risk intervals.

    For each interval *i* (defined by consecutive ``t_risk`` entries),
    find the first and last index of ``time`` that falls within it.

    Parameters
    ----------
    time : np.ndarray
        Cleaned digitized time points.
    t_risk : np.ndarray
        Times at which number-at-risk is reported.

    Returns
    -------
    lower : np.ndarray of int
        Index of the first digitized point in each interval.
    upper : np.ndarray of int
        Index of the last digitized point in each interval.
    """
    n_int = len(t_risk)
    lower = np.zeros(n_int, dtype=int)
    upper = np.zeros(n_int, dtype=int)

    for i in range(n_int):
        # First digitized point >= t_risk[i]
        candidates = np.where(time >= t_risk[i])[0]
        if len(candidates) == 0:
            lower[i] = len(time) - 1
        else:
            lower[i] = candidates[0]

        # Last digitized point before t_risk[i+1] (or end)
        if i < n_int - 1:
            candidates = np.where(time < t_risk[i + 1])[0]
            if len(candidates) == 0:
                upper[i] = lower[i]
            else:
                upper[i] = candidates[-1]
        else:
            upper[i] = len(time) - 1

    return lower, upper


# =========================================================================
# Public: Core Guyot Algorithm
# =========================================================================

def guyot_reconstruct(
    time,
    survival,
    t_risk,
    n_risk,
    tot_events: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct pseudo-IPD from a digitized KM curve (Guyot method).

    Given digitized Kaplan-Meier coordinates and a number-at-risk table,
    reconstructs individual patient-level time-to-event data.  The output
    can be fed directly into ``SurvivalFitter`` for parametric fitting.

    Parameters
    ----------
    time : array-like
        Digitized time coordinates.
    survival : array-like
        Digitized survival probabilities (0–1 scale).
    t_risk : array-like
        Times at which number-at-risk is reported.  Must be sorted.
    n_risk : array-like
        Number at risk at each ``t_risk`` time point.
    tot_events : int, optional
        Total number of events reported in the study.  Improves censoring
        estimation in the last interval when provided.

    Returns
    -------
    ipd_time : np.ndarray
        Reconstructed event/censoring times.
    ipd_event : np.ndarray
        Event indicators (1 = event, 0 = censored).

    Examples
    --------
    >>> import pyheor as ph
    >>> t = [0, 3, 6, 9, 12, 18, 24]
    >>> s = [1.0, 0.85, 0.72, 0.61, 0.50, 0.35, 0.25]
    >>> t_risk = [0, 6, 12, 18, 24]
    >>> n_risk = [120, 95, 68, 42, 25]
    >>> ipd_time, ipd_event = ph.guyot_reconstruct(t, s, t_risk, n_risk)
    >>> fitter = ph.SurvivalFitter(ipd_time, ipd_event, label="OS")
    >>> fitter.fit()

    References
    ----------
    Guyot P, Ades AE, Ouwens MJ, Welton NJ (2012).
    *BMC Med Res Methodol*, 12:9.
    """
    # --- Input validation & conversion ---
    TT, SS = clean_digitized_km(time, survival)
    t_risk = np.asarray(t_risk, dtype=float)
    n_risk = np.asarray(n_risk, dtype=float).copy()

    if len(t_risk) != len(n_risk):
        raise ValueError("t_risk and n_risk must have the same length")
    if len(t_risk) < 1:
        raise ValueError("At least one at-risk time point is required")
    if n_risk[0] < 1:
        raise ValueError("n_risk[0] must be >= 1")

    # Trim trailing zeros from n_risk (following IPDfromKM convention)
    while len(n_risk) > 1 and n_risk[-1] == 0:
        n_risk = n_risk[:-1]
        t_risk = t_risk[: len(n_risk)]

    K = len(TT)            # total digitized points
    n_int = len(t_risk)     # number of at-risk intervals

    # --- Build interval map ---
    lower, upper = _build_interval_map(TT, t_risk)

    # --- Working arrays ---
    d = np.zeros(K, dtype=int)         # events at each digitized point
    cen = np.zeros(K, dtype=int)       # censored at each digitized point
    nhat = np.zeros(K + 1)             # estimated n at risk
    KM_hat = np.ones(K)                # estimated KM survival
    ncensor = np.zeros(n_int, dtype=int)

    nhat[0] = n_risk[0]

    # Track "last event" index per interval
    last = 0

    # --- Process intervals 0 .. n_int-2 ---
    for i in range(n_int - 1):
        lo_i = lower[i]
        up_i = upper[i]
        lo_next = lower[i + 1]

        # Save "last" from previous interval to restore each iteration
        last_saved = last

        # Step 1: initial censoring estimate
        if SS[lo_i] > 0:
            ncensor[i] = int(
                round(n_risk[i] * SS[lo_next] / SS[lo_i] - n_risk[i + 1])
            )
        else:
            ncensor[i] = 0
        ncensor[i] = max(ncensor[i], 0)

        # Step 4: iterative convergence
        max_iter = 100
        for _iteration in range(max_iter):
            # Reset last to start-of-interval value each iteration
            last = last_saved
            # Step 2: distribute censored uniformly
            cen[lo_i: up_i + 1] = 0
            if ncensor[i] > 0:
                interval_start = TT[lo_i]
                interval_end = TT[lo_next]
                cen_times = [
                    interval_start
                    + j * (interval_end - interval_start) / (ncensor[i] + 1)
                    for j in range(1, ncensor[i] + 1)
                ]
                for k in range(lo_i, up_i + 1):
                    t_lo = TT[k]
                    t_hi = TT[k + 1] if k + 1 < K else TT[k] + 1
                    cen[k] = sum(1 for ct in cen_times if t_lo <= ct < t_hi)

            # Step 3: compute events
            nhat[lo_i] = n_risk[i]
            for k in range(lo_i, up_i + 1):
                if i == 0 and k == lo_i:
                    # Very first point
                    d[k] = 0
                    KM_hat[k] = 1.0
                else:
                    if KM_hat[last] > 0 and nhat[k] > 0:
                        d[k] = int(
                            round(nhat[k] * (1 - SS[k] / KM_hat[last]))
                        )
                        d[k] = max(0, min(d[k], int(nhat[k])))
                    else:
                        d[k] = 0

                    if nhat[k] > 0:
                        KM_hat[k] = KM_hat[last] * (1 - d[k] / nhat[k])
                    else:
                        KM_hat[k] = KM_hat[last]

                nhat[k + 1] = max(nhat[k] - d[k] - cen[k], 0)

                if d[k] != 0:
                    last = k

            # Check convergence
            diff = nhat[lo_next] - n_risk[i + 1]
            if abs(diff) < 0.5:
                break

            # Adjust censoring
            ncensor[i] = ncensor[i] + int(round(diff))
            ncensor[i] = max(
                0, min(ncensor[i], int(n_risk[i] - n_risk[i + 1] + 1))
            )

        # Propagate to next interval
        n_risk[i + 1] = nhat[lo_next]

    # --- Last interval ---
    i_last = n_int - 1
    lo_i = lower[i_last]
    up_i = upper[i_last]

    # Estimate censoring for last interval
    if n_int > 1:
        # Extrapolate from previous intervals' censoring rate
        prev_widths = np.diff(t_risk[:n_int])
        mean_rate = np.mean(ncensor[:i_last]) / np.mean(prev_widths)
        last_width = TT[-1] - t_risk[i_last]
        ncensor[i_last] = int(round(mean_rate * last_width))

        if tot_events is not None:
            events_so_far = int(np.sum(d[:lo_i]))
            remaining = max(tot_events - events_so_far, 0)
            ncensor[i_last] = min(
                ncensor[i_last], max(int(n_risk[i_last]) - remaining, 0)
            )
        ncensor[i_last] = max(ncensor[i_last], 0)
    else:
        # Single interval
        if tot_events is not None:
            ncensor[i_last] = max(int(n_risk[0]) - tot_events, 0)
        else:
            ncensor[i_last] = 0

    # Distribute censoring in last interval
    cen[lo_i: up_i + 1] = 0
    if ncensor[i_last] > 0:
        interval_start = TT[lo_i]
        interval_end = TT[-1]
        if interval_end > interval_start:
            cen_times = [
                interval_start
                + j
                * (interval_end - interval_start)
                / (ncensor[i_last] + 1)
                for j in range(1, ncensor[i_last] + 1)
            ]
            for k in range(lo_i, up_i + 1):
                t_lo = TT[k]
                t_hi = TT[k + 1] if k + 1 < K else TT[k] + 1
                cen[k] = sum(1 for ct in cen_times if t_lo <= ct < t_hi)

    # Compute events in last interval
    nhat[lo_i] = n_risk[i_last]
    for k in range(lo_i, up_i + 1):
        if nhat[k] > 0 and KM_hat[last] > 0:
            d[k] = int(round(nhat[k] * (1 - SS[k] / KM_hat[last])))
            d[k] = max(0, min(d[k], int(nhat[k])))
        else:
            d[k] = 0

        if nhat[k] > 0:
            KM_hat[k] = KM_hat[last] * (1 - d[k] / nhat[k])
        else:
            KM_hat[k] = KM_hat[last]

        nhat[k + 1] = max(nhat[k] - d[k] - cen[k], 0)
        if d[k] != 0:
            last = k

    # --- Generate IPD ---
    ipd_times = []
    ipd_events = []

    for k in range(K):
        # Events at this time point
        if d[k] > 0:
            ipd_times.extend([TT[k]] * d[k])
            ipd_events.extend([1] * d[k])

        # Censored between this point and next
        if cen[k] > 0:
            if k + 1 < K:
                midpoint = (TT[k] + TT[k + 1]) / 2.0
            else:
                midpoint = TT[k]
            ipd_times.extend([midpoint] * cen[k])
            ipd_events.extend([0] * cen[k])

    # Remaining patients at end (alive at last time point)
    n_remaining = int(nhat[K])
    if n_remaining > 0:
        ipd_times.extend([TT[-1]] * n_remaining)
        ipd_events.extend([0] * n_remaining)

    return np.array(ipd_times, dtype=float), np.array(ipd_events, dtype=int)
