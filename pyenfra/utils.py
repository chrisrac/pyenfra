# -*- coding: utf-8 -*-
"""
This script is part of the pyenfra module and contains functions necessary to 
perform fractal analysis for environmental data. Documentation can be found 
on repository website: https://github.com/chrisrac/pyenfra

@author: Krzysztof Raczynski
https://github.com/chrisrac
"""

import numpy as np
import warnings
import pywt
from scipy.ndimage import label
from pyts.image import RecurrencePlot

__all__ = ['_check_array_1d', '_check_positive_integer', '_generate_adaptive_nvals',
           '_generate_nvals', '_rescaled_range_analysis', '_compute_fluctuations',
           '_generate_qvals', '_check_q_mode', '_confidence_interval_rs',
           '_confidence_interval_dfa', '_MF_DFA', '_phase_space_embedding',
           '_generate_wtmm_scales', '_WTMM', '_compute_log_fit',
           '_validate_binary_sequence', '_get_line_lengths', '_count_diagonal_lines',
           '_count_vertical_lines', '_generate_recurrence_plot', 
           '_compute_diagonal_line_lengths', '_compute_vertical_line_lengths']


def _check_array_1d(arr, varname='array'):
    """
    Verify that `arr` is a 1D NumPy array with at least 2 elements. Raises ValueError otherwise.

    Parameters
    ----------
    arr : array‐like
        Input array to check.
    varname : str, optional (default='array')
        Name of the variable (used for error messages).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `arr` cannot be converted to a 1D array of length ≥ 2.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _check_array_1d
    >>> _check_array_1d(np.array([1, 2, 3]), varname='test')
    >>> _check_array_1d([10, 20], varname='x')
    """
    
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"`{varname}` must be 1D; got shape {arr.shape}.")
    if arr.size < 2:
        raise ValueError(f"`{varname}` must have at least 2 elements; got size {arr.size}.")
        
        
def _check_positive_integer(val, varname='value', minimum=1):
    """
    Verify that `val` is an integer >= `minimum`. Raises ValueError otherwise.

    Parameters
    ----------
    val : any
        Value to check.
    varname : str, optional (default='value')
        Name of the variable (for error messages).
    minimum : int, optional (default=1)
        Minimum allowed integer value.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `val` is not an integer or `val < minimum`.

    Examples
    --------
    >>> from pyenfra.utils import _check_positive_integer
    >>> _check_positive_integer(5, varname='n', minimum=1)
    """
    
    if not isinstance(val, (int, np.integer)) or val < minimum:
        raise ValueError(f"`{varname}` must be an integer ≥ {minimum}; got {val}.")
        

def _generate_adaptive_nvals(data, num=20, min_n=8, min_segments=10):
    """
    Generate an adaptive range of window sizes (nvals) for R/S and DFA analyses.

    This function chooses either a logarithmically spaced or linear sequence of
    window sizes, ensuring at least `min_segments` segments of length `n` fit into `data`.
    If `max_n = N // min_segments` is too small (≤ min_n), it falls back to a linear range.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series. Must contain at least 2 points.
    num : int, optional (default=20)
        Number of values to sample between `min_n` and `max_n` (before uniquing). Must be ≥ 1.
    min_n : int, optional (default=8)
        Minimum window size. Must be ≥ 1.
    min_segments : int, optional (default=10)
        Minimum number of segments (⌊N / n⌋) to have for a given n. Must be ≥ 1.

    Returns
    -------
    nvals : ndarray of int
        Sorted, unique window sizes to use in R/S or DFA. Each satisfies 2 ≤ n < N.

    Raises
    ------
    ValueError
        If `data` is not 1D with at least 2 points; or if `num`, `min_n`, or `min_segments`
        are not positive integers; or if `len(data) < 2 * min_n`.

    Notes
    -----
    - If `max_n = N // min_segments ≤ min_n`, returns a linear range from `min_n` to `N//2`.
    - Otherwise, returns unique integer values sampled log‐uniformly between `min_n` and `max_n`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _generate_adaptive_nvals
    >>> ts = np.random.randn(100)
    >>> nvals = _generate_adaptive_nvals(ts, num=10, min_n=5, min_segments=8)
    >>> print(nvals)
    """
    
    data = np.asarray(data)
    _check_array_1d(data, varname='data')
    _check_positive_integer(num, varname='num', minimum=1)
    _check_positive_integer(min_n, varname='min_n', minimum=1)
    _check_positive_integer(min_segments, varname='min_segments', minimum=1)
    
    N = data.size
    if N < 2 * min_n:
        raise ValueError(
            f"`data` length ({N}) is too short for min_n={min_n}; need at least {2 * min_n} points."
        )

    max_n = max(min_n, N // min_segments)

    if max_n <= min_n:
        # Fall back to a linear range when logspace is not meaningful
        candidate = np.arange(min_n, (N // 2) + 1, dtype=int)
        nvals = candidate[candidate >= 2]
        return np.unique(nvals)

    # Otherwise, generate unique log‐spaced integers between min_n and max_n
    raw = np.logspace(np.log10(min_n), np.log10(max_n), num=num)
    nvals = np.unique(raw.astype(int))
    # Keep only valid window sizes: 2 ≤ n < N
    nvals = nvals[(nvals >= 2) & (nvals < N)]
    return np.sort(nvals)


def _generate_nvals(data, num=20, min_n=10, min_segments=10):
    """
    Generate a sequence of window sizes (`nvals`) for MF‐DFA analysis.

    Chooses at least two unique logarithmically spaced integers between `min_n` 
    and `max_n = N // min_segments`. If `max_n ≤ min_n + 1`, falls back to a 
    default linear range.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series. Must be finite, numeric, and have length ≥ min_n * min_segments.
    num : int, optional (default=20)
        Number of scales to generate between `min_n` and `max_n` (before uniquing). Must be ≥ 2.
    min_n : int, optional (default=10)
        Minimum window size. Must be ≥ 1.
    min_segments : int, optional (default=10)
        Minimum number of segments (N // n) to allow for detrending. Must be ≥ 1.

    Returns
    -------
    nvals : ndarray of int
        1D array of sorted, unique window sizes. Guaranteed to contain at least 2 values.

    Raises
    ------
    ValueError
        If `data` is not 1D, or if `num < 2`, or `min_n < 1`, or `min_segments < 1`,
        or if `len(data) < min_n * min_segments`.
    RuntimeError
        If unable to generate at least two unique window sizes (e.g., too small series).

    Notes
    -----
    - If `(N // min_segments) ≤ min_n + 1`, returns `np.arange(4, max(5, N // 2))`.
    - Ensures the final `nvals` is strictly increasing.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _generate_nvals
    >>> ts = np.random.randn(200)
    >>> nvals = _generate_nvals(ts, num=15, min_n=10, min_segments=8)
    >>> print(nvals)
    """
    
    data = np.asarray(data)
    _check_array_1d(data, varname='data')
    _check_positive_integer(num, varname='num', minimum=2)
    _check_positive_integer(min_n, varname='min_n', minimum=1)
    _check_positive_integer(min_segments, varname='min_segments', minimum=1)

    N = data.size
    if N < min_n * min_segments:
        raise ValueError(
            f"Time series length (N={N}) is too short for "
            f"min_n={min_n} and min_segments={min_segments} "
            f"(requires ≥ {min_n * min_segments} points)."
        )

    max_n = max(min_n + 1, N // min_segments)
    if max_n <= min_n + 1:
        fallback = np.arange(4, max(5, N // 2))
        if fallback.size < 2:
            raise RuntimeError(
                "Could not generate at least two unique window sizes. "
                "Try lowering 'min_segments' or 'min_n'."
            )
        return fallback.astype(int)

    # Generate logarithmically spaced values and take unique integers
    raw = np.logspace(np.log10(min_n), np.log10(max_n), num=num)
    nvals = np.unique(raw.astype(int))
    # Ensure at least 2 values
    if nvals.size < 2:
        raise RuntimeError(
            "Could not generate at least two unique window sizes. "
            "Try lowering 'min_segments' or 'min_n'."
        )
    return nvals


def _rescaled_range_analysis(data, nvals):
    """
    Perform Rescaled Range (R/S) analysis on a 1D time series.

    For each window size n in `nvals`, divides `data` into non‐overlapping segments,
    computes the cumulative deviation from the mean in each segment, calculates the
    range R = max(cumulative) − min(cumulative), rescales by the segment’s standard deviation S,
    and returns the average R/S over all segments.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series. Must be finite and have at least 2 points.
    nvals : array‐like of int
        1D array of window sizes (each ≥ 2 and < N) for which to compute R/S.

    Returns
    -------
    R_S : list of float
        List containing mean R/S for each n in `nvals`. If all segments for a given n
        have S = 0, returns `np.nan` for that n.

    Raises
    ------
    ValueError
        If `data` is not 1D or if any `n` in `nvals` is < 2 or ≥ len(data).

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _rescaled_range_analysis
    >>> ts = np.random.randn(100)
    >>> nvals = [5, 10, 20]
    >>> RS = _rescaled_range_analysis(ts, nvals)
    >>> print(RS)
    """
    
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"`data` must be 1D; got shape {data.shape}.")
    N = data.size

    nvals = np.asarray(nvals, dtype=int)
    if nvals.ndim != 1:
        raise ValueError("`nvals` must be a 1D array of integers.")
    if np.any(nvals < 2) or np.any(nvals >= N):
        raise ValueError("Each window size in `nvals` must satisfy 2 ≤ n < len(data).")
    
    R_S = []
    for n in nvals:
        segments = N // n
        if segments < 1:
            R_S.append(np.nan)
            continue

        RS_vals = []
        for i in range(segments):
            segment = data[i * n : (i + 1) * n]
            mean_seg = np.mean(segment)
            Z = np.cumsum(segment - mean_seg)
            R = np.max(Z) - np.min(Z)
            S = np.std(segment)
            if S == 0:
                # All values in this segment are identical: skip
                continue
            RS_vals.append(R / S)

        if len(RS_vals) == 0:
            R_S.append(np.nan)
        else:
            R_S.append(np.mean(RS_vals))

    return R_S


def _compute_fluctuations(data, nvals):
    """
    Compute the DFA fluctuation function F(n) for each window size in `nvals`.

    Steps:
      1. Center `data` by subtracting its mean; build the cumulative profile Y.
      2. For each n in `nvals`, divide Y into N // n non‐overlapping segments.
      3. In each segment, fit a first‐degree polynomial (trend) and compute RMS of residuals.
      4. F(n) = mean of RMS values over all segments of size n.

    Parameters
    ----------
    data : ndarray of shape (N,)
        1D numeric array (time series). Must not contain NaNs or Infs.
    nvals : ndarray of int
        1D array of window sizes for which to compute F(n). Each must satisfy 2 ≤ n < len(data).

    Returns
    -------
    F_n : ndarray of float
        1D array of mean RMS fluctuation F(n) for each n in `nvals`, in the same order.

    Raises
    ------
    ValueError
        If `data` is not 1D numeric, contains NaNs or Infs,
        or if `nvals` is not a 1D array of ints with 2 ≤ n < len(data).

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _compute_fluctuations
    >>> ts = np.random.randn(200)
    >>> nvals = np.array([5, 10, 20])
    >>> F_n = _compute_fluctuations(ts, nvals)
    >>> print(F_n)
    """
    
    # Convert to array and validate
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"`data` must be 1D; got shape {data.shape}.")
    if not np.isfinite(data).all():
        raise ValueError("`data` must not contain NaN or infinite values.")

    N = data.size
    # Validate nvals
    nvals = np.asarray(nvals, dtype=int)
    if nvals.ndim != 1:
        raise ValueError("`nvals` must be a 1D array of integers.")
    if np.any(nvals < 2) or np.any(nvals >= N):
        raise ValueError("Each window size in `nvals` must satisfy 2 ≤ n < len(data).")

    # Build the integrated profile
    data_centered = data - np.mean(data)
    Y = np.cumsum(data_centered)

    F_n_list = []
    for n in nvals:
        segments = N // n
        if segments < 1:
            # Should not happen after validation, but skip if it does
            F_n_list.append(np.nan)
            continue

        rms_vals = np.zeros(segments, dtype=float)
        x = np.arange(n)

        for i in range(segments):
            seg = Y[i * n : (i + 1) * n]
            # Fit a linear trend to this segment
            coeffs = np.polyfit(x, seg, 1)
            trend = np.polyval(coeffs, x)
            # Root‐mean‐square detrended fluctuation
            rms_vals[i] = np.sqrt(np.mean((seg - trend) ** 2))

        # Mean fluctuation over all segments of size n
        F_n_list.append(np.mean(rms_vals))

    return np.array(F_n_list, dtype=float)


def _generate_qvals(mode='typical'):
    """
    Generate a sequence of q exponents for MF‐DFA analysis.

    Parameters
    ----------
    mode : {'typical', 'reliable', 'unstable'}, optional (default='typical')
        Mode for q‐values:
          - 'typical'  → 11 values uniformly from −5 to +5.
          - 'reliable' → 11 values uniformly from −4 to +4 (less extreme).
          - 'unstable' → 11 values uniformly from −10 to +10 (more extreme, may be noisy).

    Returns
    -------
    qvals : ndarray of float
        1D array of q exponents, sorted ascending.

    Raises
    ------
    ValueError
        If `mode` is not one of the accepted strings.

    Examples
    --------
    >>> from pyenfra.utils import _generate_qvals
    >>> qtyp = _generate_qvals('typical')
    >>> print(len(qtyp))  # 11
    """
    
    return _check_q_mode(mode)


def _check_q_mode(mode):
    """
    Internal: verify and return q‐values according to `mode`. Raises ValueError if invalid.

    Parameters
    ----------
    mode : {'typical', 'reliable', 'unstable'}
        Mode for q‐values.

    Returns
    -------
    qvals : ndarray of float
        Array of q exponents.

    Raises
    ------
    ValueError
        If `mode` is not recognized.
    """
    
    if mode == 'typical':
        return np.linspace(-5, 5, 11)
    elif mode == 'reliable':
        return np.linspace(-4, 4, 11)
    elif mode == 'unstable':
        return np.linspace(-10, 10, 11)
    else:
        raise ValueError("`mode` must be 'typical', 'reliable', or 'unstable'.")
        

def _confidence_interval_rs(data, alpha):
    """
    Compute two‐sided Weron empirical confidence intervals for R/S analysis.

    In R/S analysis, H = 0.5 indicates white noise (random). These formulas
    approximate finite‐sample confidence intervals around 0.5.

    Reference
    ---------
    Weron R., 2002, “Estimating long-range dependence: finite sample properties
    and confidence intervals,” Physica A, 312(1–2), 285–299,
    DOI:10.1016/S0378-4371(02)00961-5

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series; after dropping NaNs/Infs, length must be ≥ 50.
    alpha : float or int
        Confidence level. Acceptable values for:
          - 90% level: 90, 10, 0.9, 0.1
          - 95% level: 95, 5, 0.95, 0.05
          - 99% level: 99, 1, 0.99, 0.01

    Returns
    -------
    ci : list of float
        [lower_bound, upper_bound] for H. If `alpha` not recognized or length < 50,
        raises ValueError.

    Raises
    ------
    ValueError
        If `alpha` not in the accepted list, or if after filtering out non‐finite
        values, `len(data) < 50`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _confidence_interval_rs
    >>> ts = np.random.randn(100)
    >>> ci = _confidence_interval_rs(ts, alpha=95)
    >>> print(ci)  # e.g., [0.32, 0.67]
    """
    
    data = np.asarray(data, dtype=float)
    if not np.isfinite(data).all():
        data = data[np.isfinite(data)]
    n = data.size
    if n < 50:
        raise ValueError(f"Time series too short ({n} points); need ≥ 50. If not available, use DFA instead.")

    allowed = {99, 95, 90, 10, 5, 1, 0.99, 0.95, 0.9, 0.1, 0.05, 0.01}
    if alpha not in allowed:
        raise ValueError(
            f"Wrong alpha value {alpha}. Accepted: 90/10/0.9/0.1, 95/5/0.95/0.05, or 99/1/0.99/0.01."
        )
        
    N = np.log2(n)
    if alpha in {90, 10, 0.9, 0.1}:
        lower = 0.5 - np.exp(-7.35 * np.log(np.log(N)) + 4.06)
        upper = np.exp(-7.07 * np.log(np.log(N)) + 3.75) + 0.5
    elif alpha in {95, 5, 0.95, 0.05}:
        lower = 0.5 - np.exp(-7.33 * np.log(np.log(N)) + 4.21)
        upper = np.exp(-7.20 * np.log(np.log(N)) + 4.04) + 0.5
    else:  # alpha in {99, 1, 0.99, 0.01}
        lower = 0.5 - np.exp(-7.19 * np.log(np.log(N)) + 4.34)
        upper = np.exp(-7.51 * np.log(np.log(N)) + 4.58) + 0.5

    return [lower, upper]


def _confidence_interval_dfa(data, alpha):
    """
    Compute two‐sided Weron empirical confidence intervals for DFA analysis.

    In DFA, H = 0.5 indicates white noise, H = 1 indicates 1/f (pink) noise.
    These formulas approximate finite‐sample confidence intervals around 0.5 and 1.

    Reference
    ---------
    Weron R., 2002, “Estimating long-range dependence: finite sample properties
    and confidence intervals,” Physica A, 312(1–2), 285–299,
    DOI:10.1016/S0378-4371(02)00961-5

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series; after dropping NaNs/Infs, length must be ≥ 10.
    alpha : float or int
        Confidence level. Acceptable values for:
          - 90% level: 90, 10, 0.9, 0.1
          - 95% level: 95, 5, 0.95, 0.05
          - 99% level: 99, 1, 0.99, 0.01

    Returns
    -------
    ci : list of float
        [lower_white, upper_white, lower_pink, upper_pink].

    Raises
    ------
    ValueError
        If `alpha` not in the accepted list, or if after filtering out non‐finite
        values, `len(data) < 10`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _confidence_interval_dfa
    >>> ts = np.random.randn(200)
    >>> ci = _confidence_interval_dfa(ts, alpha=95)
    >>> print(ci)  # e.g., [0.40, 0.75, 0.25, 1.25]
    """
    
    data = np.asarray(data, dtype=float)
    if not np.isfinite(data).all():
        data = data[np.isfinite(data)]
    n = data.size
    if n < 10:
        raise ValueError(f"Time series too short ({n} points); need ≥ 10.")

    allowed = {99, 95, 90, 10, 5, 1, 0.99, 0.95, 0.9, 0.1, 0.05, 0.01}
    if alpha not in allowed:
        raise ValueError(
            f"Wrong alpha value {alpha}. Accepted: 90/10/0.9/0.1, 95/5/0.95/0.05, or 99/1/0.99/0.01."
        )
        
    N = np.log2(n)
    if alpha in {90, 10, 0.9, 0.1}:
        if n < 50:
            lw = 0.5 - np.exp(-2.33 * np.log(N) + 3.09)
            uw = np.exp(-2.44 * np.log(N) + 3.13) + 0.5
            lp = 1 - np.exp(-2.33 * np.log(N) + 3.09)
            up = np.exp(-2.44 * np.log(N) + 3.13) + 1
        else:
            lw = 0.5 - np.exp(-2.99 * np.log(N) + 4.45)
            uw = np.exp(-3.09 * np.log(N) + 4.57) + 0.5
            lp = 1 - np.exp(-2.99 * np.log(N) + 4.45)
            up = np.exp(-3.09 * np.log(N) + 4.57) + 1
    elif alpha in {95, 5, 0.95, 0.05}:
        if n < 50:
            lw = 0.5 - np.exp(-2.33 * np.log(N) + 3.25)
            uw = np.exp(-2.46 * np.log(N) + 3.38) + 0.5
            lp = 1 - np.exp(-2.33 * np.log(N) + 3.25)
            up = np.exp(-2.46 * np.log(N) + 3.38) + 1
        else:
            lw = 0.5 - np.exp(-2.93 * np.log(N) + 4.45)
            uw = np.exp(-3.10 * np.log(N) + 4.77) + 0.5
            lp = 1 - np.exp(-2.93 * np.log(N) + 4.45)
            up = np.exp(-3.10 * np.log(N) + 4.77) + 1
    else:  # alpha in {99, 1, 0.99, 0.01}
        if n < 50:
            lw = 0.5 - np.exp(-2.20 * np.log(N) + 3.18)
            uw = np.exp(-2.45 * np.log(N) + 3.62) + 0.5
            lp = 1 - np.exp(-2.20 * np.log(N) + 3.18)
            up = np.exp(-2.45 * np.log(N) + 3.62) + 1
        else:
            lw = 0.5 - np.exp(-2.67 * np.log(N) + 4.06)
            uw = np.exp(-3.19 * np.log(N) + 5.28) + 0.5
            lp = 1 - np.exp(-2.67 * np.log(N) + 4.06)
            up = np.exp(-3.19 * np.log(N) + 5.28) + 1

    return [lw, uw, lp, up]


def _MF_DFA(data, nvals, qvals):
    """
    Core MF‐DFA routine: computes fluctuation functions F_q(n) for each q and n.

    This helper assumes minimal validation; use `MF_DFA(...)` (public) for the full interface.

    Definition
    ----------
    1. Build the profile Y = cumsum(data − mean(data)).
    2. For each window size n in `nvals`, divide Y into M = N//n non‐overlapping segments.
    3. In each segment, fit a first‐degree polynomial (linear detrending) and compute
       RMS of residuals: RMS_k(n) for segment k.
    4. For each q:
       - If q = 0: F_q(n) = exp(0.5 * mean(ln(RMS_k(n)²))).
       - If q ≠ 0: F_q(n) = [ mean(RMS_k(n)^q) ]^(1/q).

    Parameters
    ----------
    data : ndarray of shape (N,)
        Input 1D time series (finite numeric).
    nvals : ndarray of int
        1D array of window sizes. Each n must satisfy 2 ≤ n < N. Must have length ≥ 2.
    qvals : ndarray of float
        1D array of q exponents. Must have length ≥ 1.

    Returns
    -------
    F_q : ndarray of shape (len(qvals), len(nvals))
        Fluctuation function values. 
        F_q[i, j] = F_{q_i}(n_j). Entries where detrending fails or no valid segments remain are NaN.

    Raises
    ------
    ValueError
        If `data` is not 1D, or if `nvals` is not a 1D array of at least two ints ≥ 2,
        or if `qvals` is not a 1D array of at least one element,
        or if any n ≥ N.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _MF_DFA
    >>> ts = np.random.randn(500)
    >>> nvals = np.array([10, 20, 50])
    >>> qvals = np.linspace(-5, 5, 11)
    >>> F = _MF_DFA(ts, nvals, qvals)
    >>> print(F.shape)  # (11, 3)
    """
    
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"`data` must be 1D; got shape {data.shape}.")
    N = data.size

    nvals = np.asarray(nvals, dtype=int)
    if nvals.ndim != 1 or nvals.size < 2:
        raise ValueError("`nvals` must be a 1D array with at least two window sizes.")
    if np.any(nvals < 2) or np.any(nvals >= N):
        raise ValueError("Each window size in `nvals` must satisfy 2 ≤ n < len(data).")

    qvals = np.asarray(qvals, dtype=float)
    if qvals.ndim != 1 or qvals.size < 1:
        raise ValueError("`qvals` must be a 1D array with at least one q exponent.")
    
    # Build the profile
    Y = np.cumsum(data - np.mean(data))
    F_q = np.full((len(qvals), len(nvals)), np.nan, dtype=float)

    for i, q in enumerate(qvals):
        for j, n in enumerate(nvals):
            segments = N // n
            if segments < 1:
                # no segments possible
                continue

            rms_vals = np.empty(segments, dtype=float)
            for k in range(segments):
                idx_start = k * n
                idx_end = idx_start + n
                segment = Y[idx_start:idx_end]
                x = np.arange(n)
                # 1st‐degree polynomial fit (linear detrend)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                rms_vals[k] = np.sqrt(np.mean((segment - trend) ** 2))

            # Handle zero or negative RMS:
            if np.any(rms_vals <= 0):
                # If all are zero, F_q=0; otherwise drop zeros
                if np.all(rms_vals == 0):
                    F_q[i, j] = 0.0
                    continue
                else:
                    rms_vals = rms_vals[rms_vals > 0]
                    if rms_vals.size == 0:
                        F_q[i, j] = np.nan
                        continue

            if np.isclose(q, 0.0):
                # Logarithmic averaging
                F_q[i, j] = np.exp(0.5 * np.mean(np.log(rms_vals ** 2)))
            else:
                F_q[i, j] = (np.mean(rms_vals ** q)) ** (1.0 / q)

    return F_q    


def _phase_space_embedding(data, dim, tau):
    """
    Embed a 1D time series into a `dim`‐dimensional phase space with delay `tau`.

    Given a series {x₀, x₁, …, x_{N−1}}, the embedding constructs points:
      X[i] = [ x_i, x_{i+tau}, x_{i+2*tau}, …, x_{i+(dim−1)*tau} ] 
    for i = 0 … (N − (dim−1)*tau − 1). The result has shape (M, dim), where M = N − (dim−1)*tau.

    Parameters
    ----------
    data : array‐like of shape (N,)
        Input 1D time series. Must be finite and length ≥ 2.
    dim : int
        Embedding dimension. Must be ≥ 2.
    tau : int
        Time delay (lag). Must be ≥ 1.

    Returns
    -------
    embedded : ndarray of shape (M, dim)
        Embedded trajectory, where M = N − (dim−1)*tau.

    Raises
    ------
    ValueError
        If `data` is not 1D, or `dim < 2`, or `tau < 1`, or if N − (dim−1)*tau < 2.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _phase_space_embedding
    >>> ts = np.sin(np.linspace(0, 10*np.pi, 500))
    >>> X = _phase_space_embedding(ts, dim=3, tau=5)
    >>> print(X.shape)  # (500 - 2*5, 3) = (490, 3)
    """
    
    data = np.asarray(data, dtype=float)
    _check_array_1d(data, varname='data')
    _check_positive_integer(dim, varname='dim', minimum=2)
    _check_positive_integer(tau, varname='tau', minimum=1)
    
    N = data.shape[0]
    M = N - (dim - 1) * tau
    if M < 2:
        raise ValueError(f"Cannot embed: with N={N}, dim={dim}, tau={tau}, "
                         f"N - (dim-1)*tau = {M} < 2.")
    
    embedded = np.empty((M, dim), dtype=float)
    for i in range(dim):
        embedded[:, i] = data[i * tau : i * tau + M]
    return embedded


def _generate_wtmm_scales(data, min_scale, max_fraction, num):
    """
    Generate a set of integer scales for WTMM analysis using geometric spacing.

    Parameters
    ----------
    data : array‐like of shape (N,)
        Input 1D time series. Used only to determine maximum scale (N * max_fraction).
    min_scale : int
        Minimum scale to include (must be ≥ 1).
    max_fraction : float
        Fraction of data length to use as maximum scale. Must be in (0, 1].
    num : int
        Number of scales to generate in geometric progression. Must be ≥ 1.

    Returns
    -------
    scales : ndarray of int
        Sorted, unique integer scales between `min_scale` and floor(N * max_fraction).

    Raises
    ------
    ValueError
        If `data` is empty, or if `min_scale < 1`, or if `max_fraction ≤ 0` or > 1,
        or if `num < 1`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _generate_wtmm_scales
    >>> ts = np.random.randn(200)
    >>> scales = _generate_wtmm_scales(ts, min_scale=2, max_fraction=0.3, num=10)
    >>> print(scales)
    """
    
    if data is None or len(data) == 0:
        raise ValueError("`data` must be a non‐empty array‐like sequence.")
    try:
        N = len(data)
    except Exception:
        raise ValueError("`data` must be array‐like with a defined length.")

    if not isinstance(min_scale, int) or min_scale < 1:
        raise ValueError(f"`min_scale` must be an integer ≥ 1; got {min_scale!r}.")
    if not isinstance(max_fraction, (float, int)) or not (0 < max_fraction <= 1):
        raise ValueError(f"`max_fraction` must be in (0, 1]; got {max_fraction!r}.")
    if not isinstance(num, int) or num < 1:
        raise ValueError(f"`num` must be an integer ≥ 1; got {num!r}.")

    max_scale = int(np.floor(N * max_fraction))
    if max_scale < min_scale:
        max_scale = min_scale

    raw_scales = np.geomspace(min_scale, max_scale, num=num)
    int_scales = np.unique(raw_scales.astype(int))
    return int_scales


def _WTMM(data, modulus, wavelet, scales=None, min_scale=2, max_fraction=0.25, num=50):
    """
    Compute the Wavelet Transform Modulus Maxima (WTMM) for a 1D signal.

    This function:
      1. Validates inputs and, if `scales` is None, generates scales via `_generate_wtmm_scales`.
      2. Performs a Continuous Wavelet Transform (CWT) on `data` over `scales`.
      3. Collapses the absolute coefficients at each scale according to `modulus`:
         - 'mean': average of |coefficients| over time,
         - 'norm': L2 norm across time,
         - 'max' : maximum |coefficients| over time.
      4. Returns `(scales, modulus_maxima)`.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series. Must not be empty and must be 1D.
    modulus : {'mean', 'norm', 'max'}
        Method to collapse wavelet coefficients at each scale.
    wavelet : str or pywt.ContinuousWavelet
        Wavelet to use for CWT.
    scales : array‐like of int, optional (default=None)
        Pre‐computed integer scales to use. If None, generated from `data`, `min_scale`,
        `max_fraction`, and `num`.
    min_scale : int, optional (default=2)
        Minimum scale for generation (must be ≥ 1).
    max_fraction : float, optional (default=0.25)
        Fraction of `len(data)` for maximum scale. Must be in (0, 1].
    num : int, optional (default=50)
        Number of scales to generate if `scales` is None. Must be ≥ 1.

    Returns
    -------
    scales : ndarray of int
        The integer scales used for the CWT (sorted ascending).
    modulus_maxima : ndarray of float
        The collapsed modulus maxima (one value per scale), according to `modulus`.

    Raises
    ------
    ValueError
        If `data` is None or empty, or not 1D;
        or if `modulus` not in {'mean','norm','max'};
        or if `scales` provided but not 1D, or contains non‐positive values;
        or if no valid scales are generated;
        or if the CWT computation fails (invalid `wavelet`).
    RuntimeError
        If the computed `modulus_maxima` contains negative values (should not occur).

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _WTMM
    >>> ts = np.random.randn(256)
    >>> scales, modmax = _WTMM(ts, modulus='mean', wavelet='cmor1-1.5', min_scale=2, max_fraction=0.3, num=20)
    >>> print(scales.shape, modmax.shape)
    """
    
    if data is None or len(data) == 0:
        raise ValueError("`data` must be a non‐empty array‐like sequence.")
    data_arr = np.asarray(data)
    if data_arr.ndim != 1:
        raise ValueError(f"`data` must be 1D; got shape {data_arr.shape}.")

    valid_moduli = {'mean', 'norm', 'max'}
    if modulus not in valid_moduli:
        raise ValueError(f"Invalid `modulus`: {modulus!r}. Must be one of {valid_moduli}.")

    if scales is None:
        scales = _generate_wtmm_scales(data_arr, min_scale, max_fraction, num)
    else:
        scales = np.asarray(scales, dtype=int)
        if scales.ndim != 1:
            raise ValueError("`scales` must be a 1D array of positive integers.")
        if np.any(scales < 1):
            raise ValueError("All `scales` must be ≥ 1.")

    if scales.size == 0:
        raise ValueError(
            "No valid scales generated. Check `min_scale`, `max_fraction`, and data length."
        )

    # Perform CWT
    try:
        coefficients, freqs = pywt.cwt(data_arr, scales, wavelet)
    except Exception as exc:
        raise ValueError(f"CWT computation failed for wavelet {wavelet!r}: {exc}")

    abs_coeff = np.abs(coefficients)  # shape: (len(scales), len(data))
    if modulus == 'mean':
        modulus_maxima = np.mean(abs_coeff, axis=1)
    elif modulus == 'norm':
        modulus_maxima = np.linalg.norm(coefficients, axis=1)
    else:  # modulus == 'max'
        modulus_maxima = np.max(abs_coeff, axis=1)

    if np.any(modulus_maxima < 0):
        raise RuntimeError(
            "Computed `modulus_maxima` contains negative values; this is unexpected."
        )

    return scales, modulus_maxima


def _compute_log_fit(scales, modulus_maxima, fit_range=None):
    """
    Compute linear regression of log(modulus_maxima) vs. log(scales), returning slope, intercept, and R².

    Steps:
      1. Convert `scales` and `modulus_maxima` to 1D arrays. Check that lengths match and `scales > 0`.
      2. Replace any `modulus_maxima ≤ 0` with NaN, so they get masked.
      3. If `fit_range` is provided as (min_scale, max_scale), mask out points outside that range.
      4. Fit a 1D line: log(modulus_maxima) = slope * log(scales) + intercept.
      5. Compute R² over the used points.

    Parameters
    ----------
    scales : array‐like of shape (L,)
        1D array of positive scales.
    modulus_maxima : array‐like of shape (L,)
        1D array of positive modulus maxima.
    fit_range : tuple of two floats (min_scale, max_scale), optional
        Inclusive range of scales to use. If None, use all valid points.

    Returns
    -------
    result : dict
        {
          'slope'      : float, estimated slope,
          'intercept'  : float, estimated intercept,
          'r_squared'  : float, coefficient of determination,
          'num_points' : int, number of points used in fit
        }

    Raises
    ------
    ValueError
        If `scales` or `modulus_maxima` are not 1D of equal length,
        or if any `scales ≤ 0`, or if fewer than two valid points remain after masking.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _compute_log_fit
    >>> scales = np.array([2, 4, 8, 16])
    >>> modmax = np.array([1.0, 2.0, 4.0, 8.0])
    >>> res = _compute_log_fit(scales, modmax)
    >>> print(res['slope'], res['r_squared'])
    """
    
    scales = np.asarray(scales, dtype=float)
    modmax = np.asarray(modulus_maxima, dtype=float)
    if scales.ndim != 1 or modmax.ndim != 1:
        raise ValueError("`scales` and `modulus_maxima` must be 1D arrays.")
    if scales.shape[0] != modmax.shape[0]:
        raise ValueError(
            f"`scales` length {scales.shape[0]} and "
            f"`modulus_maxima` length {modmax.shape[0]} must match."
        )
    if np.any(scales <= 0):
        raise ValueError("All `scales` must be > 0 for log transformation.")

    # Mask out non‐positive modulus_maxima
    modmax = np.where(modmax <= 0, np.nan, modmax)

    log_scales = np.log(scales)
    log_modmax = np.log(modmax)

    if fit_range is not None:
        if (not isinstance(fit_range, (tuple, list))) or len(fit_range) != 2:
            raise ValueError("`fit_range` must be a tuple/list of length 2.")
        min_r, max_r = fit_range
        mask = (scales >= min_r) & (scales <= max_r) & (~np.isnan(log_modmax))
    else:
        mask = ~np.isnan(log_modmax)

    if np.count_nonzero(mask) < 2:
        raise ValueError("At least two valid data points are required for linear regression.")

    x = log_scales[mask]
    y = log_modmax[mask]

    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'num_points': int(np.count_nonzero(mask))
    }


def _validate_binary_sequence(sequence, arg_name="sequence"):
    """
    Internal helper: Ensure that `sequence` is a 1D binary array (containing only 0s and 1s).

    Parameters
    ----------
    sequence : ndarray
        The array to validate.
    arg_name : str, optional (default="sequence")
        Name of the argument (for error messages).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `sequence` is not a 1D array or contains values other than 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _validate_binary_sequence
    >>> _validate_binary_sequence(np.array([0, 1, 1, 0]), arg_name='col')
    >>> _validate_binary_sequence([1, 2, 3], arg_name='col')  # raises ValueError
    """
    
    if not isinstance(sequence, np.ndarray):
        raise ValueError(f"`{arg_name}` must be a numpy array, got {type(sequence)}.")
    if sequence.ndim != 1:
        raise ValueError(f"`{arg_name}` must be 1D; got shape {sequence.shape}.")
    unique_vals = np.unique(sequence)
    if not set(unique_vals).issubset({0, 1}):
        raise ValueError(f"`{arg_name}` must be binary (0/1), but contains {unique_vals}.")
        
        
def _get_line_lengths(binary_array, min_length=2):
    """
    Compute the lengths of contiguous runs of 1s in a 1D binary sequence.

    Uses connected‐component labeling to identify runs of 1s, then returns
    only those runs whose length ≥ `min_length`.

    Parameters
    ----------
    binary_array : ndarray of shape (L,)
        1D array of zeros and ones.
    min_length : int, optional (default=2)
        Minimum run length to keep. Runs shorter than this are discarded.

    Returns
    -------
    List[int]
        List of all contiguous‐1s run lengths (each ≥ `min_length`).

    Raises
    ------
    ValueError
        If `binary_array` is not a 1D numpy array or contains values other than 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _get_line_lengths
    >>> arr = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1])
    >>> print(_get_line_lengths(arr, min_length=2))  # [2, 3]
    """
    
    _validate_binary_sequence(binary_array, arg_name="binary_array")

    # Label connected components of 1s
    labels, num_features = label(binary_array)
    if num_features == 0:
        return []

    # Compute length of each labeled component
    line_lengths = []
    for feature_idx in range(1, num_features + 1):
        length_i = int(np.sum(labels == feature_idx))
        if length_i >= min_length:
            line_lengths.append(length_i)
    return line_lengths


def _count_diagonal_lines(rp, min_length=2):
    """
    Count lengths of diagonal lines (runs of 1s along all diagonals) in a square recurrence plot.

    Loops over all diagonals (offsets from −(N−1) to +(N−1)) of `rp` and
    collects lengths of contiguous 1s runs ≥ `min_length`.

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D square binary recurrence plot (each element is 0 or 1).
    min_length : int, optional (default=2)
        Minimum diagonal run length to keep. Runs shorter than `min_length` are ignored.

    Returns
    -------
    List[int]
        List of all diagonal run lengths (each ≥ `min_length`).

    Raises
    ------
    ValueError
        If `rp` is not a 2D square numpy array or contains values other than 0 or 1.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _count_diagonal_lines
    >>> rp = np.array([[1,1,0],[1,1,1],[0,1,1]])
    >>> print(_count_diagonal_lines(rp, min_length=2))  # [3]
    """
    
    if not isinstance(rp, np.ndarray):
        raise ValueError(f"`rp` must be a numpy array, got {type(rp)}.")
    if rp.ndim != 2 or rp.shape[0] != rp.shape[1]:
        raise ValueError(f"`rp` must be a 2D square array, got shape {rp.shape}.")
    unique_vals = np.unique(rp)
    if not set(unique_vals).issubset({0, 1}):
        raise ValueError(f"`rp` must be binary (0/1), but contains {unique_vals}.")

    N = rp.shape[0]
    diag_lengths = []
    for offset in range(-N + 1, N):
        diag = np.diagonal(rp, offset=offset)
        try:
            lengths = _get_line_lengths(diag, min_length=min_length)
        except ValueError as e:
            warnings.warn(f"Skipping diagonal offset={offset}: {e}")
            continue
        diag_lengths.extend(lengths)
    return diag_lengths


def _count_vertical_lines(rp, min_length=2):
    """
    Count lengths of vertical lines (runs of 1s in each column) in a square recurrence plot.

    Iterates over each column of `rp` and collects contiguous runs of 1s ≥ `min_length`.

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D square binary recurrence plot (each element is 0 or 1).
    min_length : int, optional (default=2)
        Minimum vertical run length to keep. Runs shorter than `min_length` are ignored.

    Returns
    -------
    List[int]
        List of all vertical run lengths (each ≥ `min_length`).

    Raises
    ------
    ValueError
        If `rp` is not a 2D square numpy array or contains values outside {0, 1}.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _count_vertical_lines
    >>> rp = np.array([[1,1],[1,0]])
    >>> print(_count_vertical_lines(rp, min_length=2))  # [2]
    """
    
    if not isinstance(rp, np.ndarray):
        raise ValueError(f"`rp` must be a numpy array, got {type(rp)}.")
    if rp.ndim != 2 or rp.shape[0] != rp.shape[1]:
        raise ValueError(f"`rp` must be a 2D square array, got shape {rp.shape}.")
    unique_vals = np.unique(rp)
    if not set(unique_vals).issubset({0, 1}):
        raise ValueError(f"`rp` must be binary (0/1), but contains {unique_vals}.")

    vertical_lengths = []
    for col_idx in range(rp.shape[1]):
        col = rp[:, col_idx]
        try:
            lengths = _get_line_lengths(col, min_length=min_length)
        except ValueError as e:
            warnings.warn(f"Skipping column index={col_idx}: {e}")
            continue
        vertical_lengths.extend(lengths)
    return vertical_lengths


def _generate_recurrence_plot(series, threshold="point", percentage=20.0):
    """
    Compute a binary recurrence plot (RP) from a 1D time series.

    Wraps PyTS’s RecurrencePlot:
      1. Validates `series` is 1D with at least 2 points.
      2. Uses `threshold` and `percentage` to configure the transformer.
      3. Returns a binary (0/1) square matrix of shape (N, N).

    Parameters
    ----------
    series : array‐like of shape (N,)
        1D time series (numeric). Must have at least 2 time points.
    threshold : {'point'} or float, optional (default="point")
        Strategy for thresholding. If "point", the distance threshold is chosen so
        that `percentage` of neighbors are recurrent. If a float, that value is used
        as the fixed threshold (library‐dependent behavior).
    percentage : float, optional (default=20.0)
        Percentage of nearest neighbors to consider recurrent when `threshold="point"`.
        Must satisfy 0 < percentage < 100.

    Returns
    -------
    rp_bin : ndarray of shape (N, N)
        Binary recurrence plot (0/1).

    Raises
    ------
    ValueError
        If `series` is not 1D or has < 2 points, or if `percentage` not in (0, 100),
        or if `threshold` is neither "point" nor numeric.
    RuntimeError
        If the RecurrencePlot transformer fails internally.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _generate_recurrence_plot
    >>> ts = np.sin(np.linspace(0, 10, 100))
    >>> rp = _generate_recurrence_plot(ts, threshold='point', percentage=15.0)
    >>> print(rp.shape)  # (100, 100)
    """
    
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"`series` must be 1D; got shape {arr.shape}.")
    if arr.size < 2:
        raise ValueError("`series` must contain at least 2 time points.")
    if not (0.0 < percentage < 100.0):
        raise ValueError("`percentage` must be in (0, 100).")
    if not (threshold == "point" or isinstance(threshold, (int, float))):
        raise ValueError("`threshold` must be 'point' or a numeric value.")

    series_2d = arr.reshape(1, -1)
    try:
        rp_transformer = RecurrencePlot(threshold=threshold, percentage=percentage)
        rp = rp_transformer.fit_transform(series_2d)[0]
    except Exception as e:
        raise RuntimeError(f"Error computing RecurrencePlot: {e}")

    rp_bin = (rp > 0).astype(int)
    return rp_bin


def _compute_diagonal_line_lengths(rp, min_length=2):
    """
    Compute all diagonal line lengths of contiguous 1s in a recurrence plot.

    Wrapper around `_count_diagonal_lines` for clarity.

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D binary recurrence plot (0/1).
    min_length : int, optional (default=2)
        Minimum diagonal length to include.

    Returns
    -------
    List[int]
        List of diagonal line lengths ≥ `min_length`.

    Raises
    ------
    ValueError
        If `rp` is not a 2D square binary array.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _compute_diagonal_line_lengths
    >>> rp = np.array([[1,1,0],[1,1,1],[0,1,1]])
    >>> print(_compute_diagonal_line_lengths(rp, min_length=2))  # [3]
    """
    
    return _count_diagonal_lines(rp, min_length=min_length)


def _compute_vertical_line_lengths(rp, min_length=2):
    """
    Compute all vertical line lengths of contiguous 1s in a recurrence plot.

    Wrapper around `_count_vertical_lines`.

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D binary recurrence plot (0/1).
    min_length : int, optional (default=2)
        Minimum vertical length to include.

    Returns
    -------
    List[int]
        List of vertical line lengths ≥ `min_length`.

    Raises
    ------
    ValueError
        If `rp` is not a 2D square binary array.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.utils import _compute_vertical_line_lengths
    >>> rp = np.array([[1,1],[1,0]])
    >>> print(_compute_vertical_line_lengths(rp, min_length=1))  # [2, 1]
    """
    
    return _count_vertical_lines(rp, min_length=min_length)