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
from . import utils


__all__ = ['hurst', 'DFA', 'MF_DFA', 'lyapunov', 'WTMM', 'RQA', 'recurrence_rate',
           'determinism', 'l_max', 'entropy', 'divergence', 'laminarity',
           'trapping_time']


def hurst(data, num=20, min_n=8, min_segments=10):
    """
    Estimate the Hurst exponent of a time series using Rescaled Range (R/S) analysis.

    The Hurst exponent (H) measures the long‐term memory of a time series.
    Values of H > 0.5 indicate persistent behavior; H < 0.5 indicates
    anti‐persistence; H ≈ 0.5 implies a random walk.

    Reference
    ---------
    Hurst H.E., 1951, Long‐Term Storage Capacity of Reservoirs,
    Transactions of the American Society of Civil Engineers, 116, 1,
    770–799, DOI: 10.1061/TACEAT.0006518

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series to analyze. Must be finite and non‐constant.
    num : int, optional (default=20)
        Number of scale values (n) to generate.
    min_n : int, optional (default=8)
        Minimum window size to consider.
    min_segments : int, optional (default=10)
        Minimum number of segments (N // n) required for statistical validity.

    Returns
    -------
    H_rs : float
        Estimated Hurst exponent from the slope of log(R/S) versus log(n).

    Raises
    ------
    ValueError
        If `data` is not 1D, contains non‐finite values, is constant,
        or too short (length < max(50, min_segments*min_n)),
        or if insufficient valid scales remain.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import hurst
    >>> ts = np.random.randn(1000)
    >>> H = hurst(ts, num=20, min_n=8, min_segments=10)
    >>> print(f"Hurst exponent ≈ {H:.3f}")
    """
    
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("`data` must be a one‐dimensional array.")
    if not np.isfinite(data).all():
        raise ValueError("`data` must not contain NaN or infinite values.")
    if np.std(data) == 0:
        raise ValueError("`data` is constant; no fluctuation to analyze.")
    N = data.size
    if N < 50:
        raise ValueError(
            f"Time series too short ({N} points); need > 50. "
            "Use DFA for shorter series."
        )
    if N < min_segments * min_n:
        raise ValueError(
            f"Time series too short ({N} points) for "
            f"min_segments={min_segments} and min_n={min_n}."
        )

    # Generate scale values
    nvals = utils._generate_adaptive_nvals(data, num, min_n, min_segments)
    # Compute R/S
    R_S = np.array(utils._rescaled_range_analysis(data, nvals))

    # Filter out NaNs and non‐positive
    valid_mask = np.isfinite(R_S) & (R_S > 0)
    tnvals = nvals[valid_mask]
    tR_S = R_S[valid_mask]

    if tnvals.size < 5:
        raise ValueError(
            f"Insufficient valid scales ({tnvals.size}) for reliable Hurst estimation."
        )

    # Fit slope on log‐log
    log_n = np.log(tnvals)
    log_RS = np.log(tR_S)
    coeffs = np.polyfit(log_n, log_RS, 1)
    H_rs = float(coeffs[0])

    return H_rs


def DFA(data, num=20, min_n=8, min_segments=10):
    """
    Compute the Detrended Fluctuation Analysis (DFA) scaling exponent α.

    DFA quantifies long‐range correlations in a 1D time series. Steps:
      1. Integrate the mean‐centered series to obtain a profile.
      2. Window the profile into segments of length n.
      3. In each segment, fit and subtract a polynomial trend and compute RMS.
      4. Compute F(n) = sqrt(mean(square‐residuals)) for each n.
      5. Fit a line to log(F(n)) vs. log(n); the slope is α.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series to analyze. Must be finite and have non‐zero variance.
    num : int, optional (default=20)
        Number of scale values (n) to generate.
    min_n : int, optional (default=8)
        Minimum window size to consider.
    min_segments : int, optional (default=10)
        Minimum number of segments (N // n) required for validity.

    Returns
    -------
    H_dfa : float
        DFA scaling exponent α (slope of log(F(n)) vs. log(n)).

    Raises
    ------
    ValueError
        If `data` is not 1D, contains non‐finite values, is constant,
        too short (N < 2*min_n), or if fewer than two valid (n, F(n))
        pairs remain for regression.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import DFA
    >>> ts = np.sin(np.linspace(0, 10, 500)) + 0.1*np.random.randn(500)
    >>> alpha = DFA(ts, num=25, min_n=10, min_segments=8)
    >>> print(f"DFA exponent ≈ {alpha:.3f}")
    """
    
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("`data` must be a one‐dimensional array.")
    if not np.isfinite(data).all():
        raise ValueError("`data` must not contain NaN or infinite values.")
    if np.std(data) == 0:
        raise ValueError("`data` is constant; no fluctuation to analyze.")
    N = data.size
    if N < 2 * min_n:
        raise ValueError(
            f"Data length ({N}) too short for min_n={min_n}."
        )

    # Generate scale values
    nvals = utils._generate_adaptive_nvals(data, num, min_n, min_segments)

    # Compute fluctuation F(n)
    F_n = utils._compute_fluctuations(data, nvals)
    valid_mask = np.isfinite(F_n) & (F_n > 0)
    t_nvals = nvals[valid_mask]
    t_F_n = F_n[valid_mask]

    if t_nvals.size < 2:
        raise ValueError(
            "Not enough valid (n, F(n)) pairs to perform regression. "
            "Check if data is too short or constant."
        )

    # Fit slope on log‐log
    log_n = np.log(t_nvals)
    log_F = np.log(t_F_n)
    coeffs = np.polyfit(log_n, log_F, 1)
    H_dfa = float(coeffs[0])

    return H_dfa


def MF_DFA(data, q_mode='typical', num=20, min_n=10, min_segments=10, return_nvals=False):
    """
    Perform Multifractal Detrended Fluctuation Analysis (MF‐DFA).

    MF‐DFA generalizes DFA by computing fluctuation functions F_q(n)
    for a range of q‐exponents and then estimating generalized H(q).

    Definition
    ----------
    1. For each scale n in nvals:
         a. Divide the profile into ⌊N/n⌋ non‐overlapping segments.
         b. Fit a polynomial trend in each segment and compute F_j(n).
         c. Compute F_q(n) = [ (1/(2M)) Σ_{j=1}^{2M} (F_j(n))^q ]^{1/q}.
    2. For each q, fit a line to log(F_q(n)) vs. log(n); slope = H(q).

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series. Must be finite, non‐constant, and length ≥ min_n*min_segments.
    q_mode : {'typical', 'reliable', 'unstable'}, optional (default='typical')
        Mode for q‐value generation. Passed to `utils._generate_qvals`.
    num : int, optional (default=20)
        Number of scales to sample when generating nvals.
    min_n : int, optional (default=10)
        Minimum window size.
    min_segments : int, optional (default=10)
        Minimum number of segments (N // n).
    return_nvals : bool, optional (default=False)
        If True, return the array of scales (nvals) used.

    Returns
    -------
    H_q : ndarray of float, shape (len(qvals),)
        Generalized Hurst exponents for each q (sorted ascending). NaN if fit fails.
    qvals : ndarray of float, shape (len(qvals),)
        The q‐exponents used, sorted ascending.
    nvals : ndarray of int, shape (len(nvals),), optional
        The scales used. Only returned if `return_nvals=True`.

    Raises
    ------
    ValueError
        If `data` is not 1D, contains non‐finite values, is constant,
        length < min_n*min_segments, or if `q_mode` is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import MF_DFA
    >>> ts = np.sin(np.linspace(0, 20*np.pi, 2000)) + 0.1*np.random.randn(2000)
    >>> H_q, qvals, nvals = MF_DFA(
    ...     ts, q_mode='typical', num=25, min_n=20, min_segments=8, return_nvals=True
    ... )
    >>> print("q‐values:", qvals)
    >>> print("Generalized Hurst exponents:", H_q)
    """
    
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("`data` must be a one‐dimensional array.")
    if not np.isfinite(data).all():
        raise ValueError("`data` must not contain NaN or infinite values.")
    if np.std(data) == 0:
        raise ValueError("`data` is constant; no fluctuation to analyze.")
    N = data.size
    if N < min_n * min_segments:
        raise ValueError(
            f"Data length ({N}) too short for min_n={min_n} and min_segments={min_segments}."
        )

    # Validate q_mode
    valid_q_modes = {'typical', 'reliable', 'unstable'}
    if q_mode not in valid_q_modes:
        raise ValueError(
            f"`q_mode` must be one of {valid_q_modes}, got '{q_mode}'."
        )

    qvals = utils._generate_qvals(q_mode)
    nvals = utils._generate_nvals(data, num=num, min_n=min_n, min_segments=min_segments)

    # Compute F_q(n) matrix: shape (len(qvals), len(nvals))
    F_q = utils._MF_DFA(data, nvals, qvals)

    H_q = np.full_like(qvals, np.nan, dtype=float)
    # For each q, fit slope on log‐log
    for i, q in enumerate(qvals):
        F_row = F_q[i, :]
        valid_mask = ~np.isnan(F_row) & (F_row > 0)
        if np.sum(valid_mask) >= 2:
            x = np.log(nvals[valid_mask])
            y = np.log(F_row[valid_mask])
            coeffs = np.polyfit(x, y, 1)
            H_q[i] = float(coeffs[0])
        else:
            H_q[i] = np.nan

    if return_nvals:
        return H_q, qvals, nvals
    else:
        return H_q, qvals
    
    
def lyapunov(data, dim=3, tau=1, fs=1.0, max_iter=None, theiler=1):
    """
    Estimate the largest Lyapunov exponent of a scalar time series using Rosenstein’s method.

    Steps:
      1. Embed the series in a dim‐dimensional phase space with delay τ.
      2. For each point i, find its nearest neighbor j such that |i − j| > theiler.
      3. Track the divergence d_i(k) = ||X[i+k] − X[j+k]|| for k = 0..max_iter−1.
      4. Compute average log‐divergence: ⟨ln d(k)⟩ over all i.
      5. Fit a straight line to ⟨ln d(k)⟩ vs. time t = k/fs over the initial portion (≈ 1/3)
         to estimate λ_max.

    Definition
    ----------
    λ_max = lim_{k → 0} slope of ⟨ln d(k)⟩ vs. t

    Other notes
    -----------
    This implementation uses a brute‐force nearest‐neighbor search (O(N²)).  For
    large time series, consider using a KD‐tree or ball‐tree for efficiency.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series. Must be finite and have non‐zero variance.
    dim : int, optional (default=3)
        Embedding dimension. Must be ≥ 2.
    tau : int, optional (default=1)
        Time delay for embedding. Must be ≥ 1.
    fs : float, optional (default=1.0)
        Sampling frequency (samples per unit time). Must be > 0.
    max_iter : int or None, optional (default=None)
        Number of time‐steps over which to average divergence. If None, uses ⌊M/10⌋,
        where M = N − (dim−1)*tau. Must be ≥ 1 if specified.
    theiler : int, optional (default=1)
        Minimum temporal separation (in samples) to avoid trivial neighbors. Must be ≥ 0.

    Returns
    -------
    lyap_exp : float
        Estimated largest Lyapunov exponent (units: 1/time). NaN if fit fails.
    divergence : ndarray of shape (L,)
        Average log‐divergence ⟨ln d(k)⟩ for k = 0..L−1, where L = max_iter.
    times : ndarray of shape (L,)
        Corresponding times in the same units as 1/fs.

    Raises
    ------
    ValueError
        If `data` is not 1D, contains non‐finite values, is constant,
        if `dim` < 2, `tau` < 1, `fs` ≤ 0, `theiler` < 0,
        or if N − (dim−1)*tau < 2.
    Warnings
    --------
    - If fewer than two valid divergence points are found for fitting,
      a warning is issued and `lyap_exp` = NaN.
    - Assumes noise‐free data; pre‐filter noisy data as needed.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import lyapunov
    >>> ts = np.sin(np.linspace(0, 50*np.pi, 5000))  # e.g., Lorenz x‐component
    >>> λ, div, t = lyapunov(ts, dim=5, tau=10, fs=100.0)
    >>> print(f"Largest Lyapunov exponent ≈ {λ:.4f} per unit time")
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("`data` must be a one‐dimensional array.")
    if not np.isfinite(data).all():
        raise ValueError("`data` must not contain NaN or infinite values.")
    if np.std(data) == 0:
        raise ValueError("`data` is constant; no divergence to measure.")

    utils._check_positive_integer(dim, varname='dim', minimum=2)
    utils._check_positive_integer(tau, varname='tau', minimum=1)
    if fs <= 0:
        raise ValueError("`fs` (sampling frequency) must be > 0.")
    if theiler < 0:
        raise ValueError("`theiler` must be ≥ 0.")

    N = data.size
    M = N - (dim - 1) * tau
    if M < 2:
        raise ValueError(
            f"Time series too short to embed with dim={dim}, tau={tau}."
        )

    X = utils._phase_space_embedding(data, dim=dim, tau=tau)  # shape: (M, dim)

    # Determine max_iter
    if max_iter is None:
        max_iter = max(1, M // 10)
    else:
        utils._check_positive_integer(max_iter, varname='max_iter', minimum=1)
        max_iter = min(max_iter, M - 1)

    # Step 2: Find nearest neighbor with Theiler exclusion
    nn_indices = np.full(M, -1, dtype=int)
    for i in range(M):
        xi = X[i]
        excluded = np.arange(max(0, i - theiler), min(M, i + theiler + 1))
        neighbors = np.delete(np.arange(M), excluded)
        if neighbors.size == 0:
            continue
        diffs = X[neighbors] - xi  # shape: (len(neighbors), dim)
        dists = np.sum(diffs**2, axis=1)
        j_min = np.argmin(dists)
        nn_indices[i] = neighbors[j_min]

    # Step 3: Track divergence
    divergences = np.zeros((M, max_iter), dtype=float)
    valid_counts = np.zeros(max_iter, dtype=int)

    for i in range(M):
        j = nn_indices[i]
        if j < 0:
            continue
        end = min(M - i, M - j, max_iter)
        if end < 1:
            continue
        for k in range(end):
            diff_vec = X[i + k] - X[j + k]
            dist = np.linalg.norm(diff_vec)
            # If dist == 0 → log(0) = -inf; we’ll filter later
            divergences[i, k] = np.log(dist) if dist > 0 else -np.inf
            valid_counts[k] += 1

    # Average log‐divergence across all i
    avg_log_div = np.full(max_iter, np.nan, dtype=float)
    for k in range(max_iter):
        # Select only finite log‐distances
        vals = divergences[:, k]
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size > 0:
            avg_log_div[k] = np.mean(finite_vals)
        else:
            avg_log_div[k] = np.nan

    valid_mask = (~np.isnan(avg_log_div)) & (np.isfinite(avg_log_div))
    if np.sum(valid_mask) < 2:
        warnings.warn(
            "Not enough valid divergence points to estimate Lyapunov exponent; returning NaN."
        )
        times_full = np.arange(max_iter) / fs
        return np.nan, avg_log_div, times_full

    idxs = np.where(valid_mask)[0]
    cutoff = max(2, int(len(idxs) / 3))
    fit_idxs = idxs[:cutoff]
    times = fit_idxs / fs
    y = avg_log_div[fit_idxs]
    x = times
    coeffs = np.polyfit(x, y, 1)
    lyap_exp = float(coeffs[0])

    full_times = np.arange(max_iter) / fs
    return lyap_exp, avg_log_div, full_times


def WTMM(data, modulus='mean', wavelet='cmor1-1.5', log_fit_range=None, min_scale=2,
         max_fraction=0.25, num=50):
    """
    Estimate the WTMM scaling exponent (slope) from a 1D time series.

    This function:
      1. Performs a Continuous Wavelet Transform (CWT) to obtain wavelet coefficients
         across scales.
      2. Extracts modulus maxima (ridge lines) at each scale.
      3. Collapses the modulus maxima via `modulus` (e.g., mean across all positions).
      4. Fits a line to log(modulus_maxima) vs. log(scale) over the specified range.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series. Must be finite and non‐constant.
    modulus : {'mean', 'norm', 'max'}, optional (default='mean')
        Strategy to collapse wavelet coefficients at each scale:
          - 'mean': average magnitude of maxima,
          - 'norm': Euclidean norm of maxima,
          - 'max' : maximum magnitude.
    wavelet : str or pywt.ContinuousWavelet, optional (default='cmor1-1.5')
        Wavelet to use for the CWT.
    log_fit_range : tuple of two floats (min_scale, max_scale), optional
        Inclusive range of scales (not log‐scales) over which to perform the fit.
        If None, uses all scales returned by `_WTMM`.
    min_scale : int, optional (default=2)
        Minimum scale for WTMM. Must be ≥ 1.
    max_fraction : float, optional (default=0.25)
        Maximum scale as a fraction of len(data). Must satisfy 0 < max_fraction ≤ 1.
    num : int, optional (default=50)
        Number of scales to generate for WTMM. Must be ≥ 1.

    Returns
    -------
    slope : float
        Estimated slope of log(modulus_maxima) vs. log(scale).
    fit_coeffs : ndarray of float, shape (2,)
        Coefficients [slope, intercept] of the 1D linear fit.

    Raises
    ------
    ValueError
        If `data` is not 1D, contains non‐finite values, is constant,
        if `min_scale` < 1, if `max_fraction` not in (0, 1],
        if `num` < 1, if `log_fit_range` is not a (len=2) tuple/list,
        or if `log_fit_range` lies outside the computed scales,
        or if no valid points remain for fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import WTMM
    >>> ts = np.sin(np.linspace(0, 10, 1024))
    >>> slope, coeffs = WTMM(ts, modulus='mean', wavelet='cmor1-1.5', min_scale=2, max_fraction=0.3, num=40)
    >>> print(f"WTMM slope ≈ {slope:.3f}")
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("`data` must be a one‐dimensional array.")
    if not np.isfinite(data).all():
        raise ValueError("`data` must not contain NaN or infinite values.")
    if np.std(data) == 0:
        raise ValueError("`data` is constant; no scaling to compute.")

    if min_scale < 1:
        raise ValueError("`min_scale` must be ≥ 1.")
    if not (0 < max_fraction <= 1):
        raise ValueError("`max_fraction` must be in (0, 1].")
    if num < 1:
        raise ValueError("`num` (number of scales) must be ≥ 1.")

    # Obtain scales and modulus maxima
    scales, modulus_maxima = utils._WTMM(
        data,
        modulus=modulus,
        wavelet=wavelet,
        min_scale=min_scale,
        max_fraction=max_fraction,
        num=num
    )

    scales = np.asarray(scales, dtype=float)
    modulus_maxima = np.asarray(modulus_maxima, dtype=float)
    if scales.ndim != 1 or modulus_maxima.ndim != 1:
        raise ValueError("`scales` and `modulus_maxima` must be 1D arrays.")
    if scales.size != modulus_maxima.size:
        raise ValueError("`scales` and `modulus_maxima` must have the same length.")
    if not np.all(scales > 0):
        raise ValueError("`scales` must be > 0.")
    # Replace non‐positive modulus_maxima with NaN
    modulus_maxima = np.where(modulus_maxima <= 0, np.nan, modulus_maxima)

    log_scales = np.log(scales)
    log_modmax = np.log(modulus_maxima)

    # Build mask for fitting
    if log_fit_range is not None:
        if (not isinstance(log_fit_range, (tuple, list))) or len(log_fit_range) != 2:
            raise ValueError("`log_fit_range` must be a tuple/list of length 2.")
        min_range, max_range = log_fit_range
        if min_range < scales.min() or max_range > scales.max():
            raise ValueError(
                f"`log_fit_range` {log_fit_range} outside computed scales [{scales.min()}, {scales.max()}]."
            )
        fit_mask = (scales >= min_range) & (scales <= max_range) & (~np.isnan(log_modmax))
    else:
        fit_mask = ~np.isnan(log_modmax)

    if not np.any(fit_mask):
        raise ValueError(
            "No valid data points available for linear fitting on specified `log_fit_range`."
        )

    fit_coeffs = np.polyfit(log_scales[fit_mask], log_modmax[fit_mask], 1)
    slope = float(fit_coeffs[0])

    return slope, fit_coeffs


def RQA(data, threshold="point", percentage=20.0, min_length=2):
    """
    Compute a suite of standard RQA metrics from a 1D time series.

    This function:
      1. Generates a binary recurrence plot (RP) using the chosen thresholding.
      2. Computes:
         - Recurrence Rate (RR)
         - Determinism (DET)
         - Entropy (ENTR)
         - Longest diagonal line (L_max)
         - Divergence (DIV)
         - Laminarity (LAM)
         - Trapping Time (TT)

    Definition
    ----------
    RP_{i,j} = 1 if ||x_i − x_j|| ≤ ε (threshold), else 0.
    RR = (1 / N²) Σ_{i,j} RP_{i,j}.
    DET = (Σ lengths of all diagonal lines ≥ min_length) / (total number of recurrent points).
    ENTR = − Σ p(L) log p(L), where p(L) is probability of diagonal line of length L.
    L_max = max diagonal line length ≥ min_length.
    DIV = 1 / L_max if L_max > 0 else 0.
    LAM = (Σ lengths of all vertical lines ≥ min_length) / (total number of recurrent points).
    TT = average vertical line length ≥ min_length.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series. Must be finite and numeric.
    threshold : {'point'} or float, optional (default="point")
        Thresholding strategy for RP. If 'point', uses `percentage` to pick ε;
        otherwise, uses given float as ε directly.
    percentage : float, optional (default=20.0)
        If `threshold="point"`, percentage of nearest neighbors to mark as recurrent.
        Must satisfy 0 < percentage ≤ 100.
    min_length : int, optional (default=2)
        Minimum line length (diagonal or vertical) to consider for metrics.

    Returns
    -------
    rp : ndarray of shape (N, N)
        Binary recurrence plot (0/1).
    metrics : dict
        Dictionary containing RQA metrics:
          {
            'RR': float,
            'DET': float,
            'ENTR': float,
            'L_max': int,
            'DIV': float,
            'LAM': float,
            'TT': float
          }

    Raises
    ------
    ValueError
        If `series` is not 1D, contains non‐finite values, if `percentage`
        not in (0, 100], or if `min_length` < 1.
    RuntimeError
        If recurrence‐plot computation fails internally.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import RQA
    >>> ts = np.sin(np.linspace(0, 10, 200)) + 0.05*np.random.randn(200)
    >>> rp, metrics = RQA(ts, threshold='point', percentage=15.0, min_length=3)
    >>> print("Recurrence Rate:", metrics['RR'])
    """
    series = np.asarray(data, dtype=float)
    if series.ndim != 1:
        raise ValueError("`series` must be a one‐dimensional array.")
    if not np.isfinite(series).all():
        raise ValueError("`series` must not contain NaN or infinite values.")
    if series.size == 0:
        raise ValueError("`series` must not be empty.")
    if min_length < 1:
        raise ValueError("`min_length` must be ≥ 1.")
    if threshold == "point":
        if not (0 < percentage <= 100):
            raise ValueError("`percentage` must be in (0, 100].")
    elif not isinstance(threshold, (int, float)):
        raise ValueError("`threshold` must be 'point' or a numeric value.")

    # Step 1: generate RP
    try:
        rp = utils._generate_recurrence_plot(
            series,
            threshold=threshold,
            percentage=percentage
        )
    except Exception as e:
        raise RuntimeError(f"Recurrence plot generation failed: {e}")

    # Validate rp for downstream metrics
    rp = np.asarray(rp, dtype=int)
    if rp.ndim != 2 or rp.shape[0] != rp.shape[1]:
        raise ValueError("Generated `rp` must be a square 2D array.")
    uniques = np.unique(rp)
    if not set(uniques).issubset({0, 1}):
        raise ValueError("Generated `rp` must be binary (0/1).")

    # Step 2: compute metrics
    rr = recurrence_rate(rp)
    det = determinism(rp, min_length=min_length)
    lmax = l_max(rp, min_length=min_length)
    entr = entropy(rp, min_length=min_length)
    div = divergence(rp, min_length=min_length)
    lam = laminarity(rp, min_length=min_length)
    tt = trapping_time(rp, min_length=min_length)

    metrics = {
        "RR": rr,
        "DET": det,
        "ENTR": entr,
        "L_max": int(lmax),
        "DIV": div,
        "LAM": lam,
        "TT": tt,
    }
    return rp, metrics


def recurrence_rate(rp):
    """
    Compute Recurrence Rate (RR) from a binary recurrence plot.

    RR = (number of recurrent points) / (total number of points).

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D binary recurrence plot (0/1).

    Returns
    -------
    rr : float
        Recurrence Rate, in [0, 1], i.e., fraction of ones in `rp`.

    Raises
    ------
    ValueError
        If `rp` is not a 2D numpy array or contains values other than 0 or 1.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import recurrence_rate
    >>> rp = np.array([[1, 0], [0, 1]])
    >>> print(recurrence_rate(rp))  # 2 recurrent points / 4 total = 0.5
    """
    
    rp = np.asarray(rp, dtype=int)
    if rp.ndim != 2:
        raise ValueError(f"`rp` must be a 2D array, got shape {rp.shape}.")
    uniques = np.unique(rp)
    if not set(uniques).issubset({0, 1}):
        raise ValueError(f"`rp` must be binary (0/1), but contains {uniques}.")

    total_points = float(rp.size)
    if total_points == 0:
        return 0.0
    rec_points = float(np.sum(rp))
    return rec_points / total_points


def determinism(rp, min_length=2):
    """
    Compute Determinism (DET) from a binary recurrence plot.

    DET = (sum of lengths of all diagonal lines ≥ min_length) / (total number of recurrent points).

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D binary recurrence plot (0/1).
    min_length : int, optional (default=2)
        Minimum diagonal line length to include.

    Returns
    -------
    det : float
        Determinism (DET). If there are no recurrent points, returns 0.0.

    Raises
    ------
    ValueError
        If `rp` is not a 2D binary array, or if `min_length` < 1.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import determinism
    >>> rp = np.array([[1, 1], [1, 1]])
    >>> print(determinism(rp, min_length=2))  # One diagonal of length 2 over 4 points = 2/4 = 0.5
    """
    
    rp = np.asarray(rp, dtype=int)
    if rp.ndim != 2:
        raise ValueError(f"`rp` must be a 2D array, got shape {rp.shape}.")
    uniques = np.unique(rp)
    if not set(uniques).issubset({0, 1}):
        raise ValueError(f"`rp` must be binary (0/1), but contains {uniques}.")
    if min_length < 1:
        raise ValueError("`min_length` must be ≥ 1.")

    rec_points = float(np.sum(rp))
    if rec_points == 0.0:
        return 0.0

    diag_lengths = utils._compute_diagonal_line_lengths(rp, min_length=min_length)
    if not diag_lengths:
        return 0.0

    total_diag = float(np.sum(diag_lengths))
    return total_diag / rec_points


def l_max(rp, min_length=2):
    """
    Compute the length of the longest diagonal line (L_max) in a recurrence plot.

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D binary recurrence plot (0/1).
    min_length : int, optional (default=2)
        Minimum diagonal line length to consider.

    Returns
    -------
    lmax : int
        Maximum diagonal line length ≥ min_length.
        If no diagonal lines ≥ min_length, returns 0.

    Raises
    ------
    ValueError
        If `rp` is not a 2D binary array, or if `min_length` < 1.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import l_max
    >>> rp = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
    >>> print(l_max(rp, min_length=2))  # Longest diagonal of length 3 → 3
    """
    
    rp = np.asarray(rp, dtype=int)
    if rp.ndim != 2:
        raise ValueError(f"`rp` must be a 2D array, got shape {rp.shape}.")
    uniques = np.unique(rp)
    if not set(uniques).issubset({0, 1}):
        raise ValueError(f"`rp` must be binary (0/1), but contains {uniques}.")
    if min_length < 1:
        raise ValueError("`min_length` must be ≥ 1.")

    diag_lengths = utils._compute_diagonal_line_lengths(rp, min_length=min_length)
    if not diag_lengths:
        return 0
    return int(max(diag_lengths))


def entropy(rp, min_length=2):
    """
    Compute Shannon Entropy (ENTR) of diagonal line‐length distribution in a recurrence plot.

    ENTR = − Σ p(L) log p(L),
    where p(L) is the probability of a diagonal line having length L (L ≥ min_length).

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D binary recurrence plot (0/1).
    min_length : int, optional (default=2)
        Minimum diagonal line length to include.

    Returns
    -------
    entr : float
        Shannon entropy of the diagonal line‐length distribution.
        If no diagonal lines ≥ min_length, returns 0.0.

    Raises
    ------
    ValueError
        If `rp` is not a 2D binary array, or if `min_length` < 1.

    Notes
    -----
    This implementation treats each line‐length as a single “event”:
      p(L_i) = length_i / Σ(lengths).
    Some references define p(L) as (# of lines of length L) / (total # lines).

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import entropy
    >>> rp = np.array([[1, 1], [1, 1]])  # one diagonal of length 2
    >>> print(entropy(rp, min_length=2))  # p(2)=1 → entropy=0
    """
    
    rp = np.asarray(rp, dtype=int)
    if rp.ndim != 2:
        raise ValueError(f"`rp` must be a 2D array, got shape {rp.shape}.")
    uniques = np.unique(rp)
    if not set(uniques).issubset({0, 1}):
        raise ValueError(f"`rp` must be binary (0/1), but contains {uniques}.")
    if min_length < 1:
        raise ValueError("`min_length` must be ≥ 1.")

    diag_lengths = utils._compute_diagonal_line_lengths(rp, min_length=min_length)
    if not diag_lengths:
        return 0.0

    lengths = np.array(diag_lengths, dtype=float)
    total = np.sum(lengths)
    probs = lengths / total
    with np.errstate(divide="ignore", invalid="ignore"):
        entr_vals = -probs * np.log(probs)
        entr_vals = entr_vals[np.isfinite(entr_vals)]
    return float(np.sum(entr_vals)) if entr_vals.size > 0 else 0.0


def divergence(rp, min_length=2):
    """
    Compute Divergence (DIV) from a binary recurrence plot.

    DIV = 1 / L_max, where L_max is the length of the longest diagonal line ≥ min_length.
    If L_max == 0, returns 0.0.

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D binary recurrence plot (0/1).
    min_length : int, optional (default=2)
        Minimum diagonal line length to consider.

    Returns
    -------
    div : float
        Divergence = 1 / L_max if L_max > 0; otherwise 0.0.

    Raises
    ------
    ValueError
        If `rp` is not a 2D binary array, or if `min_length` < 1.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import divergence
    >>> rp = np.eye(5, dtype=int)  # only trivial diagonals of length 1
    >>> print(divergence(rp, min_length=1))  # L_max=5 → DIV=0.2
    """
    
    # Re‐use l_max’s validation
    lmax_val = l_max(rp, min_length=min_length)
    return 1.0 / float(lmax_val) if lmax_val > 0 else 0.0


def laminarity(rp, min_length=2):
    """
    Compute Laminarity (LAM) for a recurrence plot.

    LAM = (sum of lengths of all vertical lines ≥ min_length) / (total number of recurrent points).

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D binary recurrence plot (0/1).
    min_length : int, optional (default=2)
        Minimum vertical line length to include.

    Returns
    -------
    lam : float
        Laminarity (LAM). If no recurrent points or no vertical lines ≥ min_length, returns 0.0.

    Raises
    ------
    ValueError
        If `rp` is not a 2D binary array, or if `min_length` < 1.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import laminarity
    >>> rp = np.array([[1, 1], [1, 1]])
    >>> print(laminarity(rp, min_length=2))  # Two vertical lines of length 2 over 4 points → 4/4=1.0
    """
    
    rp = np.asarray(rp, dtype=int)
    if rp.ndim != 2:
        raise ValueError(f"`rp` must be a 2D array, got shape {rp.shape}.")
    uniques = np.unique(rp)
    if not set(uniques).issubset({0, 1}):
        raise ValueError(f"`rp` must be binary (0/1), but contains {uniques}.")
    if min_length < 1:
        raise ValueError("`min_length` must be ≥ 1.")

    rec_points = float(np.sum(rp))
    if rec_points == 0.0:
        return 0.0

    vert_lengths = utils._compute_vertical_line_lengths(rp, min_length=min_length)
    if not vert_lengths:
        return 0.0

    total_vert = float(np.sum(vert_lengths))
    return total_vert / rec_points


def trapping_time(rp, min_length=2):
    """
    Compute Trapping Time (TT) for a recurrence plot.

    TT = average length of vertical lines ≥ min_length.
    If no vertical lines ≥ min_length, returns 0.0.

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D binary recurrence plot (0/1).
    min_length : int, optional (default=2)
        Minimum vertical line length to include.

    Returns
    -------
    tt : float
        Average vertical line length. If no vertical lines ≥ min_length, returns 0.0.

    Raises
    ------
    ValueError
        If `rp` is not a 2D binary array, or if `min_length` < 1.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.functions import trapping_time
    >>> rp = np.array([[1, 0], [1, 1]])
    >>> print(trapping_time(rp, min_length=1))  # vertical lines: lengths [1,2] → avg=1.5
    """
    
    rp = np.asarray(rp, dtype=int)
    if rp.ndim != 2:
        raise ValueError(f"`rp` must be a 2D array, got shape {rp.shape}.")
    uniques = np.unique(rp)
    if not set(uniques).issubset({0, 1}):
        raise ValueError(f"`rp` must be binary (0/1), but contains {uniques}.")
    if min_length < 1:
        raise ValueError("`min_length` must be ≥ 1.")

    vert_lengths = utils._compute_vertical_line_lengths(rp, min_length=min_length)
    if not vert_lengths:
        return 0.0
    return float(np.mean(vert_lengths))