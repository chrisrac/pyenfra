# -*- coding: utf-8 -*-
"""
This script is part of the pyenfra module and contains functions necessary to 
perform fractal analysis for environmental data. Documentation can be found 
on repository website: https://github.com/chrisrac/pyenfra

@author: Krzysztof Raczynski
https://github.com/chrisrac
"""

import numpy as np
from . import utils
from . import functions


__all__ = [
    'interpret_hurst',
    'interpret_DFA',
    'interpret_mf_dfa',
    'interpret_lyapunov',
    'summarize_rqa',
    'table_rqa'
]


def interpret_hurst(
    data,
    use_confidence_interval=True,
    alpha=0.05,
    num=20,
    min_n=8,
    min_segments=10
):
    """
    Interpret the meaning of the Hurst exponent value, optionally using Weron’s confidence intervals.

    Steps
    -----
    1. Compute the Hurst exponent H via Rescaled Range analysis (functions.hurst).
    2. If `use_confidence_interval=True`, compute Weron’s empirical CI with `_confidence_interval_rs`.
       - If H > upper_CI: “persistent long‐term memory.”
       - If H < lower_CI: “anti‐persistent memory.”
       - Otherwise: “random‐walk‐like.”
    3. If `use_confidence_interval=False`, compare H to 0.5 only:
       - H > 0.5: “persistent long‐term memory.”
       - H < 0.5: “anti‐persistent memory.”
       - H == 0.5: “random‐walk‐like.”

    Definition
    ----------
    The Hurst exponent (H) measures long‐term memory in a time series:
      - H > 0.5 → persistent behavior
      - H < 0.5 → anti‐persistent (mean‐reverting)
      - H ≈ 0.5 → random walk

    Reference
    ---------
    Hurst H.E., 1951, “Long‐Term Storage Capacity of Reservoirs,”
    Transactions of the American Society of Civil Engineers, 116(1), 770–799.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series. Must be finite numeric and have length ≥ max(50, min_n*min_segments).
    use_confidence_interval : bool, optional (default=True)
        If True, uses Weron’s empirical CI to interpret H. Otherwise, compares H to 0.5 only.
    alpha : float or int, optional (default=0.05)
        Alpha level for CI. Acceptable values:
          - 90%: 90, 10, 0.9, 0.1
          - 95%: 95, 5, 0.95, 0.05
          - 99%: 99, 1, 0.99, 0.01
    num : int, optional (default=20)
        Number of scale values (n) to generate for Hurst estimation.
    min_n : int, optional (default=8)
        Minimum window size n.
    min_segments : int, optional (default=10)
        Minimum number of segments (⌊N/n⌋) for each n.

    Returns
    -------
    result : str
        A textual interpretation of the Hurst exponent.

    Raises
    ------
    ValueError
        If `data` is not a 1D finite array, or if it is too short to compute Hurst,
        or if `alpha` is invalid for CI computation.

    Examples
    --------
    >>> import numpy as np
    >>> from fractal_analysis.interpretes import interpret_hurst
    >>> ts = np.random.randn(200)
    >>> text = interpret_hurst(ts, use_confidence_interval=True, alpha=0.95)
    >>> print(text)
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("`data` must be a one‐dimensional array.")
    if not np.isfinite(data).all():
        raise ValueError("`data` must not contain NaN or infinite values.")
    N = data.size
    if N < 50:
        raise ValueError(f"Time series too short ({N} points); need ≥ 50.")
    if N < min_segments * min_n:
        raise ValueError(
            f"`data` length ({N}) too short for min_segments={min_segments} and min_n={min_n}."
        )

    # Compute Hurst exponent
    H = round(functions.hurst(data, num=num, min_n=min_n, min_segments=min_segments), 3)

    if use_confidence_interval:
        # Compute Weron’s confidence interval for R/S
        # Note: _confidence_interval_rs expects alpha in {90,10,0.9,0.1,...}
        ci = utils._confidence_interval_rs(data, alpha)
        lower_ci, upper_ci = ci[0], ci[1]

        if H > upper_ci:
            return (
                f"Hurst exponent H = {H}. Upper CI = {upper_ci:.3f}. "
                "The series exhibits long‐term memory with persistent patterns "
                "(increases tend to follow increases, decreases follow decreases)."
            )
        elif H < lower_ci:
            return (
                f"Hurst exponent H = {H}. Lower CI = {lower_ci:.3f}. "
                "The series exhibits long‐term memory with anti‐persistent patterns "
                "(values tend to revert to the mean)."
            )
        else:
            return (
                f"Hurst exponent H = {H}. "
                "The series is consistent with a random walk (no persistent trend)."
            )
    else:
        # Simple threshold at 0.5
        if H > 0.5:
            return (
                f"Hurst exponent H = {H}. "
                "The series exhibits long‐term memory with persistent patterns."
            )
        elif H < 0.5:
            return (
                f"Hurst exponent H = {H}. "
                "The series exhibits long‐term memory with anti‐persistent patterns."
            )
        else:  # H == 0.5
            return (
                f"Hurst exponent H = {H}. "
                "The series is consistent with a random walk (no persistent trend)."
            )


def interpret_DFA(
    data,
    use_confidence_interval=True,
    alpha=0.05,
    num=20,
    min_n=8,
    min_segments=10
):
    """
    Interpret the meaning of the DFA scaling exponent α, optionally using Weron’s confidence intervals.

    Steps
    -----
    1. Compute DFA exponent α via functions.DFA.
    2. If `use_confidence_interval=True`, compute empirical CI with `_confidence_interval_dfa`:
       - α < lower_white_CI: “anti‐persistent (mean‐reverting).”
       - lower_white_CI ≤ α < upper_white_CI: “random walk / white noise.”
       - upper_white_CI ≤ α < lower_pink_CI: “persistent long‐range correlations.”
       - lower_pink_CI ≤ α < upper_pink_CI: “1/f (pink) noise.”
       - α ≥ upper_pink_CI: “nonstationary (Brownian‐like).”
    3. If `use_confidence_interval=False`, classify using thresholds only:
       - α < 0.5: anti‐persistent
       - α == 0.5: uncorrelated random walk
       - 0.5 < α < 1.0: persistent long‐range correlations
       - α == 1.0: pink noise
       - α > 1.0: nonstationary

    Definition
    ----------
    The DFA exponent α relates to power‐law correlations:
      - α = 0.5 → uncorrelated (white noise)
      - 0.5 < α < 1 → persistent long‐range correlations
      - α = 1 → 1/f (pink) noise
      - α > 1 → nonstationary (Brownian‐like)

    Reference
    ---------
    Weron R., 2002, “Estimating long‐range dependence: finite sample properties and confidence intervals,”
    Physica A, 312(1–2), 285–299.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series. Must be finite and length ≥ max(10, min_n*min_segments).
    use_confidence_interval : bool, optional (default=True)
        If True, uses Weron’s empirical CI to interpret α. Otherwise, uses fixed thresholds.
    alpha : float or int, optional (default=0.05)
        Alpha for CI. Acceptable: 90/10/0.9/0.1, 95/5/0.95/0.05, 99/1/0.99/0.01.
    num : int, optional (default=20)
        Number of scale values (n) to generate.
    min_n : int, optional (default=8)
        Minimum window size.
    min_segments : int, optional (default=10)
        Minimum segments for largest n.

    Returns
    -------
    result : str
        A textual interpretation of the DFA exponent α.

    Raises
    ------
    ValueError
        If `data` is not 1D finite, too short for DFA, or if `alpha` is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from fractal_analysis.interpretes import interpret_DFA
    >>> ts = np.cos(np.linspace(0,10,500)) + 0.1*np.random.randn(500)
    >>> text = interpret_DFA(ts, use_confidence_interval=False)
    >>> print(text)
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("`data` must be a one‐dimensional array.")
    if not np.isfinite(data).all():
        raise ValueError("`data` must not contain NaN or infinite values.")
    N = data.size
    if N < 10:
        raise ValueError(f"Time series too short ({N} points); need ≥ 10.")

    # Compute DFA exponent
    alpha_val = round(functions.DFA(data, num=num, min_n=min_n, min_segments=min_segments), 3)

    if use_confidence_interval:
        # Compute Weron’s confidence intervals for white and pink noise
        ci = utils._confidence_interval_dfa(data, alpha)
        lower_white, upper_white, lower_pink, upper_pink = ci

        if alpha_val < lower_white:
            return (
                f"DFA exponent α = {alpha_val}. Lower white‐noise CI = {lower_white:.3f}. "
                "Anti‐persistent (mean‐reverting) behavior."
            )
        elif lower_white <= alpha_val < upper_white:
            return (
                f"DFA exponent α = {alpha_val}. White‐noise CI = [{lower_white:.3f}, {upper_white:.3f}]. "
                "Uncorrelated (random walk) behavior."
            )
        elif upper_white <= alpha_val < lower_pink:
            return (
                f"DFA exponent α = {alpha_val}. Between white‐noise upper CI and pink‐noise lower CI: "
                "Persistent long‐range correlations."
            )
        elif lower_pink <= alpha_val < upper_pink:
            return (
                f"DFA exponent α = {alpha_val}. Pink‐noise CI = [{lower_pink:.3f}, {upper_pink:.3f}]. "
                "1/f (pink) noise behavior."
            )
        else:  # alpha_val >= upper_pink
            return (
                f"DFA exponent α = {alpha_val}. Upper pink‐noise CI = {upper_pink:.3f}. "
                "Nonstationary (Brownian‐like) or trending behavior."
            )
    else:
        # Classification without CI
        if alpha_val < 0.5:
            return (
                f"DFA exponent α = {alpha_val}. Anti‐persistent (mean‐reverting) behavior."
            )
        elif alpha_val == 0.5:
            return (
                f"DFA exponent α = {alpha_val}. Uncorrelated (random walk) behavior."
            )
        elif 0.5 < alpha_val < 1.0:
            return (
                f"DFA exponent α = {alpha_val}. Persistent long‐range correlations."
            )
        elif alpha_val == 1.0:
            return (
                f"DFA exponent α = {alpha_val}. 1/f (pink) noise behavior."
            )
        else:  # alpha_val > 1.0
            return (
                f"DFA exponent α = {alpha_val}. Nonstationary (Brownian‐like) or trending behavior."
            )


def interpret_mf_dfa(H_q, qvals):
    """
    Provide a textual summary of the generalized Hurst spectrum H(q) vs. q.

    The summary includes:
      - Minimum H(q) over all q,
      - Maximum H(q),
      - H(0) if q=0 is present (else None),
      - A Boolean `monofractal` flag which is True if std(H_q) < 1e−3.

    Parameters
    ----------
    H_q : array‐like of shape (M,)
        Generalized Hurst exponents.
    qvals : array‐like of shape (M,)
        Corresponding q exponents.

    Returns
    -------
    summary : dict
        {
          'H_min'       : float,  # minimum H(q)
          'H_max'       : float,  # maximum H(q)
          'H_at_q_zero' : float or None,  # H(0) if available
          'monofractal' : bool    # True if std(H_q) < 1e−3
        }

    Raises
    ------
    ValueError
        If `H_q` and `qvals` have mismatched shapes or if all H_q are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from fractal_analysis.interpretes import interpret_mf_dfa
    >>> Hq = np.array([0.6, 0.61, 0.60, 0.59, np.nan])
    >>> q = np.array([-2, -1, 0, 1, 2])
    >>> summary = interpret_mf_dfa(Hq, q)
    >>> print(summary)
    {'H_min': 0.59, 'H_max': 0.61, 'H_at_q_zero': 0.6, 'monofractal': False}
    """
    H_q = np.asarray(H_q, dtype=float)
    qvals = np.asarray(qvals, dtype=float)
    if H_q.shape != qvals.shape:
        raise ValueError("Shapes of `H_q` and `qvals` must match.")
    if np.all(np.isnan(H_q)):
        raise ValueError("All values in `H_q` are NaN; cannot interpret.")

    valid_mask = ~np.isnan(H_q)
    H_valid = H_q[valid_mask]
    q_valid = qvals[valid_mask]

    H_min = float(np.min(H_valid))
    H_max = float(np.max(H_valid))

    # Find H at q=0 if present
    if 0 in q_valid:
        H0 = float(H_valid[np.where(q_valid == 0)[0][0]])
    else:
        H0 = None

    monofractal = bool(np.nanstd(H_valid) < 1e-3)

    return {
        'H_min': H_min,
        'H_max': H_max,
        'H_at_q_zero': H0,
        'monofractal': monofractal
    }


def interpret_lyapunov(lyap_exp):
    """
    Provide a classification of the largest Lyapunov exponent.

    Classification
    --------------
    - lyap_exp > 0   : “Positive Lyapunov exponent indicates presence of chaos.”
    - lyap_exp < 0   : “Negative Lyapunov exponent indicates periodic/quasiperiodic dynamics.”
    - lyap_exp == 0  : “Neutral or random‐walk‐like behavior.”
    - lyap_exp = NaN : “Undefined.”

    Parameters
    ----------
    lyap_exp : float
        Largest Lyapunov exponent (per time unit).

    Returns
    -------
    classification : str
        One of:
          - 'Positive Lyapunov exponent indicates presence of chaos.'
          - 'Negative Lyapunov exponent indicates periodic or quasiperiodic dynamics.'
          - 'Neutral or random walk.'
          - 'Undefined.'

    Raises
    ------
    None

    Examples
    --------
    >>> from fractal_analysis.interpretes import interpret_lyapunov
    >>> print(interpret_lyapunov(0.15))
    Positive Lyapunov exponent indicates presence of chaos.
    """
    if lyap_exp is None or (isinstance(lyap_exp, float) and np.isnan(lyap_exp)):
        return 'Undefined'
    if lyap_exp > 0:
        return 'Positive Lyapunov exponent indicates presence of chaos.'
    elif lyap_exp < 0:
        return 'Negative Lyapunov exponent indicates periodic or quasiperiodic dynamics.'
    else:
        return 'Neutral or random walk.'


def summarize_rqa(metrics):
    """
    Generate a human‐readable summary string for a dictionary of RQA metrics.

    Each metric name and its value appear on a separate line.

    Parameters
    ----------
    metrics : dict
        Dictionary mapping metric names (str) to values (float or int).

    Returns
    -------
    summary : str
        Multiline text, starting with “RQA Metrics Summary:” followed by each metric.

    Raises
    ------
    ValueError
        If `metrics` is not a non‐empty dict.

    Examples
    --------
    >>> metrics = {'RR': 0.12, 'DET': 0.45, 'L_max': 20}
    >>> from fractal_analysis.interpretes import summarize_rqa
    >>> print(summarize_rqa(metrics))
    RQA Metrics Summary:
      RR: 0.120000
      DET: 0.450000
      L_max: 20
    """
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("`metrics` must be a non‐empty dictionary.")

    lines = ["RQA Metrics Summary:"]
    for key, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"  {key}: {val:.6f}")
        else:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def table_rqa(metrics):
    """
    Return a simple text‐table representation of RQA metrics.

    The first row is “Metric | Value”, the second is a separator, and subsequent lines:
    `metric_name | formatted_value`.

    Parameters
    ----------
    metrics : dict
        Dictionary mapping metric names (str) to numerical values (float or int).

    Returns
    -------
    table : str
        Multiline text table. If `metrics` is empty, returns an empty string.

    Examples
    --------
    >>> metrics = {'RR': 0.12, 'DET': 0.45, 'L_max': 20}
    >>> from fractal_analysis.interpretes import table_rqa
    >>> print(table_rqa(metrics))
    Metric | Value
    ------ | -----
    RR | 0.120000
    DET | 0.450000
    L_max | 20
    """
    if not metrics:
        return ""
    lines = ["Metric | Value", "------ | -----"]
    for key, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"{key} | {val:.6f}")
        else:
            lines.append(f"{key} | {val}")
    return "\n".join(lines)