# -*- coding: utf-8 -*-
"""
This script is part of the pyenfra module and contains functions necessary to 
perform fractal analysis for environmental data. Documentation can be found 
on repository website: https://github.com/chrisrac/pyenfra

@author: Krzysztof Raczynski
https://github.com/chrisrac
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from . import utils


__all__ = [
    'plot_hurst',
    'plot_dfa',
    'plot_mf_dfa',
    'plot_lyapunov',
    'plot_wtmm',
    'plot_recurrence',
    'plot_line_length_histograms'
]


def plot_hurst(data, num=20, min_n=8, min_segments=10, figsize=(6, 4),
    title="Climacogram (R/S Analysis)", xlabel="log(n)", ylabel="log(R/S)",
    scatter_kwargs=None, line_kwargs=None):
    """
    Plot the climacogram: log‐log plot of window size n vs. rescaled range R/S.

    This function:
      1. Generates scale values (nvals) via `_generate_adaptive_nvals`.
      2. Computes R/S for each n ∈ nvals.
      3. Fits a line to log(R/S) vs. log(n) and reports the Hurst exponent.
      4. Produces a scatter of log(n), log(R/S) and overlays the fit.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series to analyze. Must be finite and have length ≥ max(50, min_n * min_segments).
    num : int, optional (default=20)
        Number of scales (n) to generate if needed.
    min_n : int, optional (default=8)
        Minimum scale (window size) to consider.
    min_segments : int, optional (default=10)
        Minimum number of segments (⌊N/n⌋) required for statistical validity.
    figsize : tuple of two ints, optional (default=(6, 4))
        Figure size passed to `plt.subplots`.
    title : str, optional (default="Climacogram (R/S Analysis)")
        Title for the plot.
    xlabel : str, optional (default="log(n)")
        Label for the x‐axis.
    ylabel : str, optional (default="log(R/S)")
        Label for the y‐axis.
    scatter_kwargs : dict or None, optional
        Additional keyword arguments for `ax.scatter`. e.g., `{'color':'C0', 's':20}`.
    line_kwargs : dict or None, optional
        Additional keyword arguments for `ax.plot` (the fit line). e.g., `{'color':'red', 'linestyle':'--'}`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the climacogram plot.

    Raises
    ------
    ValueError
        If `data` is not a 1D finite array, too short for R/S calculation,
        or if insufficient valid scales remain for Hurst estimation.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.plotting import plot_hurst
    >>> ts = np.random.randn(500)
    >>> ax = plot_hurst(ts, num=15, min_n=10, min_segments=8,
    ...                figsize=(5,4),
    ...                scatter_kwargs={'color':'C1'},
    ...                line_kwargs={'color':'C3','linewidth':2})
    """
    
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("`data` must be a one‐dimensional array.")
    if not np.isfinite(data).all():
        raise ValueError("`data` must not contain NaN or infinite values.")
    N = data.size
    if N < 50:
        raise ValueError(f"Time series too short ({N} points); need ≥ 50. Use DFA for shorter series.")
    if N < min_segments * min_n:
        raise ValueError(
            f"`data` length ({N}) too short for min_segments={min_segments} and min_n={min_n}."
        )

    # Generate scales and compute R/S
    nvals = utils._generate_adaptive_nvals(data, num, min_n, min_segments)
    R_S = np.array(utils._rescaled_range_analysis(data, nvals))

    # Filter valid (finite, positive) points
    nvals = np.asarray(nvals)
    valid_mask = np.isfinite(R_S) & (R_S > 0)
    tn = nvals[valid_mask]
    tRS = R_S[valid_mask]
    if tn.size < 5:
        raise ValueError(f"Insufficient valid scales ({tn.size}) for reliable Hurst estimation.")

    # Fit to obtain Hurst exponent
    log_n = np.log(tn)
    log_RS = np.log(tRS)
    coeffs = np.polyfit(log_n, log_RS, 1)
    H = float(coeffs[0])

    # Compute intercept for plotting
    intercept = float(coeffs[1])

    # Prepare figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter points
    s_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    ax.scatter(log_n, log_RS, label='Data Points', **s_kwargs)

    # Fit line
    x_line = np.linspace(np.min(log_n), np.max(log_n), 100)
    y_line = H * x_line + intercept
    l_kwargs = {'color': 'red'} if line_kwargs is None else line_kwargs
    ax.plot(x_line, y_line, label=f'Fit Line (H = {H:.4f})', **l_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    return ax


def plot_dfa(data, nvals=None, num=20, min_n=8, min_segments=10, figsize=(6, 4),
    xlabel="log₁₀(window size n)", ylabel="log₁₀(F(n))", title="DFA log‐log plot",
    show_fit=True, scatter_kwargs=None, line_kwargs=None):
    """
    Create a log‐log plot of F(n) vs. n for DFA, optionally overlaying the regression line.

    Steps:
      1. Generate or accept window sizes `nvals`.
      2. Compute fluctuation function F(n) via `_compute_fluctuations`.
      3. Plot log₁₀(n) vs. log₁₀(F(n)) points.
      4. If `show_fit=True`, fit ln(F) vs. ln(n) and overlay the corresponding line on log₁₀ axes.

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series. Must be finite and length ≥ 2*min_n.
    nvals : array‐like of int or None, optional
        Precomputed window sizes. If None, uses `_generate_adaptive_nvals(data, num, min_n, min_segments)`.
    num : int, optional (default=20)
        Number of scales to generate if `nvals` is None.
    min_n : int, optional (default=8)
        Minimum window size if `nvals` is None.
    min_segments : int, optional (default=10)
        Minimum number of segments if `nvals` is None.
    figsize : tuple of two ints, optional (default=(6,4))
        Figure size for the plot.
    xlabel : str, optional (default="log₁₀(window size n)")
        Label for x‐axis.
    ylabel : str, optional (default="log₁₀(F(n))")
        Label for y‐axis.
    title : str, optional (default="DFA log‐log plot")
        Title for the plot.
    show_fit : bool, optional (default=True)
        If True, overlays the regression line.
    scatter_kwargs : dict or None, optional
        Keyword arguments for `ax.scatter`. e.g., `{'color':'C2'}`.
    line_kwargs : dict or None, optional
        Keyword arguments for `ax.plot` (fit line). e.g., `{'color':'C3','linestyle':'--'}`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the DFA log‐log plot.

    Raises
    ------
    ValueError
        If `data` is not 1D finite or if there are fewer than two valid (n, F(n)) pairs to plot.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.plotting import plot_dfa
    >>> ts = np.sin(np.linspace(0, 20*np.pi, 1000)) + 0.1*np.random.randn(1000)
    >>> ax = plot_dfa(ts, num=30, min_n=20, min_segments=8,
    ...               figsize=(5,4),
    ...               scatter_kwargs={'marker':'x'},
    ...               line_kwargs={'linewidth':2})
    """
    
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("`data` must be a one‐dimensional array.")
    if not np.isfinite(data).all():
        raise ValueError("`data` must not contain NaN or infinite values.")
    N = data.size
    if N < 2 * min_n and nvals is None:
        raise ValueError(f"Data length ({N}) is too short for min_n={min_n}.")

    # Determine nvals
    if nvals is None:
        nvals = utils._generate_adaptive_nvals(data, num, min_n, min_segments)
    else:
        nvals = np.asarray(nvals, dtype=int)
        if nvals.ndim != 1 or np.any(nvals < 2) or np.any(nvals >= N):
            raise ValueError("`nvals` must be a 1D array of ints satisfying 2 ≤ n < len(data).")

    # Compute F(n)
    F_n = utils._compute_fluctuations(data, nvals)
    valid_mask = np.isfinite(F_n) & (F_n > 0)
    t_n = nvals[valid_mask]
    t_F = F_n[valid_mask]
    if t_n.size < 2:
        raise ValueError("Not enough valid (n, F(n)) pairs to plot DFA.")

    # Prepare figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter: plot log10(n) vs. log10(F(n))
    s_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    ax.scatter(np.log10(t_n), np.log10(t_F), label="Empirical points", **s_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Overlay fit if requested
    if show_fit:
        # Fit ln(F) = slope * ln(n) + intercept
        ln_n = np.log(t_n)
        ln_F = np.log(t_F)
        coeffs = np.polyfit(ln_n, ln_F, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

        # On log10 axes: log10(F) = (slope * ln(n) + intercept) / ln(10)
        x_line_log10 = np.linspace(np.min(np.log10(t_n)), np.max(np.log10(t_n)), 100)
        ln_n_line = x_line_log10 * np.log(10)
        ln_F_line = slope * ln_n_line + intercept
        y_line_log10 = ln_F_line / np.log(10)

        l_kwargs = {'color': 'red', 'linestyle': '--'} if line_kwargs is None else line_kwargs
        ax.plot(x_line_log10, y_line_log10, label=f"Fit: slope={slope:.3f}", **l_kwargs)
        ax.legend()

    ax.grid(True)
    return ax


def plot_mf_dfa(H_q, qvals, ax=None, figsize=(6, 4), xlabel='q', ylabel='H(q)',
    title='Hurst Spectrum', line_kwargs=None, marker_kwargs=None):
    """
    Plot the generalized Hurst exponent H(q) vs. q (the Hurst spectrum).

    Parameters
    ----------
    H_q : array‐like of shape (M,)
        Generalized Hurst exponents for each q in `qvals`.
    qvals : array‐like of shape (M,)
        Corresponding q exponents. Must match `H_q` in length.
    ax : matplotlib.axes.Axes or None, optional
        Axes on which to plot. If None, a new figure and axes are created.
    figsize : tuple of two ints, optional (default=(6,4))
        Figure size if `ax=None`.
    xlabel : str, optional (default='q')
        Label for the x‐axis.
    ylabel : str, optional (default='H(q)')
        Label for the y‐axis.
    title : str, optional (default='Hurst Spectrum')
        Title for the plot.
    line_kwargs : dict or None, optional
        Keyword arguments for the line plot (e.g., `{'color':'C2','linestyle':'-'}`).
    marker_kwargs : dict or None, optional
        Keyword arguments for the markers (e.g., `{'marker':'o','color':'C2'}`).

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the Hurst spectrum plot.

    Raises
    ------
    ValueError
        If `H_q` or `qvals` are not 1D arrays of the same length.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.plotting import plot_mf_dfa
    >>> H_q = np.array([0.5, 0.55, 0.6, 0.58, 0.52])
    >>> q = np.array([-2, -1, 0, 1, 2])
    >>> ax = plot_mf_dfa(H_q, q, figsize=(5,3), line_kwargs={'color':'C3'}, marker_kwargs={'s':50})
    """
    
    H_q = np.asarray(H_q, dtype=float)
    qvals = np.asarray(qvals, dtype=float)
    if H_q.ndim != 1 or qvals.ndim != 1 or H_q.shape != qvals.shape:
        raise ValueError("`H_q` and `qvals` must be 1D arrays of the same shape.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    l_kwargs = {'linestyle': '-', 'color': 'C0'} if line_kwargs is None else line_kwargs
    m_kwargs = {'marker': 'o', 'color': l_kwargs.get('color', 'C0')} if marker_kwargs is None else marker_kwargs

    ax.plot(qvals, H_q, **l_kwargs, label='H(q)')
    ax.scatter(qvals, H_q, **m_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    return ax


def plot_lyapunov(divergence, times=None, ax=None, figsize=(6, 4),
    xlabel='Time (samples or units)', ylabel='⟨log divergence⟩', title='Lyapunov Divergence Plot',
    fit_slope=None, line_kwargs=None, marker_kwargs=None):
    """
    Plot average log‐divergence vs. time for Lyapunov exponent estimation.

    Parameters
    ----------
    divergence : array‐like of shape (L,)
        Average log‐divergence ⟨ln d(k)⟩ at each time step k.
    times : array‐like of shape (L,) or None, optional
        Time points corresponding to `divergence`. If None, uses `np.arange(L)`.
    ax : matplotlib.axes.Axes or None, optional
        Axes on which to plot. If None, a new figure and axes are created.
    figsize : tuple of two ints, optional (default=(6,4))
        Figure size if `ax=None`.
    xlabel : str, optional (default='Time (samples or units)')
        Label for the x‐axis.
    ylabel : str, optional (default='⟨log divergence⟩')
        Label for the y‐axis.
    title : str, optional (default='Lyapunov Divergence Plot')
        Title for the plot.
    fit_slope : tuple (slope, intercept) or None, optional
        If provided, plots the line `y = slope * t + intercept` over the data.
    line_kwargs : dict or None, optional
        Keyword arguments for the fit line (e.g., `{'color':'C1','linewidth':1.5}`).
    marker_kwargs : dict or None, optional
        Keyword arguments for the data points (e.g., `{'marker':'o','color':'C0'}`).

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the Lyapunov divergence plot.

    Raises
    ------
    ValueError
        If `divergence` is not a 1D array, or if `times` does not match its shape.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.plotting import plot_lyapunov
    >>> div = np.array([0.1, 0.2, 0.35, 0.50, 0.65])
    >>> t = np.array([0, 1, 2, 3, 4])
    >>> ax = plot_lyapunov(div, times=t, fit_slope=(0.15, 0.0),
    ...                    marker_kwargs={'marker':'x'}, line_kwargs={'linestyle':'--'})
    """
    
    divergence = np.asarray(divergence, dtype=float)
    if divergence.ndim != 1:
        raise ValueError("`divergence` must be a 1D array.")
    L = divergence.size

    if times is None:
        times = np.arange(L, dtype=float)
    else:
        times = np.asarray(times, dtype=float)
        if times.shape != divergence.shape:
            raise ValueError("`times` must have the same shape as `divergence`.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    m_kwargs = {'marker': 'o', 'linestyle': '-', 'color': 'C0'} if marker_kwargs is None else marker_kwargs
    ax.plot(times, divergence, **m_kwargs, label='⟨log divergence⟩')

    if fit_slope is not None:
        slope, intercept = fit_slope
        x_line = times
        y_line = slope * x_line + intercept
        l_kwargs = {'color': 'C1', 'linestyle': '--'} if line_kwargs is None else line_kwargs
        ax.plot(x_line, y_line, label=f"Fit: slope={slope:.3f}", **l_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return ax


def plot_wtmm(data, wavelets, modulus='mean', min_scale=2, max_fraction=0.25,
    num=50, log_fit_range=None, figsize=(6, 4), xlabel='log(scale)',
    ylabel='log(modulus maxima)', title='WTMM Scaling for Different Wavelets'):
    """
    Plot WTMM log–log scaling curves and linear fits for multiple wavelets.

    For each wavelet in `wavelets`:
      1. Compute scales and modulus maxima via `_WTMM`.
      2. Plot (ln scale, ln modulus_maxima) points.
      3. Fit a line to ln(modulus_maxima) vs. ln(scale) using `_compute_log_fit`.
      4. Overlay the fit if possible, reporting slope and R².

    Parameters
    ----------
    data : array‐like of shape (N,)
        1D time series for WTMM analysis. Must be finite and length ≥ 2.
    wavelets : list of str or pywt.ContinuousWavelet
        List of wavelet names or ContinuousWavelet objects (e.g., ['cmor0.5-1.0', 'cmor1-1.5']).
    modulus : {'mean', 'norm', 'max'}, optional (default='mean')
        Method to collapse wavelet coefficients at each scale.
    min_scale : int, optional (default=2)
        Minimum scale to include in WTMM (must be ≥ 1).
    max_fraction : float, optional (default=0.25)
        Fraction of `len(data)` to use for maximum scale (0 < max_fraction ≤ 1).
    num : int, optional (default=50)
        Number of scales to generate for each wavelet.
    log_fit_range : tuple of two floats (min_scale, max_scale), optional
        Inclusive range of scales for fitting (in original scale, not log). If None, all scales used.
    figsize : tuple of two ints, optional (default=(6, 4))
        Figure size for the plot.
    xlabel : str, optional (default='log(scale)')
        Label for the x‐axis.
    ylabel : str, optional (default='log(modulus maxima)')
        Label for the y‐axis.
    title : str, optional (default='WTMM Scaling for Different Wavelets')
        Title for the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the WTMM scaling curves.
    ax : matplotlib.axes.Axes
        Axes on which the curves and fits were drawn.

    Raises
    ------
    ValueError
        If `data` is not a 1D finite array, or if `wavelets` is not a non‐empty list/tuple.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.plotting import plot_wtmm
    >>> ts = np.sin(np.linspace(0, 2*np.pi, 512)) + 0.1*np.random.randn(512)
    >>> wavelets = ['cmor0.5-1.0', 'cmor1-1.5']
    >>> fig, ax = plot_wtmm(ts, wavelets,
    ...                     modulus='mean',
    ...                     min_scale=2,
    ...                     max_fraction=0.3,
    ...                     num=20,
    ...                     log_fit_range=(2, 50),
    ...                     figsize=(5,4))
    """
    
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("`data` must be a one‐dimensional array.")
    if not np.isfinite(data).all():
        raise ValueError("`data` must not contain NaN or infinite values.")
    if not isinstance(wavelets, (list, tuple)) or len(wavelets) == 0:
        raise ValueError("`wavelets` must be a non‐empty list or tuple of wavelet names/objects.")

    fig, ax = plt.subplots(figsize=figsize)

    for wavelet in wavelets:
        try:
            scales, mod_max = utils._WTMM(
                data,
                modulus=modulus,
                wavelet=wavelet,
                min_scale=min_scale,
                max_fraction=max_fraction,
                num=num
            )
        except Exception as exc:
            warnings.warn(f"Skipping wavelet {wavelet!r}: {exc}", UserWarning)
            continue

        # Convert to ln (natural log) and handle non‐positive modulus
        scales = np.asarray(scales, dtype=float)
        mod_max = np.asarray(mod_max, dtype=float)
        if scales.ndim != 1 or mod_max.ndim != 1 or scales.size != mod_max.size:
            warnings.warn(f"Invalid output shapes for wavelet {wavelet!r}; skipping.", UserWarning)
            continue

        log_scales = np.log(scales)
        mod_max = np.where(mod_max <= 0, np.nan, mod_max)
        log_modmax = np.log(mod_max)

        # Plot raw ln‐ln points
        ax.plot(
            log_scales,
            log_modmax,
            marker='o',
            linestyle='-',
            label=f"{wavelet} data"
        )

        # Attempt linear fit
        try:
            fit_info = utils._compute_log_fit(scales, mod_max, fit_range=log_fit_range)
            slope = fit_info['slope']
            intercept = fit_info['intercept']
            r2 = fit_info['r_squared']
            fit_line = slope * log_scales + intercept
            ax.plot(
                log_scales,
                fit_line,
                linestyle='--',
                alpha=0.7,
                label=f"{wavelet} fit (s={slope:.3f}, R²={r2:.2f})"
            )
        except Exception as fit_exc:
            warnings.warn(f"Could not fit for wavelet {wavelet!r}: {fit_exc}", UserWarning)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig, ax


def plot_recurrence(rp, cmap="Greys", figsize=(6, 6), title="Recurrence Plot",
    xlabel="Time Index", ylabel="Time Index"):
    """
    Display a binary recurrence plot as an image.

    Parameters
    ----------
    rp : ndarray of shape (N, N)
        2D binary recurrence plot (0/1).
    cmap : str, optional (default="Greys")
        Matplotlib colormap for the image (e.g., "Greys", "binary").
    figsize : tuple of two ints, optional (default=(6,6))
        Figure size for the plot.
    title : str, optional (default="Recurrence Plot")
        Title for the plot.
    xlabel : str, optional (default="Time Index")
        Label for the x‐axis.
    ylabel : str, optional (default="Time Index")
        Label for the y‐axis.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the recurrence plot.

    Raises
    ------
    ValueError
        If `rp` is not a 2D numpy array.

    Examples
    --------
    >>> import numpy as np
    >>> from pyenfra.plotting import plot_recurrence
    >>> rp = np.zeros((100,100), dtype=int)
    >>> rp[10:20, 30:40] = 1
    >>> ax = plot_recurrence(rp, cmap='binary', figsize=(5,5))
    """
    
    rp = np.asarray(rp, dtype=int)
    if rp.ndim != 2:
        raise ValueError(f"`rp` must be a 2D array, got shape {rp.shape}.")

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(rp, origin="lower", cmap=cmap, interpolation="nearest")
    fig.colorbar(img, ax=ax, label="Recurrence (0 or 1)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return ax


def plot_line_length_histograms(diag_lengths, vert_lengths, bins=None, figsize=(10, 4),
    diag_title="Diagonal Line Lengths", vert_title="Vertical Line Lengths",
    xlabel="Length", ylabel="Frequency"):
    """
    Plot histograms of diagonal and vertical line‐length distributions side by side.

    Parameters
    ----------
    diag_lengths : list of int
        List of diagonal line lengths (each > 0).
    vert_lengths : list of int
        List of vertical line lengths (each > 0).
    bins : int or sequence of int or None, optional
        Number of bins or specific bin edges for both histograms. If None, matplotlib chooses automatically.
    figsize : tuple of two ints, optional (default=(10,4))
        Figure size (width, height).
    diag_title : str, optional (default="Diagonal Line Lengths")
        Title for the diagonal histogram.
    vert_title : str, optional (default="Vertical Line Lengths")
        Title for the vertical histogram.
    xlabel : str, optional (default="Length")
        Label for the x‐axes.
    ylabel : str, optional (default="Frequency")
        Label for the y‐axes.

    Returns
    -------
    axs : tuple (ax_diag, ax_vert)
        Axes of the diagonal and vertical histograms, respectively.

    Raises
    ------
    ValueError
        If `diag_lengths` or `vert_lengths` is not a list of positive integers.

    Examples
    --------
    >>> from pyenfra.plotting import plot_line_length_histograms
    >>> diag = [2,3,3,4,5,2]
    >>> vert = [1,1,2,2,3,4]
    >>> ax_diag, ax_vert = plot_line_length_histograms(diag, vert, bins=5, figsize=(8,3))
    """
    
    if not isinstance(diag_lengths, (list, tuple)) or any(
        not isinstance(x, (int, np.integer)) or x <= 0 for x in diag_lengths
    ):
        raise ValueError("`diag_lengths` must be a list of positive integers.")
    if not isinstance(vert_lengths, (list, tuple)) or any(
        not isinstance(x, (int, np.integer)) or x <= 0 for x in vert_lengths
    ):
        raise ValueError("`vert_lengths` must be a list of positive integers.")

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Diagonal histogram
    ax_diag = axs[0]
    ax_diag.hist(diag_lengths, bins=bins, edgecolor="black")
    ax_diag.set_title(diag_title)
    ax_diag.set_xlabel(xlabel)
    ax_diag.set_ylabel(ylabel)

    # Vertical histogram
    ax_vert = axs[1]
    ax_vert.hist(vert_lengths, bins=bins, edgecolor="black")
    ax_vert.set_title(vert_title)
    ax_vert.set_xlabel(xlabel)
    ax_vert.set_ylabel(ylabel)

    fig.tight_layout()
    return ax_diag, ax_vert