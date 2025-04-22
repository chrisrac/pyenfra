# -*- coding: utf-8 -*-
"""
@author: Krzysztof Raczynski
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
import numpy as np
import pywt
from scipy.ndimage import label
import warnings
warnings.filterwarnings("ignore")

register_matplotlib_converters()
sns.set_style("ticks")

plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=13)

# ---- RS analysis
def _generate_adaptive_nvals(data, num=20, min_n=8, min_segments=10):
    N = len(data)
    max_n = max(min_n, N // min_segments)

    if max_n <= min_n:
        # Fall back to linear range when logspace isn't meaningful
        nvals = np.arange(4, N // 2)
        return nvals[nvals >= 2]
    
    # Otherwise generate log-spaced values
    nvals = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), num).astype(int))
    return nvals[nvals < N]  # Make sure segments fit into the series

def _rescaled_range_analysis(data, nvals):
    N = len(data)
    R_S = []

    for n in nvals:
        segments = int(np.floor(N / n))
        RS_vals = []

        for i in range(segments):
            segment = data[i * n:(i + 1) * n]
            mean_seg = np.mean(segment)
            Z = np.cumsum(segment - mean_seg)
            R = np.max(Z) - np.min(Z)
            S = np.std(segment)
            RS_vals.append(R / S)

        R_S.append(np.mean(RS_vals))

    return R_S

def RS_analysis(data):
    nvals = _generate_adaptive_nvals(data)
    R_S = np.array(_rescaled_range_analysis(data, nvals))
    tR_S = R_S[~np.isnan(R_S)]
    tnvals = nvals[:len(tR_S)]
    coeffs = np.polyfit(np.log(tnvals), np.log(tR_S), 1)
    H_rs = coeffs[0]
    return H_rs

# ---- DFA
def _DFA(data, nvals):
    N = len(data)
    Y = np.cumsum(data - np.mean(data))
    F_n = []
    
    for n in nvals:
        segments = int(np.floor(N / n))
        RMS = []
        
        for i in range(segments):
            segment = Y[i * n:(i + 1) * n]
            x = np.arange(n)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            RMS.append(np.sqrt(np.mean((segment - trend) ** 2)))
        
        F_n.append(np.mean(RMS))
    
    F_n = np.array(F_n)
    return F_n

def DFA(data):
    nvals = _generate_adaptive_nvals(data, min_n=16)
    F_n = _DFA(data, nvals)
    t_F_n = F_n[~np.isnan(F_n)]
    t_nvals = nvals[:len(t_F_n)]
    coeffs = np.polyfit(np.log(t_nvals), np.log(t_F_n), 1)
    return coeffs[0]

# ---- MF DFA
def _generate_mfdfa_nvals(data, num=20, min_n=10, min_segments=10):
    N = len(data)
    max_n = max(min_n + 1, N // min_segments)
    if max_n <= min_n + 1:
        return np.arange(4, max(5, N // 2))
    return np.unique(np.logspace(np.log10(min_n), np.log10(max_n), num).astype(int))

def _generate_mfdfa_qvals(mode='typical'):
    if mode == 'typical':
        qvals = np.linspace(-5, 5, 11)
    elif mode == 'reliable':
        qvals = np.linspace(-4, 4, 11)
    elif mode == 'unstable':
        qvals = np.linspace(-10, 10, 11)
    else:
        raise SyntaxError("mode is incorrect. Accepted values are: 'typical', 'reliable' or 'unstable'")
    
    return qvals

def _MF_DFA(data, nvals, qvals):
    N = len(data)
    Y = np.cumsum(data - np.mean(data))
    F_q = np.zeros((len(qvals), len(nvals)))
    
    for i, q in enumerate(qvals):
        for j, n in enumerate(nvals):
            segments = int(np.floor(N / n))
            RMS = []
            
            for k in range(segments):
                segment = Y[k * n:(k + 1) * n]
                x = np.arange(n)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                RMS.append(np.sqrt(np.mean((segment - trend) ** 2)))
            
            RMS = np.array(RMS)
            if q == 0:
                F_q[i, j] = np.exp(0.5 * np.mean(np.log(RMS ** 2)))
            else:
                F_q[i, j] = (np.mean(RMS ** q)) ** (1.0 / q)
    
    return F_q

def MF_DFA(data):
    qvals = _generate_mfdfa_qvals()
    nvals = _generate_mfdfa_nvals(data)
    F_q = _MF_DFA(data, nvals, qvals)
    H_q = np.full(len(qvals), np.nan)
    for i, q in enumerate(qvals):
        valid = ~np.isnan(F_q[i])
        if valid.sum() >= 2:
            coeffs = np.polyfit(np.log(nvals[valid]), np.log(F_q[i][valid]), 1)
            H_q[i] = coeffs[0]
            
    return H_q

# ---- WTMM

def _generate_wtmm_scales(data, min_scale, max_fraction, num):
    N = len(data)
    max_scale = int(N * max_fraction)
    return np.unique(np.geomspace(min_scale, max_scale, num=num).astype(int))


def _WTMM(data, modulus, wavelet, min_scale=2, max_fraction=0.25, num=50):
    scales = _generate_wtmm_scales(data, min_scale, max_fraction, num)
    coefficients, freqs = pywt.cwt(data, scales, wavelet)
    if modulus=='mean':
        modulus_maxima = np.mean(np.abs(coefficients), axis=1)
    elif modulus=='norm':
        modulus_maxima = np.linalg.norm(coefficients, axis=1)  # L2 norm across time
    elif modulus=='max':
        modulus_maxima = np.max(np.abs(coefficients), axis=1)
    else:
        raise SyntaxError("Incorrect modulus; accepted values are: 'mean', 'max' or 'norm'.")
    return scales, modulus_maxima

def WTMM(data, modulus='mean', wavelet='cmor1-1.5', log_fit_range=None):
    scales, modulus_maxima = _WTMM(data, modulus, wavelet)
    log_scales = np.log(scales)
    log_modmax = np.log(modulus_maxima)    
    if log_fit_range:
        fit_mask = (scales >= log_fit_range[0]) & (scales <= log_fit_range[1])
    else:
        fit_mask = ~np.isnan(log_modmax)
    
    fit_coeffs = np.polyfit(log_scales[fit_mask], log_modmax[fit_mask], 1)
    slope = fit_coeffs[0]
    
    return slope, fit_coeffs


def plot_wtmm_scaling(data, wavelets, modulus='mean', min_scale=2, max_fraction=0.25, num=50, log_fit_range=None):
    """
    Plot WTMM log-log scaling for different wavelets.

    Parameters:
    - data: 1D array-like time series
    - wavelets: list of wavelet names (e.g., ['cmor0.5-1.0', 'cmor1-1.5', 'cmor1.5-2.0'])
    - scale_range: 'auto' or tuple (min_scale, max_scale)
    - num_scales: number of scales in geometric spacing
    - log_fit_range: tuple (min_scale, max_scale) for linear fit; if None, uses all
    """
   
    plt.figure(figsize=(6.5, 4))
    
    for wavelet in wavelets:
        scales, mod_max = _WTMM(data, modulus, wavelet)
                
        log_scales = np.log(scales)
        log_modmax = np.log(mod_max)

        # Choose fit range
        if log_fit_range:
            fit_mask = (scales >= log_fit_range[0]) & (scales <= log_fit_range[1])
        else:
            fit_mask = ~np.isnan(log_modmax)
        
        slope, fit_coeffs = WTMM(data, modulus, wavelet)
        fit_line = np.polyval(fit_coeffs, log_scales)

        plt.plot(log_scales, log_modmax, 'o-', label=f"{wavelet} (slope ≈ {slope:.3f})")
        plt.plot(log_scales, fit_line, '--', alpha=0.5)

    plt.xlabel("log(scale)")
    plt.ylabel("log(modulus maxima)")
    plt.title("WTMM Scaling for Different Wavelets")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def get_line_lengths(binary_array, min_length=2):
    """Count line lengths in a 1D binary sequence."""
    labels, num_features = label(binary_array)
    line_lengths = [np.sum(labels == i) for i in range(1, num_features + 1)]
    return [l for l in line_lengths if l >= min_length]

def count_diagonal_lines(rp, min_length=2):
    diagonals = []
    N = rp.shape[0]
    for offset in range(-N + 1, N):
        diag = np.diagonal(rp, offset=offset)
        diagonals += get_line_lengths(diag, min_length)
    return diagonals

def count_vertical_lines(rp, min_length=2):
    verticals = []
    for col in rp.T:
        verticals += get_line_lengths(col, min_length)
    return verticals

def valid_matrix_dims(emb_dim):
    divisors = []
    for m in range(2, emb_dim + 1):  # matrix_dim must be at least 2
        if (emb_dim - 1) % (m - 1) == 0:
            divisors.append(m)
    return divisors
