# File: examples/example_usage.py

"""
Example Usage of fractal_analysis package.

This script demonstrates how to:
  - Compute Hurst exponent and interpret it.
  - Compute DFA exponent and interpret it.
  - Perform MF-DFA and plot the Hurst spectrum.
  - Estimate the largest Lyapunov exponent.
  - Perform WTMM analysis and plot scaling.
  - Generate and visualize a recurrence plot, then compute RQA metrics and summarize them.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyenfra import functions, utils, plotting, interpreters

# ───────────────────────────────────────────────────────────────────────────────
# 1) Load or synthesize some time series
# ───────────────────────────────────────────────────────────────────────────────

# Example 1: White noise
ts_white = np.random.RandomState(0).randn(2000)

# Example 2: AR(1) persistent process
def ar1(phi, length=2000, seed=1):
    rng = np.random.RandomState(seed)
    x = np.zeros(length)
    eps = rng.randn(length)
    for i in range(1, length):
        x[i] = phi * x[i-1] + eps[i]
    return x

ts_ar_persist = ar1(phi=0.8)

# Example 3: Sinusoid + noise (for RQA)
length = 1000
t = np.linspace(0, 20*np.pi, length)
ts_sine = np.sin(t) + 0.1*np.random.randn(length)

# Example 4: Logistic map (for Lyapunov)
def logistic_map(length=2000, r=3.99, x0=0.4):
    x = np.zeros(length)
    x[0] = x0
    for n in range(1, length):
        x[n] = r * x[n-1] * (1 - x[n-1])
    return x

ts_logistic = logistic_map()


# ───────────────────────────────────────────────────────────────────────────────
# 2) HURST EXPONENT
# ───────────────────────────────────────────────────────────────────────────────
print("\n=== Hurst Exponent ===")
H_white = functions.hurst(ts_white, num=30, min_n=10, min_segments=10)
print(f"White noise H ≈ {H_white:.3f}")

H_ar = functions.hurst(ts_ar_persist, num=30, min_n=10, min_segments=10)
print(f"AR(1) (φ=0.8) H ≈ {H_ar:.3f}")

# Interpret Hurst
text1 = interpreters.interpret_hurst(ts_white, use_confidence_interval=False)
print("Interpret white noise H:", text1)
text2 = interpreters.interpret_hurst(ts_ar_persist, use_confidence_interval=True, alpha=95)
print("Interpret AR(1) H (with CI):", text2)

# Plot Hurst climacogram for AR(1)
ax_hurst = plotting.plot_hurst(ts_ar_persist, num=30, min_n=10, min_segments=10,
                               figsize=(5,4), scatter_kwargs={'color':'C0'}, line_kwargs={'color':'C1'})
ax_hurst.figure.suptitle("Climacogram: AR(1) Persistent Process")
plt.show()


# ───────────────────────────────────────────────────────────────────────────────
# 3) DFA EXPONENT
# ───────────────────────────────────────────────────────────────────────────────
print("\n=== DFA Exponent ===")
α_white = functions.DFA(ts_white, num=30, min_n=10, min_segments=10)
print(f"White noise DFA α ≈ {α_white:.3f}")

α_ar = functions.DFA(ts_ar_persist, num=30, min_n=10, min_segments=10)
print(f"AR(1) (φ=0.8) DFA α ≈ {α_ar:.3f}")

# Interpret DFA
dtext1 = interpreters.interpret_DFA(ts_white, use_confidence_interval=False)
print("Interpret white noise α:", dtext1)
dtext2 = interpreters.interpret_DFA(ts_ar_persist, use_confidence_interval=True, alpha=95)
print("Interpret AR(1) α (with CI):", dtext2)

# Plot DFA log‐log
ax_dfa = plotting.plot_dfa(ts_ar_persist, num=30, min_n=10, min_segments=10,
                           figsize=(5,4),
                           scatter_kwargs={'marker':'x','color':'C2'},
                           line_kwargs={'color':'C3','linestyle':'--'})
ax_dfa.figure.suptitle("DFA log‐log: AR(1) Persistent Process")
plt.show()


# ───────────────────────────────────────────────────────────────────────────────
# 4) MF‐DFA (Multifractal DFA)
# ───────────────────────────────────────────────────────────────────────────────
print("\n=== MF‐DFA ===")
H_q, qvals = functions.MF_DFA(ts_white, q_mode='typical', num=30, min_n=10, min_segments=10)
print(f"H(q) for white noise (first three): {H_q[:3]} ...")

# Summarize
mf_summary = interpreters.interpret_mf_dfa(H_q, qvals)
print("MF‐DFA summary:", mf_summary)

# Plot H(q) vs. q
ax_mf = plotting.plot_mf_dfa(H_q, qvals, figsize=(5,3), line_kwargs={'color':'C4'}, marker_kwargs={'s':50})
ax_mf.figure.suptitle("MF‐DFA: H(q) Spectrum (White Noise)")
plt.show()


# ───────────────────────────────────────────────────────────────────────────────
# 5) Lyapunov Exponent
# ───────────────────────────────────────────────────────────────────────────────
print("\n=== Lyapunov Exponent ===")
lyap_val, divergence, times = functions.lyapunov(ts_logistic, dim=3, tau=1, fs=1.0, max_iter=200, theiler=1)
print(f"Estimated Lyapunov exponent (logistic r=3.99): {lyap_val:.4f}")
print("Interpretation:", interpreters.interpret_lyapunov(lyap_val))

# Plot divergence vs. time and show fit
ax_lyap = plotting.plot_lyapunov(divergence, times=times, figsize=(5,3), fit_slope=(lyap_val, 0.0),
                                line_kwargs={'color':'C5'}, marker_kwargs={'marker':'o','color':'C6'})
ax_lyap.figure.suptitle("Lyapunov Divergence (Logistic Map)")
plt.show()


# ───────────────────────────────────────────────────────────────────────────────
# 6) WTMM Analysis
# ───────────────────────────────────────────────────────────────────────────────
print("\n=== WTMM Analysis ===")
slope_wtmm, coeffs_wtmm = functions.WTMM(ts_white, modulus='mean', wavelet='cmor1-1.5',
                                         min_scale=2, max_fraction=0.25, num=20)
print(f"WTMM slope (white noise): {slope_wtmm:.4f}")

fig_wtmm, ax_wtmm = plotting.plot_wtmm(ts_white, wavelets=['cmor1-1.5'], modulus='mean',
                                       min_scale=2, max_fraction=0.25, num=20,
                                       figsize=(5,4))
fig_wtmm.suptitle("WTMM Scaling (White Noise)")
plt.show()


# ───────────────────────────────────────────────────────────────────────────────
# 7) Recurrence Plot & RQA
# ───────────────────────────────────────────────────────────────────────────────
print("\n=== Recurrence Plot & RQA ===")
# 7.1 Generate recurrence plot from a noisy sine wave
rp = utils._generate_recurrence_plot(ts_sine, threshold='point', percentage=10.0)

# 7.2 Display the recurrence plot
ax_rp = plotting.plot_recurrence(rp, cmap='binary', figsize=(4,4),
                                 title="Recurrence Plot: Noisy Sine Wave")
plt.show()

# 7.3 Compute RQA metrics
rp2, metrics = functions.RQA(ts_sine, threshold='point', percentage=10.0, min_length=2)
print("RQA Metrics:", metrics)

# 7.4 Summarize RQA metrics as text
print(interpreters.summarize_rqa(metrics))

# 7.5 Show histograms of line lengths
diag_lengths = utils._compute_diagonal_line_lengths(rp2, min_length=2)
vert_lengths = utils._compute_vertical_line_lengths(rp2, min_length=2)
axd, axv = plotting.plot_line_length_histograms(diag_lengths, vert_lengths, bins=10, figsize=(6,3))
axd.figure.suptitle("Line Length Distributions (Noisy Sine Wave RQA)")
plt.show()
