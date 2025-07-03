import numpy as np
import pytest

from pyenfra import functions, utils, plotting, interpreters

# ─────────────────────────────────────────────────────────────────────────────
# 1) Synthetic Dataset Generators
# ─────────────────────────────────────────────────────────────────────────────

def white_noise(length=1000, seed=0):
    """Generate i.i.d. Gaussian white noise of given length."""
    rng = np.random.RandomState(seed)
    return rng.randn(length)


def ar1_process(length=1000, phi=0.8, seed=0):
    """
    Generate an AR(1) process: x_t = phi*x_{t-1} + ε_t, 
    with ε_t ~ N(0,1). 
    If phi > 0, persistent; if phi < 0, anti-persistent.
    """
    rng = np.random.RandomState(seed)
    x = np.zeros(length)
    eps = rng.randn(length)
    for t in range(1, length):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def pure_sine(length=1000, freq=5, noise_amp=0.0):
    """
    Generate a pure sinusoid (plus optional small additive noise).
    Useful for RQA (periodic structure).
    """
    t = np.linspace(0, 2 * np.pi * freq, length)
    signal = np.sin(t)
    if noise_amp > 0.0:
        signal += noise_amp * np.random.randn(length)
    return signal


def logistic_map(length=1000, r=3.99, x0=0.5):
    """
    Generate logistic map iterates: x_{n+1} = r * x_n * (1 - x_n).
    For r=3.99, this is chaotic (positive Liapunov).
    """
    x = np.zeros(length)
    x[0] = x0
    for n in range(1, length):
        x[n] = r * x[n - 1] * (1 - x[n - 1])
    return x


# ─────────────────────────────────────────────────────────────────────────────
# 2) HURST & DFA Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phi, expected_sign", [
    (0.0, 0),      # white noise → H ≈ 0.5
    (0.8, +1),     # persistent → H > 0.5
    (-0.8, -1),    # anti‐persistent → H < 0.5
])
def test_hurst_ar1(phi, expected_sign):
    """Check that Hurst(AR1) is roughly >0.5 if phi>0, <0.5 if phi<0, and ~0.5 if phi=0."""
    ts = ar1_process(length=5000, phi=phi, seed=42)
    H = functions.hurst(ts, num=30, min_n=10, min_segments=10)
    # White noise (phi=0) should be within ±0.1 of 0.5.
    if phi == 0.0:
        assert abs(H - 0.5) < 0.1
    else:
        if expected_sign > 0:
            assert H > 0.55
        else:
            assert H < 0.45


def test_hurst_errors():
    """Check that Hurst raises ValueError for too‐short series."""
    short = np.arange(20)  # length < 50
    with pytest.raises(ValueError):
        functions.hurst(short)
    # Also test length < min_segments*min_n
    # For example, length=80, min_n=20, min_segments=5 → 80 < 100
    ts = np.arange(80)
    with pytest.raises(ValueError):
        functions.hurst(ts, num=10, min_n=20, min_segments=5)


@pytest.mark.parametrize("phi, expected_sign", [
    (0.0, 0),     # white noise → α ≈ 0.5
    (0.8, +1),    # persistent → α > 0.5
    (-0.8, -1),   # anti
])
def test_dfa_ar1(phi, expected_sign):
    """Check that DFA exponent for AR1 matches expectations."""
    ts = ar1_process(length=5000, phi=phi, seed=24)
    α = functions.DFA(ts, num=30, min_n=10, min_segments=10)
    if phi == 0.0:
        assert abs(α - 0.5) < 0.1
    else:
        if expected_sign > 0:
            assert α > 0.55
        else:
            assert α < 0.45


def test_dfa_errors():
    """Check that DFA raises for non‐1D or too‐short series."""
    bad = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        functions.DFA(bad)
    short = np.ones(10)
    # min_n=8 → 10 < 2*8 = 16
    with pytest.raises(ValueError):
        functions.DFA(short, num=5, min_n=8, min_segments=2)


# ─────────────────────────────────────────────────────────────────────────────
# 3) MF‐DFA Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_mf_dfa_monofractal():
    """For white noise, H(q) should be nearly constant across q (monofractal)."""
    ts = white_noise(length=5000, seed=5)
    H_q, qvals = functions.MF_DFA(ts, q_mode='typical', num=30, min_n=10, min_segments=10, return_nvals=False)
    # Standard deviation small
    assert np.nanstd(H_q) < 0.05
    # H(q=0) should be ~0.5
    idx0 = np.where(qvals == 0)[0][0]
    assert abs(H_q[idx0] - 0.5) < 0.1


def test_mf_dfa_errors():
    """Verify that MF_DFA raises when data too short or invalid mode."""
    short = np.arange(5)
    with pytest.raises(ValueError):
        functions.MF_DFA(short, q_mode='typical', num=5, min_n=4, min_segments=2)
    ts = white_noise(1000)
    with pytest.raises(ValueError):
        # invalid q_mode
        functions.MF_DFA(ts, q_mode='invalid_mode')


# ─────────────────────────────────────────────────────────────────────────────
# 4) LYAUPUNOV Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_lyapunov_logistic_map():
    """
    For logistic map at r=3.99, expect positive Lyapunov exponent (chaotic).
    Check that lyap_exp > 0.
    """
    ts = logistic_map(length=2000, r=3.99, x0=0.4)
    lyap_exp, divergence, times = functions.lyapunov(ts, dim=3, tau=1, fs=1.0, max_iter=200, theiler=1)
    assert not np.isnan(lyap_exp)
    assert lyap_exp > 0.0
    # divergence and times arrays should have correct shape
    assert divergence.shape == (200,)
    assert times.shape == (200,)


def test_lyapunov_errors():
    """Check that lyapunov raises on invalid parameters."""
    ts2d = np.vstack([np.arange(100), np.arange(100)]).T
    with pytest.raises(ValueError):
        functions.lyapunov(ts2d)  # not 1D
    with pytest.raises(ValueError):
        functions.lyapunov(white_noise(100), dim=1, tau=1)  # dim<2
    with pytest.raises(ValueError):
        functions.lyapunov(white_noise(100), dim=3, tau=0)  # tau<1
    with pytest.raises(ValueError):
        functions.lyapunov(white_noise(100), dim=3, tau=1, fs=0)  # fs<=0
    with pytest.raises(ValueError):
        functions.lyapunov(white_noise(20), dim=3, tau=10)  # too short to embed


# ─────────────────────────────────────────────────────────────────────────────
# 5) WTMM Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_wtmm_scaling_simple():
    """
    For white noise, the WTMM slope (scaling) should be near zero or small, 
    since no clear multiscale structure.
    """
    ts = white_noise(length=1024, seed=7)
    slope, coeffs = functions.WTMM(ts, modulus='mean', wavelet='cmor1-1.5', min_scale=2, max_fraction=0.25, num=20)
    # Slope might be close to 0
    assert abs(slope) < 0.5


@pytest.mark.parametrize("wavelet", ['invalid_wavelet', 123])
def test_wtmm_errors(wavelet):
    """Invalid wavelet specification or bad type should raise ValueError."""
    ts = white_noise(256, seed=2)
    with pytest.raises(ValueError):
        functions.WTMM(ts, modulus='mean', wavelet=wavelet)


# ─────────────────────────────────────────────────────────────────────────────
# 6) RQA & Related Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_recurrence_rate_and_metrics():
    """Use a simple periodic signal to build a recurrence plot and check RQA metrics."""
    ts = pure_sine(length=500, freq=3, noise_amp=0.0)
    rp, metrics = functions.RQA(ts, threshold='point', percentage=10.0, min_length=2)
    # Recurrence plot must be square and binary
    assert rp.ndim == 2 and rp.shape[0] == rp.shape[1]
    assert set(np.unique(rp)).issubset({0,1})

    # Check recurrence_rate ∈ [0,1]
    rr = functions.recurrence_rate(rp)
    assert 0.0 <= rr <= 1.0

    # determinism ≤ 1
    det = functions.determinism(rp, min_length=2)
    assert 0.0 <= det <= 1.0

    # l_max is integer ≥ 0
    lmax = functions.l_max(rp, min_length=2)
    assert isinstance(lmax, int) and lmax >= 0

    # entropy, divergence, laminarity, trapping_time are ≥ 0
    assert functions.entropy(rp, min_length=2) >= 0.0
    div = functions.divergence(rp, min_length=2)
    assert div >= 0.0
    lam = functions.laminarity(rp, min_length=2)
    assert 0.0 <= lam <= 1.0
    tt = functions.trapping_time(rp, min_length=2)
    assert tt >= 0.0

    # metrics dict keys
    for key in ['RR', 'DET', 'ENTR', 'L_max', 'DIV', 'LAM', 'TT']:
        assert key in metrics


def test_rqa_errors():
    """Invalid RQA parameters should raise ValueError."""
    with pytest.raises(ValueError):
        # series not 1D or too short
        functions.RQA(np.array([[1,2],[3,4]]))
    with pytest.raises(ValueError):
        # percentage out of range
        functions.RQA(np.arange(100), threshold='point', percentage=200.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7) Utils Tests (Some Key Helpers)
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_adaptive_nvals_and_dfa_consistency():
    """Ensure _generate_adaptive_nvals and _compute_fluctuations behave as expected."""
    ts = white_noise(500, seed=3)
    nvals = utils._generate_adaptive_nvals(ts, num=10, min_n=5, min_segments=5)
    assert isinstance(nvals, np.ndarray)
    assert np.all(nvals >= 2) and np.all(nvals < ts.size)

    F_n = utils._compute_fluctuations(ts, nvals)
    assert isinstance(F_n, np.ndarray)
    assert F_n.shape == (nvals.size,)
    assert np.all(np.isfinite(F_n[nvals < ts.size//2]))  # at least some valid points


def test_confidence_intervals_boundaries():
    """Test that confidence intervals raise on invalid alpha or too short input."""
    short = np.arange(30)
    with pytest.raises(ValueError):
        utils._confidence_interval_rs(short, alpha=95)  # needs length ≥ 50
    ts = white_noise(200, seed=1)
    with pytest.raises(ValueError):
        utils._confidence_interval_rs(ts, alpha=123)  # invalid alpha
    # DFA CI: length ≥ 10
    with pytest.raises(ValueError):
        utils._confidence_interval_dfa(np.arange(5), alpha=95)
    with pytest.raises(ValueError):
        utils._confidence_interval_dfa(ts, alpha=0.32)  # invalid alpha


# ─────────────────────────────────────────────────────────────────────────────
# 8) Interpreter Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_interpret_hurst_basic():
    """Check interpret_hurst returns a string and contains expected keywords."""
    ts = white_noise(1000, seed=8)
    txt = interpreters.interpret_hurst(ts, use_confidence_interval=False)
    assert isinstance(txt, str) and "Hurst exponent" in txt

    # Use CI and an AR(1) persistent process
    ts2 = ar1_process(1000, phi=0.9, seed=9)
    txt2 = interpreters.interpret_hurst(ts2, use_confidence_interval=True, alpha=95)
    assert isinstance(txt2, str)
    # Because AR(1) with phi=0.9 is persistent, we expect the word “persistent” somewhere
    assert "persistent" in txt2.lower()


def test_interpret_DFA_basic():
    ts = white_noise(2000, seed=10)
    txt = interpreters.interpret_DFA(ts, use_confidence_interval=False)
    assert isinstance(txt, str) and "DFA exponent" in txt

    # Now AR(1) persistent
    ts2 = ar1_process(2000, phi=0.8, seed=11)
    txt2 = interpreters.interpret_DFA(ts2, use_confidence_interval=True, alpha=95)
    assert "persistent" in txt2.lower()


def test_interpret_mf_dfa_errors_and_output():
    """Check interpret_mf_dfa for shape mismatch and normal output."""
    Hq = np.array([0.6, 0.61, np.nan])
    qv = np.array([-1, 0, 1])
    out = interpreters.interpret_mf_dfa(Hq, qv)
    assert isinstance(out, dict)
    assert 'H_min' in out and 'monofractal' in out

    with pytest.raises(ValueError):
        interpreters.interpret_mf_dfa(np.array([0.5, np.nan]), np.array([0.1]))  # shape mismatch
    with pytest.raises(ValueError):
        interpreters.interpret_mf_dfa(np.array([np.nan, np.nan]), np.array([-1, 1]))


def test_interpret_lyapunov_and_rqa_summary():
    # chaotic series → positive
    lyap = 0.2
    cl = interpreters.interpret_lyapunov(lyap)
    assert "chaos" in cl.lower()

    cl2 = interpreters.interpret_lyapunov(np.nan)
    assert cl2 == 'Undefined'

    # RQA summaries
    metrics = {'RR':0.12, 'DET':0.42, 'L_max':15}
    summ = interpreters.summarize_rqa(metrics)
    assert isinstance(summ, str) and "RQA Metrics Summary" in summ

    table = interpreters.table_rqa(metrics)
    assert isinstance(table, str) and "RR |" in table

    # Empty metrics for table
    assert interpreters.table_rqa({}) == ""


# ─────────────────────────────────────────────────────────────────────────────
# 9) Plotting Smoke Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_plotting_smoke():
    """A simple “smoke test” to ensure plotting functions run without error (we don't check visual output)."""
    ts = white_noise(500, seed=15)

    # Hurst plot
    ax1 = plotting.plot_hurst(ts, num=10, min_n=5, min_segments=5, figsize=(4,3))
    assert ax1 is not None

    # DFA plot
    ax2 = plotting.plot_dfa(ts, num=10, min_n=5, min_segments=5, figsize=(4,3), show_fit=False)
    assert ax2 is not None

    # MF-DFA: plot a dummy H(q)
    Hq = np.linspace(0.4, 0.6, 11)
    qv = np.linspace(-5, 5, 11)
    ax3 = plotting.plot_mf_dfa(Hq, qv, figsize=(4,3))
    assert ax3 is not None

    # Lyapunov: pass a dummy divergence curve
    div = np.linspace(0, 1, 50)
    times = np.arange(50)
    ax4 = plotting.plot_lyapunov(div, times=times, figsize=(4,3), fit_slope=(0.02, 0.0))
    assert ax4 is not None

    # WTMM plot
    fig5, ax5 = plotting.plot_wtmm(ts, wavelets=['cmor1-1.5'], num=10, min_scale=2, max_fraction=0.2, figsize=(4,3))
    assert ax5 is not None

    # Recurrence plot (use sine without noise)
    rp = utils._generate_recurrence_plot(pure_sine(100), threshold='point', percentage=10.0)
    ax6 = plotting.plot_recurrence(rp, figsize=(3,3))
    assert ax6 is not None

    # Histograms (dummy data)
    diag = [2,2,3,4,5,2]
    vert = [1,1,2,2,3]
    axd, axv = plotting.plot_line_length_histograms(diag, vert, bins=5, figsize=(4,2))
    assert axd is not None and axv is not None
