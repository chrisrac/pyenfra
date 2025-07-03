"""
pyenfra: A Python package for fractal and complexity analysis of time series.

Includes:
    - hurst, DFA, MF-DFA
    - Lyapunov exponent estimation
    - WTMM (Wavelet Transform Modulus Maxima)
    - RQA (Recurrence Quantification Analysis)
    - Plotting utilities
    - Interpretation helpers

Repository: 
    https://github.com/chrisrac/pyenfra

Author: Krzysztof Raczynski
    https://github.com/chrisrac

"""

__version__ = "0.4.0"

# Public API
from .functions import (
    hurst,
    DFA,
    MF_DFA,
    lyapunov,
    WTMM,
    RQA,
    recurrence_rate,
    determinism,
    l_max,
    entropy,
    divergence,
    laminarity,
    trapping_time
)

from .utils import (
    _check_array_1d, 
    _check_positive_integer, 
    _generate_adaptive_nvals,
    _generate_nvals, 
    _rescaled_range_analysis, 
    _compute_fluctuations,
    _generate_qvals, 
    _check_q_mode, 
    _confidence_interval_rs,
    _confidence_interval_dfa, 
    _MF_DFA, 
    _phase_space_embedding,
    _generate_wtmm_scales, 
    _WTMM, 
    _compute_log_fit,
    _validate_binary_sequence, 
    _get_line_lengths, 
    _count_diagonal_lines,
    _count_vertical_lines, 
    _generate_recurrence_plot, 
    _compute_diagonal_line_lengths,
    _compute_vertical_line_lengths
)

from .plotting import (
    plot_hurst,
    plot_dfa,
    plot_mf_dfa,
    plot_lyapunov,
    plot_wtmm,
    plot_recurrence,
    plot_line_length_histograms
)

from .interpreters import (
    interpret_hurst,
    interpret_DFA,
    interpret_mf_dfa,
    interpret_lyapunov,
    summarize_rqa,
    table_rqa
)

__all__ = [
    # version
    '__version__',
    # functions
    'hurst',
    'DFA',
    'MF_DFA',
    'lyapunov',
    'WTMM',
    'RQA',
    'recurrence_rate',
    'determinism',
    'l_max',
    'entropy',
    'divergence',
    'laminarity',
    'trapping_time',
    # utils
    '_check_array_1d', 
    '_check_positive_integer', 
    '_generate_adaptive_nvals',
    '_generate_nvals', 
    '_rescaled_range_analysis', 
    '_compute_fluctuations',
    '_generate_qvals', 
    '_check_q_mode', 
    '_confidence_interval_rs',
    '_confidence_interval_dfa', 
    '_MF_DFA', 
    '_phase_space_embedding',
    '_generate_wtmm_scales', 
    '_WTMM', 
    '_compute_log_fit',
    '_validate_binary_sequence', 
    '_get_line_lengths', 
    '_count_diagonal_lines',
    '_count_vertical_lines', 
    '_generate_recurrence_plot', 
    '_compute_diagonal_line_lengths',
    '_compute_vertical_line_lengths',
    # plotting
    'plot_hurst',
    'plot_dfa',
    'plot_mf_dfa',
    'plot_lyapunov',
    'plot_wtmm',
    'plot_recurrence',
    'plot_line_length_histograms',
    # interpreters
    'interpret_hurst',
    'interpret_DFA',
    'interpret_mf_dfa',
    'interpret_lyapunov',
    'summarize_rqa',
    'table_rqa'
]