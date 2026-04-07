"""
Post-DGP correlation analysis for spend matrices.

Provides static and rolling correlation, drift detection, and
per-channel average absolute correlation (multicollinearity risk).
All functions are pure numpy — no plotting. A future frontend layer
will consume these return values to render the dashboard.
"""
from typing import Dict, List, Tuple

import numpy as np

from scripts.synth_input_classes.input_configurations import InputConfigurations


def compute_static_correlation(spend: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix across all T weeks.

    Args:
        spend: (T, num_channels) spend matrix.
    Returns:
        (num_channels, num_channels) correlation matrix.
    """
    return np.corrcoef(spend.T)


def compute_rolling_correlation(
    spend: np.ndarray,
    window: int = 12,
) -> np.ndarray:
    """Rolling Pearson correlation over a sliding window.

    Args:
        spend: (T, num_channels) spend matrix.
        window: number of weeks per window.
    Returns:
        (T - window, num_channels, num_channels) tensor.
        Entry [t] is the correlation matrix for weeks [t, t+window).
    """
    T, C = spend.shape
    if window > T:
        raise ValueError(f"Window ({window}) exceeds number of weeks ({T})")
    results = np.zeros((T - window, C, C))
    for t in range(T - window):
        results[t] = np.corrcoef(spend[t : t + window].T)
    return results


def compute_pairwise_drift(
    rolling_corr: np.ndarray,
    num_edge_windows: int = 5,
) -> np.ndarray:
    """Change in rolling rho between the first and last edge windows.

    Args:
        rolling_corr: (N, C, C) tensor from compute_rolling_correlation.
        num_edge_windows: how many windows to average at each end.
    Returns:
        (C, C) matrix where entry [i,j] = mean_rho_last - mean_rho_first.
        Positive means correlation increased over time.
    """
    n = min(num_edge_windows, rolling_corr.shape[0])
    start_mean = rolling_corr[:n].mean(axis=0)
    end_mean = rolling_corr[-n:].mean(axis=0)
    return end_mean - start_mean


def compute_avg_abs_correlation(
    static_corr: np.ndarray,
    channel_names: List[str],
) -> Dict[str, float]:
    """Average absolute off-diagonal correlation per channel.

    Flags which channel is most entangled with the rest — a
    multicollinearity risk indicator for Meridian.

    Returns:
        {channel_name: avg_abs_rho} sorted descending.
    """
    C = static_corr.shape[0]
    result = {}
    for i, name in enumerate(channel_names):
        off_diag = [abs(static_corr[i, j]) for j in range(C) if j != i]
        result[name] = float(np.mean(off_diag)) if off_diag else 0.0
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


def compute_most_correlated_channel(
    static_corr: np.ndarray,
    channel_names: List[str],
) -> str:
    """Return the channel with the highest average absolute correlation."""
    avg = compute_avg_abs_correlation(static_corr, channel_names)
    return max(avg, key=avg.get)


def analyze_spend_correlations(
    config: InputConfigurations,
    spend: np.ndarray,
    window: int = 12,
) -> Dict:
    """Run full correlation analysis and return all results in a dict.

    This is the main entry point. The returned dict contains everything
    a frontend needs to render the dashboard shown in the design mockups.

    Returns dict with keys:
        channel_names: List[str]
        static_corr: (C, C) ndarray
        rolling_corr: (N, C, C) ndarray
        drift: (C, C) ndarray
        avg_abs_corr: {name: float}
        most_correlated_channel: str
        pairwise_summary: list of dicts per configured pair
        window: int
    """
    channel_names = [ch.get_channel_name() for ch in config.get_channel_list()]
    T, C = spend.shape

    static_corr = compute_static_correlation(spend)

    effective_window = min(window, T)
    rolling_corr = np.empty((0, C, C))
    drift = np.zeros((C, C))
    if effective_window < T:
        rolling_corr = compute_rolling_correlation(spend, effective_window)
        drift = compute_pairwise_drift(rolling_corr)

    avg_abs = compute_avg_abs_correlation(static_corr, channel_names)
    most_corr = compute_most_correlated_channel(static_corr, channel_names)

    name_to_idx = {n: i for i, n in enumerate(channel_names)}
    pairwise_summary = []
    for entry in config.get_correlations():
        pair = entry["channels"]
        configured_rho = entry["rho"]
        i, j = name_to_idx[pair[0]], name_to_idx[pair[1]]
        observed_rho = float(static_corr[i, j])
        pair_drift = float(drift[i, j])
        if abs(pair_drift) < 0.05:
            drift_label = "stable"
        elif pair_drift > 0:
            drift_label = f"+{pair_drift:.2f}"
        else:
            drift_label = f"{pair_drift:.2f}"
        pairwise_summary.append({
            "pair": pair,
            "configured_rho": configured_rho,
            "observed_rho": observed_rho,
            "drift": pair_drift,
            "drift_label": drift_label,
        })

    return {
        "channel_names": channel_names,
        "static_corr": static_corr,
        "rolling_corr": rolling_corr,
        "drift": drift,
        "avg_abs_corr": avg_abs,
        "most_correlated_channel": most_corr,
        "pairwise_summary": pairwise_summary,
        "window": effective_window,
    }


def print_correlation_report(results: Dict) -> None:
    """Print a human-readable summary to stdout."""
    names = results["channel_names"]
    static = results["static_corr"]
    C = len(names)

    print("\n=== SPEND CORRELATION ANALYSIS ===\n")

    print(f"  Rolling window:          {results['window']} weeks")
    print(f"  Most correlated channel: {results['most_correlated_channel']}")
    avg_all = np.mean([v for v in results["avg_abs_corr"].values()])
    print(f"  Avg pairwise |rho|:      {avg_all:.2f}")

    print("\n--- Static Correlation Matrix ---")
    header = "".rjust(16) + "".join(n.rjust(12) for n in names)
    print(header)
    for i in range(C):
        row = names[i].rjust(16)
        for j in range(C):
            row += f"{static[i, j]:12.3f}"
        print(row)

    if results["pairwise_summary"]:
        print("\n--- Pairwise Summary + Drift ---")
        for p in results["pairwise_summary"]:
            print(
                f"  {p['pair'][0]} / {p['pair'][1]}:  "
                f"rho={p['observed_rho']:.2f}  "
                f"(configured={p['configured_rho']:.2f})  "
                f"drift={p['drift_label']}"
            )

    print("\n--- Avg Absolute Correlation per Channel ---")
    for name, val in results["avg_abs_corr"].items():
        print(f"  {name}: {val:.3f}")

    print()
