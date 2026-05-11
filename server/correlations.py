"""Recompute correlation diagnostics from a cached results DataFrame.

`run_with_cache` returns `corr_results=None` on cache hits to keep the fast path cheap.
The React diagnostics route still wants the panel, so we derive the same shape from
the cached spend columns + the user's saved config. This mirrors the Streamlit
fallback in `app.ui_results._build_corr_results_from_cached_df`, just without the
Streamlit dependency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _channel_names_from_config(config: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in config.get("channel_list") or []:
        ch = item.get("channel") if isinstance(item, dict) else item
        if isinstance(ch, dict) and ch.get("channel_name"):
            out.append(str(ch["channel_name"]))
    return out


def derive_correlation_results(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Return a corr_results dict matching what `run_simulation` produces, or None.

    Returns None when fewer than 2 channels exist (correlations are meaningless).
    Pairwise summary is empty when no `correlations:` are configured, but the rest
    of the panel (heatmap + multicollinearity) still has data to render — slightly
    more forgiving than the Streamlit fallback, which gates on configured pairs.
    """
    channel_names = _channel_names_from_config(config)
    if len(channel_names) < 2:
        return None

    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    spend_cols = [f"{name}_spend" for name in channel_names]
    if not all(c in df.columns for c in spend_cols):
        return None

    spend = df[spend_cols].to_numpy(dtype=float)
    t, c = spend.shape
    if t == 0 or c < 2:
        return None

    static_corr = np.corrcoef(spend.T)

    window = min(12, max(2, t - 1))
    rolling_corr = np.empty((0, c, c))
    drift = np.zeros((c, c))
    if t > window:
        rolling_corr = np.array(
            [np.corrcoef(spend[i : i + window].T) for i in range(t - window)]
        )
        edge = min(5, rolling_corr.shape[0])
        drift = rolling_corr[-edge:].mean(axis=0) - rolling_corr[:edge].mean(axis=0)

    avg_abs_corr: Dict[str, float] = {}
    for i, name in enumerate(channel_names):
        off_diag = [abs(static_corr[i, j]) for j in range(c) if j != i]
        avg_abs_corr[name] = float(np.mean(off_diag)) if off_diag else 0.0
    avg_abs_corr = dict(sorted(avg_abs_corr.items(), key=lambda kv: kv[1], reverse=True))
    most_corr = max(avg_abs_corr, key=avg_abs_corr.get) if avg_abs_corr else ""

    name_to_idx = {n: i for i, n in enumerate(channel_names)}
    pairwise_summary: List[Dict[str, Any]] = []
    for entry in config.get("correlations") or []:
        if not isinstance(entry, dict):
            continue
        pair = list(entry.get("channels") or [])
        if len(pair) != 2 or pair[0] not in name_to_idx or pair[1] not in name_to_idx:
            continue
        i, j = name_to_idx[pair[0]], name_to_idx[pair[1]]
        observed_rho = float(static_corr[i, j])
        pair_drift = float(drift[i, j]) if drift.size else 0.0
        if abs(pair_drift) < 0.05:
            drift_label = "stable"
        elif pair_drift > 0:
            drift_label = f"+{pair_drift:.2f}"
        else:
            drift_label = f"{pair_drift:.2f}"
        pairwise_summary.append(
            {
                "pair": pair,
                "configured_rho": float(entry.get("rho", 0.0)),
                "observed_rho": observed_rho,
                "drift": pair_drift,
                "drift_label": drift_label,
            }
        )

    return {
        "channel_names": channel_names,
        "static_corr": static_corr,
        "rolling_corr": rolling_corr,
        "drift": drift,
        "avg_abs_corr": avg_abs_corr,
        "most_correlated_channel": most_corr,
        "pairwise_summary": pairwise_summary,
        "window": int(window),
    }
