"""Build per-pair correlation summaries (used by analysis and Streamlit results)."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def build_pairwise_summary(
    channel_names: List[str],
    static_corr: np.ndarray,
    drift: np.ndarray,
    correlation_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """One summary row per unordered channel pair (i < j in channel order).

    ``configured_rho`` is set only when that pair appears in ``correlation_entries``;
    otherwise it is ``None`` (no copula target in YAML).
    """
    c = len(channel_names)
    if c < 2:
        return []

    name_to_idx = {n: i for i, n in enumerate(channel_names)}
    configured: Dict[Tuple[str, str], float] = {}
    for entry in correlation_entries:
        pair = list(entry.get("channels") or [])
        if len(pair) != 2:
            continue
        a, b = str(pair[0]), str(pair[1])
        if a not in name_to_idx or b not in name_to_idx:
            continue
        key = tuple(sorted((a, b)))
        rho = float(entry.get("rho", 0.0))
        configured[key] = max(-1.0, min(1.0, rho))

    out: List[Dict[str, Any]] = []
    for i in range(c):
        for j in range(i + 1, c):
            a, b = channel_names[i], channel_names[j]
            key = tuple(sorted((a, b)))
            rho_tgt: Optional[float] = configured.get(key)
            pair_drift = float(drift[i, j])
            if abs(pair_drift) < 0.05:
                drift_label = "stable"
            elif pair_drift > 0:
                drift_label = f"+{pair_drift:.2f}"
            else:
                drift_label = f"{pair_drift:.2f}"
            out.append(
                {
                    "pair": [a, b],
                    "configured_rho": rho_tgt,
                    "observed_rho": float(static_corr[i, j]),
                    "drift": pair_drift,
                    "drift_label": drift_label,
                }
            )
    return out
