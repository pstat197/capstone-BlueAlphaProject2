"""Saturation and adstock curve previews for the channel form.

Small, illustrative sparklines that mirror the math in
``scripts/revenue_simulation/revenue_generation.py``. The previews read the
user's current widget state (session state) so they update live as the user
tweaks sliders. They deliberately use synthetic x-ranges — the goal is to
show curve *shape*, not absolute revenue magnitudes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app.ui_form_state import (
    adstock_weights_key,
    effective_curve_type,
    parse_optional_num,
    parse_weights_csv,
    pc_field_key,
)
from app.ui_helpers import get_at, path_string_to_parts


# ---------------------------------------------------------------------------
# Parameter extraction (reads live widget state, falls back to YAML, then schema default)
# ---------------------------------------------------------------------------


def _resolve_numeric_param(
    i: int,
    path_suffix: str,
    data: Dict[str, Any],
    *,
    as_int: bool = False,
    fallback: float = 0.0,
) -> float:
    """Resolve one numeric per-channel parameter.

    Priority: current widget session state -> YAML config value -> fallback.
    ``parse_optional_num`` handles both numeric (slider) and string (text_input)
    session state transparently.
    """
    key = pc_field_key(i, path_suffix, list_index=None)
    raw = st.session_state.get(key, None)
    val, ok = parse_optional_num(raw, as_int=as_int)
    if ok and val is not None:
        return float(val)

    parts = path_string_to_parts(f"channel_list.{i}.{path_suffix}")
    cur = get_at(data, parts)
    if isinstance(cur, (int, float)) and not isinstance(cur, bool):
        return float(cur)
    return float(fallback)


def _resolve_weights(i: int, data: Dict[str, Any]) -> np.ndarray:
    """Resolve the adstock weights list for channel ``i`` (session -> YAML -> [1.0])."""
    key = adstock_weights_key(i)
    raw = st.session_state.get(key, None)
    parsed, ok = parse_weights_csv(raw)
    if ok and parsed is not None:
        return np.asarray(parsed, dtype=float)
    parts = path_string_to_parts(f"channel_list.{i}.channel.adstock_decay_config.weights")
    cur = get_at(data, parts)
    if isinstance(cur, list) and cur:
        try:
            return np.asarray([float(x) for x in cur], dtype=float)
        except (TypeError, ValueError):
            pass
    return np.asarray([1.0], dtype=float)


# ---------------------------------------------------------------------------
# Curve math (mirrors scripts/revenue_simulation/revenue_generation.py)
# ---------------------------------------------------------------------------


def _saturation_preview(
    sat_type: str,
    i: int,
    data: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """Return (x, y, x_label) for the saturation preview, or None if not plottable."""
    if sat_type == "linear":
        slope = _resolve_numeric_param(i, "channel.saturation_config.slope", data, fallback=1.0)
        x_max = 100.0
        x = np.linspace(0, x_max, 80)
        y = slope * x
        return x, y, "impressions (arbitrary units)"

    if sat_type == "hill":
        slope = _resolve_numeric_param(i, "channel.saturation_config.slope", data, fallback=1.0)
        K = _resolve_numeric_param(i, "channel.saturation_config.K", data, fallback=1.0)
        if K <= 0:
            K = 1.0
        x_max = max(K * 3.0, 10.0)
        x = np.linspace(0, x_max, 80)
        xa = np.power(x, slope)
        Ka = np.power(K, slope)
        y = xa / (xa + Ka + 1e-9)
        return x, y, "impressions (K-relative)"

    if sat_type == "diminishing_returns":
        beta = _resolve_numeric_param(i, "channel.saturation_config.beta", data, fallback=0.0)
        x_max = 100.0 if beta <= 0 else max(10.0, 5.0 / beta)
        x = np.linspace(0, x_max, 80)
        y = x / (1.0 + beta * x)
        return x, y, "impressions (arbitrary units)"

    return None


def _adstock_kernel_preview(
    ad_type: str,
    i: int,
    data: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (lags, weights) for the adstock kernel preview."""
    lag_raw = _resolve_numeric_param(i, "channel.adstock_decay_config.lag", data, as_int=True, fallback=0.0)
    lag = max(0, int(round(lag_raw)))

    if ad_type == "linear":
        n = lag + 1
        lags = np.arange(n)
        w = np.full(n, 1.0 / n)
        return lags, w

    if ad_type in ("geometric", "exponential"):
        lam = _resolve_numeric_param(i, "channel.adstock_decay_config.lambda", data, fallback=0.5)
        lags = np.arange(lag + 1)
        w = np.power(max(0.0, lam), lags)
        return lags, w

    if ad_type == "weighted":
        w = _resolve_weights(i, data)
        lags = np.arange(len(w))
        return lags, w

    return None


# ---------------------------------------------------------------------------
# Streamlit renderers
# ---------------------------------------------------------------------------


def render_saturation_preview(
    i: int,
    data: Dict[str, Any],
    sat_options: list,
) -> None:
    """Render a small saturation curve preview below the saturation knobs."""
    sat_type = effective_curve_type(i, "channel.saturation_config.type", data, list(sat_options))
    out = _saturation_preview(sat_type, i, data)
    if out is None:
        return
    x, y, x_label = out
    df = pd.DataFrame({x_label: x, "saturated response": y}).set_index(x_label)
    st.caption(f"Preview — current *{sat_type}* curve (shape, not absolute scale).")
    st.line_chart(df, height=160, width="stretch")


def render_adstock_preview(
    i: int,
    data: Dict[str, Any],
    ad_options: list,
) -> None:
    """Render the adstock kernel (weight per lag) below the adstock knobs."""
    ad_type = effective_curve_type(i, "channel.adstock_decay_config.type", data, list(ad_options))
    out = _adstock_kernel_preview(ad_type, i, data)
    if out is None:
        return
    lags, w = out
    if len(lags) == 0:
        return
    df = pd.DataFrame({"lag (weeks)": lags, "kernel weight": w}).set_index("lag (weeks)")
    label_hint = (
        "normalized" if ad_type == "linear"
        else "geometric decay" if ad_type in ("geometric", "exponential")
        else "user weights"
    )
    st.caption(f"Preview — *{ad_type}* kernel weight per lag ({label_hint}).")
    st.bar_chart(df, height=160, width="stretch")
