"""Seasonality UI: one radio for all modes, reference expander, cycle table + chart."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui_helpers import get_at, path_string_to_parts
from app.ui_help_markdown import SEASONALITY_OVERVIEW_MD, SEASONALITY_TYPES_GUIDE_MD
from scripts.revenue_simulation.seasonality_fit import evaluate_deterministic_fourier


def _basic_cycle_df_key(i: int) -> str:
    """Session key for the editable cycle table (not a widget key — safe to assign)."""
    return f"sea_basic_df_{i}"


def _basic_cycle_editor_key(i: int) -> str:
    """``st.data_editor`` widget key only — must never be assigned via ``st.session_state[...] =``."""
    return f"sea_basic_editor_{i}"


def yaml_sync_from_form() -> None:
    st.session_state["yaml_manual_edit"] = False


def _infer_sea_type_from_saved_sea(sea: Dict[str, Any]) -> str:
    """Map saved ``seasonality_config`` to the unified radio value."""
    if not sea:
        return "none"
    t = str(sea.get("type", "")).strip().lower()
    if t == "fourier":
        return "cycle" if sea.get("coefficients") is not None else "fourier"
    if t in ("sin", "categorical"):
        return t
    return "none"


def ensure_seasonality_widgets_warmed(cfg: Dict[str, Any]) -> None:
    """
    Initialize ``sea_type_{i}`` from ``cfg`` before merge runs (e.g. YAML preview).

    Migrates older sessions that used ``sea_ui_mode_{i}`` / ``sea_basic_sub_{i}``.
    """
    channels = cfg.get("channel_list") or []
    for i, item in enumerate(channels):
        if not isinstance(item, dict):
            continue
        ch = item.get("channel") if isinstance(item.get("channel"), dict) else item
        if not isinstance(ch, dict):
            continue
        sea = dict(ch.get("seasonality_config") or {})
        kt = f"sea_type_{i}"
        if kt not in st.session_state:
            if st.session_state.get(f"sea_ui_mode_{i}") == "Basic":
                st.session_state[kt] = (
                    "cycle" if st.session_state.get(f"sea_basic_sub_{i}") == "cycle" else "none"
                )
            else:
                st.session_state[kt] = _infer_sea_type_from_saved_sea(sea)


def _current_seasonality_dict(data: Dict[str, Any], i: int) -> Dict[str, Any]:
    parts = path_string_to_parts(f"channel_list.{i}.channel.seasonality_config")
    raw = get_at(data, parts)
    return dict(raw) if isinstance(raw, dict) else {}


def _initial_mults_for_cycle(sea: Dict[str, Any], p: int) -> List[float]:
    """Build default multipliers for one cycle of length ``p`` from saved config."""
    p = max(2, min(52, int(p)))
    if sea.get("type") == "fourier" and sea.get("coefficients"):
        period = int(sea.get("period", p))
        t_full = np.arange(period, dtype=float)
        m_full = evaluate_deterministic_fourier(t_full, sea)
        if p == period:
            return [float(x) for x in m_full]
        old_x = np.linspace(0.0, 1.0, period, endpoint=False)
        new_x = np.linspace(0.0, 1.0, p, endpoint=False)
        interp = np.interp(new_x, old_x, np.asarray(m_full, dtype=float))
        return [float(x) for x in interp]
    return [1.0] * p


def warm_basic_cycle_editor_if_needed(i: int, cfg: Dict[str, Any]) -> None:
    """Seed cycle ``DataFrame`` state before merge if the cycle editor never rendered this session."""
    df_key = _basic_cycle_df_key(i)
    if df_key in st.session_state:
        return
    lst = cfg.get("channel_list") or []
    if i >= len(lst):
        return
    item = lst[i]
    ch = item.get("channel") if isinstance(item.get("channel"), dict) else item
    if not isinstance(ch, dict):
        return
    sea = dict(ch.get("seasonality_config") or {})
    p_key = f"sea_basic_P_{i}"
    p = int(st.session_state.get(p_key, 8))
    p = max(2, min(52, p))
    mults = _initial_mults_for_cycle(sea, p)
    st.session_state[df_key] = pd.DataFrame(
        {"Week in cycle": list(range(1, p + 1)), "Multiplier": mults}
    )
    st.session_state[f"sea_basic_lastP_{i}"] = p


def _ensure_basic_cycle_dataframe(i: int, sea: Dict[str, Any]) -> None:
    """Initialize or resize the cycle table backing store (not the data_editor widget key)."""
    p_key = f"sea_basic_P_{i}"
    last_key = f"sea_basic_lastP_{i}"
    df_key = _basic_cycle_df_key(i)
    legacy_ed = f"sea_basic_ed_{i}"
    if legacy_ed in st.session_state and df_key not in st.session_state:
        leg = st.session_state.pop(legacy_ed, None)
        if isinstance(leg, pd.DataFrame):
            st.session_state[df_key] = leg
        elif isinstance(leg, dict):
            st.session_state[df_key] = pd.DataFrame(leg)

    p = int(st.session_state.get(p_key, 8))
    p = max(2, min(52, p))
    st.session_state[p_key] = p

    need_build = df_key not in st.session_state or st.session_state.get(last_key) != p
    if need_build:
        mults = _initial_mults_for_cycle(sea, p)
        st.session_state[df_key] = pd.DataFrame(
            {"Week in cycle": list(range(1, p + 1)), "Multiplier": mults}
        )
        st.session_state[last_key] = p


def _read_basic_cycle_multipliers(i: int) -> List[float]:
    raw = st.session_state.get(_basic_cycle_df_key(i))
    if raw is None:
        return [1.0, 1.0]
    if isinstance(raw, dict):
        df = pd.DataFrame(raw)
    elif isinstance(raw, pd.DataFrame):
        df = raw
    else:
        return [1.0, 1.0]
    if "Multiplier" not in df.columns:
        return [1.0, 1.0]
    out: List[float] = []
    for x in df["Multiplier"].tolist():
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            out.append(1.0)
    return out if len(out) >= 2 else [1.0, 1.0]


def _init_seasonality_param_defaults(i: int, data: Dict[str, Any]) -> None:
    """Default numeric / pattern widget values from the current channel config."""
    sea = _current_seasonality_dict(data, i)
    st.session_state.setdefault(f"sea_amp_{i}", float(sea.get("amplitude", 0.2)))
    st.session_state.setdefault(f"sea_period_{i}", int(sea.get("period", 52)))
    st.session_state.setdefault(f"sea_phase_{i}", float(sea.get("phase", 0.0)))
    st.session_state.setdefault(f"sea_k_{i}", int(sea.get("K", 2)))
    st.session_state.setdefault(f"sea_scale_{i}", float(sea.get("scale", 0.1)))
    pattern = sea.get("pattern", [1.0, 1.0, 1.0, 1.0])
    if not isinstance(pattern, list):
        pattern = [1.0, 1.0, 1.0, 1.0]
    st.session_state.setdefault(f"sea_pattern_{i}", ", ".join(f"{float(x):g}" for x in pattern))


def _sea_mode_label(v: str) -> str:
    return {
        "none": "None",
        "cycle": "Repeating cycle (table → fitted Fourier)",
        "sin": "Sin (saved as fitted Fourier)",
        "fourier": "Random Fourier (K / scale)",
        "categorical": "Comma pattern → fitted Fourier",
    }.get(v, v)


def render_seasonality_block(i: int, data: Dict[str, Any], week_range: int) -> None:
    """
    Single radio for every seasonality mode; one expander documents all modes.

    The chart is preview-only; edit multipliers in the table (no native point-drag in Streamlit).
    """
    sea = _current_seasonality_dict(data, i)
    _ = week_range

    ensure_seasonality_widgets_warmed({"channel_list": data.get("channel_list") or []})
    _init_seasonality_param_defaults(i, data)

    st.markdown("##### Baseline trend & seasonality")
    with st.expander(
        "How seasonality works (all modes — click to open)",
        expanded=False,
        key=f"sea_explain_all_{i}",
    ):
        st.markdown(SEASONALITY_OVERVIEW_MD)
        st.markdown(SEASONALITY_TYPES_GUIDE_MD)

    st.radio(
        "Seasonality",
        options=["none", "cycle", "sin", "fourier", "categorical"],
        format_func=_sea_mode_label,
        key=f"sea_type_{i}",
        help="Choose how baseline revenue is scaled by time. Details are in the expander above.",
        on_change=yaml_sync_from_form,
    )

    sea_type = str(st.session_state.get(f"sea_type_{i}", "none")).strip().lower()
    if sea_type in ("", "none"):
        st.caption("No seasonal multiplier on baseline.")
        return

    if sea_type == "cycle":
        st.caption(
            "Set **cycle length** and edit the **multiplier** column (chart is a preview only). "
            "Merged YAML stores a **fitted Fourier**, not the raw table."
        )
        _ensure_basic_cycle_dataframe(i, sea)
        st.number_input(
            "Cycle length (weeks)",
            min_value=2,
            max_value=52,
            value=int(st.session_state.get(f"sea_basic_P_{i}", 8)),
            step=1,
            key=f"sea_basic_P_{i}",
            help="Pattern repeats every this many weeks.",
            on_change=yaml_sync_from_form,
        )
        df_key = _basic_cycle_df_key(i)
        edited = st.data_editor(
            st.session_state[df_key],
            key=_basic_cycle_editor_key(i),
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "Week in cycle": st.column_config.NumberColumn(disabled=True),
                "Multiplier": st.column_config.NumberColumn(
                    min_value=0.01,
                    max_value=5.0,
                    step=0.01,
                    format="%.4f",
                ),
            },
            on_change=yaml_sync_from_form,
        )
        st.session_state[df_key] = edited
        mults = _read_basic_cycle_multipliers(i)
        fig = go.Figure(
            go.Scatter(
                x=list(range(1, len(mults) + 1)),
                y=mults,
                mode="lines+markers",
                name="Multiplier",
                marker=dict(size=10),
            )
        )
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=30, b=40),
            yaxis_title="Baseline multiplier",
            xaxis_title="Week index in cycle",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"sea_basic_chart_{i}")
        return

    if sea_type == "sin":
        a, b, c = st.columns(3)
        with a:
            st.number_input(
                "Amplitude",
                min_value=0.0,
                max_value=2.0,
                step=0.01,
                key=f"sea_amp_{i}",
                help="Height of seasonal oscillation around baseline.",
                on_change=yaml_sync_from_form,
            )
        with b:
            st.number_input(
                "Period",
                min_value=1,
                max_value=520,
                step=1,
                key=f"sea_period_{i}",
                help="Cycle length in weeks.",
                on_change=yaml_sync_from_form,
            )
        with c:
            st.number_input(
                "Phase",
                min_value=-520.0,
                max_value=520.0,
                step=1.0,
                key=f"sea_phase_{i}",
                help="Horizontal shift of the seasonal wave.",
                on_change=yaml_sync_from_form,
            )
        return

    if sea_type == "fourier":
        a, b, c = st.columns(3)
        with a:
            st.number_input(
                "Period",
                min_value=1,
                max_value=520,
                step=1,
                key=f"sea_period_{i}",
                help="Base cycle length in weeks.",
                on_change=yaml_sync_from_form,
            )
        with b:
            st.number_input(
                "Harmonics (K)",
                min_value=1,
                max_value=20,
                step=1,
                key=f"sea_k_{i}",
                help="Number of harmonics for random draw.",
                on_change=yaml_sync_from_form,
            )
        with c:
            st.number_input(
                "Scale",
                min_value=0.0,
                max_value=2.0,
                step=0.01,
                key=f"sea_scale_{i}",
                help="Overall magnitude of the random Fourier component.",
                on_change=yaml_sync_from_form,
            )
        return

    if sea_type == "categorical":
        st.text_input(
            "Pattern multipliers (comma-separated)",
            key=f"sea_pattern_{i}",
            help="Repeating baseline multipliers; saved as a smoothed Fourier fit.",
            on_change=yaml_sync_from_form,
        )
