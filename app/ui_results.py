"""Results panel: Plotly charts, preview table, YAML snapshot."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.cache import cached_dataframe_schema_ok
from app.ui_chart_constants import ACCENT2, BLUE, CHART_PAL_CVD, ORANGE
from app.ui_yaml_io import yaml_dump


def _norm_series(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def channel_names_from_results_df(df: pd.DataFrame) -> List[str]:
    names: List[str] = []
    for col in df.columns:
        c = str(col).strip().lstrip("\ufeff")
        if c.endswith("_impressions") and c != "total_impressions":
            names.append(c[: -len("_impressions")])
    return names


def results_df_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with stripped / BOM-safe headers so *_revenue lines up with *_impressions."""
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
    return out


def make_charts(
    df: pd.DataFrame,
    *,
    channel: Optional[str] = None,
    overlay: bool = False,
    night: bool = False,
    colorblind: bool = False,
) -> go.Figure:
    c1, c2, c3 = CHART_PAL_CVD if colorblind else (BLUE, ORANGE, ACCENT2)
    grid = "rgba(148,163,184,0.2)" if night else "rgba(15,23,42,0.08)"
    paper = "rgba(15,23,42,0.3)" if night else "rgba(255,255,255,0)"
    plot_bg = "rgba(30,41,59,0.5)" if night else "rgba(248,250,252,0.9)"
    title_color = "#e2e8f0" if night else "#0f172a"

    r_col, sp_col, im_col = "revenue", "total_spend", "total_impressions"
    sub_r, sub_sp, sub_im = "Revenue", "Total spend", "Total impressions"
    if channel:
        rc, sc, ic = f"{channel}_revenue", f"{channel}_spend", f"{channel}_impressions"
        if rc in df.columns and sc in df.columns and ic in df.columns:
            r_col, sp_col, im_col = rc, sc, ic
            sub_r = f"Revenue ({channel})"
            sub_sp = f"Spend ({channel})"
            sub_im = f"Impressions ({channel})"
        else:
            channel = None

    if overlay:
        r = _norm_series(df[r_col])
        sp = _norm_series(df[sp_col])
        im = _norm_series(df[im_col])
        fig = go.Figure()
        nm_r = f"{sub_r} (normalized)"
        nm_sp = f"{sub_sp} (normalized)"
        nm_im = f"{sub_im} (normalized)"
        fig.add_trace(
            go.Scatter(
                x=df["week"],
                y=r,
                name=nm_r,
                mode="lines",
                line=dict(color=c1, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["week"],
                y=sp,
                name=nm_sp,
                mode="lines",
                line=dict(color=c2, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["week"],
                y=im,
                name=nm_im,
                mode="lines",
                line=dict(color=c3, width=2),
            )
        )
        overlay_title = "Series overlaid (min–max normalized per series)"
        if channel:
            overlay_title = f"{overlay_title} · {channel}"
        fig.update_layout(
            height=480,
            title=dict(text=overlay_title, font=dict(color=title_color)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=48, r=24, t=72, b=48),
            paper_bgcolor=paper,
            plot_bgcolor=plot_bg,
            xaxis=dict(gridcolor=grid, title="Week", color=title_color),
            yaxis=dict(
                gridcolor=grid,
                title="Normalized 0–1",
                color=title_color,
                rangemode="tozero",
            ),
        )
        return fig

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(sub_r, sub_sp, sub_im),
    )
    fig.add_trace(
        go.Scatter(
            x=df["week"], y=df[r_col], name=sub_r, mode="lines", line=dict(color=c1, width=2)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["week"], y=df[sp_col], name=sub_sp, mode="lines", line=dict(color=c2, width=2)
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["week"],
            y=df[im_col],
            name=sub_im,
            mode="lines",
            line=dict(color=c3, width=2),
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        height=720,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor=paper,
        plot_bgcolor=plot_bg,
    )
    fig.update_xaxes(title_text="Week", row=3, col=1, gridcolor=grid, color=title_color)
    fig.update_yaxes(gridcolor=grid, color=title_color, rangemode="tozero")
    for a in fig.layout.annotations:
        a.font = dict(color=title_color, size=12)
    return fig


def preview_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rounded numeric preview for display."""
    prev = df.head(25).copy()
    for c in prev.columns:
        if prev[c].dtype == float or prev[c].dtype == "float64":
            prev[c] = prev[c].map(lambda x: round(float(x), 3) if pd.notna(x) else x)
    return prev


def render_results_panel(df: pd.DataFrame, *, compact_toolbar: bool) -> None:
    night = st.session_state.get("night_mode", False)
    colorblind = bool(st.session_state.get("colorblind_charts", True))
    rid = st.session_state.get("last_run_id", "")
    hit = st.session_state.get("last_cache_hit", False)
    ch = st.session_state.get("last_hash", "")

    df = results_df_clean_columns(df)

    if "overlay_results_charts" not in st.session_state:
        st.session_state.overlay_results_charts = False

    if compact_toolbar:
        st.title("Results")
    else:
        st.markdown("### Latest results")

    tb1, tb2, tb3 = st.columns([1, 1, 2])
    with tb1:
        if compact_toolbar and st.button("Edit configuration", type="primary", use_container_width=True):
            st.session_state.config_collapsed = False
            st.rerun()
    with tb2:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name=f"{rid or 'simulation'}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        f"Run **{rid}** · {'served from cache' if hit else 'newly computed'} · config `…{ch[-8:]}`"
    )

    ch_names = channel_names_from_results_df(df)
    schema_ok = cached_dataframe_schema_ok(df)
    if not schema_ok and ch_names:
        if hit:
            st.info(
                "These results were loaded from **disk cache** in an older CSV shape (no per-channel "
                "**`*_revenue`** columns). Click **Run simulation** once to rebuild, or use "
                "**Clear simulation cache** in the sidebar."
            )
        else:
            st.warning(
                "This run’s table is missing per-channel **`*_revenue`** columns next to "
                "**`*_impressions`** (unexpected for a new run). **Stop and restart** the Streamlit "
                "server so it picks up the latest `scripts.main` code, then run again."
            )
        scope_options = ["All channels (totals)"]
        st.session_state.results_chart_scope = scope_options[0]
    else:
        scope_options = ["All channels (totals)"] + ch_names

    if "results_chart_scope" not in st.session_state:
        st.session_state.results_chart_scope = scope_options[0]
    if st.session_state.results_chart_scope not in scope_options:
        st.session_state.results_chart_scope = scope_options[0]

    with st.expander("Configuration (YAML snapshot)", expanded=False):
        st.caption("Last merged settings (same structure as Advanced YAML).")
        st.code(
            yaml_dump(st.session_state.get("sim_config") or {}),
            language="yaml",
        )

    tab_chart, tab_data = st.tabs(["Chart view", "Data preview"])
    with tab_chart:
        csel1, csel2 = st.columns([1, 1])
        with csel1:
            st.selectbox(
                "Series scope",
                options=scope_options,
                key="results_chart_scope",
                help="Totals across all channels, or one channel’s revenue, spend, and impressions.",
            )
        with csel2:
            st.checkbox(
                "Overlay series (min–max normalized on one chart)",
                key="overlay_results_charts",
            )

        scope = str(st.session_state.results_chart_scope)
        ch_view: Optional[str] = None if scope == scope_options[0] else scope
        overlay = bool(st.session_state.overlay_results_charts)
        st.plotly_chart(
            make_charts(df, channel=ch_view, overlay=overlay, night=night, colorblind=colorblind),
            use_container_width=True,
        )

    with tab_data:
        st.caption("First 25 rows · values rounded for readability.")
        st.dataframe(
            preview_table(df),
            use_container_width=True,
            hide_index=True,
            height=320,
        )
