"""Results panel: Plotly charts, preview table, YAML snapshot."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.cache import cached_dataframe_schema_ok
from app.ui_chart_constants import ACCENT2, BLUE, CHART_PAL_CVD, ORANGE
from app.ui_yaml_io import yaml_dump
from scripts.spend_simulation.pairwise_summary import build_pairwise_summary


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


def _corr_cell_color(rho: float) -> str:
    if rho < 0:
        return "#e74c3c"
    if rho >= 0.8:
        return "#27ae60"
    if rho >= 0.5:
        return "#3498db"
    if rho >= 0.2:
        return "#e67e22"
    return "#95a5a6"


def _make_correlation_heatmap(
    corr: np.ndarray,
    names: List[str],
    *,
    night: bool = False,
) -> go.Figure:
    c = len(names)
    text = [[f"{corr[i, j]:.2f}" for j in range(c)] for i in range(c)]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.tolist(),
            x=names,
            y=names,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
            colorscale=[
                [0.0, "#e74c3c"],
                [0.25, "#e67e22"],
                [0.5, "#f1c40f"],
                [0.75, "#3498db"],
                [1.0, "#27ae60"],
            ],
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(title="rho"),
        )
    )
    paper = "rgba(15,23,42,0.3)" if night else "rgba(255,255,255,0)"
    plot_bg = "rgba(30,41,59,0.5)" if night else "rgba(248,250,252,0.9)"
    title_color = "#e2e8f0" if night else "#0f172a"
    fig.update_layout(
        title=dict(text="Static Correlation Matrix", font=dict(color=title_color, size=14)),
        height=380,
        margin=dict(l=80, r=40, t=50, b=40),
        paper_bgcolor=paper,
        plot_bgcolor=plot_bg,
        xaxis=dict(color=title_color, side="bottom"),
        yaxis=dict(color=title_color, autorange="reversed"),
    )
    return fig


def _make_rolling_correlation_chart(
    rolling_corr: np.ndarray,
    names: List[str],
    pair: List[str],
    window: int,
    *,
    night: bool = False,
) -> go.Figure:
    name_to_idx = {n: i for i, n in enumerate(names)}
    i, j = name_to_idx[pair[0]], name_to_idx[pair[1]]
    rho_series = rolling_corr[:, i, j]
    weeks = [f"W{t + window}" for t in range(len(rho_series))]

    paper = "rgba(15,23,42,0.3)" if night else "rgba(255,255,255,0)"
    plot_bg = "rgba(30,41,59,0.5)" if night else "rgba(248,250,252,0.9)"
    title_color = "#e2e8f0" if night else "#0f172a"
    grid = "rgba(148,163,184,0.2)" if night else "rgba(15,23,42,0.08)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=rho_series,
            mode="lines",
            fill="tozeroy",
            line=dict(color=BLUE, width=2),
            fillcolor="rgba(29,99,237,0.15)",
            name=f"{pair[0]} / {pair[1]}",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Rolling Correlation: {pair[0]} / {pair[1]}",
            font=dict(color=title_color, size=14),
        ),
        height=320,
        margin=dict(l=48, r=24, t=50, b=48),
        paper_bgcolor=paper,
        plot_bgcolor=plot_bg,
        xaxis=dict(gridcolor=grid, color=title_color, title="Week"),
        yaxis=dict(gridcolor=grid, color=title_color, title="Pearson rho", range=[-1, 1]),
        showlegend=False,
    )
    return fig


def _render_correlation_panel(corr_results: Dict[str, Any]) -> None:
    night = st.session_state.get("night_mode", False)
    names = corr_results["channel_names"]
    static_corr = corr_results["static_corr"]
    rolling_corr = corr_results["rolling_corr"]
    window = corr_results["window"]
    pairwise = corr_results["pairwise_summary"]

    st.markdown("---")
    st.markdown("### Channel Spend Correlation Analysis")
    st.caption(
        "Pearson **ρ** on weekly **spend** in the output table. **Heatmap:** all pairs. **Summary:** "
        "every unordered pair; parentheses show copula target ρ only when that pair is under `correlations` in config; "
        "trailing label = drift."
    )
    if not pairwise:
        st.caption("Need at least two channels for pairwise summaries and rolling charts.")

    m1, m2, m3 = st.columns(3)
    avg_rho = float(np.mean([v for v in corr_results["avg_abs_corr"].values()]))
    with m1:
        st.metric(
            "Avg pairwise |rho|",
            f"{avg_rho:.2f}",
            help="Mean |ρ| over all unordered channel pairs in the static spend matrix.",
        )
    with m2:
        st.metric(
            "Most correlated channel",
            corr_results["most_correlated_channel"],
            help="Channel with the largest mean |ρ| to every other channel.",
        )
    with m3:
        st.metric(
            "Rolling window",
            f"{window} wks",
            help="Window length in weeks for rolling ρ and for the drift comparison.",
        )

    col_heat, col_summary = st.columns([1, 1])
    with col_heat:
        st.caption("**Static matrix:** full-run Pearson ρ of weekly spend. Symmetric; diagonal 1.")
        st.plotly_chart(_make_correlation_heatmap(static_corr, names, night=night), width="stretch")
    with col_summary:
        st.markdown("**Pairwise Summary + Drift**")
        for p in pairwise:
            pair_label = f"{p['pair'][0]} / {p['pair'][1]}"
            rho_val = p["observed_rho"]
            rho_tgt = p.get("configured_rho")
            drift_label = p["drift_label"]
            color = _corr_cell_color(rho_val)
            drift_color = "#27ae60" if drift_label == "stable" else ("#e74c3c" if drift_label.startswith("-") else "#e67e22")
            tgt_muted = "#94a3b8" if night else "#64748b"
            tgt_html = (
                f"<span style='color:{tgt_muted};font-size:0.8rem'>(target ρ {float(rho_tgt):.2f})</span>"
                if rho_tgt is not None
                else f"<span style='color:{tgt_muted};font-size:0.8rem'>(no copula target)</span>"
            )
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;flex-wrap:wrap'>"
                f"<span style='min-width:140px;font-weight:500'>{pair_label}</span>"
                f"<span style='background:{color};color:white;padding:2px 10px;border-radius:4px;font-size:0.9rem;font-weight:600'>{rho_val:.2f}</span>"
                f"{tgt_html}"
                f"<span style='color:{drift_color};font-size:0.85rem;font-weight:500'>{drift_label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "**Drift:** mean rolling ρ over the last five windows minus the first five. "
            "Label **stable** when the absolute change stays under 0.05."
        )

    if rolling_corr is not None and rolling_corr.shape[0] > 0 and len(pairwise) > 0:
        raw_pairs = [list(p["pair"]) for p in pairwise]
        raw_labels = [f"{p[0]} / {p[1]}" for p in raw_pairs]
        all_pairs: List[List[str]] = []
        pair_labels: List[str] = []
        seen_unordered: set[tuple[str, ...]] = set()
        for lbl, pr in zip(raw_labels, raw_pairs):
            key = tuple(sorted(pr))
            if key in seen_unordered:
                continue
            seen_unordered.add(key)
            pair_labels.append(lbl)
            all_pairs.append(pr)
        if not pair_labels:
            pair_labels = raw_labels
            all_pairs = raw_pairs
        if "corr_pair_select" not in st.session_state:
            st.session_state.corr_pair_select = pair_labels[0]
        st.caption("**Rolling line:** Pearson ρ inside each sliding spend window along the timeline.")
        selected_label = st.selectbox(
            "Channel pair (rolling chart)",
            options=pair_labels,
            key="corr_pair_select",
            help="Selects the pair for the plot. Each point is ρ for one window; labels mark the window end week.",
        )
        idx = pair_labels.index(selected_label)
        selected_pair = all_pairs[idx]
        st.plotly_chart(
            _make_rolling_correlation_chart(rolling_corr, names, selected_pair, window, night=night),
            width="stretch",
        )

    with st.expander("Per-channel avg absolute correlation (multicollinearity risk)", expanded=False):
        st.caption(
            "Per channel: mean |ρ| to all **other** channels. Higher bars flag stronger joint movement with the rest of the mix."
        )
        for name, val in corr_results["avg_abs_corr"].items():
            bar_pct = min(val * 100, 100)
            color = _corr_cell_color(val)
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem'>"
                f"<span style='min-width:120px'>{name}</span>"
                f"<div style='flex:1;background:#e0e0e0;border-radius:4px;height:16px'>"
                f"<div style='width:{bar_pct}%;background:{color};height:100%;border-radius:4px'></div>"
                f"</div>"
                f"<span style='font-weight:600;min-width:45px'>{val:.3f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _build_corr_results_from_cached_df(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Fallback for cache hits: derive correlation results from spend columns + sim_config."""
    cfg = st.session_state.get("sim_config") or {}
    corr_cfg = cfg.get("correlations") or []

    channels = cfg.get("channel_list") or []
    channel_names: List[str] = []
    for item in channels:
        ch = item.get("channel") if isinstance(item, dict) else item
        if isinstance(ch, dict) and ch.get("channel_name"):
            channel_names.append(str(ch["channel_name"]))
    if not channel_names:
        return None

    spend_cols = [f"{name}_spend" for name in channel_names]
    if not all(c in df.columns for c in spend_cols):
        return None

    spend = df[spend_cols].to_numpy(dtype=float)
    t, c = spend.shape
    static_corr = np.corrcoef(spend.T)

    # Match scripts.spend_simulation.correlation_analysis.analyze_spend_correlations (default window=12).
    roll_window = 12
    effective_window = min(roll_window, t)
    rolling_corr = np.empty((0, c, c))
    drift = np.zeros((c, c))
    if effective_window < t:
        rolling_corr = np.array(
            [np.corrcoef(spend[i : i + effective_window].T) for i in range(t - effective_window)]
        )
        edge = min(5, rolling_corr.shape[0])
        drift = rolling_corr[-edge:].mean(axis=0) - rolling_corr[:edge].mean(axis=0)

    avg_abs_corr: Dict[str, float] = {}
    for i, name in enumerate(channel_names):
        off_diag = [abs(static_corr[i, j]) for j in range(c) if j != i]
        avg_abs_corr[name] = float(np.mean(off_diag)) if off_diag else 0.0
    avg_abs_corr = dict(sorted(avg_abs_corr.items(), key=lambda kv: kv[1], reverse=True))
    most_corr = max(avg_abs_corr, key=avg_abs_corr.get) if avg_abs_corr else ""

    pairwise_summary = build_pairwise_summary(channel_names, static_corr, drift, list(corr_cfg))

    return {
        "channel_names": channel_names,
        "static_corr": static_corr,
        "rolling_corr": rolling_corr,
        "drift": drift,
        "avg_abs_corr": avg_abs_corr,
        "most_correlated_channel": most_corr,
        "pairwise_summary": pairwise_summary,
        "window": effective_window,
    }


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
        if compact_toolbar and st.button("Edit configuration", type="primary", width="stretch"):
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
            width="stretch",
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
        # #region agent log
        try:
            from app.debug_ndlog import agent_dbg

            sc = st.session_state.get("sim_config") or {}
            agent_dbg(
                "H5",
                "ui_results.render_results_panel",
                "yaml_snapshot_expander",
                {
                    "n_correlations": len(sc.get("correlations") or []),
                    "correlations": sc.get("correlations"),
                },
            )
        except Exception:
            pass
        # #endregion
        st.code(
            yaml_dump(st.session_state.get("sim_config") or {}),
            language="yaml",
        )

    # Always prefer recomputing from the saved CSV + current sim_config so pairwise lists stay
    # complete after code updates and never stay stale vs. the heatmap (session can hold old
    # last_corr_results from a previous app version or partial dict).
    rebuilt = _build_corr_results_from_cached_df(df)
    if rebuilt is not None:
        st.session_state["last_corr_results"] = rebuilt
        corr_results = rebuilt
    else:
        corr_results = st.session_state.get("last_corr_results")

    tab_chart, tab_corr, tab_data = st.tabs(
        ["Chart view", "Correlation analysis", "Data preview"]
    )

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
            width="stretch",
        )

    with tab_corr:
        if corr_results is not None:
            _render_correlation_panel(corr_results)
        else:
            st.info(
                "Needs per-channel **`*_spend`** columns and channel names in the saved configuration to build this tab."
            )

    with tab_data:
        st.caption("First 25 rows · values rounded for readability.")
        st.dataframe(
            preview_table(df),
            width="stretch",
            hide_index=True,
            height=320,
        )
