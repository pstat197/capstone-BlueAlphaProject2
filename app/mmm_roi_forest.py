"""Horizontal forest (dot-and-whisker) plot for Bayesian MMM posterior ROI per channel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

# Styling
_CI_BLUE = "#378ADD"
_TRUE_RED = "#E24B4A"


def true_roi_by_channel_map(sim_config: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Map channel name (exact YAML `channel_name`) -> synthetic true ROI when present."""
    out: Dict[str, float] = {}
    if not sim_config:
        return out
    for item in sim_config.get("channel_list") or []:
        d = (item or {}).get("channel") or {}
        name = d.get("channel_name")
        if not name or "true_roi" not in d:
            continue
        try:
            out[str(name).strip()] = float(d["true_roi"])
        except (TypeError, ValueError):
            continue
    return out


def _merge_true_roi(
    channel: str, true_map: Mapping[str, float]
) -> Optional[float]:
    if channel in true_map:
        return float(true_map[channel])
    key = str(channel).strip()
    for k, v in true_map.items():
        if k.strip().lower() == key.lower():
            return float(v)
    return None


def meridian_posterior_roi_forest_rows(
    mmm: Any,
    *,
    true_map: Optional[Mapping[str, float]] = None,
) -> List[dict]:
    """Build forest-plot row dicts from Meridian MediaSummary (matches Altair ROI bar chart)."""
    from meridian import constants as c
    from meridian.analysis import visualizer

    true_map = true_map or {}
    ms = visualizer.MediaSummary(mmm, use_kpi=False)
    ds = ms.get_paid_summary_metrics(aggregate_times=True)
    roi_da = ds["roi"].sel(distribution=c.POSTERIOR)
    channels = [str(x) for x in roi_da["channel"].values]
    out: List[dict] = []
    for ch in channels:
        med = float(roi_da.sel(metric=c.MEDIAN, channel=ch).item())
        lo = float(roi_da.sel(metric=c.CI_LO, channel=ch).item())
        hi = float(roi_da.sel(metric=c.CI_HI, channel=ch).item())
        tr = _merge_true_roi(ch, true_map)
        out.append(
            {
                "channel": ch,
                "median": med,
                "ci_low": lo,
                "ci_high": hi,
                "true_roi": tr,
            }
        )
    return out


def roi_m_rhat_by_media_channel(mmm: Any) -> Optional[Dict[str, float]]:
    """Per-channel R̂ for MCMC parameter ``roi_m`` (if available)."""
    try:
        import arviz as az
    except Exception:
        return None
    idata = getattr(mmm, "inference_data", None)
    if idata is None or "roi_m" not in getattr(idata, "posterior", {}):
        return None
    try:
        rh = az.rhat(idata.posterior[["roi_m"]])
    except Exception:
        return None
    if "roi_m" not in rh:
        return None
    da = rh["roi_m"]
    if "media_channel" not in da.dims and "media_channel" not in da.coords:
        return None
    names = [str(x) for x in da["media_channel"].values]
    vals = np.asarray(da.values).ravel()
    if len(names) != len(vals):
        return None
    return {names[i]: float(vals[i]) for i in range(len(names)) if np.isfinite(vals[i])}


def _annotate_rhat(val: float) -> Tuple[str, str]:
    if val < 1.05:
        return f"R̂ {val:.3f}", "green"
    if val <= 1.1:
        return f"R̂ {val:.3f}", "#B8860B"  # darkgoldenrod / amber
    return f"R̂ {val:.3f} ⚠", "red"


def plot_mmm_roi_forest(
    rows: List[Mapping[str, Any]],
    *,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    rhat_by_channel: Optional[Mapping[str, float]] = None,
) -> "Any":
    """Matplotlib horizontal forest plot; saves PNG if ``save_path`` is set."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No channels to plot", ha="center", va="center")
        ax.set_axis_off()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        return fig

    n = len(rows)
    fig_h = max(3.0, n * 1.2 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor="white")
    ax.set_facecolor("white")
    from matplotlib.transforms import blended_transform_factory

    rhat_axes = rhat_by_channel and len(rhat_by_channel) > 0
    if rhat_axes:
        fig.subplots_adjust(right=0.78)
    trans_xaxes_ydata = blended_transform_factory(ax.transAxes, ax.transData)

    for i, r in enumerate(rows):
        ch = str(r["channel"])
        med = float(r["median"])
        lo = float(r["ci_low"])
        hi = float(r["ci_high"])
        w = hi - lo
        y = float(i)
        if w > 0:
            ax.barh(y, w, left=lo, height=0.25, color=_CI_BLUE, alpha=0.12, zorder=1)
        ax.hlines(y, lo, hi, color=_CI_BLUE, linewidth=2.5, zorder=2)
        cap = 0.12
        ax.vlines(
            [lo, hi], y - cap, y + cap, color=_CI_BLUE, linewidth=2, zorder=2
        )
        tr = r.get("true_roi")
        if tr is not None and np.isfinite(tr):
            ax.plot(
                [tr, tr],
                [y - 0.38, y + 0.38],
                linestyle="--",
                color=_TRUE_RED,
                linewidth=1.5,
                zorder=4,
            )
        ax.scatter(
            med,
            y,
            s=95,
            c=_CI_BLUE,
            zorder=6,
            edgecolors="white",
            linewidths=0.6,
        )

        rhat_map = rhat_by_channel or {}
        rh = rhat_map.get(ch)
        if rh is None and rhat_map:
            for k, v in rhat_map.items():
                if k.strip().lower() == ch.lower():
                    rh = v
                    break
        if rh is not None and np.isfinite(rh):
            txt, col = _annotate_rhat(float(rh))
            ax.text(
                1.01,
                y,
                txt,
                transform=trans_xaxes_ydata,
                ha="left",
                va="center",
                fontsize=9,
                color=col,
            )

    ax.set_yticks(np.arange(n, dtype=float))
    ax.set_yticklabels([str(r["channel"]) for r in rows], fontsize=12)
    ax.set_xlabel("ROI ($ returned per $1 spent)", fontsize=11)
    ax.set_title(
        "Recovered media ROI — posterior median with 90% credible interval",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    has_true = any(
        r.get("true_roi") is not None and np.isfinite(r["true_roi"]) for r in rows
    )
    med_line = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=_CI_BLUE,
        markeredgecolor="white",
        markersize=9,
        label="Posterior median",
    )
    ci_patch = Patch(
        facecolor=_CI_BLUE, alpha=0.35, edgecolor=_CI_BLUE, label="90% credible interval"
    )
    leg_el = [med_line, ci_patch]
    if has_true:
        leg_el.append(
            Line2D(
                [0],
                [0],
                color=_TRUE_RED,
                linestyle="--",
                linewidth=1.5,
                label="True synthetic ROI",
            )
        )
    ax.legend(handles=leg_el, loc="lower right", frameon=True)
    if not rhat_axes:
        fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def plotly_mmm_roi_forest_figure(
    rows: List[Mapping[str, Any]],
    *,
    rhat_by_channel: Optional[Mapping[str, float]] = None,
) -> "Any":
    """Interactive horizontal forest plot (Plotly) — same data as the matplotlib version."""
    import plotly.graph_objects as go  # type: ignore[import-not-found]

    if not rows:
        return go.Figure().update_layout(
            title="Recovered media ROI (posterior)",
            height=200,
        )

    rhat_map = rhat_by_channel or {}
    n = len(rows)
    fig = go.Figure()
    has_true = any(
        r.get("true_roi") is not None
        and np.isfinite(r["true_roi"])
        for r in rows
    )
    true_legend_placed = False
    for i, r in enumerate(rows):
        ch = str(r["channel"])
        med = float(r["median"])
        lo = float(r["ci_low"])
        hi = float(r["ci_high"])
        # Shaded 90% band
        fig.add_shape(
            type="rect",
            x0=lo,
            x1=hi,
            y0=i - 0.2,
            y1=i + 0.2,
            fillcolor="rgba(55, 138, 221, 0.12)",
            line_width=0,
            layer="below",
        )
        # CI line + caps
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[i, i],
                mode="lines",
                line=dict(color=_CI_BLUE, width=2.5),
                showlegend=(i == 0),
                name="90% credible interval",
                legendgroup="ci",
            )
        )
        for cx in (lo, hi):
            fig.add_trace(
                go.Scatter(
                    x=[cx, cx],
                    y=[i - 0.1, i + 0.1],
                    mode="lines",
                    line=dict(color=_CI_BLUE, width=2),
                    showlegend=False,
                    legendgroup="ci",
                )
            )
        tr = r.get("true_roi")
        if tr is not None and np.isfinite(float(tr)):
            tval = float(tr)
            sleg = has_true and (not true_legend_placed)
            if sleg:
                true_legend_placed = True
            fig.add_trace(
                go.Scatter(
                    x=[tval, tval],
                    y=[i - 0.38, i + 0.38],
                    mode="lines",
                    line=dict(color=_TRUE_RED, width=1.5, dash="dash"),
                    showlegend=sleg,
                    name="True synthetic ROI",
                    legendgroup="true",
                )
            )
        rh = rhat_map.get(ch)
        if rh is None and rhat_map:
            for k, v in rhat_map.items():
                if k.strip().lower() == ch.lower():
                    rh = v
                    break
        htext = f"{ch}<br>Median ROI: {med:.2f}<br>90% CI: [{lo:.2f}, {hi:.2f}]"
        if tr is not None and np.isfinite(float(tr)):
            htext += f"<br>True (YAML): {float(tr):.2f}"
        if rh is not None and np.isfinite(rh):
            htext += f"<br>R̂ (roi_m): {float(rh):.3f}"
        fig.add_trace(
            go.Scatter(
                x=[med],
                y=[i],
                mode="markers",
                name="Posterior median",
                showlegend=(i == 0),
                legendgroup="med",
                marker=dict(
                    size=11,
                    color=_CI_BLUE,
                    line=dict(color="white", width=1),
                ),
                hovertext=[htext],
                hoverinfo="text",
            )
        )

    xmax = max(float(r["ci_high"]) for r in rows)
    xmin = min(float(r["ci_low"]) for r in rows)
    if has_true:
        for r in rows:
            t = r.get("true_roi")
            if t is not None and np.isfinite(t):
                xmax = max(xmax, float(t))
                xmin = min(xmin, float(t))
    pad = (xmax - xmin) * 0.08 if xmax > xmin else 0.1 * (abs(xmax) + 0.1)
    x_pad_r = 1.2 if rhat_map else 1.0
    fig.update_xaxes(
        title_text="ROI ($ returned per $1 spent)",
        range=[xmin - pad, xmax + pad * x_pad_r],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)",
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=[str(r["channel"]) for r in rows],
        autorange="reversed",
        showgrid=False,
    )
    fig.update_layout(
        title="Recovered media ROI — posterior median with 90% credible interval",
        height=max(300, 80 * n + 120),
        template="plotly_white",
        margin=dict(l=24, r=24, t=64, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=bool(n),
    )
    return fig
