"""Streamlit: Bayesian MMM (Google Meridian) tab — run on the latest synthetic simulator output."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.meridian_mmm import (  # noqa: E402
    MeridianRunConfig,
    channel_names_from_simulator_df,
    fit_meridian,
    format_predictive_metrics_df,
    meridian_import_status,
    meridian_visualizations,
)
from app.pipeline_runner import run_pipeline  # noqa: E402
from app.cache import run_with_cache  # noqa: E402
from app.ui_config_merge import merge_ui_into_config  # noqa: E402
from app.ui_yaml_io import yaml_dump  # noqa: E402
from app.mmm_roi_forest import (  # noqa: E402
    meridian_posterior_roi_forest_rows,
    plotly_mmm_roi_forest_figure,
    roi_m_rhat_by_media_channel,
    true_roi_by_channel_map,
)

MERIDIAN_HELP = """
**Bayesian MMM (what changes vs a frequentist fit)**  
Classical MMM often picks a single “best” set of adstock/saturation/scale parameters by
optimizing a loss (e.g. maximizing R²) and then fits a regression. That yields **point
estimates** and can understate uncertainty, especially with correlated channels and limited time
points. A **Bayesian** MMM (Meridian) places **priors** on those parameters (e.g. ROI- or
coefficient-level priors), then uses **MCMC (NUTS in Meridian)** to draw many samples from the
**posterior**: a distribution of plausible values, not one winner.

**What Google Meridian is**  
[Meridian](https://github.com/google/meridian) is Google’s open-source Bayesian MMM library. It
models media (and optional controls) with hierarchical structure, adstock, saturation, and
treatment priors; inference is with TensorFlow Probability. Full fits are **compute-heavy**; Google
recommends **Python 3.11–3.13** and a **GPU** for large posteriors. This tab uses **revenue
KPI** and your weekly **impressions + spend** columns from the synthetic simulator.

**Core inputs (for this app)**  
- **KPI:** `revenue` (total weekly).  
- **Time:** `week` in the CSV is turned into a weekly `time` column.  
- **Population:** a constant `population` (national model).  
- **Media:** for each channel, `{name}_impressions` and `{name}_spend`.

**“Training” (posterior sampling)**  
Meridian is not trained by gradient descent on one loss. It (1) builds the model, (2) draws prior
samples, (3) runs **NUTS** chains: adaptation, burn-in, then kept samples. Chains are summarized
(e.g. **R̂** should be near 1.0 for good mixing). Tuning **chains / adapt / burn-in / keep** trades
**accuracy vs wall time**.

**Official walkthroughs**  
- [Install Meridian](https://developers.google.com/meridian/docs/user-guide/installing)  
- [Getting started (Colab)](https://developers.google.com/meridian/notebook/meridian-getting-started)  
- [Configure the model / priors](https://developers.google.com/meridian/docs/user-guide/configure-model)
"""

# Session: user clicked Run model; next rerun performs fit (shows busy button, blocks re-click)
_K_MMM_FIT_PENDING = "m_meridian_fit_pending"

# (chains, n_adapt, n_burnin, n_keep, n_prior draws)
MCMC_PRESETS: Dict[str, Tuple[int, int, int, int, int]] = {
    "Fast (smaller, quicker)": (2, 500, 200, 200, 200),
    "Balanced": (4, 1000, 500, 500, 500),
    "Slower (more reliable)": (4, 2000, 500, 1000, 1000),
}


def _budget_recommendation_html(df_rec: pd.DataFrame) -> str:
    """Per-channel reallocation lines (HTML, green / red / gray) for the optimization tab."""
    parts: List[str] = [
        '<div style="font-size:1.2rem;line-height:1.55;font-weight:500;">',
    ]
    _g, _r, _n = "#15803d", "#b91c1c", "#64748b"
    for _, row in df_rec.iterrows():
        ch = str(row["channel"])
        cur = float(row["current_weekly_spend"])
        optv = float(row["optimized_weekly_spend"])
        d = optv - cur
        if "change_pct" in row and pd.notna(row.get("change_pct")):
            pct = float(row["change_pct"])
        else:
            pct = 100.0 * d / (cur + 1e-9)
        if abs(d) < 0.5:
            parts.append(
                f'<p style="margin:0.4em 0;color:{_n}"><b>{ch}</b>: keep weekly spend at about '
                f"<b>${optv:,.0f}</b> (within ~$1 of current).</p>"
            )
        elif d > 0:
            parts.append(
                f'<p style="margin:0.4em 0;color:{_g}"><b>{ch}</b>: increase weekly spend by '
                f"<b>${d:,.0f}</b> (<b>{pct:+.1f}%</b>) — target <b>${optv:,.0f}</b>/week vs "
                f"<b>${cur:,.0f}</b>/week now.</p>"
            )
        else:
            parts.append(
                f'<p style="margin:0.4em 0;color:{_r}"><b>{ch}</b>: decrease weekly spend by '
                f"<b>${-d:,.0f}</b> (<b>{pct:+.1f}%</b>) — target <b>${optv:,.0f}</b>/week vs "
                f"<b>${cur:,.0f}</b>/week now.</p>"
            )
    parts.append("</div>")
    return "".join(parts) if len(parts) > 1 else ""


def _meridian_pie_title(chart: Any, title: str) -> Any:
    """Override Altair pie title from Meridian."""
    if chart is None:
        return None
    return chart.properties(
        title=alt.TitleParams(text=title, anchor="start", fontSize=17, fontWeight="normal")
    )


def _render_fit_diagnostics_tab(mmm: object, summ: Dict[str, Any], viz: Dict[str, Any]) -> None:
    """Tab content: max R-hat, in-sample fit metrics, and advanced diagnostics."""
    mcols = st.columns(3)
    rh = summ.get("rhat_max")
    if rh is not None:
        try:
            v = float(rh)
            mcols[0].metric(
                "Max R̂ (posterior)",
                f"{v:.4f}",
                help="Largest Gelman–Rubin R-hat over parameters. Near 1 means chains mixed well; above ~1.1 is a red flag. This is not business 'accuracy' (see R²).",
            )
        except (TypeError, ValueError):
            pass
    mcols[1].empty()
    mcols[2].empty()  # layout spacer
    st.caption(
        "**R̂ (R-hat):** MCMC mixing only. **R², MAPE, wMAPE:** in-sample **goodness of fit** for "
        "posterior-mean **predicted** revenue vs **realized** revenue. Low R² is common on short or noisy series."
    )
    fit_raw = viz.get("fit_metrics")
    fit_df = format_predictive_metrics_df(fit_raw) if fit_raw is not None else None
    if fit_df is not None and hasattr(fit_df, "empty") and not fit_df.empty:
        st.markdown("**In-sample predictive accuracy (Meridian table)**")
        st.caption(
            "**MAPE** (Mean Absolute Percentage Error) is the average of "
            "|actual − predicted| / |actual| (by week), expressed as a percent. It treats big misses as a "
            "fraction of the actual level, so a few small weeks are not up-weighted like in plain MAE. "
            "**wMAPE** (weighted MAPE) is similar but weighted so larger-revenue weeks count more. "
            "Values are in-sample: they do **not** measure holdout or business forecast accuracy on their own."
        )
        st.dataframe(fit_df, use_container_width=True, hide_index=True)
    elif viz.get("fit_metrics_error"):
        st.caption("Fit metrics: " + str(viz["fit_metrics_error"]))

    with st.expander("Advanced analytics", expanded=False):
        st.caption("Optional diagnostics: in-sample **fit** vs data and **MCMC** convergence by parameter.")
        g1, g2 = st.columns(2)
        with g1:
            if viz.get("model_fit_chart") is not None:
                st.caption("Expected vs actual revenue (in-sample)")
                st.altair_chart(viz["model_fit_chart"], use_container_width=True)
            elif viz.get("model_fit_error"):
                st.caption("Model fit plot: " + str(viz["model_fit_error"]))
        with g2:
            if viz.get("rhat_chart") is not None:
                st.caption("R̂ by parameter (MCMC convergence)")
                st.altair_chart(viz["rhat_chart"], use_container_width=True)
            elif viz.get("rhat_error"):
                st.caption("R-hat plot: " + str(viz["rhat_error"]))


def _render_roi_tab(mmm: object) -> None:
    """Tab content: posterior ROI forest plot per channel."""
    st.caption(
        "Per-**channel** incremental ROI from the **fitted** model (90% credible interval). "
        "Optional **true** ROI (red dashed) comes from your YAML. Hover a point for details. **R̂** for `roi_m` in hover when available."
    )
    try:
        tr = true_roi_by_channel_map(st.session_state.get("sim_config"))
        rows = meridian_posterior_roi_forest_rows(mmm, true_map=tr)
        rh = roi_m_rhat_by_media_channel(mmm)
        fig_ply = plotly_mmm_roi_forest_figure(rows, rhat_by_channel=rh)
        st.plotly_chart(fig_ply, use_container_width=True)
    except Exception as e:
        st.caption("ROI forest plot: " + str(e))


def _render_optimization_tab(viz: Dict[str, Any]) -> None:
    """Tab content: budget optimization — current vs optimized spend share (pies + reallocation text)."""
    st.caption(
        "Under Meridian’s default **fixed-budget** scenario, spend is reallocated to maximize **incremental "
        "revenue (outcome)** in the model window, subject to channel bounds. The pies show each channel’s "
        "**share of spend** before vs after optimization."
    )
    if viz.get("optimization_error"):
        st.warning("Budget optimization: " + str(viz["optimization_error"]))
        return

    p1, p2 = st.columns(2)
    with p1:
        if viz.get("opt_spend_pie_nonopt") is not None:
            st.altair_chart(
                _meridian_pie_title(viz["opt_spend_pie_nonopt"], "Current budget allocation"),
                use_container_width=True,
            )
    with p2:
        if viz.get("opt_spend_pie_opt") is not None:
            st.altair_chart(
                _meridian_pie_title(viz["opt_spend_pie_opt"], "Optimized budget allocation"),
                use_container_width=True,
            )

    if viz.get("spend_recommendation_df") is not None:
        df0 = st.session_state.get("last_df")
        n_weeks = max(1, int(len(df0))) if df0 is not None else 1
        rec0 = pd.DataFrame(viz["spend_recommendation_df"])
        if not rec0.empty and "spend_baseline" in rec0.columns and "channel" in rec0.columns:
            rec_opt = rec0.copy()
            rec_opt["current_weekly_spend"] = rec_opt["spend_baseline"] / n_weeks
            rec_opt["optimized_weekly_spend"] = rec_opt["spend_optimized"] / n_weeks
            rec_opt["change_pct"] = (
                rec_opt["optimized_weekly_spend"] / (rec_opt["current_weekly_spend"] + 1e-9) - 1.0
            ) * 100.0
            st.markdown(
                "<p style='font-size:1.35rem;font-weight:600;margin:0.5rem 0 0.3rem 0;'>"
                "Recommended reallocation (weekly, model suggestion)</p>",
                unsafe_allow_html=True,
            )
            rec_html = _budget_recommendation_html(rec_opt)
            if rec_html:
                st.markdown(rec_html, unsafe_allow_html=True)
            else:
                st.caption("No per-channel text could be built from the optimization table.")


def _render_meridian_outputs(mmm: object, summ: Dict[str, Any], viz: Dict[str, Any]) -> None:
    """Render result sections in clickable tabs at the top instead of stacked sections."""
    st.markdown(
        """
        <style>
        /* Remove default card / band background on this tab (metrics + row layout). */
        div[data-testid="stMetric"] {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        div[data-testid="stMetric"] > div,
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricLabel"] {
            background-color: transparent !important;
        }
        section.main div[data-testid="stHorizontalBlock"] {
            background: transparent !important;
            border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab_fit, tab_roi, tab_opt = st.tabs(
        ["Model fit & diagnostics", "Recovered media ROI", "Budget optimization"]
    )
    with tab_fit:
        _render_fit_diagnostics_tab(mmm, summ, viz)
    with tab_roi:
        _render_roi_tab(mmm)
    with tab_opt:
        _render_optimization_tab(viz)


def render_meridian_tab(schema: Dict[str, Any]) -> None:
    st.header("Bayesian MMM")
    st.caption("Uses synthetic data from the Simulator tab (in memory). Configure the run, then run the model.")

    mer_ok, mer_err = meridian_import_status()
    if not mer_ok:
        st.error(
            "**Google Meridian is not installed** in the Python environment running Streamlit. "
            "The simulator tab still works; this tab needs the optional MMM stack (TensorFlow + Meridian)."
        )
        st.markdown(
            "From the **project root**, install the optional stack **into the same venv** Streamlit uses:\n\n"
            "```\npip install -r requirements.txt\n"
            "pip install -r requirements-meridian.txt\n```\n\n"
            "Or one line: `pip install -e \".[mmm]\"`\n\n"
            "**Restart Streamlit with that interpreter**, e.g. `./scripts/run_streamlit.sh` or `.venv/bin/streamlit run app/streamlit_app.py` "
            "(running plain `streamlit` can pick a different Python without Meridian)."
        )
        if mer_err:
            with st.expander("Import error details (for debugging)", expanded=False):
                st.code(mer_err or "", language="text")
        return

    with st.expander("Optional: background on Bayesian MMM", expanded=False):
        st.markdown(MERIDIAN_HELP)

    if st.button(
        "Load synthetic data from current config",
        use_container_width=True,
        key="m_btn_load_data",
        help=(
            "Re-run the Simulator pipeline using the current YAML / form configuration and load the "
            "resulting weekly DataFrame here so the MMM uses fresh synthetic data."
        ),
    ):
        regen = True
    else:
        regen = False

    if regen:
        try:
            merged, warns = merge_ui_into_config(schema)
            for w in warns:
                st.warning(w)
            if not (merged.get("channel_list") or []):
                raise ValueError("Add at least one channel in the Simulator tab (or use example config).")
            df_out, run_id, cache_hit, cfg_hash, corr_results = run_with_cache(merged, run_pipeline)
            st.session_state["last_df"] = df_out
            st.session_state["last_run_id"] = run_id
            st.session_state["last_cache_hit"] = cache_hit
            st.session_state["last_hash"] = cfg_hash
            st.session_state["last_corr_results"] = corr_results
            st.session_state["last_error"] = None
            st.session_state.sim_config = copy.deepcopy(merged)
            st.session_state["pending_yaml_dump"] = yaml_dump(merged)
            st.session_state.yaml_manual_edit = False
            st.success("Synthetic data refreshed.")
        except Exception as e:
            st.error(str(e))

    df = st.session_state.get("last_df")
    if df is None:
        st.info("No data yet. Run the simulator in the first tab, or use **Load synthetic data** above.")
        return

    st.session_state.setdefault(_K_MMM_FIT_PENDING, False)

    chans = channel_names_from_simulator_df(df)
    n_weeks = len(df)
    st.caption(
        f"**{n_weeks}** weeks · **{len(chans)}** channels: {', '.join(chans)}. "
        "Priors apply to model **roi_m** (media treatment ROI; not raw simulator labels beyond these names)."
    )
    if n_weeks < 20:
        st.caption("Short series: expect weak in-sample R² and wide CIs even if R̂ looks fine.")

    has_fit = (
        st.session_state.get("meridian_mmm") is not None
        and st.session_state.get("meridian_viz") is not None
    )
    m_pending = bool(st.session_state.get(_K_MMM_FIT_PENDING, False))
    if m_pending and not mer_ok:
        m_pending = False
        st.session_state[_K_MMM_FIT_PENDING] = False

    # Inputs visibility:
    # - During a pending run: always hidden (only the spinner shows).
    # - Before any fit: visible so the user can configure.
    # - Fit already exists: hidden until the user clicks "Edit settings & re-run".
    st.session_state.setdefault("m_show_inputs", False)
    if m_pending:
        show_inputs = False
    elif not has_fit:
        show_inputs = True
    else:
        show_inputs = bool(st.session_state.get("m_show_inputs", False))

    if has_fit and not m_pending and not show_inputs:
        if st.button(
            "⚙ Edit settings & re-run",
            key="m_edit_toggle",
            use_container_width=True,
            help="Bring back the ROI prior and MCMC settings so you can change them and re-run the model.",
        ):
            st.session_state["m_show_inputs"] = True
            st.rerun()

    # Wrap the input sections in an st.empty() slot so they can be cleared from the
    # browser the moment "Run model" is clicked, instead of remaining visible until the
    # next rerun renders past them.
    inputs_slot = st.empty()
    run_clicked = False
    if show_inputs:
        with inputs_slot.container():
            st.markdown("##### ROI prior (lognormal on roi_m)")
            st.caption(
                "Lognormal **μ, σ** parameterize the prior on each channel’s **media ROI** in Meridian. "
                "Use *Shared* for one prior for all channels, or *Independent* to type different μ, σ for each name below."
            )
            st.radio(
                "Prior per channel",
                [
                    "Shared (same lognormal for every channel)",
                    "Independent (lognormal μ and σ for each channel)",
                ],
                index=0,
                key="m_roi_mode",
                help=(
                    "**Shared**: one lognormal prior used for every channel's ROI — fewer assumptions, faster "
                    "and a sensible default. **Independent**: type a different μ and σ per channel when you have "
                    "channel-specific beliefs (e.g. expect Search ROI ≫ Display)."
                ),
            )
            indep = st.session_state.get("m_roi_mode", "").startswith("Independent")
            if not indep:
                c1, c2 = st.columns(2)
                with c1:
                    st.number_input(
                        "Shared: lognormal μ",
                        value=0.2,
                        step=0.1,
                        key="m_roi_mu",
                        help=(
                            "Lognormal location (log-space mean) for prior media ROI. The implied prior **median** "
                            "ROI is exp(μ); e.g. μ=0.2 → median ROI ≈ 1.22, μ=0 → median ROI = 1.0."
                        ),
                    )
                with c2:
                    st.number_input(
                        "Shared: lognormal σ",
                        value=0.9,
                        min_value=0.01,
                        step=0.1,
                        key="m_roi_sig",
                        help=(
                            "Lognormal scale (log-space std) — controls how wide the prior is. Larger σ allows "
                            "much higher / lower ROI a priori; σ≈0.5 is fairly tight, σ≈0.9 is broad."
                        ),
                    )
            else:
                st.caption("Per-channel lognormal **μ** and **σ** (names match your synthetic data).")
                for i, ch in enumerate(chans):
                    a, b = st.columns(2)
                    a.number_input(
                        f"{ch} — μ",
                        value=0.2,
                        step=0.1,
                        key=f"m_pr_mu_{i}",
                        help=f"Lognormal μ for **{ch}** only. Implied prior median ROI ≈ exp(μ).",
                    )
                    b.number_input(
                        f"{ch} — σ",
                        value=0.9,
                        min_value=0.01,
                        step=0.1,
                        key=f"m_pr_sig_{i}",
                        help=f"Lognormal σ for **{ch}** only. Larger = more uncertainty about that channel's ROI.",
                    )

            st.markdown("##### MCMC (NUTS) sampling")
            prof = st.selectbox(
                "Profile",
                [
                    "Custom (set every number below)",
                    "Fast (smaller, quicker)",
                    "Balanced",
                    "Slower (more reliable)",
                ],
                index=2,
                key="m_mcmc_profile",
                help=(
                    "Pick a preset to fill chains / n_adapt / n_burnin / n_keep / prior draws automatically. "
                    "**Fast** gets you a quick sanity check; **Balanced** is a sensible default; "
                    "**Slower** runs more samples for tighter credible intervals at the cost of wall time. "
                    "Choose **Custom** to set every number yourself."
                ),
            )
            if prof in MCMC_PRESETS:
                pr = MCMC_PRESETS[prof]
                st.caption(
                    f"Using preset: **{prof}** → chains={pr[0]}, n_adapt={pr[1]}, n_burnin={pr[2]}, n_keep={pr[3]}, prior draws={pr[4]}"
                )
            a, b, c, d = st.columns(4)
            with a:
                st.number_input(
                    "Chains",
                    min_value=1,
                    max_value=32,
                    value=2,
                    step=1,
                    key="m_n_chains",
                    disabled=prof in MCMC_PRESETS,
                    help=(
                        "Number of independent NUTS chains run in parallel. More chains make R̂ more reliable "
                        "and reveal multimodality, but linearly increase compute. 2–4 is typical."
                    ),
                )
            with b:
                st.number_input(
                    "n_adapt",
                    min_value=100,
                    max_value=20_000,
                    value=500,
                    step=100,
                    key="m_n_adapt",
                    disabled=prof in MCMC_PRESETS,
                    help=(
                        "Adaptation iterations used by NUTS to tune the step size and mass matrix. Discarded "
                        "from inference. More adaptation generally helps tricky posteriors but slows the run."
                    ),
                )
            with c:
                st.number_input(
                    "n_burnin",
                    min_value=0,
                    max_value=20_000,
                    value=200,
                    step=50,
                    key="m_n_burnin",
                    disabled=prof in MCMC_PRESETS,
                    help=(
                        "Burn-in (warm-up) draws after adaptation that are discarded so the chain has time to "
                        "reach the typical set of the posterior before samples are kept."
                    ),
                )
            with d:
                st.number_input(
                    "n_keep",
                    min_value=50,
                    max_value=10_000,
                    value=200,
                    step=50,
                    key="m_n_keep",
                    disabled=prof in MCMC_PRESETS,
                    help=(
                        "Posterior draws **kept per chain** for inference. More draws → tighter credible "
                        "intervals and more stable summaries, but proportionally slower."
                    ),
                )
            p1, p2, p3 = st.columns(3)
            with p1:
                st.number_input(
                    "Prior draws (sample_prior)",
                    min_value=50,
                    max_value=5_000,
                    value=200,
                    step=50,
                    key="m_n_prior",
                    disabled=prof in MCMC_PRESETS,
                    help=(
                        "Samples drawn directly from the **prior** (no data) and used by Meridian for prior "
                        "predictive checks; usually small."
                    ),
                )
            with p2:
                st.number_input(
                    "MCMC seed",
                    min_value=0,
                    max_value=2_147_483_647,
                    value=0,
                    step=1,
                    key="m_seed",
                    help="Random seed used for sampling. Fix it (any positive integer) for reproducible runs.",
                )
            with p3:
                st.checkbox(
                    "enable_aks (adaptive knots; often slower / GPU)",
                    value=False,
                    key="m_enable_aks",
                    help=(
                        "Adaptive Knot Saturation: lets the saturation curve flex more across knots. Often "
                        "slower and is most practical with a GPU; leave off for quick CPU sanity checks."
                    ),
                )
            st.number_input(
                "Meridian analysis batch_size (lower if out-of-memory on CPU)",
                min_value=20,
                max_value=1000,
                value=80,
                step=20,
                key="m_batch_size",
                help=(
                    "Batch size used by Meridian's Analyzer when computing posterior summaries / optimization. "
                    "Lower this if you hit out-of-memory errors on CPU; higher is faster but uses more RAM."
                ),
            )

            run_clicked = st.button(
                "Run model",
                type="primary",
                use_container_width=True,
                key="m_run_go",
                help=(
                    "Fit Meridian on the loaded synthetic data using the priors and MCMC settings above. "
                    "Sampling + diagnostics + budget optimization can take many minutes on CPU."
                ),
            )

    # Handle the click OUTSIDE the with block so we can clear the slot. Triggering the
    # fit in the same script run (no st.rerun()) means the spinner below replaces the
    # inputs immediately on screen instead of after the long-running fit completes.
    if run_clicked:
        if not mer_ok:
            st.error("Install `google-meridian` in this environment, then restart Streamlit.")
        else:
            inputs_slot.empty()
            st.session_state["m_show_inputs"] = False
            st.session_state[_K_MMM_FIT_PENDING] = True
            m_pending = True

    if m_pending and mer_ok:
        psel = st.session_state.get("m_mcmc_profile", "Balanced")
        if psel in MCMC_PRESETS:
            nc, na, nb, nk, npr0 = MCMC_PRESETS[psel]
        else:
            nc = int(st.session_state.get("m_n_chains", 2))
            na = int(st.session_state.get("m_n_adapt", 500))
            nb = int(st.session_state.get("m_n_burnin", 200))
            nk = int(st.session_state.get("m_n_keep", 200))
            npr0 = int(st.session_state.get("m_n_prior", 200))
        per = bool(st.session_state.get("m_roi_mode", "").startswith("Independent"))
        mus: Optional[List[float]] = None
        sigs: Optional[List[float]] = None
        if per:
            mus = [float(st.session_state.get(f"m_pr_mu_{i}", 0.2)) for i in range(len(chans))]
            sigs = [float(st.session_state.get(f"m_pr_sig_{i}", 0.9)) for i in range(len(chans))]

        fit_succeeded = False
        with st.spinner("Sampling + diagnostics + budget optimization (can take many minutes on CPU)…"):
            try:
                cfg = MeridianRunConfig(
                    roi_log_mu=float(st.session_state.get("m_roi_mu", 0.2)),
                    roi_log_sigma=float(st.session_state.get("m_roi_sig", 0.9)),
                    per_channel_roi_priors=per,
                    channel_roi_mus=mus,
                    channel_roi_sigmas=sigs,
                    n_chains=nc,
                    n_adapt=na,
                    n_burnin=nb,
                    n_keep=nk,
                    n_prior=npr0,
                    seed=int(st.session_state.get("m_seed", 0)),
                    enable_aks=bool(st.session_state.get("m_enable_aks", False)),
                )
                mmm, summ = fit_meridian(df, cfg)
                bsz = int(st.session_state.get("m_batch_size", 80))
                viz = meridian_visualizations(mmm, analysis_batch_size=bsz)
                st.session_state["meridian_mmm"] = mmm
                st.session_state["meridian_summary"] = summ
                st.session_state["meridian_viz"] = viz
                fit_succeeded = True
            except Exception as e:
                st.error(f"Meridian failed: {e}")
                st.session_state["meridian_mmm"] = None
                st.session_state["meridian_summary"] = None
                st.session_state["meridian_viz"] = None
            finally:
                st.session_state[_K_MMM_FIT_PENDING] = False

        if fit_succeeded:
            # Hide the input sections automatically after a successful run; user can bring
            # them back via the "⚙ Edit settings & re-run" button at the top of the tab.
            st.session_state["m_show_inputs"] = False
            st.rerun()

    mmm: Optional[object] = st.session_state.get("meridian_mmm")
    summ: Optional[Dict[str, Any]] = st.session_state.get("meridian_summary")
    viz: Optional[Dict[str, Any]] = st.session_state.get("meridian_viz")
    if mmm is not None and summ and viz is not None:
        note = summ.get("note") or ""
        if note:
            st.warning(note)
        _render_meridian_outputs(mmm, summ, viz)
