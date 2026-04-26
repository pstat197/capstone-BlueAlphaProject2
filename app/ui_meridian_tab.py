"""Streamlit: Bayesian MMM (Google Meridian) tab — run on the latest synthetic simulator output."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "output"
_ROI_FOREST_PNG = _OUT_DIR / "mmm_roi_forest_plot.png"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.meridian_mmm import (  # noqa: E402
    MeridianRunConfig,
    arviz_posterior_table,
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
    plot_mmm_roi_forest,
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

# (chains, n_adapt, n_burnin, n_keep, n_prior draws)
MCMC_PRESETS: Dict[str, Tuple[int, int, int, int, int]] = {
    "Fast (smaller, quicker)": (2, 500, 200, 200, 200),
    "Balanced": (4, 1000, 500, 500, 500),
    "Slower (more reliable)": (4, 2000, 500, 1000, 1000),
}


def _render_meridian_outputs(mmm: object, summ: Dict[str, Any], viz: Dict[str, Any]) -> None:
    """Plot Altair figures and tables from `meridian_visualizations` result."""
    st.subheader("Model fit & diagnostics")
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
        "**R̂ (R-hat):** MCMC mixing only. **R² / MAPE / wMAPE:** how close **posterior-mean** predictions are to **realized** revenue in sample—low R² is common on short, noisy, or misspecified runs."
    )
    fit_raw = viz.get("fit_metrics")
    fit_df = format_predictive_metrics_df(fit_raw) if fit_raw is not None else None
    if fit_df is not None and hasattr(fit_df, "empty") and not fit_df.empty:
        st.markdown("**In-sample predictive accuracy (Meridian table)**")
        st.dataframe(fit_df, use_container_width=True, hide_index=True)
    elif viz.get("fit_metrics_error"):
        st.caption("Fit metrics: " + str(viz["fit_metrics_error"]))

    st.subheader("Recovered media ROI (posterior)")
    st.caption("Per-**channel** incremental ROI from the **fitted** model, with 90% credible intervals (Channel names match your synthetic data).")
    if viz.get("media_roi_bar_chart") is not None:
        st.altair_chart(viz["media_roi_bar_chart"], use_container_width=True)
    elif viz.get("media_roi_error"):
        st.caption("Media ROI chart: " + str(viz["media_roi_error"]))

    st.caption(
        "**Forest plot** — same posterior ROI as above, as a horizontal dot-and-whisker "
        "(median, 90% CI, optional synthetic **true ROI** from your YAML). **R̂** for `roi_m` when available."
    )
    try:
        tr = true_roi_by_channel_map(st.session_state.get("sim_config"))
        rows = meridian_posterior_roi_forest_rows(mmm, true_map=tr)
        rh = roi_m_rhat_by_media_channel(mmm)
        fig_for = plot_mmm_roi_forest(
            rows,
            save_path=_ROI_FOREST_PNG,
            dpi=150,
            rhat_by_channel=rh,
        )
        st.pyplot(fig_for, use_container_width=True)
        plt.close(fig_for)
        st.caption(f"Saved: `{_ROI_FOREST_PNG}` (150 dpi).")
    except Exception as e:
        st.caption("ROI forest plot: " + str(e))

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

    st.subheader("Budget optimization (fixed total ≈ historical)")
    st.caption(
        "Under Meridian’s default **fixed-budget** scenario, spend is reallocated to maximize **incremental "
        "revenue (outcome)** in the model window, subject to channel bounds. **Weekly $** = window total / "
        "number of weeks in your loaded CSV. Pies: share of spend; **grouped bars**: current vs optimized weekly."
    )
    if viz.get("optimization_error"):
        st.warning("Budget optimization: " + str(viz["optimization_error"]))
    else:
        df0 = st.session_state.get("last_df")
        n_weeks = max(1, int(len(df0))) if df0 is not None else 1
        rec_opt: Optional[pd.DataFrame] = None
        if viz.get("spend_recommendation_df") is not None:
            rec0 = pd.DataFrame(viz["spend_recommendation_df"])
            if not rec0.empty and "spend_baseline" in rec0.columns and "channel" in rec0.columns:
                rec_opt = rec0.copy()
                rec_opt["current_weekly_spend"] = rec_opt["spend_baseline"] / n_weeks
                rec_opt["optimized_weekly_spend"] = rec_opt["spend_optimized"] / n_weeks
                rec_opt["change_pct"] = (
                    rec_opt["optimized_weekly_spend"] / (rec_opt["current_weekly_spend"] + 1e-9) - 1.0
                ) * 100.0
                total_b = float(rec_opt["spend_baseline"].sum())
                st.markdown(
                    f"**Window total spend (all channels, model view):** ${total_b:,.0f}  ·  **Weeks in data:** {n_weeks}  "
                    f"·  **Implied average weekly budget:** ${total_b / n_weeks:,.0f}"
                )
                show_cols = [
                    c
                    for c in [
                        "channel",
                        "spend_baseline",
                        "spend_optimized",
                        "current_weekly_spend",
                        "optimized_weekly_spend",
                        "change_pct",
                        "delta",
                        "delta_pct",
                    ]
                    if c in rec_opt.columns
                ]
                st.dataframe(
                    rec_opt[show_cols] if show_cols else rec_opt,
                    use_container_width=True,
                    hide_index=True,
                )
        opt = viz.get("optimization")
        if opt is not None:
            try:
                oa = opt.optimized_data.attrs
                na = opt.nonoptimized_data.attrs
                m1, m2, m3, m4 = st.columns(4)
                if "total_incremental_outcome" in na and "total_incremental_outcome" in oa:
                    t0 = float(na["total_incremental_outcome"])
                    t1 = float(oa["total_incremental_outcome"])
                    m1.metric("Non-optimized incremental (model)", f"{t0:,.0f}")
                    m2.metric("Optimized incremental (model)", f"{t1:,.0f}")
                    m3.metric("Lift (absolute)", f"{t1 - t0:,.0f}")
                    m4.metric(
                        "Lift (%)",
                        f"{(t1 / t0 - 1.0) * 100.0:+.1f}%" if t0 and t0 > 0 else "—",
                    )
            except (KeyError, TypeError, ValueError):
                pass
        if rec_opt is not None:
            figb = go.Figure(
                data=[
                    go.Bar(
                        name="Current (weekly $)",
                        x=rec_opt["channel"],
                        y=rec_opt["current_weekly_spend"],
                    ),
                    go.Bar(
                        name="Optimized (weekly $)",
                        x=rec_opt["channel"],
                        y=rec_opt["optimized_weekly_spend"],
                    ),
                ]
            )
            figb.update_layout(
                barmode="group",
                title="Current vs optimized — weekly spend allocation",
                yaxis_title="Weekly spend ($)",
                xaxis_title="Channel",
                showlegend=True,
            )
            st.plotly_chart(figb, use_container_width=True)

    row1, row2 = st.columns(2)
    with row1:
        if viz.get("opt_outcome_delta_chart") is not None:
            st.caption("Incremental outcome — reallocation effect (waterfall)")
            st.altair_chart(viz["opt_outcome_delta_chart"], use_container_width=True)
    with row2:
        if viz.get("opt_spend_delta_chart") is not None:
            st.caption("Change in spend by channel (vs baseline allocation)")
            st.altair_chart(viz["opt_spend_delta_chart"], use_container_width=True)
        elif viz.get("opt_spend_delta_error"):
            st.caption("Spend delta chart: " + str(viz["opt_spend_delta_error"]))

    p1, p2 = st.columns(2)
    with p1:
        if viz.get("opt_spend_pie_nonopt") is not None:
            st.caption("Share of spend — baseline")
            st.altair_chart(viz["opt_spend_pie_nonopt"], use_container_width=True)
    with p2:
        if viz.get("opt_spend_pie_opt") is not None:
            st.caption("Share of spend — optimized")
            st.altair_chart(viz["opt_spend_pie_opt"], use_container_width=True)

    t = arviz_posterior_table(mmm)
    if t is not None:
        st.markdown("##### Posterior (ArviZ summary)")
        st.dataframe(t, use_container_width=True)


def render_meridian_tab(schema: Dict[str, Any]) -> None:
    st.header("Bayesian MMM")
    st.caption("Uses synthetic data from the Simulator tab (in memory). Configure the run, then run the model.")

    mer_ok, _ = meridian_import_status()
    if not mer_ok:
        st.caption("Meridian not loaded — after `pip install google-meridian`, restart Streamlit with that same Python.")

    with st.expander("Optional: background on Bayesian MMM", expanded=False):
        st.markdown(MERIDIAN_HELP)

    if st.button("Load synthetic data from current config", use_container_width=True, key="m_btn_load_data"):
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

    chans = channel_names_from_simulator_df(df)
    n_weeks = len(df)
    st.caption(
        f"**{n_weeks}** weeks · **{len(chans)}** channels: {', '.join(chans)}. "
        "Priors apply to model **roi_m** (media treatment ROI; not raw simulator labels beyond these names)."
    )
    if n_weeks < 20:
        st.caption("Short series: expect weak in-sample R² and wide CIs even if R̂ looks fine.")

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
    )
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Shared: lognormal μ", value=0.2, step=0.1, key="m_roi_mu")
    with c2:
        st.number_input("Shared: lognormal σ", value=0.9, min_value=0.01, step=0.1, key="m_roi_sig")
    indep = st.session_state.get("m_roi_mode", "").startswith("Independent")
    if indep:
        st.caption("Per-channel names match your synthetic data.")
        for i, ch in enumerate(chans):
            a, b = st.columns(2)
            a.number_input(f"{ch} — μ", value=0.2, step=0.1, key=f"m_pr_mu_{i}")
            b.number_input(f"{ch} — σ", value=0.9, min_value=0.01, step=0.1, key=f"m_pr_sig_{i}")

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
        help="Chains: parallel NUTS. n_adapt: adapt step size. n_burnin: discarded warm-up. n_keep: kept draws per chain. Prior draws: sample_prior size.",
    )
    if prof in MCMC_PRESETS:
        pr = MCMC_PRESETS[prof]
        st.caption(
            f"Using preset: **{prof}** → chains={pr[0]}, n_adapt={pr[1]}, n_burnin={pr[2]}, n_keep={pr[3]}, prior draws={pr[4]}"
        )
    a, b, c, d = st.columns(4)
    with a:
        st.number_input("Chains", min_value=1, max_value=32, value=2, step=1, key="m_n_chains", disabled=prof in MCMC_PRESETS)
    with b:
        st.number_input("n_adapt", min_value=100, max_value=20_000, value=500, step=100, key="m_n_adapt", disabled=prof in MCMC_PRESETS)
    with c:
        st.number_input("n_burnin", min_value=0, max_value=20_000, value=200, step=50, key="m_n_burnin", disabled=prof in MCMC_PRESETS)
    with d:
        st.number_input("n_keep", min_value=50, max_value=10_000, value=200, step=50, key="m_n_keep", disabled=prof in MCMC_PRESETS)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.number_input("Prior draws (sample_prior)", min_value=50, max_value=5_000, value=200, step=50, key="m_n_prior", disabled=prof in MCMC_PRESETS)
    with p2:
        st.number_input("MCMC seed", min_value=0, max_value=2_147_483_647, value=0, step=1, key="m_seed")
    with p3:
        st.checkbox("enable_aks (adaptive knots; often slower / GPU)", value=False, key="m_enable_aks")
    st.number_input(
        "Meridian analysis batch_size (lower if out-of-memory on CPU)",
        min_value=20,
        max_value=1000,
        value=80,
        step=20,
        key="m_batch_size",
    )

    go = st.button("Run model", type="primary", use_container_width=True)

    if go:
        if not mer_ok:
            st.error("Install `google-meridian` in this environment, then restart Streamlit.")
            return
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
            except Exception as e:
                st.error(f"Meridian failed: {e}")
                st.session_state["meridian_mmm"] = None
                st.session_state["meridian_summary"] = None
                st.session_state["meridian_viz"] = None
                return

    mmm: Optional[object] = st.session_state.get("meridian_mmm")
    summ: Optional[Dict[str, Any]] = st.session_state.get("meridian_summary")
    viz: Optional[Dict[str, Any]] = st.session_state.get("meridian_viz")
    if mmm is not None and summ and viz is not None:
        note = summ.get("note") or ""
        if note:
            st.warning(note)
        _render_meridian_outputs(mmm, summ, viz)
