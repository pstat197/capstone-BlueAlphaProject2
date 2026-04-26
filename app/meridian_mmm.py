"""Build Google Meridian (Bayesian MMM) inputs from the simulator DataFrame and fit the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MeridianRunConfig:
    """User-tunable settings for a Meridian MCMC run (see Google Meridian docs)."""

    roi_log_mu: float = 0.2
    roi_log_sigma: float = 0.9
    # If True, use channel_roi_mus / channel_roi_sigmas (len == n media channels). Else repeat global pair.
    per_channel_roi_priors: bool = False
    channel_roi_mus: Optional[List[float]] = None
    channel_roi_sigmas: Optional[List[float]] = None
    n_chains: int = 4
    n_adapt: int = 1000
    n_burnin: int = 500
    n_keep: int = 500
    n_prior: int = 500
    seed: int = 0
    enable_aks: bool = False


def meridian_import_status() -> tuple[bool, Optional[str]]:
    """Return (ok, err_msg). err_msg is the import exception text when ok is False."""
    try:
        import meridian  # noqa: F401
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, None


def meridian_importable() -> bool:
    return meridian_import_status()[0]


def _channel_names_from_df(df: pd.DataFrame) -> List[str]:
    spend_cols = [c for c in df.columns if c.endswith("_spend") and c != "total_spend"]
    return [c.replace("_spend", "") for c in spend_cols]


def channel_names_from_simulator_df(df: pd.DataFrame) -> List[str]:
    """Channel names in simulator column order (for per-channel priors and labels)."""
    return _channel_names_from_df(df)


def format_predictive_metrics_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Drop redundant `geo_granularity` for single-geo (national) runs."""
    if df is None or not hasattr(df, "empty") or df.empty:
        return df
    out = df.copy()
    if "geo_granularity" in out.columns and out["geo_granularity"].nunique() <= 1:
        out = out.drop(columns=["geo_granularity"])
    return out


def simulation_df_to_meridian_format(
    df: pd.DataFrame,
    time_start: str = "2024-01-01",
) -> pd.DataFrame:
    """Map lab simulator output to columns Meridian's DataFrameInputDataBuilder expects.

    - Adds a ``time`` column (weekly datetimes) from ``week`` (1-based index).
    - Adds a constant ``population`` column (national MMM; Meridian default scale).
    """
    out = df.copy()
    if "week" not in out.columns:
        raise ValueError("Simulator DataFrame must include a 'week' column.")
    w = out["week"].astype(int)
    t0 = pd.Timestamp(time_start)
    out["time"] = t0 + pd.to_timedelta((w - 1) * 7, unit="D")
    if "revenue" not in out.columns:
        raise ValueError("Simulator DataFrame must include a 'revenue' KPI column.")
    out["population"] = 1.0
    return out


def build_meridian_input_data(df: pd.DataFrame):
    """Return Meridian ``InputData`` from a prepared DataFrame (see ``simulation_df_to_meridian_format``)."""
    from meridian import constants
    from meridian.data import data_frame_input_data_builder

    channels = _channel_names_from_df(df)
    if len(channels) < 1:
        raise ValueError("No media channels found (expected * _spend columns).")

    media_cols = [f"{c}_impressions" for c in channels]
    spend_cols = [f"{c}_spend" for c in channels]
    for col in media_cols + spend_cols + ["revenue", "time", "population"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE,
        default_kpi_column="revenue",
        default_time_column="time",
    )
    builder = builder.with_kpi(df, kpi_col="revenue", time_col="time")
    builder = builder.with_population(df, population_col="population")
    builder = builder.with_media(
        df,
        media_cols=media_cols,
        media_spend_cols=spend_cols,
        media_channels=channels,
        time_col="time",
    )
    return builder.build()


def fit_meridian(
    df: pd.DataFrame,
    run_cfg: MeridianRunConfig,
) -> Tuple[Any, Dict[str, Any]]:
    """Fit Meridian and return (mmm_model, summary_dict).

    ``summary_dict`` has keys: rhat_max, n_params, ok (bool), note (str).
    """
    import arviz as az
    import tensorflow as tf
    import tensorflow_probability as tfp
    from meridian import constants
    from meridian.model import model, prior_distribution, spec

    prepared = simulation_df_to_meridian_format(df)
    data = build_meridian_input_data(prepared)

    # Quiet TF logs in Streamlit
    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass

    n_media = len(_channel_names_from_df(prepared))
    if run_cfg.per_channel_roi_priors and run_cfg.channel_roi_mus and run_cfg.channel_roi_sigmas:
        if len(run_cfg.channel_roi_mus) != n_media or len(run_cfg.channel_roi_sigmas) != n_media:
            raise ValueError("Per-channel ROI priors need one μ and one σ per media channel.")
        loc = tf.constant(run_cfg.channel_roi_mus, dtype=tf.float32)
        scale = tf.constant(run_cfg.channel_roi_sigmas, dtype=tf.float32)
    else:
        loc = tf.fill([n_media], float(run_cfg.roi_log_mu))
        scale = tf.fill([n_media], float(run_cfg.roi_log_sigma))
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(loc=loc, scale=scale, name=constants.ROI_M)
    )
    model_spec = spec.ModelSpec(prior=prior, enable_aks=run_cfg.enable_aks)
    mmm = model.Meridian(input_data=data, model_spec=model_spec)

    mmm.sample_prior(int(run_cfg.n_prior))
    mmm.sample_posterior(
        n_chains=int(run_cfg.n_chains),
        n_adapt=int(run_cfg.n_adapt),
        n_burnin=int(run_cfg.n_burnin),
        n_keep=int(run_cfg.n_keep),
        seed=int(run_cfg.seed),
    )

    summary: Dict[str, Any] = {"ok": True, "note": ""}
    idata = mmm.inference_data
    if idata is None or not hasattr(idata, "posterior"):
        summary["ok"] = False
        summary["note"] = "inference_data missing after sampling."
        return mmm, summary

    rhat_ds = az.rhat(idata)
    rhat_max = float(np.nanmax(rhat_ds.to_array().values))
    summary["rhat_max"] = rhat_max
    if np.isfinite(rhat_max) and rhat_max > 1.1:
        summary["note"] = (
            f"Max R-hat = {rhat_max:.3f} (values above ~1.1 often indicate poor chain mixing; "
            "try more adaptation/burn-in or more chains/keeps on a longer series)."
        )
    return mmm, summary


def arviz_posterior_table(mmm: Any, max_rows: int = 40) -> Optional[pd.DataFrame]:
    """Small posterior summary table for the UI."""
    try:
        import arviz as az
    except Exception:
        return None
    idata = getattr(mmm, "inference_data", None)
    if idata is None:
        return None
    s = az.summary(idata, round_to=4)
    return s.head(max_rows) if len(s) > max_rows else s


def meridian_visualizations(
    mmm: Any,
    *,
    analysis_batch_size: int = 80,
) -> Dict[str, Any]:
    """Model fit metrics/charts + fixed-budget optimization plots (Altair), best-effort."""
    from meridian.analysis import visualizer
    from meridian.analysis import optimizer as optmod

    out: Dict[str, Any] = {}

    try:
        diag = visualizer.ModelDiagnostics(mmm, use_kpi=False)
        out["fit_metrics"] = diag.predictive_accuracy_table(batch_size=analysis_batch_size)
    except Exception as e:
        out["fit_metrics_error"] = str(e)

    try:
        diag = visualizer.ModelDiagnostics(mmm, use_kpi=False)
        out["rhat_chart"] = diag.plot_rhat_boxplot()
    except Exception as e:
        out["rhat_error"] = str(e)

    try:
        mf = visualizer.ModelFit(mmm, use_kpi=False)
        out["model_fit_chart"] = mf.plot_model_fit()
    except Exception as e:
        out["model_fit_error"] = str(e)

    try:
        ms = visualizer.MediaSummary(mmm, use_kpi=False)
        out["media_roi_bar_chart"] = ms.plot_roi_bar_chart(include_ci=True)
    except Exception as e:
        out["media_roi_error"] = str(e)

    try:
        bo = optmod.BudgetOptimizer(mmm)
        opt = bo.optimize(
            use_posterior=True,
            fixed_budget=True,
            budget=None,
            batch_size=analysis_batch_size,
        )
        out["optimization"] = opt
        out["spend_recommendation_df"] = spend_recommendation_dataframe(opt)
    except Exception as e:
        out["optimization_error"] = str(e)

    o = out.get("optimization")
    if o is not None:
        try:
            out["opt_outcome_delta_chart"] = o.plot_incremental_outcome_delta()
        except Exception as e:
            out["opt_outcome_delta_error"] = str(e)
        try:
            out["opt_spend_delta_chart"] = o.plot_spend_delta()
        except Exception as e:
            out["opt_spend_delta_error"] = str(e)
        try:
            out["opt_spend_pie_nonopt"] = o.plot_budget_allocation(optimized=False)
        except Exception as e:
            out["opt_pie_nonopt_error"] = str(e)
        try:
            out["opt_spend_pie_opt"] = o.plot_budget_allocation(optimized=True)
        except Exception as e:
            out["opt_pie_opt_error"] = str(e)

    return out


def spend_recommendation_dataframe(opt: Any) -> Optional[pd.DataFrame]:
    """Per-channel current vs optimized spend and suggested change (from BudgetOptimizer)."""
    try:
        s0 = opt.nonoptimized_data["spend"]
        s1 = opt.optimized_data["spend"]
        dlt = s1 - s0
        channels = [str(c) for c in np.atleast_1d(s0["channel"].values)]
        base = np.atleast_1d(s0.values).astype(float)
        optv = np.atleast_1d(s1.values).astype(float)
        delta = np.atleast_1d(dlt.values).astype(float)
    except Exception:
        return None
    if len(channels) != len(base):
        return None
    df = pd.DataFrame(
        {
            "channel": channels,
            "spend_baseline": base,
            "spend_optimized": optv,
            "delta": delta,
        }
    )
    df["delta_pct"] = np.where(
        df["spend_baseline"] > 0,
        100.0 * df["delta"] / df["spend_baseline"],
        0.0,
    )
    df["suggestion"] = np.where(
        df["delta"].abs() < 1e-9,
        "Hold (near current)",
        np.where(df["delta"] > 0, "Increase spend", "Decrease spend"),
    )
    return df
