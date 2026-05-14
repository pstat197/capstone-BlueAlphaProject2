"""Bayesian MMM (Google Meridian) HTTP layer for the React UI.

The Streamlit ``ui_meridian_tab`` runs Meridian inline because it can keep the
fitted model object on ``st.session_state``. The React app cannot hold a Python
object across the wire, so we:

* run the fit in a background **thread** (Meridian / TFP block the GIL during
  sampling but Python's threading is enough for a single-fit-at-a-time UI),
* persist a *JSON-only* projection of the fit results so the UI can re-open
  them after a refresh / on a fresh client without re-fitting (sampling is
  minutes-long), and
* expose three small endpoints (``POST`` to start a fit, ``GET`` for status,
  ``GET`` for the serialized results) that the React MMM page polls.

Caching key is ``(simulator_config_hash, mmm_run_config_hash)``: identical
priors + MCMC against the same simulator dataset always re-uses the prior fit.
"""

from __future__ import annotations

import hashlib
import json
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.cache import _CACHE_ROOT, try_load_cached  # type: ignore[attr-defined]
from app.meridian_mmm import (
    MeridianRunConfig,
    channel_names_from_simulator_df,
    fit_meridian,
    format_predictive_metrics_df,
    meridian_import_status,
    meridian_visualizations,
    spend_recommendation_dataframe,
)
from app.mmm_roi_forest import (
    meridian_posterior_roi_forest_rows,
    roi_m_rhat_by_media_channel,
    true_roi_by_channel_map,
)
from server.store import get_run_record


_MMM_CACHE_ROOT = _CACHE_ROOT / "mmm"


def _ensure_mmm_cache_dir() -> Path:
    _MMM_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _MMM_CACHE_ROOT


# ---------------------------------------------------------------------------
# Request shape (matches Streamlit per-channel priors + MCMC presets)
# ---------------------------------------------------------------------------


# (chains, n_adapt, n_burnin, n_keep, n_prior). Mirrors ui_meridian_tab.MCMC_PRESETS.
MCMC_PRESETS: Dict[str, tuple] = {
    "fast": (2, 500, 200, 200, 200),
    "balanced": (4, 1000, 500, 500, 500),
    "slow": (4, 2000, 500, 1000, 1000),
}


@dataclass
class MmmRunRequest:
    """JSON-friendly payload posted by the React UI to start a fit."""

    config_hash: str
    profile: str = "balanced"  # "fast" | "balanced" | "slow" | "custom"
    # When profile == "custom" these are required; otherwise overridden.
    n_chains: int = 4
    n_adapt: int = 1000
    n_burnin: int = 500
    n_keep: int = 500
    n_prior: int = 500
    seed: int = 0
    enable_aks: bool = False
    knots: Optional[List[int]] = None
    # Per-channel priors. Lengths must match the data's channel count.
    channel_roi_mus: Optional[List[float]] = None
    channel_roi_sigmas: Optional[List[float]] = None

    def resolved_mcmc(self) -> tuple:
        if self.profile in MCMC_PRESETS:
            return MCMC_PRESETS[self.profile]
        return (
            int(self.n_chains),
            int(self.n_adapt),
            int(self.n_burnin),
            int(self.n_keep),
            int(self.n_prior),
        )

    def to_run_config(self, n_channels: int) -> MeridianRunConfig:
        nc, na, nb, nk, npr = self.resolved_mcmc()
        if self.channel_roi_mus and self.channel_roi_sigmas:
            mus = list(self.channel_roi_mus)
            sigs = list(self.channel_roi_sigmas)
            if len(mus) != n_channels or len(sigs) != n_channels:
                raise ValueError(
                    f"Per-channel priors length mismatch: got μ={len(mus)} σ={len(sigs)}, "
                    f"expected {n_channels} for the loaded simulator data."
                )
            return MeridianRunConfig(
                per_channel_roi_priors=True,
                channel_roi_mus=mus,
                channel_roi_sigmas=sigs,
                n_chains=nc,
                n_adapt=na,
                n_burnin=nb,
                n_keep=nk,
                n_prior=npr,
                seed=int(self.seed),
                enable_aks=bool(self.enable_aks),
                knots=list(self.knots) if self.knots else None,
            )
        return MeridianRunConfig(
            n_chains=nc,
            n_adapt=na,
            n_burnin=nb,
            n_keep=nk,
            n_prior=npr,
            seed=int(self.seed),
            enable_aks=bool(self.enable_aks),
            knots=list(self.knots) if self.knots else None,
        )

    def cache_key(self) -> str:
        """Stable hash of (priors + MCMC), used together with the simulator
        config_hash to look up a previously serialized fit. Profile is
        materialized first so 'balanced' and the equivalent 'custom' numbers
        share a cache file."""
        nc, na, nb, nk, npr = self.resolved_mcmc()
        payload = {
            "channel_roi_mus": list(self.channel_roi_mus) if self.channel_roi_mus else None,
            "channel_roi_sigmas": list(self.channel_roi_sigmas) if self.channel_roi_sigmas else None,
            "n_chains": nc,
            "n_adapt": na,
            "n_burnin": nb,
            "n_keep": nk,
            "n_prior": npr,
            "seed": int(self.seed),
            "enable_aks": bool(self.enable_aks),
            "knots": list(self.knots) if self.knots else None,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Job manager (in-process; per-uvicorn-worker)
# ---------------------------------------------------------------------------


@dataclass
class MmmJobState:
    job_id: str
    status: str  # queued | running | succeeded | failed
    config_hash: str
    cache_key: str
    profile: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    # Serialized results (see _serialize_fit). Mirrored to disk on success.
    results: Optional[Dict[str, Any]] = None
    # Note from the underlying summary (e.g. high R-hat warning).
    note: Optional[str] = None
    # Lightweight progress hint for the UI ("preparing", "sampling", "diagnostics", "optimizing").
    stage: str = "queued"
    # Whether the result was loaded from the on-disk cache (no new sampling done).
    cache_hit: bool = False
    n_channels: int = 0
    n_weeks: int = 0
    channels: List[str] = field(default_factory=list)


class _JobRegistry:
    """Thread-safe in-memory registry for pending and finished MMM fits."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, MmmJobState] = {}

    def create(self, *, config_hash: str, cache_key: str, profile: str) -> MmmJobState:
        job = MmmJobState(
            job_id=uuid.uuid4().hex[:16],
            status="queued",
            config_hash=config_hash,
            cache_key=cache_key,
            profile=profile,
            stage="queued",
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[MmmJobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)


_REGISTRY = _JobRegistry()


# ---------------------------------------------------------------------------
# Result serialization (JSON only; no Meridian / Altair objects leak to UI)
# ---------------------------------------------------------------------------


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _fit_metrics_rows(viz: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = viz.get("fit_metrics")
    df = format_predictive_metrics_df(raw) if raw is not None else None
    if df is None or not hasattr(df, "empty") or df.empty:
        return []
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: Dict[str, Any] = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (np.floating, float)):
                rec[str(col)] = _safe_float(val)
            elif isinstance(val, (np.integer, int, bool)):
                rec[str(col)] = int(val) if not isinstance(val, bool) else bool(val)
            else:
                rec[str(col)] = None if pd.isna(val) else str(val)
        out.append(rec)
    return out


def _budget_optimization_payload(
    viz: Dict[str, Any],
    n_weeks: int,
) -> Dict[str, Any]:
    """Convert Meridian's optimizer outputs into JSON-friendly rows + pies.

    The Streamlit version renders Altair pies returned by Meridian directly;
    we serialize the underlying ``spend_recommendation_dataframe`` instead and
    let the React UI draw equivalent Recharts pies + the per-channel
    reallocation copy that Streamlit shows below the pies.
    """
    err = viz.get("optimization_error")
    if err:
        return {"error": str(err)}
    df = viz.get("spend_recommendation_df")
    if df is None and viz.get("optimization") is not None:
        df = spend_recommendation_dataframe(viz["optimization"])
    if df is None or not hasattr(df, "empty") or df.empty:
        return {"error": "No spend recommendation could be computed."}

    n_weeks = max(1, int(n_weeks))
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        baseline = _safe_float(r.get("spend_baseline")) or 0.0
        optimized = _safe_float(r.get("spend_optimized")) or 0.0
        weekly_baseline = baseline / n_weeks
        weekly_optimized = optimized / n_weeks
        delta_weekly = weekly_optimized - weekly_baseline
        change_pct = (
            100.0 * delta_weekly / weekly_baseline
            if weekly_baseline > 1e-9
            else 0.0
        )
        rows.append(
            {
                "channel": str(r.get("channel")),
                "spend_baseline_total": baseline,
                "spend_optimized_total": optimized,
                "spend_baseline_weekly": weekly_baseline,
                "spend_optimized_weekly": weekly_optimized,
                "delta_weekly": delta_weekly,
                "change_pct": change_pct,
            }
        )

    total_baseline = sum(r["spend_baseline_total"] for r in rows)
    total_optimized = sum(r["spend_optimized_total"] for r in rows)
    pies = {
        "current": [
            {
                "channel": r["channel"],
                "value": r["spend_baseline_total"],
                "share": (
                    100.0 * r["spend_baseline_total"] / total_baseline
                    if total_baseline > 1e-9
                    else 0.0
                ),
            }
            for r in rows
        ],
        "optimized": [
            {
                "channel": r["channel"],
                "value": r["spend_optimized_total"],
                "share": (
                    100.0 * r["spend_optimized_total"] / total_optimized
                    if total_optimized > 1e-9
                    else 0.0
                ),
            }
            for r in rows
        ],
    }
    return {
        "rows": rows,
        "pies": pies,
        "total_spend_baseline": total_baseline,
        "total_spend_optimized": total_optimized,
        "n_weeks": n_weeks,
    }


def _serialize_fit(
    *,
    mmm: Any,
    summary: Dict[str, Any],
    viz: Dict[str, Any],
    sim_config: Dict[str, Any],
    n_weeks: int,
    channels: List[str],
) -> Dict[str, Any]:
    """All UI-bound data for a finished fit, ready to JSON-serialize."""
    # ROI forest rows already carry mean / ci / true_roi and are light-weight.
    true_map = true_roi_by_channel_map(sim_config)
    try:
        roi_rows = meridian_posterior_roi_forest_rows(mmm, true_map=true_map)
    except Exception as exc:  # noqa: BLE001
        roi_rows = []
        roi_error = str(exc)
    else:
        roi_error = None

    rhat_by_channel = roi_m_rhat_by_media_channel(mmm) or {}
    fit_rows = _fit_metrics_rows(viz)
    budget = _budget_optimization_payload(viz, n_weeks=n_weeks)

    return {
        "summary": {
            "rhat_max": _safe_float(summary.get("rhat_max")),
            "note": summary.get("note") or None,
            "ok": bool(summary.get("ok", True)),
        },
        "channels": channels,
        "n_weeks": int(n_weeks),
        "fit_metrics": fit_rows,
        "fit_metrics_error": viz.get("fit_metrics_error"),
        "roi_forest": {
            "rows": roi_rows,
            "rhat_by_channel": {k: _safe_float(v) for k, v in rhat_by_channel.items()},
            "error": roi_error,
        },
        "budget_optimization": budget,
    }


# ---------------------------------------------------------------------------
# On-disk cache (JSON; (config_hash, run_cfg_hash) keyed)
# ---------------------------------------------------------------------------


def _result_path(config_hash: str, cache_key: str) -> Path:
    return _ensure_mmm_cache_dir() / f"{config_hash}__{cache_key}.json"


def _load_cached_result(config_hash: str, cache_key: str) -> Optional[Dict[str, Any]]:
    p = _result_path(config_hash, cache_key)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_cached_result(config_hash: str, cache_key: str, payload: Dict[str, Any]) -> None:
    p = _result_path(config_hash, cache_key)
    try:
        p.write_text(json.dumps(payload, default=str), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


def status_payload() -> Dict[str, Any]:
    """Replacement for the old ``coming_soon`` stub. Reports whether the
    optional Meridian dependency is importable so the UI can render a
    friendly install hint when it is not."""
    ok, err = meridian_import_status()
    return {
        "installed": bool(ok),
        "ui_status": "available" if ok else "unavailable",
        "message": (
            "Meridian is installed; ready to fit."
            if ok
            else (
                "Meridian is not importable from the API's Python environment. "
                "Install the optional MMM extras (requirements-meridian.txt) into "
                "the same venv that runs the FastAPI server."
            )
        ),
        "error": err,
    }


def _resolve_simulator_data(config_hash: str) -> tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """Reload the cached simulator DataFrame + config for ``config_hash``."""
    record = get_run_record(config_hash)
    if record is None:
        raise LookupError(
            "Unknown config hash. Run the simulator first, then come back to MMM."
        )
    cached = try_load_cached(config_hash)
    if cached is None:
        raise LookupError(
            "Cached simulator output for this config was cleared. Re-run the simulator."
        )
    df, _meta = cached
    sim_config = record.get("config") or {}
    channels = channel_names_from_simulator_df(df)
    if not channels:
        raise ValueError("Simulator DataFrame has no channels (no *_spend columns).")
    return df, sim_config, channels


def _run_fit_thread(job_id: str, request: MmmRunRequest) -> None:
    job = _REGISTRY.get(job_id)
    if job is None:
        return
    _REGISTRY.update(
        job_id,
        status="running",
        stage="preparing",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        df, sim_config, channels = _resolve_simulator_data(request.config_hash)
        n_channels = len(channels)
        n_weeks = int(len(df))
        _REGISTRY.update(
            job_id,
            n_channels=n_channels,
            n_weeks=n_weeks,
            channels=channels,
            stage="sampling",
        )
        run_cfg = request.to_run_config(n_channels)
        mmm, summary = fit_meridian(df, run_cfg)
        _REGISTRY.update(job_id, stage="diagnostics")
        viz = meridian_visualizations(mmm)
        _REGISTRY.update(job_id, stage="serializing")
        results = _serialize_fit(
            mmm=mmm,
            summary=summary,
            viz=viz,
            sim_config=sim_config,
            n_weeks=n_weeks,
            channels=channels,
        )
        _save_cached_result(request.config_hash, request.cache_key(), results)
        _REGISTRY.update(
            job_id,
            status="succeeded",
            stage="done",
            results=results,
            note=results.get("summary", {}).get("note"),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:  # noqa: BLE001
        _REGISTRY.update(
            job_id,
            status="failed",
            stage="error",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        # Bubble traceback into the server log; the UI gets the short message.
        traceback.print_exc()


def start_fit(request: MmmRunRequest) -> Dict[str, Any]:
    """Either return a previously cached fit (synchronously) or kick off a
    background thread and return a queued job descriptor."""
    ok, err = meridian_import_status()
    if not ok:
        raise RuntimeError(
            "Meridian is not installed. " + (err or "Install requirements-meridian.txt and restart.")
        )

    cache_key = request.cache_key()
    cached = _load_cached_result(request.config_hash, cache_key)
    if cached is not None:
        # Materialize a "succeeded" job synchronously so the UI's polling
        # loop converges instantly without paying for a sampling run.
        # We also reload n_weeks / channels for badge text.
        try:
            df, _sim, channels = _resolve_simulator_data(request.config_hash)
            n_weeks = int(len(df))
        except Exception:
            channels = list(cached.get("channels") or [])
            n_weeks = int(cached.get("n_weeks") or 0)
        job = _REGISTRY.create(
            config_hash=request.config_hash,
            cache_key=cache_key,
            profile=request.profile,
        )
        _REGISTRY.update(
            job.job_id,
            status="succeeded",
            stage="done",
            results=cached,
            cache_hit=True,
            note=cached.get("summary", {}).get("note"),
            n_weeks=n_weeks,
            n_channels=len(channels),
            channels=channels,
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        refreshed = _REGISTRY.get(job.job_id)
        return _job_summary(refreshed) if refreshed else asdict(job)

    job = _REGISTRY.create(
        config_hash=request.config_hash,
        cache_key=cache_key,
        profile=request.profile,
    )
    t = threading.Thread(
        target=_run_fit_thread, args=(job.job_id, request), name=f"mmm-fit-{job.job_id}", daemon=True
    )
    t.start()
    refreshed = _REGISTRY.get(job.job_id)
    return _job_summary(refreshed) if refreshed else asdict(job)


def _job_summary(job: MmmJobState) -> Dict[str, Any]:
    """Status-only projection (results omitted; fetched via /results)."""
    d = asdict(job)
    d.pop("results", None)
    return d


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    job = _REGISTRY.get(job_id)
    if job is None:
        return None
    return _job_summary(job)


def get_job_results(job_id: str) -> Optional[Dict[str, Any]]:
    job = _REGISTRY.get(job_id)
    if job is None:
        return None
    if job.status != "succeeded" or job.results is None:
        return {"job": _job_summary(job), "results": None}
    return {"job": _job_summary(job), "results": job.results}


def lookup_cached_fit(config_hash: str, cache_key: str) -> Optional[Dict[str, Any]]:
    """Direct cache lookup (no job). Lets the UI prompt 'a fit for these
    settings already exists — open it?' before the user clicks Run."""
    return _load_cached_result(config_hash, cache_key)
