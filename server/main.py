"""FastAPI entry point for the React UI.

Run from repo root:
    uvicorn server.main:app --reload --port 8000
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Ensure repo root is importable when launched via `python -m` or uvicorn elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.cache import (  # noqa: E402
    cache_entry_exists,
    canonical_config_hash,
    clear_run_cache,
    run_with_cache,
    try_load_cached,
)
from app.pipeline_runner import run_pipeline  # noqa: E402
from app.ui_yaml_io import load_example_text, load_ui_schema, yaml_dump  # noqa: E402
from scripts.config.loader import load_config_from_dict  # noqa: E402
from scripts.ground_truth_io import extract_ground_truth  # noqa: E402
from scripts.revenue_simulation.revenue_generation import _seasonality  # noqa: E402
from scripts.revenue_simulation.seasonality_fit import (  # noqa: E402
    fit_pattern_multipliers_to_fourier,
)

from server.correlations import derive_correlation_results  # noqa: E402
from server.serializers import serialize_correlation, serialize_run_dataframe  # noqa: E402
from server.store import get_run_record, list_run_records, save_run_record  # noqa: E402
from server.validate import validate_config  # noqa: E402

app = FastAPI(
    title="BlueAlpha Simulator API",
    version="0.1.0",
    description="HTTP wrapper around the marketing simulator + Bayesian MMM pipelines.",
)

# CORS allow-list.
#
# Local dev origins (Vite dev server + Vite preview) are always allowed.
# Production origins (e.g. the Vercel deployment) are added via the
# CORS_ALLOW_ORIGINS env var as a comma-separated list:
#
#     CORS_ALLOW_ORIGINS="https://bluealpha.vercel.app,https://bluealpha-mmm.com"
#
# Set CORS_ALLOW_ORIGIN_REGEX to match the Vercel preview-deploy pattern
# ("https://bluealpha-*.vercel.app") so PR previews work without re-deploying
# the backend.
_DEFAULT_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
]
_extra_origins = [
    o.strip()
    for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",")
    if o.strip()
]
_origin_regex = os.environ.get("CORS_ALLOW_ORIGIN_REGEX") or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_DEV_ORIGINS + _extra_origins,
    allow_origin_regex=_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Merged simulator config (YAML-shaped dict) plus optional overrides."""

    config: Dict[str, Any] = Field(..., description="Full merged simulator config.")


class YamlValidateRequest(BaseModel):
    yaml_text: str


class YamlValidateResponse(BaseModel):
    ok: bool
    error: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None


class YamlDumpRequest(BaseModel):
    config: Dict[str, Any]


class YamlDumpResponse(BaseModel):
    yaml_text: str


# ---------------------------------------------------------------------------
# Bootstrap endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "name": "bluealpha-simulator-api"}


@app.get("/api/schema")
def get_schema() -> Dict[str, Any]:
    return load_ui_schema()


@app.get("/api/example-config")
def get_example_config() -> Dict[str, Any]:
    text = load_example_text()
    parsed = yaml.safe_load(text) or {}
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=500, detail="example.yaml did not parse to an object")
    return {"config": parsed, "yaml_text": text}


# ---------------------------------------------------------------------------
# YAML helpers (used by the Advanced YAML editor)
# ---------------------------------------------------------------------------


@app.post("/api/yaml/validate", response_model=YamlValidateResponse)
def validate_yaml(payload: YamlValidateRequest) -> YamlValidateResponse:
    text = payload.yaml_text or ""
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return YamlValidateResponse(ok=False, error=str(e))
    if parsed is None:
        return YamlValidateResponse(ok=True, parsed={})
    if not isinstance(parsed, dict):
        return YamlValidateResponse(ok=False, error="YAML must parse to a mapping (dict).")
    return YamlValidateResponse(ok=True, parsed=parsed)


@app.post("/api/yaml/dump", response_model=YamlDumpResponse)
def dump_yaml(payload: YamlDumpRequest) -> YamlDumpResponse:
    return YamlDumpResponse(yaml_text=yaml_dump(payload.config or {}))


class FitPatternRequest(BaseModel):
    """Categorical multipliers (one cycle), optional harmonic cap."""

    pattern: List[float] = Field(
        ..., description="One cycle of multipliers (e.g. monthly indices)."
    )
    K: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Harmonic cap. Defaults to a smoothing-friendly value computed from "
            "the pattern length when omitted."
        ),
    )


@app.post("/api/seasonality/fit-pattern")
def fit_pattern_endpoint(payload: FitPatternRequest) -> Dict[str, Any]:
    """Least-squares fit a categorical multiplier pattern to a deterministic
    Fourier seasonality config. Returns ``{type, period, K, intercept,
    coefficients}`` ready to drop into ``seasonality_config``.

    Useful for translating observed weekly / monthly indices into a smooth
    cycle the simulator can reproduce.
    """
    pattern = list(payload.pattern or [])
    if len(pattern) < 2:
        raise HTTPException(
            status_code=400,
            detail="pattern must contain at least 2 multipliers",
        )
    try:
        result = fit_pattern_multipliers_to_fourier(pattern, K=payload.K)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=400, detail="Could not fit pattern")
    return result


class EvaluateSeasonalityRequest(BaseModel):
    """Render an arbitrary ``seasonality_config`` against a week grid."""

    config: Dict[str, Any] = Field(
        ..., description="A seasonality_config dict (sin/categorical/fourier/hybrid)."
    )
    weeks: int = Field(
        52,
        ge=1,
        le=520,
        description="Number of weeks to evaluate. Capped at 520 to keep the wire payload small.",
    )
    seed: Optional[int] = Field(
        None,
        description=(
            "Fallback seed for random-Fourier components. Ignored for deterministic configs."
        ),
    )


@app.post("/api/seasonality/evaluate")
def evaluate_seasonality_endpoint(payload: EvaluateSeasonalityRequest) -> Dict[str, Any]:
    """Run the simulator's own ``_seasonality`` evaluator and return the
    resulting per-week multipliers.

    The React Fourier editor renders a pure-TS port of the deterministic
    Fourier math for snappy live previews; this endpoint is the
    source-of-truth overlay so the user can verify the preview matches what
    the pipeline will actually use, and so random-Fourier draws (no
    coefficients) still have *something* to show in the chart.
    """
    import numpy as np

    cfg = payload.config or {}
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail="config must be a mapping")

    weeks = int(payload.weeks)
    t = np.arange(weeks, dtype=float)
    try:
        multipliers = _seasonality(t, cfg, fallback_seed=payload.seed)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    values = [float(v) for v in np.asarray(multipliers, dtype=float).ravel().tolist()]
    return {"multipliers": values, "weeks": weeks}


@app.post("/api/config/validate")
def validate_config_endpoint(payload: RunRequest) -> Dict[str, Any]:
    """Structured, field-level validation for the React form.

    Returns ``{"ok": bool, "issues": [{path, message, severity, section}]}``.
    Errors carry an empty path when the issue is global (e.g. the loader
    raised), otherwise a JSON-Pointer-ish array the UI can use to pin
    badges to specific controls.
    """
    return validate_config(payload.config or {})


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


def _safe_extract_ground_truth(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Derive the ground-truth artifact from a merged user config.

    Loading the config + extracting ground truth is cheap (no simulation)
    and we want it available on both fresh and cached paths so the React
    Results page can always show it. We swallow loader errors here so
    that a broken config still lets the rest of the run payload through —
    the create_run handler validates separately and surfaces real errors.
    """
    try:
        loaded = load_config_from_dict(config)
    except Exception:
        return None
    try:
        return extract_ground_truth(loaded)
    except Exception:
        return None


def _build_run_payload(
    *,
    config: Dict[str, Any],
    config_hash: str,
    run_identifier: str,
    cache_hit: bool,
    df: Any,
    corr_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    serialized = serialize_run_dataframe(df)
    # `run_with_cache` skips correlation analysis on cache hits to keep the fast
    # path cheap. Recompute it here so the React diagnostics route works for
    # any run (fresh or cached, POST or GET).
    if corr_results is None:
        corr_results = derive_correlation_results(df, config)
    return {
        "run_id": run_identifier,
        "config_hash": config_hash,
        "cache_hit": cache_hit,
        "config": config,
        "weeks": serialized["weeks"],
        "totals": serialized["totals"],
        "channels": serialized["channels"],
        "preview": serialized["preview"],
        "correlation": serialize_correlation(corr_results),
        "ground_truth": _safe_extract_ground_truth(config),
    }


@app.post("/api/runs")
def create_run(payload: RunRequest) -> Dict[str, Any]:
    config = payload.config or {}
    if not (config.get("channel_list") or []):
        raise HTTPException(status_code=400, detail="Add at least one channel before running.")
    try:
        df, run_identifier, cache_hit, config_hash, corr_results = run_with_cache(
            config, run_pipeline
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline error: {e}") from e

    save_run_record(
        config_hash,
        run_identifier=run_identifier,
        config=config,
        cache_hit=cache_hit,
    )

    return _build_run_payload(
        config=config,
        config_hash=config_hash,
        run_identifier=run_identifier or "run",
        cache_hit=cache_hit,
        df=df,
        corr_results=corr_results,
    )


@app.get("/api/runs")
def list_runs() -> Dict[str, Any]:
    return {"runs": list_run_records()}


@app.get("/api/runs/{config_hash}")
def get_run(config_hash: str) -> Dict[str, Any]:
    record = get_run_record(config_hash)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown run id (config hash).")
    cached = try_load_cached(config_hash)
    if cached is None:
        raise HTTPException(
            status_code=410,
            detail="Run metadata exists but cached results were cleared. Re-run to regenerate.",
        )
    df, _meta = cached
    return _build_run_payload(
        config=record.get("config") or {},
        config_hash=config_hash,
        run_identifier=record.get("run_identifier") or "run",
        cache_hit=True,
        df=df,
        corr_results=None,
    )


@app.get("/api/runs/{config_hash}/csv")
def download_csv(config_hash: str) -> StreamingResponse:
    cached = try_load_cached(config_hash)
    if cached is None:
        raise HTTPException(status_code=404, detail="No cached results for this run id.")
    df, _meta = cached
    record = get_run_record(config_hash)
    rid = (record or {}).get("run_identifier") or "simulation"
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{rid}.csv"'}
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)


@app.post("/api/cache/clear")
def clear_cache() -> Dict[str, Any]:
    n = clear_run_cache()
    return {"removed": n}


# ---------------------------------------------------------------------------
# Helpers also used by `/api/runs` (for previewing what a config would hash to)
# ---------------------------------------------------------------------------


@app.post("/api/config/hash")
def hash_config(payload: RunRequest) -> Dict[str, str]:
    return {"config_hash": canonical_config_hash(payload.config or {})}


@app.get("/api/cache/{config_hash}")
def cache_status(config_hash: str) -> Dict[str, Any]:
    """Lightweight cache lookup: tells the React UI whether a hash is cached
    without loading the dataframe. The Simulator page debounces a call to
    this so the run button can show a "cached — re-open" affordance live."""
    exists = cache_entry_exists(config_hash)
    record = get_run_record(config_hash)
    return {
        "config_hash": config_hash,
        "cached": bool(exists),
        "run_identifier": (record or {}).get("run_identifier") if record else None,
        "last_seen_at": (record or {}).get("last_seen_at") if record else None,
    }


# ---------------------------------------------------------------------------
# Bayesian MMM (Meridian)
# ---------------------------------------------------------------------------


from server.mmm import (  # noqa: E402
    MmmRunRequest,
    get_job_results,
    get_job_status,
    lookup_cached_fit,
    start_fit,
    status_payload,
)


class MmmFitRequest(BaseModel):
    """Mirror of ``MmmRunRequest``. Pydantic gates / validates input shape."""

    config_hash: str = Field(..., description="Hash of the cached simulator run to fit on.")
    profile: str = Field("balanced", description="MCMC preset: fast | balanced | slow | custom.")
    n_chains: int = Field(4, ge=1, le=32)
    n_adapt: int = Field(1000, ge=100, le=20_000)
    n_burnin: int = Field(500, ge=0, le=20_000)
    n_keep: int = Field(500, ge=50, le=10_000)
    n_prior: int = Field(500, ge=50, le=5_000)
    seed: int = Field(0, ge=0, le=2_147_483_647)
    enable_aks: bool = False
    knots: Optional[List[int]] = None
    channel_roi_mus: Optional[List[float]] = None
    channel_roi_sigmas: Optional[List[float]] = None


@app.get("/api/meridian/status")
def meridian_status() -> Dict[str, Any]:
    return status_payload()


@app.post("/api/mmm/fits")
def create_mmm_fit(payload: MmmFitRequest) -> Dict[str, Any]:
    req = MmmRunRequest(**payload.model_dump())
    try:
        return start_fit(req)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/mmm/fits/{job_id}")
def get_mmm_fit_status(job_id: str) -> Dict[str, Any]:
    job = get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown MMM job id.")
    return job


@app.get("/api/mmm/fits/{job_id}/results")
def get_mmm_fit_results(job_id: str) -> Dict[str, Any]:
    out = get_job_results(job_id)
    if out is None:
        raise HTTPException(status_code=404, detail="Unknown MMM job id.")
    return out


@app.get("/api/mmm/cache")
def lookup_mmm_cache(config_hash: str, cache_key: str) -> Dict[str, Any]:
    """Probe whether a fit already exists for this (simulator, mmm-config) pair
    so the UI can flip the Run button into a 'open cached fit' affordance."""
    cached = lookup_cached_fit(config_hash, cache_key)
    return {"cached": cached is not None, "config_hash": config_hash, "cache_key": cache_key}
