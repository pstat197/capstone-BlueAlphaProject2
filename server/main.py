"""FastAPI entry point for the React UI.

Run from repo root:
    uvicorn server.main:app --reload --port 8000
"""

from __future__ import annotations

import io
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

from server.correlations import derive_correlation_results  # noqa: E402
from server.serializers import serialize_correlation, serialize_run_dataframe  # noqa: E402
from server.store import get_run_record, list_run_records, save_run_record  # noqa: E402

app = FastAPI(
    title="BlueAlpha Simulator API",
    version="0.1.0",
    description="HTTP wrapper around the marketing simulator + Bayesian MMM pipelines.",
)

# Vite dev server runs on 5173 by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
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
# Meridian (stub for now — full parity comes in a follow-up branch)
# ---------------------------------------------------------------------------


@app.get("/api/meridian/status")
def meridian_status() -> Dict[str, Any]:
    try:
        import meridian  # type: ignore  # noqa: F401

        installed = True
    except Exception:
        installed = False
    return {
        "installed": installed,
        "ui_status": "coming_soon",
        "message": (
            "The Bayesian MMM tab is still served by the Streamlit app on this branch. "
            "It will land in the React UI in a follow-up."
        ),
    }
