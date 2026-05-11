"""Server-side sidecar store: persist the merged config alongside Streamlit's CSV/JSON cache.

We never mutate `app.cache` files so the Streamlit app keeps working. Configs and run metadata
needed by the React run-history drawer live under `app/.cache/runs/server/`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.cache import _CACHE_ROOT  # type: ignore[attr-defined]

_SERVER_DIR = _CACHE_ROOT / "server"


def _ensure_dir() -> Path:
    _SERVER_DIR.mkdir(parents=True, exist_ok=True)
    return _SERVER_DIR


def _config_path(config_hash: str) -> Path:
    return _ensure_dir() / f"{config_hash}.json"


def save_run_record(
    config_hash: str,
    *,
    run_identifier: str,
    config: Dict[str, Any],
    cache_hit: bool,
) -> None:
    """Idempotent: only write if missing, but always touch `last_seen_at` for sorting."""
    path = _config_path(config_hash)
    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    now = datetime.now(timezone.utc).isoformat()
    record = {
        "config_hash": config_hash,
        "run_identifier": run_identifier or existing.get("run_identifier") or "run",
        "config": config,
        "created_at": existing.get("created_at") or now,
        "last_seen_at": now,
        "last_was_cache_hit": cache_hit,
    }
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def get_run_record(config_hash: str) -> Optional[Dict[str, Any]]:
    path = _config_path(config_hash)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_run_records() -> List[Dict[str, Any]]:
    if not _SERVER_DIR.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for p in _SERVER_DIR.iterdir():
        if not p.is_file() or p.suffix != ".json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append(
            {
                "config_hash": data.get("config_hash") or p.stem,
                "run_identifier": data.get("run_identifier") or "run",
                "created_at": data.get("created_at"),
                "last_seen_at": data.get("last_seen_at") or data.get("created_at"),
                "last_was_cache_hit": bool(data.get("last_was_cache_hit", False)),
            }
        )
    out.sort(key=lambda r: r.get("last_seen_at") or "", reverse=True)
    return out
