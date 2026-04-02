"""Disk-backed cache for simulation runs keyed by config hash."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

# Bump when simulation outputs change for the same YAML (invalidate old cache files).
CACHE_VERSION = 2

_CACHE_ROOT = Path(__file__).resolve().parent / ".cache" / "runs"


def _ensure_cache_dir() -> Path:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _CACHE_ROOT


def canonical_config_hash(user_data: Dict[str, Any]) -> str:
    """Stable SHA-256 hash of user config dict + cache version."""
    payload = {"cache_version": CACHE_VERSION, "config": user_data}
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_paths(config_hash: str) -> Tuple[Path, Path]:
    root = _ensure_cache_dir()
    return root / f"{config_hash}.csv", root / f"{config_hash}.json"


def try_load_cached(config_hash: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    csv_path, json_path = cache_paths(config_hash)
    if not csv_path.is_file():
        return None
    meta: Dict[str, Any] = {}
    if json_path.is_file():
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
    try:
        df = pd.read_csv(csv_path)
    except OSError:
        return None
    return df, meta


def save_cache(config_hash: str, df: pd.DataFrame, run_identifier: str) -> None:
    csv_path, json_path = cache_paths(config_hash)
    _ensure_cache_dir()
    df.to_csv(csv_path, index=False)
    meta = {
        "run_identifier": run_identifier,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_version": CACHE_VERSION,
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_with_cache(
    user_data: Dict[str, Any],
    runner: Callable[[Dict[str, Any]], Tuple[pd.DataFrame, str]],
) -> Tuple[pd.DataFrame, str, bool, str]:
    """
    runner: callable taking user_data -> (DataFrame, run_identifier)
    Returns (df, run_identifier, cache_hit, config_hash).
    """
    h = canonical_config_hash(user_data)
    cached = try_load_cached(h)
    if cached is not None:
        df, meta = cached
        rid = meta.get("run_identifier") or ""
        return df, rid, True, h

    df, run_identifier = runner(user_data)
    save_cache(h, df, run_identifier)
    return df, run_identifier, False, h


def clear_run_cache() -> int:
    """Delete all cached run CSV/JSON files. Returns number of files removed."""
    removed = 0
    if not _CACHE_ROOT.is_dir():
        return 0
    for p in _CACHE_ROOT.iterdir():
        if p.is_file():
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed
