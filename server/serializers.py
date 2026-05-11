"""Pure serialization helpers: pipeline outputs -> JSON-safe dicts for the React UI."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _series_to_floats(s: pd.Series) -> List[Optional[float]]:
    return [_safe_float(v) for v in s.tolist()]


def channel_names_from_df(df: pd.DataFrame) -> List[str]:
    """Channel names are inferred from `{name}_impressions` columns (excluding `total_impressions`)."""
    names: List[str] = []
    for col in df.columns:
        c = str(col).strip().lstrip("\ufeff")
        if c.endswith("_impressions") and c != "total_impressions":
            names.append(c[: -len("_impressions")])
    return names


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
    return out


def serialize_run_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Shape the simulator dataframe into the structure the React UI expects."""
    df = clean_columns(df)

    weeks: List[int] = (
        df["week"].astype(int).tolist() if "week" in df.columns else list(range(1, len(df) + 1))
    )

    def _col(name: str) -> List[Optional[float]]:
        return _series_to_floats(df[name]) if name in df.columns else []

    totals = {
        "revenue": _col("revenue"),
        "spend": _col("total_spend"),
        "impressions": _col("total_impressions"),
    }

    channels: List[Dict[str, Any]] = []
    for name in channel_names_from_df(df):
        channels.append(
            {
                "name": name,
                "revenue": _col(f"{name}_revenue"),
                "spend": _col(f"{name}_spend"),
                "impressions": _col(f"{name}_impressions"),
            }
        )

    preview_rows: List[Dict[str, Any]] = []
    head = df.head(25)
    columns = [str(c) for c in head.columns]
    for _, row in head.iterrows():
        record: Dict[str, Any] = {}
        for col in columns:
            v = row[col]
            if isinstance(v, (np.floating, float)):
                f = _safe_float(v)
                record[col] = round(f, 3) if f is not None else None
            elif isinstance(v, (np.integer, int)):
                record[col] = int(v)
            elif pd.isna(v):
                record[col] = None
            else:
                record[col] = str(v)
        preview_rows.append(record)

    return {
        "weeks": weeks,
        "totals": totals,
        "channels": channels,
        "preview": {"columns": columns, "rows": preview_rows},
    }


def serialize_correlation(corr: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not corr:
        return None
    static_corr = corr.get("static_corr")
    rolling_corr = corr.get("rolling_corr")
    drift = corr.get("drift")

    def _arr(a: Any) -> Any:
        if a is None:
            return None
        if isinstance(a, np.ndarray):
            return a.tolist()
        return a

    return {
        "channel_names": list(corr.get("channel_names") or []),
        "static_corr": _arr(static_corr),
        "rolling_corr": _arr(rolling_corr),
        "drift": _arr(drift),
        "avg_abs_corr": {k: _safe_float(v) for k, v in (corr.get("avg_abs_corr") or {}).items()},
        "most_correlated_channel": corr.get("most_correlated_channel") or "",
        "pairwise_summary": corr.get("pairwise_summary") or [],
        "window": int(corr.get("window") or 0),
    }
