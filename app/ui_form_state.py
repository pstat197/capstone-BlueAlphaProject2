"""Widget session keys, parsing helpers, and curve-type resolution for the config form."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from app.ui_helpers import get_at, path_string_to_parts


def pc_field_key(i: int, path_suffix: str, list_index: Optional[int]) -> str:
    li = list_index if list_index is not None else "x"
    return f"pc_{i}_{path_suffix.replace('.', '_')}_{li}"


def select_session_key(i: int, path_suffix: str) -> str:
    return f"sel_{i}_{path_suffix.replace('.', '_')}"


def adstock_weights_key(i: int) -> str:
    return f"adw_{i}"


def effective_curve_type(
    i: int,
    path_suffix: str,
    data: Dict[str, Any],
    options: List[str],
) -> str:
    """Widget selection if set, else YAML value, else first option."""
    key = select_session_key(i, path_suffix)
    if key in st.session_state:
        v = st.session_state[key]
        if v in options:
            return str(v)
    parts = path_string_to_parts(f"channel_list.{i}.{path_suffix}")
    cur = get_at(data, parts)
    if cur in options:
        return str(cur)
    return str(options[0]) if options else "linear"


def saturation_slider_visible(item: Dict[str, Any], sat_type: str) -> bool:
    types = item.get("saturation_types")
    if types is None:
        return True
    return sat_type in types


def adstock_slider_visible(item: Dict[str, Any], ad_type: str) -> bool:
    types = item.get("adstock_types")
    if types is None:
        return True
    return ad_type in types


def parse_optional_num(raw: Any, *, as_int: bool = False) -> Tuple[Optional[float], bool]:
    """Returns (value, ok). If empty string, returns (None, True) meaning skip override."""
    if raw is None:
        return None, True
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return (float(raw), True) if not as_int else (float(int(raw)), True)
    s = str(raw).strip()
    if s == "":
        return None, True
    try:
        if as_int:
            return float(int(s, 10)), True
        return float(s), True
    except ValueError:
        return None, False


def parse_weights_csv(raw: Any) -> Tuple[Optional[List[float]], bool]:
    """Returns (weights or None if empty, ok). None + True means skip override."""
    if raw is None:
        return None, True
    s = str(raw).strip()
    if s == "":
        return None, True
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    if not parts:
        return None, True
    try:
        return [float(p) for p in parts], True
    except ValueError:
        return None, False
