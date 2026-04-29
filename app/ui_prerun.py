"""Pre-run summary, validation, and cache fingerprint helpers for the Streamlit app."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from app.ui_channel_toggles import (
    ch_pause_end_key,
    ch_pause_kind_key,
    ch_pause_rows_key,
    ch_pause_start_key,
)


def existing_channel_names(cfg: Dict[str, Any]) -> List[str]:
    """Non-blank effective channel names (live rename widget state wins)."""
    out: List[str] = []
    for i, item in enumerate(cfg.get("channel_list") or []):
        renamed = st.session_state.get(f"ch_name_{i}")
        if isinstance(renamed, str) and renamed.strip():
            out.append(renamed.strip())
            continue
        ch = item.get("channel") if isinstance(item, dict) else None
        if not isinstance(ch, dict):
            ch = item if isinstance(item, dict) else {}
        nm = ch.get("channel_name")
        if isinstance(nm, str) and nm.strip():
            out.append(nm.strip())
    return out


def next_unique_channel_name(base: str, existing: List[str]) -> str:
    """Return ``base`` if unused, else ``base 2``, ``base 3``, … (case-insensitive)."""
    taken = {n.lower() for n in existing}
    if base.lower() not in taken:
        return base
    i = 2
    while f"{base} {i}".lower() in taken:
        i += 1
    return f"{base} {i}"


def _channel_dict_at(merged: Dict[str, Any], i: int) -> Dict[str, Any]:
    lst = merged.get("channel_list") or []
    if i >= len(lst):
        return {}
    item = lst[i]
    if not isinstance(item, dict):
        return {}
    ch = item.get("channel")
    return ch if isinstance(ch, dict) else {}


def _channel_fully_active(ch: Dict[str, Any]) -> bool:
    """False when the channel is turned off for the whole run."""
    en = ch.get("enabled")
    if en is False:
        return False
    if isinstance(en, dict):
        d = en.get("default", True)
        if isinstance(d, bool) and not d:
            return False
    return True


def _pause_rule_count(ch: Dict[str, Any]) -> int:
    n_fixed = 0
    en = ch.get("enabled")
    if isinstance(en, dict):
        for item in en.get("off_ranges") or []:
            if isinstance(item, dict):
                n_fixed += 1
    sticky = ch.get("sticky_pause_ranges")
    n_sticky = len(sticky) if isinstance(sticky, list) else 0
    return n_fixed + n_sticky


def build_run_summary_table(merged: Dict[str, Any]) -> pd.DataFrame:
    """Compact table of per-channel settings as they will be merged for the next run."""
    rows: List[Dict[str, Any]] = []
    ch_list = merged.get("channel_list") or []
    for i, _item in enumerate(ch_list):
        ch = _channel_dict_at(merged, i)
        name = str(ch.get("channel_name") or f"Channel {i + 1}").strip() or f"Channel {i + 1}"
        active = "Yes" if _channel_fully_active(ch) else "No"
        ads = ch.get("adstock_enabled", True)
        sat = ch.get("saturation_enabled", True)
        ads_s = "On" if (isinstance(ads, bool) and ads) or ads is None else "Off"
        sat_s = "On" if (isinstance(sat, bool) and sat) or sat is None else "Off"
        roi = ch.get("true_roi")
        roi_s = f"{float(roi):g}" if isinstance(roi, (int, float)) and not isinstance(roi, bool) else "—"
        bl = ch.get("baseline_revenue")
        bl_s = f"{float(bl):,.0f}" if isinstance(bl, (int, float)) and not isinstance(bl, bool) else "—"
        rows.append(
            {
                "Channel": name,
                "Active": active,
                "Adstock": ads_s,
                "Saturation": sat_s,
                "Pause rules": _pause_rule_count(ch),
                "ROI": roi_s,
                "Baseline": bl_s,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["Channel", "Active", "Adstock", "Saturation", "Pause rules", "ROI", "Baseline"]
        )
    return pd.DataFrame(rows)


def _merge_warning_blocks_run(w: str) -> bool:
    low = w.lower()
    if "invalid number" in low or "invalid list" in low:
        return True
    if "could not normalize" in low:
        return True
    if "skipped" in low:
        return True
    return False


def _pause_widget_start_after_end(i: int) -> List[str]:
    """Invalid pause rows in session (start week strictly after end week)."""
    issues: List[str] = []
    rows: List[Dict[str, Any]] = list(st.session_state.get(ch_pause_rows_key(i)) or [])
    for row in rows:
        rid = int(row["id"])
        try:
            s = int(st.session_state.get(ch_pause_start_key(i, rid), row.get("start", 1)))
            e = int(st.session_state.get(ch_pause_end_key(i, rid), row.get("end", 1)))
        except (TypeError, ValueError):
            continue
        if s > e:
            kind = str(st.session_state.get(ch_pause_kind_key(i, rid), row.get("kind", "fixed")))
            issues.append(
                f"Channel {i + 1}, pause rule #{rid} ({kind}): start week ({s}) is after end week ({e})."
            )
    return issues


def prerun_blocking_issues(merged: Dict[str, Any], merge_warns: List[str]) -> List[str]:
    """Reasons the Run button should stay disabled (empty list => OK to run)."""
    issues: List[str] = []

    wr = int(merged.get("week_range") or 0)
    if wr < 1:
        issues.append("Week range must be at least 1.")

    ch_list = merged.get("channel_list") or []
    if not ch_list:
        issues.append("Add at least one channel before running.")
        return issues

    for w in merge_warns:
        if _merge_warning_blocks_run(w):
            issues.append(w)

    n = len(ch_list)
    seen_names: set[str] = set()
    for i in range(n):
        nm = str(st.session_state.get(f"ch_name_{i}", "")).strip()
        if not nm:
            issues.append(f"Channel {i + 1}: enter a channel name (or remove the row).")
        else:
            lowered = nm.casefold()
            if lowered in seen_names:
                issues.append(f"Channel names must be unique (duplicate: {nm!r}).")
            seen_names.add(lowered)

        ch = _channel_dict_at(merged, i)
        roi = ch.get("true_roi")
        if isinstance(roi, (int, float)) and not isinstance(roi, bool) and float(roi) < 0:
            label = nm or f"Channel {i + 1}"
            issues.append(f"{label}: ROI cannot be negative.")

        issues.extend(_pause_widget_start_after_end(i))

    return list(dict.fromkeys(issues))


def predict_cache_fingerprint(merged: Dict[str, Any]) -> Tuple[str, bool]:
    """Return (full config hash hex, True if a cache file exists for that hash)."""
    from app.cache import cache_entry_exists, canonical_config_hash

    h = canonical_config_hash(merged)
    hit = cache_entry_exists(h)
    return h, hit


def informational_merge_warns(merge_warns: List[str]) -> List[str]:
    """Merge messages that do not block Run (shown in a collapsible notes section)."""
    return [w for w in merge_warns if not _merge_warning_blocks_run(w)]
