"""
Streamlit widgets for per-channel on/off toggles and the pause-week schedule.

Covers the YAML surface added by the channel toggle feature:

- `enabled`: bool  *or*  `{default, off_ranges: [{start_week, end_week}, ...]}`
- `adstock_enabled`: bool (per channel)
- `saturation_enabled`: bool (per channel)

All toggles are fail-open: if the user leaves defaults untouched, nothing is
emitted to YAML (channel stays fully on with adstock + saturation active).

The adstock / saturation enable checkboxes are rendered by
``render_channel_adstock_enable_checkbox`` / ``render_channel_saturation_enable_checkbox``
so they can be co-located with their curve configuration blocks in
``ui_channel_form``. ``render_channel_toggles_block`` renders only the
Availability section (Channel active + pause windows, with inline
start/end validation).

Week ranges are clamped to the current simulation `Week range`; invalid
rows produce warnings inline AND on merge, and are skipped rather than
raising.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import streamlit as st


# ---------------------------------------------------------------------------
# Session key helpers
# ---------------------------------------------------------------------------

def ch_enabled_key(i: int) -> str:
    return f"tog_enabled_{i}"


def ch_adstock_enabled_key(i: int) -> str:
    return f"tog_ads_{i}"


def ch_saturation_enabled_key(i: int) -> str:
    return f"tog_sat_{i}"


def ch_off_rows_key(i: int) -> str:
    """Session key for the list of pause-window rows for channel ``i``."""
    return f"tog_offrows_{i}"


def ch_off_next_id_key(i: int) -> str:
    return f"tog_offnext_{i}"


def ch_off_start_key(i: int, rid: int) -> str:
    return f"tog_offstart_{i}_{rid}"


def ch_off_end_key(i: int, rid: int) -> str:
    return f"tog_offend_{i}_{rid}"


def ch_off_remove_key(i: int, rid: int) -> str:
    return f"tog_offrm_{i}_{rid}"


# ---------------------------------------------------------------------------
# Parsing helpers (YAML <-> UI state)
# ---------------------------------------------------------------------------


def _yaml_sync_from_form() -> None:
    st.session_state["yaml_manual_edit"] = False


def _channel_at(cfg: Dict[str, Any], i: int) -> Dict[str, Any]:
    lst = cfg.get("channel_list") or []
    if i >= len(lst):
        return {}
    item = lst[i]
    if not isinstance(item, dict):
        return {}
    ch = item.get("channel") if isinstance(item.get("channel"), dict) else item
    return ch if isinstance(ch, dict) else {}


def _parse_enabled_from_yaml(
    raw: Any,
) -> Tuple[bool, List[Tuple[int, int]]]:
    """Decode the YAML `enabled:` field into (is_enabled, off_ranges)."""
    if raw is None:
        return True, []
    if isinstance(raw, bool):
        return raw, []
    if isinstance(raw, dict):
        default = raw.get("default", True)
        is_enabled = bool(default) if isinstance(default, bool) else True
        off: List[Tuple[int, int]] = []
        for item in raw.get("off_ranges") or []:
            if not isinstance(item, dict):
                continue
            s = item.get("start_week")
            e = item.get("end_week")
            if isinstance(s, bool) or isinstance(e, bool):
                continue
            try:
                si = int(s)
                ei = int(e)
            except (TypeError, ValueError):
                continue
            off.append((si, ei))
        return is_enabled, off
    return True, []


def _clamp_week(value: int, week_range: int) -> int:
    if week_range < 1:
        return 1
    return max(1, min(week_range, int(value)))


# ---------------------------------------------------------------------------
# State initialization
# ---------------------------------------------------------------------------


def ensure_channel_toggle_state_initialized(
    cfg: Dict[str, Any],
    i: int,
    week_range: int,
) -> None:
    """Populate widget state for channel ``i`` from its YAML config if missing."""
    ch = _channel_at(cfg, i)

    enabled_key = ch_enabled_key(i)
    if enabled_key not in st.session_state:
        is_enabled, off_ranges = _parse_enabled_from_yaml(ch.get("enabled"))
        st.session_state[enabled_key] = is_enabled
        rows: List[Dict[str, int]] = []
        for idx, (s_w, e_w) in enumerate(off_ranges, start=1):
            s_clamped = _clamp_week(s_w, max(week_range, 1))
            e_clamped = _clamp_week(e_w, max(week_range, 1))
            if s_clamped > e_clamped:
                s_clamped, e_clamped = e_clamped, s_clamped
            rows.append({"id": idx, "start": s_clamped, "end": e_clamped})
        st.session_state[ch_off_rows_key(i)] = rows
        st.session_state[ch_off_next_id_key(i)] = len(rows) + 1

    ads_key = ch_adstock_enabled_key(i)
    if ads_key not in st.session_state:
        v = ch.get("adstock_enabled", True)
        st.session_state[ads_key] = bool(v) if isinstance(v, bool) else True

    sat_key = ch_saturation_enabled_key(i)
    if sat_key not in st.session_state:
        v = ch.get("saturation_enabled", True)
        st.session_state[sat_key] = bool(v) if isinstance(v, bool) else True


def clear_toggle_widget_keys() -> None:
    """Remove all per-channel toggle widget keys."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(
            ("tog_enabled_", "tog_ads_", "tog_sat_", "tog_offrows_", "tog_offnext_",
             "tog_offstart_", "tog_offend_", "tog_offrm_")
        ):
            del st.session_state[k]


# ---------------------------------------------------------------------------
# Status summary (for collapsed channel expander title)
# ---------------------------------------------------------------------------


def channel_status_summary(i: int, cfg: Dict[str, Any]) -> str:
    """Build a one-line status summary for the collapsed channel expander.

    Reads from session state when available (so unsaved UI edits reflect),
    falling back to the YAML config. Returns a string like
    ``"active · adstock on · saturation off · 2 pauses"``.
    """
    ch = _channel_at(cfg, i)

    enabled = st.session_state.get(ch_enabled_key(i))
    if enabled is None:
        enabled, _ = _parse_enabled_from_yaml(ch.get("enabled"))
    enabled = bool(enabled)

    rows = st.session_state.get(ch_off_rows_key(i))
    if rows is None:
        _, off_ranges = _parse_enabled_from_yaml(ch.get("enabled"))
        n_pauses = len(off_ranges)
    else:
        n_pauses = len(rows)

    ads = st.session_state.get(ch_adstock_enabled_key(i))
    if ads is None:
        v = ch.get("adstock_enabled", True)
        ads = bool(v) if isinstance(v, bool) else True
    sat = st.session_state.get(ch_saturation_enabled_key(i))
    if sat is None:
        v = ch.get("saturation_enabled", True)
        sat = bool(v) if isinstance(v, bool) else True

    parts: List[str] = []
    parts.append("active" if enabled else "inactive")
    if enabled:
        parts.append("adstock " + ("on" if ads else "off"))
        parts.append("saturation " + ("on" if sat else "off"))
        if n_pauses:
            parts.append(f"{n_pauses} pause{'s' if n_pauses != 1 else ''}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_channel_adstock_enable_checkbox(i: int) -> None:
    """Per-channel adstock enable checkbox, intended to live at the top of the Adstock section."""
    st.checkbox(
        "Apply adstock to this channel",
        key=ch_adstock_enabled_key(i),
        help=(
            "When unchecked, this channel's adstock carry-over step is skipped "
            "(saturation and ROI still run). The curve parameters below are "
            "preserved so you can re-enable without reconfiguring."
        ),
        on_change=_yaml_sync_from_form,
    )


def render_channel_saturation_enable_checkbox(i: int) -> None:
    """Per-channel saturation enable checkbox, intended to live at the top of the Saturation section."""
    st.checkbox(
        "Apply saturation to this channel",
        key=ch_saturation_enabled_key(i),
        help=(
            "When unchecked, impressions pass straight into ROI scaling with no "
            "saturation curve. The curve parameters below are preserved so you "
            "can re-enable without reconfiguring."
        ),
        on_change=_yaml_sync_from_form,
    )


def render_channel_toggles_block(
    i: int,
    cfg: Dict[str, Any],
    week_range: int,
) -> None:
    """Render the Availability section (channel active + pause windows).

    The per-channel adstock / saturation enable checkboxes are rendered
    separately inside their respective sections in ``ui_channel_form`` via
    ``render_channel_adstock_enable_checkbox`` /
    ``render_channel_saturation_enable_checkbox``.
    """
    ensure_channel_toggle_state_initialized(cfg, i, week_range)

    st.markdown("##### Availability")
    st.caption(
        "Controls when this channel contributes. **Channel active** toggles the "
        "whole channel; pause windows zero out spend and impressions for specific "
        "weeks while letting prior adstock decay flow through into revenue (soft off)."
    )

    st.checkbox(
        "Channel active",
        key=ch_enabled_key(i),
        help=(
            "Uncheck to fully disable this channel for the whole run. "
            "A fully disabled channel contributes zero spend, impressions, "
            "and revenue (no baseline, no noise, no adstock echo)."
        ),
        on_change=_yaml_sync_from_form,
    )

    channel_active = bool(st.session_state.get(ch_enabled_key(i), True))

    if not channel_active:
        st.info(
            "Channel is fully disabled — pause windows are ignored while the channel "
            "is inactive."
        )
        return

    st.divider()

    rows: List[Dict[str, int]] = list(st.session_state.get(ch_off_rows_key(i)) or [])
    n_rows = len(rows)
    count_label = (
        f"&nbsp;·&nbsp; *{n_rows} active pause{'s' if n_rows != 1 else ''}*"
        if n_rows
        else "&nbsp;·&nbsp; *no pauses set*"
    )
    st.markdown(f"###### Pause windows{count_label}")
    st.caption(
        "Optional inclusive week ranges during which this channel pauses spend "
        "(spend and impressions go to zero; past adstock still decays into revenue — soft off). "
        f"Week numbers must fall within `1 … Week range` (currently **{int(week_range)}**); "
        "values are clamped automatically."
    )

    if rows:
        h_s, h_e, h_rm = st.columns([2, 2, 1])
        h_s.caption("Start week")
        h_e.caption("End week")
        h_rm.caption(" ")

    for row in list(rows):
        rid = int(row["id"])
        start_key = ch_off_start_key(i, rid)
        end_key = ch_off_end_key(i, rid)

        upper_bound = max(int(week_range), 1)
        # Initialize widget state lazily the first time we see this row.
        if start_key not in st.session_state:
            st.session_state[start_key] = _clamp_week(int(row.get("start", 1)), upper_bound)
        else:
            st.session_state[start_key] = _clamp_week(
                int(st.session_state[start_key]), upper_bound
            )
        if end_key not in st.session_state:
            st.session_state[end_key] = _clamp_week(int(row.get("end", upper_bound)), upper_bound)
        else:
            st.session_state[end_key] = _clamp_week(
                int(st.session_state[end_key]), upper_bound
            )

        c_s, c_e, c_rm = st.columns([2, 2, 1])
        with c_s:
            st.number_input(
                "Start week",
                min_value=1,
                max_value=upper_bound,
                step=1,
                key=start_key,
                label_visibility="collapsed",
                help=f"Inclusive start of the pause (1 … {upper_bound}).",
                on_change=_yaml_sync_from_form,
            )
        with c_e:
            st.number_input(
                "End week",
                min_value=1,
                max_value=upper_bound,
                step=1,
                key=end_key,
                label_visibility="collapsed",
                help=f"Inclusive end of the pause (1 … {upper_bound}). Must be ≥ start week.",
                on_change=_yaml_sync_from_form,
            )
        with c_rm:
            st.write("")
            if st.button("Remove", key=ch_off_remove_key(i, rid), width="stretch"):
                st.session_state[ch_off_rows_key(i)] = [
                    r for r in st.session_state[ch_off_rows_key(i)] if int(r["id"]) != rid
                ]
                for k in (start_key, end_key, ch_off_remove_key(i, rid)):
                    st.session_state.pop(k, None)
                _yaml_sync_from_form()
                st.rerun()

        # Inline validation: start must be <= end. Clamping to range is enforced
        # by the number_input min/max bounds, but start/end ordering is not.
        try:
            s_val = int(st.session_state[start_key])
            e_val = int(st.session_state[end_key])
        except (KeyError, TypeError, ValueError):
            s_val = e_val = None
        if s_val is not None and e_val is not None and s_val > e_val:
            st.warning(
                f"Pause window #{rid}: start week ({s_val}) is after end week "
                f"({e_val}). Values will be swapped when the simulation runs."
            )

    # Inline duplicate detection — soft warning only.
    finalized: List[Tuple[int, int]] = []
    for row in rows:
        rid = int(row["id"])
        try:
            s = int(st.session_state.get(ch_off_start_key(i, rid), row.get("start", 1)))
            e = int(st.session_state.get(ch_off_end_key(i, rid), row.get("end", 1)))
        except (TypeError, ValueError):
            continue
        if s > e:
            s, e = e, s
        finalized.append((s, e))
    if len(finalized) != len(set(finalized)):
        st.caption("ℹ️ Duplicate pause windows will be deduplicated on run.")

    add_disabled = int(week_range) < 1
    if st.button(
        "Add pause window",
        key=f"tog_offadd_{i}",
        width="content",
        disabled=add_disabled,
    ):
        next_id = int(st.session_state.get(ch_off_next_id_key(i), 1))
        st.session_state[ch_off_next_id_key(i)] = next_id + 1
        new_row = {"id": next_id, "start": 1, "end": max(int(week_range), 1)}
        st.session_state[ch_off_rows_key(i)] = list(rows) + [new_row]
        _yaml_sync_from_form()
        st.rerun()


# ---------------------------------------------------------------------------
# Merge helpers (UI -> YAML)
# ---------------------------------------------------------------------------


def _collect_channel_toggle(
    i: int,
    week_range: int,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Build the YAML overrides (as a patch dict) for channel ``i``'s toggles.

    Returns (patch, warnings). The patch is applied directly onto the merged
    channel dict.
    """
    warns: List[str] = []
    patch: Dict[str, Any] = {}

    enabled = bool(st.session_state.get(ch_enabled_key(i), True))
    adstock_enabled = bool(st.session_state.get(ch_adstock_enabled_key(i), True))
    saturation_enabled = bool(st.session_state.get(ch_saturation_enabled_key(i), True))

    rows: List[Dict[str, int]] = list(st.session_state.get(ch_off_rows_key(i)) or [])
    upper = max(int(week_range), 1)

    parsed_ranges: List[Tuple[int, int]] = []
    for row in rows:
        rid = int(row["id"])
        start_raw = st.session_state.get(ch_off_start_key(i, rid), row.get("start", 1))
        end_raw = st.session_state.get(ch_off_end_key(i, rid), row.get("end", upper))
        try:
            s = int(start_raw)
            e = int(end_raw)
        except (TypeError, ValueError):
            warns.append(
                f"Channel {i + 1} pause window #{rid}: start/end must be integers; skipped."
            )
            continue
        s_clamped = _clamp_week(s, upper)
        e_clamped = _clamp_week(e, upper)
        if s != s_clamped or e != e_clamped:
            warns.append(
                f"Channel {i + 1} pause window #{rid}: "
                f"clamped to [{s_clamped}, {e_clamped}] within 1..{upper}."
            )
        if s_clamped > e_clamped:
            warns.append(
                f"Channel {i + 1} pause window #{rid}: "
                f"start ({s_clamped}) > end ({e_clamped}); swapped."
            )
            s_clamped, e_clamped = e_clamped, s_clamped
        parsed_ranges.append((s_clamped, e_clamped))

    parsed_ranges = sorted(set(parsed_ranges))

    if enabled and not parsed_ranges:
        patch["enabled"] = True
    elif not enabled:
        patch["enabled"] = False
    else:
        patch["enabled"] = {
            "default": True,
            "off_ranges": [
                {"start_week": s, "end_week": e} for s, e in parsed_ranges
            ],
        }

    patch["adstock_enabled"] = adstock_enabled
    patch["saturation_enabled"] = saturation_enabled

    return patch, warns


def merge_channel_toggles_into_config(
    merged: Dict[str, Any],
    *,
    silent: bool = False,
) -> List[str]:
    """
    Apply per-channel toggle widgets onto ``merged``.

    Mutates ``merged`` in place. Returns a list of user-facing warnings
    (empty if silent or if nothing needed flagging).

    Top-level ``adstock.global`` / ``saturation.global`` flags from the
    loaded YAML are preserved unchanged — this function never reads or
    writes them. The per-channel ``adstock_enabled`` / ``saturation_enabled``
    flags are the single source of UI truth.
    """
    warns: List[str] = []
    week_range = int(merged.get("week_range") or 0)
    channel_list = merged.get("channel_list") or []

    for i, item in enumerate(channel_list):
        if not isinstance(item, dict):
            continue
        ch = item.get("channel") if isinstance(item.get("channel"), dict) else item
        if not isinstance(ch, dict):
            continue
        patch, per_warns = _collect_channel_toggle(i, week_range)
        if not silent:
            warns.extend(per_warns)
        ch.update(patch)

    return [] if silent else warns


def resync_toggle_state_from_config(cfg: Dict[str, Any]) -> None:
    """After a YAML load / reset: drop all toggle widget state so it
    gets rebuilt from the new config on the next render."""
    clear_toggle_widget_keys()
