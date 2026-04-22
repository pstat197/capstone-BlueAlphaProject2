"""
Streamlit widgets for per-channel on/off toggles and the pause-week schedule.

Covers the YAML surface added by the channel toggle feature:

- `enabled`: bool  *or*  `{default, off_ranges: [{start_week, end_week}, ...]}`
- `adstock_enabled`: bool (per channel)
- `saturation_enabled`: bool (per channel)

Global modeling switches (top-level `adstock.global` / `saturation.global`)
are rendered in ``streamlit_app.py`` and merged in ``ui_config_merge``.

All toggles are fail-open: if the user leaves defaults untouched, nothing is
emitted to YAML (channel stays fully on with adstock + saturation active).

Week ranges are clamped to the current simulation `Week range`; invalid
rows produce warnings on merge and are skipped rather than raising.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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


def global_adstock_key() -> str:
    return "tog_global_adstock"


def global_saturation_key() -> str:
    return "tog_global_saturation"


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


def ensure_global_toggle_state_initialized(cfg: Dict[str, Any]) -> None:
    """Populate the top-level global adstock/saturation switches if missing."""
    if global_adstock_key() not in st.session_state:
        adstock_section = cfg.get("adstock") if isinstance(cfg.get("adstock"), dict) else {}
        v = adstock_section.get("global", True) if adstock_section else True
        st.session_state[global_adstock_key()] = bool(v) if isinstance(v, bool) else True

    if global_saturation_key() not in st.session_state:
        sat_section = cfg.get("saturation") if isinstance(cfg.get("saturation"), dict) else {}
        v = sat_section.get("global", True) if sat_section else True
        st.session_state[global_saturation_key()] = bool(v) if isinstance(v, bool) else True


def clear_toggle_widget_keys() -> None:
    """Remove all per-channel and global toggle widget keys."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(
            ("tog_enabled_", "tog_ads_", "tog_sat_", "tog_offrows_", "tog_offnext_",
             "tog_offstart_", "tog_offend_", "tog_offrm_")
        ):
            del st.session_state[k]
    for k in (global_adstock_key(), global_saturation_key()):
        st.session_state.pop(k, None)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_channel_toggles_block(
    i: int,
    cfg: Dict[str, Any],
    week_range: int,
) -> None:
    """Render the Availability / on-off section inside a channel expander."""
    ensure_channel_toggle_state_initialized(cfg, i, week_range)

    st.markdown("##### Availability")
    st.caption(
        "Controls when this channel contributes to the simulation. **Channel active** "
        "turns media spend and impressions on or off; pause windows zero out spend and "
        "impressions for the selected week range while letting existing adstock decay "
        "flow through into revenue (soft off). Disabling adstock or saturation here "
        "skips that effect for this channel only."
    )

    top_l, top_m, top_r = st.columns([2, 2, 2])
    with top_l:
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
    with top_m:
        st.checkbox(
            "Adstock enabled",
            key=ch_adstock_enabled_key(i),
            help=(
                "Apply this channel's adstock carry-over. Uncheck to skip adstock "
                "for this channel only; saturation still runs if enabled."
            ),
            on_change=_yaml_sync_from_form,
        )
    with top_r:
        st.checkbox(
            "Saturation enabled",
            key=ch_saturation_enabled_key(i),
            help=(
                "Apply this channel's saturation curve. Uncheck to pass raw "
                "impressions straight to ROI scaling for this channel only."
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


def render_global_effect_switches() -> None:
    """Render top-level adstock/saturation global kill-switches."""
    st.markdown("##### Modeling effects (global)")
    st.caption(
        "Top-level kill-switches for the media-effect transforms. When turned off, "
        "the corresponding step is skipped for **every** channel regardless of each "
        "channel's own flag. Leave on for typical MMM runs."
    )
    col_a, col_s = st.columns(2)
    with col_a:
        st.checkbox(
            "Adstock globally enabled",
            key=global_adstock_key(),
            help=(
                "If off, disables adstock carry-over for all channels. "
                "Useful for isolating instantaneous (same-week) media effects."
            ),
            on_change=_yaml_sync_from_form,
        )
    with col_s:
        st.checkbox(
            "Saturation globally enabled",
            key=global_saturation_key(),
            help=(
                "If off, disables the saturation step for all channels so impressions "
                "flow directly into ROI scaling (pre-adstock)."
            ),
            on_change=_yaml_sync_from_form,
        )


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
    channel dict. Keys are omitted (or set to ``True``) when equivalent to the
    fail-open default so the YAML stays clean for simple cases.
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
    Apply per-channel toggle widgets + global effect switches onto ``merged``.

    Mutates ``merged`` in place. Returns a list of user-facing warnings
    (empty if silent or if nothing needed flagging).
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

    ads_on = bool(st.session_state.get(global_adstock_key(), True))
    sat_on = bool(st.session_state.get(global_saturation_key(), True))

    adstock_section = merged.get("adstock")
    if isinstance(adstock_section, dict):
        adstock_section["global"] = ads_on
    elif not ads_on:
        merged["adstock"] = {"global": False}
    else:
        merged.pop("adstock", None)

    saturation_section = merged.get("saturation")
    if isinstance(saturation_section, dict):
        saturation_section["global"] = sat_on
    elif not sat_on:
        merged["saturation"] = {"global": False}
    else:
        merged.pop("saturation", None)

    return [] if silent else warns


def resync_toggle_state_from_config(cfg: Dict[str, Any]) -> None:
    """After a YAML load / reset: drop all toggle widget state so it
    gets rebuilt from the new config on the next render."""
    clear_toggle_widget_keys()
