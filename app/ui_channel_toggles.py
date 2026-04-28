"""
Streamlit widgets for per-channel on/off toggles and the pause-week schedule.

Covers the YAML surface added by the channel toggle feature:

- `enabled`: bool  *or*  `{default, off_ranges: [{start_week, end_week}, ...]}`
- `sticky_pause_ranges`: optional list of `{start_week, end_week, start_probability,
  continue_probability}` (Markov / sticky random pauses; spend/impressions only).
- `adstock_enabled`: bool (per channel)
- `saturation_enabled`: bool (per channel)

All toggles are fail-open: if the user leaves defaults untouched, nothing is
emitted to YAML (channel stays fully on with adstock + saturation active).

The adstock / saturation enable checkboxes are rendered by
``render_channel_adstock_enable_checkbox`` / ``render_channel_saturation_enable_checkbox``
so they can be co-located with their curve configuration blocks in
``ui_channel_form``. ``render_channel_toggles_block`` renders the Availability
section (channel active + unified pause rules: each row is fixed or sticky).

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


def ch_pause_rows_key(i: int) -> str:
    """Session key for ordered pause-rule rows (fixed or sticky) for channel ``i``."""
    return f"tog_pauserows_{i}"


def ch_pause_next_id_key(i: int) -> str:
    return f"tog_pausenext_{i}"


def ch_pause_kind_key(i: int, rid: int) -> str:
    return f"tog_pkind_{i}_{rid}"


def ch_pause_start_key(i: int, rid: int) -> str:
    return f"tog_pstart_{i}_{rid}"


def ch_pause_end_key(i: int, rid: int) -> str:
    return f"tog_pend_{i}_{rid}"


def ch_pause_pstart_key(i: int, rid: int) -> str:
    return f"tog_pp0_{i}_{rid}"


def ch_pause_pcont_key(i: int, rid: int) -> str:
    return f"tog_pp1_{i}_{rid}"


def ch_pause_remove_key(i: int, rid: int) -> str:
    return f"tog_parm_{i}_{rid}"


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


def _parse_sticky_pause_ranges_from_yaml(raw: Any) -> List[Dict[str, Any]]:
    """Decode ``sticky_pause_ranges`` into UI row dicts with float probabilities."""
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        s_w = item.get("start_week")
        e_w = item.get("end_week")
        p0 = item.get("start_probability")
        p1 = item.get("continue_probability")
        if isinstance(s_w, bool) or isinstance(e_w, bool):
            continue
        try:
            si = int(s_w)
            ei = int(e_w)
        except (TypeError, ValueError):
            continue
        try:
            f0 = float(p0)
            f1 = float(p1)
        except (TypeError, ValueError):
            continue
        f0 = max(0.0, min(1.0, f0))
        f1 = max(0.0, min(1.0, f1))
        out.append({"start": si, "end": ei, "p_start": f0, "p_continue": f1})
    return out


def _pause_rows_from_channel_yaml(ch: Dict[str, Any], week_range: int) -> Tuple[List[Dict[str, Any]], int]:
    """Build unified pause row list from YAML: fixed ranges first, then sticky (stable order)."""
    upper = max(int(week_range), 1)
    rows: List[Dict[str, Any]] = []
    nid = 1
    is_enabled, off_ranges = _parse_enabled_from_yaml(ch.get("enabled"))
    if is_enabled:
        for s_w, e_w in off_ranges:
            s_clamped = _clamp_week(s_w, upper)
            e_clamped = _clamp_week(e_w, upper)
            if s_clamped > e_clamped:
                s_clamped, e_clamped = e_clamped, s_clamped
            rows.append({
                "id": nid,
                "kind": "fixed",
                "start": s_clamped,
                "end": e_clamped,
                "p_start": 0.2,
                "p_continue": 0.85,
            })
            nid += 1
    for entry in _parse_sticky_pause_ranges_from_yaml(ch.get("sticky_pause_ranges")):
        s_clamped = _clamp_week(int(entry["start"]), upper)
        e_clamped = _clamp_week(int(entry["end"]), upper)
        if s_clamped > e_clamped:
            s_clamped, e_clamped = e_clamped, s_clamped
        rows.append({
            "id": nid,
            "kind": "sticky",
            "start": s_clamped,
            "end": e_clamped,
            "p_start": float(entry["p_start"]),
            "p_continue": float(entry["p_continue"]),
        })
        nid += 1
    return rows, nid


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

    pause_key = ch_pause_rows_key(i)
    if pause_key not in st.session_state:
        pause_rows, next_id = _pause_rows_from_channel_yaml(ch, week_range)
        st.session_state[pause_key] = pause_rows
        st.session_state[ch_pause_next_id_key(i)] = next_id

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
            (
                "tog_enabled_",
                "tog_ads_",
                "tog_sat_",
                "tog_pauserows_",
                "tog_pausenext_",
                "tog_pkind_",
                "tog_pstart_",
                "tog_pend_",
                "tog_pp0_",
                "tog_pp1_",
                "tog_parm_",
                # legacy keys (older sessions)
                "tog_offrows_",
                "tog_offnext_",
                "tog_offstart_",
                "tog_offend_",
                "tog_offrm_",
                "tog_stickyrows_",
                "tog_stickynext_",
                "tog_stickystart_",
                "tog_stickyend_",
                "tog_stickyp0_",
                "tog_stickyp1_",
                "tog_stickyrm_",
            )
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
    wr = max(int(cfg.get("week_range") or 0), 1)

    enabled = st.session_state.get(ch_enabled_key(i))
    if enabled is None:
        enabled, _ = _parse_enabled_from_yaml(ch.get("enabled"))
    enabled = bool(enabled)

    pause_rows = st.session_state.get(ch_pause_rows_key(i))
    if pause_rows is None:
        pr, _ = _pause_rows_from_channel_yaml(ch, wr)
        n_fixed = sum(1 for r in pr if r.get("kind") == "fixed")
        n_sticky = sum(1 for r in pr if r.get("kind") == "sticky")
    else:
        n_fixed = sum(1 for r in pause_rows if r.get("kind") == "fixed")
        n_sticky = sum(1 for r in pause_rows if r.get("kind") == "sticky")

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
        n_rules = n_fixed + n_sticky
        if n_rules:
            bits: List[str] = []
            if n_fixed:
                bits.append(f"{n_fixed} fixed")
            if n_sticky:
                bits.append(f"{n_sticky} sticky")
            parts.append(", ".join(bits) + " pause" + ("s" if n_rules != 1 else ""))
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
        # Keep only the reactivation control visible while inactive.
        return

    st.caption(
        "Controls when this channel contributes. **Pause rules** can fix spend/impressions "
        "to zero for whole ranges or add sticky random pauses; adstock echo can still flow (soft off)."
    )

    st.divider()

    pause_rows: List[Dict[str, Any]] = list(st.session_state.get(ch_pause_rows_key(i)) or [])
    n_rules = len(pause_rows)
    count_label = (
        f"&nbsp;·&nbsp; *{n_rules} rule{'s' if n_rules != 1 else ''}*"
        if n_rules
        else "&nbsp;·&nbsp; *no rules*"
    )
    st.markdown(f"###### Pause rules{count_label}")
    st.caption(
        "Each row is one inclusive week range. Choose **Fixed** to be off every week in "
        "that range, or **Sticky** for Markov random pauses (**Start P** / **Continue P**). "
        "Fixed rules are applied before sticky; during a fixed off-week the sticky chain "
        "does not advance. Weeks are clamped to `1 … Week range` "
        f"(**{int(week_range)}**). Same run seed ⇒ same sticky pattern."
    )

    upper_bound = max(int(week_range), 1)
    if pause_rows:
        h0, h1, h2, h3, h4, h5 = st.columns([2.1, 1.0, 1.0, 1.35, 1.35, 0.65])
        h0.caption("Type")
        h1.caption("Start")
        h2.caption("End")
        h3.caption("Start P")
        h4.caption("Continue P")
        h5.caption(" ")

    def _kind_label(k: str) -> str:
        if k == "sticky":
            return "Sticky (random)"
        return "Fixed (always off)"

    for row in list(pause_rows):
        rid = int(row["id"])
        kind_key = ch_pause_kind_key(i, rid)
        start_key = ch_pause_start_key(i, rid)
        end_key = ch_pause_end_key(i, rid)
        p0_key = ch_pause_pstart_key(i, rid)
        p1_key = ch_pause_pcont_key(i, rid)

        if kind_key not in st.session_state:
            k0 = str(row.get("kind", "fixed"))
            st.session_state[kind_key] = k0 if k0 in ("fixed", "sticky") else "fixed"
        if start_key not in st.session_state:
            st.session_state[start_key] = _clamp_week(int(row.get("start", 1)), upper_bound)
        else:
            st.session_state[start_key] = _clamp_week(int(st.session_state[start_key]), upper_bound)
        if end_key not in st.session_state:
            st.session_state[end_key] = _clamp_week(int(row.get("end", upper_bound)), upper_bound)
        else:
            st.session_state[end_key] = _clamp_week(int(st.session_state[end_key]), upper_bound)
        if p0_key not in st.session_state:
            st.session_state[p0_key] = float(row.get("p_start", 0.2))
        if p1_key not in st.session_state:
            st.session_state[p1_key] = float(row.get("p_continue", 0.85))

        c0, c1, c2, c3, c4, c5 = st.columns([2.1, 1.0, 1.0, 1.35, 1.35, 0.65])
        with c0:
            st.selectbox(
                "Rule type",
                options=["fixed", "sticky"],
                format_func=_kind_label,
                key=kind_key,
                label_visibility="collapsed",
                help="Fixed: zero spend every week in range. Sticky: random Markov pauses inside the range.",
                on_change=_yaml_sync_from_form,
            )
        kind_val = str(st.session_state.get(kind_key, "fixed"))
        if kind_val not in ("fixed", "sticky"):
            kind_val = "fixed"
            st.session_state[kind_key] = kind_val

        with c1:
            st.number_input(
                "Start week",
                min_value=1,
                max_value=upper_bound,
                step=1,
                key=start_key,
                label_visibility="collapsed",
                help=f"Inclusive start (1 … {upper_bound}).",
                on_change=_yaml_sync_from_form,
            )
        with c2:
            st.number_input(
                "End week",
                min_value=1,
                max_value=upper_bound,
                step=1,
                key=end_key,
                label_visibility="collapsed",
                help=f"Inclusive end (1 … {upper_bound}).",
                on_change=_yaml_sync_from_form,
            )
        with c3:
            if kind_val == "sticky":
                st.slider(
                    "Start P",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    key=p0_key,
                    label_visibility="collapsed",
                    help="P(pause | not sticky-paused last week).",
                    on_change=_yaml_sync_from_form,
                )
            else:
                st.caption("—")
        with c4:
            if kind_val == "sticky":
                st.slider(
                    "Continue P",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    key=p1_key,
                    label_visibility="collapsed",
                    help="P(pause | sticky-paused last week).",
                    on_change=_yaml_sync_from_form,
                )
            else:
                st.caption("—")
        with c5:
            st.write("")
            if st.button("✕", key=ch_pause_remove_key(i, rid), help="Remove this rule"):
                st.session_state[ch_pause_rows_key(i)] = [
                    r for r in st.session_state[ch_pause_rows_key(i)] if int(r["id"]) != rid
                ]
                for k in (kind_key, start_key, end_key, p0_key, p1_key, ch_pause_remove_key(i, rid)):
                    st.session_state.pop(k, None)
                _yaml_sync_from_form()
                st.rerun()

        try:
            s_val = int(st.session_state[start_key])
            e_val = int(st.session_state[end_key])
        except (KeyError, TypeError, ValueError):
            s_val = e_val = None
        if s_val is not None and e_val is not None and s_val > e_val:
            st.warning(
                f"Rule #{rid} ({kind_val}): start week ({s_val}) is after end week "
                f"({e_val}); values will be swapped when the simulation runs."
            )

    finalized_fixed: List[Tuple[int, int]] = []
    for row in pause_rows:
        rid = int(row["id"])
        kv = str(st.session_state.get(ch_pause_kind_key(i, rid), "fixed"))
        if kv != "fixed":
            continue
        try:
            s = int(st.session_state.get(ch_pause_start_key(i, rid), row.get("start", 1)))
            e = int(st.session_state.get(ch_pause_end_key(i, rid), row.get("end", 1)))
        except (TypeError, ValueError):
            continue
        if s > e:
            s, e = e, s
        finalized_fixed.append((s, e))
    if len(finalized_fixed) != len(set(finalized_fixed)):
        st.caption("ℹ️ Duplicate **fixed** ranges will be deduplicated on run.")

    add_disabled = int(week_range) < 1
    if st.button(
        "Add pause rule",
        key=f"tog_pauseadd_{i}",
        width="content",
        disabled=add_disabled,
    ):
        next_id = int(st.session_state.get(ch_pause_next_id_key(i), 1))
        st.session_state[ch_pause_next_id_key(i)] = next_id + 1
        new_row: Dict[str, Any] = {
            "id": next_id,
            "kind": "fixed",
            "start": 1,
            "end": upper_bound,
            "p_start": 0.2,
            "p_continue": 0.85,
        }
        st.session_state[ch_pause_rows_key(i)] = list(pause_rows) + [new_row]
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

    pause_rows: List[Dict[str, Any]] = list(st.session_state.get(ch_pause_rows_key(i)) or [])
    upper = max(int(week_range), 1)

    parsed_ranges: List[Tuple[int, int]] = []
    sticky_yaml: List[Dict[str, Any]] = []

    for row in pause_rows:
        rid = int(row["id"])
        kind = str(st.session_state.get(ch_pause_kind_key(i, rid), row.get("kind", "fixed")))
        if kind not in ("fixed", "sticky"):
            kind = "fixed"
        start_raw = st.session_state.get(ch_pause_start_key(i, rid), row.get("start", 1))
        end_raw = st.session_state.get(ch_pause_end_key(i, rid), row.get("end", upper))
        try:
            s = int(start_raw)
            e = int(end_raw)
        except (TypeError, ValueError):
            warns.append(
                f"Channel {i + 1} pause rule #{rid}: start/end must be integers; skipped."
            )
            continue
        s_clamped = _clamp_week(s, upper)
        e_clamped = _clamp_week(e, upper)
        if s != s_clamped or e != e_clamped:
            warns.append(
                f"Channel {i + 1} pause rule #{rid}: "
                f"clamped to [{s_clamped}, {e_clamped}] within 1..{upper}."
            )
        if s_clamped > e_clamped:
            warns.append(
                f"Channel {i + 1} pause rule #{rid}: "
                f"start ({s_clamped}) > end ({e_clamped}); swapped."
            )
            s_clamped, e_clamped = e_clamped, s_clamped

        if kind == "fixed":
            parsed_ranges.append((s_clamped, e_clamped))
        else:
            sr_p0 = st.session_state.get(ch_pause_pstart_key(i, rid), row.get("p_start", 0.2))
            sr_p1 = st.session_state.get(ch_pause_pcont_key(i, rid), row.get("p_continue", 0.85))
            try:
                p0 = float(sr_p0)
                p1 = float(sr_p1)
            except (TypeError, ValueError):
                warns.append(
                    f"Channel {i + 1} pause rule #{rid} (sticky): probabilities must be numbers; skipped."
                )
                continue
            p0 = max(0.0, min(1.0, p0))
            p1 = max(0.0, min(1.0, p1))
            sticky_yaml.append(
                {
                    "start_week": s_clamped,
                    "end_week": e_clamped,
                    "start_probability": p0,
                    "continue_probability": p1,
                }
            )

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
    patch["sticky_pause_ranges"] = sticky_yaml

    return patch, warns


def merge_channel_toggles_into_config(merged: Dict[str, Any]) -> List[str]:
    """
    Apply per-channel toggle widgets onto ``merged``.

    Mutates ``merged`` in place. Returns a list of user-facing warnings.

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
        warns.extend(per_warns)
        ch.update(patch)

    return warns


def resync_toggle_state_from_config(cfg: Dict[str, Any]) -> None:
    """After a YAML load / reset: drop all toggle widget state so it
    gets rebuilt from the new config on the next render."""
    clear_toggle_widget_keys()
