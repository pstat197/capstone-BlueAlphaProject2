"""Streamlit UI for optional YAML ``budget_shifts`` (post-draw spend rules)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import streamlit as st

from app.ui_correlations import effective_channel_names
from scripts.spend_simulation.budget_shift_auto import generate_auto_budget_shift_rules
from scripts.synth_input_classes.input_configurations import _normalize_budget_shifts


def _yaml_sync_from_form() -> None:
    st.session_state["yaml_manual_edit"] = False


def _cfg_week_range(cfg: Dict[str, Any]) -> int:
    return max(1, int(cfg.get("week_range") or 52))


def _rows_from_cfg_budget_shifts(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    raw = cfg.get("budget_shifts")
    wr = _cfg_week_range(cfg)
    if raw is None:
        return [], 1
    if not isinstance(raw, list):
        return [], 1
    rows: List[Dict[str, Any]] = []
    next_id = 1
    for e in raw:
        if not isinstance(e, dict):
            continue
        t = str(e.get("type", "")).strip().lower()
        if t == "scale":
            sw = int(e.get("start_week", 1))
            ew = int(e.get("end_week", sw))
            rows.append(
                {
                    "id": next_id,
                    "kind": "scale",
                    "start_week": sw,
                    "end_week": ew,
                    "factor": float(e.get("factor", 1.0)),
                }
            )
            next_id += 1
        elif t == "scale_channel":
            sw = int(e.get("start_week", 1))
            ew = int(e.get("end_week", sw))
            rows.append(
                {
                    "id": next_id,
                    "kind": "scale_channel",
                    "channel_name": str(e.get("channel_name", "")),
                    "start_week": sw,
                    "end_week": ew,
                    "factor": float(e.get("factor", 1.0)),
                }
            )
            next_id += 1
        elif t == "reallocate":
            sw = int(e.get("start_week", 1))
            if "end_week" in e and e.get("end_week") is not None:
                ew = int(e["end_week"])
            else:
                ew = wr
            rows.append(
                {
                    "id": next_id,
                    "kind": "reallocate",
                    "start_week": sw,
                    "end_week": ew,
                    "from_channel": str(e.get("from_channel", "")),
                    "to_channel": str(e.get("to_channel", "")),
                    "fraction": float(e.get("fraction", 0.0)),
                }
            )
            next_id += 1
    return rows, next_id


def ensure_budget_shift_rows_initialized(cfg: Dict[str, Any]) -> None:
    if "budget_shift_ui_rows" in st.session_state:
        return
    rows, next_id = _rows_from_cfg_budget_shifts(cfg)
    st.session_state.budget_shift_ui_rows = rows
    st.session_state.budget_shift_next_id = next_id


def clear_budget_shift_keys() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith("bs_") or k in (
            "budget_shift_ui_rows",
            "budget_shift_next_id",
            "budget_shift_auto_sig",
            "budget_shift_auto_rules",
        ):
            del st.session_state[k]


def apply_budget_shifts_to_session_rows(cfg: Dict[str, Any]) -> None:
    """After YAML load or reset: replace budget-shift UI rows from config."""
    clear_budget_shift_keys()
    rows, next_id = _rows_from_cfg_budget_shifts(cfg)
    st.session_state.budget_shift_ui_rows = rows
    st.session_state.budget_shift_next_id = next_id


def merge_budget_shifts_from_widgets(
    merged: Dict[str, Any],
    *,
    silent: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build ``budget_shifts`` YAML list from session widgets, plus optional auto rules."""
    warns: List[str] = []
    n = len(merged.get("channel_list") or [])
    names_set = set(effective_channel_names(merged, n))
    week_cap = max(1, int(st.session_state.get("week_range_num", merged.get("week_range") or 52)))
    rows: List[Dict[str, Any]] = list(st.session_state.get("budget_shift_ui_rows") or [])
    raw_out: List[Dict[str, Any]] = []

    for row in rows:
        rid = int(row["id"])
        kind = str(st.session_state.get(f"bs_kind_{rid}", row.get("kind", "scale"))).strip().lower()
        if kind not in ("scale", "scale_channel", "reallocate"):
            kind = "scale"

        if kind == "scale":
            sw = int(st.session_state.get(f"bs_sw_{rid}", row.get("start_week", 1)))
            ew = int(st.session_state.get(f"bs_ew_{rid}", row.get("end_week", sw)))
            factor = float(st.session_state.get(f"bs_factor_{rid}", row.get("factor", 1.0)))
            if sw < 1 or ew < 1:
                if not silent:
                    warns.append(f"Budget shift row {rid} (scale): weeks must be ≥ 1; skipped.")
                continue
            if ew < sw:
                if not silent:
                    warns.append(
                        f"Budget shift row {rid} (scale): end_week ({ew}) < start_week ({sw}); skipped."
                    )
                continue
            if sw > week_cap or ew > week_cap:
                if not silent:
                    warns.append(
                        f"Budget shift row {rid} (scale): weeks exceed current week range ({week_cap}); "
                        "rule kept — extend **Week range** if you need longer horizons."
                    )
            raw_out.append({"type": "scale", "start_week": sw, "end_week": ew, "factor": factor})

        elif kind == "scale_channel":
            sw = int(st.session_state.get(f"bs_ch_sw_{rid}", row.get("start_week", 1)))
            ew = int(st.session_state.get(f"bs_ch_ew_{rid}", row.get("end_week", sw)))
            factor = float(st.session_state.get(f"bs_ch_factor_{rid}", row.get("factor", 1.0)))
            cname = str(st.session_state.get(f"bs_ch_name_{rid}", row.get("channel_name", ""))).strip()
            if sw < 1 or ew < 1:
                if not silent:
                    warns.append(f"Budget shift row {rid} (scale one channel): weeks must be ≥ 1; skipped.")
                continue
            if ew < sw:
                if not silent:
                    warns.append(f"Budget shift row {rid} (scale one channel): end < start; skipped.")
                continue
            if not cname or cname not in names_set:
                if not silent:
                    warns.append(f"Budget shift row {rid} (scale one channel): unknown channel {cname!r}; skipped.")
                continue
            raw_out.append(
                {
                    "type": "scale_channel",
                    "channel_name": cname,
                    "start_week": sw,
                    "end_week": ew,
                    "factor": factor,
                }
            )
        else:
            sw = int(st.session_state.get(f"bs_r_sw_{rid}", row.get("start_week", 1)))
            ew = int(st.session_state.get(f"bs_r_ew_{rid}", row.get("end_week", sw)))
            f_name = str(st.session_state.get(f"bs_from_{rid}", row.get("from_channel", ""))).strip()
            t_name = str(st.session_state.get(f"bs_to_{rid}", row.get("to_channel", ""))).strip()
            frac = float(st.session_state.get(f"bs_frac_{rid}", row.get("fraction", 0.0)))
            if sw < 1 or ew < 1:
                if not silent:
                    warns.append(f"Budget shift row {rid} (reallocate): weeks must be ≥ 1; skipped.")
                continue
            if ew < sw:
                if not silent:
                    warns.append(f"Budget shift row {rid} (reallocate): end_week < start_week; skipped.")
                continue
            if not f_name or not t_name:
                if not silent:
                    warns.append(f"Budget shift row {rid} (reallocate): both channels required; skipped.")
                continue
            if f_name == t_name:
                if not silent:
                    warns.append(f"Budget shift row {rid} (reallocate): from and to must differ; skipped.")
                continue
            if f_name not in names_set or t_name not in names_set:
                if not silent:
                    warns.append(
                        f"Budget shift row {rid} (reallocate): unknown channel "
                        f"(from={f_name!r}, to={t_name!r}); skipped."
                    )
                continue
            if n < 2:
                if not silent:
                    warns.append(f"Budget shift row {rid} (reallocate): need at least two channels; skipped.")
                continue
            raw_out.append(
                {
                    "type": "reallocate",
                    "start_week": sw,
                    "end_week": ew,
                    "from_channel": f_name,
                    "to_channel": t_name,
                    "fraction": frac,
                }
            )

    extra = str(st.session_state.get("budget_shift_extra_option") or "none").strip().lower()
    if extra not in ("none", "global", "global_and_channel"):
        extra = "none"

    names_list = list(effective_channel_names(merged, n))
    auto_list: List[Dict[str, Any]] = []
    if extra != "none" and names_list:
        sig = (int(merged.get("seed", 0)), extra, week_cap, tuple(names_list))
        if st.session_state.get("budget_shift_auto_sig") == sig and "budget_shift_auto_rules" in st.session_state:
            auto_list = list(st.session_state["budget_shift_auto_rules"])
        else:
            auto_list = generate_auto_budget_shift_rules(week_cap, names_list, extra, int(merged.get("seed", 0)))
            st.session_state["budget_shift_auto_sig"] = sig
            st.session_state["budget_shift_auto_rules"] = list(auto_list)
    elif extra == "none":
        st.session_state.pop("budget_shift_auto_sig", None)
        st.session_state.pop("budget_shift_auto_rules", None)

    combined = raw_out + auto_list

    try:
        normalized = _normalize_budget_shifts(combined)
    except (TypeError, ValueError) as e:
        if not silent:
            warns.append(f"budget_shifts: could not normalize rules ({e}); cleared.")
        return [], warns

    return normalized, warns


def render_budget_shifts_section(cfg: Dict[str, Any], n_channels: int) -> None:
    ensure_budget_shift_rows_initialized(cfg)

    st.markdown("##### Budget shifts")
    st.caption(
        "Rules run **after** the base weekly spend draw and **before** channel on/off masks. "
        "Weeks are **1-based**. **Scale window** multiplies all channels; **Scale one channel** multiplies "
        "a single named channel; **Reallocate** moves a fraction of spend between channels (optional "
        "YAML may omit `end_week` on reallocate to mean through end of run. "
        "**Extra shifts from seed** append reproducible random rules; they refresh when **Random seed**, "
        "this option, **Week range**, or **channel names** change."
    )

    st.selectbox(
        "Extra shifts from seed (appended after manual rules)",
        options=["none", "global", "global_and_channel"],
        format_func=lambda x: {
            "none": "None — manual rules only",
            "global": "Global — random all-channel scales + bounded reallocates",
            "global_and_channel": "Global + per-channel — also random single-channel scales",
        }[x],
        key="budget_shift_extra_option",
        on_change=_yaml_sync_from_form,
    )

    week_cap = max(1, int(st.session_state.get("week_range_num", cfg.get("week_range") or 52)))
    names = effective_channel_names(cfg, n_channels)
    rows: List[Dict[str, Any]] = st.session_state.budget_shift_ui_rows

    st.markdown("###### Manual rules")

    if rows:
        h0, h1, h2 = st.columns([2, 5, 1])
        h0.caption("Type")
        h1.caption("Parameters")
        h2.caption(" ")

    for row in list(rows):
        rid = int(row["id"])
        kind = str(st.session_state.get(f"bs_kind_{rid}", row.get("kind", "scale"))).strip().lower()
        if kind not in ("scale", "scale_channel", "reallocate"):
            kind = "scale"

        c0, c1, c2 = st.columns([2, 5, 1])
        with c0:
            st.selectbox(
                "Rule type",
                options=["scale", "scale_channel", "reallocate"],
                index={"scale": 0, "scale_channel": 1, "reallocate": 2}.get(kind, 0),
                key=f"bs_kind_{rid}",
                format_func=lambda x: {
                    "scale": "Scale (all channels)",
                    "scale_channel": "Scale one channel",
                    "reallocate": "Reallocate",
                }[x],
                label_visibility="collapsed",
                on_change=_yaml_sync_from_form,
            )
        with c1:
            kind_now = str(st.session_state.get(f"bs_kind_{rid}", kind)).strip().lower()
            if kind_now == "scale":
                a, b, c = st.columns([1, 1, 1])
                with a:
                    st.number_input(
                        "Start week",
                        min_value=1,
                        max_value=None,
                        step=1,
                        value=int(row.get("start_week", 1)),
                        key=f"bs_sw_{rid}",
                        help="First week in the inclusive scale window (1-based).",
                        on_change=_yaml_sync_from_form,
                    )
                with b:
                    st.number_input(
                        "End week",
                        min_value=1,
                        max_value=None,
                        step=1,
                        value=int(row.get("end_week", row.get("start_week", 1))),
                        key=f"bs_ew_{rid}",
                        help="Last week in the inclusive window.",
                        on_change=_yaml_sync_from_form,
                    )
                with c:
                    st.number_input(
                        "Factor",
                        min_value=0.0,
                        max_value=None,
                        step=0.05,
                        value=float(row.get("factor", 1.0)),
                        format="%.4f",
                        key=f"bs_factor_{rid}",
                        help="Multiply all channels’ spend in each week of the window.",
                        on_change=_yaml_sync_from_form,
                    )
            elif kind_now == "scale_channel":
                if not names:
                    st.info("Add a named channel to use per-channel scale.")
                else:
                    opts = list(names)
                    cv = str(row.get("channel_name", "") or "").strip()
                    if cv not in opts:
                        cv = opts[0]
                    a, b, c, d = st.columns([2, 1, 1, 1])
                    with a:
                        st.selectbox(
                            "Channel",
                            options=opts,
                            index=opts.index(cv) if cv in opts else 0,
                            key=f"bs_ch_name_{rid}",
                            label_visibility="collapsed",
                            on_change=_yaml_sync_from_form,
                        )
                    with b:
                        st.number_input(
                            "Start week",
                            min_value=1,
                            max_value=None,
                            step=1,
                            value=int(row.get("start_week", 1)),
                            key=f"bs_ch_sw_{rid}",
                            on_change=_yaml_sync_from_form,
                        )
                    with c:
                        st.number_input(
                            "End week",
                            min_value=1,
                            max_value=None,
                            step=1,
                            value=int(row.get("end_week", row.get("start_week", 1))),
                            key=f"bs_ch_ew_{rid}",
                            on_change=_yaml_sync_from_form,
                        )
                    with d:
                        st.number_input(
                            "Factor",
                            min_value=0.0,
                            max_value=None,
                            step=0.05,
                            value=float(row.get("factor", 1.0)),
                            format="%.4f",
                            key=f"bs_ch_factor_{rid}",
                            on_change=_yaml_sync_from_form,
                        )
            else:
                if n_channels < 2 or not names:
                    st.info("Add at least two channels with names to use reallocate rules.")
                else:
                    a, b, c, d, e = st.columns([1, 1, 2, 2, 1])
                    with a:
                        st.number_input(
                            "Start week",
                            min_value=1,
                            max_value=None,
                            step=1,
                            value=int(row.get("start_week", 1)),
                            key=f"bs_r_sw_{rid}",
                            help="First week of the inclusive reallocate window.",
                            on_change=_yaml_sync_from_form,
                        )
                    with b:
                        st.number_input(
                            "End week",
                            min_value=1,
                            max_value=None,
                            step=1,
                            value=int(row.get("end_week", row.get("start_week", 1))),
                            key=f"bs_r_ew_{rid}",
                            help="Last week of the window (set equal to start for a single week).",
                            on_change=_yaml_sync_from_form,
                        )
                    with c:
                        opts = list(names)
                        fv = str(row.get("from_channel", "") or "").strip()
                        if fv not in opts:
                            fv = opts[0]
                        st.selectbox(
                            "From channel",
                            options=opts,
                            index=opts.index(fv) if fv in opts else 0,
                            key=f"bs_from_{rid}",
                            label_visibility="collapsed",
                            on_change=_yaml_sync_from_form,
                        )
                    with d:
                        tv = str(row.get("to_channel", "") or "").strip()
                        if tv not in opts or tv == fv:
                            tv = opts[1] if len(opts) > 1 and opts[1] != fv else opts[0]
                        st.selectbox(
                            "To channel",
                            options=opts,
                            index=opts.index(tv) if tv in opts else 0,
                            key=f"bs_to_{rid}",
                            label_visibility="collapsed",
                            on_change=_yaml_sync_from_form,
                        )
                    with e:
                        st.number_input(
                            "Fraction",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.05,
                            value=float(min(1.0, max(0.0, row.get("fraction", 0.0)))),
                            format="%.2f",
                            key=f"bs_frac_{rid}",
                            help="Share of **from** channel spend moved to **to** each week in the window.",
                            on_change=_yaml_sync_from_form,
                        )
        with c2:
            st.write("")
            if st.button("Remove", key=f"bs_rm_{rid}", width="stretch"):
                st.session_state.budget_shift_ui_rows = [
                    r for r in st.session_state.budget_shift_ui_rows if r["id"] != rid
                ]
                for k in (
                    f"bs_kind_{rid}",
                    f"bs_sw_{rid}",
                    f"bs_ew_{rid}",
                    f"bs_factor_{rid}",
                    f"bs_ch_name_{rid}",
                    f"bs_ch_sw_{rid}",
                    f"bs_ch_ew_{rid}",
                    f"bs_ch_factor_{rid}",
                    f"bs_r_sw_{rid}",
                    f"bs_r_ew_{rid}",
                    f"bs_from_{rid}",
                    f"bs_to_{rid}",
                    f"bs_frac_{rid}",
                    f"bs_rm_{rid}",
                ):
                    st.session_state.pop(k, None)
                st.rerun()

    if st.button("Add budget shift rule", width="content"):
        nid = int(st.session_state.get("budget_shift_next_id", 1))
        st.session_state.budget_shift_next_id = nid + 1
        st.session_state.budget_shift_ui_rows = list(st.session_state.budget_shift_ui_rows) + [
            {
                "id": nid,
                "kind": "scale",
                "start_week": 1,
                "end_week": min(week_cap, 1),
                "factor": 1.0,
            }
        ]
        st.rerun()
