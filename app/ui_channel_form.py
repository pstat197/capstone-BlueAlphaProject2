"""Per-channel Streamlit widgets (spend, noise, saturation, adstock)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import streamlit as st

from app.default_channel import default_channel_dict
from app.ui_channel_toggles import (
    channel_status_summary,
    ensure_channel_toggle_state_initialized,
    render_channel_adstock_enable_checkbox,
    render_channel_saturation_enable_checkbox,
    render_channel_toggles_block,
)
from app.ui_config_merge import clear_channel_widget_keys
from app.ui_curve_preview import render_adstock_preview, render_saturation_preview
from app.ui_form_state import (
    adstock_slider_visible,
    adstock_weights_key,
    effective_curve_type,
    pc_field_key,
    saturation_slider_visible,
    select_session_key,
)
from app.ui_help_markdown import (
    ADSTOCK_TYPES_GUIDE_MD,
    NOISE_PARAMETERS_GUIDE_MD,
    SATURATION_TYPES_GUIDE_MD,
)
from app.ui_helpers import get_at, path_string_to_parts


def yaml_sync_from_form() -> None:
    st.session_state["yaml_manual_edit"] = False


def yaml_mark_dirty() -> None:
    st.session_state["yaml_manual_edit"] = True


_default_channel_template: Optional[Dict[str, Any]] = None


def _get_default_channel_template() -> Dict[str, Any]:
    global _default_channel_template
    if _default_channel_template is None:
        _default_channel_template = default_channel_dict()
    return _default_channel_template


def _channel_name(data: Dict[str, Any], index: int) -> str:
    ch = get_at(data, ["channel_list", index, "channel"])
    if isinstance(ch, dict) and ch.get("channel_name"):
        return str(ch["channel_name"])
    return f"Channel {index + 1}"


def _fmt_default(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:g}" if val != int(val) else str(int(val))
    return str(val)


def _numeric_close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9 * max(1.0, abs(a), abs(b)))


def _template_scalar_at(path_suffix: str, list_index: Optional[int]) -> Any:
    """Scalar from default channel template at the same path as the UI field."""
    tmpl_ch = _get_default_channel_template()
    ref = {"channel_list": [{"channel": tmpl_ch}]}
    v = get_at(ref, path_string_to_parts(f"channel_list.0.{path_suffix}"))
    if list_index is not None:
        if isinstance(v, list) and len(v) > list_index:
            return v[list_index]
        return None
    return v


def _matches_channel_template_default(
    path_suffix: str,
    list_index: Optional[int],
    cur: Any,
) -> bool:
    """True when YAML value equals the default channel template (form stays blank; placeholder shows default)."""
    if cur is None or isinstance(cur, bool):
        return False
    exp = _template_scalar_at(path_suffix, list_index)
    if exp is None:
        return False
    if isinstance(cur, (int, float)) and isinstance(exp, (int, float)) and not isinstance(exp, bool):
        return _numeric_close(float(cur), float(exp))
    return cur == exp


def _group_slider_items(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {
        "core": [],
        "noise": [],
        "saturation": [],
        "adstock": [],
    }
    for it in items:
        g = str(it.get("group", "core"))
        out.setdefault(g, []).append(it)
    return out


def _channel_adstock_weights_from_data(data: Dict[str, Any], i: int) -> List[float]:
    parts = path_string_to_parts(f"channel_list.{i}.channel.adstock_decay_config.weights")
    w = get_at(data, parts)
    if isinstance(w, list) and w:
        out: List[float] = []
        for x in w:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                continue
        return out if out else [1.0]
    return [1.0]


def _fmt_weights_placeholder(weights: List[float]) -> str:
    return ", ".join(_fmt_default(x) for x in weights)


def _current_seasonality_dict(data: Dict[str, Any], i: int) -> Dict[str, Any]:
    parts = path_string_to_parts(f"channel_list.{i}.channel.seasonality_config")
    raw = get_at(data, parts)
    return dict(raw) if isinstance(raw, dict) else {}


def _init_seasonality_state(i: int, data: Dict[str, Any]) -> None:
    sea = _current_seasonality_dict(data, i)
    sea_type = str(sea.get("type", "none")).strip().lower() if sea else "none"
    if sea_type not in {"none", "sin", "fourier", "categorical"}:
        sea_type = "none"
    st.session_state.setdefault(f"sea_type_{i}", sea_type)
    st.session_state.setdefault(f"sea_amp_{i}", float(sea.get("amplitude", 0.2)))
    st.session_state.setdefault(f"sea_period_{i}", int(sea.get("period", 52)))
    st.session_state.setdefault(f"sea_phase_{i}", float(sea.get("phase", 0.0)))
    st.session_state.setdefault(f"sea_k_{i}", int(sea.get("K", 2)))
    st.session_state.setdefault(f"sea_scale_{i}", float(sea.get("scale", 0.1)))
    pattern = sea.get("pattern", [1.0, 1.0, 1.0, 1.0])
    if not isinstance(pattern, list):
        pattern = [1.0, 1.0, 1.0, 1.0]
    st.session_state.setdefault(f"sea_pattern_{i}", ", ".join(_fmt_default(float(x)) for x in pattern))


def _render_pc_fields_flex(i: int, fields: List[Dict[str, Any]], data: Dict[str, Any]) -> None:
    """Lay out 1–3 per-channel numeric fields in rows of up to 3 columns."""
    if not fields:
        return
    chunk = 3
    for start in range(0, len(fields), chunk):
        row = fields[start : start + chunk]
        cols = st.columns(len(row))
        for col, it in zip(cols, row):
            with col:
                _render_one_pc_field(i, it, data)


def _render_adstock_weights_field(i: int, data: Dict[str, Any]) -> None:
    key = adstock_weights_key(i)
    default_w = _channel_adstock_weights_from_data(data, i)
    ph = f"Default: {_fmt_weights_placeholder(default_w)}"
    help_txt = (
        "Comma-separated weights for the adstock kernel (oldest → newest lag). "
        "Leave empty to keep YAML / default. Example: 0.5, 0.3, 0.2"
    )
    if key not in st.session_state:
        tmpl_ch = _get_default_channel_template()
        tmpl_ad = tmpl_ch.get("adstock_decay_config") or {}
        tmpl_list = tmpl_ad.get("weights")
        if not isinstance(tmpl_list, list):
            tmpl_list = [1.0]
        try:
            tmpl_wf = [float(x) for x in tmpl_list]
        except (TypeError, ValueError):
            tmpl_wf = [1.0]
        same_as_template = len(default_w) == len(tmpl_wf) and all(
            _numeric_close(float(a), float(b)) for a, b in zip(default_w, tmpl_wf)
        )
        if same_as_template:
            st.session_state[key] = ""
        else:
            st.session_state[key] = _fmt_weights_placeholder(default_w)
    st.text_input(
        "Adstock weights (comma-separated)",
        key=key,
        placeholder=ph,
        help=help_txt,
        on_change=yaml_sync_from_form,
    )


_SLIDER_GROUPS = {"saturation", "adstock", "noise"}


def _resolve_cur_and_default(
    i: int,
    item: Dict[str, Any],
    data: Dict[str, Any],
) -> tuple[Any, Any, Any]:
    """Return (cur, default_v, tmpl_v) for a per-channel numeric field."""
    path_suffix = item["path"]
    full_path = f"channel_list.{i}.{path_suffix}"
    parts = path_string_to_parts(full_path)
    raw = get_at(data, parts)
    list_index = item.get("list_index")
    if list_index is not None:
        if isinstance(raw, list) and len(raw) > list_index:
            cur: Any = raw[list_index]
        else:
            cur = None
    else:
        cur = raw
    if isinstance(cur, bool) or (cur is not None and not isinstance(cur, (int, float))):
        cur = None

    tmpl_v = _template_scalar_at(path_suffix, list_index)
    if isinstance(cur, (int, float)) and not isinstance(cur, bool):
        default_v: Any = cur
    elif (
        tmpl_v is not None
        and isinstance(tmpl_v, (int, float))
        and not isinstance(tmpl_v, bool)
    ):
        default_v = tmpl_v
    else:
        default_v = item["min"]
    if not isinstance(default_v, (int, float)):
        default_v = item["min"]
    return cur, default_v, tmpl_v


def _render_one_pc_field(
    i: int,
    item: Dict[str, Any],
    data: Dict[str, Any],
) -> None:
    """Dispatch a per-channel numeric field to either the slider or text renderer.

    Saturation / adstock / noise parameters use bounded sliders so the effect
    of small tweaks is immediately visible in the live curve preview. Core
    fields (ROI, baseline, CPM, spend range) stay as text inputs with a
    ``Default: …`` placeholder so they can be cleared back to defaults.
    """
    group = str(item.get("group", "core"))
    if group in _SLIDER_GROUPS:
        _render_slider_field(i, item, data)
    else:
        _render_text_field(i, item, data)


def _render_text_field(
    i: int,
    item: Dict[str, Any],
    data: Dict[str, Any],
) -> None:
    path_suffix = item["path"]
    list_index = item.get("list_index")
    cur, default_v, _tmpl_v = _resolve_cur_and_default(i, item, data)

    key = pc_field_key(i, path_suffix, list_index)
    label = item["label"]
    ph = f"Default: {_fmt_default(default_v)}"
    help_txt = item.get("help", "Leave empty to use the default in the placeholder.")
    if key not in st.session_state:
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            if _matches_channel_template_default(path_suffix, list_index, cur):
                st.session_state[key] = ""
            else:
                st.session_state[key] = _fmt_default(cur)
        else:
            st.session_state[key] = ""
    st.text_input(
        label,
        key=key,
        placeholder=ph,
        help=help_txt,
        on_change=yaml_sync_from_form,
    )


def _render_slider_field(
    i: int,
    item: Dict[str, Any],
    data: Dict[str, Any],
) -> None:
    """Bounded slider for saturation / adstock / noise parameters.

    Session state is stored as a concrete number (int for ``lag``, float
    otherwise). ``parse_optional_num`` in the merge path accepts both strings
    and numbers, so this coexists cleanly with text-field overrides for core
    fields that may still be string-typed.
    """
    path_suffix = item["path"]
    list_index = item.get("list_index")
    cur, default_v, _tmpl_v = _resolve_cur_and_default(i, item, data)

    key = pc_field_key(i, path_suffix, list_index)
    label = item["label"]
    help_txt = item.get("help", "")

    is_int = path_suffix.endswith("lag")
    if is_int:
        mn_c: Any = int(item["min"])
        mx_c: Any = int(item["max"])
        step_c: Any = int(item["step"])
        init_raw = cur if isinstance(cur, (int, float)) and not isinstance(cur, bool) else default_v
        try:
            init_val: Any = int(round(float(init_raw)))
        except (TypeError, ValueError):
            init_val = mn_c
        init_val = max(mn_c, min(mx_c, init_val))
    else:
        mn_c = float(item["min"])
        mx_c = float(item["max"])
        step_c = float(item["step"])
        init_raw = cur if isinstance(cur, (int, float)) and not isinstance(cur, bool) else default_v
        try:
            init_val = float(init_raw)
        except (TypeError, ValueError):
            init_val = mn_c
        init_val = max(mn_c, min(mx_c, init_val))

    existing = st.session_state.get(key)
    if not isinstance(existing, (int, float)) or isinstance(existing, bool):
        st.session_state[key] = init_val
    else:
        clamped = max(mn_c, min(mx_c, int(existing) if is_int else float(existing)))
        if clamped != existing:
            st.session_state[key] = clamped

    st.slider(
        label,
        min_value=mn_c,
        max_value=mx_c,
        step=step_c,
        key=key,
        help=help_txt,
        on_change=yaml_sync_from_form,
    )


def _render_type_radio_for_path(
    i: int,
    item: Dict[str, Any],
    data: Dict[str, Any],
) -> None:
    """Radio list for curve type (same session keys / YAML merge as former selectbox)."""
    path_suffix = item["path"]
    full_path = f"channel_list.{i}.{path_suffix}"
    parts = path_string_to_parts(full_path)
    cur = get_at(data, parts)
    opts = list(item.get("options", []))
    key = select_session_key(i, path_suffix)
    sel_idx = opts.index(cur) if cur in opts else 0
    st.radio(
        item["label"],
        options=opts,
        index=sel_idx,
        key=key,
        help=item.get("help"),
        on_change=yaml_sync_from_form,
    )


def render_channel_widgets(schema: Dict[str, Any], data: Dict[str, Any], n: int) -> None:
    items = list(schema.get("per_channel_sliders", []))
    grouped = _group_slider_items(items)
    selects = list(schema.get("per_channel_selects", []))
    sat_select = next(
        (s for s in selects if s.get("path") == "channel.saturation_config.type"),
        None,
    )
    ad_select = next(
        (s for s in selects if s.get("path") == "channel.adstock_decay_config.type"),
        None,
    )

    for i in range(n):
        name = _channel_name(data, i)
        if f"ch_name_{i}" not in st.session_state:
            st.session_state[f"ch_name_{i}"] = name

        week_range_for_ranges = int(
            st.session_state.get(
                "week_range_num",
                data.get("week_range", 52),
            )
        )
        # Initialize toggle state before rendering the expander so the
        # collapsed-header summary reflects live session values.
        ensure_channel_toggle_state_initialized(data, i, week_range_for_ranges)
        summary = channel_status_summary(i, data)
        header = f"{name}  —  {summary}" if summary else f"{name}"

        # `key=` makes Streamlit persist the open/closed state across reruns,
        # so toggling a checkbox or adding a pause window no longer snaps the
        # expander shut. With `key` set, `expanded` is only the initial state.
        with st.expander(header, expanded=False, key=f"ch_exp_{i}"):
            _init_seasonality_state(i, data)
            head_l, head_r = st.columns([5, 1])
            with head_l:
                st.text_input(
                    "Channel name",
                    key=f"ch_name_{i}",
                    on_change=yaml_sync_from_form,
                )
            with head_r:
                st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                if st.button("Remove", key=f"del_ch_{i}", type="secondary", help="Delete this channel"):
                    cl = st.session_state.sim_config.get("channel_list") or []
                    if 0 <= i < len(cl):
                        cl.pop(i)
                        st.session_state.sim_config["channel_list"] = cl
                        st.session_state["yaml_manual_edit"] = False
                        clear_channel_widget_keys()
                        st.rerun()

            st.caption(
                "† — template jitter may apply when values are "
                "auto-filled from defaults."
            )

            render_channel_toggles_block(i, data, week_range_for_ranges)
            channel_active = bool(st.session_state.get(f"tog_enabled_{i}", True))
            if not channel_active:
                # Auto-collapse inactive channels and hide all non-reactivation controls.
                st.session_state[f"ch_exp_{i}"] = False
                continue

            st.markdown("##### Spend & ROI")
            st.caption(
                "Spend range and gamma sampling drive weekly spend; CPM maps spend to impressions. "
                "True ROI scales effective (saturated, adstocked) media into revenue. "
                "Baseline revenue is added each week regardless of media."
            )
            core = grouped.get("core", [])
            if len(core) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    _render_one_pc_field(i, core[0], data)
                with c2:
                    _render_one_pc_field(i, core[1], data)
            if len(core) >= 5:
                c3, c4, c5 = st.columns(3)
                with c3:
                    _render_one_pc_field(i, core[2], data)
                with c4:
                    _render_one_pc_field(i, core[3], data)
                with c5:
                    _render_one_pc_field(i, core[4], data)
            if len(core) >= 6:
                _render_one_pc_field(i, core[5], data)

            st.markdown("##### Baseline trend & seasonality")
            st.caption(
                "Trend modifies baseline linearly over time. Seasonality multiplies the baseline "
                "before media contribution and noise are added."
            )
            st.selectbox(
                "Seasonality type",
                options=["none", "sin", "fourier", "categorical"],
                key=f"sea_type_{i}",
                help="none disables seasonality. Other options map directly to seasonality_config.type.",
                on_change=yaml_sync_from_form,
            )
            sea_type = str(st.session_state.get(f"sea_type_{i}", "none"))
            if sea_type == "sin":
                a, b, c = st.columns(3)
                with a:
                    st.number_input(
                        "Amplitude",
                        min_value=0.0,
                        max_value=2.0,
                        step=0.01,
                        key=f"sea_amp_{i}",
                        on_change=yaml_sync_from_form,
                    )
                with b:
                    st.number_input(
                        "Period",
                        min_value=1,
                        max_value=520,
                        step=1,
                        key=f"sea_period_{i}",
                        on_change=yaml_sync_from_form,
                    )
                with c:
                    st.number_input(
                        "Phase",
                        min_value=-520.0,
                        max_value=520.0,
                        step=1.0,
                        key=f"sea_phase_{i}",
                        on_change=yaml_sync_from_form,
                    )
            elif sea_type == "fourier":
                a, b, c = st.columns(3)
                with a:
                    st.number_input(
                        "Period",
                        min_value=1,
                        max_value=520,
                        step=1,
                        key=f"sea_period_{i}",
                        on_change=yaml_sync_from_form,
                    )
                with b:
                    st.number_input(
                        "Harmonics (K)",
                        min_value=1,
                        max_value=20,
                        step=1,
                        key=f"sea_k_{i}",
                        on_change=yaml_sync_from_form,
                    )
                with c:
                    st.number_input(
                        "Scale",
                        min_value=0.0,
                        max_value=2.0,
                        step=0.01,
                        key=f"sea_scale_{i}",
                        on_change=yaml_sync_from_form,
                    )
            elif sea_type == "categorical":
                st.text_input(
                    "Pattern multipliers (comma-separated)",
                    key=f"sea_pattern_{i}",
                    help="Repeating multiplicative factors, e.g. 1.0, 1.1, 0.9, 1.0",
                    on_change=yaml_sync_from_form,
                )

            st.markdown("##### Noise (simulation)")
            st.caption(
                "Random variation in **impressions** (right after CPM) and in **revenue** (after all media math). "
                "Both use √(your value) × a weekly level, so they scale with that week’s size — not fixed dollar noise."
            )
            with st.expander(
                "How noise values work (click to open)",
                expanded=False,
                key=f"noise_ref_{i}",
            ):
                st.markdown(NOISE_PARAMETERS_GUIDE_MD)
            noise = grouped.get("noise", [])
            if len(noise) >= 2:
                n1, n2 = st.columns(2)
                with n1:
                    _render_one_pc_field(i, noise[0], data)
                with n2:
                    _render_one_pc_field(i, noise[1], data)

            st.markdown("##### Saturation")
            st.caption(
                "Step 1 in the revenue path: turns raw impressions into effective media. "
                "Pick a type below, then adjust only the fields that appear."
            )
            render_channel_saturation_enable_checkbox(i)
            if st.session_state.get(f"tog_sat_{i}", True):
                with st.expander(
                    "Saturation types — reference (click to open)",
                    expanded=False,
                    key=f"sat_types_ref_{i}",
                ):
                    st.markdown(SATURATION_TYPES_GUIDE_MD)
                sat_opts = list(sat_select.get("options", [])) if sat_select else []
                if sat_select is not None:
                    _render_type_radio_for_path(i, sat_select, data)
                eff_sat = effective_curve_type(i, "channel.saturation_config.type", data, sat_opts)
                sat = [
                    it
                    for it in grouped.get("saturation", [])
                    if saturation_slider_visible(it, eff_sat)
                ]
                _render_pc_fields_flex(i, sat, data)
                if sat_opts:
                    render_saturation_preview(i, data, sat_opts)

            st.markdown("##### Adstock")
            st.caption(
                "Step 2: spreads each week’s saturated response over neighboring weeks (carry-over / memory)."
            )
            render_channel_adstock_enable_checkbox(i)
            if st.session_state.get(f"tog_ads_{i}", True):
                with st.expander(
                    "Adstock types — reference (click to open)",
                    expanded=False,
                    key=f"ad_types_ref_{i}",
                ):
                    st.markdown(ADSTOCK_TYPES_GUIDE_MD)
                ad_opts = list(ad_select.get("options", [])) if ad_select else []
                if ad_select is not None:
                    _render_type_radio_for_path(i, ad_select, data)
                eff_ad = effective_curve_type(i, "channel.adstock_decay_config.type", data, ad_opts)
                ads = [
                    it
                    for it in grouped.get("adstock", [])
                    if adstock_slider_visible(it, eff_ad)
                ]
                _render_pc_fields_flex(i, ads, data)
                if eff_ad == "weighted":
                    _render_adstock_weights_field(i, data)
                if ad_opts:
                    render_adstock_preview(i, data, ad_opts)

