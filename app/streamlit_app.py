"""
Streamlit entry: marketing simulator UI.

Run from repository root:
  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

# Repository root on sys.path for `scripts` imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.cache import cached_dataframe_schema_ok, clear_run_cache, run_with_cache  # noqa: E402
from app.default_channel import default_channel_dict  # noqa: E402
from app.pipeline_runner import run_pipeline  # noqa: E402
from app.theme import inject_theme_css  # noqa: E402
from app.ui_helpers import apply_overrides, get_at, path_string_to_parts  # noqa: E402

EXAMPLE_YAML_PATH = _REPO_ROOT / "example.yaml"
UI_SCHEMA_PATH = Path(__file__).resolve().parent / "ui_schema.yaml"

BLUE = "#1D63ED"
ORANGE = "#F39C59"
ACCENT2 = "#2563eb"
# Wong-inspired palette: orange / blue / green — distinct for common color-vision deficiencies
CHART_PAL_CVD = ("#E69F00", "#0072B2", "#009E73")


def _load_ui_schema() -> Dict[str, Any]:
    with open(UI_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_example_text() -> str:
    if EXAMPLE_YAML_PATH.is_file():
        return EXAMPLE_YAML_PATH.read_text(encoding="utf-8")
    return "# example.yaml not found\nrun_identifier: Demo\nweek_range: 26\nchannel_list: []\n"


def _yaml_dump(cfg: Dict[str, Any]) -> str:
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _channel_name(data: Dict[str, Any], index: int) -> str:
    ch = get_at(data, ["channel_list", index, "channel"])
    if isinstance(ch, dict) and ch.get("channel_name"):
        return str(ch["channel_name"])
    return f"Channel {index + 1}"


def _fmt_default(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:g}" if val != int(val) else str(int(val))
    return str(val)


def _pc_field_key(i: int, path_suffix: str, list_index: Optional[int]) -> str:
    li = list_index if list_index is not None else "x"
    return f"pc_{i}_{path_suffix.replace('.', '_')}_{li}"


_default_channel_template: Optional[Dict[str, Any]] = None


def _get_default_channel_template() -> Dict[str, Any]:
    global _default_channel_template
    if _default_channel_template is None:
        _default_channel_template = default_channel_dict()
    return _default_channel_template


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


def _select_session_key(i: int, path_suffix: str) -> str:
    return f"sel_{i}_{path_suffix.replace('.', '_')}"


def _effective_curve_type(
    i: int,
    path_suffix: str,
    data: Dict[str, Any],
    options: List[str],
) -> str:
    """Widget selection if set, else YAML value, else first option."""
    key = _select_session_key(i, path_suffix)
    if key in st.session_state:
        v = st.session_state[key]
        if v in options:
            return str(v)
    parts = path_string_to_parts(f"channel_list.{i}.{path_suffix}")
    cur = get_at(data, parts)
    if cur in options:
        return str(cur)
    return str(options[0]) if options else "linear"


def _saturation_slider_visible(item: Dict[str, Any], sat_type: str) -> bool:
    types = item.get("saturation_types")
    if types is None:
        return True
    return sat_type in types


def _adstock_slider_visible(item: Dict[str, Any], ad_type: str) -> bool:
    types = item.get("adstock_types")
    if types is None:
        return True
    return ad_type in types


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


def _adstock_weights_key(i: int) -> str:
    return f"adw_{i}"


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


def _parse_weights_csv(raw: Any) -> Tuple[Optional[List[float]], bool]:
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


def _render_adstock_weights_field(i: int, data: Dict[str, Any]) -> None:
    key = _adstock_weights_key(i)
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
        on_change=_yaml_sync_from_form,
    )


def _yaml_sync_from_form() -> None:
    st.session_state["yaml_manual_edit"] = False


def _yaml_mark_dirty() -> None:
    st.session_state["yaml_manual_edit"] = True


def _render_one_pc_field(
    i: int,
    item: Dict[str, Any],
    data: Dict[str, Any],
) -> None:
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

    key = _pc_field_key(i, path_suffix, list_index)
    tmpl_v = _template_scalar_at(path_suffix, list_index)
    if isinstance(cur, (int, float)) and not isinstance(cur, bool):
        default_v = cur
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
    label = item["label"]
    ph = f"Default: {_fmt_default(default_v)}"
    help_txt = item.get("help", "Leave empty to use the default in the placeholder.")
    # Widget value lives in session_state; seed when missing (e.g. after Apply YAML clears keys).
    # Equal to template default → leave blank (run still uses YAML / merged config).
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
        on_change=_yaml_sync_from_form,
    )


def _render_select_for_path(
    i: int,
    item: Dict[str, Any],
    data: Dict[str, Any],
) -> None:
    path_suffix = item["path"]
    full_path = f"channel_list.{i}.{path_suffix}"
    parts = path_string_to_parts(full_path)
    cur = get_at(data, parts)
    opts = list(item.get("options", []))
    key = _select_session_key(i, path_suffix)
    sel_idx = opts.index(cur) if cur in opts else 0
    if key not in st.session_state and opts:
        st.session_state[key] = opts[sel_idx]
    st.selectbox(
        item["label"],
        options=opts,
        index=sel_idx,
        key=key,
        help=item.get("help"),
        on_change=_yaml_sync_from_form,
    )


def _render_channel_widgets(schema: Dict[str, Any], data: Dict[str, Any], n: int) -> None:
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

        with st.expander(f"{name}", expanded=False):
            head_l, head_r = st.columns([5, 1])
            with head_l:
                st.text_input(
                    "Channel name",
                    key=f"ch_name_{i}",
                    on_change=_yaml_sync_from_form,
                )
            with head_r:
                st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                if st.button("Remove", key=f"del_ch_{i}", type="secondary", help="Delete this channel"):
                    cl = st.session_state.sim_config.get("channel_list") or []
                    if 0 <= i < len(cl):
                        cl.pop(i)
                        st.session_state.sim_config["channel_list"] = cl
                        st.session_state["yaml_manual_edit"] = False
                        _clear_channel_widget_keys()
                        st.rerun()

            st.caption(
                "† — template jitter may apply when values are "
                "auto-filled from defaults."
            )

            st.markdown("**Spend & ROI**")
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

            st.markdown("**Noise variances (simulation)**")
            noise = grouped.get("noise", [])
            if len(noise) >= 2:
                n1, n2 = st.columns(2)
                with n1:
                    _render_one_pc_field(i, noise[0], data)
                with n2:
                    _render_one_pc_field(i, noise[1], data)

            st.markdown("##### Saturation")
            st.caption("How impressions map to effective response before ROI (curve type + shape).")
            sat_opts = list(sat_select.get("options", [])) if sat_select else []
            if sat_select is not None:
                _render_select_for_path(i, sat_select, data)
            eff_sat = _effective_curve_type(i, "channel.saturation_config.type", data, sat_opts)
            sat = [
                it
                for it in grouped.get("saturation", [])
                if _saturation_slider_visible(it, eff_sat)
            ]
            _render_pc_fields_flex(i, sat, data)

            st.markdown("##### Adstock")
            st.caption("How past media carries into this week (type + decay parameters).")
            ad_opts = list(ad_select.get("options", [])) if ad_select else []
            if ad_select is not None:
                _render_select_for_path(i, ad_select, data)
            eff_ad = _effective_curve_type(i, "channel.adstock_decay_config.type", data, ad_opts)
            ads = [
                it
                for it in grouped.get("adstock", [])
                if _adstock_slider_visible(it, eff_ad)
            ]
            _render_pc_fields_flex(i, ads, data)
            if eff_ad == "weighted":
                _render_adstock_weights_field(i, data)


def _parse_optional_num(
    raw: Any, *, as_int: bool = False
) -> Tuple[Optional[float], bool]:
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


def _collect_overrides(schema: Dict[str, Any], n_channels: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    overrides: List[Dict[str, Any]] = []
    warnings: List[str] = []
    selects = list(schema.get("per_channel_selects", []))
    sat_select = next((s for s in selects if s.get("path") == "channel.saturation_config.type"), None)
    ad_select = next((s for s in selects if s.get("path") == "channel.adstock_decay_config.type"), None)
    sat_opts = list(sat_select.get("options", [])) if sat_select else []
    ad_opts = list(ad_select.get("options", [])) if ad_select else []
    base_cfg = st.session_state.sim_config

    for i in range(n_channels):
        eff_sat = _effective_curve_type(i, "channel.saturation_config.type", base_cfg, sat_opts)
        eff_ad = _effective_curve_type(i, "channel.adstock_decay_config.type", base_cfg, ad_opts)

        for item in schema.get("per_channel_sliders", []):
            if item.get("group") == "saturation" and not _saturation_slider_visible(item, eff_sat):
                continue
            if item.get("group") == "adstock" and not _adstock_slider_visible(item, eff_ad):
                continue
            path_suffix = item["path"]
            list_index = item.get("list_index")
            key = _pc_field_key(i, path_suffix, list_index)
            raw = st.session_state.get(key, "")
            is_lag = path_suffix.endswith("lag") or "adstock_decay_config.lag" in path_suffix
            val, ok = _parse_optional_num(raw, as_int=is_lag)
            if not ok:
                warnings.append(f"{item.get('label', path_suffix)}: invalid number, skipped.")
                continue
            if val is None:
                continue
            mn, mx = float(item["min"]), float(item["max"])
            val = max(mn, min(mx, val))
            full_path = f"channel_list.{i}.{path_suffix}"
            if list_index is not None:
                overrides.append(
                    {"path": full_path, "value": float(val), "list_index": list_index}
                )
            elif is_lag:
                overrides.append({"path": full_path, "value": int(val)})
            else:
                overrides.append({"path": full_path, "value": float(val)})

        if eff_ad == "weighted":
            wkey = _adstock_weights_key(i)
            raw = st.session_state.get(wkey, "")
            wl, wok = _parse_weights_csv(raw)
            if not wok:
                warnings.append(f"Channel {i + 1} adstock weights: invalid list, skipped.")
            elif wl is not None:
                overrides.append(
                    {
                        "path": f"channel_list.{i}.channel.adstock_decay_config.weights",
                        "value": wl,
                    }
                )

        for item in schema.get("per_channel_selects", []):
            path_suffix = item["path"]
            key = _select_session_key(i, path_suffix)
            if key not in st.session_state:
                continue
            full_path = f"channel_list.{i}.{path_suffix}"
            overrides.append({"path": full_path, "value": st.session_state[key]})

    return overrides, warnings


def _clear_channel_widget_keys() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith(("pc_", "sel_", "ch_name_", "del_ch_", "adw_")):
            del st.session_state[k]


def _clear_widget_keys() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith(
            (
                "tl_",
                "pc_",
                "sel_",
                "ch_name_",
                "week_range_",
                "run_identifier_",
                "seed_input",
                "del_ch_",
                "adw_",
            )
        ) or k in ("new_channel_name", "advanced_yaml"):
            del st.session_state[k]


def _merge_ui_into_config(schema: Dict[str, Any], *, silent: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    merged = copy.deepcopy(st.session_state.sim_config)
    merged["week_range"] = int(st.session_state.get("week_range_num", 52))
    merged["run_identifier"] = str(st.session_state.get("run_identifier_input", "run")).strip() or "run"
    merged["seed"] = int(st.session_state.get("seed_input", 0))

    n_channels = len(merged.get("channel_list") or [])
    overrides, warns = _collect_overrides(schema, n_channels)
    merged = apply_overrides(merged, overrides)

    for i in range(n_channels):
        key = f"ch_name_{i}"
        if key in st.session_state and i < len(merged["channel_list"]):
            nm = str(st.session_state[key]).strip()
            if nm:
                ch = merged["channel_list"][i].get("channel") or merged["channel_list"][i]
                if isinstance(ch, dict):
                    ch["channel_name"] = nm
    return merged, ([] if silent else warns)


def _norm_series(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def _channel_names_from_results_df(df: pd.DataFrame) -> List[str]:
    names: List[str] = []
    for col in df.columns:
        c = str(col).strip().lstrip("\ufeff")
        if c.endswith("_impressions") and c != "total_impressions":
            names.append(c[: -len("_impressions")])
    return names


def _results_df_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with stripped / BOM-safe headers so *_revenue lines up with *_impressions."""
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
    return out


def _make_charts(
    df: pd.DataFrame,
    *,
    channel: Optional[str] = None,
    overlay: bool = False,
    night: bool = False,
    colorblind: bool = False,
) -> go.Figure:
    c1, c2, c3 = CHART_PAL_CVD if colorblind else (BLUE, ORANGE, ACCENT2)
    grid = "rgba(148,163,184,0.2)" if night else "rgba(15,23,42,0.08)"
    paper = "rgba(15,23,42,0.3)" if night else "rgba(255,255,255,0)"
    plot_bg = "rgba(30,41,59,0.5)" if night else "rgba(248,250,252,0.9)"
    title_color = "#e2e8f0" if night else "#0f172a"

    r_col, sp_col, im_col = "revenue", "total_spend", "total_impressions"
    sub_r, sub_sp, sub_im = "Revenue", "Total spend", "Total impressions"
    if channel:
        rc, sc, ic = f"{channel}_revenue", f"{channel}_spend", f"{channel}_impressions"
        if rc in df.columns and sc in df.columns and ic in df.columns:
            r_col, sp_col, im_col = rc, sc, ic
            sub_r = f"Revenue ({channel})"
            sub_sp = f"Spend ({channel})"
            sub_im = f"Impressions ({channel})"
        else:
            channel = None

    if overlay:
        r = _norm_series(df[r_col])
        sp = _norm_series(df[sp_col])
        im = _norm_series(df[im_col])
        fig = go.Figure()
        nm_r = f"{sub_r} (normalized)"
        nm_sp = f"{sub_sp} (normalized)"
        nm_im = f"{sub_im} (normalized)"
        fig.add_trace(
            go.Scatter(
                x=df["week"],
                y=r,
                name=nm_r,
                mode="lines",
                line=dict(color=c1, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["week"],
                y=sp,
                name=nm_sp,
                mode="lines",
                line=dict(color=c2, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["week"],
                y=im,
                name=nm_im,
                mode="lines",
                line=dict(color=c3, width=2),
            )
        )
        overlay_title = "Series overlaid (min–max normalized per series)"
        if channel:
            overlay_title = f"{overlay_title} · {channel}"
        fig.update_layout(
            height=480,
            title=dict(text=overlay_title, font=dict(color=title_color)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=48, r=24, t=72, b=48),
            paper_bgcolor=paper,
            plot_bgcolor=plot_bg,
            xaxis=dict(gridcolor=grid, title="Week", color=title_color),
            yaxis=dict(
                gridcolor=grid,
                title="Normalized 0–1",
                color=title_color,
                rangemode="tozero",
            ),
        )
        return fig

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(sub_r, sub_sp, sub_im),
    )
    fig.add_trace(
        go.Scatter(
            x=df["week"], y=df[r_col], name=sub_r, mode="lines", line=dict(color=c1, width=2)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["week"], y=df[sp_col], name=sub_sp, mode="lines", line=dict(color=c2, width=2)
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["week"],
            y=df[im_col],
            name=sub_im,
            mode="lines",
            line=dict(color=c3, width=2),
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        height=720,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor=paper,
        plot_bgcolor=plot_bg,
    )
    fig.update_xaxes(title_text="Week", row=3, col=1, gridcolor=grid, color=title_color)
    fig.update_yaxes(gridcolor=grid, color=title_color, rangemode="tozero")
    for a in fig.layout.annotations:
        a.font = dict(color=title_color, size=12)
    return fig


def _preview_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rounded numeric preview for display."""
    prev = df.head(25).copy()
    for c in prev.columns:
        if prev[c].dtype == float or prev[c].dtype == "float64":
            prev[c] = prev[c].map(lambda x: round(float(x), 3) if pd.notna(x) else x)
    return prev


def _render_results_panel(df: pd.DataFrame, *, compact_toolbar: bool) -> None:
    night = st.session_state.get("night_mode", False)
    colorblind = bool(st.session_state.get("colorblind_charts", False))
    rid = st.session_state.get("last_run_id", "")
    hit = st.session_state.get("last_cache_hit", False)
    ch = st.session_state.get("last_hash", "")

    df = _results_df_clean_columns(df)

    if "overlay_results_charts" not in st.session_state:
        st.session_state.overlay_results_charts = False

    if compact_toolbar:
        st.title("Results")
    else:
        st.markdown("### Latest results")

    tb1, tb2, tb3 = st.columns([1, 1, 2])
    with tb1:
        if compact_toolbar and st.button("Edit configuration", type="primary", use_container_width=True):
            st.session_state.config_collapsed = False
            st.rerun()
    with tb2:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name=f"{rid or 'simulation'}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        f"Run **{rid}** · {'served from cache' if hit else 'newly computed'} · config `…{ch[-8:]}`"
    )

    ch_names = _channel_names_from_results_df(df)
    schema_ok = cached_dataframe_schema_ok(df)
    if not schema_ok and ch_names:
        if hit:
            st.info(
                "These results were loaded from **disk cache** in an older CSV shape (no per-channel "
                "**`*_revenue`** columns). Click **Run simulation** once to rebuild, or use "
                "**Clear simulation cache** in the sidebar."
            )
        else:
            st.warning(
                "This run’s table is missing per-channel **`*_revenue`** columns next to "
                "**`*_impressions`** (unexpected for a new run). **Stop and restart** the Streamlit "
                "server so it picks up the latest `scripts.main` code, then run again."
            )
        scope_options = ["All channels (totals)"]
        st.session_state.results_chart_scope = scope_options[0]
    else:
        scope_options = ["All channels (totals)"] + ch_names

    if "results_chart_scope" not in st.session_state:
        st.session_state.results_chart_scope = scope_options[0]
    if st.session_state.results_chart_scope not in scope_options:
        st.session_state.results_chart_scope = scope_options[0]

    csel1, csel2 = st.columns([1, 1])
    with csel1:
        st.selectbox(
            "Chart view",
            options=scope_options,
            key="results_chart_scope",
            help="Totals across all channels, or one channel’s revenue, spend, and impressions.",
        )
    with csel2:
        st.checkbox(
            "Overlay series (min–max normalized on one chart)",
            key="overlay_results_charts",
        )

    scope = str(st.session_state.results_chart_scope)
    ch_view: Optional[str] = None if scope == scope_options[0] else scope

    overlay = bool(st.session_state.overlay_results_charts)
    st.plotly_chart(
        _make_charts(df, channel=ch_view, overlay=overlay, night=night, colorblind=colorblind),
        use_container_width=True,
    )

    st.markdown("##### Data preview")
    st.caption("First 25 rows · values rounded for readability.")
    st.dataframe(
        _preview_table(df),
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    with st.expander("Configuration (YAML snapshot)", expanded=False):
        st.caption("Last merged settings (same structure as Advanced YAML).")
        st.code(
            _yaml_dump(st.session_state.get("sim_config") or {}),
            language="yaml",
        )


def main() -> None:
    st.set_page_config(
        page_title="BlueAlpha Simulator",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "night_mode" not in st.session_state:
        st.session_state.night_mode = False
    if "colorblind_charts" not in st.session_state:
        st.session_state.colorblind_charts = False

    inject_theme_css(night=bool(st.session_state.night_mode))

    schema = _load_ui_schema()

    if "sim_config" not in st.session_state:
        st.session_state.sim_config = yaml.safe_load(_load_example_text()) or {}
    if "config_collapsed" not in st.session_state:
        st.session_state.config_collapsed = False
    if "yaml_manual_edit" not in st.session_state:
        st.session_state.yaml_manual_edit = False

    if "week_range_num" not in st.session_state:
        st.session_state.week_range_num = int(st.session_state.sim_config.get("week_range", 52))
    if "run_identifier_input" not in st.session_state:
        st.session_state.run_identifier_input = str(
            st.session_state.sim_config.get("run_identifier", "run")
        )
    if "seed_input" not in st.session_state:
        s = st.session_state.sim_config.get("seed")
        st.session_state.seed_input = int(s) if s is not None else 0

    # Apply YAML updates widget-bound keys here, before those widgets are instantiated.
    if st.session_state.pop("_sync_top_widgets_from_sim_config", False):
        st.session_state.week_range_num = int(st.session_state.sim_config.get("week_range", 52))
        st.session_state.run_identifier_input = str(
            st.session_state.sim_config.get("run_identifier", "run")
        )
        s = st.session_state.sim_config.get("seed")
        st.session_state.seed_input = int(s) if s is not None else 0

    # Sync Advanced YAML from form when the user is not mid-edit in the YAML box.
    pending_yaml = st.session_state.pop("pending_yaml_dump", None)
    if pending_yaml is not None:
        st.session_state.advanced_yaml = pending_yaml
        st.session_state.yaml_manual_edit = False
    elif not st.session_state.get("yaml_manual_edit", False):
        merged_preview, _ = _merge_ui_into_config(schema, silent=True)
        st.session_state.advanced_yaml = _yaml_dump(merged_preview)
    elif "advanced_yaml" not in st.session_state:
        st.session_state.advanced_yaml = _yaml_dump(st.session_state.sim_config)

    with st.sidebar:
        st.header("Settings")
        st.checkbox("Night mode", key="night_mode")
        st.checkbox(
            "Colorblind-safe chart colors",
            key="colorblind_charts",
            help="Uses orange / blue / green lines (Wong-style) so series are easier to distinguish.",
        )

        st.divider()
        if st.button("Reset to example.yaml", use_container_width=True):
            st.session_state.sim_config = yaml.safe_load(_load_example_text()) or {}
            st.session_state.config_collapsed = False
            _clear_widget_keys()
            st.session_state.week_range_num = int(st.session_state.sim_config.get("week_range", 52))
            st.session_state.run_identifier_input = str(
                st.session_state.sim_config.get("run_identifier", "run")
            )
            s = st.session_state.sim_config.get("seed")
            st.session_state.seed_input = int(s) if s is not None else 0
            st.session_state["pending_yaml_dump"] = _yaml_dump(st.session_state.sim_config)
            st.session_state.yaml_manual_edit = False
            st.rerun()

        if st.button("Clear simulation cache", use_container_width=True, help="Remove cached runs on disk"):
            n = clear_run_cache()
            st.caption(f"Cleared {n} cached file(s).")

    user_dict = st.session_state.sim_config
    n_channels = len(user_dict.get("channel_list") or [])

    df = st.session_state.get("last_df")
    show_charts = df is not None

    if st.session_state.config_collapsed and show_charts:
        _render_results_panel(df, compact_toolbar=True)
        return

    st.title("Marketing mix simulator")
    st.caption("Configure the simulation below, then run.")

    st.markdown("##### Simulation settings")
    col_w, col_r, col_s = st.columns(3)
    with col_w:
        st.number_input(
            "Week range",
            min_value=1,
            max_value=None,
            step=1,
            key="week_range_num",
            help="Simulation length in weeks.",
            on_change=_yaml_sync_from_form,
        )
    with col_r:
        st.text_input(
            "Run Name",
            key="run_identifier_input",
            placeholder="e.g. Example Alpha",
            help="Labels CSV downloads and run summaries.",
            on_change=_yaml_sync_from_form,
        )
    with col_s:
        st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            step=1,
            key="seed_input",
            help="Fixes randomness so runs and cache keys are reproducible.",
            on_change=_yaml_sync_from_form,
        )

    st.markdown("##### Channels")
    row_a, row_b = st.columns([4, 1])
    with row_a:
        new_nm = st.text_input(
            "Name for new channel",
            key="new_channel_name",
            placeholder="e.g. TikTok",
            on_change=_yaml_sync_from_form,
        )
    with row_b:
        st.write("")
        if st.button("Add channel", use_container_width=True):
            nm = (new_nm or "").strip() or "New channel"
            ch = default_channel_dict()
            ch["channel_name"] = nm
            if "channel_list" not in st.session_state.sim_config:
                st.session_state.sim_config["channel_list"] = []
            st.session_state.sim_config["channel_list"].append({"channel": ch})
            st.session_state["yaml_manual_edit"] = False
            _clear_channel_widget_keys()
            st.rerun()

    if n_channels > 0:
        _render_channel_widgets(schema, user_dict, n_channels)
    else:
        st.info("Add at least one channel to run the simulation.")

    with st.expander("Advanced: edit full YAML", expanded=False):
        st.caption(
            "Stays in sync with the form unless you edit here—then click **Apply YAML to form** to load it. "
            "Editing the form updates this panel on the next run."
        )
        st.text_area(
            "YAML",
            height=280,
            key="advanced_yaml",
            label_visibility="collapsed",
            on_change=_yaml_mark_dirty,
        )
        if st.button("Apply YAML to form", type="secondary"):
            try:
                parsed = yaml.safe_load(st.session_state.advanced_yaml)
                if not isinstance(parsed, dict):
                    raise ValueError("YAML must parse to a mapping (dict).")
                st.session_state.sim_config = parsed
                st.session_state.yaml_manual_edit = False
                _clear_channel_widget_keys()
                st.session_state["_sync_top_widgets_from_sim_config"] = True
                st.session_state["pending_yaml_dump"] = _yaml_dump(st.session_state.sim_config)
                st.success("YAML applied.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not apply YAML: {e}")

    run_clicked = st.button("Run simulation", type="primary", use_container_width=False)

    if run_clicked:
        try:
            merged, warns = _merge_ui_into_config(schema)
            for w in warns:
                st.warning(w)
            if not (merged.get("channel_list") or []):
                raise ValueError("Add at least one channel before running.")
            df_out, run_id, cache_hit, cfg_hash = run_with_cache(merged, run_pipeline)
            st.session_state["last_df"] = df_out
            st.session_state["last_run_id"] = run_id
            st.session_state["last_cache_hit"] = cache_hit
            st.session_state["last_hash"] = cfg_hash
            st.session_state["last_error"] = None
            st.session_state.sim_config = copy.deepcopy(merged)
            st.session_state["pending_yaml_dump"] = _yaml_dump(merged)
            st.session_state.yaml_manual_edit = False
            st.session_state.config_collapsed = True
            st.rerun()
        except Exception as e:
            st.session_state["last_error"] = str(e)
            st.session_state["last_df"] = None
            st.session_state.config_collapsed = False

    err: Optional[str] = st.session_state.get("last_error")
    if err:
        st.error(err)

    df2 = st.session_state.get("last_df")
    if df2 is not None and not st.session_state.config_collapsed:
        st.markdown("---")
        _render_results_panel(df2, compact_toolbar=False)


if __name__ == "__main__":
    main()
