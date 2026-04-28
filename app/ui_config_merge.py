"""Merge Streamlit widget state into `sim_config` and apply path overrides."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import streamlit as st

from app.ui_channel_toggles import (
    clear_toggle_widget_keys,
    merge_channel_toggles_into_config,
)
from app.ui_form_state import (
    adstock_slider_visible,
    adstock_weights_key,
    effective_curve_type,
    parse_optional_num,
    parse_weights_csv,
    pc_field_key,
    saturation_slider_visible,
    select_session_key,
)
from app.ui_budget_shifts import merge_budget_shifts_from_widgets
from app.ui_correlations import merge_correlations_from_widgets
from app.ui_helpers import apply_overrides
from app.ui_seasonality_panel import _read_basic_cycle_multipliers, warm_basic_cycle_editor_if_needed
from scripts.revenue_simulation.seasonality_fit import (
    fit_pattern_multipliers_to_fourier,
    sin_to_deterministic_fourier,
)


def _parse_float_or_default(raw: Any, default: float) -> float:
    val, ok = parse_optional_num(raw, as_int=False)
    if not ok or val is None:
        return float(default)
    return float(val)


def _parse_int_or_default(raw: Any, default: int) -> int:
    val, ok = parse_optional_num(raw, as_int=True)
    if not ok or val is None:
        return int(default)
    return int(val)


def _parse_pattern_csv(raw: Any, default: List[float]) -> List[float]:
    if raw is None:
        return list(default)
    s = str(raw).strip()
    if not s:
        return list(default)
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            return list(default)
    return out if out else list(default)


def _collect_seasonality_overrides(n_channels: int) -> List[Dict[str, Any]]:
    overrides: List[Dict[str, Any]] = []
    # Use .get so tests that monkeypatch session_state with a plain dict still work.
    base_cfg = st.session_state.get("sim_config", {})
    for i in range(n_channels):
        sea_type = str(st.session_state.get(f"sea_type_{i}", "none")).strip().lower()
        if sea_type in ("", "none"):
            overrides.append({"path": f"channel_list.{i}.channel.seasonality_config", "value": {}})
            continue

        if sea_type == "cycle":
            warm_basic_cycle_editor_if_needed(i, base_cfg)
            mults = _read_basic_cycle_multipliers(i)
            cfg = fit_pattern_multipliers_to_fourier(mults)
            overrides.append({"path": f"channel_list.{i}.channel.seasonality_config", "value": cfg or {}})
            continue

        if sea_type == "sin":
            cfg = sin_to_deterministic_fourier(
                {
                    "type": "sin",
                    "amplitude": _parse_float_or_default(st.session_state.get(f"sea_amp_{i}"), 0.2),
                    "period": _parse_int_or_default(st.session_state.get(f"sea_period_{i}"), 52),
                    "phase": _parse_float_or_default(st.session_state.get(f"sea_phase_{i}"), 0.0),
                }
            )
        elif sea_type == "fourier":
            cfg = {
                "type": "fourier",
                "period": _parse_int_or_default(st.session_state.get(f"sea_period_{i}"), 52),
                "K": _parse_int_or_default(st.session_state.get(f"sea_k_{i}"), 2),
                "scale": _parse_float_or_default(st.session_state.get(f"sea_scale_{i}"), 0.1),
            }
        elif sea_type == "categorical":
            pattern = _parse_pattern_csv(
                st.session_state.get(f"sea_pattern_{i}"), [1.0, 1.0, 1.0, 1.0]
            )
            cfg = fit_pattern_multipliers_to_fourier(pattern)
            if not cfg:
                cfg = {}
        else:
            cfg = {}
        overrides.append({"path": f"channel_list.{i}.channel.seasonality_config", "value": cfg})
    return overrides


def collect_overrides(schema: Dict[str, Any], n_channels: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    overrides: List[Dict[str, Any]] = []
    warnings: List[str] = []
    selects = list(schema.get("per_channel_selects", []))
    sat_select = next((s for s in selects if s.get("path") == "channel.saturation_config.type"), None)
    ad_select = next((s for s in selects if s.get("path") == "channel.adstock_decay_config.type"), None)
    sat_opts = list(sat_select.get("options", [])) if sat_select else []
    ad_opts = list(ad_select.get("options", [])) if ad_select else []
    base_cfg = st.session_state.sim_config

    for i in range(n_channels):
        eff_sat = effective_curve_type(i, "channel.saturation_config.type", base_cfg, sat_opts)
        eff_ad = effective_curve_type(i, "channel.adstock_decay_config.type", base_cfg, ad_opts)

        for item in schema.get("per_channel_sliders", []):
            if item.get("group") == "saturation" and not saturation_slider_visible(item, eff_sat):
                continue
            if item.get("group") == "adstock" and not adstock_slider_visible(item, eff_ad):
                continue
            path_suffix = item["path"]
            list_index = item.get("list_index")
            key = pc_field_key(i, path_suffix, list_index)
            raw = st.session_state.get(key, "")
            is_lag = path_suffix.endswith("lag") or "adstock_decay_config.lag" in path_suffix
            val, ok = parse_optional_num(raw, as_int=is_lag)
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
            wkey = adstock_weights_key(i)
            raw = st.session_state.get(wkey, "")
            wl, wok = parse_weights_csv(raw)
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
            key = select_session_key(i, path_suffix)
            if key not in st.session_state:
                continue
            full_path = f"channel_list.{i}.{path_suffix}"
            overrides.append({"path": full_path, "value": st.session_state[key]})

    overrides.extend(_collect_seasonality_overrides(n_channels))
    return overrides, warnings


def clear_channel_widget_keys() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith(("pc_", "sel_", "ch_name_", "del_ch_", "adw_", "ch_exp_", "sea_")):
            del st.session_state[k]
    clear_toggle_widget_keys()


def clear_widget_keys() -> None:
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
                "ch_exp_",
                "sea_",
            )
        ) or k.startswith(("corr_a_", "corr_b_", "corr_rho_", "corr_rm_")) or k.startswith(
            "bs_"
        ) or k in (
            "new_channel_name",
            "advanced_yaml",
            "corr_ui_rows",
            "corr_next_id",
            "corr_extra_option",
            "budget_shift_ui_rows",
            "budget_shift_next_id",
            "budget_shift_extra_option",
            "budget_shifts_auto_mode",
            "correlations_auto_mode",
        ):
            del st.session_state[k]
    clear_toggle_widget_keys()


def merge_ui_into_config(schema: Dict[str, Any], *, silent: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    """Merge widget state into a config dict.

    The second tuple element always lists merge warnings (invalid numbers,
    skipped rules, etc.). The ``silent`` keyword is ignored and kept only so
    older call sites can pass ``silent=True`` without changes.
    """
    merged = copy.deepcopy(st.session_state.sim_config)
    merged["week_range"] = int(st.session_state.get("week_range_num", 52))
    merged["run_identifier"] = str(st.session_state.get("run_identifier_input", "run")).strip() or "run"
    merged["seed"] = int(st.session_state.get("seed_input", 0))

    from app.ui_seasonality_panel import ensure_seasonality_widgets_warmed

    ensure_seasonality_widgets_warmed(merged)

    n_channels = len(merged.get("channel_list") or [])
    overrides, warns = collect_overrides(schema, n_channels)
    merged = apply_overrides(merged, overrides)

    for i in range(n_channels):
        key = f"ch_name_{i}"
        if key in st.session_state and i < len(merged["channel_list"]):
            nm = str(st.session_state[key]).strip()
            if nm:
                ch = merged["channel_list"][i].get("channel") or merged["channel_list"][i]
                if isinstance(ch, dict):
                    ch["channel_name"] = nm

    corrs, corr_warns = merge_correlations_from_widgets(merged)
    merged["correlations"] = corrs
    warns.extend(corr_warns)

    shifts, shift_warns = merge_budget_shifts_from_widgets(merged)
    merged["budget_shifts"] = shifts
    warns.extend(shift_warns)

    bs_mode = str(
        st.session_state.get("budget_shifts_auto_mode")
        or merged.get("budget_shifts_auto_mode")
        or st.session_state.get("budget_shift_extra_option")
        or "none"
    ).strip().lower()
    if bs_mode not in ("none", "global", "global_and_channel"):
        bs_mode = "none"
    merged["budget_shifts_auto_mode"] = bs_mode

    corr_mode = str(
        st.session_state.get("correlations_auto_mode")
        or merged.get("correlations_auto_mode")
        or st.session_state.get("corr_extra_option")
        or "none"
    ).strip().lower()
    if corr_mode not in ("none", "random"):
        corr_mode = "none"
    merged["correlations_auto_mode"] = corr_mode

    toggle_warns = merge_channel_toggles_into_config(merged)
    warns.extend(toggle_warns)

    _ = silent
    return merged, warns
