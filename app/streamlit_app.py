"""
Streamlit entry: marketing simulator UI.

Run from repository root:
  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Optional

import streamlit as st
import yaml

# Repo root must be on sys.path before any `from app.*` (Streamlit Cloud cwd is not always the repo).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.cache import clear_run_cache, run_with_cache  # noqa: E402
from app.default_channel import default_channel_dict  # noqa: E402
from app.pipeline_runner import run_pipeline  # noqa: E402
from app.theme import inject_theme_css  # noqa: E402
from app.ui_channel_form import (  # noqa: E402
    render_channel_widgets,
    yaml_mark_dirty,
    yaml_sync_from_form,
)
from app.ui_config_merge import (  # noqa: E402
    clear_channel_widget_keys,
    clear_widget_keys,
    merge_ui_into_config,
)
from app.ui_correlations import (  # noqa: E402
    apply_correlations_to_session_rows,
    ensure_corr_rows_initialized,
    render_correlations_section,
)
from app.ui_results import render_results_panel  # noqa: E402
from app.ui_yaml_io import load_example_text, load_ui_schema, yaml_dump  # noqa: E402


def _existing_channel_names(cfg: dict) -> list[str]:
    """Channel names currently in ``cfg.channel_list`` (ignores blanks)."""
    out: list[str] = []
    for item in cfg.get("channel_list") or []:
        ch = item.get("channel") if isinstance(item, dict) else None
        if not isinstance(ch, dict):
            ch = item if isinstance(item, dict) else {}
        nm = ch.get("channel_name")
        if isinstance(nm, str) and nm.strip():
            out.append(nm.strip())
    return out


def _next_unique_channel_name(base: str, existing: list[str]) -> str:
    """Return ``base`` if unused, else ``base 2``, ``base 3``, … (case-insensitive)."""
    taken = {n.lower() for n in existing}
    if base.lower() not in taken:
        return base
    i = 2
    while f"{base} {i}".lower() in taken:
        i += 1
    return f"{base} {i}"


def _resync_form_from_sim_config() -> None:
    """Reset form widget state so it mirrors the current ``sim_config``.

    Used by both **Edit configuration** (after a run) and **Apply YAML to form**
    so the two entry points stay consistent.
    """
    clear_channel_widget_keys()
    apply_correlations_to_session_rows(st.session_state.sim_config)
    st.session_state["_sync_top_widgets_from_sim_config"] = True
    st.session_state["pending_yaml_dump"] = yaml_dump(st.session_state.sim_config)
    st.session_state["yaml_manual_edit"] = False


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
        st.session_state.colorblind_charts = True

    inject_theme_css(night=bool(st.session_state.night_mode))

    schema = load_ui_schema()

    if "sim_config" not in st.session_state:
        st.session_state.sim_config = yaml.safe_load(load_example_text()) or {}
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

    ensure_corr_rows_initialized(st.session_state.sim_config)

    if st.session_state.pop("_resync_form_from_sim_config", False):
        _resync_form_from_sim_config()

    if st.session_state.pop("_sync_top_widgets_from_sim_config", False):
        st.session_state.week_range_num = int(st.session_state.sim_config.get("week_range", 52))
        st.session_state.run_identifier_input = str(
            st.session_state.sim_config.get("run_identifier", "run")
        )
        s = st.session_state.sim_config.get("seed")
        st.session_state.seed_input = int(s) if s is not None else 0

    pending_yaml = st.session_state.pop("pending_yaml_dump", None)
    if pending_yaml is not None:
        st.session_state.advanced_yaml = pending_yaml
        st.session_state.yaml_manual_edit = False
    elif not st.session_state.get("yaml_manual_edit", False):
        merged_preview, _ = merge_ui_into_config(schema, silent=True)
        st.session_state.advanced_yaml = yaml_dump(merged_preview)
    elif "advanced_yaml" not in st.session_state:
        st.session_state.advanced_yaml = yaml_dump(st.session_state.sim_config)

    with st.sidebar:
        st.header("Settings")
        st.checkbox("Night mode", key="night_mode")
        st.checkbox(
            "Colorblind-safe chart colors",
            key="colorblind_charts",
            help="Uses orange / blue / green lines (Wong-style) so series are easier to distinguish.",
        )

        st.divider()
        if st.button("Reset to example.yaml", width="stretch"):
            st.session_state.sim_config = yaml.safe_load(load_example_text()) or {}
            st.session_state.config_collapsed = False
            clear_widget_keys()
            st.session_state.week_range_num = int(st.session_state.sim_config.get("week_range", 52))
            st.session_state.run_identifier_input = str(
                st.session_state.sim_config.get("run_identifier", "run")
            )
            s = st.session_state.sim_config.get("seed")
            st.session_state.seed_input = int(s) if s is not None else 0
            st.session_state["pending_yaml_dump"] = yaml_dump(st.session_state.sim_config)
            st.session_state.yaml_manual_edit = False
            st.rerun()

        if st.button("Clear simulation cache", width="stretch", help="Remove cached runs on disk"):
            n = clear_run_cache()
            st.caption(f"Cleared {n} cached file(s).")

    user_dict = st.session_state.sim_config
    n_channels = len(user_dict.get("channel_list") or [])

    df = st.session_state.get("last_df")
    show_charts = df is not None

    if st.session_state.config_collapsed and show_charts:
        render_results_panel(df, compact_toolbar=True)
        return

    st.title("Marketing mix simulator")
    st.caption("Configure the simulation below, then run.")

    st.markdown("##### Simulation settings")
    st.caption(
        "Length and labeling for each run. **Random seed** fixes sampling so the same settings reproduce "
        "the same series and cache behavior."
    )
    col_w, col_r, col_s = st.columns(3)
    with col_w:
        st.number_input(
            "Week range",
            min_value=1,
            max_value=None,
            step=1,
            key="week_range_num",
            help="Simulation length in weeks.",
            on_change=yaml_sync_from_form,
        )
    with col_r:
        st.text_input(
            "Run Name",
            key="run_identifier_input",
            placeholder="e.g. Example Alpha",
            help="Labels CSV downloads and run summaries.",
            on_change=yaml_sync_from_form,
        )
    with col_s:
        st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            step=1,
            key="seed_input",
            help="Fixes randomness so runs and cache keys are reproducible.",
            on_change=yaml_sync_from_form,
        )

    st.markdown("##### Channels")
    st.caption(
        "Each row is one media channel with its own spend, response curve, carry-over, and noise. "
        "Open a channel; optional **reference** expanders under Noise, Saturation, and Adstock explain formulas and when to use each option."
    )
    row_a, row_b = st.columns([4, 1])
    with row_a:
        new_nm = st.text_input(
            "Name for new channel",
            key="new_channel_name",
            placeholder="e.g. TikTok",
            on_change=yaml_sync_from_form,
        )
    with row_b:
        st.write("")
        if st.button("Add channel", width="stretch"):
            base = (new_nm or "").strip() or "New channel"
            existing = _existing_channel_names(st.session_state.sim_config)
            nm = _next_unique_channel_name(base, existing)
            ch = default_channel_dict()
            ch["channel_name"] = nm
            if "channel_list" not in st.session_state.sim_config:
                st.session_state.sim_config["channel_list"] = []
            st.session_state.sim_config["channel_list"].append({"channel": ch})
            st.session_state["yaml_manual_edit"] = False
            clear_channel_widget_keys()
            st.rerun()

    if n_channels > 0:
        render_channel_widgets(schema, user_dict, n_channels)
    else:
        st.info("Add at least one channel to run the simulation.")

    render_correlations_section(user_dict, n_channels)

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
            on_change=yaml_mark_dirty,
        )
        if st.button("Apply YAML to form", type="secondary"):
            try:
                parsed = yaml.safe_load(st.session_state.advanced_yaml)
                if not isinstance(parsed, dict):
                    raise ValueError("YAML must parse to a mapping (dict).")
                st.session_state.sim_config = parsed
                _resync_form_from_sim_config()
                st.success("YAML applied.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not apply YAML: {e}")

    run_clicked = st.button("Run simulation", type="primary", width="content")

    if run_clicked:
        try:
            merged, warns = merge_ui_into_config(schema)
            for w in warns:
                st.warning(w)
            if not (merged.get("channel_list") or []):
                raise ValueError("Add at least one channel before running.")
            df_out, run_id, cache_hit, cfg_hash, corr_results = run_with_cache(merged, run_pipeline)
            st.session_state["last_df"] = df_out
            st.session_state["last_run_id"] = run_id
            st.session_state["last_cache_hit"] = cache_hit
            st.session_state["last_hash"] = cfg_hash
            st.session_state["last_corr_results"] = corr_results
            st.session_state["last_error"] = None
            st.session_state.sim_config = copy.deepcopy(merged)
            st.session_state["pending_yaml_dump"] = yaml_dump(merged)
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
        render_results_panel(df2, compact_toolbar=False)


if __name__ == "__main__":
    main()
