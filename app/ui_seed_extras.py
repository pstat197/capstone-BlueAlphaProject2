"""Always-visible controls for seed-append modes (persisted in YAML, expanded at load time).

``budget_shifts_auto_mode`` and ``correlations_auto_mode`` are written into the merged config /
``sim_config`` so Advanced YAML and the post-run snapshot show **how** extras are requested.
:class:`~scripts.config.loader.load_config_from_dict` expands them into concrete ``budget_shifts``
and ``correlations`` using the same **Random seed** (deterministic).

Streamlit only executes the selected tab each run; these controls stay **above** the tabs.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from app.ui_channel_form import yaml_sync_from_form


def _normalize_bs_mode(raw: Any) -> str:
    s = str(raw or "none").strip().lower()
    return s if s in ("none", "global", "global_and_channel") else "none"


def _normalize_corr_mode(raw: Any) -> str:
    s = str(raw or "none").strip().lower()
    return s if s in ("none", "random") else "none"


def sync_seed_extra_modes_from_cfg(cfg: Dict[str, Any]) -> None:
    """Align widget session keys with ``sim_config`` (Apply YAML / reset / Edit configuration)."""
    st.session_state["budget_shifts_auto_mode"] = _normalize_bs_mode(
        cfg.get("budget_shifts_auto_mode") or st.session_state.get("budget_shift_extra_option")
    )
    st.session_state["correlations_auto_mode"] = _normalize_corr_mode(
        cfg.get("correlations_auto_mode") or st.session_state.get("corr_extra_option")
    )


def render_seed_extra_controls() -> None:
    cfg: Dict[str, Any] = st.session_state.get("sim_config") or {}
    st.session_state.setdefault(
        "budget_shifts_auto_mode",
        _normalize_bs_mode(cfg.get("budget_shifts_auto_mode")),
    )
    st.session_state.setdefault(
        "correlations_auto_mode",
        _normalize_corr_mode(cfg.get("correlations_auto_mode")),
    )

    st.markdown("##### Random append (same run seed)")
    st.caption(
        "Saved in YAML as **`budget_shifts_auto_mode`** and **`correlations_auto_mode`** next to your "
        "manual **`budget_shifts`** / **`correlations`** lists. The loader expands them on each run "
        "using **Random seed** — same file, same seed, same effective simulation."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox(
            "Extra budget shifts",
            options=["none", "global", "global_and_channel"],
            format_func=lambda x: {
                "none": "None — manual rules only",
                "global": "Global — random scales + bounded reallocates",
                "global_and_channel": "Global + per-channel — also random single-channel scales",
            }[x],
            key="budget_shifts_auto_mode",
            on_change=yaml_sync_from_form,
            help="YAML key: budget_shifts_auto_mode",
        )
    with c2:
        st.selectbox(
            "Extra correlation pairs",
            options=["none", "random"],
            format_func=lambda x: {
                "none": "None — manual rows only",
                "random": "Random — append reproducible ρ (manual wins on duplicate pairs)",
            }[x],
            key="correlations_auto_mode",
            on_change=yaml_sync_from_form,
            help="YAML key: correlations_auto_mode",
        )
