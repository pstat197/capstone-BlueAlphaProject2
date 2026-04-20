"""Streamlit UI for optional pairwise spend correlations (YAML `correlations` block)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import streamlit as st


def _yaml_sync_from_form() -> None:
    st.session_state["yaml_manual_edit"] = False


def _ch_at(cfg: Dict[str, Any], index: int) -> Dict[str, Any]:
    lst = cfg.get("channel_list") or []
    if index >= len(lst):
        return {}
    item = lst[index]
    return item.get("channel") or item if isinstance(item, dict) else {}


def effective_channel_names(cfg: Dict[str, Any], n: int) -> List[str]:
    """Channel names as in merged config (includes rename widgets)."""
    names: List[str] = []
    for i in range(n):
        key = f"ch_name_{i}"
        if key in st.session_state:
            nm = str(st.session_state[key]).strip()
            if nm:
                names.append(nm)
                continue
        ch = _ch_at(cfg, i)
        nm = ch.get("channel_name") if isinstance(ch, dict) else None
        names.append(str(nm).strip() if nm else f"Channel {i + 1}")
    return names


def _rows_from_cfg_correlations(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    raw = cfg.get("correlations") or []
    rows: List[Dict[str, Any]] = []
    next_id = 1
    for e in raw:
        if not isinstance(e, dict):
            continue
        ch = e.get("channels") or []
        if len(ch) != 2:
            continue
        rows.append(
            {
                "id": next_id,
                "ch0": str(ch[0]),
                "ch1": str(ch[1]),
                "rho": float(e.get("rho", 0.0)),
            }
        )
        next_id += 1
    return rows, next_id


def ensure_corr_rows_initialized(cfg: Dict[str, Any]) -> None:
    if "corr_ui_rows" in st.session_state:
        return
    rows, next_id = _rows_from_cfg_correlations(cfg)
    st.session_state.corr_ui_rows = rows
    st.session_state.corr_next_id = next_id


def clear_correlation_keys() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith(("corr_a_", "corr_b_", "corr_rho_", "corr_rm_")):
            del st.session_state[k]
    for k in ("corr_ui_rows", "corr_next_id"):
        if k in st.session_state:
            del st.session_state[k]


def merge_correlations_from_widgets(
    merged: Dict[str, Any],
    *,
    silent: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build `correlations` YAML list from session widgets; returns (list, warnings)."""
    warns: List[str] = []
    n = len(merged.get("channel_list") or [])
    names_set = set(effective_channel_names(merged, n))
    out: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = list(st.session_state.get("corr_ui_rows") or [])
    seen_pairs: set[Tuple[str, str]] = set()

    for row in rows:
        rid = row["id"]
        a = str(st.session_state.get(f"corr_a_{rid}", row.get("ch0", ""))).strip()
        b = str(st.session_state.get(f"corr_b_{rid}", row.get("ch1", ""))).strip()
        rho = float(st.session_state.get(f"corr_rho_{rid}", row.get("rho", 0.0)))
        if not a or not b:
            if not silent:
                warns.append(f"Correlation row {rid}: both channels must be selected; skipped.")
            continue
        if a == b:
            if not silent:
                warns.append(f"Correlation {a}/{b}: channels must differ; skipped.")
            continue
        if a not in names_set or b not in names_set:
            if not silent:
                warns.append(f"Correlation {a} / {b}: unknown channel name; skipped.")
            continue
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs:
            if not silent:
                warns.append(f"Duplicate correlation pair {a} / {b}; skipped duplicate.")
            continue
        seen_pairs.add(pair)
        if not (-1.0 <= rho <= 1.0):
            if not silent:
                warns.append(f"Correlation {a} / {b}: rho clipped to [-1, 1].")
            rho = max(-1.0, min(1.0, rho))
        out.append({"channels": [a, b], "rho": rho})

    return out, warns


def apply_correlations_to_session_rows(cfg: Dict[str, Any]) -> None:
    """After YAML load or reset: replace UI rows from config."""
    clear_correlation_keys()
    rows, next_id = _rows_from_cfg_correlations(cfg)
    st.session_state.corr_ui_rows = rows
    st.session_state.corr_next_id = next_id


def render_correlations_section(cfg: Dict[str, Any], n_channels: int) -> None:
    ensure_corr_rows_initialized(cfg)

    st.markdown("##### Correlated channel spend")
    st.caption(
        "Optional: joint weekly spend across pairs (Gaussian copula in log space). "
        "Leave empty for independent channels (default)."
    )

    names = effective_channel_names(cfg, n_channels)
    if n_channels < 2:
        st.info("Add at least two channels to configure spend correlations.")
        return

    rows: List[Dict[str, Any]] = st.session_state.corr_ui_rows
    if rows:
        h0, h1, h2, h3 = st.columns([2, 2, 2, 1])
        h0.caption("Channel A")
        h1.caption("Channel B")
        h2.caption("Correlation ρ")
        h3.caption(" ")

    for row in list(rows):
        rid = int(row["id"])
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            st.selectbox(
                "Channel A",
                options=names,
                key=f"corr_a_{rid}",
                index=names.index(row["ch0"]) if row["ch0"] in names else 0,
                label_visibility="collapsed",
                on_change=_yaml_sync_from_form,
            )
        with c2:
            st.selectbox(
                "Channel B",
                options=names,
                key=f"corr_b_{rid}",
                index=names.index(row["ch1"]) if row["ch1"] in names else min(1, len(names) - 1),
                label_visibility="collapsed",
                on_change=_yaml_sync_from_form,
            )
        with c3:
            st.slider(
                "ρ",
                min_value=-1.0,
                max_value=1.0,
                value=float(row.get("rho", 0.0)),
                step=0.05,
                key=f"corr_rho_{rid}",
                help="Target Pearson correlation between the two channels' weekly spend.",
                on_change=_yaml_sync_from_form,
            )
        with c4:
            st.write("")
            if st.button("Remove", key=f"corr_rm_{rid}", use_container_width=True):
                st.session_state.corr_ui_rows = [r for r in st.session_state.corr_ui_rows if r["id"] != rid]
                for k in (f"corr_a_{rid}", f"corr_b_{rid}", f"corr_rho_{rid}", f"corr_rm_{rid}"):
                    st.session_state.pop(k, None)
                st.rerun()

    if st.button("Add correlated pair", use_container_width=False):
        nid = int(st.session_state.get("corr_next_id", 1))
        st.session_state.corr_next_id = nid + 1
        st.session_state.corr_ui_rows = list(st.session_state.corr_ui_rows) + [
            {
                "id": nid,
                "ch0": names[0],
                "ch1": names[1] if len(names) > 1 else names[0],
                "rho": 0.0,
            }
        ]
        st.rerun()
