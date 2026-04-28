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
    return _dedupe_corr_rows(rows)


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


def _max_distinct_pairs(n_channels: int) -> int:
    return n_channels * (n_channels - 1) // 2 if n_channels >= 2 else 0


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def _row_channel_ab(row: Dict[str, Any], names: List[str]) -> Tuple[str, str]:
    rid = int(row["id"])
    a = str(st.session_state.get(f"corr_a_{rid}", row.get("ch0", ""))).strip()
    b = str(st.session_state.get(f"corr_b_{rid}", row.get("ch1", ""))).strip()
    if a not in names:
        a = names[0]
    if b not in names:
        b = names[min(1, len(names) - 1)]
    return a, b


def _distinct_pairs_in_rows(rows: List[Dict[str, Any]], names: List[str]) -> set[Tuple[str, str]]:
    keys: set[Tuple[str, str]] = set()
    for r in rows:
        a, b = _row_channel_ab(r, names)
        if a and b and a != b:
            keys.add(_pair_key(a, b))
    return keys


def _other_row_pair_keys(
    rows: List[Dict[str, Any]], names: List[str], skip_rid: int
) -> set[Tuple[str, str]]:
    keys: set[Tuple[str, str]] = set()
    for r in rows:
        rid = int(r["id"])
        if rid == skip_rid:
            continue
        a, b = _row_channel_ab(r, names)
        if a and b and a != b:
            keys.add(_pair_key(a, b))
    return keys


def _dedupe_corr_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """One UI row per unordered pair; last row wins on ρ (matches merge)."""
    last: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        a = str(row.get("ch0", "")).strip()
        b = str(row.get("ch1", "")).strip()
        if not a or not b or a == b:
            continue
        last[_pair_key(a, b)] = row
    out: List[Dict[str, Any]] = []
    nid = 1
    for pk in sorted(last.keys()):
        r = last[pk]
        out.append(
            {
                "id": nid,
                "ch0": pk[0],
                "ch1": pk[1],
                "rho": float(r.get("rho", 0.0)),
            }
        )
        nid += 1
    return out, nid


def _reconcile_corr_row_choices(
    row: Dict[str, Any], names: List[str], other_keys: set[Tuple[str, str]]
) -> Tuple[List[str], List[str], str, str]:
    """Clamp widget state so (A,B) is not used on another row; returns (opts_a, opts_b, a, b)."""
    rid = int(row["id"])
    for _ in range(8):
        a_now, b_now = _row_channel_ab(row, names)
        opts_b = [m for m in names if m != a_now and _pair_key(a_now, m) not in other_keys]
        if not opts_b:
            moved = False
            for a_try in names:
                cand = [m for m in names if m != a_try and _pair_key(a_try, m) not in other_keys]
                if cand:
                    st.session_state[f"corr_a_{rid}"] = a_try
                    st.session_state[f"corr_b_{rid}"] = cand[0]
                    moved = True
                    break
            if moved:
                continue
            opts_b = [m for m in names if m != a_now]
        if b_now not in opts_b:
            st.session_state[f"corr_b_{rid}"] = opts_b[0]
            continue
        opts_a = [n for n in names if n != b_now and _pair_key(n, b_now) not in other_keys]
        if not opts_a:
            moved = False
            for b_try in names:
                cand = [n for n in names if n != b_try and _pair_key(n, b_try) not in other_keys]
                if cand:
                    st.session_state[f"corr_b_{rid}"] = b_try
                    st.session_state[f"corr_a_{rid}"] = cand[0]
                    moved = True
                    break
            if moved:
                continue
            opts_a = [n for n in names if n != b_now]
        if a_now not in opts_a:
            st.session_state[f"corr_a_{rid}"] = opts_a[0]
            continue
        return opts_a, opts_b, a_now, b_now
    a_now, b_now = _row_channel_ab(row, names)
    opts_b = [m for m in names if m != a_now and _pair_key(a_now, m) not in other_keys] or [
        m for m in names if m != a_now
    ]
    opts_a = [n for n in names if n != b_now and _pair_key(n, b_now) not in other_keys] or [
        n for n in names if n != b_now
    ]
    return opts_a, opts_b, a_now, b_now


def merge_correlations_from_widgets(
    merged: Dict[str, Any],
    *,
    silent: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build `correlations` YAML list from session widgets; returns (list, warnings)."""
    warns: List[str] = []
    n = len(merged.get("channel_list") or [])
    names_set = set(effective_channel_names(merged, n))
    rows: List[Dict[str, Any]] = list(st.session_state.get("corr_ui_rows") or [])
    ordered_keys: List[Tuple[str, str]] = []
    last_rho: Dict[Tuple[str, str], float] = {}

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
        if not (-1.0 <= rho <= 1.0):
            if not silent:
                warns.append(f"Correlation {a} / {b}: rho clipped to [-1, 1].")
            rho = max(-1.0, min(1.0, rho))
        pair_key = tuple(sorted((a, b)))
        if pair_key not in last_rho:
            ordered_keys.append(pair_key)
        elif not silent and abs(last_rho[pair_key] - rho) > 1e-12:
            warns.append(
                f"Duplicate pair {pair_key[0]} / {pair_key[1]}: using ρ={rho:.2f} from the lower row "
                f"(replaces earlier ρ={last_rho[pair_key]:.2f})."
            )
        last_rho[pair_key] = rho

    out = [{"channels": [k[0], k[1]], "rho": last_rho[k]} for k in ordered_keys]

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
        "Optional: joint weekly spend via a **Gaussian copula in log‑space** (then exponentiate and clip). "
        "The slider is **correlation of log‑spend**, not of dollar spend in the CSV—results charts use spend‑level ρ. "
        "Leave empty for independent channels (default). "
        "Each channel pair may appear **once**; duplicate entries in pasted YAML are collapsed to a single row (last ρ kept)."
    )

    st.caption(
        "**Extra correlation pairs from seed** are chosen under **Random append (same run seed)** above."
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
        other_keys = _other_row_pair_keys(rows, names, rid)
        opts_a, opts_b, a_now, b_now = _reconcile_corr_row_choices(row, names, other_keys)

        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            st.selectbox(
                "Channel A",
                options=opts_a,
                key=f"corr_a_{rid}",
                index=opts_a.index(a_now) if a_now in opts_a else 0,
                label_visibility="collapsed",
                on_change=_yaml_sync_from_form,
            )
        with c2:
            st.selectbox(
                "Channel B",
                options=opts_b,
                key=f"corr_b_{rid}",
                index=opts_b.index(b_now) if b_now in opts_b else 0,
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
                help="Correlation in the **log‑spend** Gaussian copula (−1…1). Dollar spend in the output will not match this ρ exactly.",
                on_change=_yaml_sync_from_form,
            )
        with c4:
            st.write("")
            if st.button("Remove", key=f"corr_rm_{rid}", width="stretch"):
                st.session_state.corr_ui_rows = [r for r in st.session_state.corr_ui_rows if r["id"] != rid]
                for k in (f"corr_a_{rid}", f"corr_b_{rid}", f"corr_rho_{rid}", f"corr_rm_{rid}"):
                    st.session_state.pop(k, None)
                st.rerun()

    max_pairs = _max_distinct_pairs(n_channels)
    distinct_pairs = _distinct_pairs_in_rows(rows, names)
    if len(distinct_pairs) < max_pairs and st.button("Add correlated pair", width="content"):
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
