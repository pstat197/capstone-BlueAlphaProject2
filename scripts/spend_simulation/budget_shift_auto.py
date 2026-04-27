"""Deterministic pseudo-random ``budget_shifts`` rules for reproducible scenarios."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def generate_auto_budget_shift_rules(
    week_range: int,
    channel_names: List[str],
    mode: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Build extra ``budget_shifts`` dicts to **append** after user-defined rules.

    ``mode`` is ``"global"`` (all-channel scale + bounded-window reallocate) or
    ``"global_and_channel"`` (adds ``scale_channel`` rules). Uses a dedicated RNG
    stream derived from ``seed`` so spend draws stay independent of this choice.
    """
    mode = str(mode or "").strip().lower()
    if mode not in ("global", "global_and_channel") or week_range < 1:
        return []

    names = [str(n).strip() for n in channel_names if str(n).strip()]
    if not names:
        return []

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xB05E11AF]))
    W = int(week_range)
    rules: List[Dict[str, Any]] = []

    def rand_window() -> tuple[int, int]:
        a = int(rng.integers(1, W + 1))
        b = int(rng.integers(a, W + 1))
        return a, b

    n_rules = int(rng.integers(2, 5)) if mode == "global" else int(rng.integers(3, 6))
    for _ in range(n_rules):
        if mode == "global_and_channel" and len(names) >= 1 and rng.random() < 0.45:
            ch = str(rng.choice(names))
            sw, ew = rand_window()
            factor = float(rng.uniform(0.88, 1.22))
            rules.append(
                {
                    "type": "scale_channel",
                    "channel_name": ch,
                    "start_week": sw,
                    "end_week": ew,
                    "factor": factor,
                }
            )
            continue

        if len(names) >= 2 and rng.random() < 0.55:
            idx = rng.permutation(len(names))[:2]
            f_name, t_name = names[int(idx[0])], names[int(idx[1])]
            sw = int(rng.integers(1, W + 1))
            max_span = max(1, W - sw + 1)
            span_cap = min(3, max_span)
            span = int(rng.integers(1, span_cap + 1))
            ew = min(sw + span - 1, W)
            frac = float(rng.uniform(0.12, 0.42))
            rules.append(
                {
                    "type": "reallocate",
                    "start_week": sw,
                    "end_week": ew,
                    "from_channel": f_name,
                    "to_channel": t_name,
                    "fraction": frac,
                }
            )
        else:
            sw, ew = rand_window()
            factor = float(rng.uniform(0.86, 1.18))
            rules.append({"type": "scale", "start_week": sw, "end_week": ew, "factor": factor})

    if mode == "global_and_channel" and names and not any(
        r.get("type") == "scale_channel" for r in rules
    ):
        ch = str(rng.choice(names))
        sw, ew = rand_window()
        factor = float(rng.uniform(0.88, 1.22))
        rules.append(
            {
                "type": "scale_channel",
                "channel_name": ch,
                "start_week": sw,
                "end_week": ew,
                "factor": factor,
            }
        )

    return rules
