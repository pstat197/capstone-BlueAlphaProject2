"""Deterministic pseudo-random pairwise ``correlations`` entries (log-spend copula ρ)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def generate_auto_correlation_entries(channel_names: List[str], seed: int) -> List[Dict[str, Any]]:
    """
    Build extra ``correlations`` YAML-style dicts to **append** after manual pairs.

    Uses a dedicated RNG stream derived from ``seed`` so spend sampling stays
    independent. Draws a random count of **distinct** unordered channel pairs
    and assigns each a ρ in roughly ``[-0.92, 0.92]`` (log-spend Gaussian copula).
    """
    names = sorted({str(x).strip() for x in channel_names if str(x).strip()})
    n = len(names)
    if n < 2:
        return []

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC0FFA1AB]))
    all_pairs: List[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((names[i], names[j]))

    rng.shuffle(all_pairs)
    max_extra = min(3, len(all_pairs))
    n_extra = int(rng.integers(1, max_extra + 1))
    picked = all_pairs[:n_extra]

    out: List[Dict[str, Any]] = []
    for a, b in picked:
        rho = float(rng.uniform(-0.92, 0.92))
        out.append({"channels": [a, b], "rho": rho})
    return out
