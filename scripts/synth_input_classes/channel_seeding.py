"""Order-independent seed mixing for per-channel auxiliary RNG branches.

``channel_list`` order in YAML must not change stochastic realizations for the
same ``channel_name`` and run ``seed``. Integer list positions are replaced by
a stable 32-bit word derived from the UTF-8 channel name (SHA-256 prefix).
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

# Magic words separate SeedSequence namespaces (sticky vs seasonality vs …).
_STICKY_PAUSE_TAG = 0x53544B59
_SEASONALITY_FALLBACK_TAG = 0x5EA50A1


def channel_name_seed_u32(channel_name: str) -> int:
    """First four bytes of SHA-256(channel name) as a little-endian uint32."""
    digest = hashlib.sha256(str(channel_name).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def sticky_pause_seed_sequence(
    config_seed: Optional[int], channel_name: str
) -> np.random.SeedSequence:
    seed_part = 0 if config_seed is None else int(config_seed)
    return np.random.SeedSequence(
        [seed_part, int(channel_name_seed_u32(channel_name)), _STICKY_PAUSE_TAG]
    )


def outcome_seasonality_fallback_seed(
    run_seed: Optional[int],
    channel_name: Optional[str] = None,
) -> int:
    """
    Deterministic seed for random Fourier / hybrid stochastic seasonality on the
    outcome path.

    When ``channel_name`` is set (implicit outcome from a ``channel_list`` row),
    the mix is ``(run_seed, hash(name), 0x5EA50A1)``. When it is omitted or blank
    (explicit top-level ``outcome_revenue``), the mix is ``(run_seed, 0x0FFC0DE)``
    so existing configs keep the same draws as before that branch existed.
    """
    seed_part = 0 if run_seed is None else int(run_seed)
    if channel_name is None or not str(channel_name).strip():
        return int(
            np.random.SeedSequence([seed_part, 0x0FFC0DE]).generate_state(1)[0]
        )
    nm = str(channel_name).strip()
    seq = np.random.SeedSequence(
        [seed_part, int(channel_name_seed_u32(nm)), _SEASONALITY_FALLBACK_TAG]
    )
    return int(seq.generate_state(1)[0])
