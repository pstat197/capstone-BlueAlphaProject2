"""
Random noise for config generation. Variance scales with default value size.
σ = max(α|d|, ε); value = d + N(0, σ²).
Replace or wrap this function to change noise behavior.
"""
from typing import Optional

import numpy as np

# Default RNG; can be overridden for tests or initialized from config seed
_default_rng: Optional[np.random.Generator] = None


def get_default_rng() -> np.random.Generator:
    global _default_rng
    if _default_rng is None:
        _default_rng = np.random.default_rng()
    return _default_rng


def init_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Initialize the default RNG with the given seed.
    If seed is None, use a random seed (from entropy) so each run can differ but is reproducible within the run.
    Returns the RNG instance.
    """
    global _default_rng
    if seed is not None:
        _default_rng = np.random.default_rng(int(seed))
    else:
        _default_rng = np.random.default_rng()
    return _default_rng


def set_default_rng(rng: Optional[np.random.Generator]) -> None:
    """Set the global RNG (e.g. for tests with a fixed seed)."""
    global _default_rng
    _default_rng = rng


def add_noise_to_value(
    default: float,
    alpha: float = 0.1,
    epsilon_floor: float = 1e-9,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Return default + ε where ε ~ N(0, σ²), σ = max(α|d|, ε_floor).
    So larger defaults get proportionally larger randomness.
    """
    if rng is None:
        rng = get_default_rng()
    d = float(default)
    sigma = max(alpha * abs(d), epsilon_floor)
    return d + rng.normal(0, sigma)
