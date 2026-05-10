"""
Deterministic Fourier seasonality helpers.

- Fit repeating multiplier patterns to smoothed Fourier series (least squares).
- Convert single-harmonic sin specs to equivalent deterministic Fourier.
- Normalize YAML/UI configs so categorical/sin inputs become type=fourier with
  ``intercept`` + ``coefficients``; random Fourier (``scale``/``K``, no coefficients) passes through unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np


def _coerce_pattern(pattern: Any) -> Optional[np.ndarray]:
    if pattern is None:
        return None
    if isinstance(pattern, (list, tuple)):
        try:
            arr = np.asarray([float(x) for x in pattern], dtype=float)
        except (TypeError, ValueError):
            return None
        if arr.size == 0:
            return None
        return arr
    return None


def _choose_K_smooth(pattern_len: int) -> int:
    """Cap harmonics for a smoother fit than full-rank interpolation."""
    if pattern_len <= 1:
        return 1
    return min(8, max(1, pattern_len // 3))


def fit_pattern_multipliers_to_fourier(
    pattern: List[float],
    *,
    K: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fit one cycle of multipliers m[0..P-1] (e.g. categorical pattern) to:

        m_hat(t) = 1 + c0 + sum_{k=1..K} a_k sin(2πkt/P) + b_k cos(2πkt/P)

    Targets y = m - 1 (same deviation convention as categorical seasonality).

    Returns a ``seasonality_config`` dict with type fourier, ``period``,
    ``K``, ``intercept`` (= c0), and ``coefficients`` as [[a1,b1], ...].
    """
    m = np.asarray(pattern, dtype=float)
    if m.size < 2:
        return {}
    # One full cycle is the pattern length (repeating multipliers).
    P = int(m.size)
    y = m - 1.0
    t = np.arange(P, dtype=float)
    nyquist_cap = max(1, P // 2)
    K_chosen = int(K) if K is not None else _choose_K_smooth(P)
    K_use = max(1, min(K_chosen, nyquist_cap))

    cols = [np.ones(P, dtype=float)]
    for k in range(1, K_use + 1):
        ang = 2.0 * np.pi * k * t / float(P)
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept = float(coef[0])
    coeff_rows: List[List[float]] = []
    for j in range(K_use):
        coeff_rows.append([float(coef[1 + 2 * j]), float(coef[1 + 2 * j + 1])])

    return {
        "type": "fourier",
        "period": P,
        "K": K_use,
        "intercept": intercept,
        "coefficients": coeff_rows,
    }


def sin_to_deterministic_fourier(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Exact equivalence: one harmonic Fourier for sin(amplitude, period, phase)."""
    amplitude = float(cfg.get("amplitude", 0.2))
    period = int(cfg.get("period", 52))
    phase = float(cfg.get("phase", 0.0))
    if period < 1:
        period = 52
    # sin(2π(t+φ)/P) = sin(2πt/P)cos(2πφ/P) + cos(2πt/P)sin(2πφ/P)
    a1 = amplitude * float(np.cos(2.0 * np.pi * phase / float(period)))
    b1 = amplitude * float(np.sin(2.0 * np.pi * phase / float(period)))
    return {
        "type": "fourier",
        "period": period,
        "K": 1,
        "intercept": 0.0,
        "coefficients": [[a1, b1]],
    }


def normalize_seasonality_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return a config safe for simulation: categorical and sin become deterministic
    fourier; empty dict unchanged; random fourier (coefficients absent) unchanged.
    """
    if not cfg:
        return {}
    if not isinstance(cfg, dict):
        return {}
    st = str(cfg.get("type", "")).strip().lower()
    if st == "categorical":
        pat = _coerce_pattern(cfg.get("pattern"))
        if pat is None or pat.size < 2:
            return {}
        return fit_pattern_multipliers_to_fourier(pat.tolist())
    if st == "sin":
        return sin_to_deterministic_fourier(cfg)
    if st == "fourier":
        # Deterministic path if coefficients present; else random-draw Fourier.
        if cfg.get("coefficients") is not None:
            out = dict(cfg)
            out["type"] = "fourier"
            out["period"] = int(out.get("period", 52))
            out["K"] = int(out.get("K", len(out.get("coefficients") or []) or 1))
            out["intercept"] = float(out.get("intercept", 0.0))
            return out
        return dict(cfg)
    # hybrid and others: pass through (hybrid may still contain categorical inside)
    return dict(cfg)


def evaluate_deterministic_fourier(t: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Evaluate 1 + intercept + sum_k a_k sin(2πkt/P) + b_k cos(2πkt/P)."""
    t = np.asarray(t, dtype=float)
    period = int(cfg.get("period", 52))
    if period < 1:
        period = 52
    intercept = float(cfg.get("intercept", 0.0))
    coeffs = cfg.get("coefficients") or []
    s = np.full_like(t, intercept, dtype=float)
    for k, pair in enumerate(coeffs, start=1):
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        a_k = float(pair[0])
        b_k = float(pair[1])
        ang = 2.0 * np.pi * k * t / float(period)
        s += a_k * np.sin(ang) + b_k * np.cos(ang)
    return 1.0 + s
