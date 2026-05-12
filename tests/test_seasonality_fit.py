"""Tests for deterministic Fourier fitting and normalization."""
from __future__ import annotations

import numpy as np

from scripts.revenue_simulation.revenue_generation import _seasonality
from scripts.revenue_simulation.seasonality_fit import (
    fit_pattern_multipliers_to_fourier,
    normalize_seasonality_config,
    sin_to_deterministic_fourier,
)


def test_sin_normalizes_to_fourier_matching_sin_formula() -> None:
    t = np.arange(0, 52, dtype=float)
    sin_cfg = {"type": "sin", "amplitude": 0.15, "period": 52, "phase": 3.0}
    f_cfg = normalize_seasonality_config(sin_cfg)
    np.testing.assert_allclose(
        _seasonality(t, sin_cfg),
        _seasonality(t, f_cfg),
        rtol=0,
        atol=1e-9,
    )


def test_categorical_normalizes_to_close_fit_for_two_level_pattern() -> None:
    t = np.arange(0, 8, dtype=float)
    cat = {"type": "categorical", "pattern": [1.0, 2.0]}
    f_cfg = normalize_seasonality_config(cat)
    assert f_cfg.get("type") == "fourier"
    assert "coefficients" in f_cfg
    np.testing.assert_allclose(
        _seasonality(t, cat),
        _seasonality(t, f_cfg),
        rtol=0,
        atol=1e-9,
    )


def test_fit_pattern_preserves_alternating_pattern_at_integers() -> None:
    cfg = fit_pattern_multipliers_to_fourier([1.0, 2.0])
    t = np.array([0.0, 1.0, 2.0, 3.0])
    m = _seasonality(t, cfg)
    np.testing.assert_allclose(m, [1.0, 2.0, 1.0, 2.0], rtol=0, atol=1e-6)


def test_sin_to_fourier_analytic_coefficients() -> None:
    cfg = sin_to_deterministic_fourier(
        {"type": "sin", "amplitude": 0.25, "period": 26, "phase": 2.0}
    )
    assert cfg["type"] == "fourier"
    assert cfg["period"] == 26
    assert cfg["K"] == 1
    a1, b1 = cfg["coefficients"][0]
    exp_a = 0.25 * float(np.cos(2.0 * np.pi * 2.0 / 26.0))
    exp_b = 0.25 * float(np.sin(2.0 * np.pi * 2.0 / 26.0))
    assert abs(a1 - exp_a) < 1e-12 and abs(b1 - exp_b) < 1e-12
