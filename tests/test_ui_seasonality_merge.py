"""
Tests for Streamlit seasonality merge helpers.

Run: python -m pytest tests/test_ui_seasonality_merge.py -v
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

from app import ui_config_merge as ucm
from scripts.revenue_simulation.seasonality_fit import (
    fit_pattern_multipliers_to_fourier,
    sin_to_deterministic_fourier,
)


def test_collect_seasonality_overrides_none_clears_config(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_state: Dict[str, Any] = {
        "sea_type_0": "none",
    }
    monkeypatch.setattr(ucm.st, "session_state", fake_state, raising=False)
    out = ucm._collect_seasonality_overrides(1)
    assert out == [{"path": "channel_list.0.channel.seasonality_config", "value": {}}]


def test_collect_seasonality_overrides_sin_builds_expected_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_state: Dict[str, Any] = {
        "sea_type_0": "sin",
        "sea_amp_0": 0.25,
        "sea_period_0": 26,
        "sea_phase_0": 2.0,
    }
    monkeypatch.setattr(ucm.st, "session_state", fake_state, raising=False)
    out = ucm._collect_seasonality_overrides(1)
    expected = sin_to_deterministic_fourier(
        {"type": "sin", "amplitude": 0.25, "period": 26, "phase": 2.0}
    )
    assert out == [{"path": "channel_list.0.channel.seasonality_config", "value": expected}]


def test_collect_seasonality_overrides_categorical_invalid_pattern_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_state: Dict[str, Any] = {
        "sea_type_0": "categorical",
        "sea_pattern_0": "1.0, bad, 0.9",
    }
    monkeypatch.setattr(ucm.st, "session_state", fake_state, raising=False)
    out = ucm._collect_seasonality_overrides(1)
    expected = fit_pattern_multipliers_to_fourier([1.0, 1.0, 1.0, 1.0])
    assert out == [{"path": "channel_list.0.channel.seasonality_config", "value": expected}]


def test_collect_seasonality_overrides_cycle_fits_table(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_state: Dict[str, Any] = {
        "sim_config": {"channel_list": [{"channel": {"channel_name": "A", "seasonality_config": {}}}]},
        "sea_type_0": "cycle",
        "sea_basic_df_0": pd.DataFrame(
            {"Week in cycle": [1, 2, 3], "Multiplier": [1.1, 1.0, 0.95]}
        ),
    }
    monkeypatch.setattr(ucm.st, "session_state", fake_state, raising=False)
    out = ucm._collect_seasonality_overrides(1)
    cfg = out[0]["value"]
    assert cfg.get("type") == "fourier"
    assert cfg.get("coefficients")
