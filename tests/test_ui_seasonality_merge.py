"""
Tests for Streamlit seasonality merge helpers.

Run: python -m pytest tests/test_ui_seasonality_merge.py -v
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from app import ui_config_merge as ucm


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
    assert out == [
        {
            "path": "channel_list.0.channel.seasonality_config",
            "value": {"type": "sin", "amplitude": 0.25, "period": 26, "phase": 2.0},
        }
    ]


def test_collect_seasonality_overrides_categorical_invalid_pattern_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_state: Dict[str, Any] = {
        "sea_type_0": "categorical",
        "sea_pattern_0": "1.0, bad, 0.9",
    }
    monkeypatch.setattr(ucm.st, "session_state", fake_state, raising=False)
    out = ucm._collect_seasonality_overrides(1)
    assert out == [
        {
            "path": "channel_list.0.channel.seasonality_config",
            "value": {"type": "categorical", "pattern": [1.0, 1.0, 1.0, 1.0]},
        }
    ]
