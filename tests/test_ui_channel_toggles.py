"""
Tests for unified pause rules in ``app.ui_channel_toggles`` (YAML helpers + merge).

Run: python -m pytest tests/test_ui_channel_toggles.py -v
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from app import ui_channel_toggles as uct


def test_pause_rows_from_yaml_fixed_then_sticky_order() -> None:
    ch: Dict[str, Any] = {
        "enabled": {
            "default": True,
            "off_ranges": [{"start_week": 2, "end_week": 3}],
        },
        "sticky_pause_ranges": [
            {
                "start_week": 5,
                "end_week": 6,
                "start_probability": 0.5,
                "continue_probability": 0.9,
            },
        ],
    }
    rows, next_id = uct._pause_rows_from_channel_yaml(ch, week_range=10)
    assert len(rows) == 2
    assert rows[0]["kind"] == "fixed"
    assert rows[0]["id"] == 1
    assert rows[0]["start"] == 2 and rows[0]["end"] == 3
    assert rows[1]["kind"] == "sticky"
    assert rows[1]["id"] == 2
    assert rows[1]["start"] == 5 and rows[1]["end"] == 6
    assert rows[1]["p_start"] == 0.5 and rows[1]["p_continue"] == 0.9
    assert next_id == 3


def test_pause_rows_from_yaml_disabled_channel_skips_fixed_keeps_sticky() -> None:
    ch: Dict[str, Any] = {
        "enabled": False,
        "sticky_pause_ranges": [
            {
                "start_week": 1,
                "end_week": 2,
                "start_probability": 0.1,
                "continue_probability": 0.2,
            },
        ],
    }
    rows, next_id = uct._pause_rows_from_channel_yaml(ch, week_range=8)
    assert len(rows) == 1
    assert rows[0]["kind"] == "sticky"
    assert next_id == 2


def test_pause_rows_from_yaml_clamps_weeks() -> None:
    ch: Dict[str, Any] = {
        "enabled": {"default": True, "off_ranges": [{"start_week": 1, "end_week": 99}]},
    }
    rows, _ = uct._pause_rows_from_channel_yaml(ch, week_range=5)
    assert rows[0]["start"] == 1 and rows[0]["end"] == 5


def test_collect_channel_toggle_splits_fixed_and_sticky(monkeypatch: pytest.MonkeyPatch) -> None:
    """Merge path: one fixed + one sticky row → correct ``enabled`` + ``sticky_pause_ranges``."""
    fake_state: Dict[str, Any] = {
        uct.ch_enabled_key(0): True,
        uct.ch_adstock_enabled_key(0): True,
        uct.ch_saturation_enabled_key(0): True,
        uct.ch_pause_rows_key(0): [
            {
                "id": 1,
                "kind": "fixed",
                "start": 1,
                "end": 2,
                "p_start": 0.2,
                "p_continue": 0.85,
            },
            {
                "id": 2,
                "kind": "sticky",
                "start": 3,
                "end": 4,
                "p_start": 0.11,
                "p_continue": 0.88,
            },
        ],
        uct.ch_pause_kind_key(0, 1): "fixed",
        uct.ch_pause_kind_key(0, 2): "sticky",
        uct.ch_pause_start_key(0, 1): 1,
        uct.ch_pause_end_key(0, 1): 2,
        uct.ch_pause_start_key(0, 2): 3,
        uct.ch_pause_end_key(0, 2): 4,
        uct.ch_pause_pstart_key(0, 2): 0.11,
        uct.ch_pause_pcont_key(0, 2): 0.88,
    }
    monkeypatch.setattr(uct.st, "session_state", fake_state, raising=False)

    patch, warns = uct._collect_channel_toggle(0, week_range=10)
    assert not warns
    assert patch["enabled"] == {
        "default": True,
        "off_ranges": [{"start_week": 1, "end_week": 2}],
    }
    assert patch["sticky_pause_ranges"] == [
        {
            "start_week": 3,
            "end_week": 4,
            "start_probability": 0.11,
            "continue_probability": 0.88,
        },
    ]
    assert patch["adstock_enabled"] is True
    assert patch["saturation_enabled"] is True


def test_collect_channel_toggle_sticky_only_enabled_true(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_state: Dict[str, Any] = {
        uct.ch_enabled_key(0): True,
        uct.ch_adstock_enabled_key(0): True,
        uct.ch_saturation_enabled_key(0): True,
        uct.ch_pause_rows_key(0): [
            {
                "id": 1,
                "kind": "sticky",
                "start": 2,
                "end": 5,
                "p_start": 0.3,
                "p_continue": 0.7,
            },
        ],
        uct.ch_pause_kind_key(0, 1): "sticky",
        uct.ch_pause_start_key(0, 1): 2,
        uct.ch_pause_end_key(0, 1): 5,
        uct.ch_pause_pstart_key(0, 1): 0.3,
        uct.ch_pause_pcont_key(0, 1): 0.7,
    }
    monkeypatch.setattr(uct.st, "session_state", fake_state, raising=False)

    patch, _ = uct._collect_channel_toggle(0, week_range=10)
    assert patch["enabled"] is True
    assert len(patch["sticky_pause_ranges"]) == 1


def test_collect_channel_toggle_type_switch_sticky_to_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row metadata says sticky but widget says fixed → emit only off_ranges."""
    fake_state: Dict[str, Any] = {
        uct.ch_enabled_key(0): True,
        uct.ch_adstock_enabled_key(0): True,
        uct.ch_saturation_enabled_key(0): True,
        uct.ch_pause_rows_key(0): [
            {"id": 7, "kind": "sticky", "start": 1, "end": 3, "p_start": 0.9, "p_continue": 0.9},
        ],
        uct.ch_pause_kind_key(0, 7): "fixed",
        uct.ch_pause_start_key(0, 7): 1,
        uct.ch_pause_end_key(0, 7): 3,
    }
    monkeypatch.setattr(uct.st, "session_state", fake_state, raising=False)

    patch, _ = uct._collect_channel_toggle(0, week_range=10)
    assert patch["sticky_pause_ranges"] == []
    assert patch["enabled"] == {
        "default": True,
        "off_ranges": [{"start_week": 1, "end_week": 3}],
    }


def main() -> None:
    """Invoke tests without pytest (used by ``python test.py``)."""
    test_pause_rows_from_yaml_fixed_then_sticky_order()
    test_pause_rows_from_yaml_disabled_channel_skips_fixed_keeps_sticky()
    test_pause_rows_from_yaml_clamps_weeks()
    mp = pytest.MonkeyPatch()
    try:
        test_collect_channel_toggle_splits_fixed_and_sticky(mp)
        test_collect_channel_toggle_sticky_only_enabled_true(mp)
        test_collect_channel_toggle_type_switch_sticky_to_fixed(mp)
    finally:
        mp.undo()
