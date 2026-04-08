"""Load / dump YAML used by the Streamlit UI."""

from __future__ import annotations

from typing import Any, Dict

import yaml

from app.paths import EXAMPLE_YAML_PATH, UI_SCHEMA_PATH


def load_ui_schema() -> Dict[str, Any]:
    with open(UI_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_example_text() -> str:
    if EXAMPLE_YAML_PATH.is_file():
        return EXAMPLE_YAML_PATH.read_text(encoding="utf-8")
    return "# example.yaml not found\nrun_identifier: Demo\nweek_range: 26\nchannel_list: []\n"


def yaml_dump(cfg: Dict[str, Any]) -> str:
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
