"""Load / dump YAML used by the Streamlit UI."""

from __future__ import annotations

from typing import Any, Dict

import yaml

from app.paths import EXAMPLE_YAML_PATH, UI_SCHEMA_PATH


def sanitize_for_yaml_export(obj: Any) -> Any:
    """Recursively convert values to types PyYAML always serializes (e.g. NumPy scalars → Python)."""
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return sanitize_for_yaml_export(obj.tolist())
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): sanitize_for_yaml_export(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_yaml_export(v) for v in obj]
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj) if isinstance(obj, float) else int(obj)
    return obj


def load_ui_schema() -> Dict[str, Any]:
    with open(UI_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_example_text() -> str:
    if EXAMPLE_YAML_PATH.is_file():
        return EXAMPLE_YAML_PATH.read_text(encoding="utf-8")
    return "# example.yaml not found\nrun_identifier: Demo\nweek_range: 26\nchannel_list: []\n"


def yaml_dump(cfg: Dict[str, Any]) -> str:
    safe = sanitize_for_yaml_export(cfg if isinstance(cfg, dict) else {})
    return yaml.dump(safe, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
