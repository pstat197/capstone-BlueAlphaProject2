"""
Single source of truth for per-channel defaults: loaded from default.yaml.
Change default.yaml only; code here just reads it and raises if config is invalid.
"""
from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_YAML_PATH = Path(__file__).resolve().parent / "default.yaml"


def get_default_channel_template() -> Dict[str, Any]:
    """
    Return the default channel dict from default.yaml (first channel).

    Reads from disk each call (no in-process cache) so edits to default.yaml
    are always visible without restarting a long-lived process.
    """
    if not _DEFAULT_YAML_PATH.exists():
        raise FileNotFoundError(f"Required default config not found: {_DEFAULT_YAML_PATH}")

    with open(_DEFAULT_YAML_PATH, "r") as f:
        data = yaml.safe_load(f) or {}

    raw = data.get("channel_list") or []
    if not raw:
        raise ValueError(
            f"Invalid default config at {_DEFAULT_YAML_PATH}: 'channel_list' must contain at least one channel"
        )

    item = raw[0]
    ch = item.get("channel") or item
    return dict(ch)
