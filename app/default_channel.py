"""Load a default channel dict from scripts/config/default.yaml for new channels."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_YAML = _REPO_ROOT / "scripts" / "config" / "default.yaml"


def default_channel_dict() -> Dict[str, Any]:
    with open(_DEFAULT_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("channel_list") or []
    if not raw:
        return {
            "channel_name": "New channel",
            "cpm": 10.0,
            "spend_sampling_gamma_params": {"shape": 2.5, "scale": 1000},
            "noise_variance": {"impression": 0.0225, "revenue": 0.1},
            "true_roi": 2.5,
            "spend_range": [1000, 50000],
            "baseline_revenue": 5000,
            "saturation_config": {
                "type": "linear",
                "slope": 1.0,
                "K": 50000.0,
                "beta": 0.5,
            },
            "adstock_decay_config": {
                "type": "linear",
                "lambda": 0.5,
                "lag": 10,
                "weights": [1.0],
            },
        }
    first = raw[0].get("channel") or raw[0]
    return copy.deepcopy(dict(first))
