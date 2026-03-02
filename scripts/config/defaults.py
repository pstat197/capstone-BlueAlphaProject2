"""
Single source of truth for per-channel defaults: loaded from default.yaml.
Change default.yaml only; code here just reads it (with a minimal fallback if file is missing/empty).
"""
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_YAML_PATH = Path(__file__).resolve().parent / "default.yaml"
_cached_default_channel: Optional[Dict[str, Any]] = None


def get_default_channel_template() -> Dict[str, Any]:
    """
    Return the default channel dict from default.yaml (first channel).
    Cached after first load. If file is missing or has no channels, returns a minimal fallback.
    """
    global _cached_default_channel
    if _cached_default_channel is not None:
        return dict(_cached_default_channel)

    if not _DEFAULT_YAML_PATH.exists():
        _cached_default_channel = _minimal_fallback_channel()
        return dict(_cached_default_channel)

    with open(_DEFAULT_YAML_PATH, "r") as f:
        data = yaml.safe_load(f) or {}

    raw = data.get("channel_list") or []
    if not raw:
        _cached_default_channel = _minimal_fallback_channel()
        return dict(_cached_default_channel)

    item = raw[0]
    ch = item.get("channel") or item
    _cached_default_channel = dict(ch)
    return dict(_cached_default_channel)


def _minimal_fallback_channel() -> Dict[str, Any]:
    """Used only when default.yaml is missing or has no channels."""
    return {
        "channel_name": "Channel 1",
        "spend_sampling_gamma_params": {"shape": 2.5, "scale": 1000},
        "noise_variance": {"impression": 0.1, "revenue": 0.1},
        "true_roi": 1.0,
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
