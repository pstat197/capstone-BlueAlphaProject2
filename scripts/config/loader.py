"""
Load user YAML and config/default.yaml; fill or generate missing fields per design plan.
- Top-level: run_identifier (or run_YYYYMMDD_HHMM), week_range from default.
- number_of_channels: add channels up to N, named "Generated Channel 1", ... with default + noise.
- Per-channel: missing fields filled with default or default + noise (saturation_function = default only).
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from scripts.synth_input_classes.input_configurations import InputConfigurations

from .noise import add_noise_to_value, init_rng

# Default channel template keys that get default-as-is (no noise)
NO_NOISE_KEYS = {"saturation_function", "channel_name"}


def _default_channel_template(default_data: Dict[str, Any]) -> Dict[str, Any]:
    """First channel from default config as a template (flat channel dict)."""
    raw = default_data.get("channel_list") or []
    if not raw:
        return {
            "channel_name": "Channel 1",
            "spend_sampling_gamma_params": {"shape": 2.5, "scale": 1000},
            "noise_variance": {"impression": 0.1, "revenue": 0.1},
            "true_roi": 1.0,
            "spend_range": [1000, 50000],
            "baseline_revenue": 5000,
            "saturation_function": "log(x+1)",
        }
    item = raw[0]
    ch = item.get("channel") or item
    return dict(ch)


def _deep_merge(user: Any, default: Any) -> Any:
    """Recursively merge user dict over default. User values take precedence."""
    if not isinstance(user, dict) or not isinstance(default, dict):
        return user if user is not None else default
    out = dict(default)
    for k, v in user.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(v, out[k])
        else:
            out[k] = v
    return out


def _add_noise_to_channel_template(
    template: Dict[str, Any],
) -> Dict[str, Any]:
    """Copy template and add normal noise to every numeric field; no noise for NO_NOISE_KEYS."""
    out = {}
    for k, v in template.items():
        if k in NO_NOISE_KEYS:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {kk: add_noise_to_value(float(vv)) for kk, vv in v.items()}
        elif isinstance(v, list):
            out[k] = [add_noise_to_value(float(x)) for x in v]
        elif isinstance(v, (int, float)):
            out[k] = add_noise_to_value(float(v))
        else:
            out[k] = v
    return out


def _fill_channel_missing_fields(
    ch: Dict[str, Any],
    default_channel: Dict[str, Any],
    index_1based: int,
) -> Dict[str, Any]:
    """Fill any missing channel fields: default + noise for numerics; default only for saturation_function; placeholder name if missing."""
    out = dict(ch)
    for key, default_val in default_channel.items():
        if key not in out or out[key] is None or out[key] == "":
            if key == "channel_name":
                out[key] = f"Unnamed Channel {index_1based}"
            elif key == "saturation_function":
                out[key] = default_val
            elif isinstance(default_val, dict):
                out[key] = {
                    k: add_noise_to_value(float(v))
                    for k, v in default_val.items()
                }
            elif isinstance(default_val, list):
                out[key] = [add_noise_to_value(float(x)) for x in default_val]
            elif isinstance(default_val, (int, float)):
                out[key] = add_noise_to_value(float(default_val))
            else:
                out[key] = default_val
    return out


def load_config(user_yaml_path: str) -> InputConfigurations:
    """
    Load user YAML and default.yaml; fill missing top-level and per-channel fields;
    add channels up to number_of_channels with default+noise; build InputConfigurations.
    """
    config_dir = Path(__file__).resolve().parent
    default_path = config_dir / "default.yaml"
    with open(default_path, "r") as f:
        default_data = yaml.safe_load(f) or {}

    user_path = Path(user_yaml_path)
    if not user_path.exists():
        raise FileNotFoundError(f"Config file not found: {user_path}")
    with open(user_path, "r") as f:
        user_data = yaml.safe_load(f) or {}

    # Merge user over default (deep)
    merged = _deep_merge(user_data, default_data)
    default_channel = _default_channel_template(default_data)

    # Seed: use from config if present for RNG; InputConfigurations will store it from merged
    seed = merged.get("seed")
    if seed is not None:
        init_rng(int(seed))
    else:
        init_rng(None)  # random seed for this run

    # Step 2: Fill missing top-level
    if not merged.get("run_identifier"):
        merged["run_identifier"] = "run_" + datetime.now().strftime("%Y%m%d_%H%M")
    if "week_range" not in merged or merged.get("week_range") is None:
        merged["week_range"] = default_data.get("week_range") or default_data.get("weeks", 52)
    if "weeks" in merged and "week_range" not in merged:
        merged["week_range"] = merged.get("weeks")
    merged.pop("weeks", None)

    # Step 3: Ensure enough channels (number_of_channels)
    num_channels_opt = merged.pop("number_of_channels", None)
    channel_list: List[Dict[str, Any]] = []
    raw_list = merged.get("channel_list") or []
    for item in raw_list:
        ch = item.get("channel") or item
        channel_list.append({"channel": dict(ch)})

    target_count = None
    if num_channels_opt is not None and num_channels_opt > 0:
        target_count = int(num_channels_opt)
    if target_count is not None and target_count > len(channel_list):
        for i in range(len(channel_list), target_count):
            noised = _add_noise_to_channel_template(default_channel)
            # 1-based index among generated channels only
            noised["channel_name"] = f"Generated Channel {i - len(channel_list) + 1}"
            channel_list.append({"channel": noised})

    merged["channel_list"] = channel_list

    # Step 4: Fill missing fields inside each channel
    filled = []
    for i, item in enumerate(channel_list):
        ch = item.get("channel") or item
        filled_ch = _fill_channel_missing_fields(ch, default_channel, i + 1)
        filled.append({"channel": filled_ch})
    merged["channel_list"] = filled

    # Step 5: Build
    return InputConfigurations.from_yaml_dict(merged)
