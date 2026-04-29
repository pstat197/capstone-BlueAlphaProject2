"""
Load user YAML and config/default.yaml; fill or generate missing fields per design plan.
- Top-level: run_identifier (or run_YYYYMMDD_HHMM), week_range from default.
- number_of_channels: add channels up to N, named "Generated Channel 1", ... with default + noise.
- Per-channel: missing fields filled with default or default + noise (saturation_config and adstock_decay_config = default only).
"""
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from scripts.spend_simulation.budget_shift_auto import generate_auto_budget_shift_rules
from scripts.spend_simulation.correlation_auto import generate_auto_correlation_entries
from scripts.synth_input_classes.input_configurations import InputConfigurations, _normalize_budget_shifts

from .defaults import get_default_channel_template

from .noise import add_noise_to_value

# Default channel template keys that get default-as-is (no noise); all config dicts processed the same way
NO_NOISE_KEYS = {
    "saturation_config",
    "adstock_decay_config",
    "seasonality_config",
    "spend_sampling_gamma_params",
    "noise_variance",
    "trend_slope",
    "channel_name",
    "cpm_sampling_range",  # used only to sample cpm when cpm is missing
    # On/off toggle fields are copied verbatim (never noised): they are
    # boolean/structured config, not sampled numerics.
    "enabled",
    "adstock_enabled",
    "saturation_enabled",
}


def _default_channel_template(default_data: Dict[str, Any]) -> Dict[str, Any]:
    """First channel from default config as a template (from default.yaml)."""
    raw = default_data.get("channel_list") or []
    if not raw:
        return get_default_channel_template()
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
    *,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Copy template and add normal noise to every numeric field; no noise for NO_NOISE_KEYS."""
    out = {}
    for k, v in template.items():
        if k in NO_NOISE_KEYS:
            # Copy dicts so we don't mutate shared defaults
            out[k] = dict(v) if isinstance(v, dict) else v
        elif isinstance(v, dict):
            out[k] = {kk: add_noise_to_value(float(vv), rng=rng) for kk, vv in v.items()}
        elif isinstance(v, list):
            out[k] = [add_noise_to_value(float(x), rng=rng) for x in v]
        elif isinstance(v, (int, float)):
            out[k] = add_noise_to_value(float(v), rng=rng)
        else:
            out[k] = v
    return out


def _fill_channel_missing_fields(
    ch: Dict[str, Any],
    default_channel: Dict[str, Any],
    index_1based: int,
    *,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Fill any missing channel fields: default + noise for numerics; config dicts (saturation, adstock, gamma_params, noise_variance) get default only; placeholder name if missing. If cpm is missing, sample from Uniform(cpm_sampling_range)."""
    out = dict(ch)
    for key, default_val in default_channel.items():
        if key == "cpm_sampling_range":
            continue  # meta key for sampling cpm when missing; do not copy into channel
        if key not in out or out[key] is None or out[key] == "":
            if key == "channel_name":
                out[key] = f"Unnamed Channel {index_1based}"
            elif key in ("saturation_config", "adstock_decay_config", "seasonality_config", "spend_sampling_gamma_params", "noise_variance"):
                out[key] = dict(default_val) if isinstance(default_val, dict) else default_val
            elif isinstance(default_val, dict):
                out[key] = {
                    k: add_noise_to_value(float(v), rng=rng)
                    for k, v in default_val.items()
                }
            elif isinstance(default_val, list):
                out[key] = [add_noise_to_value(float(x), rng=rng) for x in default_val]
            elif isinstance(default_val, (int, float)):
                out[key] = add_noise_to_value(float(default_val), rng=rng)
            else:
                out[key] = default_val
    # If cpm not provided, sample per channel from Uniform(cpm_sampling_range)
    if "cpm" not in out or out["cpm"] is None:
        low, high = 0.5, 50.0
        cpm_range = default_channel.get("cpm_sampling_range")
        if isinstance(cpm_range, (list, tuple)) and len(cpm_range) >= 2:
            low, high = float(cpm_range[0]), float(cpm_range[1])
        out["cpm"] = float(rng.uniform(low, high))
    return out


def _channel_names_in_order(merged: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in merged.get("channel_list") or []:
        ch = item.get("channel") or item
        nm = str(ch.get("channel_name", "")).strip() if isinstance(ch, dict) else ""
        if nm and nm not in seen:
            seen.add(nm)
            out.append(nm)
    return out


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def _validate_correlation_matrix_psd(
    channel_names_ordered: List[str],
    correlations_raw: List[Dict[str, Any]],
    *,
    tol: float = 1e-8,
) -> None:
    """Reject invalid pairwise rho sets that cannot form a PSD correlation matrix."""
    n = len(channel_names_ordered)
    if n <= 1:
        return
    name_to_idx = {nm: i for i, nm in enumerate(channel_names_ordered)}
    corr = np.eye(n, dtype=float)
    for entry in correlations_raw:
        pair = entry.get("channels") or []
        if len(pair) != 2:
            continue
        a, b = str(pair[0]), str(pair[1])
        if a not in name_to_idx or b not in name_to_idx or a == b:
            continue
        i, j = name_to_idx[a], name_to_idx[b]
        rho = float(entry.get("rho", 0.0))
        corr[i, j] = rho
        corr[j, i] = rho

    min_eig = float(np.linalg.eigvalsh(corr).min())
    if min_eig < -tol:
        raise ValueError(
            "Correlations form a non-PSD matrix; please adjust pairwise rho values "
            f"(minimum eigenvalue {min_eig:.6f})."
        )


def apply_seed_append_expansion(merged: Dict[str, Any]) -> None:
    """
    Read ``budget_shifts_auto_mode`` / ``correlations_auto_mode``, expand into concrete
    ``budget_shifts`` / ``correlations`` lists for this run, then **remove** the mode keys
    so :class:`InputConfigurations` only sees canonical YAML fields.
    """
    seed = int(merged.get("seed") or 0)
    week_range = max(1, int(merged.get("week_range") or 52))
    names_list = _channel_names_in_order(merged)
    names_set = set(names_list)

    bs_mode = str(merged.pop("budget_shifts_auto_mode", "none") or "none").strip().lower()
    if bs_mode not in ("none", "global", "global_and_channel"):
        bs_mode = "none"

    manual_bs_raw = merged.get("budget_shifts")
    manual_bs = _normalize_budget_shifts(manual_bs_raw)
    if bs_mode in ("global", "global_and_channel") and names_list:
        extra = generate_auto_budget_shift_rules(week_range, names_list, bs_mode, seed)
        merged["budget_shifts"] = _normalize_budget_shifts(manual_bs + extra)
    else:
        merged["budget_shifts"] = manual_bs

    corr_mode = str(merged.pop("correlations_auto_mode", "none") or "none").strip().lower()
    if corr_mode not in ("none", "random"):
        corr_mode = "none"

    manual_corr: List[Dict[str, Any]] = []
    for entry in merged.get("correlations") or []:
        if isinstance(entry, dict):
            manual_corr.append(dict(entry))

    ordered_keys: List[Tuple[str, str]] = []
    last_rho: Dict[Tuple[str, str], float] = {}
    for entry in manual_corr:
        pair = entry.get("channels") or []
        if len(pair) != 2:
            continue
        a, b = str(pair[0]).strip(), str(pair[1]).strip()
        if not a or not b or a == b:
            continue
        pk = _pair_key(a, b)
        if pk not in last_rho:
            ordered_keys.append(pk)
        last_rho[pk] = float(entry.get("rho", 0.0))

    if corr_mode == "random" and len(names_list) >= 2:
        names_sorted = sorted(names_set)
        for entry in generate_auto_correlation_entries(names_sorted, seed):
            ch = entry.get("channels") or []
            if len(ch) != 2:
                continue
            a, b = str(ch[0]).strip(), str(ch[1]).strip()
            if not a or not b or a == b or a not in names_set or b not in names_set:
                continue
            pk = _pair_key(a, b)
            if pk in last_rho:
                continue
            rho = float(entry.get("rho", 0.0))
            rho = max(-1.0, min(1.0, rho))
            last_rho[pk] = rho
            ordered_keys.append(pk)

    merged["correlations"] = [{"channels": [k[0], k[1]], "rho": last_rho[k]} for k in ordered_keys]


def load_config_from_dict(user_data: Dict[str, Any]) -> InputConfigurations:
    """
    Same as load_config but user config is provided as a dict (merged with default.yaml).
    """
    config_dir = Path(__file__).resolve().parent
    default_path = config_dir / "default.yaml"
    with open(default_path, "r") as f:
        default_data = yaml.safe_load(f) or {}

    # Deep copy so downstream merges and validation never mutate the caller's nested lists
    # (e.g. `budget_shifts`, `correlations`) — important for Streamlit snapshots of the same dict.
    user_data = copy.deepcopy(user_data or {})
    merged = _deep_merge(user_data, default_data)
    default_channel = _default_channel_template(default_data)

    seed = merged.get("seed")
    merge_rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()

    if not merged.get("run_identifier"):
        merged["run_identifier"] = "run_" + datetime.now().strftime("%Y%m%d_%H%M")
    if "week_range" not in merged or merged.get("week_range") is None:
        merged["week_range"] = default_data.get("week_range") or default_data.get("weeks", 52)
    merged.pop("weeks", None)

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
            noised = _add_noise_to_channel_template(default_channel, rng=merge_rng)
            noised["channel_name"] = f"Generated Channel {i - len(channel_list) + 1}"
            channel_list.append({"channel": noised})

    merged["channel_list"] = channel_list

    filled = []
    for i, item in enumerate(channel_list):
        ch = item.get("channel") or item
        filled_ch = _fill_channel_missing_fields(ch, default_channel, i + 1, rng=merge_rng)
        filled.append({"channel": filled_ch})
    merged["channel_list"] = filled

    apply_seed_append_expansion(merged)

    # Step 5: Validate correlations block (if present)
    correlations_raw = merged.get("correlations") or []
    channel_names = set()
    for item in merged.get("channel_list", []):
        ch = item.get("channel") or item
        channel_names.add(ch.get("channel_name", ""))
    for entry in correlations_raw:
        pair = entry.get("channels", [])
        if len(pair) != 2:
            raise ValueError(f"Each correlation entry must specify exactly 2 channels, got {pair}")
        for name in pair:
            if name not in channel_names:
                raise ValueError(f"Correlation references unknown channel '{name}'. Available: {sorted(channel_names)}")
        rho = float(entry.get("rho", 0.0))
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation rho must be in [-1, 1], got {rho} for {pair}")
    _validate_correlation_matrix_psd(sorted(channel_names), correlations_raw)

    # Step 6: Build (inject default channel so builder uses default.yaml defaults for any missing keys)
    return InputConfigurations.from_yaml_dict(merged, default_channel_template=default_channel)


def load_config(user_yaml_path: str) -> InputConfigurations:
    """
    Load user YAML and default.yaml; fill missing top-level and per-channel fields;
    add channels up to number_of_channels with default+noise; build InputConfigurations.
    """
    user_path = Path(user_yaml_path)
    if not user_path.exists():
        raise FileNotFoundError(f"Config file not found: {user_path}")
    with open(user_path, "r") as f:
        user_data = yaml.safe_load(f) or {}

    return load_config_from_dict(user_data)
