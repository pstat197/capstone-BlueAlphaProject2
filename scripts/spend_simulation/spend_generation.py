from typing import Any, List

import numpy as np
from scripts.synth_input_classes.input_configurations import InputConfigurations

# Default gamma params if channel config is missing shape/scale (from default.yaml)
DEFAULT_GAMMA_SHAPE = 2.5
DEFAULT_GAMMA_SCALE = 1000.0


def _clip_spend_to_ranges(spend: np.ndarray, channels: List[Any]) -> None:
    """In-place clip each channel column to that channel's spend_range."""
    for c, ch in enumerate(channels):
        spend_range = ch.get_spend_range()
        low = spend_range[0] if len(spend_range) >= 1 else 0.0
        high = spend_range[1] if len(spend_range) >= 2 else np.inf
        spend[:, c] = np.clip(spend[:, c], low, high)


def _generate_base_spend(config: InputConfigurations) -> np.ndarray:
    num_weeks = config.get_week_range()
    channels = config.get_channel_list()
    num_channels = len(channels)
    rng = config.get_rng()

    out = np.zeros((num_weeks, num_channels))
    for c in range(num_channels):
        params = channels[c].get_spend_sampling_gamma_params()
        shape = params.get("shape", DEFAULT_GAMMA_SHAPE)
        scale = params.get("scale", DEFAULT_GAMMA_SCALE)
        spend_range = channels[c].get_spend_range()
        low = spend_range[0] if len(spend_range) >= 1 else 0.0
        high = spend_range[1] if len(spend_range) >= 2 else np.inf
        for w in range(num_weeks):
            out[w, c] = rng.gamma(shape, scale)
            out[w, c] = np.clip(out[w, c], low, high)
    return out


def _apply_budget_shifts(spend: np.ndarray, config: InputConfigurations) -> None:
    """Apply optional budget rules in order (after base draw). Mutates spend in place; clips at end."""
    shifts = config.get_budget_shifts()
    if not shifts:
        return

    channels = config.get_channel_list()
    num_weeks = spend.shape[0]
    names = [ch.get_channel_name() for ch in channels]
    name_to_idx = {n: i for i, n in enumerate(names)}

    for rule in shifts:
        t = rule["type"]
        if t == "scale":
            start_w = int(rule["start_week"])
            end_w = int(rule["end_week"])
            factor = float(rule["factor"])
            for w in range(num_weeks):
                week_1based = w + 1
                if start_w <= week_1based <= end_w:
                    spend[w, :] *= factor
        elif t == "reallocate":
            start_w = int(rule["start_week"])
            f_name = rule["from_channel"]
            t_name = rule["to_channel"]
            fraction = float(rule["fraction"])
            if f_name not in name_to_idx or t_name not in name_to_idx:
                raise ValueError(
                    f"budget_shifts reallocate: unknown channel "
                    f"(from_channel={f_name!r}, to_channel={t_name!r}); "
                    f"known: {names}"
                )
            fi = name_to_idx[f_name]
            ti = name_to_idx[t_name]
            if fi == ti:
                continue
            for w in range(num_weeks):
                if w + 1 < start_w:
                    continue
                move_amt = spend[w, fi] * fraction
                spend[w, fi] -= move_amt
                spend[w, ti] += move_amt
        else:
            raise ValueError(f"unsupported budget_shifts type: {t!r}")

    _clip_spend_to_ranges(spend, channels)
    spend[:] = np.maximum(spend, 0.0)


def generate_spend(config: InputConfigurations) -> np.ndarray:
    spend = _generate_base_spend(config)
    _apply_budget_shifts(spend, config)
    return spend
