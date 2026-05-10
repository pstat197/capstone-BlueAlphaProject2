from typing import List

import numpy as np

from scripts.synth_input_classes.input_configurations import InputConfigurations


def _channel_cpm_list(config: InputConfigurations) -> np.ndarray:
    """Return array of CPM values, one per channel."""
    channels = config.get_channel_list()
    cpms: List[float] = []
    for ch in channels:
        cpm = ch.get_cpm()
        if cpm <= 0:
            raise ValueError(f"CPM for channel {ch.get_channel_name()} must be positive, got {cpm}.")
        cpms.append(float(cpm))
    return np.asarray(cpms, dtype=float)


def generate_impressions(config: InputConfigurations, spend_matrix: np.ndarray) -> np.ndarray:
    """
    Map weekly spend per channel to impressions per channel.

    For each channel c and week w:
        impressions[w, c] = (spend[w, c] / cpm[c]) * 1000

    Outcome-level revenue noise is applied later in ``generate_revenue``; impressions
    stay deterministic given spend and CPM.

    This respects per-channel CPM from the config and the same spend-allowed masks
    as spend generation (fully disabled channels and off weeks).
    """
    num_weeks = config.get_week_range()
    channels = config.get_channel_list()
    num_channels = len(channels)

    if spend_matrix.shape != (num_weeks, num_channels):
        raise ValueError(
            f"spend_matrix shape {spend_matrix.shape} does not match "
            f"(week_range={num_weeks}, num_channels={num_channels})."
        )

    cpms = _channel_cpm_list(config)  # shape (num_channels,)
    # base_impressions: weeks x channels
    base_impressions = (spend_matrix / cpms[None, :]) * 1000.0
    impressions = np.clip(base_impressions.astype(float, copy=True), a_min=0.0, a_max=None)

    # Safety-net channel toggle masking. Spend is already masked upstream so
    # base_impressions is zero on off weeks.
    seed = config.get_seed()
    for c, ch in enumerate(channels):
        if ch.is_fully_disabled():
            impressions[:, c] = 0.0
            continue
        mask = ch.spend_allowed_mask(num_weeks, channel_index=c, config_seed=seed)
        if bool(np.all(mask)):
            continue
        impressions[~mask, c] = 0.0

    return impressions

# # TEST

# spend_matrix = np.array([[1, 2, 3],
#                          [4, 5, 7],
#                          [8, 9, 6]])
# weeks = 3
# channels = 3
# var = [0, 1, 2]
# std = np.sqrt(var)
# std_scaled = std * spend_matrix
# means = np.zeros((weeks, channels))
# cpm = [1, 25, 48]
# noise = np.random.normal(loc=means, scale = std_scaled, size = (weeks, channels))
# impressions = ((spend_matrix / cpm) * 1000) + noise
# print(impressions)
# print(noise)
