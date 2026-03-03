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
        base_impressions[w, c] = (spend[w, c] / cpm[c]) * 1000
        noise ~ N(0, sigma^2), sigma = sqrt(noise_impression) * base_impressions[w, c]
        impressions = max(base_impressions + noise, 0)

    This respects:
      - per-channel CPM from the config
      - per-channel impression noise variance
      - global RNG seeded via the config loader
    """
    num_weeks = config.get_week_range()
    channels = config.get_channel_list()
    num_channels = len(channels)

    if spend_matrix.shape != (num_weeks, num_channels):
        raise ValueError(
            f"spend_matrix shape {spend_matrix.shape} does not match "
            f"(week_range={num_weeks}, num_channels={num_channels})."
        )

    rng = config.get_rng()

    cpms = _channel_cpm_list(config)  # shape (num_channels,)
    # base_impressions: weeks x channels
    base_impressions = (spend_matrix / cpms[None, :]) * 1000.0

    noise_std = np.zeros_like(base_impressions, dtype=float)
    for c, ch in enumerate(channels):
        noise_cfg = ch.get_noise_variance() or {}
        var_imp = float(noise_cfg.get("impression", 0.0))
        if var_imp < 0:
            raise ValueError(
                f"Impression noise variance for channel {ch.get_channel_name()} must be non-negative, got {var_imp}."
            )
        noise_std[:, c] = np.sqrt(var_imp) * base_impressions[:, c]

    noise = rng.normal(loc=0.0, scale=noise_std)
    impressions = base_impressions + noise
    # Ensure non-negative impressions
    impressions = np.clip(impressions, a_min=0.0, a_max=None)

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
