from typing import Dict, List, Any

import numpy as np
from scripts.synth_input_classes.input_configurations import InputConfigurations

DEFAULT_GAMMA_SHAPE = 2.5
DEFAULT_GAMMA_SCALE = 1000.0


def _build_correlation_matrix(
    channel_names: List[str],
    correlations: List[Dict[str, Any]],
) -> np.ndarray:
    """Build a symmetric correlation matrix from pairwise entries.
    Unspecified pairs default to 0 (independent). Diagonal is 1."""
    n = len(channel_names)
    name_to_idx = {name: i for i, name in enumerate(channel_names)}
    corr = np.eye(n)
    for entry in correlations:
        pair = entry["channels"]
        rho = entry["rho"]
        i, j = name_to_idx[pair[0]], name_to_idx[pair[1]]
        corr[i, j] = rho
        corr[j, i] = rho
    return corr


def _generate_correlated_spend(config: InputConfigurations) -> np.ndarray:
    """Draw correlated spend via Multivariate Normal in log space, then exponentiate."""
    num_weeks = config.get_week_range()
    channels = config.get_channel_list()
    num_channels = len(channels)
    rng = config.get_rng()

    channel_names = [ch.get_channel_name() for ch in channels]

    mu = np.zeros(num_channels)
    sigma = np.zeros(num_channels)

    for c, ch in enumerate(channels):
        params = ch.get_spend_sampling_gamma_params()
        shape = params.get("shape", DEFAULT_GAMMA_SHAPE)
        scale = params.get("scale", DEFAULT_GAMMA_SCALE)
        gamma_mean = shape * scale
        gamma_var = shape * (scale ** 2)
        # Lognormal parameterisation: match the gamma's mean and variance
        mu[c] = np.log(gamma_mean ** 2 / np.sqrt(gamma_var + gamma_mean ** 2))
        sigma[c] = np.sqrt(np.log(1 + gamma_var / gamma_mean ** 2))

    corr = _build_correlation_matrix(channel_names, config.get_correlations())
    cov = np.diag(sigma) @ corr @ np.diag(sigma)

    log_spend = rng.multivariate_normal(mu, cov, size=num_weeks)
    spend = np.exp(log_spend)

    for c, ch in enumerate(channels):
        spend_range = ch.get_spend_range()
        low = spend_range[0] if len(spend_range) >= 1 else 0.0
        high = spend_range[1] if len(spend_range) >= 2 else np.inf
        spend[:, c] = np.clip(spend[:, c], low, high)

    return spend


def _generate_independent_spend(config: InputConfigurations) -> np.ndarray:
    """Original gamma-sampling path for independent channels."""
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


def _apply_channel_toggles(config: InputConfigurations, spend: np.ndarray) -> np.ndarray:
    """
    Zero out spend for channels that are fully disabled or in per-week off ranges.

    Masking happens at the end of spend generation so that downstream stages
    (correlation analysis, CPM -> impressions, revenue) all see a spend matrix
    that is already consistent with the channel on/off schedule. Channels with
    no toggles configured stay fully on (fail-open).
    """
    if spend.size == 0:
        return spend

    num_weeks = spend.shape[0]
    out = spend.astype(float, copy=True)
    seed = config.get_seed()
    for c, ch in enumerate(config.get_channel_list()):
        if ch.is_fully_disabled():
            out[:, c] = 0.0
            continue
        mask = ch.spend_allowed_mask(num_weeks, channel_index=c, config_seed=seed)
        if bool(np.all(mask)):
            continue
        out[~mask, c] = 0.0
    return out


def generate_spend(config: InputConfigurations) -> np.ndarray:
    if config.get_correlations():
        spend = _generate_correlated_spend(config)
    else:
        spend = _generate_independent_spend(config)
    return _apply_channel_toggles(config, spend)