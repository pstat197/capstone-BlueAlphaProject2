from typing import Any, Dict, List, Sequence

import numpy as np
from scripts.synth_input_classes.input_configurations import InputConfigurations

DEFAULT_GAMMA_SHAPE = 2.5
DEFAULT_GAMMA_SCALE = 1000.0


def _simulation_draw_order(channels: Sequence[Any]) -> List[int]:
    """YAML column indices sorted by ``channel_name`` (then index) for order-invariant RNG."""
    n = len(channels)
    return sorted(range(n), key=lambda i: (channels[i].get_channel_name(), i))


def _scatter_sorted_draws_to_yaml_order(
    work: np.ndarray, draw_order: Sequence[int]
) -> np.ndarray:
    """Map draws with columns in ``draw_order`` back to YAML ``channel_list`` column order.

    ``work[:, k]`` belongs to ``channels[draw_order[k]]``.
    """
    num_weeks, n = work.shape
    out = np.zeros((num_weeks, n), dtype=float)
    for k in range(n):
        yaml_col = int(draw_order[k])
        out[:, yaml_col] = work[:, k]
    return out


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


def _clip_spend_to_ranges(spend: np.ndarray, channels: List[Any]) -> None:
    """In-place clip each channel column to that channel's spend_range."""
    for c, ch in enumerate(channels):
        spend_range = ch.get_spend_range()
        low = spend_range[0] if len(spend_range) >= 1 else 0.0
        high = spend_range[1] if len(spend_range) >= 2 else np.inf
        spend[:, c] = np.clip(spend[:, c], low, high)


def _generate_correlated_spend(config: InputConfigurations) -> np.ndarray:
    """Draw correlated spend via Multivariate Normal in log space, then exponentiate."""
    num_weeks = config.get_week_range()
    channels = config.get_channel_list()
    num_channels = len(channels)
    rng = config.get_rng()

    draw_order = _simulation_draw_order(channels)
    sorted_names = [channels[i].get_channel_name() for i in draw_order]

    mu = np.zeros(num_channels)
    sigma = np.zeros(num_channels)

    for k in range(num_channels):
        ch = channels[draw_order[k]]
        params = ch.get_spend_sampling_gamma_params()
        shape = params.get("shape", DEFAULT_GAMMA_SHAPE)
        scale = params.get("scale", DEFAULT_GAMMA_SCALE)
        gamma_mean = shape * scale
        gamma_var = shape * (scale ** 2)
        # Lognormal parameterisation: match the gamma's mean and variance
        mu[k] = np.log(gamma_mean ** 2 / np.sqrt(gamma_var + gamma_mean ** 2))
        sigma[k] = np.sqrt(np.log(1 + gamma_var / gamma_mean ** 2))

    corr = _build_correlation_matrix(sorted_names, config.get_correlations())
    cov = np.diag(sigma) @ corr @ np.diag(sigma)

    log_spend_sorted = rng.multivariate_normal(mu, cov, size=num_weeks)
    work = np.exp(log_spend_sorted)
    for k in range(num_channels):
        ch = channels[draw_order[k]]
        spend_range = ch.get_spend_range()
        low = spend_range[0] if len(spend_range) >= 1 else 0.0
        high = spend_range[1] if len(spend_range) >= 2 else np.inf
        work[:, k] = np.clip(work[:, k], low, high)
    spend = _scatter_sorted_draws_to_yaml_order(work, draw_order)
    return spend


def _generate_independent_spend(config: InputConfigurations) -> np.ndarray:
    """Gamma-sampling path for independent channels (draw order by ``channel_name``)."""
    num_weeks = config.get_week_range()
    channels = config.get_channel_list()
    num_channels = len(channels)
    rng = config.get_rng()

    draw_order = _simulation_draw_order(channels)
    work = np.zeros((num_weeks, num_channels))
    for k in range(num_channels):
        ch = channels[draw_order[k]]
        params = ch.get_spend_sampling_gamma_params()
        shape = params.get("shape", DEFAULT_GAMMA_SHAPE)
        scale = params.get("scale", DEFAULT_GAMMA_SCALE)
        spend_range = ch.get_spend_range()
        low = spend_range[0] if len(spend_range) >= 1 else 0.0
        high = spend_range[1] if len(spend_range) >= 2 else np.inf
        draws = rng.gamma(shape, scale, size=num_weeks)
        work[:, k] = np.clip(draws, low, high)
    return _scatter_sorted_draws_to_yaml_order(work, draw_order)


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
            end_w = rule.get("end_week")
            if end_w is not None:
                end_w = int(end_w)
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
                week_1based = w + 1
                if week_1based < start_w:
                    continue
                if end_w is not None and week_1based > end_w:
                    continue
                move_amt = spend[w, fi] * fraction
                spend[w, fi] -= move_amt
                spend[w, ti] += move_amt
        elif t == "scale_channel":
            start_w = int(rule["start_week"])
            end_w = int(rule["end_week"])
            factor = float(rule["factor"])
            cname = rule["channel_name"]
            if cname not in name_to_idx:
                raise ValueError(
                    f"budget_shifts scale_channel: unknown channel {cname!r}; known: {names}"
                )
            ci = name_to_idx[cname]
            for w in range(num_weeks):
                week_1based = w + 1
                if start_w <= week_1based <= end_w:
                    spend[w, ci] *= factor
        else:
            raise ValueError(f"unsupported budget_shifts type: {t!r}")

    _clip_spend_to_ranges(spend, channels)
    spend[:] = np.maximum(spend, 0.0)


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
        mask = ch.spend_allowed_mask(num_weeks, config_seed=seed)
        if bool(np.all(mask)):
            continue
        out[~mask, c] = 0.0
    return out


def generate_spend(config: InputConfigurations) -> np.ndarray:
    """Return spend after budget shifts and channel toggle masking."""
    return generate_spend_with_details(config)[1]


def generate_spend_with_details(config: InputConfigurations) -> tuple[np.ndarray, np.ndarray]:
    """Return (pre_mask_spend, post_mask_spend) for analysis and downstream stages."""
    if config.get_correlations():
        spend = _generate_correlated_spend(config)
    else:
        spend = _generate_independent_spend(config)
    _apply_budget_shifts(spend, config)
    pre_mask = spend.astype(float, copy=True)
    post_mask = _apply_channel_toggles(config, spend)
    return pre_mask, post_mask
