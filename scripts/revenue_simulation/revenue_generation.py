from typing import Dict
import numpy as np

from scripts.synth_input_classes.input_configurations import InputConfigurations
from scripts.synth_input_classes.channel import Channel


def _saturation_fn(impressions: np.ndarray, saturation_config: Dict) -> np.ndarray:
    """
    Apply saturation function to impressions.

    Supported types:
      - 'hill':            x^slope / (x^slope + K^slope)
      - 'diminishing_returns': x / (1 + beta * x)
      - 'linear':          linear scaling slope * impressions (default slope=1 → identity)
    """
    saturation_type = (saturation_config or {}).get("type", "linear")

    if saturation_type == "hill":
        slope = float(saturation_config.get("slope", 1.0))
        K = float(saturation_config.get("K", 1.0))
        x_a = np.power(impressions, slope)
        K_a = np.power(K, slope)
        return x_a / (x_a + K_a + 1e-9)

    if saturation_type == "diminishing_returns":
        beta = float(saturation_config.get("beta", 0.0))
        return impressions / (1.0 + beta * impressions)

    if saturation_type == "linear":
        slope = float(saturation_config.get("slope", 1.0))
        return slope * impressions.astype(float, copy=True)

    raise ValueError(
        f'Unknown saturation_config type "{saturation_type}". '
        'Expected one of: "hill", "diminishing_returns", "linear".'
    )


def _adstock_decay(impressions: np.ndarray, adstock_config: Dict) -> np.ndarray:
    """
    Apply adstock decay to impressions.

    Supported types:
      - 'geometric' or 'exponential': geometric decay with rate lambda and truncated lag L
      - 'weighted': arbitrary finite impulse response with provided weights
      - 'linear': if lag <= 0, return impressions; else uniform moving average over lag+1 weeks
    """
    adstock_config = adstock_config or {}
    adstock_type = adstock_config.get("type", "linear")

    if adstock_type == "linear":
        lag = int(adstock_config.get("lag", 0))
        if lag < 0:
            raise ValueError(f"adstock lag must be non-negative, got {lag}.")
        if lag == 0:
            return impressions.astype(float, copy=True)
        weights = np.ones(lag + 1, dtype=float) / (lag + 1)
        return np.convolve(impressions, weights, mode="full")[: len(impressions)]

    if adstock_type in ("geometric", "exponential"):
        lambda_ = float(adstock_config.get("lambda", adstock_config.get("decay_rate", 0.0)))
        lag = int(adstock_config.get("lag", 0))
        if lag < 0:
            raise ValueError(f"adstock lag must be non-negative, got {lag}.")
        lag_array = np.arange(lag + 1)
        decay_weights = np.power(lambda_, lag_array)
        return np.convolve(impressions, decay_weights, mode="full")[: len(impressions)]

    if adstock_type == "weighted":
        weights = adstock_config.get("weights", [1.0])
        weights_arr = np.asarray(weights, dtype=float)
        return np.convolve(impressions, weights_arr, mode="full")[: len(impressions)]

    raise ValueError(
        f'Unknown adstock_decay_config type "{adstock_type}". '
        'Expected one of: "geometric", "exponential", "weighted", "linear".'
    )

# seasonality functions
def _fourier_seasonality(t, period, K=1, scale=0.1, seed=None):
    '''
    t: time index array (np.ndarray)
    period: seasonality period (e.g. 52 for weekly data with yearly seasonality)
    K: number of Fourier harmonics (higher K = more complex patterns)
    scale: overall amplitude of seasonality
    seed: random seed for reproducibility (optional)
    '''
    rng = np.random.default_rng(seed)
    s = np.zeros_like(t, dtype=float)

    for k in range(1, K + 1):
        a_k = rng.normal(0, scale / k)
        b_k = rng.normal(0, scale / k)
        s += a_k * np.sin(2 * np.pi * k * t / period)
        s += b_k * np.cos(2 * np.pi * k * t / period)

    return s

# 'categorical': discrete repeating pattern (e.g., weekly effects); assigns fixed multipliers per period index (t % period);
def _categorical_seasonality(t, pattern):
    pattern = np.array(pattern, dtype=float)
    return pattern[t % len(pattern)] - 1.0  # convert to deviation

# 'spikes': sparse, non-periodic multiplicative shocks; simulates irregular events (promotions, outages, viral bursts)
def _event_spikes(t, prob=0.02, magnitude=(0.5, 1.5), seed=None):
    rng = np.random.default_rng(seed)
    spikes = np.zeros_like(t, dtype=float)
    mask = rng.random(len(t)) < prob
    spikes[mask] = rng.uniform(*magnitude, size=mask.sum())
    return spikes


def _seasonality(t: np.ndarray, seasonality_config: Dict) -> np.ndarray:
    """
    Flexible seasonality generator.

    Supported types:
      - 'sin': basic sinusoidal
      - 'fourier': multi-harmonic smooth seasonality
      - 'categorical': discrete pattern (e.g. weekly)
      - 'hybrid': combination of components
    """

    if not seasonality_config:
        return np.ones_like(t, dtype=float)

    stype = seasonality_config.get("type", "sin")

    if stype == "sin":
        amplitude = seasonality_config.get("amplitude", 0.2)
        period = seasonality_config.get("period", 52)
        phase = seasonality_config.get("phase", 0)

        return 1 + amplitude * np.sin(2 * np.pi * (t + phase) / period)

    if stype == "fourier":
        period = seasonality_config["period"]
        K = seasonality_config.get("K", 2)
        scale = seasonality_config.get("scale", 0.1)

        s = _fourier_seasonality(t, period, K, scale)
        return 1 + s

    if stype == "categorical":
        pattern = seasonality_config["pattern"]

        s = _categorical_seasonality(t, pattern)
        return 1 + s

    if stype == "hybrid":
        components = seasonality_config.get("components", [])
        s = np.ones_like(t, dtype=float)

        for comp in components:
            ctype = comp["type"]

            if ctype == "fourier":
                val = _fourier_seasonality(
                    t,
                    comp["period"],
                    comp.get("K", 1),
                    comp.get("scale", 0.1),
                    comp.get("seed", None)
                )
                s *= (1 + val)

            elif ctype == "categorical":
                pattern = comp["pattern"]
                val = np.array(pattern)[t % len(pattern)]
                s *= val

            elif ctype == "spikes":
                val = _event_spikes(
                    t,
                    prob=comp.get("prob", 0.02),
                    magnitude=comp.get("magnitude", (0.5, 1.5)),
                    seed=comp.get("seed", None)
                )
                s *= (1 + val)

            else:
                raise ValueError(f"Unknown hybrid component type: {ctype}")

        return s

    raise ValueError(f'Unknown seasonality type "{stype}"')

def _generate_baseline_revenue(week_range, base_revenue, trend_slope, seasonality_config):
    t = np.arange(week_range)
    baseline = base_revenue + trend_slope * t
    if seasonality_config:
        baseline *= _seasonality(t, seasonality_config)
    return baseline

def _calculate_channel_revenue(
    channel: Channel,
    spend: np.ndarray,
    impressions: np.ndarray,
    rng: np.random.Generator,
    *,
    adstock_global: bool = True,
    saturation_global: bool = True,
) -> np.ndarray:
    """
    Compute weekly revenue contribution for a single channel.

    Pipeline:
      impressions
        → saturation (optional per-channel / global toggle)
        → adstock (optional per-channel / global toggle)
        → ROI scaling (true_roi)
        → + baseline_revenue
        → + Gaussian noise (variance from noise_variance['revenue'])

    On/off semantics (Policy A - "soft off"):
      - Fully disabled channels (`enabled=False`) contribute zero across the
        entire run: no baseline, no noise, no adstock echo.
      - Channels with per-week off ranges have impressions of zero on off
        weeks (masked upstream). Adstock carry-over from prior active weeks
        is intentionally preserved, so off-week rows may still show revenue
        from the decaying tail plus baseline + noise.
    """
    if channel.is_fully_disabled():
        return np.zeros_like(impressions, dtype=float)

    saturation_on = saturation_global and channel.saturation_enabled
    adstock_on = adstock_global and channel.adstock_enabled

    x = _saturation_fn(impressions, channel.saturation_config) if saturation_on else impressions.astype(float, copy=True)
    transformed_imp = _adstock_decay(x, channel.adstock_decay_config) if adstock_on else x
    beta = channel.true_roi * (spend / (impressions + 1e-9))
    revenue = transformed_imp * beta
    trend_slope = float(getattr(channel, "trend_slope", 0.0))
    seasonality_config = getattr(channel, "seasonality_config", None)
    revenue += _generate_baseline_revenue(len(spend), channel.baseline_revenue, trend_slope, seasonality_config)

    noise_cfg = channel.noise_variance or {}
    var_rev = float(noise_cfg.get("revenue", 0.0))
    if var_rev < 0:
        raise ValueError(
            f"Revenue noise variance for channel {channel.channel_name} must be non-negative, got {var_rev}."
        )
    if var_rev > 0:
        sigma = np.sqrt(var_rev) * np.abs(revenue)
        noise = rng.normal(loc=0.0, scale=sigma)
        revenue = revenue + noise

    return revenue

def generate_revenue(config: InputConfigurations, impressions_matrix: np.ndarray) -> np.ndarray:
    """
    Map impressions to weekly revenue per channel and total.

    Parameters
    ----------
    config : InputConfigurations
        Configuration with channel list, week_range, and RNG.
    impressions_matrix : np.ndarray, shape (num_weeks, num_channels)
        Impressions per channel per week, as produced by generate_impressions.

    Returns
    -------
    revenue_matrix : np.ndarray, shape (num_weeks, num_channels)
        Weekly revenue attributed to each channel.
    """
    num_weeks, num_channels = impressions_matrix.shape
    expected_weeks = config.get_week_range()
    channels = config.get_channel_list()

    if num_weeks != expected_weeks:
        raise ValueError(
            f"impressions_matrix has {num_weeks} weeks, config expects {expected_weeks}."
        )
    if num_channels != len(channels):
        raise ValueError(
            f"impressions_matrix has {num_channels} channels, config has {len(channels)}."
        )

    rng = config.get_rng()
    adstock_global = config.get_adstock_global()
    saturation_global = config.get_saturation_global()
    out = np.zeros((num_weeks, num_channels), dtype=float)

    for c, channel in enumerate(channels):
        cpm = float(channel.get_cpm())
        channel_spend = (impressions_matrix[:, c].astype(float) * cpm) / 1000.0
        channel_impressions = impressions_matrix[:, c].astype(float)
        out[:, c] = _calculate_channel_revenue(
            channel,
            channel_spend,
            channel_impressions,
            rng,
            adstock_global=adstock_global,
            saturation_global=saturation_global,
        )

    return out
