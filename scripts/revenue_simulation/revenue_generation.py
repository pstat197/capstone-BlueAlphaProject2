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
      - 'linear':          return impressions unchanged
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
        return impressions.astype(float, copy=True)

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
      - 'linear': no adstock (return impressions)
    """
    adstock_config = adstock_config or {}
    adstock_type = adstock_config.get("type", "linear")

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

    if adstock_type == "linear":
        return impressions.astype(float, copy=True)

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
    cpm: float
) -> np.ndarray:
    """
    Compute weekly revenue contribution for a single channel.

    Pipeline:
      impressions
        → saturation (diminishing returns)
        → adstock (carry-over effects)
        → ROI scaling (beta = true_roi * spend / expected_transformed_imp(alpha, spend))
        → + baseline_revenue
        → + Gaussian noise (variance from noise_variance['revenue'])
    """

    adstocked_imp = _adstock_decay(impressions, channel.adstock_decay_config)
    transformed_imp = _saturation_fn(adstocked_imp, channel.saturation_config)

    expected_imp = spend / cpm * 1000
    expected_imp = _adstock_decay(expected_imp, channel.adstock_decay_cfg)
    expected_transformed_imp = _saturation_fn(expected_imp, channel.saturation_cfg)

    beta = channel.true_roi * spend / expected_transformed_imp
    revenue = transformed_imp * beta
    revenue += _generate_baseline_revenue(len(spend), channel.baseline_revenue, channel.trend_slope, channel.seasonality_config)

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

def generate_revenue(config: InputConfigurations, spend_matrix: np.ndarray, impressions_matrix: np.ndarray, cpm_list: list[float]) -> np.ndarray:
    """
    Map impressions to total weekly revenue across all channels.

    Parameters
    ----------
    config : InputConfigurations
        Configuration with channel list, week_range, and RNG.
    spend_matrix : np.ndarray, shape (num_weeks, num_channels)
        Spend per channel per week, as produced by generate_spend.
    impressions_matrix : np.ndarray, shape (num_weeks, num_channels)
        Impressions per channel per week, as produced by generate_impressions.
    coefficients : list[float]        
        Coefficients for each channel's impressions contribution.

    Returns
    -------
    revenue_matrix : np.ndarray, shape (num_weeks, num_channels+1)
        Weekly revenue across all channels, appended with total revenue in the last column.
    """
    num_weeks, num_channels = impressions_matrix.shape
    expected_weeks = config.week_range()
    channels = config.channel_list()

    if num_weeks != expected_weeks:
        raise ValueError(
            f"impressions_matrix has {num_weeks} weeks, config expects {expected_weeks}."
        )
    if num_channels != len(channels):
        raise ValueError(
            f"impressions_matrix has {num_channels} channels, config has {len(channels)}."
        )

    rng = config.rng()
    weekly_revenue = np.zeros(num_weeks, dtype=float)

    for c, channel in enumerate(channels):
        channel_spend = spend_matrix[:, c].astype(float)
        channel_impressions = impressions_matrix[:, c].astype(float)
        weekly_revenue[:, c] = _calculate_channel_revenue(channel, channel_spend, channel_impressions, rng, cpm_list[c])

    total_revenue = np.sum(weekly_revenue, axis=1)
    weekly_revenue = np.concatenate((weekly_revenue, total_revenue.reshape(-1, 1)), axis=1)

    return weekly_revenue
