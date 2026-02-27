import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import scripts.dataclasses.channel as Channel
import scripts.dataclasses.input_configurations as InputConfigurations

def _saturation_fn(impressions: np.ndarray, saturation_cfg: dict):
    """
    Apply saturation function to impressions.

    We incorporate the following:
    'hill' - Hill function: x^slope / (x^slope + K^slope)
            - slope: shape parameter
            - K: half-saturation point

    'diminishing_returns' - Square-root decay: x / (1 + beta * x)
            - beta: decay rate

    'linear' - No saturation applied
            - No parameters

    impressions: shape (num_weeks,)
    returns: shape (num_weeks,) saturated values
    """
    saturation_type = saturation_cfg.get('type', 'linear')
    
    if saturation_type == 'hill':
        slope = saturation_cfg['slope']
        K = saturation_cfg['K']
        x_a = np.power(impressions, slope)
        K_a = np.power(K, slope)
        return x_a / (x_a + K_a)

    elif saturation_type == 'diminishing_returns':
        beta = saturation_cfg['beta']
        return impressions / (1.0 + beta * impressions)

    elif saturation_type == 'linear':
        return impressions.copy()

    else:
        raise ValueError(
            f'Unknown saturation_function type "{saturation_type}". '
            'Expected one of: "hill", "diminishing_returns", "linear".'
        )

def _adstock_decay(impressions: np.ndarray, adstock_decay_cfg: dict):
    """
    Apply adstock decay function to impressions.
    """
    adstock_decay_type = adstock_decay_cfg["type"]
    lambda_ = adstock_decay_cfg["decay_rate"]
    lag = adstock_decay_cfg["lag"]

    if adstock_decay_type == 'geometric':
        lag_array = np.arange(lag+1)
        decay_weights = np.power(lambda_, lag_array)
        return  np.convolve(impressions, decay_weights, mode='full')[:len(impressions)]
    elif adstock_decay_type == 'weighted':
        weights = adstock_decay_cfg.get('weights', [1.0])
        return np.convolve(impressions, weights, mode='full')[:len(impressions)]
    elif adstock_decay_type == 'linear':
        return impressions.copy()
    else:
        raise ValueError(
            f'Unknown adstock_decay_function type "{adstock_decay_type}". '
            'Expected one of: "geometric", "weighted", "linear".'
        )


def _channel_revenue(
    channel: Channel,
    impressions: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
) -> np.ndarray:
    """
    Compute weekly revenue contribution for a single channel.

    Pipeline:
      -> saturation - diminishing returns on reach
      -> ROI scaling - saturated_impressions * beta
      -> baseline - added baseline_revenue each week
      -> noise - additive gaussian, variance = noise_variance

    returns shape (num_weeks,)
    """
    saturated = _saturation_fn(impressions, channel.saturation_function)
    transformed_imp = saturated * _adstock_decay(impressions, channel.adstock_decay_cfg)

    beta = channel.true_roi / alpha
    revenue = transformed_imp * beta
    revenue += channel.baseline_revenue
    revenue += rng.normal(loc = 0.0, scale = np.sqrt(channel.noise_variance)*transformed_imp, size = len(impressions))
    return revenue

def generate_revenue(config: InputConfigurations, impressions_matrix: np.ndarray, coefficients: list[float]) -> np.ndarray:
    """
    Map impressions to total weekly revenue across all channels.

    parameters:

    config: InputConfigurations
    impressions_matrix: np.ndarray, shape (num_weeks, num_channels)
        Impressions per channel per week, as produced by generate_impressions.
    coefficients: list[float], shape (num_channels,)
        Coefficients for each channel's impressions contribution.

    returns: 

    revenue_vector: np.ndarray, shape (num_weeks,)
        Total revenue per week, across all channels.
    """
    num_weeks, num_channels = impressions_matrix.shape

    assert num_channels == config.num_channels, (
        f'impressions_matrix has {num_channels} channels, '
        f'config expects {config.num_channels}.'
    )

    assert num_weeks == config.num_weeks, (
        f'impressions_matrix has {num_weeks} weeks,'
        f'config expects {config.num_weeks}'
    )

    weekly_revenue = np.zeros(num_weeks, dtype=float)

    for c, channel in enumerate(config.channel_list):
        channel_impressions = impressions_matrix[:, c].astype(float)
        weekly_revenue += _channel_revenue(channel, channel_impressions, config.rng, coefficients[c])

    return weekly_revenue
"""
Model Assumptions:
    - true_ROI is constant across all weeks
    - baseline revenue is constant across all weeks
    
"""     
