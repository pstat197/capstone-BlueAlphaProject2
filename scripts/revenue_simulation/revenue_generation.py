import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
from scripts.dataclasses.channel import Channel
from scripts.dataclasses.input_configurations import InputConfigurations

def _saturation_fn(impressions: np.ndarray, saturation_cfg: dict):
    """
    Apply saturation function to impressions.

    We incorporate the following:
    'hill' - Hill function: x^slope / (x^slope + K^slope)
            - slope: shape parameter
            - K: half-saturation point

    'diminishing_returns' - Square-root decay: x / (1 + beta * x)
            - beta: decay rate (of the derivative of the saturation function)

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

    if adstock_decay_type == 'geometric':
        lambda_ = adstock_decay_cfg["decay_rate"]
        lag = adstock_decay_cfg["lag"]
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


def _calculate_channel_revenue(
    channel: Channel,
    spend: np.ndarray,
    impressions: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
) -> np.ndarray:
    """
    Compute weekly revenue contribution for a single channel.

    Pipeline:
      -> saturation+adstock - applying saturation and adstock functions to impressions
      -> derive beta from true_roi, spend, and alpha
      -> ROI scaling - saturated_impressions * beta
      -> baseline - added baseline_revenue each week
      -> noise - additive gaussian, variance = noise_variance

    returns shape (num_weeks,)
    """

    adstocked_imp = _adstock_decay(impressions, channel.adstock_decay_cfg)
    transformed_imp = _saturation_fn(adstocked_imp, channel.saturation_cfg)

    expected_imp = alpha * spend
    expected_imp = _adstock_decay(expected_imp, channel.adstock_decay_cfg)
    expected_transformed_imp = _saturation_fn(expected_imp, channel.saturation_cfg)

    beta = channel.true_roi * spend / expected_transformed_imp
    #print(f"beta: {beta}")
    revenue = transformed_imp * beta
    revenue += channel.baseline_revenue
    revenue += rng.normal(loc = 0.0, scale = np.sqrt(channel.noise_variance)*transformed_imp, size = len(impressions))
    return revenue

def generate_revenue(config: InputConfigurations, spend_matrix: np.ndarray, impressions_matrix: np.ndarray, coefficients: list[float]) -> np.ndarray:
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

    """
   assert num_channels == config.num_channels, (
        f'impressions_matrix has {num_channels} channels, '
        f'config expects {config.num_channels}.'
    )
    """

    assert num_weeks == config.num_weeks, (
        f'impressions_matrix has {num_weeks} weeks,'
        f'config expects {config.num_weeks}'
    )

    weekly_revenue = np.zeros((num_weeks, num_channels), dtype=float)

    for c, channel in enumerate(config.channel_list):
        channel_spend = spend_matrix[:, c].astype(float)
        channel_impressions = impressions_matrix[:, c].astype(float)
        weekly_revenue[:, c] = _calculate_channel_revenue(channel, channel_spend, channel_impressions, np.random.default_rng(), coefficients[c])

    total_revenue = np.sum(weekly_revenue, axis=1)
    weekly_revenue = np.concatenate((weekly_revenue, total_revenue.reshape(-1, 1)), axis=1)

    return weekly_revenue
"""
Model Assumptions:
    - true_ROI is constant across all weeks
    - baseline revenue is constant across all weeks
    
"""  

# TEST CASES

"""
ADSTOCK DECAY TEST CASES
print(_adstock_decay(np.array([1, 2, 3, 4, 5]), {"type": "geometric", "decay_rate": 0.5, "lag": 3}))

print(_adstock_decay(np.array([1, 2, 3, 4, 5]), {"type": "weighted", "weights": [0.5, 0.3, 0.2]}))

print(_adstock_decay(np.array([1, 2, 3, 4, 5]), {"type": "linear"}))

SATURATION TEST CASES
print(_saturation_fn(np.array([1, 2, 3, 4, 5, 10, 999999999]), {"type": "hill", "slope": 1.0, "K": 10}))

print(_saturation_fn(np.array([1, 2, 3, 4, 5, 10]), {"type": "hill", "slope": 2.0, "K": 999999999}))

print(_saturation_fn(np.array([1, 2, 3, 4, 5, 20, 999999999]), {"type": "diminishing_returns", "beta": 1}))

CHANNEL REVENUE TEST CASES
decay_cfg = {"type": "geometric", "decay_rate": 0.8, "lag": 3}
saturation_cfg = {"type": "hill", "slope": 1.0, "K": 60000}

channel_cfg= Channel(channel_name="Test", true_roi=6.0, spend_range=[10000000, 20000000], baseline_revenue=0.0, 
saturation_cfg=saturation_cfg, adstock_decay_cfg=decay_cfg, noise_variance=0.0, spend_sampling_gamma_params=None)

print(_calculate_channel_revenue(channel_cfg, np.array([10000, 20000, 30000, 40000, 50000]), np.array([110000, 210000, 280000, 390000, 500000]), np.random.default_rng(), 11))

"""
spend_matrix = np.array([[1, 6, 11], [2, 7, 9], [3, 10, 17], [4, 13, 22], [5, 12, 24], [6, 14, 21]])
alphas = np.array([10, 2, 3])
noise_matrix = np.random.normal(loc = 0.0, scale = 0.1, size = spend_matrix.shape)
impressions_matrix = spend_matrix * alphas + noise_matrix
print(f"=== impressions_matrix: === \n{impressions_matrix}\n")
print(f"=== spend_matrix: === \n{spend_matrix}\n")
decay_cfg = {"type": "geometric", "decay_rate": 0, "lag": 3}
decay_cfg2 = {"type": "geometric", "decay_rate": 0, "lag": 2}
decay_cfg3 = {"type": "geometric", "decay_rate": 0.3, "lag": 1}
saturation_cfg = {"type": "hill", "slope": 1.0, "K": 40000}
saturation_cfg2 = {"type": "hill", "slope": 2.0, "K": 80000}
saturation_cfg3 = {"type": "hill", "slope": 3.0, "K": 30000}

channel_cfg= Channel(channel_name="Test1", true_roi=6.0, spend_range=[10000000, 20000000], baseline_revenue=0.0, 
saturation_cfg=saturation_cfg, adstock_decay_cfg=decay_cfg, noise_variance=0.0, spend_sampling_gamma_params=None)
channel_cfg2 = Channel(channel_name="Test2", true_roi=5.0, spend_range=[10000000, 20000000], baseline_revenue=0.0, 
saturation_cfg=saturation_cfg2, adstock_decay_cfg=decay_cfg2, noise_variance=0.0, spend_sampling_gamma_params=None)
channel_cfg3 = Channel(channel_name="Test3", true_roi=2.0, spend_range=[10000000, 20000000], baseline_revenue=0.0, 
saturation_cfg=saturation_cfg3, adstock_decay_cfg=decay_cfg3, noise_variance=0.0, spend_sampling_gamma_params=None)

config = InputConfigurations(run_identifier="Test", num_weeks=6, channel_list=[channel_cfg, channel_cfg2, channel_cfg3])
print(generate_revenue(config, spend_matrix, impressions_matrix, alphas))

