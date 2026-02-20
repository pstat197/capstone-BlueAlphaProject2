import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from synth_input_classes.input_configurations import InputConfigurations

def generate_impressions(config: InputConfigurations, spend_matrix: np.ndarray, coeffs: list) -> np.ndarray:
    weeks = config.get_week_range()
    channels = len(config.get_channel_list())
    var = []
    for i in config.get_channel_list():
        var.append(i.get_noise_variance[0])
    
    std = np.sqrt(var)

    means = np.zeros(channels)

    noise = np.random.normal(loc = means, scale = std, size = (weeks, channels))

    impressions = spend_matrix * coeffs + noise

    return impressions

# TEST

# spend_matrix = np.array([[1, 2, 3],
                        #  [4, 5, 7],
                        #  [8, 9, 6]])
# weeks = 3
# channels = 3
# var = [0, 1, 2]
# std = np.sqrt(var)
# means = [0, 0, 0]
# coefficients = [1, 2, 3]
# noise = np.random.normal(loc=means, scale = std, size = (weeks, channels))
# impressions = spend_matrix * coefficients + noise
# print(impressions)

