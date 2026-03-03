import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from synth_input_classes.input_configurations import InputConfigurations

def generate_impressions(config: InputConfigurations, spend_matrix: np.ndarray, cpm: list) -> np.ndarray:
    weeks = config.get_week_range()
    channels = len(config.get_channel_list())
    var = []
    for i in config.get_channel_list():
        var.append(i.get_noise_variance[0])
    
    std = np.sqrt(var)

    std_scaled = std * spend_matrix

    means = np.zeros((weeks, channels))

    noise = np.random.normal(loc = means, scale = std_scaled, size = (weeks, channels))

    impressions = ((spend_matrix / cpm) * 1000) + noise

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
