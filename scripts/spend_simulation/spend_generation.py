import numpy as np
from scripts.synth_input_classes.input_configurations import InputConfigurations

# Default gamma params if channel config is missing shape/scale (from default.yaml)
DEFAULT_GAMMA_SHAPE = 2.5
DEFAULT_GAMMA_SCALE = 1000.0


def generate_spend(config: InputConfigurations) -> np.ndarray:
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