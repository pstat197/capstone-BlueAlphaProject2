import numpy as np


def get_spend_and_impressions(rng: np.random.Generator) -> tuple[float, int]: #todo temporary
    """
    Temporary spend and impressions sampler until the real distribution is implemented.
    Currently: uniform(8000, 15000) dollars and uniform(1000, 2000) impressions.
    """
    return rng.uniform(8, 15) * 1000, rng.uniform(1000, 2000)