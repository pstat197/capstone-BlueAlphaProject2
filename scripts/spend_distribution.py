import numpy as np


def spend_distribution(rng: np.random.Generator) -> float: #todo temporary
    """
    Temporary spend sampler until the real distribution is implemented.
    Currently: uniform(8000, 15000) dollars.
    """
    return rng.uniform(8, 15) * 1000