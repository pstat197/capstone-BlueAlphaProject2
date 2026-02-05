"""Generate random noise scaled by the magnitude of a number."""

import random


def add_random_noise(x, scale: float = 0.1, min_std: float = 1e-6):
    """
    Add random Gaussian noise around a number, with variance scaled by its size.

    Larger numbers get proportionally larger noise; small numbers use min_std
    so variance never goes to zero.

    Parameters
    ----------
    x : float
        The input number.
    scale : float, optional
        Fraction of |x| to use as standard deviation (default 0.1 = 10%).
    min_std : float, optional
        Minimum standard deviation when x is near zero (default 1e-6).

    Returns
    -------
    float
        x plus random noise: x + N(0, std**2) with std = max(scale * |x|, min_std).
    """
    
    magnitude = max(abs(x), 1e-10)
    std = max(scale * magnitude, min_std)
    return x + random.gauss(0, std)
