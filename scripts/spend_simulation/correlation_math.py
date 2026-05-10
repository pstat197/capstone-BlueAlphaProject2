"""Numpy-only helpers for spend correlation matrices (kept small for reliable imports)."""

from __future__ import annotations

import numpy as np


def safe_corrcoef(obs: np.ndarray, *, rowvar: bool = True) -> np.ndarray:
    """Pearson correlation matrix like ``np.corrcoef`` without divide/invalid warnings.

    Channels (or windows) with zero variance yield NaNs in ``np.corrcoef``; those become
    0 on off-diagonals after cleanup, with the diagonal forced to 1.
    """
    obs = np.asarray(obs, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(obs, rowvar=rowvar)
    c = np.asarray(c, dtype=np.float64).copy()
    np.fill_diagonal(c, 1.0)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 1.0)
    return c
