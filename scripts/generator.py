from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.config import DEFAULT_CONFIG
from scripts.random_noise import add_random_noise

def draw_from_dist(rng: np.random.Generator) -> float: #todo temporary
    """
    Temporary spend sampler until the real distribution is implemented.
    Currently: uniform(8000, 15000) dollars.
    """
    return rng.uniform(8, 15) * 1000


def generate_synthetic_data(config: dict):
    """
    Generate a synthetic MMM dataset from a config dict.

    Expected config keys:
      - seed: int
      - weeks: int
      - channels: list[{"name": str, "roi": float, ...}]

    Returns:
      df: DataFrame with columns [week, revenue, <channel spends...>]
      ground_truth: dict mapping channel -> true ROI
    """

    seed = config.get("seed", 133233)
    weeks = int(config["weeks"])
    channels_cfg = config["channels"]

    rng = np.random.default_rng(seed)

    channel_names = [c["name"] for c in channels_cfg]
    true_rois = np.array([float(c["roi"]) for c in channels_cfg])
    ground_truth = {c["name"]: float(c["roi"]) for c in channels_cfg}

    spend_matrix = np.zeros((weeks, len(channels_cfg)), dtype=float)

    for j, channel_cfg in enumerate(channels_cfg):
        for w in range(weeks):
            spend = draw_from_dist(rng)

            # per channel spend noise
            spend = add_random_noise(
                spend,
                rng=rng,
            )

            spend_matrix[w, j] = spend

    # --- Revenue from spend and true ROI ---
    revenue = spend_matrix @ true_rois

    # revenue noise
    for w in range(weeks):
        revenue[w] = add_random_noise(
            revenue[w],
            rng=rng,
        )

    # --- DataFrame ---
    df = pd.DataFrame(spend_matrix, columns=channel_names)
    df.insert(0, "revenue", revenue)
    df.insert(0, "week", np.arange(1, weeks + 1))

    return df, ground_truth


if __name__ == "__main__":
    df, gt = generate_synthetic_data(DEFAULT_CONFIG)

    print("Ground Truth ROIs (per channel):")
    for ch, roi in gt.items():
        print(f"  {ch}: {roi:.4f}")

    print("\nFirst 5 rows of generated data:\n")
    print(df.head(5).to_string(index=False))