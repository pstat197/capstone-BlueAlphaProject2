import numpy as np
import pandas as pd

from scripts.config import DEFAULT_CONFIG
from scripts.random_noise import add_random_noise
from scripts.spend_distribution import get_spend_and_impressions



def generate_synthetic_data(config: dict):
    """
    Generate a synthetic MMM dataset from a config dict.

    Expected config keys:
      - seed: int
      - weeks: int
      - channels: list[{"name": str, "roi": float, ...}]

    Returns:
      df: DataFrame with columns [week, revenue, <channel>_spend, <channel>_impressions, ...]
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
    impressions_matrix = np.zeros((weeks, len(channels_cfg)), dtype=np.int64)

    for j, channel_cfg in enumerate(channels_cfg):
        for w in range(weeks):
            spend, impressions = get_spend_and_impressions(rng)
            spend_matrix[w, j] = spend
            impressions_matrix[w, j] = int(impressions)

    # --- Revenue from spend and true ROI ---
    revenue = spend_matrix @ true_rois

    # revenue noise
    for w in range(weeks):
        revenue[w] = add_random_noise(
            revenue[w],
            rng=rng,
        )

    # --- DataFrame: week, revenue, <channel>_spend, <channel>_impressions, ... ---
    records = []
    for w in range(weeks):
        row = {"week": w + 1, "revenue": revenue[w]}
        for j, name in enumerate(channel_names):
            row[f"{name}_spend"] = spend_matrix[w, j]
            row[f"{name}_impressions"] = impressions_matrix[w, j]
        records.append(row)
    df = pd.DataFrame(records)

    return df, ground_truth


if __name__ == "__main__":
    df, gt = generate_synthetic_data(DEFAULT_CONFIG)

    print("Ground Truth ROIs (per channel):")
    for ch, roi in gt.items():
        print(f"  {ch}: {roi:.4f}")

    print("\nFirst 5 rows of generated data:\n")
    print(df.head(5).to_string(index=False))