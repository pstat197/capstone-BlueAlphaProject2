"""
Overall / full pipeline tests (config -> spend -> impressions -> revenue -> CSV).
Run from project root: python -m tests.test_pipeline  or  python test.py
"""
from pathlib import Path
import os

import numpy as np

from scripts.config.loader import load_config
from scripts.impressions_simulation.impressions_generation import generate_impressions
from scripts.main import construct_csv
from scripts.revenue_simulation.revenue_generation import generate_revenue
from scripts.spend_simulation.spend_generation import generate_spend


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_example_config():
    example_path = _project_root() / "example.yaml"
    assert example_path.exists(), f"example.yaml not found at {example_path}"
    return load_config(str(example_path))


def test_full_pipeline_construct_csv_and_shapes(tmp_path):
    """
    End-to-end: config -> spend -> impressions -> revenue -> construct_csv.
    Validate shapes, column names, and basic properties.
    """
    config = _load_example_config()
    spend = generate_spend(config)
    impressions = generate_impressions(config, spend)
    revenue = generate_revenue(config, impressions)

    df = construct_csv(config, spend, impressions, revenue)

    num_weeks = config.get_week_range()
    channels = config.get_channel_list()
    num_channels = len(channels)

    # Shape: one row per week
    assert df.shape[0] == num_weeks

    # Required columns
    assert "week" in df.columns
    assert "revenue" in df.columns
    assert "total_impressions" in df.columns
    assert "total_spend" in df.columns

    # Per-channel columns
    for ch in channels:
        name = ch.get_channel_name()
        assert f"{name}_impressions" in df.columns
        assert f"{name}_spend" in df.columns

    # Totals equal sum across channels (within numerical tolerance)
    total_impressions = np.zeros(num_weeks)
    total_spend = np.zeros(num_weeks)
    for ch in channels:
        name = ch.get_channel_name()
        total_impressions += df[f"{name}_impressions"].to_numpy()
        total_spend += df[f"{name}_spend"].to_numpy()

    np.testing.assert_allclose(df["total_impressions"].to_numpy(), total_impressions)
    np.testing.assert_allclose(df["total_spend"].to_numpy(), total_spend)

    # week column is 1..num_weeks
    assert list(df["week"]) == list(range(1, num_weeks + 1))


def main():
    print("Pipeline tests...")
    # Use a temporary directory for any artifacts if needed in the future.
    test_full_pipeline_construct_csv_and_shapes(tmp_path=Path(os.getcwd()) / "tmp")
    print("Pipeline tests passed.")


if __name__ == "__main__":
    main()
