"""
Overall / full pipeline tests (config -> spend -> impressions -> revenue -> CSV).
Run from project root: python -m tests.test_pipeline  or  python test.py
"""
from pathlib import Path
import os

import numpy as np

from scripts.config.loader import load_config_from_dict
from scripts.impressions_simulation.impressions_generation import generate_impressions
from scripts.main import construct_csv, run_simulation
from scripts.revenue_simulation.revenue_generation import generate_revenue
from scripts.spend_simulation.spend_generation import generate_spend


def test_full_pipeline_construct_csv_and_shapes(tmp_path, example_config):
    """
    End-to-end: config -> spend -> impressions -> revenue -> construct_csv.
    Validate shapes, column names, and basic properties.
    """
    config = example_config
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
        assert f"{name}_revenue" in df.columns

    # Totals equal sum across channels (within numerical tolerance)
    total_impressions = np.zeros(num_weeks)
    total_spend = np.zeros(num_weeks)
    for ch in channels:
        name = ch.get_channel_name()
        total_impressions += df[f"{name}_impressions"].to_numpy()
        total_spend += df[f"{name}_spend"].to_numpy()

    np.testing.assert_allclose(df["total_impressions"].to_numpy(), total_impressions)
    np.testing.assert_allclose(df["total_spend"].to_numpy(), total_spend)

    media_sum = np.zeros(num_weeks)
    for ch in channels:
        name = ch.get_channel_name()
        media_sum += df[f"{name}_revenue"].to_numpy()
    np.testing.assert_allclose(df["revenue"].to_numpy(), revenue.total_revenue)
    np.testing.assert_allclose(media_sum, revenue.channel_media_revenue.sum(axis=1))

    # week column is 1..num_weeks
    assert list(df["week"]) == list(range(1, num_weeks + 1))


def _pipe_channel(name: str) -> dict:
    inner = {
        "channel_name": name,
        "true_roi": 1.5,
        "spend_range": [100, 8000],
        "baseline_revenue": 200.0,
        "trend_slope": 0.0,
        "seasonality_config": {},
        "saturation_config": {"type": "linear", "slope": 1.0},
        "adstock_decay_config": {"type": "linear", "lag": 0},
        "spend_sampling_gamma_params": {"shape": 2.0, "scale": 400.0},
        "noise_variance": {"revenue": 0.0},
        "cpm": 10.0,
    }
    return {"channel": inner}


def test_full_pipeline_identical_per_channel_when_channel_list_permuted():
    """End-to-end: permuting ``channel_list`` does not change per-name series or total revenue."""
    base = {
        "run_identifier": "perm_pipe",
        "week_range": 10,
        "seed": 9001,
        "correlations": [{"channels": ["Alpha", "Zed"], "rho": 0.25}],
    }
    cfg_az = load_config_from_dict(
        {**base, "channel_list": [_pipe_channel("Alpha"), _pipe_channel("Zed")]}
    )
    cfg_za = load_config_from_dict(
        {**base, "channel_list": [_pipe_channel("Zed"), _pipe_channel("Alpha")]}
    )
    df_az, _ = run_simulation(cfg_az)
    df_za, _ = run_simulation(cfg_za)
    np.testing.assert_allclose(df_az["revenue"].to_numpy(), df_za["revenue"].to_numpy())
    for nm in ("Alpha", "Zed"):
        np.testing.assert_allclose(
            df_az[f"{nm}_spend"].to_numpy(), df_za[f"{nm}_spend"].to_numpy()
        )
        np.testing.assert_allclose(
            df_az[f"{nm}_impressions"].to_numpy(), df_za[f"{nm}_impressions"].to_numpy()
        )
        np.testing.assert_allclose(
            df_az[f"{nm}_revenue"].to_numpy(), df_za[f"{nm}_revenue"].to_numpy()
        )


def main():
    print("Pipeline tests...")
    # Use a temporary directory for any artifacts if needed in the future.
    test_full_pipeline_construct_csv_and_shapes(tmp_path=Path(os.getcwd()) / "tmp")
    test_full_pipeline_identical_per_channel_when_channel_list_permuted()
    print("Pipeline tests passed.")


if __name__ == "__main__":
    main()
