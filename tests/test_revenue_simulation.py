"""
Tests for revenue simulation (generate_revenue).
Run from project root: python -m tests.test_revenue_simulation  or  python test.py
"""
from pathlib import Path

import numpy as np

from scripts.config.loader import load_config
from scripts.impressions_simulation.impressions_generation import generate_impressions
from scripts.revenue_simulation.revenue_generation import (
    _adstock_decay,
    _saturation_fn,
    generate_revenue,
    generate_subscriptions,
)
from scripts.spend_simulation.spend_generation import generate_spend


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_example_config():
    example_path = _project_root() / "example.yaml"
    assert example_path.exists(), f"example.yaml not found at {example_path}"
    return load_config(str(example_path))


def test_generate_revenue_shape_and_finite():
    """generate_revenue returns (num_weeks, num_channels) with finite values."""
    config = _load_example_config()
    spend = generate_spend(config)
    impressions = generate_impressions(config, spend)

    revenue = generate_revenue(config, impressions)

    assert isinstance(revenue, np.ndarray)
    assert revenue.ndim == 2
    assert revenue.shape == (config.get_week_range(), len(config.get_channel_list()))
    assert np.all(np.isfinite(revenue))


def test_linear_saturation_scales_by_slope():
    x = np.array([1.0, 2.0, 3.0])
    out = _saturation_fn(x, {"type": "linear", "slope": 2.0})
    np.testing.assert_array_almost_equal(out, np.array([2.0, 4.0, 6.0]))
    same = _saturation_fn(x, {"type": "linear"})
    np.testing.assert_array_almost_equal(same, x)


def test_linear_adstock_uniform_moving_average_when_lag_positive():
    x = np.array([0.0, 0.0, 9.0, 0.0, 0.0])
    out = _adstock_decay(x, {"type": "linear", "lag": 2})
    w = np.ones(3) / 3.0
    expected = np.convolve(x, w, mode="full")[: len(x)]
    np.testing.assert_array_almost_equal(out, expected)


def test_linear_adstock_lag_zero_is_identity():
    x = np.array([1.0, 5.0, 2.0])
    out = _adstock_decay(x, {"type": "linear", "lag": 0})
    np.testing.assert_array_almost_equal(out, x)


def test_generate_revenue_reproducible_with_seed():
    """With a fixed seed in the config, two runs produce identical revenue vectors."""
    example_path = _project_root() / "example.yaml"
    config1 = load_config(str(example_path))
    spend1 = generate_spend(config1)
    imps1 = generate_impressions(config1, spend1)
    rev1 = generate_revenue(config1, imps1)

    config2 = load_config(str(example_path))
    spend2 = generate_spend(config2)
    imps2 = generate_impressions(config2, spend2)
    rev2 = generate_revenue(config2, imps2)

    np.testing.assert_array_almost_equal(spend1, spend2)
    np.testing.assert_array_almost_equal(imps1, imps2)
    np.testing.assert_array_almost_equal(rev1, rev2)
    assert rev1.shape[1] == len(config1.get_channel_list())


def test_generate_subscriptions_shape_and_integer():
    """generate_subscriptions returns (num_weeks, num_channels) with non-negative integers."""
    config = _load_example_config()
    spend = generate_spend(config)
    impressions = generate_impressions(config, spend)
    subs = generate_subscriptions(config, impressions)

    assert isinstance(subs, np.ndarray)
    assert subs.ndim == 2
    assert subs.shape == (config.get_week_range(), len(config.get_channel_list()))
    assert np.all(subs >= 0), "Subscriptions must be non-negative"
    assert subs.dtype in (np.int64, np.int32, int), f"Expected integer dtype, got {subs.dtype}"


def test_subscriptions_zero_conversion_rate():
    """With conversion_rate=0 and baseline_subscriptions=0, output is all zeros (no noise)."""
    from scripts.synth_input_classes.channel import Channel
    from scripts.synth_input_classes.input_configurations import InputConfigurations
    from scripts.config.noise import init_rng

    init_rng(42)
    ch = Channel(
        channel_name="Test",
        true_roi=1.0,
        spend_range=[100, 1000],
        baseline_revenue=0,
        saturation_config={"type": "linear", "slope": 1.0},
        adstock_decay_config={"type": "linear", "lag": 0},
        spend_sampling_gamma_params={"shape": 2.0, "scale": 500},
        noise_variance={"impression": 0.0, "revenue": 0.0, "subscription": 0.0},
        cpm=10.0,
        conversion_rate=0.0,
        baseline_subscriptions=0,
    )
    config = InputConfigurations(
        run_identifier="test",
        week_range=10,
        channel_list=[ch],
        seed=42,
        kpi_mode="both",
    )
    impressions = np.ones((10, 1)) * 1000
    subs = generate_subscriptions(config, impressions)
    assert np.all(subs == 0), f"Expected all zeros, got {subs}"


def main():
    print("Revenue simulation tests...")
    test_generate_revenue_shape_and_finite()
    test_linear_saturation_scales_by_slope()
    test_linear_adstock_uniform_moving_average_when_lag_positive()
    test_linear_adstock_lag_zero_is_identity()
    test_generate_revenue_reproducible_with_seed()
    test_generate_subscriptions_shape_and_integer()
    test_subscriptions_zero_conversion_rate()
    print("Revenue simulation tests passed.")


if __name__ == "__main__":
    main()
