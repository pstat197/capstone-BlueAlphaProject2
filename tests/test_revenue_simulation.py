"""
Tests for revenue simulation (generate_revenue).
Run from project root: python -m tests.test_revenue_simulation  or  python test.py
"""
import numpy as np

from scripts.config.loader import load_config, load_config_from_dict
from scripts.impressions_simulation.impressions_generation import generate_impressions
from scripts.revenue_simulation.revenue_generation import (
    _adstock_decay,
    _saturation_fn,
    generate_revenue,
)
from scripts.spend_simulation.spend_generation import generate_spend


def test_generate_revenue_shape_and_finite(example_config):
    """generate_revenue returns (num_weeks, num_channels) with finite values."""
    config = example_config
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


def test_generate_revenue_reproducible_with_seed(example_config_path):
    """With a fixed seed in the config, two runs produce identical revenue vectors."""
    config1 = load_config(str(example_config_path))
    spend1 = generate_spend(config1)
    imps1 = generate_impressions(config1, spend1)
    rev1 = generate_revenue(config1, imps1)

    config2 = load_config(str(example_config_path))
    spend2 = generate_spend(config2)
    imps2 = generate_impressions(config2, spend2)
    rev2 = generate_revenue(config2, imps2)

    np.testing.assert_array_almost_equal(spend1, spend2)
    np.testing.assert_array_almost_equal(imps1, imps2)
    np.testing.assert_array_almost_equal(rev1, rev2)
    assert rev1.shape[1] == len(config1.get_channel_list())


def test_generate_revenue_includes_trend_and_seasonality_baseline():
    """Seasonality/trend fields from config affect baseline revenue end-to-end."""
    cfg = load_config_from_dict(
        {
            "run_identifier": "SeasonalityE2E",
            "week_range": 4,
            "seed": 123,
            "channel_list": [
                {
                    "channel": {
                        "channel_name": "A",
                        "true_roi": 0.0,
                        "spend_range": [0, 0],
                        "baseline_revenue": 100.0,
                        "trend_slope": 10.0,
                        "seasonality_config": {
                            "type": "categorical",
                            "pattern": [1.0, 2.0],
                        },
                        "saturation_config": {"type": "linear", "slope": 1.0},
                        "adstock_decay_config": {"type": "linear", "lag": 0},
                        "spend_sampling_gamma_params": {"shape": 1.0, "scale": 1.0},
                        "noise_variance": {"impression": 0.0, "revenue": 0.0},
                        "cpm": 10.0,
                    }
                }
            ],
        }
    )
    impressions = np.zeros((4, 1), dtype=float)
    rev = generate_revenue(cfg, impressions)
    expected = np.array([100.0, 220.0, 120.0, 260.0])
    np.testing.assert_allclose(rev[:, 0], expected)


def main():
    print("Revenue simulation tests...")
    test_generate_revenue_shape_and_finite()
    test_linear_saturation_scales_by_slope()
    test_linear_adstock_uniform_moving_average_when_lag_positive()
    test_linear_adstock_lag_zero_is_identity()
    test_generate_revenue_reproducible_with_seed()
    test_generate_revenue_includes_trend_and_seasonality_baseline()
    print("Revenue simulation tests passed.")


if __name__ == "__main__":
    main()
