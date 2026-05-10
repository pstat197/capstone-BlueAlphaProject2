"""
Tests for revenue simulation (generate_revenue).
Run from project root: python -m tests.test_revenue_simulation  or  python test.py
"""
import numpy as np

from scripts.config.loader import load_config, load_config_from_dict
from scripts.impressions_simulation.impressions_generation import generate_impressions
from scripts.revenue_simulation.revenue_generation import (
    _adstock_decay,
    _outcome_revenue_noise,
    _saturation_fn,
    generate_revenue,
)
from scripts.spend_simulation.spend_generation import generate_spend


def test_generate_revenue_shape_and_finite(example_config):
    """generate_revenue returns media matrix + total vector with finite values."""
    config = example_config
    spend = generate_spend(config)
    impressions = generate_impressions(config, spend)

    revenue = generate_revenue(config, impressions)

    assert revenue.channel_media_revenue.ndim == 2
    assert revenue.channel_media_revenue.shape == (
        config.get_week_range(),
        len(config.get_channel_list()),
    )
    assert revenue.total_revenue.shape == (config.get_week_range(),)
    assert np.all(np.isfinite(revenue.channel_media_revenue))
    assert np.all(np.isfinite(revenue.total_revenue))


def test_outcome_revenue_noise_is_homoskedastic():
    """Weekly shock σ = √(variance); std of residuals ~ σ regardless of revenue level."""
    var_ = 4.0
    sigma = 2.0
    n = 8000
    low = np.full(n, 50.0)
    high = np.full(n, 1.0e6)
    r0 = np.random.default_rng(0)
    r1 = np.random.default_rng(1)
    res_low = _outcome_revenue_noise(low, r0, revenue_variance=var_) - low
    res_high = _outcome_revenue_noise(high, r1, revenue_variance=var_) - high
    assert abs(float(np.std(res_low)) - sigma) < 0.05
    assert abs(float(np.std(res_high)) - sigma) < 0.05


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


def test_default_adstock_is_geometric_when_type_omitted():
    """Omitted ``type`` uses geometric decay (Meridian-style), not uniform linear MA."""
    x = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    out_default = _adstock_decay(x, {"lag": 2, "lambda": 0.5})
    out_geo = _adstock_decay(x, {"type": "geometric", "lag": 2, "lambda": 0.5})
    np.testing.assert_array_almost_equal(out_default, out_geo)
    # Geometric kernel favors recent weeks vs flat 1/3, 1/3, 1/3 linear MA.
    out_lin = _adstock_decay(x, {"type": "linear", "lag": 2})
    assert not np.allclose(out_default, out_lin)


def test_binomial_adstock_meridian_kernel_l2_alpha_half():
    """Meridian binomial: L=2, α=0.5 → α*=1, weights ∝ 1, 2/3, 1/3."""
    x = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    out = _adstock_decay(x, {"type": "binomial", "lambda": 0.5, "lag": 2})
    raw = np.array([1.0, 2.0 / 3.0, 1.0 / 3.0])
    expected = np.convolve(x, raw / raw.sum(), mode="full")[: len(x)]
    np.testing.assert_array_almost_equal(out, expected)


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
    np.testing.assert_array_almost_equal(rev1.channel_media_revenue, rev2.channel_media_revenue)
    np.testing.assert_array_almost_equal(rev1.total_revenue, rev2.total_revenue)
    assert rev1.channel_media_revenue.shape[1] == len(config1.get_channel_list())


def test_media_transform_order_swaps_pipeline():
    """saturation_first vs adstock_first changes revenue when both curves are active."""
    ch = {
        "channel_name": "A",
        "true_roi": 2.0,
        "spend_range": [0, 0],
        "baseline_revenue": 0.0,
        "trend_slope": 0.0,
        "seasonality_config": {},
        "saturation_config": {"type": "hill", "slope": 1.0, "K": 5000.0},
        "adstock_decay_config": {"type": "geometric", "lambda": 0.6, "lag": 4},
        "spend_sampling_gamma_params": {"shape": 1.0, "scale": 1.0},
        "noise_variance": {"impression": 0.0, "revenue": 0.0},
        "cpm": 10.0,
    }
    common = {
        "run_identifier": "order_test",
        "week_range": 12,
        "seed": 42,
        "channel_list": [{"channel": ch}],
    }
    cfg_ad = load_config_from_dict({**common, "media_transform_order": "adstock_first"})
    cfg_sat = load_config_from_dict({**common, "media_transform_order": "saturation_first"})
    rng = np.random.default_rng(0)
    impressions = rng.integers(100, 5000, size=(12, 1)).astype(float)
    r_ad = generate_revenue(cfg_ad, impressions)
    r_sat = generate_revenue(cfg_sat, impressions)
    assert not np.allclose(r_ad.channel_media_revenue, r_sat.channel_media_revenue)


def test_generate_revenue_includes_trend_and_seasonality_baseline():
    """Outcome seasonality/trend affect total revenue; media-only columns stay zero with zero impressions."""
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
    np.testing.assert_allclose(rev.total_revenue, expected)
    np.testing.assert_allclose(rev.channel_media_revenue[:, 0], np.zeros(4))


def main():
    print("Revenue simulation tests...")
    test_generate_revenue_shape_and_finite()
    test_outcome_revenue_noise_is_homoskedastic()
    test_linear_saturation_scales_by_slope()
    test_linear_adstock_uniform_moving_average_when_lag_positive()
    test_linear_adstock_lag_zero_is_identity()
    test_generate_revenue_reproducible_with_seed()
    test_media_transform_order_swaps_pipeline()
    test_generate_revenue_includes_trend_and_seasonality_baseline()
    print("Revenue simulation tests passed.")


if __name__ == "__main__":
    main()
