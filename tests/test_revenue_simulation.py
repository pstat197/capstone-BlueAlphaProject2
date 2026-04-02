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
)
from scripts.spend_simulation.spend_generation import generate_spend


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_example_config():
    example_path = _project_root() / "example.yaml"
    assert example_path.exists(), f"example.yaml not found at {example_path}"
    return load_config(str(example_path))


def test_generate_revenue_shape_and_finite():
    """generate_revenue returns a 1D vector of length num_weeks with finite values."""
    config = _load_example_config()
    spend = generate_spend(config)
    impressions = generate_impressions(config, spend)

    revenue = generate_revenue(config, impressions)

    assert isinstance(revenue, np.ndarray)
    assert revenue.ndim == 1
    assert revenue.shape[0] == config.get_week_range()
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


def main():
    print("Revenue simulation tests...")
    test_generate_revenue_shape_and_finite()
    test_linear_saturation_scales_by_slope()
    test_linear_adstock_uniform_moving_average_when_lag_positive()
    test_linear_adstock_lag_zero_is_identity()
    test_generate_revenue_reproducible_with_seed()
    print("Revenue simulation tests passed.")


if __name__ == "__main__":
    main()
