"""
Tests for spend generation (gamma sampling per channel/week).
Run from project root: python -m tests.test_spend_generation  or  python test.py
"""
from pathlib import Path

import numpy as np

from scripts.synth_input_classes.input_configurations import InputConfigurations
from scripts.config.loader import load_config
from scripts.config.noise import init_rng
from scripts.spend_simulation.spend_generation import generate_spend


def _project_root():
    return Path(__file__).resolve().parent.parent


def _load_example_config():
    example_path = _project_root() / "example.yaml"
    assert example_path.exists(), f"example.yaml not found at {example_path}"
    return load_config(str(example_path))


def test_generate_spend_shape():
    """generate_spend returns matrix of shape (num_weeks, num_channels)."""
    config = _load_example_config()
    matrix = generate_spend(config)
    assert isinstance(matrix, np.ndarray)
    assert matrix.ndim == 2
    assert matrix.shape[0] == config.get_week_range()
    assert matrix.shape[1] == len(config.get_channel_list())


def test_generate_spend_reproducible():
    """Same seed produces identical spend matrix when config is reloaded and generate_spend called."""
    import tempfile
    yaml_content = (
        "run_identifier: SpendRepro\nseed: 999\nweek_range: 4\n"
        "channel_list:\n  - channel:\n      channel_name: A\n"
        "      spend_sampling_gamma_params: { shape: 2.0, scale: 500 }\n"
        "      spend_range: [100, 20000]\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp = f.name
    try:
        config1 = load_config(tmp)
        matrix1 = generate_spend(config1)
        config2 = load_config(tmp)
        matrix2 = generate_spend(config2)
        np.testing.assert_array_almost_equal(matrix1, matrix2)
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_generate_spend_in_spend_range():
    """Each channel's column lies within that channel's spend_range."""
    config = _load_example_config()
    matrix = generate_spend(config)
    channels = config.get_channel_list()
    for c in range(matrix.shape[1]):
        low, high = channels[c].get_spend_range()[0], channels[c].get_spend_range()[1]
        assert np.all(matrix[:, c] >= low), f"Channel {c} has values below spend_range min"
        assert np.all(matrix[:, c] <= high), f"Channel {c} has values above spend_range max"


def test_generate_spend_non_negative_finite():
    """All spend entries are non-negative and finite."""
    config = _load_example_config()
    matrix = generate_spend(config)
    assert np.all(matrix >= 0)
    assert np.all(np.isfinite(matrix))


def test_generate_spend_default_gamma_params():
    """Channel with missing shape/scale uses defaults and does not crash."""
    init_rng(100)
    data = {
        "run_identifier": "DefaultGamma",
        "week_range": 3,
        "seed": 100,
        "channel_list": [
            {
                "channel": {
                    "channel_name": "PartialGamma",
                    "spend_sampling_gamma_params": {},
                    "spend_range": [0, 100000],
                    "true_roi": 1.0,
                    "baseline_revenue": 0,
                    "saturation_config": {"type": "linear", "slope": 1.0, "K": 50000.0, "beta": 0.5},
                    "adstock_decay_config": {"type": "linear", "lambda": 0.5, "lag": 10, "weights": [1.0]},
                    "noise_variance": {},
                }
            }
        ],
    }
    config = InputConfigurations.from_yaml_dict(data)
    matrix = generate_spend(config)
    assert matrix.shape == (3, 1)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0)
    assert np.all(matrix <= 100000)


def test_generate_spend_clipping():
    """Spend is clipped to spend_range; narrow range [100, 100] yields all 100."""
    init_rng(77)
    data = {
        "run_identifier": "Clip",
        "week_range": 5,
        "seed": 77,
        "channel_list": [
            {
                "channel": {
                    "channel_name": "Fixed",
                    "spend_sampling_gamma_params": {"shape": 2.5, "scale": 1000},
                    "spend_range": [100, 100],
                    "true_roi": 1.0,
                    "baseline_revenue": 0,
                    "saturation_config": {"type": "linear", "slope": 1.0, "K": 50000.0, "beta": 0.5},
                    "adstock_decay_config": {"type": "linear", "lambda": 0.5, "lag": 10, "weights": [1.0]},
                    "noise_variance": {},
                }
            }
        ],
    }
    config = InputConfigurations.from_yaml_dict(data)
    matrix = generate_spend(config)
    assert matrix.shape == (5, 1)
    np.testing.assert_array_almost_equal(matrix[:, 0], np.full(5, 100.0))


def test_generate_spend_single_week_single_channel():
    """Edge case: 1 week, 1 channel gives shape (1, 1)."""
    init_rng(0)
    data = {
        "run_identifier": "Tiny",
        "week_range": 1,
        "seed": 0,
        "channel_list": [
            {
                "channel": {
                    "channel_name": "Only",
                    "spend_sampling_gamma_params": {"shape": 1.0, "scale": 100},
                    "spend_range": [0, 10000],
                    "true_roi": 1.0,
                    "baseline_revenue": 0,
                    "saturation_config": {"type": "linear", "slope": 1.0, "K": 50000.0, "beta": 0.5},
                    "adstock_decay_config": {"type": "linear", "lambda": 0.5, "lag": 10, "weights": [1.0]},
                    "noise_variance": {},
                }
            }
        ],
    }
    config = InputConfigurations.from_yaml_dict(data)
    matrix = generate_spend(config)
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] >= 0 and matrix[0, 0] <= 10000 and np.isfinite(matrix[0, 0])


def _make_correlated_config(seed=42, rho=0.7, num_weeks=200):
    """Helper: build a 2-channel config with a correlation entry."""
    init_rng(seed)
    data = {
        "run_identifier": "CorrTest",
        "week_range": num_weeks,
        "seed": seed,
        "channel_list": [
            {"channel": {
                "channel_name": "Search",
                "spend_sampling_gamma_params": {"shape": 2.5, "scale": 1000},
                "spend_range": [100, 100000],
                "true_roi": 2.0,
                "baseline_revenue": 5000,
                "saturation_config": {"type": "linear", "slope": 1.0, "K": 50000.0, "beta": 0.5},
                "adstock_decay_config": {"type": "linear", "lambda": 0.5, "lag": 10, "weights": [1.0]},
                "noise_variance": {"impression": 0.02, "revenue": 0.1},
                "cpm": 10.0,
            }},
            {"channel": {
                "channel_name": "Social",
                "spend_sampling_gamma_params": {"shape": 2.0, "scale": 800},
                "spend_range": [100, 100000],
                "true_roi": 1.5,
                "baseline_revenue": 3000,
                "saturation_config": {"type": "linear", "slope": 1.0, "K": 50000.0, "beta": 0.5},
                "adstock_decay_config": {"type": "linear", "lambda": 0.5, "lag": 10, "weights": [1.0]},
                "noise_variance": {"impression": 0.02, "revenue": 0.1},
                "cpm": 15.0,
            }},
        ],
        "correlations": [
            {"channels": ["Search", "Social"], "rho": rho},
        ],
    }
    return InputConfigurations.from_yaml_dict(data)


def test_correlated_spend_shape():
    """Correlated path produces correct (T, C) shape."""
    config = _make_correlated_config(num_weeks=52)
    matrix = generate_spend(config)
    assert matrix.shape == (52, 2)


def test_correlated_spend_positive_finite():
    """All correlated spend values are positive and finite."""
    config = _make_correlated_config()
    matrix = generate_spend(config)
    assert np.all(matrix > 0)
    assert np.all(np.isfinite(matrix))


def test_correlated_spend_in_spend_range():
    """Correlated spend is clipped to each channel's spend_range."""
    config = _make_correlated_config()
    matrix = generate_spend(config)
    channels = config.get_channel_list()
    for c in range(matrix.shape[1]):
        low, high = channels[c].get_spend_range()
        assert np.all(matrix[:, c] >= low)
        assert np.all(matrix[:, c] <= high)


def test_correlated_spend_reproducible():
    """Same seed yields identical correlated spend matrices."""
    config1 = _make_correlated_config(seed=123)
    matrix1 = generate_spend(config1)
    config2 = _make_correlated_config(seed=123)
    matrix2 = generate_spend(config2)
    np.testing.assert_array_almost_equal(matrix1, matrix2)


def test_correlated_spend_shows_correlation():
    """With rho=0.8 and enough samples, Pearson r of log-spend should be
    significantly positive (> 0.4). This is the key test that proves
    the MVN draw actually induces correlation between channels."""
    config = _make_correlated_config(seed=7, rho=0.8, num_weeks=500)
    matrix = generate_spend(config)
    log_spend = np.log(matrix)
    r = np.corrcoef(log_spend[:, 0], log_spend[:, 1])[0, 1]
    assert r > 0.4, f"Expected strong positive correlation, got r={r:.3f}"


def test_correlated_spend_zero_rho_near_independent():
    """With rho=0.0, the correlation should be near zero (|r| < 0.2)."""
    config = _make_correlated_config(seed=99, rho=0.0, num_weeks=500)
    matrix = generate_spend(config)
    log_spend = np.log(matrix)
    r = np.corrcoef(log_spend[:, 0], log_spend[:, 1])[0, 1]
    assert abs(r) < 0.2, f"Expected near-zero correlation with rho=0, got r={r:.3f}"


def test_no_correlations_uses_independent_path():
    """Empty correlations list falls back to gamma sampling (independent)."""
    init_rng(55)
    data = {
        "run_identifier": "IndepFallback",
        "week_range": 10,
        "seed": 55,
        "channel_list": [
            {"channel": {
                "channel_name": "A",
                "spend_sampling_gamma_params": {"shape": 2.0, "scale": 500},
                "spend_range": [100, 20000],
                "true_roi": 1.0,
                "baseline_revenue": 0,
                "saturation_config": {"type": "linear", "slope": 1.0, "K": 50000.0, "beta": 0.5},
                "adstock_decay_config": {"type": "linear", "lambda": 0.5, "lag": 10, "weights": [1.0]},
                "noise_variance": {},
                "cpm": 10.0,
            }},
        ],
    }
    config = InputConfigurations.from_yaml_dict(data)
    assert config.get_correlations() == []
    matrix = generate_spend(config)
    assert matrix.shape == (10, 1)
    assert np.all(np.isfinite(matrix))


def main():
    print("Spend generation tests...")
    test_generate_spend_shape()
    test_generate_spend_reproducible()
    test_generate_spend_in_spend_range()
    test_generate_spend_non_negative_finite()
    test_generate_spend_default_gamma_params()
    test_generate_spend_clipping()
    test_generate_spend_single_week_single_channel()
    test_correlated_spend_shape()
    test_correlated_spend_positive_finite()
    test_correlated_spend_in_spend_range()
    test_correlated_spend_reproducible()
    test_correlated_spend_shows_correlation()
    test_correlated_spend_zero_rho_near_independent()
    test_no_correlations_uses_independent_path()
    print("Spend generation tests passed.")


if __name__ == "__main__":
    main()
