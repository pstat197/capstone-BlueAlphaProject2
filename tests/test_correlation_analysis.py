"""
Tests for spend correlation analysis (static, rolling, drift, per-channel avg).
Run from project root: python -m tests.test_correlation_analysis  or  python test.py
"""
import numpy as np

from scripts.config.noise import init_rng
from scripts.synth_input_classes.input_configurations import InputConfigurations
from scripts.spend_simulation.spend_generation import generate_spend
from scripts.spend_simulation.correlation_analysis import (
    compute_static_correlation,
    compute_rolling_correlation,
    compute_pairwise_drift,
    compute_avg_abs_correlation,
    compute_most_correlated_channel,
    analyze_spend_correlations,
    print_correlation_report,
)


def _make_config(seed=42, rho=0.7, num_weeks=200):
    """Build a 3-channel config with two correlation entries."""
    init_rng(seed)
    base_ch = {
        "spend_sampling_gamma_params": {"shape": 2.5, "scale": 1000},
        "spend_range": [100, 100000],
        "true_roi": 2.0,
        "baseline_revenue": 5000,
        "saturation_config": {"type": "linear", "slope": 1.0, "K": 50000.0, "beta": 0.5},
        "adstock_decay_config": {"type": "linear", "lambda": 0.5, "lag": 10, "weights": [1.0]},
        "noise_variance": {"impression": 0.02, "revenue": 0.1},
        "cpm": 10.0,
    }
    data = {
        "run_identifier": "CorrAnalysisTest",
        "week_range": num_weeks,
        "seed": seed,
        "channel_list": [
            {"channel": {**base_ch, "channel_name": "Search"}},
            {"channel": {**base_ch, "channel_name": "Social"}},
            {"channel": {**base_ch, "channel_name": "Display"}},
        ],
        "correlations": [
            {"channels": ["Search", "Social"], "rho": rho},
            {"channels": ["Search", "Display"], "rho": 0.3},
        ],
    }
    return InputConfigurations.from_yaml_dict(data)


def test_static_correlation_shape():
    """Static correlation is (C, C) with 1s on diagonal."""
    spend = np.random.default_rng(0).standard_normal((100, 3))
    corr = compute_static_correlation(spend)
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr), 1.0)


def test_static_correlation_symmetric():
    spend = np.random.default_rng(1).standard_normal((50, 4))
    corr = compute_static_correlation(spend)
    np.testing.assert_allclose(corr, corr.T)


def test_rolling_correlation_shape():
    """Rolling output has correct time dimension."""
    spend = np.random.default_rng(2).standard_normal((52, 3))
    window = 12
    rolling = compute_rolling_correlation(spend, window)
    assert rolling.shape == (52 - window, 3, 3)


def test_rolling_correlation_values_valid():
    """All rolling correlations are in [-1, 1] with 1s on diagonal."""
    spend = np.random.default_rng(3).standard_normal((100, 2))
    rolling = compute_rolling_correlation(spend, 20)
    assert np.all(rolling >= -1.0 - 1e-10)
    assert np.all(rolling <= 1.0 + 1e-10)
    for t in range(rolling.shape[0]):
        np.testing.assert_allclose(np.diag(rolling[t]), 1.0)


def test_rolling_window_exceeds_T():
    """Window larger than T raises ValueError."""
    spend = np.random.default_rng(4).standard_normal((10, 2))
    try:
        compute_rolling_correlation(spend, 20)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_drift_shape():
    spend = np.random.default_rng(5).standard_normal((100, 3))
    rolling = compute_rolling_correlation(spend, 12)
    drift = compute_pairwise_drift(rolling)
    assert drift.shape == (3, 3)


def test_drift_stable_for_iid():
    """IID spend should have drift near zero with enough data."""
    spend = np.random.default_rng(6).standard_normal((2000, 2))
    rolling = compute_rolling_correlation(spend, 50)
    drift = compute_pairwise_drift(rolling, num_edge_windows=20)
    assert abs(drift[0, 1]) < 0.15, f"Expected near-zero drift for iid, got {drift[0,1]:.3f}"


def test_avg_abs_correlation():
    corr = np.array([
        [1.0, 0.8, 0.2],
        [0.8, 1.0, 0.5],
        [0.2, 0.5, 1.0],
    ])
    names = ["A", "B", "C"]
    avg = compute_avg_abs_correlation(corr, names)
    assert set(avg.keys()) == {"A", "B", "C"}
    np.testing.assert_almost_equal(avg["A"], (0.8 + 0.2) / 2)
    np.testing.assert_almost_equal(avg["B"], (0.8 + 0.5) / 2)
    np.testing.assert_almost_equal(avg["C"], (0.2 + 0.5) / 2)


def test_most_correlated_channel():
    corr = np.array([
        [1.0, 0.8, 0.2],
        [0.8, 1.0, 0.5],
        [0.2, 0.5, 1.0],
    ])
    names = ["A", "B", "C"]
    assert compute_most_correlated_channel(corr, names) == "B"


def test_analyze_returns_all_keys():
    """analyze_spend_correlations returns the full result dict."""
    config = _make_config(num_weeks=100)
    spend = generate_spend(config)
    results = analyze_spend_correlations(config, spend, window=12)
    expected_keys = {
        "channel_names", "static_corr", "rolling_corr", "drift",
        "avg_abs_corr", "most_correlated_channel", "pairwise_summary", "window",
    }
    assert set(results.keys()) == expected_keys


def test_analyze_pairwise_summary_covers_all_pairs():
    """Pairwise summary lists every unordered pair; YAML fills configured_rho when set."""
    config = _make_config(num_weeks=100)
    spend = generate_spend(config)
    results = analyze_spend_correlations(config, spend)
    assert len(results["pairwise_summary"]) == 3
    pairs = [tuple(p["pair"]) for p in results["pairwise_summary"]]
    assert ("Search", "Social") in pairs
    assert ("Search", "Display") in pairs
    assert ("Display", "Social") in pairs or ("Social", "Display") in pairs
    by_pair = {tuple(p["pair"]): p for p in results["pairwise_summary"]}
    assert by_pair[("Search", "Social")]["configured_rho"] is not None
    assert by_pair[("Search", "Display")]["configured_rho"] is not None
    social_display = by_pair.get(("Social", "Display")) or by_pair.get(("Display", "Social"))
    assert social_display is not None
    assert social_display["configured_rho"] is None


def test_analyze_correlated_rho_close_to_configured():
    """With enough data, observed rho should be in the right ballpark."""
    config = _make_config(seed=10, rho=0.7, num_weeks=500)
    spend = generate_spend(config)
    results = analyze_spend_correlations(config, spend)
    for p in results["pairwise_summary"]:
        if p["pair"] == ["Search", "Social"]:
            assert p["observed_rho"] > 0.3, (
                f"Expected observed rho near 0.7, got {p['observed_rho']:.3f}"
            )


def test_analyze_no_correlations():
    """With no correlations block, analysis still works (all pairs independent)."""
    init_rng(88)
    data = {
        "run_identifier": "NoCorrAnalysis",
        "week_range": 50,
        "seed": 88,
        "channel_list": [
            {"channel": {
                "channel_name": "A",
                "spend_sampling_gamma_params": {"shape": 2.0, "scale": 500},
                "spend_range": [100, 20000],
                "true_roi": 1.0, "baseline_revenue": 0,
                "saturation_config": {"type": "linear", "slope": 1.0, "K": 50000.0, "beta": 0.5},
                "adstock_decay_config": {"type": "linear", "lambda": 0.5, "lag": 10, "weights": [1.0]},
                "noise_variance": {}, "cpm": 10.0,
            }},
            {"channel": {
                "channel_name": "B",
                "spend_sampling_gamma_params": {"shape": 2.0, "scale": 500},
                "spend_range": [100, 20000],
                "true_roi": 1.0, "baseline_revenue": 0,
                "saturation_config": {"type": "linear", "slope": 1.0, "K": 50000.0, "beta": 0.5},
                "adstock_decay_config": {"type": "linear", "lambda": 0.5, "lag": 10, "weights": [1.0]},
                "noise_variance": {}, "cpm": 10.0,
            }},
        ],
    }
    config = InputConfigurations.from_yaml_dict(data)
    spend = generate_spend(config)
    results = analyze_spend_correlations(config, spend)
    assert len(results["pairwise_summary"]) == 1
    assert results["pairwise_summary"][0]["pair"] == ["A", "B"]
    assert results["pairwise_summary"][0]["configured_rho"] is None
    assert results["static_corr"].shape == (2, 2)


def test_print_report_runs_without_error():
    """print_correlation_report doesn't crash."""
    config = _make_config(num_weeks=50)
    spend = generate_spend(config)
    results = analyze_spend_correlations(config, spend, window=10)
    print_correlation_report(results)


def main():
    print("Correlation analysis tests...")
    test_static_correlation_shape()
    test_static_correlation_symmetric()
    test_rolling_correlation_shape()
    test_rolling_correlation_values_valid()
    test_rolling_window_exceeds_T()
    test_drift_shape()
    test_drift_stable_for_iid()
    test_avg_abs_correlation()
    test_most_correlated_channel()
    test_analyze_returns_all_keys()
    test_analyze_pairwise_summary_covers_all_pairs()
    test_analyze_correlated_rho_close_to_configured()
    test_analyze_no_correlations()
    test_print_report_runs_without_error()
    print("Correlation analysis tests passed.")


if __name__ == "__main__":
    main()
