"""
Tests for impressions simulation (generate_impressions).
Run from project root: python -m tests.test_impressions_simulation  or  python test.py
"""
import numpy as np

from scripts.config.loader import load_config
from scripts.impressions_simulation.impressions_generation import generate_impressions
from scripts.spend_simulation.spend_generation import generate_spend


def test_generate_impressions_shape_and_non_negative(example_config):
    """generate_impressions returns non-negative matrix with same shape as spend_matrix."""
    config = example_config
    spend = generate_spend(config)
    impressions = generate_impressions(config, spend)

    assert isinstance(impressions, np.ndarray)
    assert impressions.shape == spend.shape
    assert np.all(impressions >= 0.0)


def test_generate_impressions_reproducible_with_seed(example_config_path):
    """With a fixed seed in the config, two runs produce identical impressions."""
    config1 = load_config(str(example_config_path))
    spend1 = generate_spend(config1)
    imps1 = generate_impressions(config1, spend1)

    # Reload config so RNG is reinitialized with the same seed
    config2 = load_config(str(example_config_path))
    spend2 = generate_spend(config2)
    imps2 = generate_impressions(config2, spend2)

    np.testing.assert_array_almost_equal(spend1, spend2)
    np.testing.assert_array_almost_equal(imps1, imps2)


def test_generate_impressions_cpm_effect_direction(example_config_path):
    """
    Channels with lower CPM should tend to have higher impressions
    for comparable spend (in expectation).
    """
    config = load_config(str(example_config_path))
    spend = generate_spend(config)
    impressions = generate_impressions(config, spend)

    channels = config.get_channel_list()
    assert len(channels) >= 2

    cpm_values = [ch.get_cpm() for ch in channels]
    # Sort channels by CPM
    sorted_idx = np.argsort(cpm_values)

    low_cpm_idx = sorted_idx[0]
    high_cpm_idx = sorted_idx[-1]

    # Compare average impressions per week
    avg_low_cpm = impressions[:, low_cpm_idx].mean()
    avg_high_cpm = impressions[:, high_cpm_idx].mean()

    assert avg_low_cpm >= avg_high_cpm, (
        "Channel with lower CPM should not have lower average impressions "
        f"(low CPM avg={avg_low_cpm}, high CPM avg={avg_high_cpm})."
    )


def main():
    print("Impressions simulation tests...")
    test_generate_impressions_shape_and_non_negative()
    test_generate_impressions_reproducible_with_seed()
    test_generate_impressions_cpm_effect_direction()
    print("Impressions simulation tests passed.")


if __name__ == "__main__":
    main()
