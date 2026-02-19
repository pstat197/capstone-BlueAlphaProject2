"""
Test loading example.yaml and the dataclasses (InputConfigurations, Channel).
Run from project root: python test.py
"""
from pathlib import Path

import yaml

# Import via scripts package so stdlib 'dataclasses' is not shadowed
from scripts.dataclasses.input_configurations import InputConfigurations
from scripts.dataclasses.channel import Channel


def test_load_example_yaml():
    """Load example.yaml and build InputConfigurations."""
    example_path = Path(__file__).resolve().parent / "example.yaml"
    assert example_path.exists(), f"example.yaml not found at {example_path}"

    with open(example_path, "r") as f:
        data = yaml.safe_load(f)

    config = InputConfigurations.from_yaml_dict(data)
    return config


def test_config_getters(config: InputConfigurations):
    """Check InputConfigurations getters match example.yaml."""
    assert config.get_run_identifier() == "Example Alpha"
    assert config.get_week_range() == 26
    noise = config.get_noise_level()
    assert noise["impression"] == 0.2
    assert noise["revenue"] == 0.15


def test_channel_list(config: InputConfigurations):
    """Check we have the right number of channels."""
    channels = config.get_channel_list()
    assert len(channels) == 2


def test_channel_tiktok(config: InputConfigurations):
    """Check TikTok channel fields."""
    channels = config.get_channel_list()
    tiktok = next(c for c in channels if c.get_channel_name() == "TikTok")
    assert tiktok.get_true_roi() == 3.1
    assert tiktok.get_spend_range() == [1500, 40000]
    assert tiktok.get_baseline_revenue() == 6000
    assert tiktok.get_saturation_function() == "log(x+1)"


def test_channel_linkedin(config: InputConfigurations):
    """Check LinkedIn channel fields."""
    channels = config.get_channel_list()
    linkedin = next(c for c in channels if c.get_channel_name() == "LinkedIn")
    assert linkedin.get_true_roi() == 2.2
    assert linkedin.get_spend_range() == [3000, 60000]
    assert linkedin.get_baseline_revenue() == 9500
    assert linkedin.get_saturation_function() == "sqrt(x)"


def main():
    print("Loading example.yaml and creating InputConfigurations...")
    config = test_load_example_yaml()
    print("  run_identifier:", config.get_run_identifier())
    print("  week_range:", config.get_week_range())
    print("  noise_level:", config.get_noise_level())
    print("  channels:", [c.get_channel_name() for c in config.get_channel_list()])

    print("\nRunning assertions...")
    test_config_getters(config)
    test_channel_list(config)
    test_channel_tiktok(config)
    test_channel_linkedin(config)

    print("All tests passed.")


if __name__ == "__main__":
    main()
