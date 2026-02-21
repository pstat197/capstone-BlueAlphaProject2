"""
Test loading example.yaml and the dataclasses (InputConfigurations, Channel).
Uses scripts.config.loader.load_config (merge with default.yaml).
Run from project root: python test.py
"""
from pathlib import Path

import yaml

# Import via scripts package so stdlib 'dataclasses' is not shadowed
from scripts.dataclasses.input_configurations import InputConfigurations
from scripts.dataclasses.channel import Channel
from scripts.config.loader import load_config


def test_load_example_yaml():
    """Load example.yaml via loader (merge with default)."""
    example_path = Path(__file__).resolve().parent / "example.yaml"
    assert example_path.exists(), f"example.yaml not found at {example_path}"
    config = load_config(str(example_path))
    return config


def test_config_getters(config: InputConfigurations):
    """Check InputConfigurations getters match example.yaml."""
    assert config.get_run_identifier() == "Example Alpha"
    assert config.get_week_range() == 26


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
    gamma = tiktok.get_spend_sampling_gamma_params()
    assert gamma["shape"] == 2.5
    assert gamma["scale"] == 1000
    noise = tiktok.get_noise_variance()
    assert noise["impression"] == 0.2
    assert noise["revenue"] == 0.15


def test_channel_linkedin(config: InputConfigurations):
    """Check LinkedIn channel fields."""
    channels = config.get_channel_list()
    linkedin = next(c for c in channels if c.get_channel_name() == "LinkedIn")
    assert linkedin.get_true_roi() == 2.2
    assert linkedin.get_spend_range() == [3000, 60000]
    assert linkedin.get_baseline_revenue() == 9500
    assert linkedin.get_saturation_function() == "sqrt(x)"
    gamma = linkedin.get_spend_sampling_gamma_params()
    assert gamma["shape"] == 2.5
    assert gamma["scale"] == 1000
    noise = linkedin.get_noise_variance()
    assert noise["impression"] == 0.2
    assert noise["revenue"] == 0.15


def test_load_default_yaml():
    """Load scripts/Default.yaml via loader and sanity-check structure."""
    default_path = Path(__file__).resolve().parent / "scripts" / "Default.yaml"
    if not default_path.exists():
        return  # skip if no Default.yaml
    config = load_config(str(default_path))
    assert config.get_run_identifier() == "Default"
    assert config.get_week_range() == 52
    channels = config.get_channel_list()
    assert len(channels) >= 1
    for c in channels:
        assert c.get_channel_name()
        assert c.get_spend_sampling_gamma_params() is not None
        assert c.get_noise_variance() is not None


def test_number_of_channels_generates_from_default():
    """If only number_of_channels is set, loader generates that many default channels."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("run_identifier: GeneratedRun\nnumber_of_channels: 3\n")
        tmp = f.name
    try:
        config = load_config(tmp)
        assert config.get_run_identifier() == "GeneratedRun"
        channels = config.get_channel_list()
        assert len(channels) == 3
        # With no channel_list, merge has default's 1 channel; we add 2 generated to reach 3
        names = [c.get_channel_name() for c in channels]
        assert len(names) == 3
        assert names[0] == "Channel 1"  # from default.yaml
        assert names[1].startswith("Generated Channel ") and names[2].startswith("Generated Channel ")
        assert config.get_week_range() == 52  # from default
        for c in channels:
            assert c.get_spend_sampling_gamma_params()
            assert c.get_noise_variance()
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_missing_fields_filled_from_default():
    """Minimal YAML gets run_identifier from user, week_range from default."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("run_identifier: MinimalRun\nchannel_list:\n  - channel:\n      channel_name: Only\n")
        tmp = f.name
    try:
        config = load_config(tmp)
        assert config.get_run_identifier() == "MinimalRun"
        assert config.get_week_range() == 52  # from default.yaml
        channels = config.get_channel_list()
        assert len(channels) == 1
        assert channels[0].get_channel_name() == "Only"
        # default channel has gamma and noise; merged in
        assert channels[0].get_spend_sampling_gamma_params()
        assert channels[0].get_noise_variance()
    finally:
        Path(tmp).unlink(missing_ok=True)


def main():
    print("Loading example.yaml and creating InputConfigurations...")
    config = test_load_example_yaml()
    print("  run_identifier:", config.get_run_identifier())
    print("  week_range:", config.get_week_range())
    print("  channels:", [c.get_channel_name() for c in config.get_channel_list()])
    for c in config.get_channel_list():
        print("    ", c.get_channel_name(),
              "gamma:", c.get_spend_sampling_gamma_params(),
              "noise_variance:", c.get_noise_variance())

    print("\nRunning assertions...")
    test_config_getters(config)
    test_channel_list(config)
    test_channel_tiktok(config)
    test_channel_linkedin(config)
    test_load_default_yaml()
    test_number_of_channels_generates_from_default()
    test_missing_fields_filled_from_default()

    print("All tests passed.")


if __name__ == "__main__":
    main()
