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


def test_seed_stored_when_in_yaml(config: InputConfigurations):
    """example.yaml has seed; config stores and returns it."""
    assert config.get_seed() is not None
    assert config.get_seed() == 133233  # from example.yaml


def test_get_rng_returns_generator(config: InputConfigurations):
    """get_rng() returns a numpy Generator (has .normal, .random)."""
    rng = config.get_rng()
    assert rng is not None
    assert hasattr(rng, "normal")
    assert hasattr(rng, "random")
    # Can draw a number
    x = rng.normal(0, 1)
    assert isinstance(x, float)


def test_seed_absent_returns_none():
    """When YAML has no seed, get_seed() returns None."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("run_identifier: NoSeedRun\nchannel_list:\n  - channel:\n      channel_name: A\n")
        tmp = f.name
    try:
        config = load_config(tmp)
        assert config.get_seed() is None
        assert config.get_rng() is not None  # RNG still available (randomly seeded)
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_missing_run_identifier_gets_timestamp():
    """When run_identifier is empty/missing, loader sets run_YYYYMMDD_HHMM."""
    import tempfile
    # Empty run_identifier so loader fills with timestamp (default would otherwise supply "Default")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write('run_identifier: ""\nweek_range: 12\nchannel_list:\n  - channel:\n      channel_name: X\n')
        tmp = f.name
    try:
        config = load_config(tmp)
        rid = config.get_run_identifier()
        assert rid.startswith("run_")
        # Format run_YYYYMMDD_HHMM
        rest = rid[4:]
        assert "_" in rest
        parts = rest.split("_")
        assert len(parts) == 2
        assert len(parts[0]) == 8 and parts[0].isdigit()  # YYYYMMDD
        assert len(parts[1]) == 4 and parts[1].isdigit()  # HHMM
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_rng_reproducible_with_same_seed():
    """Loading two configs with the same seed and drawing from get_rng() gives same sequence (after second load)."""
    import tempfile
    yaml_content = "run_identifier: Reproducible\nseed: 12345\nchannel_list:\n  - channel:\n      channel_name: C\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp = f.name
    try:
        config1 = load_config(tmp)
        rng1 = config1.get_rng()
        draws1 = [rng1.normal(0, 1) for _ in range(3)]
        config2 = load_config(tmp)
        rng2 = config2.get_rng()
        draws2 = [rng2.normal(0, 1) for _ in range(3)]
        assert draws1 == draws2, "Same seed should give same RNG sequence"
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
    test_seed_stored_when_in_yaml(config)
    test_get_rng_returns_generator(config)
    test_load_default_yaml()
    test_number_of_channels_generates_from_default()
    test_missing_fields_filled_from_default()
    test_seed_absent_returns_none()
    test_missing_run_identifier_gets_timestamp()
    test_rng_reproducible_with_same_seed()

    print("All tests passed.")


if __name__ == "__main__":
    main()
