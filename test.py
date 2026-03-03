"""
Run all test modules: config, spend generation, impressions, revenue, pipeline.
Run from project root: python test.py
"""
import sys
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import yaml

# Import via scripts package so stdlib 'dataclasses' is not shadowed
from scripts.synth_input_classes.input_configurations import InputConfigurations
from scripts.synth_input_classes.channel import Channel


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
    """Load scripts/Default.yaml and sanity-check structure."""
    default_path = Path(__file__).resolve().parent / "scripts" / "Default.yaml"
    if not default_path.exists():
        return  # skip if no Default.yaml
    with open(default_path, "r") as f:
        data = yaml.safe_load(f)
    config = InputConfigurations.from_yaml_dict(data)
    assert config.get_run_identifier() == "Default"
    assert config.get_week_range() == 52
    channels = config.get_channel_list()
    assert len(channels) >= 1
    for c in channels:
        assert c.get_channel_name()
        assert c.get_spend_sampling_gamma_params() is not None
        assert c.get_noise_variance() is not None


def main():
    from tests import test_config
    from tests import test_spend_generation
    from tests import test_impressions_simulation
    from tests import test_revenue_simulation
    from tests import test_pipeline

    test_config.main()
    test_spend_generation.main()
    test_impressions_simulation.main()
    test_revenue_simulation.main()
    test_pipeline.main()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
