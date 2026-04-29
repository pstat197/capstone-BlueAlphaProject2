"""
Tests for configuration loading, input synth_input_classes, and config step.
Uses scripts.config.loader.load_config (merge with default.yaml).
Run from project root: python -m tests.test_config  or  python test.py
"""
from pathlib import Path

import pytest

from scripts.synth_input_classes.input_configurations import InputConfigurations
from scripts.synth_input_classes.channel import Channel
from scripts.config.loader import apply_seed_append_expansion, load_config, load_config_from_dict


def _project_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def config() -> InputConfigurations:
    """Load example.yaml for tests that need a full config."""
    example_path = _project_root() / "example.yaml"
    assert example_path.exists(), f"example.yaml not found at {example_path}"
    return load_config(str(example_path))


def test_load_example_yaml():
    """Load example.yaml via loader (merge with default)."""
    example_path = _project_root() / "example.yaml"
    assert example_path.exists(), f"example.yaml not found at {example_path}"
    load_config(str(example_path))


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
    assert tiktok.get_trend_slope() == 0
    seasonality = tiktok.get_seasonality_config()
    assert seasonality == {}
    sat = tiktok.get_saturation_config()
    assert sat["type"] == "linear" and sat["slope"] == 1.0 and sat["K"] == 50000.0 and sat["beta"] == 0.5
    adstock = tiktok.get_adstock_decay_config()
    assert adstock["type"] == "linear" and adstock["lambda"] == 0.5 and adstock["lag"] == 10 and adstock["weights"] == [1.0]
    gamma = tiktok.get_spend_sampling_gamma_params()
    assert gamma["shape"] == 2.5
    assert gamma["scale"] == 1000
    noise = tiktok.get_noise_variance()
    assert noise["impression"] == 0.1
    assert noise["revenue"] == 0.15
    assert tiktok.get_cpm() == 25


def test_channel_linkedin(config: InputConfigurations):
    """Check LinkedIn channel fields."""
    channels = config.get_channel_list()
    linkedin = next(c for c in channels if c.get_channel_name() == "LinkedIn")
    assert linkedin.get_true_roi() == 2.2
    assert linkedin.get_spend_range() == [3000, 60000]
    assert linkedin.get_baseline_revenue() == 9500
    assert linkedin.get_trend_slope() == 0
    seasonality = linkedin.get_seasonality_config()
    assert seasonality == {}
    sat = linkedin.get_saturation_config()
    assert sat["type"] == "linear" and sat["slope"] == 0.8
    adstock = linkedin.get_adstock_decay_config()
    assert adstock["lag"] == 7 and adstock["weights"] == [0.7, 0.2, 0.1]
    gamma = linkedin.get_spend_sampling_gamma_params()
    assert gamma["shape"] == 2.5
    assert gamma["scale"] == 1000
    noise = linkedin.get_noise_variance()
    assert noise["impression"] == 0.0025
    assert noise["revenue"] == 0.15
    assert linkedin.get_cpm() == 10


def test_load_default_yaml():
    """Load scripts/config/default.yaml via loader and sanity-check structure."""
    default_path = _project_root() / "scripts" / "config" / "default.yaml"
    if not default_path.exists():
        return  # skip if missing
    config = load_config(str(default_path))
    assert config.get_run_identifier() == "Default"
    assert config.get_week_range() == 52
    channels = config.get_channel_list()
    assert len(channels) >= 1
    for c in channels:
        assert c.get_channel_name()
        assert c.get_spend_sampling_gamma_params() is not None
        assert c.get_noise_variance() is not None
        assert c.get_seasonality_config() is not None
        assert c.get_saturation_config() is not None
        assert c.get_adstock_decay_config() is not None
        assert 0.5 <= c.get_cpm() <= 50.0, "cpm should be in default sampling range when loaded from default"


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
        names = [c.get_channel_name() for c in channels]
        assert len(names) == 3
        assert names[0] == "Channel 1"  # from default.yaml
        assert names[1].startswith("Generated Channel ") and names[2].startswith("Generated Channel ")
        assert config.get_week_range() == 52  # from default
        for c in channels:
            assert c.get_spend_sampling_gamma_params()
            assert c.get_noise_variance()
            assert c.get_seasonality_config() is not None
            assert c.get_saturation_config() is not None
            assert c.get_adstock_decay_config() is not None
            assert 0.5 <= c.get_cpm() <= 50.0, "generated channels should have cpm in default sampling range"
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
        assert channels[0].get_spend_sampling_gamma_params()
        assert channels[0].get_noise_variance()
        assert channels[0].get_seasonality_config() is not None
        assert 0.5 <= channels[0].get_cpm() <= 50.0, "minimal channel without cpm should get sampled cpm in range"
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_cpm_omitted_sampled_per_channel_in_range():
    """When cpm is omitted, each channel gets a value in cpm_sampling_range; multiple channels get different values."""
    import tempfile
    yaml_content = (
        "run_identifier: CpmSampled\nseed: 42\nweek_range: 4\n"
        "channel_list:\n"
        "  - channel:\n      channel_name: A\n"
        "  - channel:\n      channel_name: B\n"
        "  - channel:\n      channel_name: C\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp = f.name
    try:
        config = load_config(tmp)
        channels = config.get_channel_list()
        cpms = [c.get_cpm() for c in channels]
        for cpm in cpms:
            assert 0.5 <= cpm <= 50.0, f"cpm {cpm} should be in default range [0.5, 50]"
        assert len(cpms) == len(set(cpms)), "each channel should get a different sampled cpm"
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write('run_identifier: ""\nweek_range: 12\nchannel_list:\n  - channel:\n      channel_name: X\n')
        tmp = f.name
    try:
        config = load_config(tmp)
        rid = config.get_run_identifier()
        assert rid.startswith("run_")
        rest = rid[4:]
        assert "_" in rest
        parts = rest.split("_")
        assert len(parts) == 2
        assert len(parts[0]) == 8 and parts[0].isdigit()  # YYYYMMDD
        assert len(parts[1]) == 4 and parts[1].isdigit()  # HHMM
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_rng_reproducible_with_same_seed():
    """Loading two configs with the same seed and drawing from get_rng() gives same sequence."""
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


def _minimal_channel(name: str) -> dict:
    return {
        "channel": {
            "channel_name": name,
            "true_roi": 1.0,
            "spend_range": [1, 1000],
            "baseline_revenue": 0.0,
            "trend_slope": 0.0,
            "seasonality_config": {},
            "saturation_config": {"type": "linear", "slope": 1.0},
            "adstock_decay_config": {"type": "linear", "lag": 0},
            "spend_sampling_gamma_params": {"shape": 1.0, "scale": 100.0},
            "noise_variance": {"impression": 0.0, "revenue": 0.0},
            "cpm": 10.0,
        }
    }


def test_budget_shifts_auto_mode_global_expands_at_load():
    """Empty manual budget_shifts + global auto mode yields non-empty effective rules (loader expansion)."""
    cfg = {
        "run_identifier": "AutoBS",
        "week_range": 12,
        "seed": 7,
        "budget_shifts": [],
        "budget_shifts_auto_mode": "global",
        "channel_list": [_minimal_channel("A"), _minimal_channel("B")],
    }
    ic = load_config_from_dict(cfg)
    bs = ic.get_budget_shifts()
    assert len(bs) >= 2
    for r in bs:
        assert r.get("type") in ("scale", "scale_channel", "reallocate")


def test_budget_shifts_auto_mode_reproducible():
    """Same YAML dict twice -> identical expanded budget_shifts inside InputConfigurations."""
    cfg = {
        "run_identifier": "AutoBS2",
        "week_range": 12,
        "seed": 7,
        "budget_shifts": [],
        "budget_shifts_auto_mode": "global",
        "channel_list": [_minimal_channel("A"), _minimal_channel("B")],
    }
    ic1 = load_config_from_dict(cfg)
    ic2 = load_config_from_dict(cfg)
    assert ic1.get_budget_shifts() == ic2.get_budget_shifts()


def test_correlations_auto_mode_random_expands_at_load():
    """correlations_auto_mode random with empty manual list yields 1..3 pairs for 3+ channels."""
    cfg = {
        "run_identifier": "AutoCorr",
        "week_range": 8,
        "seed": 99,
        "correlations": [],
        "correlations_auto_mode": "random",
        "channel_list": [
            _minimal_channel("Z"),
            _minimal_channel("A"),
            _minimal_channel("M"),
        ],
    }
    ic = load_config_from_dict(cfg)
    corr = ic.get_correlations()
    assert 1 <= len(corr) <= 3
    for row in corr:
        assert len(row.get("channels") or []) == 2
        rho = float(row.get("rho", 0.0))
        assert -1.0 <= rho <= 1.0


def test_correlations_manual_pair_not_overwritten_by_random():
    """Manual rho for a pair wins; random mode does not add a duplicate unordered pair."""
    cfg = {
        "run_identifier": "ManualCorrWins",
        "week_range": 8,
        "seed": 99,
        "correlations": [{"channels": ["A", "B"], "rho": 0.35}],
        "correlations_auto_mode": "random",
        "channel_list": [_minimal_channel("A"), _minimal_channel("B")],
    }
    ic = load_config_from_dict(cfg)
    corr = ic.get_correlations()
    assert len(corr) == 1
    assert corr[0]["channels"] == ["A", "B"]
    assert abs(float(corr[0]["rho"]) - 0.35) < 1e-9


def test_apply_seed_append_expansion_pops_mode_keys():
    """Expansion consumes mode keys so merged dict is safe for from_yaml_dict-style consumers."""
    merged = {
        "seed": 1,
        "week_range": 10,
        "budget_shifts": [],
        "budget_shifts_auto_mode": "global",
        "correlations_auto_mode": "none",
        "correlations": [],
        "channel_list": [
            {"channel": {"channel_name": "A"}},
            {"channel": {"channel_name": "B"}},
        ],
    }
    apply_seed_append_expansion(merged)
    assert "budget_shifts_auto_mode" not in merged
    assert "correlations_auto_mode" not in merged
    assert merged.get("budget_shifts")
    assert merged.get("correlations") == []


def test_load_config_from_dict_preserves_callers_budget_shifts_list():
    """Loader must not mutate the caller's ``budget_shifts`` list in place (Streamlit snapshots)."""
    rules = [
        {"type": "scale", "start_week": 1, "end_week": 2, "factor": 1.05},
        {"type": "scale_channel", "channel_name": "A", "start_week": 1, "end_week": 2, "factor": 0.95},
    ]
    cfg_dict = {
        "run_identifier": "PreserveBudgetShifts",
        "week_range": 5,
        "seed": 1,
        "budget_shifts": rules,
        "channel_list": [
            {
                "channel": {
                    "channel_name": "A",
                    "true_roi": 1.0,
                    "spend_range": [1, 1000],
                    "baseline_revenue": 0.0,
                    "trend_slope": 0.0,
                    "seasonality_config": {},
                    "saturation_config": {"type": "linear", "slope": 1.0},
                    "adstock_decay_config": {"type": "linear", "lag": 0},
                    "spend_sampling_gamma_params": {"shape": 1.0, "scale": 100.0},
                    "noise_variance": {"impression": 0.0, "revenue": 0.0},
                    "cpm": 10.0,
                }
            }
        ],
    }
    rules_id = id(cfg_dict["budget_shifts"])
    n_before = len(rules)
    ic = load_config_from_dict(cfg_dict)
    assert id(cfg_dict["budget_shifts"]) == rules_id
    assert len(cfg_dict["budget_shifts"]) == n_before
    assert len(ic.get_budget_shifts()) == n_before


def main():
    print("Config/loading tests...")
    root = _project_root()
    example_path = root / "example.yaml"
    assert example_path.exists(), f"example.yaml not found at {example_path}"
    config = load_config(str(example_path))
    print("  run_identifier:", config.get_run_identifier())
    print("  week_range:", config.get_week_range())
    print("  channels:", [c.get_channel_name() for c in config.get_channel_list()])
    test_config_getters(config)
    test_channel_list(config)
    test_channel_tiktok(config)
    test_channel_linkedin(config)
    test_seed_stored_when_in_yaml(config)
    test_get_rng_returns_generator(config)
    test_load_default_yaml()
    test_number_of_channels_generates_from_default()
    test_missing_fields_filled_from_default()
    test_cpm_omitted_sampled_per_channel_in_range()
    test_seed_absent_returns_none()
    test_missing_run_identifier_gets_timestamp()
    test_rng_reproducible_with_same_seed()
    test_load_config_from_dict_preserves_callers_budget_shifts_list()
    print("Config tests passed.")


if __name__ == "__main__":
    main()
