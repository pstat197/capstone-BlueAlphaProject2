"""
Tests for channel on/off toggles (Policy A - soft off).

Covers:
  - Channel dataclass defaults (fail-open).
  - Channel.is_on() / Channel.on_vector() semantics.
  - Loader / InputConfigurations parsing of all toggle YAML shapes.
  - Spend generation masking.
  - Impressions safety-net masking.
  - Revenue: fully-disabled short-circuit, Policy A adstock echo on off weeks,
    per-channel and global adstock/saturation gates.
  - Backward compatibility (no toggle fields = identical output).

Run from project root:
    python -m tests.test_channel_toggles
"""
from pathlib import Path

import numpy as np
import pytest

from scripts.config.loader import load_config_from_dict
from scripts.impressions_simulation.impressions_generation import generate_impressions
from scripts.main import construct_csv
from scripts.revenue_simulation.revenue_generation import generate_revenue
from scripts.spend_simulation.spend_generation import generate_spend
from scripts.synth_input_classes.channel import Channel, StickyPauseRange
from scripts.synth_input_classes.input_configurations import InputConfigurations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel_dict(
    name: str = "A",
    *,
    enabled=None,
    adstock_enabled=None,
    saturation_enabled=None,
    cpm: float = 10.0,
    true_roi: float = 2.0,
    baseline_revenue: float = 0.0,
    spend_range=(1000, 10000),
    gamma=(2.5, 1000),
    impression_noise: float = 0.0,
    revenue_noise: float = 0.0,
    saturation=None,
    adstock=None,
    sticky_pause_ranges=None,
) -> dict:
    """Build a raw channel dict for load_config_from_dict, with optional toggles."""
    ch = {
        "channel_name": name,
        "cpm": cpm,
        "true_roi": true_roi,
        "spend_range": list(spend_range),
        "baseline_revenue": baseline_revenue,
        "spend_sampling_gamma_params": {"shape": gamma[0], "scale": gamma[1]},
        "noise_variance": {"impression": impression_noise, "revenue": revenue_noise},
        "saturation_config": saturation or {"type": "linear", "slope": 1.0},
        "adstock_decay_config": adstock or {"type": "linear", "lag": 0},
    }
    if enabled is not None:
        ch["enabled"] = enabled
    if adstock_enabled is not None:
        ch["adstock_enabled"] = adstock_enabled
    if saturation_enabled is not None:
        ch["saturation_enabled"] = saturation_enabled
    if sticky_pause_ranges is not None:
        ch["sticky_pause_ranges"] = sticky_pause_ranges
    return {"channel": ch}


def _make_config(
    channels,
    *,
    week_range: int = 12,
    seed: int = 42,
    adstock_global=None,
    saturation_global=None,
    run_identifier: str = "ToggleTest",
) -> InputConfigurations:
    cfg_dict = {
        "run_identifier": run_identifier,
        "week_range": week_range,
        "seed": seed,
        "channel_list": channels,
    }
    if adstock_global is not None:
        cfg_dict["adstock"] = {"global": adstock_global}
    if saturation_global is not None:
        cfg_dict["saturation"] = {"global": saturation_global}
    return load_config_from_dict(cfg_dict)


# ---------------------------------------------------------------------------
# Channel dataclass
# ---------------------------------------------------------------------------


def test_channel_toggle_defaults_fail_open():
    """Channel built with no toggle fields is fully on, no effect gates."""
    ch = Channel(
        channel_name="X",
        true_roi=1.0,
        spend_range=[0, 1],
        baseline_revenue=0.0,
        saturation_config={"type": "linear"},
        adstock_decay_config={"type": "linear", "lag": 0},
        spend_sampling_gamma_params={"shape": 1.0, "scale": 1.0},
        noise_variance={"impression": 0.0, "revenue": 0.0},
        cpm=10.0,
    )
    assert ch.enabled is True
    assert ch.off_ranges == ()
    assert ch.adstock_enabled is True
    assert ch.saturation_enabled is True
    assert ch.sticky_pause_ranges == ()
    assert ch.is_fully_disabled() is False
    assert ch.is_on(1) is True
    assert ch.is_on(1000) is True
    np.testing.assert_array_equal(ch.on_vector(5), np.array([True] * 5))


def test_channel_is_on_handles_off_ranges_inclusively():
    ch = Channel(
        channel_name="X", true_roi=1.0, spend_range=[0, 1], baseline_revenue=0.0,
        saturation_config={}, adstock_decay_config={"lag": 0},
        spend_sampling_gamma_params={"shape": 1.0, "scale": 1.0},
        noise_variance={"impression": 0.0, "revenue": 0.0}, cpm=10.0,
        off_ranges=((3, 5), (8, 8)),
    )
    assert ch.is_on(2) is True
    assert ch.is_on(3) is False  # inclusive start
    assert ch.is_on(4) is False
    assert ch.is_on(5) is False  # inclusive end
    assert ch.is_on(6) is True
    assert ch.is_on(7) is True
    assert ch.is_on(8) is False  # single-week range
    assert ch.is_on(9) is True


def test_channel_fully_disabled_is_always_off():
    ch = Channel(
        channel_name="X", true_roi=1.0, spend_range=[0, 1], baseline_revenue=0.0,
        saturation_config={}, adstock_decay_config={"lag": 0},
        spend_sampling_gamma_params={"shape": 1.0, "scale": 1.0},
        noise_variance={"impression": 0.0, "revenue": 0.0}, cpm=10.0,
        enabled=False,
    )
    assert ch.is_fully_disabled() is True
    assert ch.is_on(1) is False
    np.testing.assert_array_equal(ch.on_vector(4), np.array([False] * 4))


def test_channel_on_vector_matches_is_on():
    ch = Channel(
        channel_name="X", true_roi=1.0, spend_range=[0, 1], baseline_revenue=0.0,
        saturation_config={}, adstock_decay_config={"lag": 0},
        spend_sampling_gamma_params={"shape": 1.0, "scale": 1.0},
        noise_variance={"impression": 0.0, "revenue": 0.0}, cpm=10.0,
        off_ranges=((2, 3), (7, 9)),
    )
    mask = ch.on_vector(10)
    assert mask.dtype == bool
    for i, active in enumerate(mask):
        assert bool(active) == ch.is_on(i + 1), f"mismatch at week {i + 1}"


def test_channel_on_vector_rejects_negative_num_weeks():
    ch = Channel(
        channel_name="X", true_roi=1.0, spend_range=[0, 1], baseline_revenue=0.0,
        saturation_config={}, adstock_decay_config={"lag": 0},
        spend_sampling_gamma_params={"shape": 1.0, "scale": 1.0},
        noise_variance={"impression": 0.0, "revenue": 0.0}, cpm=10.0,
    )
    with pytest.raises(ValueError):
        ch.on_vector(-1)


def test_spend_allowed_mask_reproducible():
    ch = Channel(
        channel_name="X",
        true_roi=1.0,
        spend_range=[0, 1],
        baseline_revenue=0.0,
        saturation_config={},
        adstock_decay_config={"lag": 0},
        spend_sampling_gamma_params={"shape": 1.0, "scale": 1.0},
        noise_variance={"impression": 0.0, "revenue": 0.0},
        cpm=10.0,
        sticky_pause_ranges=(
            StickyPauseRange(1, 10, 0.25, 0.75),
        ),
    )
    m1 = ch.spend_allowed_mask(10, channel_index=0, config_seed=123)
    m2 = ch.spend_allowed_mask(10, channel_index=0, config_seed=123)
    np.testing.assert_array_equal(m1, m2)


def test_spend_allowed_mask_all_sticky_when_always_continue():
    ch = Channel(
        channel_name="X",
        true_roi=1.0,
        spend_range=[0, 1],
        baseline_revenue=0.0,
        saturation_config={},
        adstock_decay_config={"lag": 0},
        spend_sampling_gamma_params={"shape": 1.0, "scale": 1.0},
        noise_variance={"impression": 0.0, "revenue": 0.0},
        cpm=10.0,
        sticky_pause_ranges=(
            StickyPauseRange(1, 4, 1.0, 1.0),
        ),
    )
    m = ch.spend_allowed_mask(8, channel_index=0, config_seed=7)
    np.testing.assert_array_equal(m[:4], np.array([False] * 4))
    np.testing.assert_array_equal(m[4:], np.array([True] * 4))


def test_sticky_markov_alternating_when_continue_is_zero():
    ch = Channel(
        channel_name="X",
        true_roi=1.0,
        spend_range=[0, 1],
        baseline_revenue=0.0,
        saturation_config={},
        adstock_decay_config={"lag": 0},
        spend_sampling_gamma_params={"shape": 1.0, "scale": 1.0},
        noise_variance={"impression": 0.0, "revenue": 0.0},
        cpm=10.0,
        sticky_pause_ranges=(
            StickyPauseRange(1, 5, 1.0, 0.0),
        ),
    )
    m = ch.spend_allowed_mask(5, channel_index=0, config_seed=99)
    expected = np.array([False, True, False, True, False])
    np.testing.assert_array_equal(m, expected)


def test_sticky_freezes_markov_state_across_deterministic_off_week():
    ch = Channel(
        channel_name="X",
        true_roi=1.0,
        spend_range=[0, 1],
        baseline_revenue=0.0,
        saturation_config={},
        adstock_decay_config={"lag": 0},
        spend_sampling_gamma_params={"shape": 1.0, "scale": 1.0},
        noise_variance={"impression": 0.0, "revenue": 0.0},
        cpm=10.0,
        off_ranges=((3, 3),),
        sticky_pause_ranges=(
            StickyPauseRange(2, 4, 1.0, 1.0),
        ),
    )
    m = ch.spend_allowed_mask(5, channel_index=0, config_seed=1)
    # Week 2 sticky off; week 3 hard off (skip sticky draw, prev stays True); week 4 uses continue 1 -> off
    expected = np.array([True, False, False, False, True])
    np.testing.assert_array_equal(m, expected)


# ---------------------------------------------------------------------------
# Loader / InputConfigurations parsing
# ---------------------------------------------------------------------------


def test_parse_enabled_bool_false():
    cfg = _make_config([_make_channel_dict("A", enabled=False)])
    ch = cfg.get_channel_list()[0]
    assert ch.enabled is False
    assert ch.off_ranges == ()


def test_parse_enabled_mapping_with_ranges():
    cfg = _make_config([
        _make_channel_dict(
            "A",
            enabled={
                "default": True,
                "off_ranges": [
                    {"start_week": 3, "end_week": 5},
                    {"start_week": 9, "end_week": 9},
                ],
            },
        )
    ])
    ch = cfg.get_channel_list()[0]
    assert ch.enabled is True
    assert ch.off_ranges == ((3, 5), (9, 9))


def test_parse_sticky_pause_ranges():
    cfg = _make_config([
        _make_channel_dict(
            "A",
            sticky_pause_ranges=[
                {
                    "start_week": 2,
                    "end_week": 5,
                    "start_probability": 0.1,
                    "continue_probability": 0.9,
                },
            ],
        )
    ])
    specs = cfg.get_channel_list()[0].sticky_pause_ranges
    assert len(specs) == 1
    assert specs[0].start_week == 2
    assert specs[0].end_week == 5
    assert specs[0].start_probability == 0.1
    assert specs[0].continue_probability == 0.9


def test_parse_enabled_mapping_default_false_disables_channel():
    cfg = _make_config([
        _make_channel_dict("A", enabled={"default": False})
    ])
    ch = cfg.get_channel_list()[0]
    assert ch.is_fully_disabled() is True


def test_parse_per_channel_effect_toggles():
    cfg = _make_config([
        _make_channel_dict("A", adstock_enabled=False, saturation_enabled=False)
    ])
    ch = cfg.get_channel_list()[0]
    assert ch.adstock_enabled is False
    assert ch.saturation_enabled is False


def test_parse_global_effect_switches():
    cfg = _make_config(
        [_make_channel_dict("A")],
        adstock_global=False,
        saturation_global=False,
    )
    assert cfg.get_adstock_global() is False
    assert cfg.get_saturation_global() is False


def test_global_switches_default_to_true_when_missing():
    cfg = _make_config([_make_channel_dict("A")])
    assert cfg.get_adstock_global() is True
    assert cfg.get_saturation_global() is True


def test_invalid_off_ranges_start_after_end_raises():
    with pytest.raises(ValueError, match="start_week"):
        _make_config([
            _make_channel_dict(
                "A",
                enabled={"default": True, "off_ranges": [{"start_week": 10, "end_week": 5}]},
            )
        ])


def test_invalid_enabled_type_raises():
    with pytest.raises(ValueError, match="enabled"):
        _make_config([_make_channel_dict("A", enabled=42)])


def test_invalid_off_ranges_missing_keys_raises():
    with pytest.raises(ValueError, match="start_week and end_week"):
        _make_config([
            _make_channel_dict(
                "A",
                enabled={"default": True, "off_ranges": [{"start_week": 3}]},
            )
        ])


# ---------------------------------------------------------------------------
# Spend generation masking
# ---------------------------------------------------------------------------


def test_generate_spend_zeros_fully_disabled_channel():
    cfg = _make_config([
        _make_channel_dict("On"),
        _make_channel_dict("Off", enabled=False),
    ], week_range=10)

    spend = generate_spend(cfg)
    assert spend.shape == (10, 2)
    assert (spend[:, 1] == 0).all(), "fully disabled channel must have zero spend"
    assert (spend[:, 0] > 0).any(), "enabled channel should have some positive spend"


def test_generate_spend_zeros_off_range_weeks():
    cfg = _make_config([
        _make_channel_dict(
            "A",
            enabled={"default": True, "off_ranges": [{"start_week": 4, "end_week": 6}]},
        )
    ], week_range=10)

    spend = generate_spend(cfg)
    # weeks are 1-indexed; row 3 = week 4
    np.testing.assert_array_equal(spend[3:6, 0], np.zeros(3))
    assert (spend[:3, 0] > 0).any()
    assert (spend[6:, 0] > 0).any()


def test_generate_spend_preserves_output_when_no_toggles():
    """Fail-open: channels without any toggle fields behave identically to before."""
    channels = [_make_channel_dict("A"), _make_channel_dict("B")]
    cfg = _make_config(channels, week_range=6, seed=7)
    spend = generate_spend(cfg)
    assert spend.shape == (6, 2)
    assert (spend > 0).all(), "no toggles = no masking, all spend should be positive"


# ---------------------------------------------------------------------------
# Impressions safety-net masking
# ---------------------------------------------------------------------------


def test_generate_impressions_zeros_off_week_rows():
    cfg = _make_config([
        _make_channel_dict(
            "A",
            enabled={"default": True, "off_ranges": [{"start_week": 2, "end_week": 3}]},
            impression_noise=0.05,
        )
    ], week_range=6)

    spend = generate_spend(cfg)
    imp = generate_impressions(cfg, spend)
    np.testing.assert_array_equal(imp[1:3, 0], np.zeros(2))


def test_generate_impressions_zeros_fully_disabled_channel():
    cfg = _make_config([
        _make_channel_dict("A"),
        _make_channel_dict("B", enabled=False, impression_noise=0.05),
    ], week_range=5)
    spend = generate_spend(cfg)
    imp = generate_impressions(cfg, spend)
    assert (imp[:, 1] == 0).all()


# ---------------------------------------------------------------------------
# Revenue semantics (Policy A)
# ---------------------------------------------------------------------------


def test_revenue_fully_disabled_channel_is_zero_everywhere():
    cfg = _make_config([
        _make_channel_dict(
            "Off",
            enabled=False,
            baseline_revenue=12345.0,
            revenue_noise=0.2,
        )
    ], week_range=8)
    spend = generate_spend(cfg)
    imp = generate_impressions(cfg, spend)
    rev = generate_revenue(cfg, imp)
    # No spend, no impressions, no baseline, no noise for fully-disabled channels.
    np.testing.assert_array_equal(rev[:, 0], np.zeros(8))


def test_revenue_policy_a_preserves_adstock_echo_on_off_weeks():
    """
    Channel A has spend on weeks 1-4, then off weeks 5-7, then spend again.
    With geometric adstock (lambda=0.5) revenue should decay on off weeks,
    not jump to zero. Spend and impressions must be zero on those weeks.
    """
    cfg = _make_config([
        _make_channel_dict(
            "A",
            enabled={"default": True, "off_ranges": [{"start_week": 5, "end_week": 7}]},
            baseline_revenue=0.0,
            adstock={"type": "exponential", "lambda": 0.5, "lag": 5},
            saturation={"type": "linear", "slope": 1.0},
            impression_noise=0.0,
            revenue_noise=0.0,
        )
    ], week_range=10, seed=101)

    spend = generate_spend(cfg)
    imp = generate_impressions(cfg, spend)
    rev = generate_revenue(cfg, imp)

    # Spend and impressions zero on off weeks (rows 4..6 = weeks 5..7).
    np.testing.assert_array_equal(spend[4:7, 0], np.zeros(3))
    np.testing.assert_array_equal(imp[4:7, 0], np.zeros(3))

    # Revenue on the first off week must still be positive (adstock echo).
    assert rev[4, 0] > 0, "Policy A: week 5 should have non-zero echo revenue"
    # Echo should decay while no new spend arrives.
    assert rev[4, 0] > rev[5, 0] > rev[6, 0], (
        f"echo should decay across off weeks, got {rev[4:7, 0]}"
    )
    # After spend resumes (week 8 = row 7), revenue should be higher again
    # (new spend + residual echo >= echo-only tail).
    assert rev[7, 0] > rev[6, 0]


def test_revenue_adstock_disabled_per_channel_skips_echo():
    """With adstock_enabled: false, off-week rows only carry baseline (+ noise)."""
    baseline = 1000.0
    cfg = _make_config([
        _make_channel_dict(
            "A",
            enabled={"default": True, "off_ranges": [{"start_week": 5, "end_week": 6}]},
            adstock_enabled=False,
            baseline_revenue=baseline,
            impression_noise=0.0,
            revenue_noise=0.0,
            adstock={"type": "exponential", "lambda": 0.5, "lag": 5},
            saturation={"type": "linear", "slope": 1.0},
        )
    ], week_range=8, seed=7)

    spend = generate_spend(cfg)
    imp = generate_impressions(cfg, spend)
    rev = generate_revenue(cfg, imp)

    # Off weeks: impressions are 0, adstock is off → transformed_imp=0 → revenue = baseline.
    np.testing.assert_allclose(rev[4:6, 0], np.array([baseline, baseline]))


def test_revenue_global_adstock_off_disables_adstock_everywhere():
    baseline = 500.0
    cfg = _make_config(
        [
            _make_channel_dict(
                "A",
                enabled={"default": True, "off_ranges": [{"start_week": 3, "end_week": 4}]},
                baseline_revenue=baseline,
                impression_noise=0.0,
                revenue_noise=0.0,
                adstock={"type": "exponential", "lambda": 0.5, "lag": 5},
                saturation={"type": "linear", "slope": 1.0},
            )
        ],
        week_range=6,
        seed=3,
        adstock_global=False,
    )
    spend = generate_spend(cfg)
    imp = generate_impressions(cfg, spend)
    rev = generate_revenue(cfg, imp)

    # Adstock is globally off → on off weeks revenue collapses to baseline.
    np.testing.assert_allclose(rev[2:4, 0], np.array([baseline, baseline]))


def test_revenue_saturation_disabled_uses_raw_impressions():
    """
    With saturation disabled (per-channel), revenue on active weeks follows
    impressions * true_roi + baseline exactly when adstock has lag=0.
    """
    cfg = _make_config([
        _make_channel_dict(
            "A",
            saturation_enabled=False,
            true_roi=3.0,
            baseline_revenue=0.0,
            impression_noise=0.0,
            revenue_noise=0.0,
            adstock={"type": "linear", "lag": 0},
            saturation={"type": "hill", "slope": 2.0, "K": 1.0},  # would squash values
        )
    ], week_range=4, seed=11)

    spend = generate_spend(cfg)
    imp = generate_impressions(cfg, spend)
    rev = generate_revenue(cfg, imp)

    # If saturation were on, a hill with K=1 and large impressions saturates to ~roi.
    # With saturation off we should get revenue == impressions * 3.0 exactly.
    np.testing.assert_allclose(rev[:, 0], imp[:, 0] * 3.0)


# ---------------------------------------------------------------------------
# End-to-end pipeline + CSV
# ---------------------------------------------------------------------------


def test_construct_csv_totals_reflect_masked_matrices():
    cfg = _make_config([
        _make_channel_dict(
            "A",
            enabled={"default": True, "off_ranges": [{"start_week": 2, "end_week": 3}]},
            impression_noise=0.0,
            revenue_noise=0.0,
            adstock={"type": "exponential", "lambda": 0.5, "lag": 3},
            saturation={"type": "linear", "slope": 1.0},
        ),
        _make_channel_dict("B", enabled=False, baseline_revenue=9999.0),
    ], week_range=6, seed=21)

    spend = generate_spend(cfg)
    imp = generate_impressions(cfg, spend)
    rev = generate_revenue(cfg, imp)
    df = construct_csv(cfg, spend, imp, rev)

    # Disabled channel contributes nothing anywhere.
    assert (df["B_spend"] == 0).all()
    assert (df["B_impressions"] == 0).all()
    assert (df["B_revenue"] == 0).all()

    # Off weeks (rows 1..2 == weeks 2..3): spend & impressions zero, revenue may
    # be non-zero thanks to adstock echo (Policy A).
    assert (df.loc[df["week"].isin([2, 3]), "A_spend"] == 0).all()
    assert (df.loc[df["week"].isin([2, 3]), "A_impressions"] == 0).all()
    # At least one off week should show the echo (after week 1 had positive spend).
    assert (df.loc[df["week"].isin([2, 3]), "A_revenue"] > 0).any()

    # Totals are consistent with the masked columns.
    np.testing.assert_allclose(
        df["total_spend"].to_numpy(),
        (df["A_spend"] + df["B_spend"]).to_numpy(),
    )
    np.testing.assert_allclose(
        df["total_impressions"].to_numpy(),
        (df["A_impressions"] + df["B_impressions"]).to_numpy(),
    )
    np.testing.assert_allclose(
        df["revenue"].to_numpy(),
        (df["A_revenue"] + df["B_revenue"]).to_numpy(),
    )


def test_invalid_sticky_pause_probability_raises():
    with pytest.raises(ValueError):
        _make_config([
            _make_channel_dict(
                "A",
                sticky_pause_ranges=[
                    {
                        "start_week": 1,
                        "end_week": 2,
                        "start_probability": 1.5,
                        "continue_probability": 0.5,
                    },
                ],
            )
        ])


def test_generate_spend_respects_sticky_always_pause_window():
    cfg = _make_config(
        [
            _make_channel_dict(
                "A",
                sticky_pause_ranges=[
                    {
                        "start_week": 1,
                        "end_week": 3,
                        "start_probability": 1.0,
                        "continue_probability": 1.0,
                    },
                ],
            )
        ],
        week_range=6,
        seed=500,
    )
    spend = generate_spend(cfg)
    assert (spend[:3, 0] == 0).all()
    assert (spend[3:, 0] > 0).all()


def test_generate_impressions_matches_sticky_spend_mask():
    cfg = _make_config(
        [
            _make_channel_dict(
                "A",
                sticky_pause_ranges=[
                    {
                        "start_week": 1,
                        "end_week": 2,
                        "start_probability": 1.0,
                        "continue_probability": 1.0,
                    },
                ],
                impression_noise=0.0,
            )
        ],
        week_range=5,
        seed=501,
    )
    spend = generate_spend(cfg)
    imp = generate_impressions(cfg, spend)
    assert (imp[:2, 0] == 0).all()
    assert (imp[2:, 0] > 0).all()


def test_backward_compat_no_toggle_fields_identical_output():
    """
    A config with no toggle fields anywhere is a no-op at the masking layer.
    We verify reproducibility by interleaving (load, generate) pairs with the
    same seed -- the loader reseeds the shared RNG each time, so each
    generate_* call starts from the same state.
    """
    channels = [
        _make_channel_dict("A", impression_noise=0.0, revenue_noise=0.0),
        _make_channel_dict("B", impression_noise=0.0, revenue_noise=0.0),
    ]

    cfg1 = _make_config(channels, week_range=8, seed=99)
    s1 = generate_spend(cfg1)
    i1 = generate_impressions(cfg1, s1)
    r1 = generate_revenue(cfg1, i1)

    cfg2 = _make_config(channels, week_range=8, seed=99)
    s2 = generate_spend(cfg2)
    i2 = generate_impressions(cfg2, s2)
    r2 = generate_revenue(cfg2, i2)

    np.testing.assert_allclose(s1, s2)
    np.testing.assert_allclose(i1, i2)
    np.testing.assert_allclose(r1, r2)

    # No off-week masking was applied (all spend should be positive).
    assert (s1 > 0).all()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    print("Channel toggle tests...")

    # Channel dataclass
    test_channel_toggle_defaults_fail_open()
    test_channel_is_on_handles_off_ranges_inclusively()
    test_channel_fully_disabled_is_always_off()
    test_channel_on_vector_matches_is_on()
    try:
        test_channel_on_vector_rejects_negative_num_weeks()
    except Exception as e:
        raise AssertionError(f"negative num_weeks test failed: {e}")
    test_spend_allowed_mask_reproducible()
    test_spend_allowed_mask_all_sticky_when_always_continue()
    test_sticky_markov_alternating_when_continue_is_zero()
    test_sticky_freezes_markov_state_across_deterministic_off_week()

    # Loader parsing
    test_parse_enabled_bool_false()
    test_parse_enabled_mapping_with_ranges()
    test_parse_sticky_pause_ranges()
    test_parse_enabled_mapping_default_false_disables_channel()
    test_parse_per_channel_effect_toggles()
    test_parse_global_effect_switches()
    test_global_switches_default_to_true_when_missing()
    try:
        test_invalid_off_ranges_start_after_end_raises()
    except Exception as e:
        raise AssertionError(f"invalid off_ranges test failed: {e}")
    try:
        test_invalid_enabled_type_raises()
    except Exception as e:
        raise AssertionError(f"invalid enabled type test failed: {e}")
    try:
        test_invalid_off_ranges_missing_keys_raises()
    except Exception as e:
        raise AssertionError(f"missing off_ranges keys test failed: {e}")
    try:
        test_invalid_sticky_pause_probability_raises()
    except Exception as e:
        raise AssertionError(f"invalid sticky probability test failed: {e}")

    # Spend / impressions
    test_generate_spend_zeros_fully_disabled_channel()
    test_generate_spend_zeros_off_range_weeks()
    test_generate_spend_preserves_output_when_no_toggles()
    test_generate_impressions_zeros_off_week_rows()
    test_generate_impressions_zeros_fully_disabled_channel()
    test_generate_spend_respects_sticky_always_pause_window()
    test_generate_impressions_matches_sticky_spend_mask()

    # Revenue
    test_revenue_fully_disabled_channel_is_zero_everywhere()
    test_revenue_policy_a_preserves_adstock_echo_on_off_weeks()
    test_revenue_adstock_disabled_per_channel_skips_echo()
    test_revenue_global_adstock_off_disables_adstock_everywhere()
    test_revenue_saturation_disabled_uses_raw_impressions()

    # Pipeline
    test_construct_csv_totals_reflect_masked_matrices()
    test_backward_compat_no_toggle_fields_identical_output()

    print("Channel toggle tests passed.")


if __name__ == "__main__":
    main()
