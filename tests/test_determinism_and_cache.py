from __future__ import annotations

import numpy as np

from app.cache import canonical_config_hash
from scripts.config.loader import load_config_from_dict
from scripts.impressions_simulation.impressions_generation import generate_impressions
from scripts.main import run_simulation
from scripts.revenue_simulation.revenue_generation import generate_revenue


def _minimal_channel(name: str, *, enabled=None, seasonality_config=None) -> dict:
    ch = {
        "channel_name": name,
        "true_roi": 1.0,
        "spend_range": [200.0, 2000.0],
        "baseline_revenue": 100.0,
        "trend_slope": 0.0,
        "seasonality_config": seasonality_config or {},
        "saturation_config": {"type": "linear", "slope": 1.0},
        "adstock_decay_config": {"type": "linear", "lag": 0},
        "spend_sampling_gamma_params": {"shape": 2.0, "scale": 300.0},
        "noise_variance": {"impression": 0.0, "revenue": 0.0},
        "cpm": 10.0,
    }
    if enabled is not None:
        ch["enabled"] = enabled
    return {"channel": ch}


def test_loader_rejects_non_psd_correlations() -> None:
    cfg = {
        "run_identifier": "bad_psd",
        "week_range": 8,
        "seed": 11,
        "channel_list": [
            _minimal_channel("A"),
            _minimal_channel("B"),
            _minimal_channel("C"),
        ],
        "correlations": [
            {"channels": ["A", "B"], "rho": 0.9},
            {"channels": ["A", "C"], "rho": 0.9},
            {"channels": ["B", "C"], "rho": -0.9},
        ],
    }
    try:
        load_config_from_dict(cfg)
    except ValueError as exc:
        assert "non-PSD" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-PSD correlation matrix")


def test_random_fourier_without_explicit_seed_is_reproducible_from_run_seed() -> None:
    cfg_dict = {
        "run_identifier": "rand_fourier",
        "week_range": 16,
        "seed": 123,
        "channel_list": [
            _minimal_channel(
                "A",
                seasonality_config={"type": "fourier", "period": 8, "K": 3, "scale": 0.2},
            )
        ],
    }
    cfg1 = load_config_from_dict(cfg_dict)
    imp1 = generate_impressions(cfg1, np.full((16, 1), 500.0, dtype=float))
    rev1 = generate_revenue(cfg1, imp1)

    cfg2 = load_config_from_dict(cfg_dict)
    imp2 = generate_impressions(cfg2, np.full((16, 1), 500.0, dtype=float))
    rev2 = generate_revenue(cfg2, imp2)

    np.testing.assert_allclose(rev1, rev2)


def test_cache_hash_is_stable_for_numpy_scalar_equivalents() -> None:
    a = {"seed": np.int64(7), "week_range": np.int64(12), "channel_list": [{"channel": {"channel_name": "A"}}]}
    b = {"seed": 7, "week_range": 12, "channel_list": [{"channel": {"channel_name": "A"}}]}
    assert canonical_config_hash(a) == canonical_config_hash(b)


def test_run_simulation_emits_operational_and_generative_correlations() -> None:
    cfg = load_config_from_dict(
        {
            "run_identifier": "dual_corr",
            "week_range": 20,
            "seed": 3,
            "channel_list": [
                _minimal_channel(
                    "A",
                    enabled={"default": True, "off_ranges": [{"start_week": 6, "end_week": 15}]},
                ),
                _minimal_channel("B"),
            ],
            "correlations": [{"channels": ["A", "B"], "rho": 0.7}],
        }
    )
    _, corr = run_simulation(cfg)
    assert corr is not None
    assert "operational_corr" in corr and "generative_corr" in corr
    op = corr["operational_corr"]["static_corr"]
    gen = corr["generative_corr"]["static_corr"]
    assert op.shape == gen.shape
    assert not np.allclose(op, gen), "toggle masking should alter operational correlation vs generative"
