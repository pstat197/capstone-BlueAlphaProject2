from dataclasses import dataclass
from typing import Dict

import numpy as np

from scripts.revenue_simulation.seasonality_fit import (
    evaluate_deterministic_fourier,
    fit_pattern_multipliers_to_fourier,
    normalize_seasonality_config,
)
from scripts.synth_input_classes.channel import Channel
from scripts.synth_input_classes.channel_seeding import outcome_seasonality_fallback_seed
from scripts.synth_input_classes.input_configurations import InputConfigurations


def _saturation_fn(impressions: np.ndarray, saturation_config: Dict) -> np.ndarray:
    """
    Apply saturation function to impressions.

    Supported types:
      - 'hill':            x^slope / (x^slope + K^slope)
      - 'diminishing_returns': x / (1 + beta * x)
      - 'linear':          linear scaling slope * impressions (default slope=1 → identity)
    """
    saturation_type = (saturation_config or {}).get("type", "linear")

    if saturation_type == "hill":
        slope = float(saturation_config.get("slope", 1.0))
        K = float(saturation_config.get("K", 1.0))
        x_a = np.power(impressions, slope)
        K_a = np.power(K, slope)
        return x_a / (x_a + K_a + 1e-9)

    if saturation_type == "diminishing_returns":
        beta = float(saturation_config.get("beta", 0.0))
        return impressions / (1.0 + beta * impressions)

    if saturation_type == "linear":
        slope = float(saturation_config.get("slope", 1.0))
        return slope * impressions.astype(float, copy=True)

    raise ValueError(
        f'Unknown saturation_config type "{saturation_type}". '
        'Expected one of: "hill", "diminishing_returns", "linear".'
    )


def _adstock_decay(impressions: np.ndarray, adstock_config: Dict) -> np.ndarray:
    """
    Apply adstock decay to impressions.

    Supported types:
      - 'geometric' or 'exponential': Meridian-style geometric decay
        (weights α^s, s=0..L, normalized; α in [0,1] via ``lambda`` / ``decay_rate``).
      - 'binomial': Meridian binomial decay on the same α ∈ [0,1] scale (see Google Meridian docs).
      - 'weighted': finite impulse response; weights are normalized to sum to 1
      - 'linear': if lag <= 0, return impressions; else uniform moving average (weights sum to 1)

    Default type (when omitted) is ``geometric`` to match Meridian-style decay kernels.
    """
    adstock_config = adstock_config or {}
    adstock_type = adstock_config.get("type", "geometric")

    if adstock_type == "linear":
        lag = int(adstock_config.get("lag", 0))
        if lag < 0:
            raise ValueError(f"adstock lag must be non-negative, got {lag}.")
        if lag == 0:
            return impressions.astype(float, copy=True)
        weights = np.ones(lag + 1, dtype=float) / (lag + 1)
        return np.convolve(impressions, weights, mode="full")[: len(impressions)]

    if adstock_type in ("geometric", "exponential"):
        alpha = float(adstock_config.get("lambda", adstock_config.get("decay_rate", 0.5)))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        lag = int(adstock_config.get("lag", 0))
        if lag < 0:
            raise ValueError(f"adstock lag must be non-negative, got {lag}.")
        lag_array = np.arange(lag + 1, dtype=float)
        if alpha <= 0.0:
            decay_weights = np.zeros(lag + 1, dtype=float)
            decay_weights[0] = 1.0
        else:
            decay_weights = np.power(alpha, lag_array)
            decay_weights /= decay_weights.sum()
        return np.convolve(impressions, decay_weights, mode="full")[: len(impressions)]

    if adstock_type == "binomial":
        # Meridian: w(s;α) = (1 - s/(1+L))^α*,  α* = 1/α - 1,  L = max lag; then normalize.
        alpha = float(adstock_config.get("lambda", adstock_config.get("decay_rate", 0.5)))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        L = int(adstock_config.get("lag", 0))
        if L < 0:
            raise ValueError(f"adstock lag must be non-negative, got {L}.")
        lag_array = np.arange(L + 1, dtype=float)
        if alpha <= 0.0:
            decay_weights = np.zeros(L + 1, dtype=float)
            decay_weights[0] = 1.0
        elif alpha >= 1.0:
            decay_weights = np.ones(L + 1, dtype=float)
            decay_weights /= decay_weights.sum()
        else:
            alpha_star = 1.0 / alpha - 1.0
            base = 1.0 - lag_array / (1.0 + float(L))
            base = np.maximum(base, 0.0)
            decay_weights = np.power(base, alpha_star)
            decay_weights /= decay_weights.sum()
        return np.convolve(impressions, decay_weights, mode="full")[: len(impressions)]

    if adstock_type == "weighted":
        weights = adstock_config.get("weights", [1.0])
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr /= weights_arr.sum()
        return np.convolve(impressions, weights_arr, mode="full")[: len(impressions)]

    raise ValueError(
        f'Unknown adstock_decay_config type "{adstock_type}". '
        'Expected one of: "geometric", "exponential", "binomial", "weighted", "linear".'
    )

# seasonality functions
def _fourier_seasonality(t, period, K=1, scale=0.1, seed=None):
    '''
    t: time index array (np.ndarray)
    period: seasonality period (e.g. 52 for weekly data with yearly seasonality)
    K: number of Fourier harmonics (higher K = more complex patterns)
    scale: overall amplitude of seasonality
    seed: random seed for reproducibility (optional)
    '''
    rng = np.random.default_rng(seed)
    s = np.zeros_like(t, dtype=float)

    for k in range(1, K + 1):
        a_k = rng.normal(0, scale / k)
        b_k = rng.normal(0, scale / k)
        s += a_k * np.sin(2 * np.pi * k * t / period)
        s += b_k * np.cos(2 * np.pi * k * t / period)

    return s

# 'spikes': sparse, non-periodic multiplicative shocks; simulates irregular events (promotions, outages, viral bursts)
def _event_spikes(t, prob=0.02, magnitude=(0.5, 1.5), seed=None):
    rng = np.random.default_rng(seed)
    spikes = np.zeros_like(t, dtype=float)
    mask = rng.random(len(t)) < prob
    spikes[mask] = rng.uniform(*magnitude, size=mask.sum())
    return spikes


def _seasonality(
    t: np.ndarray,
    seasonality_config: Dict,
    *,
    fallback_seed: int | None = None,
) -> np.ndarray:
    """
    Flexible seasonality generator.

    Supported types:
      - 'sin' / 'categorical': normalized to deterministic ``fourier`` (same as load path; supports raw YAML)
      - 'fourier': random multi-harmonic (scale/K) or deterministic (coefficients + intercept)
      - 'hybrid': combination of components
    """

    if not seasonality_config:
        return np.ones_like(t, dtype=float)

    stype = str(seasonality_config.get("type", "sin")).strip().lower()

    if stype in ("sin", "categorical"):
        normalized = normalize_seasonality_config(seasonality_config)
        if not normalized:
            return np.ones_like(t, dtype=float)
        return _seasonality(t, normalized, fallback_seed=fallback_seed)

    if stype == "fourier":
        period = int(seasonality_config["period"])
        if seasonality_config.get("coefficients") is not None:
            return evaluate_deterministic_fourier(t, seasonality_config)
        K = seasonality_config.get("K", 2)
        scale = seasonality_config.get("scale", 0.1)
        seed = seasonality_config.get("seed", None)
        if seed is None:
            seed = fallback_seed
        s = _fourier_seasonality(t, period, K, scale, seed)
        return 1 + s

    if stype == "hybrid":
        components = seasonality_config.get("components", [])
        s = np.ones_like(t, dtype=float)

        for comp in components:
            ctype = comp["type"]

            if ctype == "fourier":
                if comp.get("coefficients") is not None:
                    mult = evaluate_deterministic_fourier(t, comp)
                    s *= mult
                else:
                    comp_seed = comp.get("seed", None)
                    if comp_seed is None:
                        comp_seed = fallback_seed
                    val = _fourier_seasonality(
                        t,
                        comp["period"],
                        comp.get("K", 1),
                        comp.get("scale", 0.1),
                        comp_seed,
                    )
                    s *= (1 + val)

            elif ctype == "categorical":
                raw_pat = comp.get("pattern") or []
                try:
                    plist = [float(x) for x in raw_pat]
                except (TypeError, ValueError):
                    plist = [1.0, 1.0]
                cfg_f = fit_pattern_multipliers_to_fourier(plist)
                if cfg_f:
                    s *= evaluate_deterministic_fourier(t, cfg_f)
                else:
                    s *= 1.0

            elif ctype == "spikes":
                comp_seed = comp.get("seed", None)
                if comp_seed is None:
                    comp_seed = fallback_seed
                val = _event_spikes(
                    t,
                    prob=comp.get("prob", 0.02),
                    magnitude=comp.get("magnitude", (0.5, 1.5)),
                    seed=comp_seed,
                )
                s *= (1 + val)

            else:
                raise ValueError(f"Unknown hybrid component type: {ctype}")

        return s

    raise ValueError(f'Unknown seasonality type "{stype}"')

def _outcome_baseline_plus_trend(
    week_range: int, base_revenue: float, trend_slope: float
) -> np.ndarray:
    """Weekly linear ramp ``baseline + trend_slope * t`` with ``t`` in ``0 .. week_range-1``."""
    t = np.arange(week_range, dtype=float)
    return base_revenue + trend_slope * t


def _simulator_outcome_mu_t(
    baseline_trend: np.ndarray,
    t: np.ndarray,
    seasonality_config: Dict,
    *,
    fallback_seed: int | None,
) -> np.ndarray:
    """
    Time-varying intercept ``μ_t`` used in this simulator's additive revenue mean.

    **Meridian (reference):** ``μ_t`` is a time-varying intercept from knot values
    ``b_1,…,b_K`` at times ``s_1,…,s_K``, with linear interpolation between the two
    bracketing knots:
    ``μ_t = w(t) b_{ℓ(t)} + (1 - w(t)) b_{u(t)}`` (see Meridian model spec, § μ_t
    parameters).

    **This simulator:** we do not draw or interpolate knot values. For synthetic
    ground truth we set

    ``μ_t^sim = (baseline + trend·t) · σ_t``,

    where ``σ_t`` is the outcome seasonality multiplier from YAML (≈ 1 when
    disabled) and ``t`` is the same 0-based week index as ``baseline_trend``.
    The **mean** weekly KPI still follows Meridian's **additive** layout
    ``y_t ≈ μ_t^sim + Σ_c M_{c,t}`` before homoskedastic Gaussian noise.
    """
    sigma_t = _seasonality(t, seasonality_config, fallback_seed=fallback_seed)
    return baseline_trend * sigma_t


def _outcome_revenue_noise(
    base_series: np.ndarray,
    rng: np.random.Generator,
    *,
    revenue_variance: float,
) -> np.ndarray:
    """
    One independent Gaussian draw per week on **total** revenue (Meridian-style additive error).

    ``revenue_variance`` is the variance of that shock in **squared KPI units** (same units as
    revenue squared); standard deviation is ``sqrt(revenue_variance)`` and does **not** scale
    with the level of revenue that week.
    """
    if revenue_variance < 0:
        raise ValueError(
            f"Outcome revenue noise variance must be non-negative, got {revenue_variance}."
        )
    if revenue_variance <= 0:
        return base_series
    sigma = float(np.sqrt(revenue_variance))
    noise = rng.normal(loc=0.0, scale=sigma, size=base_series.shape)
    return base_series + noise


def _calculate_channel_media_revenue(
    channel: Channel,
    num_weeks: int,
    impressions: np.ndarray,
    *,
    adstock_global: bool = True,
    saturation_global: bool = True,
    saturation_before_adstock: bool = False,
) -> np.ndarray:
    """
    Weekly **media-only** revenue for one channel (no baseline, seasonality, trend, or noise).

    Pipeline (default ``saturation_before_adstock=False``):
      impressions
        → adstock (optional per-channel / global toggle)
        → saturation (optional per-channel / global toggle)
        → ROI scaling (true_roi)

    If ``saturation_before_adstock=True``, adstock and saturation steps are swapped.

    On/off semantics (Policy A - "soft off"):
      - Fully disabled channels (`enabled=False`) contribute zero across the
        entire run: no adstock echo.
      - Channels with per-week off ranges have impressions of zero on off
        weeks (masked upstream). Adstock carry-over from prior active weeks
        is intentionally preserved, so off-week rows may still show non-zero
        media revenue from the decaying tail.
    """
    if channel.is_fully_disabled():
        return np.zeros_like(impressions, dtype=float)

    saturation_on = saturation_global and channel.saturation_enabled
    adstock_on = adstock_global and channel.adstock_enabled
    imp_f = impressions.astype(float, copy=True)

    if saturation_before_adstock:
        x = _saturation_fn(imp_f, channel.saturation_config) if saturation_on else imp_f
        transformed_imp = _adstock_decay(x, channel.adstock_decay_config) if adstock_on else x
    else:
        x = _adstock_decay(imp_f, channel.adstock_decay_config) if adstock_on else imp_f
        transformed_imp = _saturation_fn(x, channel.saturation_config) if saturation_on else x
    return transformed_imp * float(channel.true_roi)


@dataclass(frozen=True)
class RevenueGenerationResult:
    """Media contributions per channel plus total weekly revenue (MMM-style)."""

    channel_media_revenue: np.ndarray
    total_revenue: np.ndarray


def generate_revenue(
    config: InputConfigurations, impressions_matrix: np.ndarray
) -> RevenueGenerationResult:
    """
    Map impressions to weekly **media** revenue per channel and one **total** revenue series.

    **Meridian mean structure** (same additive *shape* as in the model spec):
    ``y_{g,t} = μ_t + … + (\\text{media terms}) + ε`` (as referenced in https://developers.google.com/meridian/docs/advanced-modeling/model-spec).

    **Meridian’s μ_t:** knot values ``b_k`` at times ``s_k``, with
    ``μ_t = w(t) b_{ℓ(t)} + (1 - w(t)) b_{u(t)}`` (linear interpolation between
    neighboring knots).

    **This simulator’s μ_t:** closed-form ``μ_t^sim`` from
    :func:`_simulator_outcome_mu_t` — ``(baseline + trend·t) · σ_t`` where ``σ_t``
    is the YAML outcome seasonality multiplier (not knot-based). Week index ``t``
    runs ``0 … T-1`` (``T = week_range``).

    Weekly **mean** before noise: ``Σ_c media_c[t] + μ_t^sim``. Noise uses
    ``noise_variance.revenue`` as shock variance (homoskedastic).

    Parameters
    ----------
    config : InputConfigurations
        Configuration with channel list, week_range, RNG, and outcome revenue parameters.
    impressions_matrix : np.ndarray, shape (num_weeks, num_channels)
        Impressions per channel per week, as produced by generate_impressions.

    Returns
    -------
    RevenueGenerationResult
        ``channel_media_revenue`` shape (num_weeks, num_channels); ``total_revenue`` shape (num_weeks,).
    """
    num_weeks, num_channels = impressions_matrix.shape
    expected_weeks = config.get_week_range()
    channels = config.get_channel_list()

    if num_weeks != expected_weeks:
        raise ValueError(
            f"impressions_matrix has {num_weeks} weeks, config expects {expected_weeks}."
        )
    if num_channels != len(channels):
        raise ValueError(
            f"impressions_matrix has {num_channels} channels, config has {len(channels)}."
        )

    rng = config.get_rng()
    adstock_global = config.get_adstock_global()
    saturation_global = config.get_saturation_global()
    saturation_before_adstock = config.get_media_transform_order() == "saturation_first"
    run_seed = config.get_seed()
    out = np.zeros((num_weeks, num_channels), dtype=float)

    for c, channel in enumerate(channels):
        channel_impressions = impressions_matrix[:, c].astype(float)
        out[:, c] = _calculate_channel_media_revenue(
            channel,
            num_weeks,
            channel_impressions,
            adstock_global=adstock_global,
            saturation_global=saturation_global,
            saturation_before_adstock=saturation_before_adstock,
        )

    media_sum = out.sum(axis=1)
    t = np.arange(num_weeks, dtype=float)
    seasonality_seed = outcome_seasonality_fallback_seed(
        run_seed, config.get_outcome_seasonality_seed_channel_name()
    )
    baseline_trend = _outcome_baseline_plus_trend(
        num_weeks,
        config.get_outcome_baseline_revenue(),
        config.get_outcome_trend_slope(),
    )
    outcome_mu_t = _simulator_outcome_mu_t(
        baseline_trend,
        t,
        config.get_outcome_seasonality_config(),
        fallback_seed=seasonality_seed,
    )
    combined = media_sum + outcome_mu_t
    var_rev = float(config.get_outcome_noise_variance().get("revenue", 0.0))
    total = _outcome_revenue_noise(combined, rng, revenue_variance=var_rev)

    return RevenueGenerationResult(channel_media_revenue=out, total_revenue=total)
