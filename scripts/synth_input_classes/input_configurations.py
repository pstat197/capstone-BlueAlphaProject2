import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scripts.config.defaults import get_default_channel_template
from scripts.revenue_simulation.seasonality_fit import normalize_seasonality_config

from .channel import Channel, StickyPauseRange, WeekOffRange


def _get_defaults(template: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve per-channel config defaults from injected template or default.yaml."""
    resolved_template = template or get_default_channel_template()
    return {
        "saturation_config": dict(resolved_template.get("saturation_config") or {}),
        "adstock_decay_config": dict(resolved_template.get("adstock_decay_config") or {}),
        "spend_sampling_gamma_params": dict(resolved_template.get("spend_sampling_gamma_params") or {}),
        "noise_variance": dict(resolved_template.get("noise_variance") or {}),
        "seasonality_config": dict(resolved_template.get("seasonality_config") or {}),
        "trend_slope": float(resolved_template.get("trend_slope", 0.0)),
        "cpm": resolved_template.get("cpm"),
    }


def _parse_media_transform_order(raw: Any) -> str:
    """Return ``adstock_first`` or ``saturation_first`` (fail-open default: adstock first)."""
    if raw is None:
        return "adstock_first"
    s = str(raw).strip().lower().replace("-", "_")
    if s in ("saturation_first", "saturation_before_adstock"):
        return "saturation_first"
    if s in ("adstock_first", "adstock_before_saturation", ""):
        return "adstock_first"
    # Unknown label: keep runs working; YAML authors can fix typos when they notice defaults.
    return "adstock_first"


def _coerce_bool(value: Any, *, context: str, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{context} must be a boolean if provided, got {type(value).__name__}")


def _parse_off_ranges(value: Any, *, context: str) -> Tuple[WeekOffRange, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of {{start_week, end_week}} entries if provided")

    out: List[WeekOffRange] = []
    for i, item in enumerate(value):
        item_ctx = f"{context}[{i}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{item_ctx} must be a mapping with start_week and end_week")
        if "start_week" not in item or "end_week" not in item:
            raise ValueError(f"{item_ctx} must contain start_week and end_week")
        start_raw = item["start_week"]
        end_raw = item["end_week"]
        if isinstance(start_raw, bool) or isinstance(end_raw, bool):
            raise ValueError(f"{item_ctx} start_week/end_week must be integers")
        try:
            start = int(start_raw)
            end = int(end_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{item_ctx} start_week/end_week must be integers") from exc
        if start > end:
            raise ValueError(f"{item_ctx} has start_week ({start}) > end_week ({end})")
        out.append((start, end))
    return tuple(out)


def _parse_probability(value: Any, *, context: str) -> float:
    if value is None:
        raise ValueError(f"{context} is required")
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a number, not boolean")
    try:
        p = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite number in [0, 1]") from exc
    if not math.isfinite(p):
        raise ValueError(f"{context} must be finite, got {p}")
    if p < 0.0 or p > 1.0:
        raise ValueError(f"{context} must be in [0, 1], got {p}")
    return p


def _parse_sticky_pause_ranges(value: Any, *, context: str) -> Tuple[StickyPauseRange, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(
            f"{context} must be a list of {{start_week, end_week, start_probability, continue_probability}} entries if provided"
        )
    out: List[StickyPauseRange] = []
    for i, item in enumerate(value):
        item_ctx = f"{context}[{i}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{item_ctx} must be a mapping")
        for key in ("start_week", "end_week", "start_probability", "continue_probability"):
            if key not in item:
                raise ValueError(f"{item_ctx} must contain {key}")
        start_raw = item["start_week"]
        end_raw = item["end_week"]
        if isinstance(start_raw, bool) or isinstance(end_raw, bool):
            raise ValueError(f"{item_ctx} start_week/end_week must be integers")
        try:
            start = int(start_raw)
            end = int(end_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{item_ctx} start_week/end_week must be integers") from exc
        if start > end:
            raise ValueError(f"{item_ctx} has start_week ({start}) > end_week ({end})")
        p_start = _parse_probability(
            item["start_probability"],
            context=f"{item_ctx}.start_probability",
        )
        p_cont = _parse_probability(
            item["continue_probability"],
            context=f"{item_ctx}.continue_probability",
        )
        out.append(
            StickyPauseRange(
                start_week=start,
                end_week=end,
                start_probability=p_start,
                continue_probability=p_cont,
            )
        )
    return tuple(out)


def _parse_channel_toggles(
    ch: Mapping[str, Any],
    *,
    context: str,
) -> Tuple[bool, Tuple[WeekOffRange, ...], bool, bool]:
    """
    Extract (enabled, off_ranges, adstock_enabled, saturation_enabled) from a
    raw channel dict. Fail-open: all missing keys default to True / empty.
    """
    enabled_val = ch.get("enabled", True)
    if isinstance(enabled_val, bool):
        enabled = enabled_val
        off_ranges: Tuple[WeekOffRange, ...] = ()
    elif enabled_val is None:
        enabled = True
        off_ranges = ()
    elif isinstance(enabled_val, Mapping):
        enabled = _coerce_bool(
            enabled_val.get("default", True),
            context=f"{context}.enabled.default",
            default=True,
        )
        off_ranges = _parse_off_ranges(
            enabled_val.get("off_ranges"),
            context=f"{context}.enabled.off_ranges",
        )
    else:
        raise ValueError(
            f"{context}.enabled must be a boolean or a mapping with 'default' and/or 'off_ranges'"
        )

    adstock_enabled = _coerce_bool(
        ch.get("adstock_enabled", True),
        context=f"{context}.adstock_enabled",
        default=True,
    )
    saturation_enabled = _coerce_bool(
        ch.get("saturation_enabled", True),
        context=f"{context}.saturation_enabled",
        default=True,
    )

    return enabled, off_ranges, adstock_enabled, saturation_enabled


def _normalize_budget_shifts(raw: Any) -> List[Dict[str, Any]]:
    """Parse top-level budget_shifts from YAML. Week numbers are 1-based (same as CSV week column)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("budget_shifts must be a list")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"budget_shifts[{i}] must be a mapping")
        t = str(item.get("type", "")).strip().lower()
        if t == "scale":
            try:
                sw = int(item["start_week"])
                factor = float(item["factor"])
            except KeyError as exc:
                raise ValueError(
                    f"budget_shifts[{i}] scale requires start_week and factor"
                ) from exc
            ew = int(item.get("end_week", sw))
            if ew < sw:
                raise ValueError("budget_shifts scale: end_week must be >= start_week")
            out.append({"type": "scale", "start_week": sw, "end_week": ew, "factor": factor})
        elif t == "reallocate":
            try:
                frac = float(item["fraction"])
                sw = int(item["start_week"])
                from_channel = str(item["from_channel"])
                to_channel = str(item["to_channel"])
            except KeyError as exc:
                raise ValueError(
                    f"budget_shifts[{i}] reallocate requires start_week, from_channel, to_channel, and fraction"
                ) from exc
            frac = max(0.0, min(1.0, frac))
            rec: Dict[str, Any] = {
                "type": "reallocate",
                "start_week": sw,
                "from_channel": from_channel,
                "to_channel": to_channel,
                "fraction": frac,
            }
            if "end_week" in item and item["end_week"] is not None:
                ew = int(item["end_week"])
                if ew < sw:
                    raise ValueError("budget_shifts reallocate: end_week must be >= start_week")
                rec["end_week"] = ew
            out.append(rec)
        elif t == "scale_channel":
            try:
                sw = int(item["start_week"])
                factor = float(item["factor"])
                channel_name = str(item["channel_name"])
            except KeyError as exc:
                raise ValueError(
                    f"budget_shifts[{i}] scale_channel requires channel_name, start_week, and factor"
                ) from exc
            ew = int(item.get("end_week", sw))
            if ew < sw:
                raise ValueError("budget_shifts scale_channel: end_week must be >= start_week")
            out.append(
                {
                    "type": "scale_channel",
                    "channel_name": channel_name,
                    "start_week": sw,
                    "end_week": ew,
                    "factor": factor,
                }
            )
        else:
            raise ValueError(f"budget_shifts: unknown type {item.get('type')!r}")
    return out


def _resolve_outcome_revenue_params(
    data: Dict[str, Any],
    channels: List[Channel],
    default_channel_template: Optional[Dict[str, Any]],
) -> Tuple[float, float, Dict[str, Any], Dict[str, float]]:
    """
    Outcome-level baseline / trend / seasonality / noise for total revenue (MMM-style).

    If ``outcome_revenue`` is present and non-empty in ``data``, those values are used
    (missing sub-keys fall back to the default channel template). Otherwise values
    are taken from the first channel in ``channels`` (after merge), or the template
    when there are no channels.
    """
    tmpl = default_channel_template or get_default_channel_template()
    raw = data.get("outcome_revenue")
    if isinstance(raw, Mapping) and raw:
        baseline = float(raw.get("baseline_revenue", tmpl.get("baseline_revenue", 0.0)))
        trend_slope = float(raw.get("trend_slope", tmpl.get("trend_slope", 0.0)))
        sea_raw = raw.get("seasonality_config")
        if isinstance(sea_raw, dict) and sea_raw:
            seasonality_config = normalize_seasonality_config(dict(sea_raw))
        else:
            sc = tmpl.get("seasonality_config") or {}
            seasonality_config = normalize_seasonality_config(dict(sc)) if sc else {}
        nv_raw = raw.get("noise_variance")
        if isinstance(nv_raw, dict) and nv_raw:
            noise_variance = {str(k): float(v) for k, v in nv_raw.items()}
        else:
            noise_variance = {
                str(k): float(v) for k, v in (tmpl.get("noise_variance") or {}).items()
            }
        return baseline, trend_slope, seasonality_config, noise_variance

    if channels:
        fc = channels[0]
        sea = dict(fc.get_seasonality_config() or {})
        return (
            float(fc.get_baseline_revenue()),
            float(fc.get_trend_slope()),
            normalize_seasonality_config(sea),
            {str(k): float(v) for k, v in (fc.get_noise_variance() or {}).items()},
        )

    seasonality_template = tmpl.get("seasonality_config") or {}
    return (
        float(tmpl.get("baseline_revenue", 0.0)),
        float(tmpl.get("trend_slope", 0.0)),
        normalize_seasonality_config(dict(seasonality_template)) if seasonality_template else {},
        {str(k): float(v) for k, v in (tmpl.get("noise_variance") or {}).items()},
    )


def _validate_week_range(week_range: int) -> None:
    if int(week_range) < 1:
        raise ValueError(
            f"week_range must be a positive integer (number of simulated weeks), got {week_range!r}."
        )


def validate_budget_shifts_channel_refs(
    budget_shifts: List[Dict[str, Any]],
    *,
    known_channel_names: set[str],
) -> None:
    """Ensure reallocate / scale_channel rules reference existing ``channel_name`` values (exact match)."""
    for i, rule in enumerate(budget_shifts):
        t = rule.get("type")
        if t == "reallocate":
            f_name = str(rule.get("from_channel", ""))
            t_name = str(rule.get("to_channel", ""))
            if f_name not in known_channel_names:
                raise ValueError(
                    f"budget_shifts[{i}] reallocate from_channel {f_name!r} is not a known channel_name. "
                    f"Known channels: {sorted(known_channel_names)}"
                )
            if t_name not in known_channel_names:
                raise ValueError(
                    f"budget_shifts[{i}] reallocate to_channel {t_name!r} is not a known channel_name. "
                    f"Known channels: {sorted(known_channel_names)}"
                )
        elif t == "scale_channel":
            cname = str(rule.get("channel_name", ""))
            if cname not in known_channel_names:
                raise ValueError(
                    f"budget_shifts[{i}] scale_channel channel_name {cname!r} is not a known channel_name. "
                    f"Known channels: {sorted(known_channel_names)}"
                )


def validate_outcome_noise_variance(noise_variance: Dict[str, float]) -> None:
    """Outcome-level shock variances must be finite and non-negative (same rules as revenue path runtime)."""
    for k, v in noise_variance.items():
        if not math.isfinite(float(v)):
            raise ValueError(
                f"Resolved outcome noise_variance[{k!r}] must be finite, got {v!r}."
            )
        if float(v) < 0.0:
            raise ValueError(
                f"Resolved outcome noise_variance[{k!r}] must be non-negative, got {v}."
            )


def validate_channel_list_for_simulation(channels: List[Channel]) -> None:
    """
    Fail fast on invalid per-channel parameters before spend/impressions/revenue run.

    Enforces: non-empty unique channel names (after strip), non-negative finite ``true_roi``,
    positive finite ``cpm``, ``spend_range`` with two finite bounds ``0 <= min <= max``,
    strictly positive effective Gamma ``shape`` / ``scale``, non-negative finite per-channel
    ``noise_variance`` entries, non-negative ``adstock_decay_config`` ``lag`` when present,
    and for geometric / exponential / binomial adstock, any explicit ``lambda`` or ``decay_rate``
    must lie in ``[0, 1)`` so the decay kernel is not an unintended clipped or divergent-in-length setting.
    """
    tmpl = get_default_channel_template()
    tmpl_gamma = tmpl.get("spend_sampling_gamma_params") or {}
    default_shape = float(tmpl_gamma.get("shape", 2.5))
    default_scale = float(tmpl_gamma.get("scale", 1000.0))

    seen: Dict[str, int] = {}
    for idx, ch in enumerate(channels):
        name = str(ch.get_channel_name() or "").strip()
        if not name:
            raise ValueError(
                f"channel_list[{idx}].channel.channel_name is empty or whitespace-only; "
                "every channel needs a non-empty name."
            )
        if name in seen:
            raise ValueError(
                f"Duplicate channel_name {name!r} (after stripping): entries "
                f"channel_list[{seen[name]}] and channel_list[{idx}] conflict. "
                "Channel names must be unique."
            )
        seen[name] = idx

        roi = float(ch.get_true_roi())
        if not math.isfinite(roi):
            raise ValueError(f"Channel {name!r}: true_roi must be finite, got {roi!r}.")
        if roi < 0.0:
            raise ValueError(
                f"Channel {name!r}: true_roi must be non-negative, got {roi}."
            )

        cpm = float(ch.get_cpm())
        if not math.isfinite(cpm):
            raise ValueError(f"Channel {name!r}: cpm must be finite, got {cpm!r}.")
        if cpm <= 0.0:
            raise ValueError(
                f"Channel {name!r}: cpm must be positive (impressions use spend/cpm), got {cpm}."
            )

        spend_range = ch.get_spend_range()
        if len(spend_range) < 2:
            raise ValueError(
                f"Channel {name!r}: spend_range must have at least two entries [min_spend, max_spend], "
                f"got {spend_range!r}."
            )
        low_sr, high_sr = float(spend_range[0]), float(spend_range[1])
        if not math.isfinite(low_sr) or not math.isfinite(high_sr):
            raise ValueError(
                f"Channel {name!r}: spend_range bounds must be finite, got [{low_sr!r}, {high_sr!r}]."
            )
        if low_sr < 0.0 or high_sr < 0.0:
            raise ValueError(
                f"Channel {name!r}: spend_range bounds must be non-negative, got [{low_sr}, {high_sr}]."
            )
        if low_sr > high_sr:
            raise ValueError(
                f"Channel {name!r}: spend_range must satisfy min <= max, got [{low_sr}, {high_sr}]."
            )

        g_params = ch.get_spend_sampling_gamma_params() or {}
        eff_shape = float(g_params.get("shape", default_shape))
        eff_scale = float(g_params.get("scale", default_scale))
        if not math.isfinite(eff_shape) or not math.isfinite(eff_scale):
            raise ValueError(
                f"Channel {name!r}: spend_sampling_gamma_params shape/scale must be finite "
                f"(effective shape={eff_shape!r}, scale={eff_scale!r})."
            )
        if eff_shape <= 0.0 or eff_scale <= 0.0:
            raise ValueError(
                f"Channel {name!r}: spend_sampling_gamma_params require strictly positive shape and scale "
                f"for Gamma sampling (effective shape={eff_shape}, scale={eff_scale})."
            )

        nv = ch.get_noise_variance() or {}
        for nk, nv_val in nv.items():
            v = float(nv_val)
            if not math.isfinite(v):
                raise ValueError(
                    f"Channel {name!r}: noise_variance[{nk!r}] must be finite, got {nv_val!r}."
                )
            if v < 0.0:
                raise ValueError(
                    f"Channel {name!r}: noise_variance[{nk!r}] must be non-negative, got {v}."
                )

        cfg = ch.get_adstock_decay_config() or {}
        if "lag" in cfg:
            lag_v = int(cfg["lag"])
            if lag_v < 0:
                raise ValueError(
                    f"Channel {name!r}: adstock_decay_config lag must be non-negative, got {lag_v}."
                )

        adstock_type = str(cfg.get("type", "geometric")).strip().lower()
        if adstock_type in ("geometric", "exponential", "binomial"):
            raw = cfg.get("lambda", cfg.get("decay_rate"))
            if raw is not None:
                alpha = float(raw)
                if not math.isfinite(alpha):
                    raise ValueError(
                        f"Channel {name!r}: adstock lambda/decay_rate must be finite, got {raw!r}."
                    )
                if alpha < 0.0 or alpha >= 1.0:
                    raise ValueError(
                        f"Channel {name!r}: for adstock type {adstock_type!r}, "
                        f"lambda/decay_rate must be in [0, 1) (decay on the unit interval); got {alpha}."
                    )


@dataclass
class InputConfigurations:
    run_identifier: str
    week_range: int
    channel_list: List[Channel]
    seed: Optional[int] = None
    correlations: List[Dict[str, Any]] = field(default_factory=list)
    # Global kill-switches for modeling effects. Fail-open defaults: everything on.
    adstock_global: bool = True
    saturation_global: bool = True
    # Weekly media → revenue: apply adstock then saturation (default), or saturation then adstock.
    media_transform_order: str = "adstock_first"
    budget_shifts: List[Dict[str, Any]] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    # Single outcome-level path for total revenue (baseline + linear trend × seasonality + homoskedastic noise).
    outcome_baseline_revenue: float = 0.0
    outcome_trend_slope: float = 0.0
    outcome_seasonality_config: Dict[str, Any] = field(default_factory=dict)
    outcome_noise_variance: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml_dict(
        cls,
        data: Dict[str, Any],
        default_channel_template: Optional[Dict[str, Any]] = None,
    ) -> "InputConfigurations":
        """Build InputConfigurations from a dict using provided defaults or default.yaml."""
        seed = data.get("seed", None)
        if seed is not None:
            seed = int(seed)
        channel_list_raw = data.get("channel_list") or []
        defaults = _get_defaults(default_channel_template)
        channels = []
        for idx, item in enumerate(channel_list_raw):
            ch = item.get("channel") or item
            ctx = f"channel_list[{idx}].channel"
            baseline = ch.get("baseline_revenue", 0.0)
            gamma_cfg = ch.get("spend_sampling_gamma_params")
            spend_sampling_gamma_params = dict(gamma_cfg) if isinstance(gamma_cfg, dict) else dict(defaults["spend_sampling_gamma_params"])
            noise_cfg = ch.get("noise_variance")
            noise_variance = dict(noise_cfg) if isinstance(noise_cfg, dict) else dict(defaults["noise_variance"])
            # Ensure numeric values in gamma_params and noise_variance
            spend_sampling_gamma_params = {k: float(v) for k, v in spend_sampling_gamma_params.items()}
            noise_variance = {k: float(v) for k, v in noise_variance.items()}
            sat_cfg = ch.get("saturation_config")
            saturation_config = dict(sat_cfg) if isinstance(sat_cfg, dict) else dict(defaults["saturation_config"])
            adstock_cfg = ch.get("adstock_decay_config")
            adstock_decay_config = dict(adstock_cfg) if isinstance(adstock_cfg, dict) else dict(defaults["adstock_decay_config"])
            seasonality_cfg = ch.get("seasonality_config")
            seasonality_config = (
                dict(seasonality_cfg) if isinstance(seasonality_cfg, dict) else dict(defaults["seasonality_config"])
            )
            seasonality_config = normalize_seasonality_config(seasonality_config)
            trend_slope = float(ch.get("trend_slope", defaults["trend_slope"]))
            # Ensure numeric types for adstock_decay_config
            if "lag" in adstock_decay_config:
                adstock_decay_config["lag"] = int(adstock_decay_config["lag"])
            if "weights" in adstock_decay_config:
                adstock_decay_config["weights"] = [float(w) for w in adstock_decay_config["weights"]]
            cpm = float(ch.get("cpm", defaults["cpm"]))

            enabled, off_ranges, adstock_enabled, saturation_enabled = _parse_channel_toggles(
                ch, context=ctx
            )
            sticky_pause_ranges = _parse_sticky_pause_ranges(
                ch.get("sticky_pause_ranges"),
                context=f"{ctx}.sticky_pause_ranges",
            )

            channels.append(
                Channel(
                    channel_name=ch.get("channel_name", ""),
                    true_roi=float(ch.get("true_roi", 0.0)),
                    spend_range=list(ch.get("spend_range", [0, 0])),
                    baseline_revenue=float(baseline),
                    trend_slope=trend_slope,
                    seasonality_config=seasonality_config,
                    saturation_config=saturation_config,
                    adstock_decay_config=adstock_decay_config,
                    spend_sampling_gamma_params=spend_sampling_gamma_params,
                    noise_variance=noise_variance,
                    cpm=cpm,
                    enabled=enabled,
                    off_ranges=off_ranges,
                    sticky_pause_ranges=sticky_pause_ranges,
                    adstock_enabled=adstock_enabled,
                    saturation_enabled=saturation_enabled,
                )
            )
        validate_channel_list_for_simulation(channels)

        correlations_raw = data.get("correlations") or []
        correlations = []
        for entry in correlations_raw:
            correlations.append({
                "channels": list(entry.get("channels", [])),
                "rho": float(entry.get("rho", 0.0)),
            })

        adstock_section = data.get("adstock") if isinstance(data.get("adstock"), Mapping) else {}
        saturation_section = data.get("saturation") if isinstance(data.get("saturation"), Mapping) else {}
        adstock_global = _coerce_bool(
            adstock_section.get("global", True) if adstock_section else True,
            context="adstock.global",
            default=True,
        )
        saturation_global = _coerce_bool(
            saturation_section.get("global", True) if saturation_section else True,
            context="saturation.global",
            default=True,
        )

        media_transform_order = _parse_media_transform_order(data.get("media_transform_order"))

        budget_shifts = _normalize_budget_shifts(data.get("budget_shifts"))
        known_names = {ch.get_channel_name() for ch in channels}
        validate_budget_shifts_channel_refs(budget_shifts, known_channel_names=known_names)

        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        ob, ot, osea, onv = _resolve_outcome_revenue_params(data, channels, default_channel_template)
        validate_outcome_noise_variance(onv)

        week_range = int(data.get("week_range", 0))
        _validate_week_range(week_range)

        return cls(
            run_identifier=str(data.get("run_identifier", "")),
            week_range=week_range,
            channel_list=channels,
            seed=seed,
            correlations=correlations,
            adstock_global=adstock_global,
            saturation_global=saturation_global,
            media_transform_order=media_transform_order,
            budget_shifts=budget_shifts,
            rng=rng,
            outcome_baseline_revenue=ob,
            outcome_trend_slope=ot,
            outcome_seasonality_config=osea,
            outcome_noise_variance=onv,
        )

    def get_run_identifier(self) -> str:
        return self.run_identifier

    def get_week_range(self) -> int:
        return self.week_range

    def get_channel_list(self) -> List[Channel]:
        return self.channel_list

    def get_seed(self) -> Optional[int]:
        return self.seed

    def get_correlations(self) -> List[Dict[str, Any]]:
        return self.correlations

    def get_adstock_global(self) -> bool:
        return self.adstock_global

    def get_saturation_global(self) -> bool:
        return self.saturation_global

    def get_media_transform_order(self) -> str:
        """``adstock_first`` or ``saturation_first``."""
        return self.media_transform_order

    def get_budget_shifts(self) -> List[Dict[str, Any]]:
        return self.budget_shifts

    def get_rng(self) -> np.random.Generator:
        """Return this run's dedicated RNG instance."""
        return self.rng

    def get_outcome_baseline_revenue(self) -> float:
        return float(self.outcome_baseline_revenue)

    def get_outcome_trend_slope(self) -> float:
        return float(self.outcome_trend_slope)

    def get_outcome_seasonality_config(self) -> Dict[str, Any]:
        return dict(self.outcome_seasonality_config)

    def get_outcome_noise_variance(self) -> Dict[str, float]:
        return dict(self.outcome_noise_variance)
