from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .channel import Channel, WeekOffRange


def _get_defaults(template: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve per-channel config defaults from injected template or default.yaml."""
    from scripts.config.defaults import get_default_channel_template
    resolved_template = template or get_default_channel_template()
    return {
        "saturation_config": dict(resolved_template.get("saturation_config") or {}),
        "adstock_decay_config": dict(resolved_template.get("adstock_decay_config") or {}),
        "spend_sampling_gamma_params": dict(resolved_template.get("spend_sampling_gamma_params") or {}),
        "noise_variance": dict(resolved_template.get("noise_variance") or {}),
        "cpm": resolved_template.get("cpm"),
    }


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
            # Ensure numeric types for adstock_decay_config
            if "lag" in adstock_decay_config:
                adstock_decay_config["lag"] = int(adstock_decay_config["lag"])
            if "weights" in adstock_decay_config:
                adstock_decay_config["weights"] = [float(w) for w in adstock_decay_config["weights"]]
            cpm = float(ch.get("cpm", defaults["cpm"]))

            enabled, off_ranges, adstock_enabled, saturation_enabled = _parse_channel_toggles(
                ch, context=ctx
            )

            channels.append(
                Channel(
                    channel_name=ch.get("channel_name", ""),
                    true_roi=float(ch.get("true_roi", 0.0)),
                    spend_range=list(ch.get("spend_range", [0, 0])),
                    baseline_revenue=float(baseline),
                    saturation_config=saturation_config,
                    adstock_decay_config=adstock_decay_config,
                    spend_sampling_gamma_params=spend_sampling_gamma_params,
                    noise_variance=noise_variance,
                    cpm=cpm,
                    enabled=enabled,
                    off_ranges=off_ranges,
                    adstock_enabled=adstock_enabled,
                    saturation_enabled=saturation_enabled,
                )
            )
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

        return cls(
            run_identifier=str(data.get("run_identifier", "")),
            week_range=int(data.get("week_range", 0)),
            channel_list=channels,
            seed=seed,
            correlations=correlations,
            adstock_global=adstock_global,
            saturation_global=saturation_global,
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

    def get_rng(self):  # -> np.random.Generator (avoid np import at module level)
        """Return the RNG for this config (same one used during load, seeded with get_seed() if set)."""
        from scripts.config.noise import get_default_rng
        return get_default_rng()
