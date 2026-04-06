from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .channel import Channel

# Minimal fallback when from_yaml_dict is called without default_channel_template (e.g. tests).
# Normal runs use the loader, which injects the template from default.yaml.
_MINIMAL_CHANNEL_DEFAULTS: Dict[str, Any] = {
    "saturation_config": {"type": "linear", "slope": 1.0, "K": 50000.0, "beta": 0.5},
    "adstock_decay_config": {"type": "linear", "lambda": 0.5, "lag": 10, "weights": [1.0]},
    "spend_sampling_gamma_params": {"shape": 2.5, "scale": 1000},
    "noise_variance": {"impression": 0.1, "revenue": 0.1},
    "cpm": 10.0,
}


def _get_defaults(template: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve per-channel config defaults from injected template or minimal fallback."""
    if template:
        return {
            "saturation_config": dict(template.get("saturation_config") or _MINIMAL_CHANNEL_DEFAULTS["saturation_config"]),
            "adstock_decay_config": dict(template.get("adstock_decay_config") or _MINIMAL_CHANNEL_DEFAULTS["adstock_decay_config"]),
            "spend_sampling_gamma_params": dict(template.get("spend_sampling_gamma_params") or _MINIMAL_CHANNEL_DEFAULTS["spend_sampling_gamma_params"]),
            "noise_variance": dict(template.get("noise_variance") or _MINIMAL_CHANNEL_DEFAULTS["noise_variance"]),
            "cpm": float(template.get("cpm", _MINIMAL_CHANNEL_DEFAULTS["cpm"])),
        }
    return {k: dict(v) if isinstance(v, dict) else v for k, v in _MINIMAL_CHANNEL_DEFAULTS.items()}


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
            sw = int(item["start_week"])
            factor = float(item["factor"])
            ew = int(item.get("end_week", sw))
            if ew < sw:
                raise ValueError("budget_shifts scale: end_week must be >= start_week")
            out.append({"type": "scale", "start_week": sw, "end_week": ew, "factor": factor})
        elif t == "reallocate":
            frac = float(item["fraction"])
            frac = max(0.0, min(1.0, frac))
            out.append(
                {
                    "type": "reallocate",
                    "start_week": int(item["start_week"]),
                    "from_channel": str(item["from_channel"]),
                    "to_channel": str(item["to_channel"]),
                    "fraction": frac,
                }
            )
        else:
            raise ValueError(f"budget_shifts: unknown type {item.get('type')!r}")
    return out


@dataclass
class InputConfigurations:
    run_identifier: str
    week_range: int
    channel_list: List[Channel]
    seed: Optional[int] = None
    budget_shifts: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_yaml_dict(
        cls,
        data: Dict[str, Any],
        default_channel_template: Optional[Dict[str, Any]] = None,
    ) -> "InputConfigurations":
        """Build InputConfigurations from a dict. When a channel omits a config, use default_channel_template if provided (e.g. from loader), else minimal fallback."""
        seed = data.pop("seed", None)
        if seed is not None:
            seed = int(seed)
        channel_list_raw = data.get("channel_list") or []
        defaults = _get_defaults(default_channel_template)
        channels = []
        for item in channel_list_raw:
            ch = item.get("channel") or item
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
            cpm = float(ch.get("cpm") if ch.get("cpm") is not None else defaults["cpm"])
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
                )
            )
        budget_shifts = _normalize_budget_shifts(data.get("budget_shifts"))
        return cls(
            run_identifier=str(data.get("run_identifier", "")),
            week_range=int(data.get("week_range", 0)),
            channel_list=channels,
            seed=seed,
            budget_shifts=budget_shifts,
        )

    def get_run_identifier(self) -> str:
        return self.run_identifier

    def get_week_range(self) -> int:
        return self.week_range

    def get_channel_list(self) -> List[Channel]:
        return self.channel_list

    def get_seed(self) -> Optional[int]:
        return self.seed

    def get_budget_shifts(self) -> List[Dict[str, Any]]:
        return self.budget_shifts

    def get_rng(self):  # -> np.random.Generator (avoid np import at module level)
        """Return the RNG for this config (same one used during load, seeded with get_seed() if set)."""
        from scripts.config.noise import get_default_rng
        return get_default_rng()