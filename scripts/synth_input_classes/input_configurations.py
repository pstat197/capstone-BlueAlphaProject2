from synth_input_classes import dataclass
from typing import List, Dict, Any, Optional

from .channel import Channel


@dataclass
class InputConfigurations:
    run_identifier: str
    week_range: int
    channel_list: List[Channel]
    seed: Optional[int] = None

    @classmethod
    def from_yaml_dict(cls, data: Dict[str, Any]) -> "InputConfigurations":
        """Build InputConfigurations from a dict loaded from YAML."""
        seed = data.pop("seed", None)
        if seed is not None:
            seed = int(seed)
        channel_list_raw = data.get("channel_list") or []
        channels = []
        for item in channel_list_raw:
            ch = item.get("channel") or item
            baseline = ch.get("baseline_revenue", 0.0)
            gamma_params = ch.get("spend_sampling_gamma_params") or {}
            noise_variance = ch.get("noise_variance") or {}
            channels.append(
                Channel(
                    channel_name=ch.get("channel_name", ""),
                    true_roi=float(ch.get("true_roi", 0.0)),
                    spend_range=list(ch.get("spend_range", [0, 0])),
                    baseline_revenue=float(baseline),
                    saturation_function=str(ch.get("saturation_function", "")),
                    spend_sampling_gamma_params=dict(gamma_params),
                    noise_variance=dict(noise_variance),
                )
            )
        return cls(
            run_identifier=str(data.get("run_identifier", "")),
            week_range=int(data.get("week_range", 0)),
            channel_list=channels,
            seed=seed,
        )

    def get_run_identifier(self) -> str:
        return self.run_identifier

    def get_week_range(self) -> int:
        return self.week_range

    def get_channel_list(self) -> List[Channel]:
        return self.channel_list

    def get_seed(self) -> Optional[int]:
        return self.seed

    def get_rng(self):  # -> np.random.Generator (avoid np import at module level)
        """Return the RNG for this config (same one used during load, seeded with get_seed() if set)."""
        from scripts.config.noise import get_default_rng
        return get_default_rng()