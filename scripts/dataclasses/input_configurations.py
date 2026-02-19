from dataclasses import dataclass
from typing import List, Dict, Any

from .channel import Channel


@dataclass
class InputConfigurations:
    run_identifier: str
    week_range: int
    noise_level: Dict[str, float]
    channel_list: List[Channel]

    @classmethod
    def from_yaml_dict(cls, data: Dict[str, Any]) -> "InputConfigurations":
        """Build InputConfigurations from a dict loaded from YAML."""
        noise_level = data.get("noise_level") or {}
        channel_list_raw = data.get("channel_list") or []
        channels = []
        for item in channel_list_raw:
            ch = item.get("channel") or item
            baseline = ch.get("baseline_revenue", 0.0)
            channels.append(
                Channel(
                    channel_name=ch.get("channel_name", ""),
                    true_roi=float(ch.get("true_roi", 0.0)),
                    spend_range=list(ch.get("spend_range", [0, 0])),
                    baseline_revenue=float(baseline),
                    saturation_function=str(ch.get("saturation_function", "")),
                )
            )
        return cls(
            run_identifier=str(data.get("run_identifier", "")),
            week_range=int(data.get("week_range", 0)),
            noise_level=dict(noise_level),
            channel_list=channels,
        )

    def get_run_identifier(self) -> str:
        return self.run_identifier

    def get_week_range(self) -> int:
        return self.week_range

    def get_noise_level(self) -> Dict[str, float]:
        return self.noise_level

    def get_channel_list(self) -> List[Channel]:
        return self.channel_list
