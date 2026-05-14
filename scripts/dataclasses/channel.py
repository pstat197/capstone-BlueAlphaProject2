from dataclasses import dataclass
from typing import List, Union


@dataclass
class Channel:
    channel_name: str
    true_roi: float
    spend_range: List[float]
    baseline_revenue: float
    saturation_function: str

    def get_channel_name(self) -> str:
        return self.channel_name

    def get_true_roi(self) -> float:
        return self.true_roi

    def get_spend_range(self) -> List[float]:
        return self.spend_range

    def get_baseline_revenue(self) -> float:
        return self.baseline_revenue

    def get_saturation_function(self) -> str:
        return self.saturation_function
