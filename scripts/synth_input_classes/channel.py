from synth_input_classes import dataclass
from typing import Dict, List


@dataclass
class Channel:
    channel_name: str
    true_roi: float
    spend_range: List[float]
    baseline_revenue: float
    saturation_function: str
    spend_sampling_gamma_params: Dict[str, float]
    noise_variance: Dict[str, float]

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

    def get_spend_sampling_gamma_params(self) -> Dict[str, float]:
        return self.spend_sampling_gamma_params

    def get_noise_variance(self) -> Dict[str, float]:
        return self.noise_variance
