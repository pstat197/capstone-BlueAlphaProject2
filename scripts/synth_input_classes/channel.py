from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


WeekOffRange = Tuple[int, int]


@dataclass
class Channel:
    channel_name: str
    true_roi: float
    spend_range: List[float]
    baseline_revenue: float
    saturation_config: Dict[str, Any]
    adstock_decay_config: Dict[str, Any]
    spend_sampling_gamma_params: Dict[str, float]
    noise_variance: Dict[str, float]
    cpm: float

    # On/off toggles. Fail-open: defaults leave the channel fully active.
    enabled: bool = True
    off_ranges: Tuple[WeekOffRange, ...] = field(default_factory=tuple)
    adstock_enabled: bool = True
    saturation_enabled: bool = True

    def get_channel_name(self) -> str:
        return self.channel_name

    def get_true_roi(self) -> float:
        return self.true_roi

    def get_spend_range(self) -> List[float]:
        return self.spend_range

    def get_baseline_revenue(self) -> float:
        return self.baseline_revenue

    def get_saturation_config(self) -> Dict[str, Any]:
        return self.saturation_config

    def get_adstock_decay_config(self) -> Dict[str, Any]:
        return self.adstock_decay_config

    def get_spend_sampling_gamma_params(self) -> Dict[str, float]:
        return self.spend_sampling_gamma_params

    def get_noise_variance(self) -> Dict[str, float]:
        return self.noise_variance

    def get_cpm(self) -> float:
        return self.cpm

    def is_fully_disabled(self) -> bool:
        """Channel is fully off for the entire run (no spend, no baseline, no noise)."""
        return not self.enabled

    def is_on(self, week: int) -> bool:
        """
        True iff this channel is active in the given 1-indexed week.

        Fully disabled channels are always off. Otherwise, the channel is off
        in any week covered by an inclusive (start_week, end_week) range.
        """
        if not self.enabled:
            return False
        for start, end in self.off_ranges:
            if start <= week <= end:
                return False
        return True

    def on_vector(self, num_weeks: int) -> np.ndarray:
        """
        Boolean mask of shape (num_weeks,) where element w is True iff the
        channel is active in week w+1 (weeks are 1-indexed).
        """
        if num_weeks < 0:
            raise ValueError(f"num_weeks must be non-negative, got {num_weeks}")
        if not self.enabled:
            return np.zeros(num_weeks, dtype=bool)
        mask = np.ones(num_weeks, dtype=bool)
        if not self.off_ranges:
            return mask
        weeks = np.arange(1, num_weeks + 1)
        for start, end in self.off_ranges:
            mask &= ~((weeks >= start) & (weeks <= end))
        return mask
