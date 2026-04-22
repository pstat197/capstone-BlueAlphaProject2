from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np


WeekOffRange = Tuple[int, int]


class StickyPauseRange(NamedTuple):
    """Inclusive week window with Markov (sticky) stochastic pauses.

    Only weeks that are *deterministically* on (see ``on_vector``) participate
    in the chain. When a week is a hard ``off_ranges`` pause, the Markov state
    is frozen (not advanced) so a later week in the same window still uses
    ``continue_probability`` if the previous *sticky* outcome was paused.

    ``start_probability``: P(random pause this week | previous week was not
    sticky-paused), evaluated only at deterministically-on weeks in the window.

    ``continue_probability``: P(random pause this week | previous week was
    sticky-paused), same gating.
    """

    start_week: int
    end_week: int
    start_probability: float
    continue_probability: float


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
    sticky_pause_ranges: Tuple[StickyPauseRange, ...] = field(default_factory=tuple)
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

    def _sticky_pause_off_mask(
        self,
        num_weeks: int,
        rng: np.random.Generator,
        base_on: np.ndarray,
    ) -> np.ndarray:
        """Weeks forced off by sticky stochastic rules (subset of base_on weeks)."""
        combined = np.zeros(num_weeks, dtype=bool)
        for spec in self.sticky_pause_ranges:
            prev_sticky_paused = False
            for w in range(spec.start_week, spec.end_week + 1):
                if w < 1 or w > num_weeks:
                    continue
                idx = w - 1
                if not base_on[idx]:
                    continue
                p = spec.continue_probability if prev_sticky_paused else spec.start_probability
                paused = rng.random() < p
                if paused:
                    combined[idx] = True
                prev_sticky_paused = paused
        return combined

    def spend_allowed_mask(
        self,
        num_weeks: int,
        *,
        channel_index: int,
        config_seed: Optional[int],
    ) -> np.ndarray:
        """
        Boolean mask (num_weeks,) — True where this channel may have non-zero spend.

        Combines deterministic ``off_ranges`` with optional sticky stochastic
        pauses. Sticky draws use a dedicated ``numpy.random.SeedSequence`` branch
        from ``(config_seed, channel_index)`` so this mask is identical whenever
        recomputed (e.g. spend vs impressions) without consuming the global
        simulation RNG used for gamma / noise draws.

        Weeks follow 1-indexed convention (element 0 is week 1).
        """
        if num_weeks < 0:
            raise ValueError(f"num_weeks must be non-negative, got {num_weeks}")
        if self.is_fully_disabled():
            return np.zeros(num_weeks, dtype=bool)
        base = self.on_vector(num_weeks)
        if not self.sticky_pause_ranges:
            return base
        seed_part = 0 if config_seed is None else int(config_seed)
        sticky_rng = np.random.default_rng(
            np.random.SeedSequence([seed_part, int(channel_index), 0x53544B59])
        )
        sticky_off = self._sticky_pause_off_mask(num_weeks, sticky_rng, base)
        return base & ~sticky_off
