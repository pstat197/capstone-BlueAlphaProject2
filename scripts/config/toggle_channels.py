"""
Toggle configuration for MMM synthetic data generation.

This module is import-safe and self-contained: it has no runtime dependency on
the rest of the ``scripts/`` pipeline. Other modules are expected to import
from here and call in at well-defined points.

## Intended integration points

- ``generate_spend``          -> wrap output with ``mask_spend_matrix``
- ``generate_impressions``    -> wrap output with ``mask_impressions_matrix``
                                 (preferably mask spend *before* deriving
                                 impressions so CPM math isn't skewed)
- ``_channel_revenue``        -> replace direct saturation/adstock calls with
                                 ``apply_effect_gates`` so per-channel toggles
                                 can disable those effects
- ``construct_csv``           -> optionally pass rows through
                                 ``mask_row_for_disabled_channels`` as a final
                                 belt-and-braces pass

## Fail-open semantics

Every toggle defaults to ``True`` when unspecified. Unknown channels are treated
as active. Dropping this module into a run with no toggle config is a no-op.

## Known limitation

Week-level off schedules only affect their own weeks. Adstock carry-over from
active weeks into subsequent off-weeks is not suppressed here -- that requires
additional work inside the revenue pipeline where the full impression series is
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Mapping, Iterable, Sequence

import yaml


__all__ = [
    "ToggleChannels",
    "WeekOffRange",
    "save_default_config",
    "mask_value_if_channel_off",
    "mask_row_for_disabled_channels",
    "mask_spend_matrix",
    "mask_impressions_matrix",
    "apply_effect_gates",
]


DEFAULT_CONFIG_YAML: str = """\
# Toggle config for MMM synthetic data generation.
#
# All toggles are "fail-open": if a key is omitted, it is treated as `true`.

# This default template is shaped like `scripts/config/default.yaml` so you can
# keep your generation config and on/off toggles in one place.

run_identifier: Default
week_range: 52

channel_list:
  - channel:
      channel_name: Channel 1

      # Optional on/off controls (fail-open defaults to true).
      # You can set it as a boolean:
      #   enabled: false
      # Or as a schedule with off week ranges (inclusive):
      enabled:
        default: true
        off_ranges:
          - start_week: 10
            end_week: 12

      # Optional per-feature toggles (fail-open to true).
      adstock_enabled: true
      saturation_enabled: true

      # --- the rest mirrors default.yaml ---
      cpm: 10.0
      cpm_sampling_range: [0.5, 50]
      spend_sampling_gamma_params:
        shape: 2.5
        scale: 1000
      noise_variance:
        impression: 0.0225
        revenue: 0.1
      true_roi: 2.5
      spend_range: [1000, 50000]
      baseline_revenue: 5000
      saturation_config:
        type: linear
        slope: 1.0
        K: 50000.0
        beta: 0.5
      adstock_decay_config:
        type: linear
        lambda: 0.5
        lag: 10
        weights: [1.0]

# Legacy toggle-only format is also supported:
#
# channels:
#   Channel 1: true
#   Channel 2:
#     default: true
#     off_ranges:
#       - start_week: 10
#         end_week: 12
# adstock:
#   global: true
#   channels:
#     Channel 1: true
# saturation:
#   global: true
#   channels:
#     Channel 1: true
"""


def save_default_config(path: str | Path) -> None:
    Path(path).write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")


def _as_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{context} must be a mapping/dict if provided")


def _as_bool(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{context} must be a boolean if provided")


def _as_bool_mapping(value: Any, *, context: str) -> Mapping[str, bool]:
    mapping = _as_mapping(value, context=context)
    out: dict[str, bool] = {}
    for k, v in mapping.items():
        if not isinstance(k, str):
            k = str(k)
        out[k] = _as_bool(v, context=f"{context}.{k}")
    return out


WeekOffRange = tuple[int, int]


def _as_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer if provided")
    if isinstance(value, int):
        return value
    raise ValueError(f"{context} must be an integer if provided")


def _parse_off_ranges(value: Any, *, context: str) -> Sequence[WeekOffRange]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list if provided")

    out: list[WeekOffRange] = []
    for i, item in enumerate(value):
        item_ctx = f"{context}[{i}]"
        item_map = _as_mapping(item, context=item_ctx)
        if "start_week" not in item_map or "end_week" not in item_map:
            raise ValueError(f"{item_ctx} must contain start_week and end_week")
        start = _as_int(item_map["start_week"], context=f"{item_ctx}.start_week")
        end = _as_int(item_map["end_week"], context=f"{item_ctx}.end_week")
        if start > end:
            raise ValueError(f"{item_ctx} has start_week > end_week")
        out.append((start, end))
    return tuple(out)


def _week_in_ranges(week: int, ranges: Sequence[WeekOffRange]) -> bool:
    for start, end in ranges:
        if start <= week <= end:
            return True
    return False


def _parse_channels_section(value: Any) -> tuple[Mapping[str, bool], Mapping[str, Sequence[WeekOffRange]]]:
    """
    Parse the `channels:` section.

    Supported per-channel formats:
      - bool (simple always on/off)
      - {default: bool, off_ranges: [{start_week: int, end_week: int}, ...]}
    """
    channels_map = _as_mapping(value, context="channels")
    defaults: dict[str, bool] = {}
    off_ranges_by_channel: dict[str, Sequence[WeekOffRange]] = {}

    for raw_key, raw_val in channels_map.items():
        key = raw_key if isinstance(raw_key, str) else str(raw_key)
        if isinstance(raw_val, bool):
            defaults[key] = raw_val
            off_ranges_by_channel[key] = ()
            continue
        if raw_val is None:
            defaults[key] = True
            off_ranges_by_channel[key] = ()
            continue
        if isinstance(raw_val, Mapping):
            val_map = _as_mapping(raw_val, context=f"channels.{key}")
            default = _as_bool(val_map["default"], context=f"channels.{key}.default") if "default" in val_map else True
            defaults[key] = default
            off_ranges_by_channel[key] = _parse_off_ranges(val_map.get("off_ranges"), context=f"channels.{key}.off_ranges")
            continue
        raise ValueError(f"channels.{key} must be a boolean or mapping if provided")

    return defaults, off_ranges_by_channel


def _parse_channel_list_section(
    value: Any,
) -> tuple[
    Mapping[str, bool],
    Mapping[str, Sequence[WeekOffRange]],
    Mapping[str, bool],
    Mapping[str, bool],
]:
    """
    Parse `channel_list:` (shape from `scripts/config/default.yaml`).

    Each element should contain a `channel:` mapping with a `channel_name`.
    Optional toggle keys inside each channel:
      - enabled: bool OR {default: bool, off_ranges: [...]}
      - adstock_enabled: bool
      - saturation_enabled: bool
    """
    if value is None:
        return {}, {}, {}, {}
    if not isinstance(value, list):
        raise ValueError("channel_list must be a list if provided")

    channel_defaults: dict[str, bool] = {}
    channel_off_ranges: dict[str, Sequence[WeekOffRange]] = {}
    adstock_channels: dict[str, bool] = {}
    saturation_channels: dict[str, bool] = {}

    for i, item in enumerate(value):
        item_ctx = f"channel_list[{i}]"
        item_map = _as_mapping(item, context=item_ctx)
        channel_map = _as_mapping(item_map.get("channel"), context=f"{item_ctx}.channel")

        if "channel_name" not in channel_map:
            raise ValueError(f"{item_ctx}.channel must contain channel_name")
        channel_name = channel_map["channel_name"]
        if not isinstance(channel_name, str):
            channel_name = str(channel_name)

        enabled_val = channel_map.get("enabled", True)
        if isinstance(enabled_val, bool):
            channel_defaults[channel_name] = enabled_val
            channel_off_ranges[channel_name] = ()
        elif enabled_val is None or isinstance(enabled_val, Mapping):
            enabled_map = _as_mapping(enabled_val, context=f"{item_ctx}.channel.enabled")
            channel_defaults[channel_name] = (
                _as_bool(enabled_map["default"], context=f"{item_ctx}.channel.enabled.default")
                if "default" in enabled_map
                else True
            )
            channel_off_ranges[channel_name] = _parse_off_ranges(
                enabled_map.get("off_ranges"), context=f"{item_ctx}.channel.enabled.off_ranges"
            )
        else:
            raise ValueError(f"{item_ctx}.channel.enabled must be a boolean or mapping if provided")

        if "adstock_enabled" in channel_map:
            adstock_channels[channel_name] = _as_bool(
                channel_map["adstock_enabled"], context=f"{item_ctx}.channel.adstock_enabled"
            )

        if "saturation_enabled" in channel_map:
            saturation_channels[channel_name] = _as_bool(
                channel_map["saturation_enabled"], context=f"{item_ctx}.channel.saturation_enabled"
            )

    return channel_defaults, channel_off_ranges, adstock_channels, saturation_channels


@dataclass(frozen=True, slots=True)
class ToggleChannels:
    channel_defaults: Mapping[str, bool]
    channel_off_ranges: Mapping[str, Sequence[WeekOffRange]]
    adstock_global: bool
    adstock_channels: Mapping[str, bool]
    saturation_global: bool
    saturation_channels: Mapping[str, bool]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ToggleChannels":
        raw_text = Path(path).read_text(encoding="utf-8")
        loaded = yaml.safe_load(raw_text)

        root = _as_mapping(loaded, context="root")

        channel_defaults: Mapping[str, bool] = {}
        channel_off_ranges: Mapping[str, Sequence[WeekOffRange]] = {}
        adstock_channels_from_list: Mapping[str, bool] = {}
        saturation_channels_from_list: Mapping[str, bool] = {}

        if "channel_list" in root:
            (
                channel_defaults,
                channel_off_ranges,
                adstock_channels_from_list,
                saturation_channels_from_list,
            ) = _parse_channel_list_section(root.get("channel_list"))
        else:
            channel_defaults, channel_off_ranges = _parse_channels_section(root.get("channels"))

        adstock = _as_mapping(root.get("adstock"), context="adstock")
        adstock_global = (
            _as_bool(adstock["global"], context="adstock.global") if "global" in adstock else True
        )
        adstock_channels = dict(
            _as_bool_mapping(adstock.get("channels"), context="adstock.channels")
        )
        for ch, enabled in adstock_channels_from_list.items():
            adstock_channels.setdefault(ch, enabled)

        saturation = _as_mapping(root.get("saturation"), context="saturation")
        saturation_global = (
            _as_bool(saturation["global"], context="saturation.global")
            if "global" in saturation
            else True
        )
        saturation_channels = dict(
            _as_bool_mapping(saturation.get("channels"), context="saturation.channels")
        )
        for ch, enabled in saturation_channels_from_list.items():
            saturation_channels.setdefault(ch, enabled)

        return cls(
            channel_defaults=channel_defaults,
            channel_off_ranges=channel_off_ranges,
            adstock_global=adstock_global,
            adstock_channels=adstock_channels,
            saturation_global=saturation_global,
            saturation_channels=saturation_channels,
        )

    def channel_on(self, channel: str, *, week: int | None = None) -> bool:
        default_on = self.channel_defaults.get(channel, True)
        if not default_on:
            return False
        if week is None:
            return True
        ranges = self.channel_off_ranges.get(channel, ())
        return not _week_in_ranges(week, ranges)

    def adstock_on(self, channel: str) -> bool:
        if not self.adstock_global:
            return False
        return self.adstock_channels.get(channel, True)

    def saturation_on(self, channel: str) -> bool:
        if not self.saturation_global:
            return False
        return self.saturation_channels.get(channel, True)

    def active_channels(self, *, week: int | None = None) -> list[str]:
        channels = set(self.channel_defaults.keys()) | set(self.channel_off_ranges.keys())
        return sorted([ch for ch in channels if self.channel_on(ch, week=week)])

    def active_channels_from(
        self, names: Iterable[str], *, week: int | None = None
    ) -> list[str]:
        """
        Return the subset of `names` (preserving input order) whose channels are
        currently active.

        Prefer this over `active_channels` when the caller has a canonical list
        of channel names from the pipeline config, which may include channels
        not mentioned in the toggle YAML (those fail-open to True).
        """
        return [ch for ch in names if self.channel_on(ch, week=week)]

    def channel_on_vector(self, channel: str, num_weeks: int) -> "Any":
        """
        Return a boolean numpy array of length `num_weeks` where element w is
        True iff `channel` is active in week w+1 (weeks are 1-indexed).

        numpy is imported lazily so this module stays import-safe for callers
        that don't need vectorized masking.
        """
        import numpy as np

        if num_weeks < 0:
            raise ValueError(f"num_weeks must be non-negative, got {num_weeks}")

        if not self.channel_defaults.get(channel, True):
            return np.zeros(num_weeks, dtype=bool)

        mask = np.ones(num_weeks, dtype=bool)
        ranges = self.channel_off_ranges.get(channel, ())
        if not ranges:
            return mask

        weeks = np.arange(1, num_weeks + 1)
        for start, end in ranges:
            mask &= ~((weeks >= start) & (weeks <= end))
        return mask


def mask_value_if_channel_off(
    toggles: ToggleChannels,
    *,
    week: int | None = None,
    channel: str,
    value: Any,
    off_value: str = "-",
) -> Any:
    """
    Convenience helper for generators: if a channel is off, return a placeholder.
    Otherwise return the original value.
    """
    return value if toggles.channel_on(channel, week=week) else off_value


def mask_row_for_disabled_channels(
    toggles: ToggleChannels,
    row: Mapping[str, Any],
    *,
    week: int | None = None,
    week_key: str = "week",
    channel_to_columns: Mapping[str, Iterable[str]],
    off_value: int = 0,
) -> dict[str, Any]:
    """
    Return a copy of `row` where any columns for disabled channels are replaced
    with `off_value` (e.g. '-', 'na').

    This is intentionally schema-agnostic: you provide a mapping of channel name
    to the column names in your row/CSV that should be masked when that channel
    is turned off.
    """
    resolved_week = week
    if resolved_week is None and week_key in row and isinstance(row[week_key], int) and not isinstance(row[week_key], bool):
        resolved_week = row[week_key]

    out = dict(row)
    for channel, cols in channel_to_columns.items():
        if toggles.channel_on(channel, week=resolved_week):
            continue
        for col in cols:
            if col in out:
                out[col] = off_value
    return out


def _mask_matrix_by_channel(
    toggles: ToggleChannels,
    matrix: "Any",
    channel_names: Sequence[str],
    *,
    off_value: float = 0.0,
) -> "Any":
    import numpy as np

    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
    num_weeks, num_channels = matrix.shape
    if len(channel_names) != num_channels:
        raise ValueError(
            f"channel_names length {len(channel_names)} does not match "
            f"matrix columns {num_channels}"
        )

    out = matrix.astype(float, copy=True)
    for c, name in enumerate(channel_names):
        mask = toggles.channel_on_vector(name, num_weeks)
        out[~mask, c] = off_value
    return out


def mask_spend_matrix(
    toggles: ToggleChannels,
    spend_matrix: "Any",
    channel_names: Sequence[str],
    *,
    off_value: float = 0.0,
) -> "Any":
    """
    Return a copy of `spend_matrix` (shape (num_weeks, num_channels)) where
    cells for (week, channel) pairs with the channel off are replaced with
    `off_value`.

    Weeks are 1-indexed (row 0 == week 1). This is the canonical hook for
    wrapping the output of `generate_spend` in the pipeline.
    """
    return _mask_matrix_by_channel(
        toggles, spend_matrix, channel_names, off_value=off_value
    )


def mask_impressions_matrix(
    toggles: ToggleChannels,
    impressions_matrix: "Any",
    channel_names: Sequence[str],
    *,
    off_value: float = 0.0,
) -> "Any":
    """
    Return a copy of `impressions_matrix` (shape (num_weeks, num_channels)) where
    cells for (week, channel) pairs with the channel off are replaced with
    `off_value`.

    Prefer masking spend before impressions are derived; use this as a safety
    net or when spend cannot be modified upstream.
    """
    return _mask_matrix_by_channel(
        toggles, impressions_matrix, channel_names, off_value=off_value
    )


def apply_effect_gates(
    toggles: ToggleChannels,
    channel: str,
    *,
    impressions: "Any",
    saturation_fn,
    adstock_fn,
) -> "Any":
    """
    Conditionally apply saturation and adstock to `impressions` based on the
    per-channel toggles in `toggles`.

    Intended for use inside the revenue-per-channel pipeline step:
    swap direct calls to saturation/adstock with this gated equivalent so
    `saturation_enabled: false` / `adstock_enabled: false` (per channel or
    globally) take effect.

    Order mirrors the existing pipeline: saturation first, then adstock.
    """
    x = saturation_fn(impressions) if toggles.saturation_on(channel) else impressions
    x = adstock_fn(x) if toggles.adstock_on(channel) else x
    return x


def _print_group(title: str, items: Mapping[str, bool]) -> None:
    print(f"{title}:")
    for k in sorted(items.keys()):
        print(f"  - {k}: {items[k]}")


def _demo(cfg_path: Path) -> None:
    toggles = ToggleChannels.from_yaml(cfg_path)

    print(f"Loaded toggles from: {cfg_path}")
    print("Active channels (week 1):", toggles.active_channels(week=1))
    all_channels = sorted(
        set(toggles.channel_defaults.keys()) | set(toggles.channel_off_ranges.keys())
    )
    _print_group(
        "Channels (week 1)", {ch: toggles.channel_on(ch, week=1) for ch in all_channels}
    )
    _print_group(
        "Channels (week 11)",
        {ch: toggles.channel_on(ch, week=11) for ch in all_channels},
    )

    print("Adstock:")
    print(f"  - global: {toggles.adstock_global}")
    _print_group(
        "  - per_channel_effective",
        {ch: toggles.adstock_on(ch) for ch in all_channels},
    )

    print("Saturation:")
    print(f"  - global: {toggles.saturation_global}")
    _print_group(
        "  - per_channel_effective",
        {ch: toggles.saturation_on(ch) for ch in all_channels},
    )

    demo_row: dict[str, Any] = {
        "week": 11,
        "Channel 1_impressions": 1234,
        "Channel 1_revenue": 250.0,
    }
    demo_map = {
        "Channel 1": ("Channel 1_impressions", "Channel 1_revenue"),
    }
    print("Demo row (before):", demo_row)
    print(
        "Demo row (after):",
        mask_row_for_disabled_channels(toggles, demo_row, channel_to_columns=demo_map),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect a ToggleChannels YAML config."
    )
    parser.add_argument(
        "yaml_path",
        nargs="?",
        default=None,
        help="Path to a toggle YAML file (uses the embedded default template if omitted).",
    )
    args = parser.parse_args()

    if args.yaml_path is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "config.yaml"
            save_default_config(cfg_path)
            _demo(cfg_path)
    else:
        _demo(Path(args.yaml_path))
