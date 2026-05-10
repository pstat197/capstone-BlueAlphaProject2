from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from scripts.synth_input_classes.input_configurations import InputConfigurations


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable primitives."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (str, int, float)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "_asdict"):
        # NamedTuple (e.g. StickyPauseRange)
        return _to_jsonable(obj._asdict())  # type: ignore[attr-defined]
    return str(obj)


def extract_ground_truth(config: InputConfigurations) -> Dict[str, Any]:
    """Extract the generative parameters (the "true" model settings) used for this run."""
    channels: List[Dict[str, Any]] = []
    for ch in config.get_channel_list():
        channels.append(
            {
                "channel_name": ch.get_channel_name(),
                "enabled": bool(getattr(ch, "enabled", True)),
                "off_ranges": [
                    [int(start), int(end)] for (start, end) in list(getattr(ch, "off_ranges", ()))
                ],
                "sticky_pause_ranges": [
                    {
                        "start_week": int(r.start_week),
                        "end_week": int(r.end_week),
                        "start_probability": float(r.start_probability),
                        "continue_probability": float(r.continue_probability),
                    }
                    for r in list(getattr(ch, "sticky_pause_ranges", ()))
                ],
                "true_roi": float(ch.get_true_roi()),
                "baseline_revenue": float(ch.get_baseline_revenue()),
                "trend_slope": float(getattr(ch, "trend_slope", 0.0)),
                "seasonality_config": _to_jsonable(ch.get_seasonality_config()),
                "saturation_enabled": bool(getattr(ch, "saturation_enabled", True)),
                "saturation_config": _to_jsonable(ch.get_saturation_config()),
                "adstock_enabled": bool(getattr(ch, "adstock_enabled", True)),
                "adstock_decay_config": _to_jsonable(ch.get_adstock_decay_config()),
                "spend_range": [float(x) for x in ch.get_spend_range()],
                "cpm": float(ch.get_cpm()),
                "spend_sampling_gamma_params": _to_jsonable(ch.get_spend_sampling_gamma_params()),
                "noise_variance": _to_jsonable(ch.get_noise_variance()),
            }
        )

    return _to_jsonable(
        {
            "ground_truth_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_identifier": config.get_run_identifier(),
            "week_range": int(config.get_week_range()),
            "seed": config.get_seed(),
            "global_toggles": {
                "adstock_global": bool(config.get_adstock_global()),
                "saturation_global": bool(config.get_saturation_global()),
            },
            "correlations": config.get_correlations(),
            "budget_shifts": config.get_budget_shifts(),
            "channels": channels,
        }
    )


def write_ground_truth_json(
    out_path: Path,
    ground_truth: Dict[str, Any],
    *,
    overwrite: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(str(out_path))
    out_path.write_text(json.dumps(ground_truth, indent=2, sort_keys=True), encoding="utf-8")

