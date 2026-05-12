"""
Structured validation for user-provided simulator configs.

Two layers, both surfaced through ``/api/config/validate``:

1. ``_collect_structural_issues`` runs cheap, deterministic checks at known
   field paths and returns issues with stable JSON-Pointer-ish ``path``
   arrays so the React UI can pin each error to a specific control.

2. We then attempt the full backend load with
   :func:`scripts.config.loader.load_config_from_dict` and surface any
   exception as a top-level issue with ``severity="error"`` and no path.

Keeping #1 separate from #2 means the form can highlight obvious mistakes
(missing channel name, ROI < 0, etc.) without paying for the full loader,
and #2 acts as the catch-all backstop for anything we don't model
inline yet.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Union

from scripts.config.loader import load_config_from_dict

PathToken = Union[str, int]


class _Issue(dict):
    """Lightweight TypedDict-ish wrapper so callers see a Dict[str, Any]."""

    def __init__(
        self,
        *,
        message: str,
        path: Optional[List[PathToken]] = None,
        section: Optional[str] = None,
        severity: str = "error",
    ) -> None:
        super().__init__(
            message=message,
            path=list(path or []),
            section=section,
            severity=severity,
        )


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _check_outcome(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    outcome = config.get("outcome_revenue") or {}
    if not isinstance(outcome, dict):
        issues.append(
            _Issue(
                message="outcome_revenue must be a mapping",
                path=["outcome_revenue"],
                section="outcome",
            )
        )
        return issues

    baseline = outcome.get("baseline_revenue")
    if baseline is not None and not _is_finite_number(baseline):
        issues.append(
            _Issue(
                message="Baseline revenue must be a finite number",
                path=["outcome_revenue", "baseline_revenue"],
                section="outcome",
            )
        )
    elif _is_finite_number(baseline) and float(baseline) < 0:
        issues.append(
            _Issue(
                message="Baseline revenue must be ≥ 0",
                path=["outcome_revenue", "baseline_revenue"],
                section="outcome",
            )
        )

    noise = outcome.get("noise_variance") or {}
    if isinstance(noise, dict):
        rev = noise.get("revenue")
        if rev is not None and (not _is_finite_number(rev) or float(rev) < 0):
            issues.append(
                _Issue(
                    message="Outcome revenue noise variance must be a non-negative finite number",
                    path=["outcome_revenue", "noise_variance", "revenue"],
                    section="outcome",
                )
            )

    sea = outcome.get("seasonality_config")
    issues.extend(_check_seasonality(sea, path_prefix=["outcome_revenue", "seasonality_config"], section="outcome"))
    return issues


def _check_seasonality(
    sea: Any,
    *,
    path_prefix: List[PathToken],
    section: str,
) -> List[Dict[str, Any]]:
    if not isinstance(sea, dict) or not sea:
        return []
    issues: List[Dict[str, Any]] = []
    st = str(sea.get("type", "")).strip().lower()
    if st == "sin":
        amp = sea.get("amplitude")
        if amp is not None and (not _is_finite_number(amp) or float(amp) < 0 or float(amp) > 1):
            issues.append(
                _Issue(
                    message="Seasonality amplitude must be in [0, 1]",
                    path=[*path_prefix, "amplitude"],
                    section=section,
                )
            )
        per = sea.get("period")
        if per is not None and (not _is_finite_number(per) or int(per) < 1):
            issues.append(
                _Issue(
                    message="Seasonality period must be a positive integer",
                    path=[*path_prefix, "period"],
                    section=section,
                )
            )
    if st == "fourier" and sea.get("coefficients") is not None:
        period_raw = sea.get("period")
        period = int(period_raw) if _is_finite_number(period_raw) and int(period_raw) >= 1 else None
        coeffs = sea.get("coefficients") or []
        if not isinstance(coeffs, list):
            issues.append(
                _Issue(
                    message="Fourier coefficients must be a list of [a_k, b_k] pairs",
                    path=[*path_prefix, "coefficients"],
                    section=section,
                )
            )
        else:
            if period is not None and len(coeffs) > max(1, period // 2):
                issues.append(
                    _Issue(
                        message=(
                            f"Fourier K = {len(coeffs)} exceeds the Nyquist limit ⌊period/2⌋ = "
                            f"{max(1, period // 2)}"
                        ),
                        path=[*path_prefix, "coefficients"],
                        section=section,
                        severity="warning",
                    )
                )
            for idx, pair in enumerate(coeffs):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    issues.append(
                        _Issue(
                            message=f"Fourier coefficient k={idx + 1} must be [a_k, b_k]",
                            path=[*path_prefix, "coefficients", idx],
                            section=section,
                        )
                    )
                    continue
                if not _is_finite_number(pair[0]) or not _is_finite_number(pair[1]):
                    issues.append(
                        _Issue(
                            message=f"Fourier coefficient k={idx + 1} has non-numeric values",
                            path=[*path_prefix, "coefficients", idx],
                            section=section,
                        )
                    )
    return issues


def _check_channels(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    raw = config.get("channel_list")
    if raw is None:
        issues.append(
            _Issue(
                message="At least one channel is required to run the simulator",
                path=["channel_list"],
                section="channels",
            )
        )
        return issues
    if not isinstance(raw, list) or len(raw) == 0:
        issues.append(
            _Issue(
                message="At least one channel is required to run the simulator",
                path=["channel_list"],
                section="channels",
            )
        )
        return issues

    seen_names: Dict[str, int] = {}
    for idx, item in enumerate(raw):
        ch = (item or {}).get("channel") if isinstance(item, dict) else None
        if not isinstance(ch, dict):
            issues.append(
                _Issue(
                    message=f"Channel #{idx + 1} is malformed",
                    path=["channel_list", idx],
                    section="channels",
                )
            )
            continue

        name = ch.get("channel_name")
        if not isinstance(name, str) or not name.strip():
            issues.append(
                _Issue(
                    message="Channel name is required",
                    path=["channel_list", idx, "channel", "channel_name"],
                    section="channels",
                )
            )
        else:
            seen_names.setdefault(name.strip(), 0)
            seen_names[name.strip()] += 1

        roi = ch.get("true_roi")
        if roi is not None and (not _is_finite_number(roi) or float(roi) < 0):
            issues.append(
                _Issue(
                    message="True ROI must be a non-negative finite number",
                    path=["channel_list", idx, "channel", "true_roi"],
                    section="channels",
                )
            )

        cpm = ch.get("cpm")
        if cpm is not None and (not _is_finite_number(cpm) or float(cpm) <= 0):
            issues.append(
                _Issue(
                    message="CPM must be a positive finite number",
                    path=["channel_list", idx, "channel", "cpm"],
                    section="channels",
                )
            )

        spend = ch.get("spend_range")
        if spend is not None:
            if (
                not isinstance(spend, (list, tuple))
                or len(spend) != 2
                or not _is_finite_number(spend[0])
                or not _is_finite_number(spend[1])
            ):
                issues.append(
                    _Issue(
                        message="Spend range must be [min, max] of finite numbers",
                        path=["channel_list", idx, "channel", "spend_range"],
                        section="channels",
                    )
                )
            else:
                lo, hi = float(spend[0]), float(spend[1])
                if lo < 0 or hi < 0:
                    issues.append(
                        _Issue(
                            message="Spend range must be non-negative",
                            path=["channel_list", idx, "channel", "spend_range"],
                            section="channels",
                        )
                    )
                if lo > hi:
                    issues.append(
                        _Issue(
                            message="Spend range min must be ≤ max",
                            path=["channel_list", idx, "channel", "spend_range"],
                            section="channels",
                        )
                    )

        sea = ch.get("seasonality_config")
        issues.extend(
            _check_seasonality(
                sea,
                path_prefix=["channel_list", idx, "channel", "seasonality_config"],
                section="channels",
            )
        )

    for name, count in seen_names.items():
        if count > 1:
            # Find all offending indexes for the path list. We surface a single
            # issue with no specific index so the error banner is readable, and
            # leave the per-row dot to the structural check (duplicates trip
            # correlation resolution downstream).
            issues.append(
                _Issue(
                    message=f"Duplicate channel name '{name}'. Names must be unique.",
                    path=["channel_list"],
                    section="channels",
                )
            )
    return issues


def _check_correlations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    corrs = config.get("correlations")
    if corrs is None:
        return issues
    if not isinstance(corrs, list):
        issues.append(
            _Issue(
                message="correlations must be a list",
                path=["correlations"],
                section="correlations",
            )
        )
        return issues
    valid_names: set[str] = set()
    for item in config.get("channel_list") or []:
        ch = (item or {}).get("channel") if isinstance(item, dict) else None
        if isinstance(ch, dict):
            nm = ch.get("channel_name")
            if isinstance(nm, str) and nm.strip():
                valid_names.add(nm.strip())

    for idx, entry in enumerate(corrs):
        if not isinstance(entry, dict):
            issues.append(
                _Issue(
                    message=f"correlations[{idx}] must be a mapping",
                    path=["correlations", idx],
                    section="correlations",
                )
            )
            continue
        pair = entry.get("channels") or []
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            issues.append(
                _Issue(
                    message="Each correlation must reference exactly 2 channels",
                    path=["correlations", idx, "channels"],
                    section="correlations",
                )
            )
        else:
            for a in pair:
                if not isinstance(a, str) or not a.strip():
                    issues.append(
                        _Issue(
                            message="Correlation channel names must be non-empty strings",
                            path=["correlations", idx, "channels"],
                            section="correlations",
                        )
                    )
                elif valid_names and a not in valid_names:
                    issues.append(
                        _Issue(
                            message=f"Correlation references unknown channel '{a}'",
                            path=["correlations", idx, "channels"],
                            section="correlations",
                        )
                    )
        rho = entry.get("rho")
        if rho is not None and (not _is_finite_number(rho) or float(rho) < -1.0 or float(rho) > 1.0):
            issues.append(
                _Issue(
                    message="Correlation rho must be in [-1, 1]",
                    path=["correlations", idx, "rho"],
                    section="correlations",
                )
            )
    return issues


def _check_budget_shifts(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    shifts = config.get("budget_shifts")
    if shifts is None:
        return issues
    if not isinstance(shifts, list):
        issues.append(
            _Issue(
                message="budget_shifts must be a list",
                path=["budget_shifts"],
                section="budget_shifts",
            )
        )
        return issues
    for idx, item in enumerate(shifts):
        if not isinstance(item, dict):
            issues.append(
                _Issue(
                    message=f"budget_shifts[{idx}] must be a mapping",
                    path=["budget_shifts", idx],
                    section="budget_shifts",
                )
            )
            continue
        t = str(item.get("type", "")).strip().lower()
        if t not in {"scale", "scale_channel", "reallocate"}:
            issues.append(
                _Issue(
                    message=f"budget_shifts[{idx}].type must be one of scale / scale_channel / reallocate",
                    path=["budget_shifts", idx, "type"],
                    section="budget_shifts",
                )
            )
            continue
        start = item.get("start_week")
        end = item.get("end_week")
        if start is not None and (not _is_finite_number(start) or int(start) < 0):
            issues.append(
                _Issue(
                    message="budget_shifts start_week must be a non-negative integer",
                    path=["budget_shifts", idx, "start_week"],
                    section="budget_shifts",
                )
            )
        if end is not None and (not _is_finite_number(end) or int(end) < 0):
            issues.append(
                _Issue(
                    message="budget_shifts end_week must be a non-negative integer",
                    path=["budget_shifts", idx, "end_week"],
                    section="budget_shifts",
                )
            )
        if _is_finite_number(start) and _is_finite_number(end) and int(end) < int(start):
            issues.append(
                _Issue(
                    message="budget_shifts end_week must be ≥ start_week",
                    path=["budget_shifts", idx, "end_week"],
                    section="budget_shifts",
                )
            )
        if t in {"scale", "scale_channel"}:
            f = item.get("factor")
            if f is not None and (not _is_finite_number(f) or float(f) < 0):
                issues.append(
                    _Issue(
                        message="budget_shifts factor must be a non-negative finite number",
                        path=["budget_shifts", idx, "factor"],
                        section="budget_shifts",
                    )
                )
        if t == "reallocate":
            fr = item.get("fraction")
            if fr is not None and (not _is_finite_number(fr) or float(fr) < 0 or float(fr) > 1):
                issues.append(
                    _Issue(
                        message="budget_shifts reallocate fraction must be in [0, 1]",
                        path=["budget_shifts", idx, "fraction"],
                        section="budget_shifts",
                    )
                )
    return issues


def _check_top_level(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    wr = config.get("week_range")
    if wr is not None and (not _is_finite_number(wr) or int(wr) < 1):
        issues.append(
            _Issue(
                message="week_range must be a positive integer",
                path=["week_range"],
                section="general",
            )
        )
    n = config.get("number_of_channels")
    if n is not None and (not _is_finite_number(n) or int(n) < 0):
        issues.append(
            _Issue(
                message="number_of_channels must be a non-negative integer",
                path=["number_of_channels"],
                section="general",
            )
        )
    order = config.get("media_transform_order")
    if order is not None and order not in {"adstock_first", "saturation_first"}:
        issues.append(
            _Issue(
                message="media_transform_order must be 'adstock_first' or 'saturation_first'",
                path=["media_transform_order"],
                section="general",
            )
        )
    return issues


def _collect_structural_issues(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    issues.extend(_check_top_level(config))
    issues.extend(_check_outcome(config))
    issues.extend(_check_channels(config))
    issues.extend(_check_correlations(config))
    issues.extend(_check_budget_shifts(config))
    return issues


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``{"ok": bool, "issues": [...]}`` describing problems with the config."""
    if not isinstance(config, dict):
        return {
            "ok": False,
            "issues": [
                dict(_Issue(message="Config payload must be a mapping", section="general"))
            ],
        }
    issues = _collect_structural_issues(config)

    # Backstop: if structural checks pass, also try the real loader so that
    # any deeper validation it does (PSD correlations, schema typing, etc.)
    # surfaces as a single top-level error.
    has_error = any(i["severity"] == "error" for i in issues)
    if not has_error:
        try:
            load_config_from_dict(copy.deepcopy(config))
        except Exception as exc:  # noqa: BLE001
            issues.append(
                dict(
                    _Issue(
                        message=str(exc),
                        section="general",
                        severity="error",
                    )
                )
            )

    ok = not any(i["severity"] == "error" for i in issues)
    return {"ok": ok, "issues": issues}
