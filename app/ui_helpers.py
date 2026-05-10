"""Path helpers and merging UI widget values into config dicts."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Union

PathPart = Union[str, int]


def path_string_to_parts(path: str) -> List[PathPart]:
    """Parse 'channel_list.0.channel.true_roi' into ['channel_list', 0, 'channel', 'true_roi']."""
    parts: List[PathPart] = []
    for segment in path.split("."):
        segment = segment.strip()
        if not segment:
            continue
        if re.fullmatch(r"\d+", segment):
            parts.append(int(segment))
        else:
            parts.append(segment)
    return parts


def get_at(data: Any, parts: List[PathPart]) -> Any:
    cur = data
    for key in parts:
        if cur is None:
            return None
        if isinstance(key, int):
            if not isinstance(cur, list) or key >= len(cur):
                return None
            cur = cur[key]
        else:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
    return cur


def set_at(data: Any, parts: List[PathPart], value: Any) -> None:
    """Mutate nested dict/list in place."""
    if not parts:
        return
    if len(parts) == 1:
        last = parts[0]
        if isinstance(last, int):
            if not isinstance(data, list):
                raise TypeError("expected list at leaf")
            while len(data) <= last:
                data.append(None)
            data[last] = value
        else:
            if not isinstance(data, dict):
                raise TypeError("expected dict at leaf")
            data[last] = value
        return
    key, *rest = parts
    if isinstance(key, int):
        if not isinstance(data, list):
            raise TypeError("expected list")
        while len(data) <= key:
            data.append({})
        set_at(data[key], rest, value)
    else:
        if not isinstance(data, dict):
            raise TypeError("expected dict")
        if key not in data or data[key] is None:
            data[key] = [] if isinstance(rest[0], int) else {}
        set_at(data[key], rest, value)


def apply_overrides(
    base: Dict[str, Any], overrides: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply {path, value, list_index?} entries onto a deep copy of base."""
    out = copy.deepcopy(base)
    for ov in overrides:
        path = ov["path"]
        value = ov["value"]
        list_index: Optional[int] = ov.get("list_index")
        parts = path_string_to_parts(path)
        if list_index is not None:
            cur = get_at(out, parts)
            if not isinstance(cur, list):
                set_at(out, parts, [])
                cur = get_at(out, parts)
            if not isinstance(cur, list):
                continue
            while len(cur) <= list_index:
                cur.append(0.0)
            cur[list_index] = value
        else:
            set_at(out, parts, value)
    return out
