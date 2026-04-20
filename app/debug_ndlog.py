"""NDJSON debug logging for Cursor debug sessions (remove after investigation)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

_LOG = Path(__file__).resolve().parents[1] / "debug-ba89b7.log"


def agent_dbg(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "sessionId": "ba89b7",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with _LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
