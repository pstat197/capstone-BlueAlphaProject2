"""Legacy convenience wrapper for the full pytest suite.

Preferred usage:
    pytest tests/ -q
"""
from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()