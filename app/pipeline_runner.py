"""Parse YAML config and run the core simulation (no Streamlit imports)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
import yaml

from scripts.config.loader import load_config_from_dict
from scripts.main import run_simulation


def parse_yaml_text(yaml_text: str) -> Dict[str, Any]:
    """Parse user YAML string into a dict; empty string becomes {}."""
    if not yaml_text or not yaml_text.strip():
        return {}
    data = yaml.safe_load(yaml_text)
    return data if isinstance(data, dict) else {}


def run_pipeline(user_data: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """
    Load config from merged user dict and return (DataFrame, run_identifier).
    """
    config = load_config_from_dict(user_data)
    df = run_simulation(config)
    return df, config.get_run_identifier()
