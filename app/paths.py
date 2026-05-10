"""Repository and app paths shared by the Streamlit UI."""

from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = _APP_DIR.parent
EXAMPLE_YAML_PATH = REPO_ROOT / "example.yaml"
UI_SCHEMA_PATH = _APP_DIR / "ui_schema.yaml"
