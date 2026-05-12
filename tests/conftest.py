from __future__ import annotations

from pathlib import Path

import pytest

from scripts.config.loader import load_config


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def example_config_path(project_root: Path) -> Path:
    path = project_root / "example.yaml"
    assert path.exists(), f"example.yaml not found at {path}"
    return path


@pytest.fixture
def example_config(example_config_path: Path):
    return load_config(str(example_config_path))
