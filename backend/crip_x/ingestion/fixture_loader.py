"""
CRIP-X Fixture Loader

Loads JSON test fixtures for development, API testing,
and frontend demo mode.

In production this is replaced by the real wfdb loader
pulling from PhysioNet. The interface is identical -
swap the loader, everything else stays the same.
"""

import json
from pathlib import Path
from typing import Optional
import numpy as np

from crip_x.utils.config import ROOT_DIR
from crip_x.utils.logger import get_logger

logger = get_logger(__name__)

FIXTURES_DIR = ROOT_DIR / "data" / "fixtures"


def load_fixture(fixture_filename: str) -> dict:
    """Load a fixture file by filename."""
    path = FIXTURES_DIR / fixture_filename
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")

    with open(path, "r") as f:
        fixture = json.load(f)

    logger.debug(f"Loaded fixture: {fixture_filename}")
    return fixture


def get_signal_array(fixture: dict) -> np.ndarray:
    """
    Extract signal as numpy array.
    Handles null values (JSON null → np.nan).
    """
    raw = fixture["signal"]
    signal = np.array(
        [np.nan if v is None else float(v) for v in raw],
        dtype=np.float64
    )
    return signal


def get_neighboring_arrays(fixture: dict) -> dict:
    """Extract neighboring signal arrays."""
    neighbors = {}
    for sig_name, values in fixture.get("neighboring_signals", {}).items():
        neighbors[sig_name] = np.array(
            [np.nan if v is None else float(v) for v in values],
            dtype=np.float64
        )
    return neighbors


def get_motion_array(fixture: dict) -> Optional[np.ndarray]:
    """Extract motion signal if present."""
    if "motion_signal" not in fixture:
        return None
    return np.array(fixture["motion_signal"], dtype=np.float64)


def list_fixtures() -> list[str]:
    """List all available fixture files."""
    if not FIXTURES_DIR.exists():
        return []
    return [f.name for f in FIXTURES_DIR.glob("*.json")]