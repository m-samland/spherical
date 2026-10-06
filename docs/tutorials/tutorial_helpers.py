"""Helpers shared by the tutorial notebooks, run scripts and docs tools (stdlib only)."""

from __future__ import annotations

import os
from pathlib import Path

ENV_TUTORIAL_DIR = "SPHERICAL_TUTORIAL_DIR"


def tutorial_dir() -> Path:
    """Directory holding the tutorial runs; `$SPHERICAL_TUTORIAL_DIR`, read at call time."""
    value = os.environ.get(ENV_TUTORIAL_DIR)
    return Path(value).expanduser() if value else Path.home() / "data" / "sphere_tutorials"
