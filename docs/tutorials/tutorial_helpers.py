"""Helpers shared by the tutorial notebooks, run scripts and docs tools (stdlib only)."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

ENV_TUTORIAL_DIR = "SPHERICAL_TUTORIAL_DIR"


def tutorial_dir() -> Path:
    """Directory holding the tutorial runs; `$SPHERICAL_TUTORIAL_DIR`, read at call time."""
    value = os.environ.get(ENV_TUTORIAL_DIR)
    return Path(value).expanduser() if value else Path.home() / "data" / "sphere_tutorials"


def tidy_path(path) -> str:
    """The path as text with the home directory written as `~`, so outputs stay private."""
    return str(path).replace(str(Path.home()), "~")


def show_log_excerpt(path: Path, head: int = 15, tail: int = 15) -> None:
    """Print the first and last lines of a log file, home directory hidden."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) > head + tail:
        lines = lines[:head] + [f"... {len(lines) - head - tail} lines omitted ..."] + lines[-tail:]
    for line in lines:
        print(tidy_path(line))


def require_run(label: str, needed: list[str], source_hash: str | None = None) -> Path:
    """Return the folder of a tutorial run, or explain which run is missing what.

    `needed` lists paths relative to the run folder that the notebook reads. With
    `source_hash`, warn when the configuration code has changed since the run, so the
    code shown on the page is no longer the code that produced the products.
    """
    run = tutorial_dir() / label
    missing = [name for name in ["run_provenance.json", *needed] if not (run / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"The tutorial run in {tidy_path(run)} is missing {', '.join(missing)}. "
            f"Run docs/tutorials/runs/{label}.py first (see docs/tutorials/runs/README.md).")
    recorded = json.loads((run / "run_provenance.json").read_text(encoding="utf-8")).get("source_hash")
    if source_hash is not None and recorded != source_hash:
        warnings.warn(f"The configuration code has changed since the run in {tidy_path(run)}; "
                      f"rerun docs/tutorials/runs/{label}.py.", UserWarning, stacklevel=2)
    return run
