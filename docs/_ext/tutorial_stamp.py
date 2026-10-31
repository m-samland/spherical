"""The provenance line of a tutorial notebook, from its metadata (stdlib only).

`docs/tools/stamp_notebook.py` writes `metadata["spherical"] = {"run": ..., "rendered": ...}`.
"""

from __future__ import annotations

METADATA_KEY = "spherical"
# Same tuple as stamp_notebook.PROVENANCE_KEYS; a test keeps them equal.
REQUIRED_KEYS = ("spherical", "charis", "trap", "date", "cpu", "cores", "ram_gb", "os", "ncpu")


def format_stamp(meta: dict | None) -> str:
    """One sentence naming versions, run date, machine and, if later, the render date."""
    if not meta or "run" not in meta or "rendered" not in meta:
        raise ValueError("notebook has no stamp; run `pixi run -e dev docs-tutorials <name>`")
    run = meta["run"]
    missing = [key for key in REQUIRED_KEYS if key not in run]
    if missing:
        raise ValueError(f"notebook stamp lacks {', '.join(missing)}")
    versions = f"spherical {run['spherical']} (charis {run['charis']}, TRAP {run['trap']})"
    machine = f"on {run['cpu']} with {run['cores']} cores"
    if run["ram_gb"] is not None:
        machine += f" and {run['ram_gb']:.0f} GB RAM"
    if run["ncpu"] is not None:
        machine += f", set_ncpu({run['ncpu']})"
    if run["date"] == meta["rendered"]:
        return f"Verified with {versions} on {run['date']}, {machine}."
    return f"Reduced with {versions} on {run['date']}, {machine}. Figures rendered on {meta['rendered']}."
