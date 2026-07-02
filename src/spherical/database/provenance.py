"""Provenance for spherical database tables.

Writes a visible ``database_provenance.json`` beside the tables and embeds a
compact subset into each table's astropy ``.meta`` (serialised to FITS headers).
Query dates stand in for Gaia/MOCA versioning; the Gaia data release is a
hardcoded literal.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

GAIA_DATA_RELEASE = "GaiaDR3"
PROVENANCE_FILENAME = "database_provenance.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spherical_version() -> str:
    try:
        from spherical import __version__
        return str(__version__)
    except Exception:
        return "unknown"


@dataclass
class TableProvenance:
    instrument: str
    mode: str
    source: str = "eso-extend"
    spherical_version: str = ""
    generated_utc: str = ""
    eso_query_utc: str | None = None
    eso_coverage_start: str | None = None
    eso_coverage_end: str | None = None
    gaia_data_release: str = GAIA_DATA_RELEASE
    gaia_query_utc: str | None = None
    moca_query_utc: str | None = None
    enrichment: dict = field(default_factory=lambda: {"gaia": "skipped", "moca": "skipped"})
    build_parameters: dict = field(default_factory=dict)
    zenodo: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TableProvenance":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def provenance_path(dest) -> Path:
    return Path(dest) / PROVENANCE_FILENAME


def read_provenance(dest) -> dict[str, TableProvenance]:
    path = provenance_path(dest)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    tables = raw.get("tables", {})
    return {mode: TableProvenance.from_dict(rec) for mode, rec in tables.items()}


def write_provenance(dest, records: dict[str, TableProvenance]) -> None:
    """Merge ``records`` (mode -> TableProvenance) into the provenance file."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    existing = {}
    path = provenance_path(dest)
    if path.exists():
        try:
            existing = json.loads(path.read_text()).get("tables", {})
        except (json.JSONDecodeError, OSError):
            existing = {}
    for mode, rec in records.items():
        existing[mode] = rec.to_dict()
    payload = {
        "spherical_version": spherical_version(),
        "generated_utc": now_utc(),
        "tables": existing,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
