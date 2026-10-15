"""Provenance for spherical database tables.

Writes a visible ``database_provenance.json`` beside the tables and embeds a
compact subset into each table's astropy ``.meta`` (serialised to FITS headers).
Query dates stand in for Gaia/MOCA versioning; the Gaia data release is a
hardcoded literal.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

GAIA_DATA_RELEASE = "GaiaDR3"
PROVENANCE_FILENAME = "database_provenance.json"

#: Format of the tables this code writes and fully supports. Raise it when published
#: tables need newer code to be used correctly, so older installs warn (#226). Tables
#: written before the stamp existed count as format 1.
TABLE_FORMAT = 2


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spherical_version() -> str:
    try:
        from spherical import __version__

        return str(__version__)
    except ImportError:
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
    # Per-source enrichment health: {"gaia": {"status", "n_total", "n_valid_ids",
    # "n_matched", "frac"}, "moca": {..., "tier2_ok"}}. Legacy tables stored bare
    # status strings ({"gaia": "ok"}); readers tolerate that shape.
    enrichment: dict = field(default_factory=lambda: {"gaia": {"status": "skipped"}, "moca": {"status": "skipped"}})
    build_parameters: dict = field(default_factory=dict)
    zenodo: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TableProvenance":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def provenance_path(dest: "str | Path") -> Path:
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
    try:
        return {mode: TableProvenance.from_dict(rec) for mode, rec in tables.items()}
    except (TypeError, KeyError, AttributeError):
        return {}


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
        "table_format": TABLE_FORMAT,
        "generated_utc": now_utc(),
        "tables": existing,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


# Compact FITS-safe header keywords (<=8 chars, uppercase) -> logical name.
_META_KEYS = {
    "SPHVERS": "spherical_version",
    "SPHSRC": "source",
    "SPHINST": "instrument",
    "SPHMODE": "mode",
    "SPHCOV0": "eso_coverage_start",
    "SPHCOV1": "eso_coverage_end",
    "SPHGAIA": "gaia_data_release",
    "SPHGENUT": "generated_utc",
    "SPHTFMT": "table_format",
}


def embed_in_meta(table, prov: "TableProvenance") -> None:
    """Embed a compact provenance subset into ``table.meta`` (FITS-safe)."""
    values = {
        "SPHVERS": prov.spherical_version,
        "SPHSRC": prov.source,
        "SPHINST": prov.instrument,
        "SPHMODE": prov.mode,
        "SPHCOV0": prov.eso_coverage_start or "",
        "SPHCOV1": prov.eso_coverage_end or "",
        "SPHGAIA": prov.gaia_data_release,
        "SPHGENUT": prov.generated_utc,
        "SPHTFMT": TABLE_FORMAT,
    }
    for key, value in values.items():
        table.meta[key] = value


def extract_from_meta(table) -> dict:
    """Return provenance logical name -> value from ``table.meta`` (case-insensitive)."""
    upper = {str(k).upper(): v for k, v in dict(table.meta).items()}
    return {name: upper.get(key) for key, name in _META_KEYS.items()}


def read_table_format(dest):
    """Table format recorded in ``dest``'s provenance file, or None if it records none."""
    try:
        return json.loads(provenance_path(dest).read_text()).get("table_format")
    except (json.JSONDecodeError, OSError, AttributeError):
        return None


def newer_table_format_message(table_format) -> str | None:
    """Upgrade advice if ``table_format`` is newer than this code supports, else None."""
    try:
        table_format = int(table_format)
    except (TypeError, ValueError):
        return None
    if table_format <= TABLE_FORMAT:
        return None
    return (
        f"These tables have table format {table_format}, but spherical {spherical_version()} "
        f"supports format {TABLE_FORMAT}; upgrade spherical before using them."
    )


def warn_if_newer_table_format(table) -> None:
    """Warn if ``table`` was written for a newer spherical than this one."""
    message = newer_table_format_message(extract_from_meta(table)["table_format"])
    if message:
        warnings.warn(message, UserWarning, stacklevel=3)
