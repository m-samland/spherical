"""Get an up-to-date SPHERE database: download the Zenodo baseline, then update.

Two steps, both also available as console commands:
  1. spherical-sync-tables --dest <dir>            (download latest Zenodo tables)
  2. spherical-update-database --dest <dir>         (extend to today + rebuild)

This script shows the equivalent library calls. `update_in_place` no longer
exists: update mode always writes the canonical Zenodo-style filenames; pass a
`suffix` only for experimental/scratch builds. A build from scratch is just an
update with an explicit early start date and no baseline tables present.
"""
from pathlib import Path

from spherical.database import build

# Where the database tables live.
table_path = Path.home() / "data/sphere/database"
table_path.mkdir(parents=True, exist_ok=True)

# Step 1 (optional): download the latest pre-built tables from Zenodo.
# Skip this if you already have up-to-date files in `table_path`.
download_from_zenodo = True
if download_from_zenodo:
    build.sync_from_zenodo(table_path, instrument="all", include_polarimetry=True)

# Step 2: bring the tables up to date with the current ESO archive.
# start_date/end_date default to (provenance resume date) .. today.
for instrument in ("ifs", "irdis"):
    build.update_database(
        table_path,
        instrument,
        # start_date="2014-05-01",  # uncomment for a full build from scratch
        overlap_days=7,
        skip_sam=False,
        enrich=True,
    )

print(f"Done. Up-to-date tables and database_provenance.json in {table_path}")
