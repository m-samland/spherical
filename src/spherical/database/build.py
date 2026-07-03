"""Orchestration for building and updating spherical database tables.

Thin, testable functions used by the ``spherical-sync-tables`` and
``spherical-update-database`` CLIs. Heavy network calls (ESO/SIMBAD/Gaia/MOCA)
are delegated to the existing database modules.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np
from astropy.table import Table

from spherical.database import file_table, observation_table, target_table
from spherical.database import provenance as prov
from spherical.database.database_utils import resolve_mode_name
from spherical.database.gaia_astrophysical_params import query_gaia_astrophysical_params
from spherical.database.mocadb_matching import query_mocadb_for_targets

logger = logging.getLogger("spherical.build")


def _valid_night_starts(table_of_files) -> list[str]:
    if "NIGHT_START" not in table_of_files.colnames:
        return []
    return [str(v) for v in np.asarray(table_of_files["NIGHT_START"]) if str(v) not in ("", "INVALID_DATE", "--")]


def _max_night_start(table_of_files) -> str | None:
    values = _valid_night_starts(table_of_files)
    return max(values) if values else None


def _min_night_start(table_of_files) -> str | None:
    values = _valid_night_starts(table_of_files)
    return min(values) if values else None


def _enrich(target_tbl):
    """Run MOCA then Gaia enrichment. Never raises. Returns
    (table, status_dict, gaia_utc, moca_utc)."""
    status = {"gaia": "skipped", "moca": "skipped"}
    gaia_utc = moca_utc = None
    try:
        target_tbl = query_mocadb_for_targets(target_tbl, include_tier2=True)
        status["moca"] = "ok"
        moca_utc = prov.now_utc()
    except Exception as exc:
        logger.warning("MOCAdb enrichment failed: %s", exc)
        status["moca"] = "failed"
    try:
        target_tbl = query_gaia_astrophysical_params(target_tbl)
        status["gaia"] = "ok"
        gaia_utc = prov.now_utc()
    except Exception as exc:
        logger.warning("Gaia enrichment failed: %s", exc)
        status["gaia"] = "failed"
    return target_tbl, status, gaia_utc, moca_utc


def build_tables(
    dest,
    instrument,
    table_of_files,
    *,
    polarimetry=False,
    sparse_aperture_masking=False,
    parallax_limit=1e-3,
    J_mag_limit=14.0,
    search_radius=3.0,
    cone_size_science=15.0,
    enrich=True,
    suffix="",
    source="eso-extend",
):
    """Build target + observation tables for one mode and write them with
    embedded provenance. Returns the ``TableProvenance`` record."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    mode = resolve_mode_name(instrument, polarimetry, sparse_aperture_masking)

    target_tbl, _not_found = target_table.make_target_list_with_SIMBAD(
        table_of_files=table_of_files,
        instrument=instrument,
        polarimetry=polarimetry,
        sparse_aperture_masking=sparse_aperture_masking,
        remove_fillers=False,
        parallax_limit=parallax_limit,
        J_mag_limit=J_mag_limit,
        search_radius=search_radius,
        batch_size=250,
        min_delay=1.0,
        group_by_healpix=True,
    )

    status = {"gaia": "skipped", "moca": "skipped"}
    gaia_utc = moca_utc = None
    if enrich:
        target_tbl, status, gaia_utc, moca_utc = _enrich(target_tbl)

    obs_tbl, target_tbl = observation_table.create_observation_table(
        table_of_files=table_of_files,
        table_of_targets=target_tbl,
        instrument=instrument,
        polarimetry=polarimetry,
        sparse_aperture_masking=sparse_aperture_masking,
        cone_size_science=cone_size_science,
        remove_fillers=False,
        reorder_columns=True,
    )

    # Coverage reflects the actual data span of the file table (earliest and
    # latest observing night), not the incremental resume window — so an
    # extended baseline honestly reports its full 2014-onward coverage.
    coverage_start = _min_night_start(table_of_files)
    coverage_end = _max_night_start(table_of_files)
    record = prov.TableProvenance(
        instrument=instrument.lower(),
        mode=mode,
        source=source,
        spherical_version=prov.spherical_version(),
        generated_utc=prov.now_utc(),
        eso_query_utc=prov.now_utc(),
        eso_coverage_start=coverage_start,
        eso_coverage_end=coverage_end,
        gaia_query_utc=gaia_utc,
        moca_query_utc=moca_utc,
        enrichment=status,
        build_parameters={
            "polarimetry": bool(polarimetry),
            "sparse_aperture_masking": bool(sparse_aperture_masking),
            "parallax_limit": parallax_limit,
            "J_mag_limit": J_mag_limit,
            "search_radius": search_radius,
            "cone_size_science": cone_size_science,
        },
    )

    for tbl in (target_tbl, obs_tbl):
        prov.embed_in_meta(tbl, record)
    target_tbl.write(dest / f"table_of_targets_{mode}{suffix}.fits", format="fits", overwrite=True)
    obs_tbl.write(dest / f"table_of_observations_{mode}{suffix}.fits", format="fits", overwrite=True)
    return record


def enrich_tables(
    dest,
    instrument,
    *,
    polarimetry=False,
    sparse_aperture_masking=False,
    cone_size_science=15.0,
    suffix="",
):
    """Re-run Gaia/MOCA enrichment on existing tables without querying ESO.

    Reads the existing target table and the local file-table CSV, refreshes
    enrichment, rewrites target + observation tables, and updates provenance
    query dates and status for this mode.
    """
    dest = Path(dest)
    mode = resolve_mode_name(instrument, polarimetry, sparse_aperture_masking)
    target_path = dest / f"table_of_targets_{mode}{suffix}.fits"
    files_path = dest / f"table_of_files_{instrument.lower()}{suffix}.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"No target table to enrich: {target_path}")
    if not files_path.exists():
        raise FileNotFoundError(f"No file table found (needed to rebuild observations): {files_path}")

    target_tbl = Table.read(target_path)
    table_of_files = Table.read(files_path, format="csv")

    target_tbl, status, gaia_utc, moca_utc = _enrich(target_tbl)

    obs_tbl, target_tbl = observation_table.create_observation_table(
        table_of_files=table_of_files,
        table_of_targets=target_tbl,
        instrument=instrument,
        polarimetry=polarimetry,
        sparse_aperture_masking=sparse_aperture_masking,
        cone_size_science=cone_size_science,
        remove_fillers=False,
        reorder_columns=True,
    )

    existing = prov.read_provenance(dest).get(mode)
    record = existing or prov.TableProvenance(instrument=instrument.lower(), mode=mode)
    record.spherical_version = prov.spherical_version()
    record.generated_utc = prov.now_utc()
    record.enrichment = status
    record.gaia_query_utc = gaia_utc
    record.moca_query_utc = moca_utc
    # Coverage is a property of the file table (not the enrichment), so refresh
    # it from the data on hand — this also repairs any stale value written by an
    # older build. eso_query_utc is left untouched: no ESO query happens here.
    record.eso_coverage_start = _min_night_start(table_of_files)
    record.eso_coverage_end = _max_night_start(table_of_files)

    for tbl in (target_tbl, obs_tbl):
        prov.embed_in_meta(tbl, record)
    target_tbl.write(target_path, format="fits", overwrite=True)
    obs_tbl.write(dest / f"table_of_observations_{mode}{suffix}.fits", format="fits", overwrite=True)
    prov.write_provenance(dest, {mode: record})
    return record


def derive_start_date(dest, instrument, overlap_days, suffix=""):
    """Return the resume start date (YYYY-MM-DD) for an incremental update.

    Prefers the standard-mode provenance ``eso_coverage_end``; falls back to the
    max NIGHT_START in the local file-table CSV. Backs off by ``overlap_days``.
    Returns None if no baseline exists.
    """
    dest = Path(dest)
    instrument = instrument.lower()
    coverage_end = None
    record = prov.read_provenance(dest).get(instrument)
    if record is not None:
        coverage_end = record.eso_coverage_end
    if coverage_end is None:
        csv_path = dest / f"table_of_files_{instrument}{suffix}.csv"
        if csv_path.exists():
            coverage_end = _max_night_start(Table.read(csv_path, format="csv"))
    if coverage_end is None:
        return None
    end_dt = datetime.datetime.strptime(coverage_end, "%Y-%m-%d")
    return (end_dt - datetime.timedelta(days=overlap_days)).strftime("%Y-%m-%d")


def modes_for_instrument(instrument, skip_sam=False):
    """Return the list of modes to build for an instrument, as kwargs dicts."""
    instrument = instrument.lower()
    modes = [{"polarimetry": False, "sparse_aperture_masking": False}]
    if instrument == "irdis":
        modes.append({"polarimetry": True, "sparse_aperture_masking": False})
    if not skip_sam:
        modes.append({"polarimetry": False, "sparse_aperture_masking": True})
    return modes


def update_database(
    dest,
    instrument,
    *,
    start_date=None,
    end_date=None,
    overlap_days=7,
    suffix="",
    skip_sam=False,
    enrich=True,
    parallax_limit=1e-3,
    J_mag_limit=14.0,
    search_radius=3.0,
    cone_size_science=15.0,
    batch_size=150,
):
    """Extend the file table to today, then rebuild every mode's tables.

    ``start_date`` defaults to the provenance-derived resume date minus
    ``overlap_days``; ``end_date`` defaults to today. Raises ``ValueError`` if
    there is no baseline and no explicit ``start_date``.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    instrument = instrument.lower()

    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = derive_start_date(dest, instrument, overlap_days, suffix)
    if start_date is None:
        raise ValueError(
            "No baseline tables found and no --start-date given. For a build from "
            "scratch, pass an explicit start date (e.g. 2014-05-01)."
        )

    existing_csv = dest / f"table_of_files_{instrument}{suffix}.csv"
    table_of_files = file_table.make_file_table(
        output_dir=dest,
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        output_suffix=suffix,
        existing_table_path=existing_csv if existing_csv.exists() else None,
        batch_size=batch_size,
        cache=False,
    )

    records = {}
    for mode_kw in modes_for_instrument(instrument, skip_sam=skip_sam):
        record = build_tables(
            dest,
            instrument,
            table_of_files,
            polarimetry=mode_kw["polarimetry"],
            sparse_aperture_masking=mode_kw["sparse_aperture_masking"],
            parallax_limit=parallax_limit,
            J_mag_limit=J_mag_limit,
            search_radius=search_radius,
            cone_size_science=cone_size_science,
            enrich=enrich,
            suffix=suffix,
        )
        records[record.mode] = record

    prov.write_provenance(dest, records)
    return records


def sync_from_zenodo(dest, **kwargs):
    """Thin wrapper delegating to the Zenodo sync CLI's ``sync_tables``.

    Provided so the whole workflow is reachable from one module. Keyword
    arguments are forwarded to ``sync_tables`` (doi_or_record, instrument,
    include_polarimetry, timeout, force, dry_run).
    """
    from spherical.scripts.sync_zenodo_tables import DEFAULT_DOI, sync_tables

    kwargs.setdefault("doi_or_record", DEFAULT_DOI)
    kwargs.setdefault("instrument", "all")
    kwargs.setdefault("include_polarimetry", True)
    kwargs.setdefault("timeout", 120)
    kwargs.setdefault("force", False)
    kwargs.setdefault("dry_run", False)
    return sync_tables(dest=Path(dest), **kwargs)
