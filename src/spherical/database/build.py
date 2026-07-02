"""Orchestration for building and updating spherical database tables.

Thin, testable functions used by the ``spherical-sync-tables`` and
``spherical-update-database`` CLIs. Heavy network calls (ESO/SIMBAD/Gaia/MOCA)
are delegated to the existing database modules.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from astropy.table import Table

from spherical.database import file_table, observation_table, target_table
from spherical.database.database_utils import resolve_mode_name
from spherical.database.gaia_astrophysical_params import query_gaia_astrophysical_params
from spherical.database.mocadb_matching import query_mocadb_for_targets
from spherical.database import provenance as prov

logger = logging.getLogger("spherical.build")


def _max_night_start(table_of_files) -> str | None:
    if "NIGHT_START" not in table_of_files.colnames:
        return None
    values = [str(v) for v in np.asarray(table_of_files["NIGHT_START"])
              if str(v) not in ("", "INVALID_DATE", "--")]
    return max(values) if values else None


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

    coverage_end = _max_night_start(table_of_files)
    record = prov.TableProvenance(
        instrument=instrument.lower(),
        mode=mode,
        source=source,
        spherical_version=prov.spherical_version(),
        generated_utc=prov.now_utc(),
        eso_query_utc=prov.now_utc(),
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
