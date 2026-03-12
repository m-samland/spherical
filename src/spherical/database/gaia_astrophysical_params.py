#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gaia DR3 astrophysical parameter enrichment for spherical target tables.

This module queries the ESA Gaia DR3 archive to enrich a spherical target
table with GSP-Phot astrophysical parameters: effective temperature,
surface gravity, metallicity, and extinction.

The cross-match is performed via Gaia DR3 ``source_id`` using a TAP
upload-join against ``gaiadr3.astrophysical_parameters``.  This is a single
ADQL query regardless of list size.

References
----------
- Andrae et al. (2023), A&A, 674, A27 (GSP-Phot)
- Gaia DR3 documentation: https://gea.esac.esa.int/archive/documentation/GDR3/
"""

from __future__ import annotations

import logging
import re
import warnings

import numpy as np
from astropy.table import Column, Table

logger = logging.getLogger(__name__)

__all__ = ["query_gaia_astrophysical_params"]

# ---------------------------------------------------------------------------
# Gaia TAP endpoint
# ---------------------------------------------------------------------------
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

# ---------------------------------------------------------------------------
# ADQL query template
# ---------------------------------------------------------------------------
_ADQL_QUERY = """
SELECT u.source_id,
       ap.teff_gspphot, ap.teff_gspphot_lower, ap.teff_gspphot_upper,
       ap.logg_gspphot, ap.logg_gspphot_lower, ap.logg_gspphot_upper,
       ap.mh_gspphot,   ap.mh_gspphot_lower,   ap.mh_gspphot_upper,
       ap.ag_gspphot,   ap.ag_gspphot_lower,    ap.ag_gspphot_upper
FROM TAP_UPLOAD.targets AS u
JOIN gaiadr3.astrophysical_parameters AS ap
    ON ap.source_id = u.source_id
"""

# ---------------------------------------------------------------------------
# Output column definitions (all float)
# ---------------------------------------------------------------------------
_GAIA_COLUMNS = {
    "GAIA_TEFF":       "teff_gspphot",
    "GAIA_TEFF_LOWER": "teff_gspphot_lower",
    "GAIA_TEFF_UPPER": "teff_gspphot_upper",
    "GAIA_LOGG":       "logg_gspphot",
    "GAIA_LOGG_LOWER": "logg_gspphot_lower",
    "GAIA_LOGG_UPPER": "logg_gspphot_upper",
    "GAIA_MH":         "mh_gspphot",
    "GAIA_MH_LOWER":   "mh_gspphot_lower",
    "GAIA_MH_UPPER":   "mh_gspphot_upper",
    "GAIA_AG":         "ag_gspphot",
    "GAIA_AG_LOWER":   "ag_gspphot_lower",
    "GAIA_AG_UPPER":   "ag_gspphot_upper",
}

# ---------------------------------------------------------------------------
# Helper: parse Gaia DR3 IDs
# ---------------------------------------------------------------------------

def _clean_gaia_id(value) -> int | None:
    """Return an integer Gaia DR3 source_id, or ``None`` if not parseable."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    s = str(value).strip()
    if s.lower() in ("", "nan", "none", "--"):
        return None
    m = re.search(r"(?:Gaia\s*DR3\s*)?(\d{5,})$", s, re.I)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def query_gaia_astrophysical_params(
    target_table: Table,
    *,
    gaia_id_column: str = "ID_GAIA_DR3",
    timeout: int = 120,
    debug: bool = False,
) -> Table:
    """Enrich a target table with Gaia DR3 GSP-Phot astrophysical parameters.

    Queries the ESA Gaia TAP archive for effective temperature, surface
    gravity, metallicity, and G-band extinction from GSP-Phot, using the
    existing Gaia DR3 source IDs in the target table.

    New columns are prefixed with ``GAIA_`` to avoid collisions.

    Parameters
    ----------
    target_table : `~astropy.table.Table`
        Target table with a column containing Gaia DR3 identifiers.
    gaia_id_column : str, optional
        Name of the column containing Gaia DR3 identifiers (default:
        ``"ID_GAIA_DR3"``).
    timeout : int, optional
        TAP query timeout in seconds (default: 120).
    debug : bool, optional
        If *True*, log the ADQL query at DEBUG level.

    Returns
    -------
    enriched_table : `~astropy.table.Table`
        A copy of the input table with 12 ``GAIA_*`` columns appended.
        Targets not found in the Gaia archive have ``NaN`` values.
    """
    try:
        from astroquery.utils.tap import TapPlus
    except ImportError:
        logger.warning(
            "astroquery is not installed – skipping Gaia enrichment."
        )
        return _attach_empty_columns(target_table)

    if gaia_id_column not in target_table.colnames:
        raise ValueError(
            f"Column '{gaia_id_column}' not found in target table. "
            f"Available columns: {target_table.colnames}"
        )

    # ----- Parse Gaia IDs -----
    raw_ids = list(target_table[gaia_id_column])
    gaia_ids = [_clean_gaia_id(v) for v in raw_ids]
    valid_ids = [gid for gid in gaia_ids if gid is not None]

    if not valid_ids:
        logger.warning(
            "Gaia: No valid Gaia DR3 IDs found in column '%s' "
            "(checked %d rows) – returning table with empty GAIA columns.",
            gaia_id_column, len(raw_ids),
        )
        return _attach_empty_columns(target_table)

    n_total = len(gaia_ids)
    n_valid = len(valid_ids)
    n_invalid = n_total - n_valid
    logger.info("Gaia: %d/%d targets have valid Gaia DR3 IDs.", n_valid, n_total)
    if n_invalid > 0:
        logger.info(
            "Gaia: %d targets have missing/unparseable Gaia DR3 IDs and will be skipped.",
            n_invalid,
        )

    # ----- Build upload table -----
    unique_ids = list(set(valid_ids))
    upload_table = Table({"source_id": unique_ids})

    if debug:
        logger.debug("Gaia ADQL query:\n%s", _ADQL_QUERY.strip())
        logger.debug("Gaia: Uploading %d unique source IDs.", len(unique_ids))

    # ----- Execute TAP query -----
    try:
        tap = TapPlus(url=GAIA_TAP_URL)
        job = tap.launch_job_async(
            _ADQL_QUERY,
            upload_resource=upload_table,
            upload_table_name="targets",
        )
        result_table = job.get_results()
    except Exception as e:
        logger.warning(
            "Gaia: TAP query failed (%s). Returning table with empty GAIA columns.", e
        )
        return _attach_empty_columns(target_table)

    logger.info("Gaia: Query returned %d rows for %d unique IDs.", len(result_table), len(unique_ids))

    if len(result_table) == 0:
        logger.warning(
            "Gaia: No matches found for %d valid source IDs. "
            "Returning table with empty GAIA columns.",
            len(unique_ids),
        )
        return _attach_empty_columns(target_table)

    # ----- Merge results -----
    return _merge_results(target_table, gaia_ids, result_table)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_gaia_ap_columns(table: Table) -> Table:
    """Remove any existing ``GAIA_`` columns so the table can be re-enriched."""
    existing = [c for c in table.colnames if c.startswith("GAIA_")]
    if existing:
        logger.debug("Gaia: Removing %d existing GAIA_ columns before re-enrichment.", len(existing))
        table.remove_columns(existing)
    return table


def _merge_results(
    target_table: Table,
    gaia_ids: list[int | None],
    result_table: Table,
) -> Table:
    """Map Gaia query results back to the original target table order."""
    result = target_table.copy()
    _strip_gaia_ap_columns(result)

    # Build source_id → row lookup (keep first if duplicates)
    row_map: dict[int, dict] = {}
    for row in result_table:
        sid = int(row["source_id"])
        if sid not in row_map:
            row_map[sid] = row

    n = len(result)
    n_matched = 0

    for col_name, src_key in _GAIA_COLUMNS.items():
        values = []
        for gid in gaia_ids:
            if gid is not None and gid in row_map:
                val = row_map[gid][src_key]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        fval = float(val)
                except (TypeError, ValueError):
                    fval = np.nan
                values.append(fval)
            else:
                values.append(np.nan)
        result.add_column(Column(values, name=col_name))

    # Count matches (based on TEFF being non-NaN)
    n_matched = sum(
        1 for gid in gaia_ids
        if gid is not None and gid in row_map
    )
    logger.info("Gaia: %d/%d targets matched in Gaia astrophysical_parameters.", n_matched, n)

    return result


def _attach_empty_columns(target_table: Table) -> Table:
    """Return a copy with all GAIA_ columns filled with NaN."""
    result = target_table.copy()
    _strip_gaia_ap_columns(result)
    n = len(result)

    for col_name in _GAIA_COLUMNS:
        result.add_column(Column([np.nan] * n, name=col_name))

    return result
