#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MOCAdb cross-matching for young-association membership and stellar ages.

This module queries the Montreal Open Clusters and Associations database
(MOCAdb; Gagné et al. 2026) to enrich a spherical target table with young
stellar association membership, ages, and ancillary youth indicators.

The cross-match is performed via Gaia DR3 ``source_id`` using a direct
MySQL connection to the public MOCAdb endpoint.  Target IDs are uploaded
as a temporary table and joined server-side for efficiency.

The query is split into three steps to avoid performance issues with
the ``summary_all_members`` VIEW:

1. **Resolve** Gaia DR3 source_id → ``moca_oid`` via ``cat_gaiadr3``
   (indexed temp-table JOIN).
2. **Tier-1** query: ``summary_all_objects`` + association metadata +
   adopted age, queried via ``WHERE moca_oid IN (...)``.
3. **Tier-2** query: ``summary_all_members`` (activity, kinematics),
   also queried via ``WHERE moca_oid IN (...)``.

References
----------
- Gagné et al. (2026), ApJS, arXiv:2602.15695
- MOCAdb website: https://mocadb.ca/
- MOCAdb schema: https://mocadb.ca/en/schema
"""

from __future__ import annotations

import logging
import re
import warnings

import numpy as np
import pandas as pd
from astropy.table import Column, Table

logger = logging.getLogger(__name__)

__all__ = ["query_mocadb_for_targets"]

# ---------------------------------------------------------------------------
# MOCAdb public connection parameters
# ---------------------------------------------------------------------------
MOCADB_HOST = "104.248.106.21"
MOCADB_PORT = 3306
MOCADB_USER = "public"
MOCADB_PASSWORD = "z@nUg_2h7_%?31y88"
MOCADB_DATABASE = "mocadb"


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


def _execute_query(conn, sql: str, debug: bool = False) -> pd.DataFrame:
    """Execute a SQL query via pymysql cursor and return a DataFrame.

    Uses the cursor interface directly (rather than ``pd.read_sql``) to
    avoid spurious SQLAlchemy compatibility warnings.
    """
    if debug:
        preview = sql.strip()[:500]
        logger.debug("SQL query:\n%s", preview)
    cursor = conn.cursor()
    cursor.execute(sql)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# SQL query templates
# ---------------------------------------------------------------------------

# Step 1: Resolve Gaia source_id → moca_oid via the indexed temp table
_SQL_RESOLVE = """
SELECT
    t.gaiadr3_source_id,
    cg.moca_oid
FROM tmp_sphere_targets AS t
JOIN cat_gaiadr3 AS cg ON cg.source_id = t.gaiadr3_source_id
"""

# Step 2 (Tier-1): association membership + adopted age, queried by moca_oid
_SQL_TIER1 = """
SELECT
    sam.moca_oid,
    sam.moca_aid,
    sam.moca_mtid,
    sam.designation          AS moca_designation,
    sam.spectral_type        AS moca_spectral_type,
    sam.banyan_prob,
    sam.banyan_uvw_sep_kms,
    sam.ya_prob,
    ma.name                  AS association_name,
    ma.physical_nature       AS association_type,
    daa.age_myr              AS association_age_myr,
    daa.age_myr_unc          AS association_age_myr_unc,
    daa.age_myr_unc_pos      AS association_age_myr_unc_pos,
    daa.age_myr_unc_neg      AS association_age_myr_unc_neg
FROM summary_all_objects       AS sam
LEFT JOIN moca_associations    AS ma  ON ma.moca_aid   = sam.moca_aid
LEFT JOIN data_association_ages AS daa
    ON daa.moca_aid = sam.moca_aid AND daa.adopted = 1
WHERE sam.moca_oid IN ({oid_list})
"""

# Step 3 (Tier-2): activity, kinematics, rotation, queried by moca_oid.
# summary_all_members is a VIEW (may have several rows per moca_oid, one
# per association membership); we deduplicate in Python.
_SQL_TIER2 = """
SELECT
    samem.moca_oid,
    samem.sptn               AS moca_sptn,
    samem.parallax_mas       AS moca_parallax_mas,
    samem.x_pc               AS moca_x_pc,
    samem.y_pc               AS moca_y_pc,
    samem.z_pc               AS moca_z_pc,
    samem.u_kms              AS moca_u_kms,
    samem.v_kms              AS moca_v_kms,
    samem.w_kms              AS moca_w_kms,
    samem.prot_days,
    samem.gaia_act,
    samem.ewli,
    samem.ewha,
    samem.dr3_ruwe           AS moca_dr3_ruwe
FROM summary_all_members AS samem
WHERE samem.moca_oid IN ({oid_list})
"""


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def query_mocadb_for_targets(
    target_table: Table,
    *,
    gaia_id_column: str = "ID_GAIA_DR3",
    include_tier2: bool = True,
    timeout: int = 30,
    debug: bool = False,
) -> Table:
    """Cross-match a spherical target table with MOCAdb using Gaia DR3 IDs.

    Connects to the public MOCAdb MySQL endpoint and retrieves young-
    association membership, adopted association age, and (optionally) tier-2
    columns such as kinematics, rotation period, and activity indicators.

    New columns are prefixed with ``MOCA_`` to avoid collisions with existing
    SIMBAD-derived columns.

    Parameters
    ----------
    target_table : `~astropy.table.Table`
        Target table produced by
        `spherical.database.target_table.make_target_list_with_SIMBAD`.
        Must contain a column with Gaia DR3 identifiers.
    gaia_id_column : str, optional
        Name of the column containing Gaia DR3 identifiers (default:
        ``"ID_GAIA_DR3"``).  Values may be strings like
        ``"Gaia DR3 12345…"`` or plain integers.
    include_tier2 : bool, optional
        If *True* (default), also retrieve tier-2 columns (activity,
        rotation, kinematics) from ``summary_all_members``.
    timeout : int, optional
        MySQL connection and read timeout in seconds (default: 30).
    debug : bool, optional
        If *True*, print the SQL queries and connection info.

    Returns
    -------
    enriched_table : `~astropy.table.Table`
        A copy of the input table with MOCAdb columns appended.  Targets
        not found in MOCAdb have ``None`` / ``NaN`` in those columns.
    """
    try:
        import pymysql
    except ImportError:
        warnings.warn(
            "pymysql is not installed – skipping MOCAdb enrichment. "
            "Install with:  pip install pymysql",
            stacklevel=2,
        )
        return _attach_empty_columns(target_table, include_tier2)

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
            "MOCAdb: No valid Gaia DR3 IDs found in column '%s' "
            "(checked %d rows) – returning table with empty MOCA columns.",
            gaia_id_column, len(raw_ids),
        )
        return _attach_empty_columns(target_table, include_tier2)

    n_total = len(gaia_ids)
    n_valid = len(valid_ids)
    n_invalid = n_total - n_valid
    logger.info("MOCAdb: %d/%d targets have valid Gaia DR3 IDs.", n_valid, n_total)
    if n_invalid > 0:
        logger.info(
            "MOCAdb: %d targets have missing/unparseable Gaia DR3 IDs and will be skipped.",
            n_invalid,
        )

    # ----- Connect -----
    try:
        conn = pymysql.connect(
            host=MOCADB_HOST,
            port=MOCADB_PORT,
            user=MOCADB_USER,
            password=MOCADB_PASSWORD,
            database=MOCADB_DATABASE,
            connect_timeout=timeout,
            read_timeout=timeout * 4,
            charset="utf8mb4",
        )
    except Exception as e:
        logger.warning(
            "MOCAdb: Could not connect to %s:%s (%s). "
            "Returning table with empty MOCA columns.",
            MOCADB_HOST, MOCADB_PORT, e,
        )
        return _attach_empty_columns(target_table, include_tier2)

    try:
        # -- Step 0: Upload Gaia IDs as indexed temporary table --------
        _upload_temp_table(conn, valid_ids, debug)

        # -- Step 1: Resolve Gaia source_id → moca_oid -----------------
        df_resolve = _execute_query(conn, _SQL_RESOLVE, debug=debug)
        logger.info("MOCAdb: %d/%d targets resolved to moca_oid.", len(df_resolve), n_valid)

        if len(df_resolve) == 0:
            logger.warning(
                "MOCAdb: None of the %d valid Gaia DR3 IDs were found in MOCAdb. "
                "Returning table with empty MOCA columns.",
                n_valid,
            )
            return _attach_empty_columns(target_table, include_tier2)

        # Build source_id → moca_oid map
        oid_map: dict[int, int] = {}
        for _, row in df_resolve.iterrows():
            oid_map[int(row["gaiadr3_source_id"])] = int(row["moca_oid"])

        moca_oids = sorted(set(oid_map.values()))
        oid_list_str = ",".join(str(o) for o in moca_oids)

        # -- Step 2: Tier-1 query (association + age) -------------------
        query_t1 = _SQL_TIER1.format(oid_list=oid_list_str)
        df_t1 = _execute_query(conn, query_t1, debug=debug)
        logger.info("MOCAdb: Tier-1 query returned %d rows.", len(df_t1))

        # -- Step 3: Tier-2 query (activity/kinematics, optional) ------
        df_t2 = None
        if include_tier2:
            query_t2 = _SQL_TIER2.format(oid_list=oid_list_str)
            try:
                df_t2 = _execute_query(conn, query_t2, debug=debug)
                logger.info("MOCAdb: Tier-2 query returned %d rows.", len(df_t2))
            except Exception as e:
                logger.warning("MOCAdb: Tier-2 query failed (%s), continuing without.", e)
                df_t2 = None

    finally:
        conn.close()

    # ----- Merge results into target table -----
    enriched = _merge_results(
        target_table, gaia_ids, oid_map, df_t1, df_t2, include_tier2
    )
    return enriched


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_moca_columns(table: Table) -> Table:
    """Remove any existing ``MOCA_`` columns so the table can be re-enriched."""
    existing = [c for c in table.colnames if c.startswith("MOCA_")]
    if existing:
        logger.debug("MOCAdb: Removing %d existing MOCA_ columns before re-enrichment.", len(existing))
        table.remove_columns(existing)
    return table


def _upload_temp_table(conn, valid_ids: list[int], debug: bool) -> None:
    """Create and populate an indexed temporary table with Gaia DR3 IDs."""
    unique_ids = list(set(valid_ids))
    cursor = conn.cursor()
    cursor.execute("DROP TEMPORARY TABLE IF EXISTS tmp_sphere_targets")
    cursor.execute(
        "CREATE TEMPORARY TABLE tmp_sphere_targets ("
        "  gaiadr3_source_id BIGINT NOT NULL,"
        "  INDEX(gaiadr3_source_id)"
        ")"
    )
    # Batch insert in chunks of 1000
    chunk_size = 1000
    for i in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[i : i + chunk_size]
        values = ",".join(f"({gid})" for gid in chunk)
        cursor.execute(
            f"INSERT INTO tmp_sphere_targets (gaiadr3_source_id) VALUES {values}"
        )
    conn.commit()
    cursor.close()
    if debug:
        logger.debug("MOCAdb: Uploaded %d unique Gaia IDs to temp table.", len(unique_ids))


def _merge_results(
    target_table: Table,
    gaia_ids: list[int | None],
    oid_map: dict[int, int],
    df_t1: pd.DataFrame,
    df_t2: pd.DataFrame | None,
    include_tier2: bool,
) -> Table:
    """Map MOCAdb query results back to the original target table order."""
    result = target_table.copy()
    _strip_moca_columns(result)

    # Build lookup by moca_oid (deduplicate: keep first row)
    t1_map: dict[int, pd.Series] = {}
    if len(df_t1) > 0:
        df_t1_dedup = df_t1.drop_duplicates(subset="moca_oid", keep="first")
        for _, row in df_t1_dedup.iterrows():
            t1_map[int(row["moca_oid"])] = row

    t2_map: dict[int, pd.Series] = {}
    if df_t2 is not None and len(df_t2) > 0:
        df_t2_dedup = df_t2.drop_duplicates(subset="moca_oid", keep="first")
        for _, row in df_t2_dedup.iterrows():
            t2_map[int(row["moca_oid"])] = row

    n = len(result)

    def _resolve(gid):
        """Resolve a Gaia ID to its moca_oid, or None."""
        if gid is None:
            return None
        return oid_map.get(gid)

    # --- Tier-1 columns ---
    tier1_cols = {
        "MOCA_OID": ("moca_oid", "int"),
        "MOCA_AID": ("moca_aid", "str"),
        "MOCA_MEMBERSHIP_TYPE": ("moca_mtid", "str"),
        "MOCA_DESIGNATION": ("moca_designation", "str"),
        "MOCA_SPECTRAL_TYPE": ("moca_spectral_type", "str"),
        "MOCA_BANYAN_PROB": ("banyan_prob", "float"),
        "MOCA_BANYAN_UVW_SEP": ("banyan_uvw_sep_kms", "float"),
        "MOCA_YA_PROB": ("ya_prob", "float"),
        "MOCA_ASSOCIATION_NAME": ("association_name", "str"),
        "MOCA_ASSOCIATION_TYPE": ("association_type", "str"),
        "MOCA_AGE_MYR": ("association_age_myr", "float"),
        "MOCA_AGE_MYR_UNC": ("association_age_myr_unc", "float"),
        "MOCA_AGE_MYR_UNC_POS": ("association_age_myr_unc_pos", "float"),
        "MOCA_AGE_MYR_UNC_NEG": ("association_age_myr_unc_neg", "float"),
    }

    for col_name, (src_key, dtype) in tier1_cols.items():
        values = []
        for gid in gaia_ids:
            oid = _resolve(gid)
            row = t1_map.get(oid) if oid is not None else None
            if row is not None and src_key in row.index and pd.notna(row[src_key]):
                values.append(row[src_key])
            else:
                values.append(None)

        if dtype == "str":
            str_vals = [str(v) if v is not None else "" for v in values]
            result.add_column(Column(str_vals, name=col_name))
        elif dtype in ("float", "int"):
            float_vals = [float(v) if v is not None else np.nan for v in values]
            result.add_column(Column(float_vals, name=col_name))

    # --- Tier-2 columns ---
    if include_tier2:
        tier2_cols = {
            "MOCA_SPTN": ("moca_sptn", "float"),
            "MOCA_PARALLAX_MAS": ("moca_parallax_mas", "float"),
            "MOCA_X_PC": ("moca_x_pc", "float"),
            "MOCA_Y_PC": ("moca_y_pc", "float"),
            "MOCA_Z_PC": ("moca_z_pc", "float"),
            "MOCA_U_KMS": ("moca_u_kms", "float"),
            "MOCA_V_KMS": ("moca_v_kms", "float"),
            "MOCA_W_KMS": ("moca_w_kms", "float"),
            "MOCA_PROT_DAYS": ("prot_days", "float"),
            "MOCA_GAIA_ACT": ("gaia_act", "float"),
            "MOCA_EWLI": ("ewli", "float"),
            "MOCA_EWHA": ("ewha", "float"),
            "MOCA_DR3_RUWE": ("moca_dr3_ruwe", "float"),
        }

        for col_name, (src_key, _) in tier2_cols.items():
            values = []
            for gid in gaia_ids:
                oid = _resolve(gid)
                row = t2_map.get(oid) if oid is not None else None
                if row is not None and src_key in row.index and pd.notna(row[src_key]):
                    values.append(float(row[src_key]))
                else:
                    values.append(np.nan)
            result.add_column(Column(values, name=col_name))

    # --- Summary ---
    n_matched = sum(
        1 for gid in gaia_ids if gid is not None and oid_map.get(gid) in t1_map
    )
    logger.info("MOCAdb: %d/%d targets matched in MOCAdb.", n_matched, n)
    if include_tier2:
        n_matched_t2 = sum(
            1 for gid in gaia_ids if gid is not None and oid_map.get(gid) in t2_map
        )
        logger.info("MOCAdb: %d/%d targets have tier-2 data.", n_matched_t2, n)

    return result


def _attach_empty_columns(target_table: Table, include_tier2: bool) -> Table:
    """Return a copy of the table with all MOCAdb columns filled with NaN/empty."""
    result = target_table.copy()
    _strip_moca_columns(result)
    n = len(result)

    str_cols_t1 = [
        "MOCA_AID", "MOCA_MEMBERSHIP_TYPE", "MOCA_DESIGNATION",
        "MOCA_SPECTRAL_TYPE", "MOCA_ASSOCIATION_NAME", "MOCA_ASSOCIATION_TYPE",
    ]
    float_cols_t1 = [
        "MOCA_OID", "MOCA_BANYAN_PROB", "MOCA_BANYAN_UVW_SEP", "MOCA_YA_PROB",
        "MOCA_AGE_MYR", "MOCA_AGE_MYR_UNC",
        "MOCA_AGE_MYR_UNC_POS", "MOCA_AGE_MYR_UNC_NEG",
    ]
    for col in str_cols_t1:
        result.add_column(Column([""] * n, name=col))
    for col in float_cols_t1:
        result.add_column(Column([np.nan] * n, name=col))

    if include_tier2:
        float_cols_t2 = [
            "MOCA_SPTN", "MOCA_PARALLAX_MAS",
            "MOCA_X_PC", "MOCA_Y_PC", "MOCA_Z_PC",
            "MOCA_U_KMS", "MOCA_V_KMS", "MOCA_W_KMS",
            "MOCA_PROT_DAYS", "MOCA_GAIA_ACT", "MOCA_EWLI", "MOCA_EWHA",
            "MOCA_DR3_RUWE",
        ]
        for col in float_cols_t2:
            result.add_column(Column([np.nan] * n, name=col))

    return result
