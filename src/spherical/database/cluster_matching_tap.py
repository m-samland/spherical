#!/usr/bin/env python
"""cluster_matching_tap.py - ADQL-powered Gaia to cluster join (Hunt et al. 2024)

*One server-side ADQL query, zero network loops.*

Given Gaia DR3 `source_id` values this helper attaches open-cluster membership
and age information from Hunt et al. (2024) (VizieR cat.
`J/A+A/686/A42`).

Highlights
~~~~~~~~~~
* Uses **ADQL JOIN** executed inside the CDS TAP service – only one HTTP call.
* Accepts an **iterable** of IDs or an **Astropy Table** (with configurable
  `id_column`).  Strings like "Gaia DR3 123…" are auto-cleaned.
* Handles large lists by uploading a VOTable and joining it server-side.
* Returns the original table with five extra columns or, for iterables, a fresh
  table with those six columns.
"""
from __future__ import annotations

import re
from typing import Iterable, Union

import numpy as np
from astropy.table import Column, Table
from astroquery.utils.tap import TapPlus

__all__ = ["map_gaia_ids_to_cluster_ages"]

# -----------------------------------------------------------------------------
# TAP configuration (VizieR public endpoint)  
# -----------------------------------------------------------------------------
# Try multiple possible TAP URLs
TAP_URLS = [
    "http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/",  # Working TAPVizieR endpoint
]

# Try the first URL by default
TAP_URL = TAP_URLS[0]
TAP = TapPlus(url=TAP_URL)

# Fully-qualified table names (need quotes in ADQL)
MEMBER_TAB = '"J/A+A/686/A42/members"'
CLUSTER_TAB = '"J/A+A/686/A42/clusters"'

# Alternative table names to try
ALT_MEMBER_TAB = 'J/A+A/686/A42/members'
ALT_CLUSTER_TAB = 'J/A+A/686/A42/clusters'

# Threshold: if number of IDs > this, switch to TAP upload method
UPLOAD_THRESHOLD = 800  # empirically safe for IN-lists

# -----------------------------------------------------------------------------
# Helper: parse / clean Gaia IDs
# -----------------------------------------------------------------------------

def _clean_gid(value: Union[int, str]) -> int | None:
    """Return integer Gaia DR3 ID or ``None`` if not parseable."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    m = re.search(r"(?:Gaia\s*DR3\s*)?(\d{5,})$", str(value).strip(), re.I)
    return int(m.group(1)) if m else None

# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------

def map_gaia_ids_to_cluster_ages(
    data: Union[Iterable[Union[int, str]], Table],
    *,
    id_column: str = "gaia_id",
    asynchronous: bool = True,
    debug: bool = False,
) -> Table:
    """Attach cluster membership & ages to Gaia DR3 IDs using **one ADQL query**.

    Parameters
    ----------
    data
        Iterable of IDs *or* an `astropy.table.Table` with a Gaia ID column.
    id_column
        Column name when *data* is a Table.
    asynchronous
        Run TAP query asynchronously (recommended).  Synchronous for small sets
        if you prefer.
    debug
        Print debug information including the ADQL query.
    """

    # --------------------------------------------------
    # Normalise input → list of *clean* IDs (may contain None)
    # --------------------------------------------------
    if isinstance(data, Table):
        if id_column not in data.colnames:
            raise ValueError(f"Input table lacks column '{id_column}'.")
        base_tbl = data.copy()
        raw_ids = list(base_tbl[id_column])
    else:
        base_tbl = None
        raw_ids = list(data)

    ids = [_clean_gid(x) for x in raw_ids]

    # --------------------------------------------------
    # Build ADQL query (IN-list or TAP upload)
    # --------------------------------------------------
    valid_ids = [i for i in ids if i is not None]

    if not valid_ids:  # nothing to resolve → just attach empty columns
        return _attach_empty(base_tbl, ids)

    if len(valid_ids) <= UPLOAD_THRESHOLD:
        # --- IN-list variant ------------------------------------------------
        id_list = ",".join(str(i) for i in set(valid_ids))
        adql = f"""
        SELECT m."GaiaDR3"  AS gaia_id,
               c."ID"       AS cluster_id,
               c."Name"     AS cluster_name,
               c."logAge16", c."logAge50", c."logAge84"
        FROM   {MEMBER_TAB}  AS m
        JOIN   {CLUSTER_TAB} AS c ON m."ID" = c."ID"
        WHERE  m."GaiaDR3" IN ({id_list})
        """
        
        if debug:
            print(f"TAP URL: {TAP_URL}")
            print(f"ADQL Query:\n{adql}")
            print(f"Valid IDs: {valid_ids}")
        
        match_tbl = None
        
        # Try different TAP URLs if needed
        for i, tap_url in enumerate(TAP_URLS):
            if i > 0:  # Only try alternatives if first fails
                if debug:
                    print(f"Trying alternative TAP URL: {tap_url}")
                global TAP
                TAP = TapPlus(url=tap_url)
            
            try:
                job = TAP.launch_job_async(adql) if asynchronous else TAP.launch_job(adql)
                match_tbl = job.get_results()
                if debug:
                    print(f"Success with TAP URL: {tap_url}")
                break  # Success, exit the loop
            except Exception as e:
                if debug:
                    print(f"TAP query failed with {tap_url}: {e}")
                if i == len(TAP_URLS) - 1:  # Last URL, try alternative table names
                    print("Trying alternative table name format...")
                    # Try without quotes
                    adql_alt = f"""
                    SELECT m.GaiaDR3  AS gaia_id,
                           c.ID       AS cluster_id,
                           c.Name     AS cluster_name,
                           c.logAge16, c.logAge50, c.logAge84
                    FROM   {ALT_MEMBER_TAB}  AS m
                    JOIN   {ALT_CLUSTER_TAB} AS c ON m.ID = c.ID
                    WHERE  m.GaiaDR3 IN ({id_list})
                    """
                    
                    if debug:
                        print(f"Alternative ADQL Query:\n{adql_alt}")
                    
                    try:
                        job = TAP.launch_job_async(adql_alt) if asynchronous else TAP.launch_job(adql_alt)
                        match_tbl = job.get_results()
                        break
                    except Exception as e2:
                        print(f"Alternative query also failed: {e2}")
                        print("TAP service unavailable - returning empty results")
                        return _attach_empty(base_tbl, ids)
                
        # If we still don't have results, return empty
        if match_tbl is None or len(match_tbl) == 0:
            return _attach_empty(base_tbl, ids)
    else:
        # --- Upload variant -----------------------------------------------
        upload_tbl = Table({"gaia_id": list(set(valid_ids))})
        adql = f"""
        SELECT u.gaia_id,
               c."ID"        AS cluster_id,
               c."Name"      AS cluster_name,
               c."logAge16", c."logAge50", c."logAge84"
        FROM   TAP_UPLOAD.ids        AS u
        JOIN   {MEMBER_TAB} AS m ON m."GaiaDR3" = u.gaia_id
        JOIN   {CLUSTER_TAB} AS c ON c."ID"     = m."ID"
        """
        
        if debug:
            print(f"TAP URL: {TAP_URL}")
            print(f"ADQL Query (with upload):\n{adql}")
            print(f"Upload table: {len(upload_tbl)} unique IDs")
        
        try:
            job = TAP.launch_job_async(adql, upload_resource=upload_tbl, upload_table_name="ids") if asynchronous else TAP.launch_job(adql, upload_resource=upload_tbl, upload_table_name="ids")
            match_tbl = job.get_results()
        except Exception as e:
            print(f"Upload query failed: {e}")
            # Return empty results if TAP is unavailable
            return _attach_empty(base_tbl, ids)

    # --------------------------------------------------
    # Map matches back to original order / duplicates
    # --------------------------------------------------
    match_map = {row["gaia_id"]: row for row in match_tbl}

    cluster_id = []
    cluster_name = []
    age16 = []
    age50 = []
    age84 = []

    for gid in ids:
        r = match_map.get(gid)
        if r is None:
            cluster_id.append(None)
            cluster_name.append(None)
            age16.append(None)
            age50.append(None)
            age84.append(None)
        else:
            cluster_id.append(int(r["cluster_id"]))
            cluster_name.append(r["cluster_name"])  # already str
            age16.append(r["logAge16"])
            age50.append(r["logAge50"])
            age84.append(r["logAge84"])

    # --------------------------------------------------
    # Return combined table
    # --------------------------------------------------
    new_cols = {
        "cluster_id": Column(cluster_id),
        "cluster_name": Column(cluster_name),
        "logAge16": Column(age16),
        "logAge50": Column(age50),
        "logAge84": Column(age84),
    }

    if base_tbl is not None:
        for name, col in new_cols.items():
            base_tbl.add_column(col, name=name)
        return base_tbl

    return Table({"gaia_id": ids, **new_cols})

# -----------------------------------------------------------------------------
# Helper: attach empty result columns (all None)
# -----------------------------------------------------------------------------

def _attach_empty(base_tbl: Table | None, ids: list[int | None]) -> Table:
    blank = [None] * len(ids)
    cols = {
        "cluster_id": Column(blank.copy()),
        "cluster_name": Column(blank.copy()),
        "logAge16": Column(blank.copy()),
        "logAge50": Column(blank.copy()),
        "logAge84": Column(blank.copy()),
    }
    if base_tbl is not None:
        for n, c in cols.items():
            base_tbl.add_column(c, name=n)
        return base_tbl
    return Table({"gaia_id": ids, **cols})

# -----------------------------------------------------------------------------
# Manual smoke test
# -----------------------------------------------------------------------------

def _example():
    """Simple example - only tries ADQL TAP query."""
    sample = [
        "Gaia DR3 6202242567124829312", "Gaia DR3 331639215355235968", "bad id"]
    return map_gaia_ids_to_cluster_ages(sample, debug=True)
