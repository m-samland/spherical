#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "M. Samland @ MPIA (Heidelberg, Germany)"

import time
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time
from astroquery.simbad import Simbad

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from spherical.database.database_utils import filter_for_science_frames


def get_table_with_unique_keys(
    table_of_files,
    column_name: str,
    check_coordinates: bool = False,
    add_noname_objects: bool = False,
    group_by_healpix: bool = False,
):
    """
    Reduce a set of science files to a single representative row per unique object,
    selecting the best-quality file (based on lowest airmass) for downstream use
    such as SIMBAD queries, target list generation, or spatial indexing.

    This function is designed to deduplicate observations that share the same object
    label or sky position, ensuring each target is represented once in subsequent
    catalog cross-matching steps. It provides two modes of grouping:

    1. OBJECT-based grouping (default):
       Groups all rows by the value in the 'OBJECT' keyword. This assumes OBJECT
       labels are consistently and meaningfully used in the data (e.g., "HD123456").
       It supports optional coordinate consistency checking and special handling
       of anonymous objects ("No name") by splitting them by observation date.

    2. HEALPix-based grouping (group_by_healpix=True):
       Groups rows spatially using the 'healpix_idx' column. This is useful when
       OBJECT names are unreliable or when spatial proximity is more relevant
       than naming conventions. It provides robust grouping by sky position using
       the angular resolution defined by the HEALPix nside. 'add_noname_objects'
       is disabled in this mode, as the concept of naming no longer applies.

    Parameters
    ----------
    table_of_files : astropy.table.Table
        Input table of files, already filtered to relevant science frames.
    column_name : str, optional
        Column to group by (default: 'OBJECT'). Ignored if group_by_healpix is True.
    check_coordinates : bool, optional
        If True, verifies that all entries for a given group have consistent
        sky coordinates (RA/DEC) within 5 arcsec. Raises an error on mismatch.
    add_noname_objects : bool, optional
        If True, adds one representative row per DATE_SHORT group for entries
        where OBJECT == "No name". Helps preserve calibrators or anonymous targets.
        Ignored when group_by_healpix is True.
    group_by_healpix : bool, optional
        If True, groups rows by their 'healpix_idx' value instead of object name.
        This enables spatially robust grouping based on sky position.

    Returns
    -------
    table_of_objects : astropy.table.Table
        A new table containing one representative row per group, based on lowest airmass.

    Notes
    -----
    - This function is typically used before querying external catalogs like SIMBAD,
      which require a single coordinate per target.
    - Selecting a single row per object avoids redundant queries and ensures consistency
      in derived target lists.
    - HEALPix-based grouping is especially helpful when OBJECT labels are missing,
      inconsistent, or reused across observations.
    """

    if group_by_healpix:
        if "healpix_idx" not in table_of_files.colnames:
            raise ValueError("Column 'healpix_idx' is missing but group_by_healpix=True.")
        column_name = "healpix_idx"
        add_noname_objects = False  # Not meaningful when grouping spatially

    grouped = table_of_files.group_by(column_name)
    selected_rows = []

    for i, key in enumerate(tqdm(grouped.groups.keys, desc=f"Selecting by {column_name}")):
        group = grouped.groups[i]

        if check_coordinates:
            coords = SkyCoord(ra=group["RA"] * u.deg, dec=group["DEC"] * u.deg)
            max_sep = np.max(coords.separation(coords))
            if max_sep >= 5 * u.arcsec:
                raise ValueError(
                    f"Inconsistent coordinates (>5 arcsec) in group {column_name} = '{key[0]}'"
                )

        best_idx = np.nanargmin(group["AIRMASS"])
        selected_rows.append(group[best_idx])

    table_of_objects = Table(
        rows=selected_rows,
        names=table_of_files.colnames,
        meta=table_of_files.meta,
    )

    if not group_by_healpix and add_noname_objects:
        no_name_mask = table_of_files["OBJECT"] == "No name"
        no_name_files = table_of_files[no_name_mask]

        if len(no_name_files) > 0:
            grouped_by_date = no_name_files.group_by("DATE_SHORT")
            print(
                f'Number of "No name" objects with different dates: {len(grouped_by_date.groups.keys)}'
            )
            for i in range(len(grouped_by_date.groups.keys)):
                group = grouped_by_date.groups[i]
                table_of_objects.add_row(group[0])

    return table_of_objects


def query_SIMBAD_for_names(
    table_of_files,
    search_radius=3.0,
    number_of_retries=3.0,
    parallax_limit=1e-3,
    J_mag_limit=15.,
    verbose=False,
    batch_size=250,
    min_delay=1.0,
):
    """
    Cross-match observation metadata with SIMBAD to retrieve canonical source names and identifiers.

    This function queries the SIMBAD TAP service in batches using uploaded RA/Dec coordinates 
    and observation times (MJD) to find matching astronomical sources within a specified 
    search radius. It corrects for proper motion to the epoch of observation and filters 
    matches based on parallax and J-band magnitude. The output includes cross-identifications 
    from catalogs like Gaia DR3, 2MASS, HIP, and HD.

    Parameters
    ----------
    table_of_files : astropy.table.Table
        Table containing observation metadata with the following required columns:
        - 'OBJECT' (str): Target identifier as recorded in the FITS header.
        - 'RA' (float): Right Ascension of the target in degrees (ICRS).
        - 'DEC' (float): Declination of the target in degrees (ICRS).
        - 'MJD_OBS' (float): Modified Julian Date of the observation (UTC).

    search_radius : float, optional
        Radius in arcminutes used to search for SIMBAD matches around each target coordinate.
        Defaults to 3.0 arcmin.

    number_of_retries : float, optional
        Number of retry attempts for failed SIMBAD TAP queries. Defaults to 3.0.

    parallax_limit : float, optional
        Minimum required parallax in milliarcseconds (mas) to retain a matched object.
        This helps reject distant background stars. Defaults to 1e-3 mas.

    J_mag_limit : float, optional
        Limiting J-band magnitude. Only SIMBAD entries brighter than this limit are kept.
        Defaults to 15 (Vega system).

    verbose : bool, optional
        If True, print detailed progress and error messages during querying. Default is False.

    batch_size : int, optional
        Number of targets to upload in each SIMBAD TAP batch query. Default is 250.

    min_delay : float, optional
        Minimum delay between successive queries in seconds to avoid SIMBAD rate-limiting.
        Defaults to 1.0 seconds.

    Returns
    -------
    simbad_table : astropy.table.Table
        A table of SIMBAD matches with the following columns:

        - 'OBJ_HEADER' (str): Original 'OBJECT' value from the FITS header.
        - 'RA_HEADER' (float): Right Ascension from the input table [deg, ICRS].
        - 'DEC_HEADER' (float): Declination from the input table [deg, ICRS].
        - 'MAIN_ID' (str): Canonical SIMBAD identifier for the matched object.
        - 'RA_DEG' (Quantity): Right Ascension from SIMBAD [deg, J2000].
        - 'DEC_DEG' (Quantity): Declination from SIMBAD [deg, J2000].
        - 'OTYPE' (str): SIMBAD object type classification (e.g., 'Star', 'BrownD', 'PM*').
        - 'SP_TYPE' (str): Spectral type of the object (e.g., 'G2V', 'M4.5').
        - 'FLUX_V' (float or masked): V-band magnitude [mag, Vega system].
        - 'FLUX_R' (float or masked): R-band magnitude [mag, Vega system].
        - 'FLUX_I' (float or masked): I-band magnitude [mag, Vega system].
        - 'FLUX_J' (float or masked): J-band magnitude [mag, Vega system].
        - 'FLUX_H' (float or masked): H-band magnitude [mag, Vega system].
        - 'FLUX_K' (float or masked): K-band magnitude [mag, Vega system].
        - 'PMRA' (float): Proper motion in Right Ascension [mas/yr], includes cos(Dec).
        - 'PMDEC' (float): Proper motion in Declination [mas/yr].
        - 'PM_ERR_MAJA' (float or masked): Major axis uncertainty of proper motion [mas/yr].
        - 'PM_ERR_MINA' (float or masked): Minor axis uncertainty of proper motion [mas/yr].
        - 'PLX' (float): Parallax [mas].
        - 'PLX_ERROR' (float or masked): Uncertainty in parallax [mas].
        - 'PLX_BIBCODE' (str or masked): Bibliographic reference for the parallax measurement.
        - 'RV_VALUE' (float or masked): Radial velocity [km/s].
        - 'RVZ_ERROR' (float or masked): Uncertainty in radial velocity [km/s].
        - 'POS_DIFF_ORIG' (Quantity): Angular separation [arcsec] between header coordinates and SIMBAD (no PM correction).
        - 'POS_DIFF' (Quantity): Angular separation [arcsec] after proper motion correction to epoch of observation.
        - 'STARS_IN_CONE' (int): Number of SIMBAD sources within the search cone for a given target.
        - 'ID_GAIA_DR3' (str or masked): Gaia DR3 identifier (if available).
        - 'ID_2MASS' (str or masked): 2MASS identifier.
        - 'ID_TYC' (str or masked): Tycho catalog identifier.
        - 'ID_HD' (str or masked): Henry Draper catalog identifier.
        - 'ID_HIP' (str or masked): HIPPARCOS catalog identifier.
        - 'DISTANCE' (Quantity): Distance inferred from parallax [pc], computed as 1000 / parallax.

    not_found_list : np.ndarray
        Array of 'OBJECT' names from the input that could not be matched to any SIMBAD entry.

    Raises
    ------
    IOError
        If the required ADQL query file cannot be loaded.

    Notes
    -----
    - Coordinates are matched in the ICRS frame (equinox J2000.0), with optional proper motion correction.
    - Assumes the 'simbad_tap_query.adql' file exists in the same directory as the module.
    - The function uses the CDS TAP service with retries and rate-limiting to handle large batches robustly.
    - The resulting table is filtered to contain reliable astrometric data and IDs from major catalogs.
    """


    search_radius = search_radius * u.arcmin

    # Set up simbad
    simbad = Simbad()
    simbad.clear_cache()
    simbad.ROW_LIMIT = 20_000
    simbad.TIMEOUT = 60

    # Convert table data to a format suitable for TAP upload
    object_list = table_of_files["OBJECT", "RA", "DEC", "MJD_OBS"].copy()

    # Load the ADQL query from file
    query_file_path = Path(__file__).parent / "simbad_tap_query.adql"
    with open(query_file_path, 'r') as file:
        query_template = file.read()

    # format query with search radius
    search_radius_deg = search_radius.to(u.deg).value
    query = query_template.format(search_radius_deg=search_radius_deg)
    
    print(f"Querying SIMBAD with search radius: {search_radius} ({search_radius_deg:.2e} degrees)")
    print(f"Number of objects to query: {len(object_list)}")
    print(f"Using batch size: {batch_size} with minimum delay of {min_delay} seconds between queries")
    
    # Split the object list into batches
    total_objects = len(object_list)
    num_batches = (total_objects + batch_size - 1) // batch_size  # Ceiling division
    
    all_results = []
    
    # Process each batch with progress bar
    for batch_idx in tqdm(range(num_batches), desc="Querying SIMBAD in batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_objects)
        
        batch = object_list[start_idx:end_idx]
        
        # if verbose:
        #     print(f"Processing batch {batch_idx+1}/{num_batches}, objects {start_idx+1}-{end_idx}")
        
        # Track query time to ensure minimum delay
        query_start_time = time.time()
        
        # Execute the TAP query with retries
        for attempt in range(int(number_of_retries)):
            try:
                batch_results = Simbad.query_tap(query, object_data=batch)
                if batch_results is not None:
                    all_results.append(batch_results)
                    break
            except Exception as e:
                if verbose:
                    print(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt == number_of_retries - 1:
                    print(f"Failed to query batch {batch_idx+1} after {number_of_retries} attempts")
                time.sleep(0.5)
        
        # Ensure minimum delay between queries
        query_duration = time.time() - query_start_time
        if query_duration < min_delay:
            time.sleep(min_delay - query_duration)
    
    # Merge all results
    if not all_results:
        print("No results returned from SIMBAD")
        return Table(), object_list["OBJECT"]
    
    # Combine all result tables
    results = vstack(all_results)

    # Filter results to only allow objects with J band magnitude, parallax information and proper motion
    columns_to_check = ['flux_j', 'pmra', 'pmdec', 'plx_value']

    # Build a combined mask: True where all are **not** masked
    valid_rows = ~results[columns_to_check[0]].mask
    for col in columns_to_check[1:]:
        valid_rows &= ~results[col].mask

    # Apply the mask to filter the table
    results = results[valid_rows]
    results = results[results["flux_j"] <= J_mag_limit]
    results = results[results["plx_value"] >= parallax_limit]

    # Queried coordinates
    queried_coords = SkyCoord(
        ra=results["user_specified_ra"] * u.deg,
        dec=results["user_specified_dec"] * u.deg,
        frame="icrs",
        # obstime=obs_times,
    )

    # Observation times from your input list
    obs_times = Time(results["user_specified_mjd_obs"], format="mjd")

    simbad_coords = SkyCoord(
        ra=results["ra"],
        dec=results["dec"],
        frame="icrs",
        pm_ra_cosdec=results["pmra"],
        pm_dec=results["pmdec"],
        obstime=Time("J2000.0")
    )
    # Apply space motion to coordinates of queried objects to accounting for observing time
    corrected_coords = simbad_coords.apply_space_motion(new_obstime=obs_times)

    # Compute separation of query from on-sky position before and after proper motion correction
    results["sep_orig"] = simbad_coords.separation(queried_coords).arcsecond
    results["sep_corr"] = corrected_coords.separation(queried_coords).arcsecond

    # Convert Astropy Table to a Pandas DataFrame
    df = results.to_pandas()

    # Add a column with the number of stars in the cone that match the search criteria
    unique_counts = (
        df.groupby("user_specified_mjd_obs")["main_id"]
        .nunique()
        .rename("STARS_IN_CONE")
    )
    # Merge back to original DataFrame
    df = df.merge(unique_counts, on="user_specified_mjd_obs")

    # Find the closest match for each observation
    df_closest = df.loc[df.groupby('user_specified_mjd_obs')['sep_corr'].idxmin()]
    
    # If there are multiple matches with the same name, keep only the one with the smallest separation
    df_unique = df_closest.loc[df_closest.groupby('main_id')['sep_corr'].idxmin()]

    df_unique['user_specified_id'] = df_unique['user_specified_id'].apply(lambda x: x.strip())

    # Extract all requested catalogue IDs from the 'all_ids' column
    catalogues = ["Gaia DR3", "2MASS", "TYC", "HD", "HIP"]
    def extract_ids(all_ids_value):
        # If the value is nan, return a dict with all keys and np.nan as values.
        if pd.isnull(all_ids_value):
            return {f"ID_{prefix.replace(' ', '_').upper()}": np.nan for prefix in catalogues}
        
        # Split on '|' and strip each part. If no '|' is present, split still returns a one-element list.
        parts = [s.strip() for s in str(all_ids_value).split('|')]
        
        # Build a dictionary for each prefix
        result = {}
        for prefix in catalogues:
            # Filter parts that start with the given prefix
            matched = [s for s in parts if s.startswith(prefix)]
            # Define the column name; replace spaces with underscores and convert to upper-case
            col_name = f"ID_{prefix.replace(' ', '_').upper()}"
            # Join the matching strings with '|' or assign np.nan if there is no match
            result[col_name] = '|'.join(matched) if matched else np.nan
        return result
    
    # Apply the extract_ids function to the 'all_ids' column row-wise
    df_extracted_ids = df_unique['all_ids'].apply(extract_ids).apply(pd.Series)
    
    # Drop the 'all_ids' column
    df_unique.drop(columns=['all_ids'], inplace=True)

    # Concatenate the new columns with the original dataframe
    df_unique = pd.concat([df_unique, df_extracted_ids], axis=1, sort=False).copy()

    requested_ids = object_list.to_pandas()["OBJECT"].astype(str).str.strip()
    matched_ids = df_unique["user_specified_id"].astype(str).str.strip()

    # Now check for which requested IDs were not matched
    not_found_mask = ~requested_ids.isin(matched_ids)
    not_found_list = requested_ids[not_found_mask].to_numpy()

    # Convert column names to upper-case
    df_unique.drop(columns=['user_specified_mjd_obs'], inplace=True)
    df_unique.columns = df_unique.columns.str.upper()
   
    # Convert back to Astropy Table
    simbad_table = Table.from_pandas(df_unique)

    # cast ID columns to strings
    simbad_table['ID_GAIA_DR3'] = simbad_table['ID_GAIA_DR3'].astype(str)
    simbad_table['ID_2MASS'] = simbad_table['ID_2MASS'].astype(str)
    simbad_table['ID_TYC'] = simbad_table['ID_TYC'].astype(str)
    simbad_table['ID_HD'] = simbad_table['ID_HD'].astype(str)
    simbad_table['ID_HIP'] = simbad_table['ID_HIP'].astype(str)

    # Add required columns
    simbad_table.rename_column('PLX_VALUE', 'PLX')
    simbad_table['DISTANCE'] = np.round(1. / (1e-3 * simbad_table['PLX'].data), 3) * u.pc

    simbad_table.rename_column('USER_SPECIFIED_ID', 'OBJ_HEADER')
    simbad_table.rename_column('MAIN_ID', 'MAIN_ID')

    simbad_table.rename_column('RA', 'RA_DEG')
    simbad_table['RA_DEG'] *= u.degree
    simbad_table.rename_column('USER_SPECIFIED_RA', 'RA_HEADER')

    simbad_table.rename_column('DEC', 'DEC_DEG')
    simbad_table['DEC_DEG'] *= u.degree
    simbad_table.rename_column('USER_SPECIFIED_DEC', 'DEC_HEADER')

    simbad_table.rename_column('SEP_CORR', 'POS_DIFF')
    simbad_table['POS_DIFF'] = np.round(simbad_table['POS_DIFF'], 3) * u.arcsec

    simbad_table.rename_column('SEP_ORIG', 'POS_DIFF_ORIG')
    simbad_table['POS_DIFF_ORIG'] = np.round(simbad_table['POS_DIFF_ORIG'], 3) * u.arcsec

    return simbad_table, not_found_list


def make_target_list_with_SIMBAD(
    table_of_files,
    instrument,
    polarimetry: bool = False,
    search_radius: float = 0.5,
    parallax_limit: float = 1e-3,
    J_mag_limit: float = 15.0,
    number_of_retries: int = 1,
    remove_fillers: bool = True,
    use_center_files_only: bool = False,
    check_coordinates: bool = True,
    add_noname_objects: bool = True,
    batch_size: int = 250,
    min_delay: float = 1.0,
    verbose: bool = False,
    group_by_healpix: bool = False,
):
    """
    Generate a list of science targets by filtering observational files,
    identifying unique target fields, and resolving their coordinates and
    names via SIMBAD.

    This function supports grouping by object name (OBJECT) or by sky
    position using HEALPix indices to ensure a unique target list suitable
    for catalog crossmatching and scientific analysis.

    Parameters
    ----------
    table_of_files : astropy.table.Table
        Table containing FITS header metadata from science observations.
    instrument : str
        Instrument identifier ('irdis' or 'ifs').
    polarimetry : bool
        If True, include only frames with 'DPR_TECH' containing 'POLARIMETRY'.
        If False, exclude such frames.
    search_radius : float
        SIMBAD search radius in arcseconds.
    parallax_limit : float
        Minimum parallax (in arcsec) to exclude distant targets.
    J_mag_limit : float
        Maximum J-band magnitude threshold.
    number_of_retries : int
        Number of retries for each SIMBAD batch query on failure.
    remove_fillers : bool
        Whether to exclude OBJECTs containing "filler".
    use_center_files_only : bool
        If True, restrict to OBJECT,CENTER frames only.
    check_coordinates : bool
        If True, assert coordinate consistency per target group (within 5 arcsec).
    add_noname_objects : bool
        Whether to add one row per DATE_SHORT for OBJECT == 'No name'.
        Ignored if group_by_healpix is True.
    batch_size : int
        SIMBAD query batch size.
    min_delay : float
        Minimum delay between SIMBAD queries (seconds).
    verbose : bool
        If True, prints verbose output during SIMBAD query.
    group_by_healpix : bool
        If True, group targets by HEALPix index instead of OBJECT name.

    Returns
    -------
    table_of_targets : astropy.table.Table
        Resolved target list with SIMBAD metadata.
    not_found_list : list of str
        List of target identifiers or coordinates not matched in SIMBAD.
    """
    print("Filtering for science frames...")

    _, t_center, _, t_center_coro, _ = filter_for_science_frames(
        table_of_files=table_of_files,
        instrument=instrument,
        polarimetry=polarimetry,
        remove_fillers=remove_fillers,
    )

    # Decide early which table to use
    input_source = t_center if use_center_files_only else t_center_coro

    print("Assigning HEALPix indices to input source...")

    target_coords = SkyCoord(
        ra=input_source["RA"] * u.deg,
        dec=input_source["DEC"] * u.deg,
    )
    phi = target_coords.ra.radian
    theta = target_coords.dec.radian + np.pi / 2.0  # HEALPix theta = colatitude
    nside = 2**15  # ~0.44 arcsec resolution
    input_source["healpix_idx"] = hp.ang2pix(nside, theta, phi)

    print(f"Selecting one representative file per {'HEALPix cell' if group_by_healpix else 'OBJECT'}...")

    input_file_table = get_table_with_unique_keys(
        table_of_files=input_source,
        column_name="OBJECT",
        check_coordinates=check_coordinates,
        add_noname_objects=add_noname_objects,
        group_by_healpix=group_by_healpix,
    )

    input_file_table.sort("MJD_OBS")

    print("Querying SIMBAD for resolved names and coordinates...")

    table_of_targets, not_found_list = query_SIMBAD_for_names(
        input_file_table,
        search_radius=search_radius,
        parallax_limit=parallax_limit,
        J_mag_limit=J_mag_limit,
        number_of_retries=number_of_retries,
        batch_size=batch_size,
        min_delay=min_delay,
        verbose=verbose,
    )

    return table_of_targets, not_found_list
