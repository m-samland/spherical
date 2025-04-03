#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "M. Samland @ MPIA (Heidelberg, Germany)"

import re
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
from tqdm import tqdm


def filter_for_science_frames(table_of_files, instrument, remove_fillers=True):
    """Takes the master table of files (or subset thereof) and return 3 tables.
    One with the corongraphic files, one with the centering files, and one with both.
    Instrument: IRDIS or IFS
    """
    if instrument == "IRDIS":
        t_instrument = table_of_files[table_of_files["DET_ID"] == "IRDIS"]
    elif instrument == "IFS":
        t_instrument = table_of_files[table_of_files["DET_ID"] == "IFS"]

    def get_boolean_mask_from_true(df, column_name):
        boolean_mask = df[column_name].astype(str).str.lower().isin(['true', 't', '1'])
        return boolean_mask

    try:
        shutter_mask = get_boolean_mask_from_true(t_instrument.to_pandas(), "SHUTTER")
    except AttributeError:
        shutter_mask = get_boolean_mask_from_true(t_instrument, "SHUTTER")

    science_mask = np.logical_and.reduce(
        (
            t_instrument["DEC"] != -10000,
            t_instrument["DPR_TYPE"] != "DARK",
            t_instrument["DPR_TYPE"] != "FLAT,LAMP",
            t_instrument["DPR_TYPE"] != "OBJECT,ASTROMETRY",
            t_instrument["DPR_TYPE"] != "STD",
            t_instrument["CORO"] != "N/A",
            t_instrument["READOUT_MODE"] == "Nondest",
            shutter_mask
            # t_instrument["SHUTTER"] == shutter_filter,
        )
    )


    t_science = t_instrument[science_mask]
    if remove_fillers:
        index_of_fillers = [
            i for i, item in enumerate(t_science["OBJECT"]) if "filler" in item
        ]
        mask_filler = np.ones(len(t_science), dtype=bool)
        mask_filler[index_of_fillers] = False
        t_science = t_science[mask_filler]

    # List of science cubes and center cube and both
    try:
        t_phot = t_science[t_science["DPR_TYPE"] == "OBJECT,FLUX"]
        print(
            "Number of Object keys for flux sequence: {}".format(
                len(t_phot.group_by("OBJECT").groups.keys)
            )
        )
    except IndexError:
        t_phot = None
        print("No flux frames.")
    try:
        t_coro = t_science[t_science["DPR_TYPE"] == "OBJECT"]
        print(
            "Number of Object keys for Coronagraphic sequence: {}".format(
                len(t_coro.group_by("OBJECT").groups.keys)
            )
        )
    except IndexError:
        t_coro = None
        print("No coro frames.")
    try:
        t_center = t_science[t_science["DPR_TYPE"] == "OBJECT,CENTER"]
        print(
            "Number of Object keys for Center frames: {}".format(
                len(t_center.group_by("OBJECT").groups.keys)
            )
        )
    except IndexError:
        t_center = None
        print("No center frames")
    try:
        t_center_coro = t_science[
            np.logical_or.reduce(
                (
                    t_science["DPR_TYPE"] == "OBJECT",
                    t_science["DPR_TYPE"] == "OBJECT,CENTER",
                )
            )
        ]
        print(
            "Number of Object keys for Center+Coro frames: {}".format(
                len(t_center_coro.group_by("OBJECT").groups.keys)
            )
        )
    except IndexError:
        t_center_coro = None
        print("No Center or Coro frames")
    try:
        t_science = t_science[
            np.logical_or.reduce(
                (
                    t_science["DPR_TYPE"] == "OBJECT",
                    t_science["DPR_TYPE"] == "OBJECT,CENTER",
                    t_science["DPR_TYPE"] == "OBJECT,FLUX",
                    t_science["DPR_TYPE"] == "SKY",
                )
            )
        ]
        print(
            "Number of Object keys for all science frames: {}".format(
                len(t_science.group_by("OBJECT").groups.keys)
            )
        )
    except IndexError:
        t_science = None
        print("No science frames at all!")

    return t_coro, t_center, t_center_coro, t_science


def get_table_with_unique_keys(
    table_of_files, column_name, check_coordinates=False, add_noname_objects=False
):
    """Takes the master table of files (or subset thereof) and a column name as a
    string returns it with only one file per object key. Should be prefiltered to
    only include science frames.
    The files are checked for consistency in coordinates before only one of them is selected.
    If there is a larger than 5 arcsec deviation an exception is raised.

    """

    counter = 0
    for key in tqdm(table_of_files.group_by(column_name).groups.keys):
        files = table_of_files[table_of_files[column_name] == key[0]]
        if check_coordinates:
            list_of_coords = SkyCoord(
                ra=files["RA"] * u.degree, dec=files["DEC"] * u.degree
            )
            maximum_coord_difference = np.max(list_of_coords.separation(list_of_coords))
            assert (
                maximum_coord_difference < 5 * u.arcsec
            ), "Differences in coordinates for same object: larger than 5 arcsec."
        # assert len(files.group_by('RA').groups.keys)==1,"Different RA values for same Object: {}".format(key[0])
        # assert len(files.group_by('DEC').groups.keys)==1,"Different DEC values for same Object: {}".format(key[0])
        lowest_airmass = np.nanargmin(files["AIRMASS"])
        row = files[lowest_airmass]  # First entry of that key only
        dtypes = []
        for i in range(len(table_of_files.dtype)):
            dtypes.append(table_of_files.dtype[i])

        if counter == 0:  # Create table from one row for first iteration
            table_of_objects = Table(
                rows=row,
                names=table_of_files.colnames,
                dtype=dtypes,
                meta=table_of_files.meta,
            )
            counter += 1
        else:
            table_of_objects.add_row(row)  # Add subsequent rows to the table

    # Add row for each "No name"-object of a different date
    if add_noname_objects is True:
        files_no_name = table_of_files[table_of_files["OBJECT"] == "No name"]
        if len(files_no_name) > 0:
            dates_of_noname = files_no_name.group_by("DATE_SHORT").groups.keys
            print(
                'Number of "No name"-Objects with different date: {}'.format(
                    len(dates_of_noname)
                )
            )
            if len(dates_of_noname) > 1:
                for key in dates_of_noname:
                    row = files_no_name[files_no_name["DATE_SHORT"] == key[0]][0]
                    table_of_objects.add_row(row)

    return table_of_objects


def matches_pattern(s):
    """
    Check if the last two characters of a string match the pattern.

    Parameters
    ----------
    s : str
        The string to be checked.

    Returns
    -------
    bool
        True if the string matches the pattern, False otherwise.
    """

    pattern = r"( [b-h]|[1-9][b-h]|[1-9][B-D])$"
    return bool(re.search(pattern, s))


def retry_query(function, number_of_retries=3, verbose=False, **kwargs):
    for attempt in range(int(number_of_retries)):
        if verbose:
            print(attempt)
        try:
            result = function(**kwargs)
            if result is not None:
                return result
            time.sleep(0.3)
        except:
            pass
    return None


def correct_for_proper_motion(
    coordinates,
    pm_ra,
    pm_dec,
    time_of_observation,
    sign="positive",
):
    """Correct J2000 coordinates for subsequent proper motion.

    Parameters
    ----------
    coordinates : coordinate object in J2000 epoch and equinox.
        Astropy coordinate object
    pm_ra : quantity
        Proper motion in mas / arcsec as astropy quantity.
    pm_dec : quantity
        Proper motion in mas / arcsec as astropy quantity.
    time_of_observation : time object
        Astropy time object.
    sign : type
        "positive": proper motions added.
        "negative": proper motinos subtracted.

    Returns
    -------
    type
        Proper motion corrected coordinates.

    """

    epoch = Time("2000-01-01 11:58:55.816", scale="utc")
    time_difference_years = (time_of_observation - epoch).to(u.year)

    ra_offset = (time_difference_years * pm_ra / np.cos(coordinates.dec)).to(u.arcsec)
    dec_offset = (time_difference_years * pm_dec).to(u.arcsec)
    if sign == "positive":
        new_ra = coordinates.ra + ra_offset
        new_dec = coordinates.dec + dec_offset
    elif sign == "negative":
        new_ra = coordinates.ra - ra_offset
        new_dec = coordinates.dec - dec_offset
    # Put in old coordinate for objects without PM information
    mask = np.isnan(new_ra)
    new_ra[mask] = coordinates.ra[mask]
    new_dec[mask] = coordinates.dec[mask]

    new_coordinates = SkyCoord(ra=new_ra, dec=new_dec)

    return new_coordinates


def query_SIMBAD_for_names(
    table_of_files,
    search_radius=3.0,
    number_of_retries=3.0,
    parallax_limit=1e-3,
    J_mag_limit=15.,
    verbose=False,
    batch_size=250,
    min_delay=1.0, # down to 0.25 should be ok
):
    """Query SIMBAD for object names in batches.

    Parameters
    ----------
    table_of_files : type
        Table containing 'OBJECT', 'RA', and 'DEC' keywords.
    search_radius : type
        Search radius in arcminutes.
    number_of_retries : type
        Number of times to repeat any query upon failure.
    J_mag_limit : type
        Limiting magnitude in J band.
    verbose : bool
        Whether to print verbose output.
    batch_size : int
        Maximum number of objects to query at once.
    min_delay : float
        Minimum delay between queries in seconds.

    Returns
    -------
    type
        Simbad results table and list of objects not found.
    """

    search_radius = search_radius * u.arcmin

    # set up simbad
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
    search_radius=0.5,
    parallax_limit=1e-3,
    J_mag_limit=15.,
    number_of_retries=1,
    remove_fillers=True,
    use_center_files_only=False,
    check_coordinates=True,
    add_noname_objects=True,
    batch_size=250,
    min_delay=1.0,
    verbose=False,
):
    print("Filter for science frames only...")
    if instrument == "IRDIS":
        t_coro, t_center, t_center_coro, t_science = filter_for_science_frames(
            table_of_files, "IRDIS", remove_fillers
        )
    elif instrument == "IFS":
        t_coro, t_center, t_center_coro, t_science = filter_for_science_frames(
            table_of_files, "IFS", remove_fillers
        )
    else:
        raise NotImplementedError(
            "Instrument: {} is not implemented.".format(instrument)
        )

    print("Make list of unique object keys...")

    observed_coords = SkyCoord(
        ra=t_center_coro["RA"] * u.degree, dec=t_center_coro["DEC"] * u.degree
    )
    phi = observed_coords.ra.radian
    theta = observed_coords.dec.radian + np.pi / 2.0
    nside = int(2 ** 15) # 32768
    
    # print((hp.nside2resol(nside) * u.radian).to(u.arcsec))

    pixel_indices = hp.ang2pix(nside, theta, phi)
    t_center_coro["healpix_idx"] = pixel_indices

    # Minimum requirement for being one sequence: same 'OBJECT' keyword.
    # The set of unique object keywords should be larger than the set of unique real target names.
    if use_center_files_only is True:
        input_file_table = get_table_with_unique_keys(
            t_center,
            column_name="OBJECT",
            check_coordinates=True,
            add_noname_objects=add_noname_objects,
        )
    else:
        input_file_table = get_table_with_unique_keys(
            table_of_files=t_center_coro,
            column_name="OBJECT",
            check_coordinates=True,
            add_noname_objects=add_noname_objects,
        )

    input_file_table.sort("MJD_OBS")
    print("Query simbad for MAIN_ID and coordinates.")
    
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
