#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "M. Samland @ MPIA (Heidelberg, Germany)"
__all__ = [
    "filter_for_IRDIS_science_frames",
    "get_table_with_unique_keys",
    "retry_query",
    "correct_for_proper_motion",
    "query_SIMBAD_for_names",
    "make_IRDIS_target_list_with_SIMBAD",
]

import re
import time

import healpy as hp
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
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

    # fk5_obs = FK5(equinox='J{0:.3f}'.format(time_of_observation.jyear))
    # coordinates.transform_to(fk5_obs)
    # ipsh()
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
    J_mag_limit=15,
    verbose=False,
):
    """Short summary.

    Parameters
    ----------
    table_of_files : type
        Table containing 'OBJECT', 'RA', and 'DEC' keywords.
    search_radius : type
        Search radius in arcminutes.
    number_of_retries : type
        Number of times to repeat any query upon failure.

    Returns
    -------
    type
        Description of returned object.

    """
    # Takes the master table of files (or subset thereof) and queries
    # simbad with the 'RA' and 'DEC' keys.
    # Returns a table containing the resulting MAIN_ID of the target,
    # coordinates, and deviation from the search coordinates.

    # Simbad.ROW_LIMIT = 10
    # Simbad.TIMEOUT = 60
    not_found_list = []
    # object_list = []
    objects_in_cone = []
    separation_nearest = []  # in arcsec
    nearest_star = []
    nearest_star_otype = []
    nearest_star_stype = []
    nearest_star_flux_h = []
    companion_name = []
    number_of_companions = []
    search_separation = []
    search_separation_orig = []
    _counter = 0
    # header_ra = []
    # header_dec = []

    search_radius = search_radius * u.arcminute

    customSimbad = Simbad()
    customSimbad.ROW_LIMIT = 15
    customSimbad.TIMEOUT = 60
    customSimbad.add_votable_fields(
        "id(HD)",
        "id(HIP)",
        # "ids",
        "id(Gaia DR3)",
        "flux(V)",
        "flux(R)",
        "flux(I)",
        "flux(J)",
        "flux(H)",
        "flux(K)",
        "otype",
        "otype(V)",
        "otype(3)",
        "sptype",
        "pm",
        "parallax",
        "rv_value",
        "rvz_error",
        "fe_h",
    )

    failed = np.zeros(len(table_of_files), dtype="bool")
    for idx, (OBJECT, RA, DEC, MJD) in enumerate(
        tqdm(table_of_files["OBJECT", "RA", "DEC", "MJD_OBS"])
    ):
        observed_coords = SkyCoord(
            ra=RA * u.degree, dec=DEC * u.degree
        )  # , frame='fk5')
        time_of_observation = Time(MJD, format="mjd")

        # Query region
        query_result = retry_query(
            customSimbad.query_region,
            verbose=verbose,
            number_of_retries=number_of_retries,
            coordinates=observed_coords,
            radius=search_radius,
        )

        if query_result is None:
            # If region failed, query object name
            print("Trying with OBJECT name instead.")
            query_result = retry_query(
                customSimbad.query_object,
                verbose=verbose,
                number_of_retries=number_of_retries,
                object_name=OBJECT,
            )

        if query_result is None or len(query_result) == 0:
            not_found_list.append((OBJECT, RA, DEC, MJD))
            print("Not found: {}".format(OBJECT))
            failed[idx] = True
            continue

        photometric_values = query_result[
            "FLUX_V", "FLUX_R", "FLUX_I", "FLUX_J", "FLUX_H", "FLUX_K"
        ].to_pandas()

        fluxes_exist_mask = ~np.all(~np.isfinite(photometric_values.values), axis=1)

        J_mag_mask = photometric_values.values[:, 3] < J_mag_limit
        photometric_preselection_mask = np.logical_and(fluxes_exist_mask, J_mag_mask)

        query_result = query_result[photometric_preselection_mask]

        if len(query_result) == 1:
            primary_index = 0
            objects_in_cone.append(0)
            separation_nearest.append(0)
            nearest_star.append("none")
            number_of_companions.append(0)
            companion_name.append("none")
            nearest_star_otype.append("none")
            nearest_star_stype.append("none")
            nearest_star_flux_h.append("none")

            # Make input for proper motion correction
            queried_coordinates = SkyCoord(
                ra=query_result["RA"],
                dec=query_result["DEC"],
                unit=(u.hourangle, u.deg),
            )

            pm_ra = query_result["PMRA"]
            pm_dec = query_result["PMDEC"]

            corrected_coordinates = correct_for_proper_motion(
                queried_coordinates, pm_ra, pm_dec, time_of_observation, sign="positive"
            )

            separation_to_obs = corrected_coordinates.separation(observed_coords).to(
                u.arcsec
            )

            separation_orig = queried_coordinates.separation(observed_coords).to(
                u.arcsec
            )

            search_separation.append(np.nanmin(separation_to_obs).value)
            search_separation_orig.append(np.nanmin(separation_orig).value)

        elif len(query_result) > 1:
            # check for planets
            
            companion_indices = []
            mask_companion = np.zeros(len(query_result), dtype="bool")
            for idx in range(len(mask_companion)):
                ids = query_result[idx]["MAIN_ID"]  # .decode('utf-8')
                if matches_pattern(ids):
                    mask_companion[idx] = True
                    companion_indices.append(idx)

            found_companions = len(companion_indices)
            number_of_companions.append(found_companions)
            if found_companions == 0:
                companion_name.append("none")
            else:
                # In table only one companion name will show up
                companion_name.append(
                    query_result[companion_indices[0]]["MAIN_ID"]
                )  # .decode('utf-8'))

            # Only look for "star-like" objects
            # http://simbad.u-strasbg.fr/simbad/sim-display?data=otypes
            # allowed_object_types = [
            #     '*', 'star', 'Star', 'YSO',
            #     'BlueStraggler', 'HotSubdwarf',
            #     'OH/IR', 'CH', 'FUOr',
            #     'RCrB_Candidate',
            #     'Pulsar', 'BYDra',
            #     'RSCVn', 'RRLyr',
            #     'Cepheid', 'deltaCep',
            #     'gammaDor', 'Mira', 'SN']
            allowed_object_types = ["*"]

            mask_non_stars = np.zeros(len(query_result), dtype="bool")
            for idx in range(len(mask_non_stars)):
                object_type = query_result["OTYPE_3"][idx]  # .decode('utf-8')
                if not any(map(object_type.__contains__, allowed_object_types)):
                    mask_non_stars[idx] = True
                # Do not include double star summary entries
                if "**" in object_type:
                    mask_non_stars[idx] = True
            # Exclude BD from being selected as primary
            mask_non_stars_and_bd = mask_non_stars.copy()
            for idx in range(len(query_result)):
                object_type = query_result["OTYPE"][idx]  # .decode('utf-8')
                if "brown" in object_type:
                    mask_non_stars_and_bd[idx] = True

            queried_coordinates = SkyCoord(
                ra=query_result["RA"],
                dec=query_result["DEC"],
                unit=(u.hourangle, u.deg),
            )

            pm_ra = query_result["PMRA"]
            pm_dec = query_result["PMDEC"]

            corrected_coordinates = correct_for_proper_motion(
                queried_coordinates, pm_ra, pm_dec, time_of_observation, sign="positive"
            )

            # Identify primary as closest to observed position
            separation_to_obs = corrected_coordinates.separation(observed_coords).to(
                u.arcsec
            )

            separation_orig = queried_coordinates.separation(observed_coords).to(
                u.arcsec
            )

            masked_separations = np.ma.masked_array(
                separation_to_obs.value,
                mask=np.logical_or(mask_non_stars_and_bd, mask_companion),
            )

            masked_separations_orig = np.ma.masked_array(
                separation_orig.value,
                mask=np.logical_or(mask_non_stars_and_bd, mask_companion),
            )

            primary_index = np.nanargmin(masked_separations)
            mask_non_stars[primary_index] = True  # Exclude primary further
            search_separation.append(np.nanmin(masked_separations))
            search_separation_orig.append(np.nanmin(masked_separations_orig))

            separation_in_field = (
                queried_coordinates[primary_index]
                .separation(queried_coordinates)
                .to(u.arcsec)
            )

            # Exclude probable duplicate simbad entries with slightly displace
            # position but different name, beyond that, even if confusion
            # below 0.35 arcsec happens, it does not matter for us, we only
            # care about avoiding confusion in crowded fields with other stars
            close_binary = separation_in_field < 0.35 * u.arcsec
            mask_non_stars[close_binary] = True
            objects_in_cone.append(np.nansum(~mask_non_stars))

            # Include BD in search for nearest. This search, in rare cases this can
            # result in a low mass stars designed as otype star, but with a
            # designation of 'b' at the end... Wrong entries in simbad
            masked_separation_in_field = np.ma.masked_array(
                separation_in_field.value, mask=mask_non_stars
            )

            if np.sum(~mask_non_stars) > 0:
                separation_to_closest_object = np.nanmin(masked_separation_in_field)
                nearest_star_index = np.nanargmin(masked_separation_in_field)
                separation_nearest.append(separation_to_closest_object)
                nearest_star.append(
                    query_result["MAIN_ID"][nearest_star_index]
                )  # .decode('utf-8'))
                nearest_star_otype.append(
                    query_result["OTYPE"][nearest_star_index]
                )  # .decode('utf-8'))
                nearest_star_stype.append(
                    query_result["SP_TYPE"][nearest_star_index]
                )  # .decode('utf-8'))
                near_flux_h = query_result["FLUX_H"][nearest_star_index]
                if np.isfinite(near_flux_h):
                    nearest_star_flux_h.append(
                        query_result["FLUX_H"][nearest_star_index]
                    )
                else:
                    nearest_star_flux_h.append(-10000)

            else:
                separation_nearest.append(0)
                nearest_star.append("none")
                nearest_star_otype.append("none")
                nearest_star_stype.append("none")
                nearest_star_flux_h.append("none")

        if verbose:
            print("Queried object: {}".format(OBJECT))
            print("Retrieved object: {}".format(query_result["MAIN_ID"][primary_index]))

        try:
            main_row = query_result[
                "MAIN_ID",
                "ID_HD",
                "ID_HIP",
                "ID_Gaia_DR3",
                "RA",
                "DEC",
                "OTYPE",
                "SP_TYPE",
                "FLUX_V",
                "FLUX_R",
                "FLUX_I",
                "FLUX_J",
                "FLUX_H",
                "FLUX_K",
                "PMRA",
                "PMDEC",
                "PM_ERR_MAJA",
                "PM_ERR_MINA",
                "PLX_VALUE",
                "PLX_ERROR",
                "PLX_BIBCODE",
                "RV_VALUE",
                "RVZ_ERROR",
                "Fe_H_Fe_H"
            ][primary_index]
        except IndexError:
            not_found_list.append((OBJECT, RA, DEC, MJD))
            print("Not found: {}".format(OBJECT))
            failed[idx] = True
            continue

        dtypes = []
        for i in range(len(main_row.dtype)):
            dtypes.append(main_row.dtype[i])

        if _counter == 0:  # Create table from one row for first iteration
            simbad_table = Table(
                rows=main_row, names=main_row.colnames, dtype=dtypes, meta=main_row.meta
            )
            _counter += 1
        else:
            simbad_table.add_row(main_row)

    # Add additional columns
    absolute_proper_motion = np.sqrt(
        simbad_table["PMRA"] ** 2 + simbad_table["PMDEC"] ** 2
    )
    col_absolute_proper_motion = Table.MaskedColumn(
        name="ABS_PM", data=absolute_proper_motion, unit=u.mas / u.yr
    )
    col_stars_in_cone = Table.Column(name="STARS_IN_CONE", data=objects_in_cone)
    col_nearest_star = Table.Column(name="NEAREST_STAR", data=nearest_star)
    col_separation_nearest = Table.Column(
        name="SEPARATION_NEAREST", data=separation_nearest, unit=u.arcsec
    )
    col_nearest_otype = Table.Column(name="NEAREST_OTYPE", data=nearest_star_otype)
    col_nearest_stype = Table.Column(name="NEAREST_STYPE", data=nearest_star_stype)
    col_nearest_flux_h = Table.Column(name="NEAREST_FLUX_H", data=nearest_star_flux_h)
    col_companion_name = Table.Column(name="COMPANION_NAME", data=companion_name)
    col_number_of_companions = Table.Column(
        name="NUMBER_COMPANIONS", data=number_of_companions
    )

    # Transform coordinates from hexagesimal to degree
    ra = []
    dec = []
    for i in range(len(simbad_table)):
        ra.append(simbad_table["RA"][i])
        dec.append(simbad_table["DEC"][i])

    list_of_coords = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.degree))
    list_of_coords_header = SkyCoord(
        ra=table_of_files["RA"], dec=table_of_files["DEC"], unit=(u.degree, u.degree)
    )
    # search_separation = list_of_coords.separation(list_of_coords_header).to(u.arcsec)

    col_OBJ_header = Table.Column(
        name="OBJ_HEADER", data=table_of_files["OBJECT"][~failed]
    )
    col_RA_deg = Table.Column(
        name="RA_DEG", data=list_of_coords.ra.value, unit=u.degree
    )
    col_DEC_deg = Table.Column(
        name="DEC_DEG", data=list_of_coords.dec.value, unit=u.degree
    )
    col_RA_header = Table.Column(
        name="RA_HEADER", data=table_of_files["RA"][~failed], unit=u.degree
    )
    col_DEC_header = Table.Column(
        name="DEC_HEADER", data=table_of_files["DEC"][~failed], unit=u.degree
    )
    col_position_diff = Table.Column(
        name="POS_DIFF", data=search_separation, unit=u.arcsec
    )
    col_position_diff_orig = Table.Column(
        name="POS_DIFF_ORIG", data=search_separation_orig, unit=u.arcsec
    )
    # col_alpha = Table.Column(name='TARG_Alpha', data=table_of_files['TARG_ALPHA'])
    # col_delta = Table.Column(name='TARG_Delta', data=table_of_files['TARG_DELTA'])
    # col_PMA = Table.Column(name='PM_Alpha', data=table_of_files['TARG_PMA'], unit=u.arcsec / u.year)
    # col_PMD = Table.Column(name='PM_Delta', data=table_of_files['TARG_PMA'], unit=u.arcsec / u.year)

    simbad_table.add_column(col_absolute_proper_motion, 13)
    simbad_table.add_column(col_OBJ_header)
    simbad_table.add_column(col_RA_deg)
    simbad_table.add_column(col_DEC_deg)
    simbad_table.add_column(col_RA_header)
    simbad_table.add_column(col_DEC_header)
    simbad_table.add_column(col_position_diff)
    simbad_table.add_column(col_position_diff_orig)
    # simbad_table.add_column(col_alpha)
    # simbad_table.add_column(col_delta)
    # simbad_table.add_column(col_PMA)
    # simbad_table.add_column(col_PMD)

    # simbad_table = vstack(object_list)
    simbad_table.add_column(col_stars_in_cone)
    simbad_table.add_column(col_nearest_star)
    simbad_table.add_column(col_separation_nearest)
    simbad_table.add_column(col_nearest_otype)
    simbad_table.add_column(col_nearest_stype)
    simbad_table.add_column(col_nearest_flux_h)
    simbad_table.add_column(col_companion_name)
    simbad_table.add_column(col_number_of_companions)

    simbad_table.rename_column("ID_HD", "HD_ID")
    simbad_table.rename_column("ID_HIP", "HIP_ID")
    simbad_table.rename_column("ID_Gaia_DR3", "Gaia_ID")
    simbad_table.rename_column("PMRA", "PM_RA")
    simbad_table.rename_column("PMDEC", "PM_DEC")
    simbad_table.rename_column("PM_ERR_MAJA", "PM_RA_ERR")
    simbad_table.rename_column("PM_ERR_MINA", "PM_DEC_ERR")
    simbad_table.rename_column("PLX_VALUE", "PLX")
    simbad_table.rename_column("PLX_ERROR", "PLX_ERR")
    simbad_table.rename_column("Fe_H_Fe_H", "Fe_H")
    simbad_table.rename_column("RV_VALUE", "RV")
    simbad_table.rename_column("RVZ_ERROR", "RV_ERR")

    simbad_table["MAIN_ID"] = simbad_table["MAIN_ID"].astype(str)
    simbad_table["HD_ID"] = simbad_table["HD_ID"].astype(str)
    simbad_table["HIP_ID"] = simbad_table["HIP_ID"].astype(str)
    simbad_table["NEAREST_STAR"] = simbad_table["NEAREST_STAR"].astype(str)
    simbad_table["COMPANION_NAME"] = simbad_table["COMPANION_NAME"].astype(str)
    simbad_table["OTYPE"] = simbad_table["OTYPE"].astype(str)
    simbad_table["PLX_BIBCODE"] = simbad_table["PLX_BIBCODE"].astype(str)
    simbad_table["SP_TYPE"] = simbad_table["SP_TYPE"].astype(str)
    simbad_table['Gaia_ID'] = simbad_table['Gaia_ID'].astype(str)

    simbad_table["RA"].unit = None
    simbad_table["DEC"].unit = None

    return simbad_table, not_found_list


def make_target_list_with_SIMBAD(
    table_of_files,
    instrument,
    search_radius=3.0,
    J_mag_limit=9.0,
    number_of_retries=3,
    remove_fillers=True,
    use_center_files_only=False,
    check_coordinates=True,
    add_noname_objects=True,
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
    # ipsh()
    observed_coords = SkyCoord(
        ra=t_center_coro["RA"] * u.degree, dec=t_center_coro["DEC"] * u.degree
    )
    phi = observed_coords.ra.radian
    theta = observed_coords.dec.radian + np.pi / 2.0
    nside = 32768  # 2^15
    print((hp.nside2resol(nside) * u.radian).to(u.arcsec))

    pixel_indices = hp.ang2pix(nside, theta, phi)
    t_center_coro["healpix_idx"] = pixel_indices

    # Minimum requirement for being one sequence: same 'OBJECT' keyword.
    # The set of unique object keywords should be larger than the set of unique real target names.
    if use_center_files_only is True:
        table_of_targets = get_table_with_unique_keys(
            t_center,
            column_name="OBJECT",
            check_coordinates=True,
            add_noname_objects=add_noname_objects,
        )
    else:
        table_of_targets = get_table_with_unique_keys(
            t_center_coro,
            column_name="OBJECT",
            check_coordinates=True,
            add_noname_objects=add_noname_objects,
        )

    table_of_targets.sort("MJD_OBS")
    print("Query simbad for MAIN_ID and coordinates.")
    simbad_table, not_found_list = query_SIMBAD_for_names(
        table_of_targets,
        search_radius=search_radius,
        J_mag_limit=J_mag_limit,
        number_of_retries=number_of_retries,
        verbose=verbose,
    )
    # Remove multiple spaces from MAIN_ID
    for idx, ID in enumerate(simbad_table["MAIN_ID"]):
        simbad_table[idx]["MAIN_ID"] = " ".join(ID.split())

    unique_ids, unique_indices, number_of_observations = np.unique(
        simbad_table["MAIN_ID"], return_index=True, return_counts=True
    )
    table_of_targets = simbad_table[unique_indices]

    from spherical.sphere_database.database_utils import replace_substrings, query_gspphot

    # Query Gaia astrophysical parameters
    table_of_targets_df = table_of_targets.to_pandas()
    table_of_targets_df = replace_substrings(table_of_targets_df, 'Gaia_ID', 'Gaia DR2 ', '')
    table_of_targets_df = replace_substrings(table_of_targets_df, 'Gaia_ID', 'Gaia DR3 ', '')
    target_ids = np.unique(table_of_targets_df['Gaia_ID'].astype('str').values)

    print("Querying Gaia astrophysical parameters...")
    gaia_param = query_gspphot(target_ids)
    gaia_param['SOURCE_ID'] = gaia_param['SOURCE_ID'].astype(str)
    table_of_targets_df = pd.merge(table_of_targets_df, gaia_param, how='left', left_on='Gaia_ID', right_on='SOURCE_ID')
    table_of_targets_df.drop('SOURCE_ID', axis=1, inplace=True)
    table_of_targets = Table.from_pandas(table_of_targets_df)

    table_of_targets['RA_DEG'] = table_of_targets['RA_DEG'] * u.degree
    table_of_targets['DEC_DEG'] = table_of_targets['DEC_DEG'] * u.degree
    table_of_targets['RA_HEADER'] = table_of_targets['RA_HEADER'] * u.degree
    table_of_targets['DEC_HEADER'] = table_of_targets['DEC_HEADER'] * u.degree
    table_of_targets['POS_DIFF'] = table_of_targets['POS_DIFF'] * u.arcsec
    table_of_targets['POS_DIFF_ORIG'] = table_of_targets['POS_DIFF_ORIG'] * u.arcsec   

    table_of_targets['DISTANCE'] = 1. / (1e-3 * table_of_targets['PLX'].data) * u.pc

    return table_of_targets, not_found_list
