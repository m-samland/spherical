#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "M. Samland @ MPIA (Heidelberg, Germany)"
# __all__ = []

import collections
import copy
import glob
import os
import time

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astroquery.eso import Eso
from spherical.sphere_database.database_utils import add_night_start_date
from tqdm import tqdm

header_list = collections.OrderedDict(
    [
        ("OBJECT", ("OBJECT", "N/A")),
        # Coordinates
        ("RA", ("RA", -10000)),
        ("DEC", ("DEC", -10000)),
        ("TARG_ALPHA", ("HIERARCH ESO TEL TARG ALPHA", -10000)),
        ("TARG_DELTA", ("HIERARCH ESO TEL TARG DELTA", -10000)),
        ("TARG_PMA", ("HIERARCH ESO TEL TARG PMA", -10000)),
        ("TARG_PMD", ("HIERARCH ESO TEL TARG PMD", -10000)),
        ("DROT2_RA", ("HIERARCH ESO INS4 DROT2 RA", -10000)),
        ("DROT2_DEC", ("HIERARCH ESO INS4 DROT2 DEC", -10000)),
        ("TEL GEOELEV", ("HIERARCH ESO TEL GEOELEV", -10000)),
        ("GEOLAT", ("HIERARCH ESO TEL GEOLAT", -10000)),
        ("GEOLON", ("HIERARCH ESO TEL GEOLON", -10000)),
        ("ALT_BEGIN", ("HIERARCH ESO TEL ALT", -10000)),
        ("TEL AZ", ("HIERARCH ESO TEL AZ", -10000)),
        ("TEL PARANG START", ("HIERARCH ESO TEL PARANG START", -10000)),
        ("TEL PARANG END", ("HIERARCH ESO TEL PARANG END", -10000)),
        # Detector
        ("NDIT", ("HIERARCH ESO DET NDIT", -10000)),
        ("SEQ1_DIT", ("HIERARCH ESO DET SEQ1 DIT", -10000)),
        ("EXPTIME", ("EXPTIME", -10000)),
        ("DET SEQ1 DIT", ("HIERARCH ESO DET SEQ1 DIT", -10000)),  # duplicate
        ("DET NDIT", ("HIERARCH ESO DET NDIT", -10000)),  # duplicate
        ("DIT_DELAY", ("HIERARCH ESO DET DITDELAY", -10000)),
        # CPI
        ("SEQ_ARM", ("HIERARCH ESO SEQ ARM", "N/A")),
        ("CORO", ("HIERARCH ESO INS COMB ICOR", "N/A")),
        ("DB_FILTER", ("HIERARCH ESO INS COMB IFLT", "N/A")),
        ("POLA", ("HIERARCH ESO INS COMB POLA", "N/A")),
        ("ND_FILTER", ("HIERARCH ESO INS4 FILT2 NAME", "N/A")),
        ("INS4 DROT2 MODE", ("HIERARCH ESO INS4 DROT2 MODE", "N/A")),
        ("SHUTTER", ("HIERARCH ESO INS4 SHUT ST", "N/A")),
        ("DROT2_BEGIN", ("HIERARCH ESO INS4 DROT2 BEGIN", -10000)),
        ("DROT2_END", ("HIERARCH ESO INS4 DROT2 END", -10000)),
        ("INS4 DROT2 POSANG", ("HIERARCH ESO INS4 DROT2 POSANG", -10000)),
        # IFS
        ("INS2 MODE", ("HIERARCH ESO INS2 MODE", "N/A")),
        ("IFS_MODE", ("HIERARCH ESO INS2 COMB IFS", "N/A")),
        ("PRISM", ("HIERARCH ESO INS2 OPTI2 ID", "N/A")),
        ("LAMP", ("HIERARCH ESO INS2 CAL", "N/A")),
        # IRDIS
        ("INS1 MODE", ("HIERARCH ESO INS1 MODE", "N/A")),
        ("INS1 FILT NAME", ("HIERARCH ESO INS1 FILT NAME", "N/A")),
        ("INS1 OPTI2 NAME", ("HIERARCH ESO INS1 OPTI2 NAME", "N/A")),
        ("PAC_X", ("HIERARCH ESO INS1 PAC X", -10000)),
        ("PAC_Y", ("HIERARCH ESO INS1 PAC Y", -10000)),
        # DPR
        ("DPR_CATG", ("HIERARCH ESO DPR CATG", "N/A")),
        ("DPR_TYPE", ("HIERARCH ESO DPR TYPE", "N/A")),
        ("DPR_TECH", ("HIERARCH ESO DPR TECH", "N/A")),
        ("DET_ID", ("HIERARCH ESO DET ID", "ZPL")),
        ("DEROTATOR_MODE", ("HIERARCH ESO INS4 COMB ROT", "N/A")),
        ("READOUT_MODE", ("HIERARCH ESO DET READ CURNAME", "N/A")),
        ("WAFFLE_AMP", ("HIERARCH ESO OCS WAFFLE AMPL", -10000)),
        ("WAFFLE_ORIENT", ("HIERARCH ESO OCS WAFFLE ORIENT", "N/A")),
        ("SPATIAL_FILTER", ("HIERARCH ESO INS4 OPTI22 NAME", "N/A")),
        # SAXO
        ("AOS TTLOOP STATE", ("HIERARCH ESO AOS TTLOOP STATE", "N/A")),
        ("AOS HOLOOP STATE", ("HIERARCH ESO AOS HOLOOP STATE", "N/A")),
        ("AOS IRLOOP STATE", ("HIERARCH ESO AOS IRLOOP STATE", "N/A")),
        ("AOS PUPLOOP STATE", ("HIERARCH ESO AOS PUPLOOP STATE", "N/A")),
        ("AOS VISWFS MODE", ("HIERARCH ESO AOS VISWFS MODE", "N/A")),
        # Observing conditions
        ("AMBI_FWHM_START", ("HIERARCH ESO TEL AMBI FWHM START", -10000)),
        ("AMBI_FWHM_END", ("HIERARCH ESO TEL AMBI FWHM END", -10000)),
        ("AIRMASS", ("HIERARCH ESO TEL AIRM START", -10000)),
        ("TEL AIRM END", ("HIERARCH ESO TEL AIRM END", -10000)),
        ("TEL IA FWHM", ("HIERARCH ESO TEL IA FWHM", -10000)),
        ("AMBI_TAU", ("HIERARCH ESO TEL AMBI TAU0", -10000)),
        ("TEL AMBI TEMP", ("HIERARCH ESO TEL AMBI TEMP", -10000)),
        ("TEL AMBI WINDSP", ("HIERARCH ESO TEL AMBI WINDSP", -10000)),
        ("TEL AMBI WINDDIR", ("HIERARCH ESO TEL AMBI WINDDIR", -10000)),
        # Misc
        # ('', ('HIERARCH ESO ', 'N/A')),
        ("OBS_PROG_ID", ("HIERARCH ESO OBS PROG ID", "N/A")),
        ("OBS_PI_COI", ("HIERARCH ESO OBS PI-COI NAME", "N/A")),
        ("OBS_ID", ("HIERARCH ESO OBS ID", -10000)),
        ("OBS_NAME", ("HIERARCH ESO OBS NAME", -10000)),
        ("ORIGFILE", ("ORIGFILE", "N/A")),
        ("DATE", ("DATE", "N/A")),
        ("DATE_OBS", ("DATE-OBS", "N/A")),
        ("SEQ_UTC", ("HIERARCH ESO DET SEQ UTC", "N/A")),
        ("FRAM_UTC", ("HIERARCH ESO DET FRAM UTC", "N/A")),
        ("MJD_OBS", ("MJD-OBS", -10000)),
        ("DP.ID", ("DP.ID", "N/A")),
    ]
)

keep_columns = []
for key in header_list:
    keep_columns.append(header_list[key][0])
keep_columns = list(np.unique(keep_columns))

keep_columns_set = set(keep_columns)


def make_master_file_table(folder, file_ending='myrun',
                           start_date=None, end_date=None,
                           cache=True, save=True, savefull=False,
                           existing_master_file_table_path=None):
    """
    Generate a master table from input file containing list of filenames for calibration and science files.

    Parameters:
        folder (str): The folder path where the files are located.
        file_ending (str, optional): The file ending to be used for the output files. Defaults to 'myrun'.
        existing_master_file_table_path (str, optional): The path to the existing master file table. 
                    If provided, only download headers of new files. Defaults to None.

    Returns:
        pandas.DataFrame: The generated file table.

    """

    if existing_master_file_table_path is not None:
        previous_master_file_table = pd.read_csv(existing_master_file_table_path)
        existing_entries = set(previous_master_file_table["DP.ID"].values)

    eso = Eso()
    eso.ROW_LIMIT = 10000000
    # Set search box to 360 degree
    # Set SEQ_ARM to 'IFS' for IFS or 'IRDIS' for IRDIS
    # Set DPR CATG to 'SCIENCE' for science or 'CALIB' for calibration
    # For IFS only calib needed is DPR TYPE = 'WAVE,LAMP'

    calibration_columns = {
        'dp_cat': 'CALIB',
        'dp_type': "WAVE,LAMP",
        'box':360,
        'seq_arm': 'IFS'}
    
    science_columns = {
        'dp_cat': 'SCIENCE',
        'box':360,
        'seq_arm': 'IFS'}
    
    if start_date is not None:
        calibration_columns['stime'] = start_date
        science_columns['stime'] = start_date
    if end_date is not None:
        calibration_columns['etime'] = end_date
        science_columns['etime'] = end_date    

    dp_ids_calib = list(eso.query_instrument(
        instrument='sphere',
        column_filters=calibration_columns)['DP.ID'].value)
    dp_ids_sci = list(eso.query_instrument(
        instrument='sphere',
        column_filters=science_columns)['DP.ID'].value)

    dp_ids = dp_ids_sci + dp_ids_calib
    
    # If previous master file table is given, only download headers of new files
    if existing_master_file_table_path is not None:
        dp_ids = set(dp_ids) - existing_entries
        dp_ids = list(dp_ids)

    time0 = time.perf_counter()
    hdrs = eso.get_headers(product_ids=dp_ids, cache=cache)
    time1 = time.perf_counter()
    print(f"Runtime so far: {time1 - time0:0.4f} seconds")

    if savefull:
        hdrs.to_pandas().to_csv(os.path.join(folder, f"table_of_files_{file_ending}_allheader.csv"), index=False)

    col_hdrs_set = set(hdrs.keys())
    remove_cols = list(col_hdrs_set.difference(keep_columns_set))

    hdrs_df = hdrs.to_pandas()
    hdrs_df = hdrs_df[hdrs_df.columns[~hdrs_df.columns.isin(remove_cols)]]

    hdrs_spherical = hdrs_df.copy()
    for key in header_list:
        try:
            hdrs_spherical.rename(columns={header_list[key][0]: key}, inplace=True)
        except:
            continue

    date_short = copy.deepcopy(hdrs_spherical["DATE_OBS"])
    for i in range(len(date_short)):
        date_short[i] = date_short[i][0:10]
    hdrs_spherical["DATE_SHORT"] = date_short
    hdrs_spherical["FILE_SIZE"] = 1.0

    hdrs_spherical_astropy = Table.from_pandas(hdrs_spherical)
    hdrs_spherical_astropy = add_night_start_date(hdrs_spherical_astropy, key="DATE_OBS")
    
    hdrs_spherical_df = hdrs_spherical_astropy.to_pandas()

    if existing_master_file_table_path is not None:
        hdrs_spherical_df = pd.concat([previous_master_file_table, hdrs_spherical_df], axis=0)
        hdrs_spherical_astropy = Table.from_pandas(hdrs_spherical_df)
    
    if save:
        hdrs_spherical_df.to_csv(os.path.join(folder, f"table_of_files_{file_ending}.csv"), index=False)

    return hdrs_spherical_astropy


def get_headerval_with_exeption(header, keyword, default_val):
    try:
        val = header[keyword]
    except KeyError:
        val = default_val
    except IOError:
        val = default_val
    return val


def make_master_file_table_old(raw_path, recursive, glob_search_pattern="SPHER.*.fits"):
    """Reads in all fits file containing the search pattern
    in the raw_path. Subfolders are included with recursive keyword.

    """

    start = time.time()
    # type_keys = ['DARK', 'DARK,BACKGROUND', 'FLAT,LAMP', 'LAMP,DISTORT', 'OBJECT', 'OBJECT,CENTER', 'OBJECT,FLUX', 'SKY']
    if recursive is False:
        files = [
            os.path.join(raw_path, f)
            for f in os.listdir(raw_path)
            if f.endswith(".fits")
        ]
    else:
        files = [
            y
            for x in os.walk(raw_path)
            for y in glob.glob(os.path.join(x[0], glob_search_pattern))
        ]
    # Add some pre-filtering for files here.
    # print(files)
    file_list = []
    for f in files:
        if "SPARTA" not in f:
            if "RAW_" not in f:
                file_list.append(f)
    files = file_list

    header_list = collections.OrderedDict(
        [
            ("OBJECT", ("OBJECT", "N/A")),
            # Coordinates
            ("RA", ("RA", -10000)),
            ("DEC", ("DEC", -10000)),
            ("TARG_ALPHA", ("HIERARCH ESO TEL TARG ALPHA", -10000)),
            ("TARG_DELTA", ("HIERARCH ESO TEL TARG DELTA", -10000)),
            ("TARG_PMA", ("HIERARCH ESO TEL TARG PMA", -10000)),
            ("TARG_PMD", ("HIERARCH ESO TEL TARG PMD", -10000)),
            ("DROT2_RA", ("HIERARCH ESO INS4 DROT2 RA", -10000)),
            ("DROT2_DEC", ("HIERARCH ESO INS4 DROT2 DEC", -10000)),
            ("TEL GEOELEV", ("HIERARCH ESO TEL GEOELEV", -10000)),
            ("GEOLAT", ("HIERARCH ESO TEL GEOLAT", -10000)),
            ("GEOLON", ("HIERARCH ESO TEL GEOLON", -10000)),
            ("ALT_BEGIN", ("HIERARCH ESO TEL ALT", -10000)),
            ("TEL AZ", ("HIERARCH ESO TEL AZ", -10000)),
            # Detector
            ("NDIT", ("HIERARCH ESO DET NDIT", -10000)),
            ("SEQ1_DIT", ("HIERARCH ESO DET SEQ1 DIT", -10000)),
            ("EXPTIME", ("EXPTIME", -10000)),
            ("DET SEQ1 DIT", ("HIERARCH ESO DET SEQ1 DIT", -10000)),  # duplicate
            ("DET NDIT", ("HIERARCH ESO DET NDIT", -10000)),  # duplicate
            ("DIT_DELAY", ("HIERARCH ESO DET DITDELAY", -10000)),
            # CPI
            ("SEQ_ARM", ("HIERARCH ESO SEQ ARM", "N/A")),
            ("CORO", ("HIERARCH ESO INS COMB ICOR", "N/A")),
            ("DB_FILTER", ("HIERARCH ESO INS COMB IFLT", "N/A")),
            ("POLA", ("HIERARCH ESO INS COMB POLA", "N/A")),
            ("ND_FILTER", ("HIERARCH ESO INS4 FILT2 NAME", "N/A")),
            ("INS4 DROT2 MODE", ("HIERARCH ESO INS4 DROT2 MODE", "N/A")),
            ("SHUTTER", ("HIERARCH ESO INS4 SHUT ST", "N/A")),
            ("DROT2_BEGIN", ("HIERARCH ESO INS4 DROT2 BEGIN", -10000)),
            ("DROT2_END", ("HIERARCH ESO INS4 DROT2 END", -10000)),
            ("INS4 DROT2 POSANG", ("HIERARCH ESO INS4 DROT2 POSANG", -10000)),
            # IFS
            ("INS2 MODE", ("HIERARCH ESO INS2 MODE", "N/A")),
            ("IFS_MODE", ("HIERARCH ESO INS2 COMB IFS", "N/A")),
            ("PRISM", ("HIERARCH ESO INS2 OPTI2 ID", "N/A")),
            ("LAMP", ("HIERARCH ESO INS2 CAL", "N/A")),
            # IRDIS
            ("INS1 MODE", ("HIERARCH ESO INS1 MODE", "N/A")),
            ("INS1 FILT NAME", ("HIERARCH ESO INS1 FILT NAME", "N/A")),
            ("INS1 OPTI2 NAME", ("HIERARCH ESO INS1 OPTI2 NAME", "N/A")),
            ("PAC_X", ("HIERARCH ESO INS1 PAC X", -10000)),
            ("PAC_Y", ("HIERARCH ESO INS1 PAC Y", -10000)),
            # DPR
            ("DPR_CATG", ("HIERARCH ESO DPR CATG", "N/A")),
            ("DPR_TYPE", ("HIERARCH ESO DPR TYPE", "N/A")),
            ("DPR_TECH", ("HIERARCH ESO DPR TECH", "N/A")),
            ("DET_ID", ("HIERARCH ESO DET ID", "ZPL")),
            ("DEROTATOR_MODE", ("HIERARCH ESO INS4 COMB ROT", "N/A")),
            ("READOUT_MODE", ("HIERARCH ESO DET READ CURNAME", "N/A")),
            ("WAFFLE_AMP", ("HIERARCH ESO OCS WAFFLE AMPL", -10000)),
            ("WAFFLE_ORIENT", ("HIERARCH ESO OCS WAFFLE ORIENT", "N/A")),
            ("SPATIAL_FILTER", ("HIERARCH ESO INS4 OPTI22 NAME", "N/A")),
            # SAXO
            ("AOS TTLOOP STATE", ("HIERARCH ESO AOS TTLOOP STATE", "N/A")),
            ("AOS HOLOOP STATE", ("HIERARCH ESO AOS HOLOOP STATE", "N/A")),
            ("AOS IRLOOP STATE", ("HIERARCH ESO AOS IRLOOP STATE", "N/A")),
            ("AOS PUPLOOP STATE", ("HIERARCH ESO AOS PUPLOOP STATE", "N/A")),
            ("AOS VISWFS MODE", ("HIERARCH ESO AOS VISWFS MODE", "N/A")),
            # Observing conditions
            ("AMBI_FWHM_START", ("HIERARCH ESO TEL AMBI FWHM START", -10000)),
            ("AMBI_FWHM_END", ("HIERARCH ESO TEL AMBI FWHM END", -10000)),
            ("AIRMASS", ("HIERARCH ESO TEL AIRM START", -10000)),
            ("TEL AIRM END", ("HIERARCH ESO TEL AIRM END", -10000)),
            ("TEL IA FWHM", ("HIERARCH ESO TEL IA FWHM", -10000)),
            ("AMBI_TAU", ("HIERARCH ESO TEL AMBI TAU0", -10000)),
            ("TEL AMBI TEMP", ("HIERARCH ESO TEL AMBI TEMP", -10000)),
            ("TEL AMBI WINDSP", ("HIERARCH ESO TEL AMBI WINDSP", -10000)),
            ("TEL AMBI WINDDIR", ("HIERARCH ESO TEL AMBI WINDDIR", -10000)),
            # Misc
            # ('', ('HIERARCH ESO ', 'N/A')),
            ("OBS_PROG_ID", ("HIERARCH ESO OBS PROG ID", "N/A")),
            ("OBS_PI_COI", ("HIERARCH ESO OBS PI-COI NAME", "N/A")),
            ("OBS_ID", ("HIERARCH ESO OBS ID", -10000)),
            ("OBS_NAME", ("HIERARCH ESO OBS NAME", -10000)),
            ("ORIGFILE", ("ORIGFILE", "N/A")),
            ("DATE", ("DATE", "N/A")),
            ("DATE_OBS", ("DATE-OBS", "N/A")),
            ("SEQ_UTC", ("HIERARCH ESO DET SEQ UTC", "N/A")),
            ("FRAM_UTC", ("HIERARCH ESO DET FRAM UTC", "N/A")),
            ("MJD_OBS", ("MJD-OBS", -10000)),
        ]
    )

    # Shutter. T: open F: closed
    file_sizes = []
    non_corrupt_files = []
    bad_files = []

    header_result_table = collections.OrderedDict()
    for key in header_list.keys():
        header_result_table[key] = []

    for f in tqdm(files):
        try:
            orig_hdr = fits.getheader(f)
            header = orig_hdr.copy()
            # hdulist = fits.open(f)
            # header = hdulist[0].header
            for key in header_list.keys():
                if key == "HIERARCH ESO INS4 DROT2 BEGIN":
                    # in June 2021 ESO changed INS4 DROT2 BEGIN to INS4 DROT2 START
                    v_begin = header.get("HIERARCH ESO INS4 DROT2 BEGIN")
                    v_start = header.get("HIERARCH ESO INS4 DROT2 START")
                    header[key] = v_begin if v_begin else v_start
                    if v_begin is None:
                        header[key] = -10000
                elif key == "HIERARCH ESO INS4 DROT3 BEGIN":
                    # in June 2021 ESO changed INS3 DROT3 BEGIN to INS4 DROT3 START
                    v_begin = header.get("HIERARCH ESO INS3 DROT2 BEGIN")
                    v_start = header.get("HIERARCH ESO INS3 DROT2 START")
                    header[key] = v_begin if v_begin else v_start
                    if v_begin is None:
                        header[key] = -10000
                else:
                    header[key] = get_headerval_with_exeption(
                        orig_hdr, header_list[key][0], header_list[key][1]
                    )

            for key in header_list.keys():
                header_result_table[key].append(
                    get_headerval_with_exeption(
                        header, header_list[key][0], header_list[key][1]
                    )
                )
            file_sizes.append(os.path.getsize(f))
            non_corrupt_files.append(f)
        except IOError:
            bad_files.append(f)
        # print('Read in File: {}'.format(f))

    # Add list of files names as first entry to the ordered dictionary
    header_result_table.update({"FILE": non_corrupt_files})
    header_result_table.move_to_end("FILE", last=False)

    date_short = copy.deepcopy(header_result_table["DATE_OBS"])
    for i in range(len(date_short)):
        date_short[i] = date_short[i][0:10]
    header_result_table.update({"DATE_SHORT": date_short})

    file_sizes = np.array(file_sizes) / 1e6
    header_result_table.update({"FILE_SIZE": file_sizes})
    table_of_files = Table(header_result_table)
    # Sort frames according to order in which they were taken
    table_of_files.sort("MJD_OBS")
    # Filter out first part of file name giving date and instrument in order to remove files
    # that have been imported multiple times with different names

    # Pre-filtering file table
    mask_zero_file_size = table_of_files["FILE_SIZE"] > 0
    bad_files.append(table_of_files[~mask_zero_file_size].copy())
    table_of_files = table_of_files[mask_zero_file_size]
    # Remove duplicates from list
    lenght_of_shortstring = len("SPHER.2016-01-19T11_28_15.700IFS")
    short_list = []
    for file in table_of_files["FILE"]:
        short_list.append(os.path.basename(file)[:32])
    file_arr = np.array(short_list)
    mask_unique = np.zeros(len(file_arr), dtype=bool)
    u_arr, u_idx = np.unique(file_arr, return_index=True)
    mask_unique[u_idx] = True

    table_of_files = table_of_files[mask_unique]

    table_of_files = add_night_start_date(table_of_files, key="DATE_OBS")

    print("RAW Data that produced errors: ")
    print(bad_files)
    bad_files = np.array(bad_files)
    end = time.time()
    print("Time elapsed (minutes): {0}".format((end - start) / 60.0))

    return table_of_files, bad_files
