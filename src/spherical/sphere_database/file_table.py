#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "M. Samland @ MPIA (Heidelberg, Germany)"

import collections
import datetime
import logging
import os
import time

import numpy as np
import pandas as pd
from astropy.io.fits import Header
from astropy.table import Table
from astroquery.eso import Eso
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from spherical.sphere_database.database_utils import add_night_start_date, compute_fits_header_data_size

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('spherical.file_table')

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
        ("FILE_SIZE", ("FILE_SIZE", 0.)),
    ]
)

keep_columns = []
for key in header_list:
    keep_columns.append(header_list[key][0])
keep_columns = list(np.unique(keep_columns))
keep_columns_set = set(keep_columns)


def make_file_table(folder, file_ending='myrun',
                           start_date=None, end_date=None,
                           cache=True, save=True,
                           existing_file_table_path=None,
                           batch_size=100, date_batch_months=1):
    """
    Create a file table of SPHERE science and calibration observations by retrieving 
    and parsing ESO archive headers.

    This function queries the ESO archive for all SPHERE IFS (and optionally IRDIS) files 
    within a given date range and collects their metadata headers. It extracts only a 
    predefined set of relevant header keywords (defined in `header_list`) to simplify 
    later processing and analysis in the spherical pipeline.

    It supports incremental updating of existing file tables, caching, and batch querying
    over long date ranges.

    Parameters
    ----------
    folder : str
        Directory where the output table (CSV) will be saved.
    
    file_ending : str, optional
        Suffix for naming the output file. Default is 'myrun'.
    
    start_date : str or None, optional
        Start date of the observation query in 'YYYY-MM-DD' format.
    
    end_date : str or None, optional
        End date of the observation query in 'YYYY-MM-DD' format.
    
    cache : bool, optional
        Whether to use local cache when retrieving ESO headers. Defaults to True.
    
    save : bool, optional
        If True, the resulting table is saved to disk as a CSV file. Defaults to True.
    
    existing_file_table_path : str or None, optional
        If provided, will attempt to load an existing file table and append only 
        newly retrieved files to it.
    
    batch_size : int, optional
        Number of files to process per header retrieval batch. Defaults to 100.
    
    date_batch_months : int or None, optional
        If provided, the full date range will be split into monthly batches of this size 
        to avoid timeouts. Set to None to query the entire range at once.

    Returns
    -------
    final_table : astropy.table.Table
        Table containing metadata for all science and calibration frames within the 
        specified date range. Only selected header entries are included.

    Header Fields Extracted
    -----------------------
    A subset of relevant ESO FITS header keywords are extracted and renamed to 
    user-friendly column names. These include:

    - **Target and Coordinates**:
        - `OBJECT`: Target name from header.
        - `RA`, `DEC`: Sky coordinates in degrees (ICRS).
        - `TARG_ALPHA`, `TARG_DELTA`: Telescope target coordinates (RA/Dec) at epoch of observation.
        - `TARG_PMA`, `TARG_PMD`: Proper motion in RA/Dec [mas/yr].
        - `DROT2_RA`, `DROT2_DEC`: Telescope derotator coordinates.

    - **Telescope and Observing Conditions**:
        - `TEL GEOELEV`, `GEOLAT`, `GEOLON`: Telescope location (elevation and geodetic coords).
        - `ALT_BEGIN`, `TEL AZ`: Telescope altitude and azimuth at start.
        - `TEL PARANG START`, `TEL PARANG END`: Start and end parallactic angles [deg].
        - `AMBI_TAU`: Atmospheric coherence time Tau₀ [s].
        - `AMBI_FWHM_START`, `AMBI_FWHM_END`: FWHM seeing estimates at start/end [arcsec].
        - `AIRMASS`, `TEL AIRM END`: Start and end airmass.
        - `TEL AMBI TEMP`, `TEL AMBI WINDSP`, `TEL AMBI WINDDIR`: Ambient conditions.

    - **Detector and Exposure Info**:
        - `EXPTIME`: Exposure time per frame [s].
        - `SEQ1_DIT`, `DIT_DELAY`, `NDIT`: DIT and NDIT for detector sequencing.
        - `READOUT_MODE`, `SEQ_UTC`, `FRAM_UTC`, `MJD_OBS`: Timing metadata.

    - **Instrument Configuration**:
        - `SEQ_ARM`: Instrument arm (IFS or IRDIS).
        - `IFS_MODE`, `INS2 MODE`, `PRISM`: IFS configuration (e.g., 'IFS-H', 'YJ').
        - `INS1 FILT NAME`, `INS1 MODE`: IRDIS filters and modes.
        - `CORO`: Coronagraph type.
        - `ND_FILTER`, `ND_FILTER_FLUX`: Neutral density filters.
        - `SHUTTER`, `DROT2_BEGIN`, `DROT2_END`, `INS4 DROT2 POSANG`: Derotator and shutter state.

    - **SAXO AO Subsystem**:
        - `AOS TTLOOP STATE`, `AOS HOLOOP STATE`, `AOS PUPLOOP STATE`, `AOS IRLOOP STATE`: AO loop statuses.
        - `AOS VISWFS MODE`: Visible wavefront sensor mode.

    - **Data Provenance and Meta**:
        - `DPR_CATG`, `DPR_TYPE`, `DPR_TECH`: Data category (SCIENCE or CALIB), type, and technique.
        - `OBS_PROG_ID`, `OBS_PI_COI`, `OBS_ID`, `OBS_NAME`: ESO observing program metadata.
        - `ORIGFILE`, `DATE_OBS`, `DATE`, `DP.ID`: File identifiers and timing.
        - `FILE_SIZE`: Estimated file size in megabytes (based on header structure).

    Notes
    -----
    - The resulting table is optimized for further filtering and use in the SPHERE 
      pipeline for extracting and calibrating IFS data.
    - The function uses `astroquery.eso.Eso` to interact with the ESO Science Archive.
    - The renaming and keyword filtering are defined by `header_list`, which is designed 
      to keep only scientifically relevant metadata and simplify downstream usage.
    - Date fields are enhanced with a `NIGHT_START` column for grouping by observing night.
    - Observations from both calibration and science categories are included.
    - Use `existing_file_table_path` to incrementally update a file table over time.
    """

    logger.info(f"Starting file table generation with date range: {start_date} to {end_date}")
    
    previous_file_table = None
    existing_entries = set()
    if existing_file_table_path is not None:
        logger.info(f"Loading existing file table from: {existing_file_table_path}")
        try:
            previous_file_table = pd.read_csv(existing_file_table_path)
            existing_entries = set(previous_file_table["DP.ID"].values)
            logger.info(f"Found {len(existing_entries)} existing entries")
        except Exception as e:
            logger.error(f"Error loading existing file table: {e}")
            raise

    logger.info("Initializing ESO query interface")
    eso = Eso()
    eso.ROW_LIMIT = 10000000  # Set high limit for queries

    # Generate date batches if specified
    date_batches = []
    if date_batch_months is not None and start_date is not None and end_date is not None:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        while current < end:
            batch_end = min(current + relativedelta(months=date_batch_months), end)
            date_batches.append((
                current.strftime("%Y-%m-%d"),
                batch_end.strftime("%Y-%m-%d")
            ))
            current = batch_end
        logger.info(f"Split date range into {len(date_batches)} batches")
    else:
        # Single date range
        date_batches = [(start_date, end_date)]

    # Collect all DP.IDs across all date batches
    all_dp_ids_calib = []
    all_dp_ids_sci = []

    # Use tqdm to show progress
    for batch_idx, (batch_start, batch_end) in tqdm(enumerate(date_batches), total=len(date_batches), 
                                                desc="Processing date batches", unit="batch"):
        # logger.info(f"Processing date batch {batch_idx+1}/{len(date_batches)}: {batch_start} to {batch_end}")
        
        # Define query parameters for this date batch
        calibration_columns = {
            'dp_cat': 'CALIB',
            'dp_type': "WAVE,LAMP",
            'box': 360,
            'seq_arm': 'IFS',
            'stime': batch_start,
            'etime': batch_end
        }
        science_columns = {
            'dp_cat': 'SCIENCE',
            'box': 360,
            'seq_arm': 'IFS',
            'stime': batch_start,
            'etime': batch_end
        }
        
        # Query ESO archive for calibration data
        # logger.info(f"Querying ESO archive for calibration data in batch {batch_idx+1}")
        try:
            calibration_results = eso.query_instrument(
                instrument='sphere',
                column_filters=calibration_columns
            )
            if calibration_results is None:
                logger.warning(f"No calibration results found in batch {batch_idx+1}")
                dp_ids_calib = []
            else:
                dp_ids_calib = list(calibration_results['DP.ID'].value)
                # logger.info(f"Found {len(dp_ids_calib)} calibration files in batch {batch_idx+1}")
                all_dp_ids_calib.extend(dp_ids_calib)
        except Exception as e:
            logger.error(f"Error querying calibration data in batch {batch_idx+1}: {e}")
            dp_ids_calib = []
        
        # Query ESO archive for science data
        # logger.info(f"Querying ESO archive for science data in batch {batch_idx+1}")
        try:
            science_results = eso.query_instrument(
                instrument='sphere',
                column_filters=science_columns
            )
            if science_results is None:
                logger.warning(f"No science results found in batch {batch_idx+1}")
                dp_ids_sci = []
            else:
                dp_ids_sci = list(science_results['DP.ID'].value)
                # logger.info(f"Found {len(dp_ids_sci)} science files in batch {batch_idx+1}")
                all_dp_ids_sci.extend(dp_ids_sci)
        except Exception as e:
            logger.error(f"Error querying science data in batch {batch_idx+1}: {e}")
            dp_ids_sci = []

    # Combine all results
    dp_ids = all_dp_ids_sci + all_dp_ids_calib

    # Remove duplicates
    dp_ids = list(set(dp_ids))
    logger.info(f"Total unique files found across all date batches: {len(dp_ids)}")
    
    # If a previous file table exists, only download headers of new files
    if existing_file_table_path is not None:
        new_dp_ids = set(dp_ids) - existing_entries
        dp_ids = list(new_dp_ids)
        logger.info(f"After filtering existing entries: {len(dp_ids)} new files to process")
    
    if not dp_ids:
        logger.info("No new files to process")
        if existing_file_table_path is not None:
            logger.info("Returning existing file table")
            return Table.from_pandas(previous_file_table)
        else:
            logger.warning("No files found and no existing table. Returning empty table.")
            return Table()
    
    # Prepare output file (if saving) by backing up any existing file
    if save:
        output_path = os.path.join(folder, f"table_of_IFS_files_{file_ending}.csv")
        if os.path.exists(output_path):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = f"{output_path}_{timestamp}.bak"
            os.rename(output_path, backup_path)
            logger.info(f"Existing file backed up to {backup_path}")
        # Create a blank file
        open(output_path, "w").close()
        first_batch_file = True
    else:
        output_path = None
        first_batch_file = False

    processed_batches = []
    # total_batches = (len(dp_ids) + batch_size - 1) // batch_size
    time0 = time.perf_counter()
    
    # Retrieve headers in batches with a progress bar via tqdm
    for i in tqdm(range(0, len(dp_ids), batch_size), desc="Retrieving batches", unit="batch"):
        batch_end = min(i + batch_size, len(dp_ids))
        batch = dp_ids[i:batch_end]
        try:
            hdrs_batch = eso.get_headers(product_ids=batch, cache=cache)
            if hdrs_batch is None:
                continue
        except Exception as e:
            logger.error(f"Error retrieving headers for batch starting at index {i}: {e}")
            continue
        
        # Process the batch
        batch_df = hdrs_batch.to_pandas()
        
        # Add file size for each row estimated from header
        batch_df["FILE_SIZE"] = batch_df.apply(
            lambda row: compute_fits_header_data_size(Header(row.dropna().to_dict())),
            axis=1
        )

        # Keep only the desired columns
        cols_to_keep = [col for col in batch_df.columns if col in keep_columns_set]
        batch_df = batch_df[cols_to_keep]
        
        # Rename columns to SPHERICAL format
        rename_dict = { header_list[key][0]: key for key in header_list if header_list[key][0] in batch_df.columns }
        batch_df.rename(columns=rename_dict, inplace=True)
        
        # Process date fields: extract first 10 characters from DATE_OBS to create DATE_SHORT
        if "DATE_OBS" in batch_df.columns:
            batch_df["DATE_SHORT"] = batch_df["DATE_OBS"].apply(lambda x: x[0:10] if isinstance(x, str) else x)
        
        # Convert to an Astropy table to add night start date, then back to DataFrame
        batch_table = Table.from_pandas(batch_df)
        batch_table = add_night_start_date(batch_table, key="DATE_OBS")
        processed_df = batch_table.to_pandas()
        processed_batches.append(processed_df)
        
        # Append the processed batch to the output CSV file if saving
        if save:
            if first_batch_file:
                processed_df.to_csv(output_path, mode='a', header=True, index=False)
                first_batch_file = False
            else:
                processed_df.to_csv(output_path, mode='a', header=False, index=False)
    
    time1 = time.perf_counter()
    logger.info(f"Header retrieval completed in {time1 - time0:.2f} seconds")
    
    # Combine all new batches
    if processed_batches:
        new_data_df = pd.concat(processed_batches, ignore_index=True)
    else:
        new_data_df = pd.DataFrame()
    
    # Combine with existing data if provided
    if previous_file_table is not None:
        combined_df = pd.concat([previous_file_table, new_data_df], ignore_index=True)
    else:
        combined_df = new_data_df
    
    # Save final combined results (overwrite the file)
    if save:
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Final combined table saved to {output_path}")
    
    final_table = Table.from_pandas(combined_df)
    logger.info("File table generation completed successfully")
    return final_table

