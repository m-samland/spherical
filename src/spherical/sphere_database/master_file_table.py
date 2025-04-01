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
from astropy.table import Table
from astroquery.eso import Eso
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from spherical.sphere_database.database_utils import add_night_start_date

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('spherical.master_file_table')

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
                           existing_master_file_table_path=None,
                           batch_size=100, date_batch_months=1):
    """
    Generate a master table from input file containing list of filenames for calibration and science files.

    Parameters:
        folder (str): The folder path where the files are located.
        file_ending (str, optional): The file ending to be used for the output files. Defaults to 'myrun'.
        start_date (str, optional): Start date for query in format 'YYYY-MM-DD'. Defaults to None.
        end_date (str, optional): End date for query in format 'YYYY-MM-DD'. Defaults to None.
        cache (bool, optional): Whether to use cache. Defaults to True.
        save (bool, optional): Whether to save the output. Defaults to True.
        savefull (bool, optional): Whether to save full headers. Defaults to False.
        existing_master_file_table_path (str, optional): The path to the existing master file table. 
                    If provided, only download headers of new files. Defaults to None.
        batch_size (int, optional): Number of products to query at once. Defaults to 250.
        date_batch_months (int, optional): Split the date range into batches of this many months.
                    If None, the entire date range is queried at once. Defaults to None.

    Returns:
        astropy.table.Table: The generated file table.
    """

    logger.info(f"Starting master file table generation with date range: {start_date} to {end_date}")
    
    previous_master_file_table = None
    existing_entries = set()
    if existing_master_file_table_path is not None:
        logger.info(f"Loading existing master file table from: {existing_master_file_table_path}")
        try:
            previous_master_file_table = pd.read_csv(existing_master_file_table_path)
            existing_entries = set(previous_master_file_table["DP.ID"].values)
            logger.info(f"Found {len(existing_entries)} existing entries")
        except Exception as e:
            logger.error(f"Error loading existing master file table: {e}")
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
    
    # If a previous master table exists, only download headers of new files
    if existing_master_file_table_path is not None:
        new_dp_ids = set(dp_ids) - existing_entries
        dp_ids = list(new_dp_ids)
        logger.info(f"After filtering existing entries: {len(dp_ids)} new files to process")
    
    if not dp_ids:
        logger.info("No new files to process")
        if existing_master_file_table_path is not None:
            logger.info("Returning existing master file table")
            return Table.from_pandas(previous_master_file_table)
        else:
            logger.warning("No files found and no existing table. Returning empty table.")
            return Table()
    
    # Prepare output file (if saving) by backing up any existing file
    if save:
        output_path = os.path.join(folder, f"table_of_files_{file_ending}.csv")
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
        # Keep only the desired columns
        cols_to_keep = [col for col in batch_df.columns if col in keep_columns_set]
        batch_df = batch_df[cols_to_keep]
        
        # Rename columns to SPHERICAL format
        rename_dict = { header_list[key][0]: key for key in header_list if header_list[key][0] in batch_df.columns }
        batch_df.rename(columns=rename_dict, inplace=True)
        
        # Process date fields: extract first 10 characters from DATE_OBS to create DATE_SHORT
        if "DATE_OBS" in batch_df.columns:
            batch_df["DATE_SHORT"] = batch_df["DATE_OBS"].apply(lambda x: x[0:10] if isinstance(x, str) else x)
        # Add constant file size column
        batch_df["FILE_SIZE"] = 1.0
        
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
    if previous_master_file_table is not None:
        combined_df = pd.concat([previous_master_file_table, new_data_df], ignore_index=True)
    else:
        combined_df = new_data_df
    
    # Save final combined results (overwrite the file)
    if save:
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Final combined master table saved to {output_path}")
    
    final_table = Table.from_pandas(combined_df)
    logger.info("Master file table generation completed successfully")
    return final_table


def get_headerval_with_exeption(header, keyword, default_val):
    try:
        val = header[keyword]
    except KeyError:
        val = default_val
    except IOError:
        val = default_val
    return val


