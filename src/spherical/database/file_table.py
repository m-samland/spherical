"""
Module for querying the ESO archive and generating SPHERE IFS/IRDIS file metadata tables.

This module allows users to query SPHERE instrument data (IFS or IRDIS) from the ESO archive,
retrieve relevant FITS headers, and produce structured metadata tables for science and
calibration frames. It is primarily used in the Spherical pipeline to prepare data for
further calibration and scientific analysis.

All header fields, query settings, and output formatting are carefully tuned to astronomical
needs, respecting units, coordinate frames (ICRS), time standards (UTC), and observing
conditions.

Main Components
---------------
- `header_list`: Ordered mapping of extracted FITS header keywords to standardized output names.
- `keep_columns`: List of unique FITS keywords needed from ESO queries.
- `query_eso_data`: Helper to safely query the ESO archive with retries and error handling.
- `make_file_table`: Main function to generate a complete file table for SPHERE observations.

Notes
-----
- All date fields assume UTC unless otherwise specified.
- Coordinate fields (RA/Dec) are in degrees (ICRS reference frame).
- Atmospheric conditions are extracted where available (seeing, tau₀, airmass, etc.).
- File size is estimated from FITS header metadata using `compute_fits_header_data_size`.
"""


__author__ = "M. Samland @ MPIA (Heidelberg, Germany)"

import collections
import datetime
import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from astropy.io.fits import Header
from astropy.table import Table, vstack
from astroquery.eso import Eso
from dateutil.relativedelta import relativedelta

from spherical.database.database_utils import add_night_start_date, compute_fits_header_data_size, normalize_shutter_column
from tqdm.auto import tqdm

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
        ("NAXIS", ("NAXIS", 3)),
        ("NAXIS1", ("NAXIS1", 2048)),
        ("NAXIS2", ("NAXIS2", 2048)),
        ("NAXIS3", ("NAXIS3", 0)),
        ("BITPIX", ("BITPIX", -32)),
    ]
)

keep_columns = []
for key in header_list:
    keep_columns.append(header_list[key][0])
keep_columns = list(np.unique(keep_columns))
keep_columns_set = set(keep_columns)


def query_eso_data(eso, column_filters: Dict[str, Any], batch_idx: int, data_type: str) -> List[str]:
    """
    Query the ESO archive for SPHERE files matching specific filters.

    This helper wraps `astroquery.eso.Eso.query_instrument` with error handling
    to robustly fetch product IDs (DP.ID) matching the desired metadata criteria.

    Parameters
    ----------
    eso : astroquery.eso.Eso
        An active ESO query interface instance.
    
    column_filters : dict
        Filters to apply to the archive query (e.g., date range, instrument arm, data type).
        Keys correspond to archive fields such as 'dp_cat', 'seq_arm', 'stime', 'etime'.
    
    batch_idx : int
        Index of the current batch in multi-batch retrieval (for logging and diagnostics).
    
    data_type : str
        Type of data being queried (e.g., 'science', 'calibration') for logging purposes.

    Returns
    -------
    dp_ids : list of str
        List of dataset product IDs (DP.ID) retrieved from the query.

    Notes
    -----
    - Returns an empty list if no results are found.
    - Logs warnings for empty batches and errors for query failures.
    - Assumes SPHERE instrument data only.
    """

    try:
        results = eso.query_instrument(instrument='sphere', column_filters=column_filters)
        if results is None:
            logger.warning(f"No {data_type} results found in batch {batch_idx+1}")
            return []
        else:
            dp_ids = list(results['DP.ID'].value)
            # logger.info(f"Found {len(dp_ids)} {data_type} files in batch {batch_idx+1}")
            return dp_ids
    except Exception as e:
        logger.error(f"Error querying {data_type} data in batch {batch_idx+1}: {e}")
        return []


def make_file_table(output_dir, 
                    instrument='ifs',
                    output_suffix='myrun',
                    start_date=None,
                    end_date=None,
                    cache=False,
                    existing_table_path=None,
                    batch_size=100,
                    date_batch_months=1,
                    resume=True):
    """
    Create a file table of SPHERE science and calibration observations 
    by retrieving and parsing ESO archive headers.

    This function queries the ESO archive for all SPHERE IFS or IRDIS files 
    within a given date range and collects their metadata headers. It extracts only a 
    predefined set of relevant header keywords (defined in `header_list`) to simplify 
    later processing and analysis in the spherical pipeline.

    It supports incremental updating of existing file tables, caching, batch querying
    over long date ranges, and output file versioning.

    Parameters
    ----------
    output_dir : str
        Directory where the output table (CSV) will be saved.
    
    instrument : str
        Instrument type to query ('ifs' or 'irdis'). Default is 'ifs'.

    output_suffix : str, optional
        Suffix for naming the output file. Default is 'myrun'.
    
    start_date : str or None, optional
        Start date of the observation query in 'YYYY-MM-DD' format.
    
    end_date : str or None, optional
        End date of the observation query in 'YYYY-MM-DD' format.
    
    cache : bool, optional
        Whether to use local cache when retrieving ESO headers. Defaults to False.
        Creating the entire database can lead to 20+ GB of cached header data.
  
    existing_table_path : str or None, optional
        If provided, will attempt to load an existing file table and append only 
        newly retrieved files to it.
    
    batch_size : int, optional
        Number of files to process per header retrieval batch. Defaults to 100.
    
    date_batch_months : int or None, optional
        If provided, the full date range will be split into monthly batches of this size 
        to avoid timeouts. Set to None to query the entire range at once.

    resume : bool, optional
        If True (default), automatically resumes an interrupted run by detecting a
        partial file (``*_partial.csv``) and/or the existing output file. Already
        downloaded DP.IDs from both files are skipped. New data is written to the
        partial file and merged into the final output only upon successful completion.

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
    - Use `existing_table_path` to incrementally update a file table over time.
    - When ``resume=True``, interrupted runs are automatically resumed. New data
      is written to a ``*_partial.csv`` file; the final output is only updated
      upon successful completion.
    """

    logger.info(f"Starting file table generation with date range: {start_date} to {end_date}")

    instrument = instrument.lower()

    logger.info("Initializing ESO query interface")
    eso = Eso()
    eso.ROW_LIMIT = 10000000

    # Prepare output and partial file paths
    output_path = os.path.join(output_dir, f"table_of_files_{instrument}{output_suffix}.csv".lower())
    partial_path = output_path.replace(".csv", "_partial.csv")

    # Collect known DP.IDs from all available sources to skip already-downloaded files
    existing_entries = set()

    # Load completed output file if it exists
    if os.path.exists(output_path):
        try:
            previous_output = pd.read_csv(output_path)
            existing_entries.update(previous_output["DP.ID"].values)
            logger.info(f"Found {len(previous_output)} entries in completed output file")
        except Exception as e:
            logger.error(f"Error reading existing output file: {e}")
            raise

    # Load partial file from a previously interrupted run (resume mode)
    if resume and os.path.exists(partial_path):
        try:
            partial_df = pd.read_csv(partial_path)
        except Exception:
            logger.warning("Partial file could not be read normally, attempting recovery with on_bad_lines='skip'")
            partial_df = pd.read_csv(partial_path, on_bad_lines='skip')
        existing_entries.update(partial_df["DP.ID"].values)
        logger.info(f"Resuming: found {len(partial_df)} entries in partial file")
    elif not resume and os.path.exists(partial_path):
        # Not resuming — remove stale partial file
        os.remove(partial_path)
        logger.info("Resume disabled: removed existing partial file")

    # Load explicitly provided existing table (backward compatibility)
    if existing_table_path is not None:
        existing_table_path = str(existing_table_path)
        if os.path.exists(existing_table_path) and os.path.abspath(existing_table_path) != os.path.abspath(output_path):
            logger.info(f"Loading additional existing file table from: {existing_table_path}")
            try:
                ext_table = pd.read_csv(existing_table_path)
                existing_entries.update(ext_table["DP.ID"].values)
                logger.info(f"Added {len(ext_table)} entries from existing_table_path")
            except Exception as e:
                logger.error(f"Error loading existing file table: {e}")
                raise

    if existing_entries:
        logger.info(f"Total known DP.IDs to skip: {len(existing_entries)}")

    first_batch_file = not os.path.exists(partial_path) or not resume

    # Clamp end_date to tomorrow if it extends beyond the current date
    if end_date is not None:
        today = datetime.datetime.now()
        tomorrow = today + datetime.timedelta(days=1)
        end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        if end_date_dt > tomorrow:
            end_date = tomorrow.strftime("%Y-%m-%d")
            logger.info(f"end_date was beyond current date, clamped to {end_date}")

    # Set up date batching
    date_batches = []
    if date_batch_months is not None and start_date and end_date:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        while current < end:
            batch_end = min(current + relativedelta(months=date_batch_months), end)
            date_batches.append((current.strftime("%Y-%m-%d"), batch_end.strftime("%Y-%m-%d")))
            current = batch_end
        logger.info(f"Split date range into {len(date_batches)} batches")
    else:
        date_batches = [(start_date, end_date)]

    time0 = time.perf_counter()

    for batch_idx, (batch_start, batch_end) in tqdm(enumerate(date_batches), total=len(date_batches),
                                                    desc="Processing date batches", unit="batch"):
        
        calibration_templates = {
            'ifs': [{'dp_cat': 'CALIB', 'dp_type': 'WAVE,LAMP'}],
            'irdis': [
                {'dp_cat': 'CALIB', 'dp_type': 'FLAT,LAMP'},
                {'dp_cat': 'CALIB', 'dp_type': 'DARK,BACKGROUND'}
            ]
        }

        if instrument not in calibration_templates:
            raise ValueError(f"Unsupported instrument: {instrument}. Use 'ifs' or 'irdis'.")

        common_query_fields = {
            'box': 360,
            'seq_arm': instrument.upper(),
            'stime': batch_start,
            'etime': batch_end
        }

        batch_dp_ids = []
        for calib_filter in calibration_templates[instrument]:
            calib_ids = query_eso_data(eso, {**calib_filter, **common_query_fields}, batch_idx, data_type='calibration')
            batch_dp_ids.extend(calib_ids)

        sci_ids = query_eso_data(eso, {**common_query_fields, 'dp_cat': 'SCIENCE'}, batch_idx, data_type='science')
        batch_dp_ids.extend(sci_ids)

        batch_dp_ids = list(set(batch_dp_ids))

        if existing_entries:
            batch_dp_ids = list(set(batch_dp_ids) - existing_entries)
            logger.info(f"After filtering existing entries: {len(batch_dp_ids)} new files to process in this batch")

        if not batch_dp_ids:
            logger.info("No new files to process in this batch")
            continue

        # Process headers in sub-batches
        for i in tqdm(range(0, len(batch_dp_ids), batch_size), desc=f"Retrieving headers (batch {batch_idx+1})", unit="sub-batch"):
            subbatch_end = min(i + batch_size, len(batch_dp_ids))
            subbatch = batch_dp_ids[i:subbatch_end]

            try:
                hdrs_batch = eso.get_headers(product_ids=subbatch, cache=cache)
                if hdrs_batch is None:
                    continue
            except Exception as e:
                logger.error(f"Error retrieving headers for batch starting at index {i}: {e}")
                continue

            header_batch_df = hdrs_batch.to_pandas()

            if header_batch_df.empty:
                continue

            # Add file size
            header_batch_df["FILE_SIZE"] = header_batch_df.apply(
                lambda row: compute_fits_header_data_size(Header(row.dropna().to_dict())),
                axis=1
            )

            # Keep only necessary columns
            cols_to_keep = [col for col in header_batch_df.columns if col in keep_columns_set]
            header_batch_df = header_batch_df[cols_to_keep]

            # Rename columns
            rename_dict = {header_list[key][0]: key for key in header_list if header_list[key][0] in header_batch_df.columns}
            header_batch_df.rename(columns=rename_dict, inplace=True)

            # Create DATE_SHORT
            if "DATE_OBS" in header_batch_df.columns:
                header_batch_df["DATE_SHORT"] = pd.to_datetime(
                    header_batch_df["DATE_OBS"], errors='coerce', utc=True
                ).dt.strftime('%Y-%m-%d')
                header_batch_df["DATE_SHORT"] = header_batch_df["DATE_SHORT"].fillna("INVALID_DATE")

            # Add NIGHT_START
            header_batch_df = add_night_start_date(header_batch_df, key="DATE_OBS")

            # Ensure all expected columns
            expected_columns = list(header_list.keys()) + ["DATE_SHORT", "NIGHT_START"]
            for col in expected_columns:
                if col not in header_batch_df.columns:
                    if col in header_list:
                        header_batch_df[col] = header_list[col][1]
                    else:
                        header_batch_df[col] = "INVALID_DATE"

            header_batch_df = header_batch_df[expected_columns]

            # Save immediately to partial file
            header_batch_df = normalize_shutter_column(header_batch_df)
            if first_batch_file:
                header_batch_df.to_csv(partial_path, mode='w', header=True, index=False)
                first_batch_file = False
            else:
                header_batch_df.to_csv(partial_path, mode='a', header=False, index=False)

    time1 = time.perf_counter()
    logger.info(f"Header retrieval completed in {time1 - time0:.2f} seconds")

    # Merge all data sources into the final output
    tables_to_merge = []

    if os.path.exists(output_path):
        try:
            tables_to_merge.append(Table.read(output_path, format='csv'))
            logger.info("Loaded existing output file for merge")
        except Exception as e:
            logger.error(f"Error reading output file during merge: {e}")

    if os.path.exists(partial_path):
        try:
            tables_to_merge.append(Table.read(partial_path, format='csv'))
            logger.info("Loaded partial file for merge")
        except Exception as e:
            logger.error(f"Error reading partial file during merge: {e}")

    if existing_table_path is not None:
        existing_table_path_abs = os.path.abspath(str(existing_table_path))
        output_path_abs = os.path.abspath(output_path)
        if os.path.exists(str(existing_table_path)) and existing_table_path_abs != output_path_abs:
            try:
                tables_to_merge.append(Table.read(str(existing_table_path), format='csv'))
                logger.info("Loaded existing_table_path for merge")
            except Exception as e:
                logger.error(f"Error reading existing_table_path during merge: {e}")

    if tables_to_merge:
        final_table = vstack(tables_to_merge) if len(tables_to_merge) > 1 else tables_to_merge[0]
        final_table = normalize_shutter_column(final_table)

        # Deduplicate on DP.ID
        if "DP.ID" in final_table.colnames:
            _, unique_idx = np.unique(final_table["DP.ID"], return_index=True)
            final_table = final_table[np.sort(unique_idx)]
            logger.info(f"Deduplicated: {len(final_table)} unique entries")

        # Sort by MJD_OBS
        if "MJD_OBS" in final_table.colnames:
            try:
                if hasattr(final_table['MJD_OBS'], 'mask'):
                    final_table = final_table[np.argsort(
                        np.where(final_table['MJD_OBS'].mask, np.inf, final_table['MJD_OBS'])
                    )]
                else:
                    final_table.sort('MJD_OBS')
                logger.info("Sorted final table by MJD_OBS")
            except Exception as e:
                logger.warning(f"Sorting by MJD_OBS failed: {e}")

        # Write final merged table atomically (write to temp, then replace)
        temp_output_path = output_path + ".tmp"
        final_table.write(temp_output_path, format='csv', overwrite=True)
        os.replace(temp_output_path, output_path)
        logger.info(f"Final table written to {output_path}")

        # Clean up partial file — merge is complete
        if os.path.exists(partial_path):
            os.remove(partial_path)
            logger.info("Removed partial file after successful merge")

    else:
        logger.warning("No data found. Returning empty table.")
        empty_df = pd.DataFrame(columns=list(header_list.keys()) + ["DATE_SHORT", "NIGHT_START"])
        final_table = Table.from_pandas(empty_df)

    logger.info("File table generation completed successfully")
    return final_table