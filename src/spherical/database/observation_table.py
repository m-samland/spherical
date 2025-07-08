__author__ = "M. Samland @ MPIA (Heidelberg, Germany)"

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from spherical.database import metadata
from spherical.database.database_utils import filter_for_science_frames


def remove_objects_from_simbad_list(target_table: Table, exclude_names: List[str]) -> Table:
    """Remove specified astronomical objects from a target list.

    Parameters
    ----------
    target_table : Table
        Astropy Table containing target data with a 'MAIN_ID' column.
    exclude_names : list of str
        List of target identifiers (MAIN_ID) to exclude from the table.

    Returns
    -------
    Table
        The input table with specified targets removed.

    Examples
    --------
    >>> filtered_table = remove_objects_from_simbad_list(target_table, ["HD 100546", "HR 8799"])
    """
    for name in exclude_names:
        mask = target_table["MAIN_ID"] == name
        if np.any(mask):
            target_table.remove_row(np.where(mask)[0][0])
        else:
            print(f"Name: {name} not found.")
    return target_table


def enumerate_observations_by_timediff(table: Table, key: str = "DATE_OBS") -> Table:
    """Assign numeric identifiers to observation sequences based on time intervals.

    Consecutive entries separated by more than one hour are assigned distinct observation numbers.

    Parameters
    ----------
    table : Table
        Astropy Table with observation data, including timestamps.
    key : str, optional
        Column name containing timestamps (UTC), by default "DATE_OBS".

    Returns
    -------
    Table
        Input table with an additional 'OBS_NUMBER' column indicating the observation sequence.
    """
    times = Time(table[key]).mjd * 24  # Convert to hours
    jumps = np.diff(times) > 1
    obs_numbers = np.cumsum(np.insert(jumps, 0, 0))
    table["OBS_NUMBER"] = obs_numbers
    return table


def match_files_to_target(science_table: Table, target: Table.Row, cone_size: u.Quantity) -> Table:
    """Select science files matching a given astronomical target based on angular proximity.

    Parameters
    ----------
    science_table : Table
        Table of science observations containing 'RA' and 'DEC' columns in degrees.
    target : Table.Row
        Single row from a target table, containing 'RA_HEADER' and 'DEC_HEADER' in degrees.
    cone_size : astropy.units.Quantity
        Angular radius within which files are associated with the target (e.g., 15*u.arcsec).

    Returns
    -------
    Table
        Subset of input science_table matching the target coordinates.
    """
    target_coords = SkyCoord(ra=target["RA_HEADER"] * u.deg, dec=target["DEC_HEADER"] * u.deg)
    file_coords = SkyCoord(ra=science_table["RA"] * u.deg, dec=science_table["DEC"] * u.deg)
    separation_mask = target_coords.separation(file_coords) < cone_size
    return science_table[separation_mask]


def compute_basic_metadata(observation_files: Table, instrument: str, ndit_key: str, mode_key: str) -> OrderedDict:
    """Compute summary statistics and basic metadata for a given astronomical observation sequence.

    Parameters
    ----------
    observation_files : Table
        Files associated with a single observation sequence.
    instrument : str
        Instrument identifier ('irdis' or 'ifs').
    ndit_key : str
        Column name representing number of detector integrations (e.g., 'NDIT' or 'NAXIS3').
    mode_key : str
        Column name representing the instrument mode (e.g., 'IFS_MODE' or 'DB_FILTER').

    Returns
    -------
    OrderedDict
        Dictionary of basic metadata for the observation sequence.
    """
    mid_index = len(observation_files) // 2
    FWHM = (observation_files["AMBI_FWHM_START"] + observation_files["AMBI_FWHM_END"]) / 2.0

    metadata = OrderedDict()
    metadata["OBS_START"] = observation_files["MJD_OBS"][0]
    metadata["OBS_END"] = observation_files["MJD_OBS"][-1]
    metadata["MJD_MEAN"] = np.mean(observation_files["MJD_OBS"])
    metadata["TOTAL_EXPTIME_SCI"] = round(np.sum(observation_files["EXPTIME"] * observation_files[ndit_key]) / 60, 3)
    metadata["MEAN_TAU"] = round(np.mean(observation_files["AMBI_TAU"]), 3)
    metadata["STDDEV_TAU"] = round(np.std(observation_files["AMBI_TAU"]), 3)
    metadata["MEAN_FWHM"] = round(np.mean(FWHM), 3)
    metadata["STDDEV_FWHM"] = round(np.std(FWHM), 3)
    metadata["MEAN_AIRMASS"] = round(np.mean(observation_files["AIRMASS"]), 3)
    metadata["DIT"] = observation_files["EXPTIME"][mid_index]
    metadata["NDIT"] = observation_files[ndit_key][mid_index]
    metadata["NCUBES"] = len(observation_files)
    metadata["DEROTATOR_MODE"] = observation_files["DEROTATOR_MODE"][mid_index]
    metadata["ND_FILTER"] = observation_files["ND_FILTER"][mid_index]
    metadata["OBS_PROG_ID"] = observation_files["OBS_PROG_ID"][mid_index]
    metadata["OBS_ID"] = observation_files["OBS_ID"][mid_index]
    metadata["TOTAL_FILE_SIZE_MB"] = round(np.sum(observation_files["FILE_SIZE"]), 2)
    metadata["FILTER"] = observation_files[mode_key][mid_index]
    return metadata


def compute_derotation_info(observation_files: Table, polarimetry: bool) -> Dict[str, float | bool]:
    """Compute derotation angle and detect derotator fallback behavior.

    Uses metadata module to compute angular rotation across observation sequence.

    Parameters
    ----------
    observation_files : Table
        Files in the observation sequence.
    polarimetry : bool
        Files are taken in polarimetric mode.

    Returns
    -------
    dict
        Dictionary with 'ROTATION' (float) and 'DEROTATOR_FLAG' (bool).
    """
    if polarimetry:
        return {"ROTATION": -10000, "DEROTATOR_FLAG": True}
    
    try:
        frames_metadata = metadata.prepare_dataframe(observation_files)
        metadata.compute_times(frames_metadata)
        metadata.compute_angles(frames_metadata)
        derot_angles = frames_metadata["DEROT ANGLE"]
        rotation = round(abs(derot_angles.iloc[-1] - derot_angles.iloc[0]), 3) if len(derot_angles) > 1 else 0.0
        return {"ROTATION": rotation, "DEROTATOR_FLAG": False}
    except Exception:
        return {"ROTATION": -10000, "DEROTATOR_FLAG": True}


def calculate_observation_metadata(observation_files: Table, instrument: str, polarimetry: bool, ndit_key: str, mode_key: str) -> OrderedDict:
    """Wrapper to compute full observation metadata including derotation angles.

    Parameters
    ----------
    observation_files : Table
        Files in the observation group.
    instrument : str
        Instrument name (e.g., 'irdis').
    ndit_key : str
        NDIT or NAXIS3 column.
    mode_key : str
        Filter/mode column name.

    Returns
    -------
    OrderedDict
        Complete observational metadata dictionary.
    """
    metadata_dict = compute_basic_metadata(observation_files, instrument, ndit_key, mode_key)
    metadata_dict.update(compute_derotation_info(observation_files, polarimetry))
    return metadata_dict


def select_primary_science_frames(obs_group: Table, ndit_key: str) -> Tuple[str, Table, Dict[str, float]]:
    """Return primary science type, its rows, and total exptimes for all science types."""
    kinds = {
        "CORO": obs_group[obs_group["DPR_TYPE"] == "OBJECT"],
        "CENTER": obs_group[obs_group["DPR_TYPE"] == "OBJECT,CENTER"],
        "FLUX": obs_group[obs_group["DPR_TYPE"] == "OBJECT,FLUX"],
    }

    total_exptimes = {}
    best_kind = None
    best_exp = -1.0

    for kind, rows in kinds.items():
        total = float(np.sum(rows["EXPTIME"] * rows[ndit_key])) if len(rows) else 0.0
        total_exptimes[f"TOTAL_EXPTIME_{kind}"] = round(total / 60., 3)
        if total > best_exp or (total == best_exp and best_kind is None):
            best_kind, best_exp = kind, total

    return best_kind, kinds[best_kind] if best_kind else Table(), total_exptimes


def evaluate_observation_flags(obs_group: Table, ndit_key: str) -> Dict[str, object]:
    """Determine quality flags and mode classification for an observation sequence.

    Parameters
    ----------
    obs_group : Table
        Observation group (single sequence).
    ndit_key : str
        Column name for detector integration count.

    Returns
    -------
    dict
        Dictionary of flags and breakdown counts (e.g., WAFFLE_MODE, FLUX_FLAG, NCENTER).
    """
    t_coro = obs_group[obs_group["DPR_TYPE"] == "OBJECT"]
    t_center = obs_group[obs_group["DPR_TYPE"] == "OBJECT,CENTER"]
    t_flux = obs_group[obs_group["DPR_TYPE"] == "OBJECT,FLUX"]

    flags = {
        "NCENTER": len(t_center),
        "NFLUX": len(t_flux),
        "FLUX_FLAG": False,
        "FLUX_DIT_FLAG": False,
        "CENTER_FLAG": False,
        "CENTER_DIT_FLAG": False,
        "CORO_FLAG": False,
        "CORO_DIT_FLAG": False,
        "DIT_CENTER": 0.0,
        "NDIT_CENTER": 0,
        "DIT_FLUX": 0.0,
        "NDIT_FLUX": 0,
        "DIT_CORO": 0.0,
        "NDIT_CORO": 0,
        "ND_FILTER_FLUX": "N/A",
        "WAFFLE_AMP": 0.0,
    }

    if len(t_flux) == 0:
        flags["FLUX_FLAG"] = True
    elif len(t_flux) > 0 and len(t_flux.group_by("EXPTIME").groups.keys) != 1:
        flags["FLUX_DIT_FLAG"] = True
    else:
        flags["DIT_FLUX"] = t_flux["EXPTIME"][-1]
        flags["NDIT_FLUX"] = t_flux[ndit_key][-1]
        flags["ND_FILTER_FLUX"] = t_flux["ND_FILTER"][-1]
    
    if len(t_center) == 0:
        flags["CENTER_FLAG"] = True
    elif len(t_center) > 0 and len(t_center.group_by("EXPTIME").groups.keys) != 1:
        flags["CENTER_DIT_FLAG"] = True
    else:
        flags["DIT_CENTER"] = t_center["EXPTIME"][-1]
        flags["NDIT_CENTER"] = t_center[ndit_key][-1]

    if len(t_coro) == 0:
        flags["CORO_FLAG"] = True
    elif len(t_coro) > 0 and len(t_coro.group_by("EXPTIME").groups.keys) != 1:
        flags["CORO_DIT_FLAG"] = True
    else:
        flags["DIT_CORO"] = t_coro["EXPTIME"][-1]
        flags["NDIT_CORO"] = t_coro[ndit_key][-1]

    if len(t_center) > 0:
        flags["WAFFLE_AMP"] = t_center["WAFFLE_AMP"][-1]

    return flags


def group_observation_sequences(
    matched_files: Table,
    mode_key: str,
    time_key: str = "DATE_OBS",
    night_key: str = "NIGHT_START",
    time_gap_hours: float = 1.0,
    group_by_obs_number: bool = False
) -> List[Tuple[Dict[str, object], Table]]:
    """Group observation files into sequences by night, mode, and optionally time gaps.

    Parameters
    ----------
    matched_files : Table
        Science files matched to a given target.
    mode_key : str
        Column representing instrument mode (e.g., 'IFS_MODE').
    time_key : str, optional
        Timestamp column, by default 'DATE_OBS'.
    night_key : str, optional
        Night boundary column, by default 'NIGHT_START'.
    time_gap_hours : float, optional
        Maximum allowed time gap to consider frames as same sequence.
    group_by_obs_number : bool, optional
        If True, uses N-hour time gaps to segment into multiple sequences.
        If False, all files on a given night and in the same mode are grouped together (default).

    Returns
    -------
    list of (dict, Table)
        Each tuple contains a group key dictionary and the corresponding observation group.
    """
    matched_files.sort(time_key)

    if group_by_obs_number:
        times = Time(matched_files[time_key]).mjd * 24
        jumps = np.diff(times) > time_gap_hours
        matched_files["OBS_NUMBER"] = np.cumsum(np.insert(jumps, 0, 0))
    else:
        matched_files["OBS_NUMBER"] = np.zeros(len(matched_files), dtype=int)
    
    group_cols = [night_key, "OBS_NUMBER", mode_key]
    grouped = matched_files.group_by(group_cols)
    result = []

    for key_row, group in zip(grouped.groups.keys, grouped.groups):
        key_dict = {col: key_row[col] for col in group_cols}
        result.append((key_dict, group))

    return result


def create_observation_table(
    table_of_files: Table,
    table_of_targets: Table,
    instrument: str,
    polarimetry: bool = False,
    cone_size_science: float = 15.0,
    remove_fillers: bool = True,
    group_by_time_gaps: bool = False,
    reorder_columns: bool = True,
) -> Tuple[Table, Table]:

    """Generate a summary table of SPHERE observations matched to astronomical targets.

    This function matches science observation files from the SPHERE instrument to astronomical
    targets based on sky coordinates and generates a summary of each observation sequence. Each
    sequence is identified by observation date, observing mode, and angular proximity to the target.

    Parameters
    ----------
    table_of_files : Table
        Astropy Table containing SPHERE observation files metadata, including essential columns such as
        'RA', 'DEC', 'DATE_OBS', and instrument-specific columns ('IFS_MODE' or 'DB_FILTER').
    table_of_targets : Table
        Astropy Table of astronomical targets to match, containing at least 'MAIN_ID', 'RA_HEADER',
        and 'DEC_HEADER' columns.
    instrument : str
        Instrument name ('ifs' or 'irdis') indicating which SPHERE instrument the data originates from.
    polarimetry : bool, optional
        Flag indicating if the data are from polarimetric observations, by default False.
    cone_size_science : float, optional
        Angular radius (arcseconds) used to associate files with targets, by default 15.0.
    remove_fillers : bool, optional
        If True, calibration and filler frames are removed from consideration, by default True.
    group_by_time_gaps : bool, optional
        If True (default), splits observation sequences within a night using 1-hour time gaps.
        If False, assumes one observation per night per mode.
    reorder_columns : bool, optional
    If True (default), the output columns will be arranged in a user-friendly order:
    target metadata → instrument setup → timing → exposures → flags → conditions → rotation → program info.
    Set to False to preserve native column order.
        
    Returns
    -------
    obs_table : Table
        Summary table of observations, with each row representing an observation sequence and containing
        metadata such as exposure times, observing conditions, instrument setup, and target information.
    table_of_targets : Table
        Updated input target table with an additional 'NUMBER_OF_OBS' column indicating the number of
        observations matched per target.

    Notes
    -----
    - The function assumes RA and DEC coordinates are in degrees (ICRS frame).
    - Observations are separated based on a one-hour gap threshold.
    - For polarimetric observations, the function will not compute derotation angles and set derotator_flag.
      It will ignore the derotator_flag for in the HCI_READY flag.
    """
    if instrument.lower() == 'ifs':
        polarimetry = False
    
    ndit_key = "NAXIS3" if "NAXIS3" in table_of_files.colnames else "NDIT"
    mode_key = "IFS_MODE" if instrument.lower() == "ifs" else "DB_FILTER"
    _, _, _, _, t_science = filter_for_science_frames(table_of_files, instrument, polarimetry, remove_fillers)
    cone_size = cone_size_science * u.arcsec

    obs_table_rows = []
    science_coords = SkyCoord(ra=t_science["RA"] * u.deg, dec=t_science["DEC"] * u.deg)

    for target in tqdm(table_of_targets):
        target_coords = SkyCoord(ra=target["RA_HEADER"] * u.deg, dec=target["DEC_HEADER"] * u.deg)
        matched_files = t_science[target_coords.separation(science_coords) < cone_size]

        if matched_files is None or len(matched_files) == 0:
            print(f"❌ No matching science files for target: {target['MAIN_ID']}")
            continue

        grouped_sequences = group_observation_sequences(
            matched_files,
            mode_key=mode_key,
            group_by_obs_number=group_by_time_gaps
        )

        for group_keys, obs_group in grouped_sequences:
            night = group_keys["NIGHT_START"]
            obs_number = group_keys["OBS_NUMBER"]
            mode = group_keys[mode_key]

            # Decide science frames based on total exposure time
            primary_type, active_science, exptime_per_type = select_primary_science_frames(obs_group, ndit_key)
            if len(active_science) == 0:
                print(f"⚠️ No usable science frames for {target['MAIN_ID']} on {night} in mode {mode}.")
                continue

            flags = evaluate_observation_flags(obs_group, ndit_key)
            # Replace obs_group with active_science for metadata
            obs_metadata = calculate_observation_metadata(
                observation_files=active_science, instrument=instrument, polarimetry=polarimetry,
                 ndit_key=ndit_key, mode_key=mode_key,
            )

            # Combine both sets of metadata
            obs_metadata.update(flags)
            # ------------------------------------------------------------------
            # Compute the new high‑contrast‑pipeline readiness flag
            # ------------------------------------------------------------------
            has_center = obs_metadata["NCENTER"] > 0 and not obs_metadata["CENTER_FLAG"]
            has_flux   = obs_metadata["NFLUX"]   > 0 and not obs_metadata["FLUX_FLAG"]
            dit_issues = any(
                obs_metadata.get(flag, False)
                for flag in ("CENTER_DIT_FLAG", "FLUX_DIT_FLAG", "CORO_DIT_FLAG")
            )
            obs_metadata["HCI_READY"] = (
                has_center
                and has_flux
                and not dit_issues
                and (polarimetry or not obs_metadata["DEROTATOR_FLAG"])
            )

            # Add all target metadata to the observation row
            for colname in target.colnames:
                obs_metadata[colname] = target[colname]

            # Add observation-specific metadata
            obs_metadata.update({
                "OBS_NUMBER": obs_number,
                "INSTRUMENT": instrument.lower(),
                "POLARIMETRY": polarimetry,
                "FILTER": mode,
                "NIGHT_START": night,
                "PRIMARY_SCIENCE": primary_type,
                "WAFFLE_MODE": primary_type == "CENTER",
            })
            obs_metadata.update(exptime_per_type) # Add total exptimes for each type
            obs_table_rows.append(obs_metadata)

    obs_table = Table(rows=obs_table_rows)

    if reorder_columns:
        preferred_column_order = [
            # Target metadata
            "MAIN_ID", "OBJ_HEADER", "RA_HEADER", "DEC_HEADER", "RA_DEG", "DEC_DEG",
            "ID_HD", "ID_HIP", "ID_TYC", "ID_GAIA_DR3", "ID_2MASS",
            "SP_TYPE", "OTYPE", "DISTANCE", "PLX", "PLX_ERROR", "PLX_BIBCODE",
            "PMRA", "PMDEC", "PM_ERR_MAJA", "PM_ERR_MINA",
            "RV_VALUE", "RVZ_ERROR",
            "POS_DIFF", "POS_DIFF_ORIG", "STARS_IN_CONE",
            "FLUX_V", "FLUX_R", "FLUX_I", "FLUX_J", "FLUX_H", "FLUX_K", 

            # Instrument setup
            "INSTRUMENT", "POLARIMETRY", "FILTER", "IFS_MODE", "DB_FILTER",
            "ND_FILTER", "ND_FILTER_FLUX", "DEROTATOR_MODE", 
            "PRIMARY_SCIENCE", "WAFFLE_MODE",

            # Observation time/grouping
            "NIGHT_START", "OBS_NUMBER", "OBS_START", "OBS_END", "MJD_MEAN", 

            # Exposure settings
            "DIT", "NDIT", "NCUBES", "DIT_FLUX", "NDIT_FLUX", "NFLUX", 
            "DIT_CENTER", "NDIT_CENTER", "NCENTER", "DIT_CORO", "NDIT_CORO",
            "TOTAL_EXPTIME_SCI", "TOTAL_EXPTIME_FLUX",
            "TOTAL_EXPTIME_CENTER", "TOTAL_EXPTIME_CORO",

            # Quality flags
            "HCI_READY", "FLUX_FLAG", "FLUX_DIT_FLAG", "CENTER_FLAG", "CENTER_DIT_FLAG",
            "CORO_FLAG", "CORO_DIT_FLAG", "DEROTATOR_FLAG",

            # Atmospheric conditions
            "MEAN_TAU", "STDDEV_TAU", "MEAN_FWHM", "STDDEV_FWHM", "MEAN_AIRMASS",

            # Angular coverage / derotation
            "ROTATION", "WAFFLE_AMP",

            # Program/archive info
            "OBS_PROG_ID", "OBS_ID", "TOTAL_FILE_SIZE_MB"
        ]

        existing_cols = [col for col in preferred_column_order if col in obs_table.colnames]
        remaining_cols = [col for col in obs_table.colnames if col not in existing_cols]
        obs_table = obs_table[existing_cols + remaining_cols]
    if not group_by_time_gaps:
        obs_table.remove_column("OBS_NUMBER")

    return obs_table, table_of_targets
