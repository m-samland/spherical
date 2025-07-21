"""
Utilities to download and organise raw ESO/SPHERE data.

The public surface is intentionally minimal:

* ``convert_paths_to_filenames`` – strip file extensions and return the DP.ID
  of a list of FITS(.​Z) files.
* ``download_data_for_observation`` – fetch all science + calibration frames
  for an ``IFSObservation``/``IRDISObservation`` instance and move them into the
  canonical folder hierarchy.

All functions are safe to import in multiprocessing contexts and thread‑safe
with respect to directory creation.
"""
from __future__ import annotations

import os
import shutil
import time
import warnings
from glob import glob
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from astropy.table import Table, setdiff, vstack
from astroquery.eso import Eso

from spherical.pipeline.logging_utils import optional_logger

__all__ = ["convert_paths_to_filenames", "download_data_for_observation", "update_observation_file_paths"]


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _ensure_dir(path: Path) -> None:
    """
    Recursively create *path* if it does not yet exist.

    Parameters
    ----------
    path : pathlib.Path
        Target directory.
    """
    path.mkdir(parents=True, exist_ok=True)


def _dp_id_from_path(p: str | Path) -> str:
    """
    Strip ``.fits`` or ``.fits.Z`` and return the ESO *DP.ID*.

    Parameters
    ----------
    p : str or pathlib.Path
        File path.

    Returns
    -------
    str
        Dataset identifier.
    """
    name = Path(p).name
    if name.endswith(".fits.Z"):
        return name[:-7]
    if name.endswith(".fits"):
        return name[:-5]
    return Path(p).stem


# -----------------------------------------------------------------------------#
# Public utilities
# -----------------------------------------------------------------------------#
def convert_paths_to_filenames(full_paths: Iterable[str | os.PathLike]) -> List[str]:
    """
    Convert filesystem paths to ESO dataset identifiers (*DP.ID*).

    Parameters
    ----------
    full_paths : iterable of str or os.PathLike
        Absolute or relative paths pointing to FITS files.

    Returns
    -------
    list of str
        Each entry is the DP.ID corresponding to the input path.

    Examples
    --------
    >>> convert_paths_to_filenames(["/SPHERE.2023-10-13T20:17:27.018.fits"])
    ['SPHERE.2023-10-13T20:17:27.018']
    """
    return [_dp_id_from_path(Path(p)) for p in full_paths]


@optional_logger
def download_data_for_observation(
    raw_directory: str | os.PathLike,
    observation,
    logger,
    eso_username: str | None = None,
    store_password: bool = True,
    extra_calibration_keys: Sequence[str] | None = None,
) -> None:
    """Download and organize raw SPHERE/IFS data files for a given observation.

    This is the first step in the SPHERE/IFS data reduction pipeline. It downloads
    all required science and calibration frames from the ESO archive and organizes
    them into a standardized directory structure.

    Required Input Files
    -------------------
    None - This is the first step in the pipeline.

    Generated Output Files
    ---------------------
    The following directory structure is created under raw_directory:
    
    IFS/
    ├── science/
    │   └── target_name/
    │       └── obs_band/
    │           └── date/
    │               ├── CORO/*.fits
    │               ├── CENTER/*.fits
    │               └── FLUX/*.fits
    └── calibration/
        └── obs_band/
            └── WAVECAL/*.fits

    Parameters
    ----------
    raw_directory : str or os.PathLike
        Root directory where raw files will be stored. The function creates
        the standard <instrument>/<science|calibration>/... hierarchy inside
        this folder.
    observation : IFSObservation or IRDISObservation
        An instantiated observation object containing frame information.
        Must have a 'frames' attribute containing astropy.table.Table objects
        with a 'DP.ID' column.
    logger :
        Logger instance for logging progress and issues.
    eso_username : str or None, optional
        ESO archive username. If None, attempts anonymous download.
    extra_calibration_keys : sequence of str or None, optional
        Additional calibration frame categories to treat like default ones.
        Each key must match names used in observation.frames. Use for new
        reduction steps requiring additional reference files.

    Notes
    -----
    - By default downloads:
        * IFS: WAVECAL frames
        * IRDIS: FLAT and BG_SCIENCE frames
    - Any extra_calibration_keys are appended to the default list
    - The routine is idempotent: running again only downloads missing files
    - Creates all necessary directories if they don't exist
    - Thread-safe with respect to directory creation

    Examples
    --------
    >>> download_data_for_observation(
    ...     "/data/eso_raw",
    ...     obs,
    ...     eso_username="john_doe",
    ...     extra_calibration_keys=("BG_FLUX",),
    ... )
    """
    logger.info("Starting download_data_for_observation", extra={"step": "download_data", "status": "started"})
    # ------------------------------------------------------------------#
    # 0.  Basic constants & paths
    # ------------------------------------------------------------------#
    raw_root = Path(raw_directory).expanduser().resolve()
    instrument = "IFS" if observation.filter in {"OBS_H", "OBS_YJ"} else "IRDIS"
    logger.debug(f"Download attempted for instrument: {instrument}")

    science_keys: tuple[str, ...] = ("CORO", "CENTER", "FLUX")
    calib_keys_by_instrument = {
        "IFS": ("WAVECAL",),
        "IRDIS": ("FLAT", "BG_SCIENCE"),
    }
    calibration_keys: tuple[str, ...] = calib_keys_by_instrument[instrument]
    if extra_calibration_keys:
        calibration_keys = tuple(dict.fromkeys((*calibration_keys, *extra_calibration_keys)))
    logger.debug(f"Science keys: {science_keys}, Calibration keys: {calibration_keys}")

    # ------------------------------------------------------------------#
    # 1.  Folder skeleton
    # ------------------------------------------------------------------#
    science_root = raw_root / instrument / "science"
    calib_root = raw_root / instrument / "calibration"
    _ensure_dir(science_root)
    _ensure_dir(calib_root)

    target_name = "_".join(observation.observation["MAIN_ID"][0].split())
    obs_band = observation.filter
    obs_date = observation.observation["NIGHT_START"][0]

    target_directory = science_root / target_name / obs_band / obs_date
    calib_root_band = calib_root / obs_band
    _ensure_dir(target_directory)
    _ensure_dir(calib_root_band)
    logger.debug(f"Target directory: {target_directory}, Calibration directory: {calib_root_band}")

    # ------------------------------------------------------------------#
    # 2.  Determine what needs downloading
    # ------------------------------------------------------------------#
    needed_keys = (*science_keys, *calibration_keys)
    all_frames = vstack(
        [
            observation.frames[k]
            for k in needed_keys
            if k in observation.frames and observation.frames[k] is not None and len(observation.frames[k]) > 0
        ]
    )
    required_ids = Table({"DP.ID": all_frames["DP.ID"]})

    # Build global lookup: DP.ID → absolute path found anywhere under raw_root
    existing_paths = glob(str(raw_root / "**" / "SPHER.*"), recursive=True)
    id_to_path: dict[str, Path] = {_dp_id_from_path(Path(p)): Path(p) for p in existing_paths}
    existing_ids = list(id_to_path)

    # Fix: astropy TableColumns is not always iterable, use flat list of str
    required_id_list = [str(x) for x in np.array(required_ids["DP.ID"]).flatten()]
    download_list = (
        setdiff(required_ids, Table({"DP.ID": existing_ids}), keys=["DP.ID"]) if existing_ids else required_ids
    )
    download_id_list = [str(x) for x in np.array(download_list["DP.ID"]).flatten()]
    logger.debug(f"Required IDs: {required_id_list}")
    logger.debug(f"Existing IDs: {existing_ids}")
    logger.debug(f"Download list: {download_id_list}")

    # ------------------------------------------------------------------#
    # 3.  Download via astroquery.eso
    # ------------------------------------------------------------------#
    if len(download_id_list) > 0:
        eso = Eso()
        if eso_username:
            eso.login(username=eso_username, store_password=store_password)

        logger.info(f"Retrieving {len(download_id_list)} file(s)…", extra={"step": "download_data"})
        eso.retrieve_data(
            datasets=download_id_list,
            destination=str(raw_root),
            with_calib=None,
            unzip=True,
        )
        time.sleep(3)  # filesystem latency

        # Update lookup with freshly downloaded files
        for dp_id in download_id_list:
            for ext in (".fits", ".fits.Z"):
                p = raw_root / f"{dp_id}{ext}"
                if p.exists():
                    id_to_path[dp_id] = p
                    break
    else:
        logger.info("All required files already present – nothing to download.", extra={"step": "download_data"})

    # ------------------------------------------------------------------#
    # 4.  Move files into final locations
    # ------------------------------------------------------------------#
    def _move_ids(ids: Iterable[str], destination: Path) -> None:
        """Move the given DP.IDs to *destination* if they are not there yet."""
        _ensure_dir(destination)
        for dp_id in ids:
            # Already in destination?
            if any((destination / f"{dp_id}{ext}").exists() for ext in (".fits", ".fits.Z")):
                continue

            src = id_to_path.get(dp_id)
            if src and src.exists():
                shutil.move(src, destination / src.name)
                id_to_path[dp_id] = destination / src.name  # update mapping
            else:
                logger.warning(
                    f"Expected file {dp_id}.[fits|fits.Z] not found in source or destination",
                    extra={"step": "download_data", "status": "failed"}
                )

    # science frames
    for key in science_keys:
        tbl = observation.frames.get(key)
        if tbl is not None and len(tbl) > 0:
            _move_ids([str(x) for x in np.array(tbl["DP.ID"]).flatten()], target_directory / key)

    # calibration frames
    for key in calibration_keys:
        tbl = observation.frames.get(key)
        if tbl is not None and len(tbl) > 0:
            _move_ids([str(x) for x in np.array(tbl["DP.ID"]).flatten()], calib_root_band / key)

    # ------------------------------------------------------------------#
    # 5.  Sanity check
    # ------------------------------------------------------------------#
    expected_ids = set(required_id_list)
    found_ids = set(id_to_path)
    missing_ids = expected_ids - found_ids
    if missing_ids:
        id_to_type = {
            dp_id: key
            for key in (*science_keys, *calibration_keys)
            for dp_id in ([str(x) for x in np.array(observation.frames.get(key, Table())["DP.ID"]).flatten()] if key in observation.frames and len(observation.frames[key]) > 0 else [])
        }
        details = ", ".join(f"{dp_id} ({id_to_type.get(dp_id, '?')})" for dp_id in sorted(missing_ids))
        logger.exception(
            f"The following datasets were not found after download+move: {details}",
            extra={"step": "download_data", "status": "failed"}
        )
        raise FileNotFoundError(
            f"The following datasets were not found after download+move: {details}"
        )

    logger.info("Download & sorting complete", extra={"step": "download_data", "status": "success"})


@optional_logger
def update_observation_file_paths(
    existing_file_paths: Sequence[str | Path],
    observation,
    logger,
    used_keys: Iterable[str] = ("CORO", "CENTER", "FLUX", "WAVECAL"),
) -> None:
    """
    Populate/overwrite the ``FILE`` column of *observation.frames* in‑place.

    Parameters
    ----------
    existing_file_paths
        Flat list or tuple of FITS (or FITS.Z) paths already present on disk.
    observation
        ``IFSObservation`` / ``IRDISObservation`` instance whose ``frames``
        attribute is a dict of `~astropy.table.Table` or *pandas* DataFrame.
    logger :
        Logger instance for logging progress and issues.
    used_keys
        Subset of frame categories (e.g. ``("CORO", "FLUX")``) that the caller
        needs for the current reduction step. Defaults to ("CORO", "CENTER", "FLUX", "WAVECAL").

    Notes
    -----
    * The function is **tolerant** of missing files – when a ``DP.ID`` cannot be
      matched, a warning is emitted and the corresponding element in
      ``FILE`` is set to ``None`` (or ``<NA>`` for pandas).
    * Empty or absent frame tables are skipped silently.
    * The function modifies the tables **in place** and returns *None*.
    """
    logger.info("Starting update_observation_file_paths", extra={"step": "update_observation_file_paths", "status": "started"})
    # ------------------------------------------------------------------#
    # 0.  Fast lookup: DP.ID → absolute path
    # ------------------------------------------------------------------#
    id_to_path = {
        _dp_id_from_path(p): str(p) for p in existing_file_paths
    }
    logger.debug(f"ID to path mapping: {id_to_path}")

    # ------------------------------------------------------------------#
    # 1.  Iterate over the requested frame types
    # ------------------------------------------------------------------#
    for key in used_keys:
        frame_tbl = observation.frames.get(key)
        if frame_tbl is None or len(frame_tbl) == 0:
            logger.debug(f"Frame list for key '{key}' is empty – skipped.")
            continue

        # a) Make sure we are working with a DataFrame for convenience
        if isinstance(frame_tbl, Table):
            df = frame_tbl.to_pandas()
        elif isinstance(frame_tbl, pd.DataFrame):
            df = frame_tbl.copy()
        else:
            raise TypeError(
                f"Unsupported table type for key '{key}': {type(frame_tbl)}"
            )

        # b) Decode existing FILE column if it contains bytes
        if "FILE" in df.columns and df["FILE"].dtype == object:
            # Only decode bytes objects; leave str untouched
            df["FILE"] = df["FILE"].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            )

        # c) Map DP.ID → absolute path
        missing_ids = []
        df["FILE"] = [
            id_to_path.get(dp_id) if dp_id in id_to_path else None
            for dp_id in df["DP.ID"]
        ]
        if df["FILE"].isna().any():
            missing_ids = df.loc[df["FILE"].isna(), "DP.ID"].tolist()

        # d) Write back into observation.frames in the original format
        if isinstance(frame_tbl, Table):
            # Need numpy string dtype for FITS compatibility
            frame_tbl["FILE"] = np.asarray(df["FILE"].fillna("").values, dtype="U256")
        else:  # pandas DataFrame
            observation.frames[key] = df

        # e) Log any unresolved IDs
        if missing_ids:
            warnings.warn(
                f"{len(missing_ids)} file(s) for frame type '{key}' "
                "could not be matched on disk – entries left as <missing>.",
                RuntimeWarning,
            )
            logger.warning(
                f"Missing files for key '{key}': {', '.join(missing_ids[:5]) + ('…' if len(missing_ids) > 5 else '')}",
                extra={"step": "update_observation_file_paths", "status": "failed"}
            )
    logger.info("update_observation_file_paths complete", extra={"step": "update_observation_file_paths", "status": "success"})
