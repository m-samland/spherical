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

import logging
import os
import shutil
import time
from glob import glob
from pathlib import Path
from typing import Iterable, List, Sequence

from astropy.table import Table, setdiff, vstack
from astroquery.eso import Eso

__all__ = ["convert_paths_to_filenames", "download_data_for_observation"]

_LOG = logging.getLogger(__name__)


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


def _extract_dp_id(path: Path) -> str:
    """
    Return the ESO dataset identifier (**DP.ID**) corresponding to *path*.

    Parameters
    ----------
    path : pathlib.Path
        A file with suffix ``.fits`` or ``.fits.Z``.

    Returns
    -------
    str
        The filename without the FITS extension(s).

    Examples
    --------
    >>> _extract_dp_id(Path("SPHERE.2023-10-13T20:17:27.018.fits"))
    'SPHERE.2023-10-13T20:17:27.018'
    >>> _extract_dp_id(Path("SPHERE.2023-10-13T20:17:27.018.fits.Z"))
    'SPHERE.2023-10-13T20:17:27.018'
    """
    name = path.name
    if name.endswith(".fits.Z"):
        return name[:-7]
    if name.endswith(".fits"):
        return name[:-5]
    return path.stem


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
    return [_extract_dp_id(Path(p)) for p in full_paths]


def download_data_for_observation(
    raw_directory: str | os.PathLike,
    observation,
    eso_username: str | None = None,
    extra_calibration_keys: Sequence[str] | None = None,
) -> None:
    """
    Download all files required by *observation* and arrange them on disk.

    Parameters
    ----------
    raw_directory : str or os.PathLike
        Root directory where the raw files will be stored.  The function will
        create the standard ``<instrument>/<science|calibration>/…`` hierarchy
        inside this folder if it does not yet exist.
    observation : IFSObservation or IRDISObservation
        An instantiated observation object as defined elsewhere in the
        pipeline.  Its ``frames`` attribute must contain ``astropy.table.Table``
        objects with a ``DP.ID`` column.
    eso_username : str or None, optional
        ESO archive username.  If *None*, the function tries to download
        anonymously.
    extra_calibration_keys : sequence of str or None, optional
        Additional calibration frame *categories* that should be **treated like
        the default ones for the corresponding instrument**.  Each key must
        match exactly the names used in ``observation.frames``.  Use this when
        new reduction steps require further reference files.
        For example: 'SPECPOS', 'BG_WAVECAL', 'BG_SPECPOS', 'FLAT' for IFS
        or 'FLAT' and 'BG_FLUX' for IRDIS.

    Notes
    -----
    By default the function downloads and moves:

    * **IFS**   → ``WAVECAL``
    * **IRDIS** → ``FLAT`` and ``BG_SCIENCE``

    Any *extra_calibration_keys* are appended to that list **per call**.

    The routine is idempotent: running it again only triggers downloads for
    missing files.

    Examples
    --------
    >>> download_data_for_observation(
    ...     "/data/eso_raw",
    ...     obs,
    ...     eso_username="john_doe",
    ...     extra_calibration_keys=("BG_FLUX",),
    ... )
    """
    # ---------------------------------------------------------------------#
    # 0.  Basic constants & paths
    # ---------------------------------------------------------------------#
    raw_root = Path(raw_directory).expanduser().resolve()
    instrument = "IFS" if observation.filter in {"OBS_H", "OBS_YJ"} else "IRDIS"
    _LOG.info("Download attempted for %s", instrument)

    science_keys: tuple[str, ...] = ("CORO", "CENTER", "FLUX")

    calib_keys_by_instrument: dict[str, tuple[str, ...]] = {
        "IFS": ("WAVECAL",),
        "IRDIS": ("FLAT", "BG_SCIENCE"),
    }
    calibration_keys: tuple[str, ...] = calib_keys_by_instrument[instrument]

    if extra_calibration_keys:
        # Preserve order while avoiding duplicates
        calibration_keys = tuple(
            dict.fromkeys((*calibration_keys, *extra_calibration_keys))
        )

    # ---------------------------------------------------------------------#
    # 1.  Folder skeleton
    # ---------------------------------------------------------------------#
    science_root = raw_root / instrument / "science"
    calib_root = raw_root / instrument / "calibration"
    _ensure_dir(science_root)
    _ensure_dir(calib_root)

    target_name = "_".join(observation.observation["MAIN_ID"][0].split())
    obs_band = observation.filter
    obs_date = observation.observation["NIGHT_START"][0]

    target_directory = science_root / target_name / obs_band / obs_date
    _ensure_dir(target_directory)

    calib_root_band = calib_root / obs_band
    _ensure_dir(calib_root_band)

    # ---------------------------------------------------------------------#
    # 2.  Determine what needs downloading
    # ---------------------------------------------------------------------#
    needed_keys = (*science_keys, *calibration_keys)
    all_frames = vstack(
        [
            observation.frames[k]
            for k in needed_keys
            if k in observation.frames
            and observation.frames[k] is not None
            and len(observation.frames[k]) > 0
        ]
    )
    required_ids = Table({"DP.ID": all_frames["DP.ID"]})

    existing_ids = convert_paths_to_filenames(
        glob(str(raw_root / "**" / "SPHER.*"), recursive=True)
    )
    download_list = (
        setdiff(required_ids, Table({"DP.ID": existing_ids}), keys=["DP.ID"])
        if existing_ids
        else required_ids
    )

    # ---------------------------------------------------------------------#
    # 3.  Download via astroquery.eso
    # ---------------------------------------------------------------------#
    if len(download_list) > 0:
        eso = Eso()
        if eso_username:
            eso.login(username=eso_username)

        _LOG.info("Retrieving %d file(s)…", len(download_list))
        eso.retrieve_data(
            datasets=list(download_list["DP.ID"].data),
            destination=str(raw_root),
            with_calib=None,
            unzip=True,
        )
        time.sleep(3)  # filesystem latency
    else:
        _LOG.info("All required files already present – nothing to download.")

    # ---------------------------------------------------------------------#
    # 4.  Move into final locations
    # ---------------------------------------------------------------------#
    def _move_ids(ids: Iterable[str], destination: Path) -> None:
        """
        Move the given DP.IDs to *destination*.

        The function is silent if the target file is already present in
        *destination*.

        Parameters
        ----------
        ids : iterable of str
            Dataset identifiers (without FITS extension).
        destination : pathlib.Path
            Folder where the FITS files should live.
        """
        _ensure_dir(destination)
        for dp_id in ids:
            # 1. already at destination?
            if any(
                (destination / f"{dp_id}{ext}").exists() for ext in (".fits", ".fits.Z")
            ):
                continue

            # 2. still in raw_root?
            moved = False
            for ext in (".fits", ".fits.Z"):
                src = raw_root / f"{dp_id}{ext}"
                if src.exists():
                    shutil.move(src, destination / src.name)
                    moved = True
                    break

            # 3. nowhere to be found
            if not moved:
                _LOG.warning(
                    "Expected file %s.[fits|fits.Z] not found in source or destination",
                    dp_id,
                )

    # Science frames
    for key in science_keys:
        tbl = observation.frames.get(key)
        if tbl is not None and len(tbl) > 0:
            _move_ids(tbl["DP.ID"], target_directory / key)

    # Calibration frames
    for key in calibration_keys:
        tbl = observation.frames.get(key)
        if tbl is not None and len(tbl) > 0:
            _move_ids(tbl["DP.ID"], calib_root_band / key)

    _LOG.info(
        "Download & sorting complete for %s / %s / %s",
        target_name,
        obs_band,
        obs_date,
    )
