"""
Step: Extract IFS data cubes with multiprocessing support.

This module provides the extract_cubes_with_multiprocessing function and its helper for use in the IFS pipeline.
"""

import os
from multiprocessing import Pool, cpu_count
from typing import Any, Collection, Dict, List, Tuple

import charis
from astropy.io import fits
from tqdm import tqdm

from spherical.database.database_utils import find_nearest
from spherical.pipeline.logging_utils import optional_logger


@optional_logger
def extract_cubes_with_multiprocessing(
    observation: Any,
    frame_types_to_extract: Collection[str],
    extraction_parameters: Dict[str, Any],
    reduction_parameters: Dict[str, Any],
    wavecal_outputdir: str,
    cube_outputdir: str,
    *,
    logger,
    non_least_square_methods: Collection[str] = ('optext', 'apphot3', 'apphot5'),
    extract_cubes: bool = True,
) -> None:
    """Extract data cubes from SPHERE/IFS observations with optional parallel processing.

    This is the third step in the SPHERE/IFS data reduction pipeline. It processes
    raw FITS files into wavelength-calibrated data cubes for different frame types
    (CORO, CENTER, FLUX) with optional background subtraction and parallel processing.

    Required Input Files
    -------------------
    From previous steps:
    1. From download_data:
        - IFS/science/target_name/obs_band/date/CORO/*.fits
        - IFS/science/target_name/obs_band/date/CENTER/*.fits
        - IFS/science/target_name/obs_band/date/FLUX/*.fits
    2. From wavelength_calibration:
        - wavecal_outputdir/*key*.fits
            Wavelength calibration key files

    Generated Output Files
    ---------------------
    In cube_outputdir:
    - CORO/
        - cube_*.fits
            Extracted and wavelength-calibrated coronagraphic data cubes
    - CENTER/
        - cube_*.fits
            Extracted and wavelength-calibrated center data cubes
    - FLUX/
        - cube_*.fits
            Extracted and wavelength-calibrated flux data cubes

    Parameters
    ----------
    observation : object
        Observation object containing frames and background data. Must have:
        - frames[key]['FILE']: List of input FITS file paths
        - frames[key]['MJD_OBS']: List of observation times
        - frames['BG_SCIENCE']['FILE']: Optional background frame path
    frame_types_to_extract : collection of str
        Frame types to process (e.g., ['CORO', 'CENTER', 'FLUX']).
    extraction_parameters : dict
        Parameters controlling extraction:
        - bgsub: bool
            Whether to subtract background
        - fitbkgnd: bool
            Whether to fit background
        - fitshift: bool
            Whether to fit shifts (forced False for FLUX)
        - bg_scaling_without_mask: bool
            Whether to scale background without mask
    reduction_parameters : dict
        Reduction settings:
        - ncpu_cubebuilding: int
            Number of CPUs for parallel processing
        - bg_pca: bool
            Whether to use PCA background subtraction
        - subtract_coro_from_center: bool
            Whether to use CORO frames as background for CENTER
    wavecal_outputdir : str
        Directory containing wavelength calibration products.
    cube_outputdir : str
        Directory where extracted cubes will be written.
    non_least_square_methods : collection of str, optional
        Methods requiring special wavelength handling. Default is
        ('optext', 'apphot3', 'apphot5').
    extract_cubes : bool, optional
        If False, skip extraction. Default is True.

    Returns
    -------
    None
        This function writes extracted cubes to disk and does not return a value.

    Notes
    -----
    - Cannot enable both 'bgsub' and 'fitbkgnd' simultaneously
    - For CENTER frames, can use CORO frames as background if available
    - For FLUX frames, fitshift is always forced to False
    - Uses multiprocessing for parallel extraction when ncpu_cubebuilding > 1
    - Creates output directories if they don't exist
    - Logs errors but continues processing if individual extractions fail

    Examples
    --------
    >>> extract_cubes_with_multiprocessing(
    ...     observation=obs,
    ...     frame_types_to_extract=["CORO", "CENTER", "FLUX"],
    ...     extraction_parameters={
    ...         "bgsub": True,
    ...         "fitbkgnd": False,
    ...         "fitshift": True
    ...     },
    ...     reduction_parameters={
    ...         "ncpu_cubebuilding": 8,
    ...         "bg_pca": False,
    ...         "subtract_coro_from_center": True
    ...     },
    ...     wavecal_outputdir="/path/to/wavecal",
    ...     cube_outputdir="/path/to/output"
    ... )
    """

    # Ensure the output directory exists.
    os.makedirs(cube_outputdir, exist_ok=True)

    logger.info("Starting extract_cubes_with_multiprocessing", extra={"step": "extract_cubes", "status": "started"})

    if not extract_cubes:
        logger.info("Extraction disabled by caller – returning early.", extra={"step": "extract_cubes", "status": "skipped"})
        return

    if extraction_parameters.get('bgsub') and extraction_parameters.get('fitbkgnd'):
        logger.error("Cannot enable both 'bgsub' and 'fitbkgnd' simultaneously.", extra={"step": "extract_cubes", "status": "failed"})
        raise ValueError("Cannot enable both 'bgsub' and 'fitbkgnd'.")

    # If PCA background is requested we do *not* pass a file path to getcube.
    if reduction_parameters.get('bg_pca'):
        bgpath_global: str | None = None
    else:
        try:
            bgpath_global = observation.frames['BG_SCIENCE']['FILE'].iloc[-1]
        except (KeyError, IndexError):
            logger.info("No BG_SCIENCE frame found – falling back to PCA fit.", extra={"step": "extract_cubes", "status": "info"})
            bgpath_global = None

    fitshift_original: bool = extraction_parameters.get('fitshift', False)

    # ---------------------------------------------------------------------
    # Collect *every* extraction task first
    # ---------------------------------------------------------------------
    tasks: List[Tuple[str, int, str | None, str, str, Dict[str, Any]]] = []

    for key in frame_types_to_extract:
        if len(observation.frames.get(key, [])) == 0:
            logger.info(f"No files to reduce for frame‑type {key}.", extra={"step": "extract_cubes", "status": "skipped"})
            continue

        # Ensure output folder exists *per* frame type
        cube_type_outputdir = os.path.join(cube_outputdir, key)
        os.makedirs(cube_type_outputdir, exist_ok=True)

        for idx, filename in tqdm(
            enumerate(observation.frames[key]['FILE']),
            desc=f"Collect tasks for {key}",
            unit="file",
            leave=False,
        ):
            hdr = fits.getheader(filename)
            ndit: int = int(hdr['HIERARCH ESO DET NDIT'])

            # -----------------------------------------------------------------
            # Derive background frame for this particular file
            # -----------------------------------------------------------------
            if key == 'CENTER':
                if (
                    len(observation.frames.get('CORO', [])) > 0
                    and reduction_parameters.get('subtract_coro_from_center', False)
                ):
                    extraction_parameters['bg_scaling_without_mask'] = True
                    idx_nearest = find_nearest(  # type: ignore[name-defined]
                        observation.frames['CORO']['MJD_OBS'].values,
                        observation.frames['CENTER'].iloc[idx]['MJD_OBS'],
                    )
                    bg_frame: str | None = observation.frames['CORO']['FILE'].iloc[idx_nearest]
                    logger.debug(
                        "CENTER [%d] uses CORO background: %s",
                        idx, bg_frame,
                        extra={"step": "extract_cubes"}
                    )
                else:
                    extraction_parameters['bg_scaling_without_mask'] = False
                    bg_frame = None
                    logger.debug(
                        "CENTER [%d] uses PCA background.",
                        idx,
                        extra={"step": "extract_cubes"}
                    )
            else:
                extraction_parameters['bg_scaling_without_mask'] = False
                bg_frame = bgpath_global

            # -----------------------------------------------------------------
            # Frame‑type‑specific flag tweaks
            # -----------------------------------------------------------------
            if key == 'FLUX':
                extraction_parameters['fitshift'] = False
            else:
                extraction_parameters['fitshift'] = fitshift_original

            # -----------------------------------------------------------------
            # Serial extraction if only one CPU requested
            # -----------------------------------------------------------------
            if reduction_parameters.get('ncpu_cubebuilding', 1) == 1:
                for dit_index in tqdm(
                    range(ndit),
                    desc=f"Extract {key} (1 CPU)",
                    unit="DIT",
                    leave=False,
                ):
                    try:
                        logger.debug(
                            "Extracting cube: %s, DIT=%d, bg_frame=%s",
                            filename, dit_index, bg_frame,
                            extra={"step": "extract_cubes"}
                        )
                        charis.extractcube.getcube(
                            filename=filename,
                            dit=dit_index,
                            bgpath=bg_frame,
                            calibdir=wavecal_outputdir,
                            outdir=cube_type_outputdir,
                            **extraction_parameters,
                        )
                    except ValueError as err:
                        logger.warning(
                            "ValueError in %s (DIT=%d): %s",
                            filename, dit_index, err,
                            extra={"step": "extract_cubes", "status": "failed"}
                        )
                    except Exception:
                        logger.exception(
                            "Unexpected exception in %s (DIT=%d)",
                            filename, dit_index,
                            extra={"step": "extract_cubes", "status": "failed"}
                        )
                        raise
            else:
                # Defer to the multiprocessing pool
                for dit_index in range(ndit):
                    tasks.append(
                        (
                            filename,
                            dit_index,
                            bg_frame,
                            wavecal_outputdir,
                            cube_type_outputdir,
                            extraction_parameters.copy(),
                        )
                    )

            # Reset mutable flags for next file
            extraction_parameters['bg_scaling_without_mask'] = False
            extraction_parameters['fitshift'] = fitshift_original

    # ---------------------------------------------------------------------
    # Parallel processing of all accumulated tasks
    # ---------------------------------------------------------------------
    if tasks and reduction_parameters.get('ncpu_cubebuilding', 1) != 1:
        ncpus = min(reduction_parameters['ncpu_cubebuilding'], cpu_count())
        logger.info(
            "Starting parallel extraction of %d DITs using up to %d worker(s).",
            len(tasks), ncpus,
            extra={"step": "extract_cubes", "status": "started"}
        )
        try:
            with Pool(processes=ncpus) as pool:
                results = pool.starmap(
                    _parallel_extraction_worker,
                    [(task, logger.name) for task in tasks]
                )
            num_failed = sum(not ok for ok in results)
            if num_failed:
                logger.warning(
                    "%d extraction task(s) failed with ValueError and were skipped.",
                    num_failed,
                    extra={"step": "extract_cubes", "status": "partial-failure"}
                )
        except Exception:
            logger.exception("Unexpected exception during parallel extraction.", extra={"step": "extract_cubes", "status": "failed"})
            raise

    logger.info("Step finished", extra={"step": "extract_cubes", "status": "success"})


def _parallel_extraction_worker(
    task: Tuple[str, int, str | None, str, str, Dict[str, Any]],
    logger_name: str,
) -> bool:
    """
    Worker helper for :func:`extract_cubes_with_multiprocessing`.

    Parameters
    ----------
    task
        ``(filename, dit_index, bg_frame, wavecal_outputdir, outputdir,
        extraction_params)``
    logger_name
        Name of the logger to use in the worker process.
    Returns
    -------
    bool
        ``True`` on success, ``False`` if ``charis.extractcube.getcube`` raised
        *ValueError* (all other exceptions propagate).
    """
    import logging

    logger = logging.getLogger(logger_name)
    (
        filename,
        dit_index,
        bg_frame,
        wavecal_outputdir,
        outputdir,
        extraction_params,
    ) = task

    try:
        logger.debug(
            "[Worker] Extracting cube: %s, DIT=%d, bg_frame=%s",
            filename, dit_index, bg_frame,
            extra={"step": "extract_cubes"}
        )
        charis.extractcube.getcube(
            filename=filename,
            dit=dit_index,
            bgpath=bg_frame,
            calibdir=wavecal_outputdir,
            outdir=outputdir,
            **extraction_params,
        )
        return True
    except ValueError:
        logger.warning(
            "[Worker] Extraction failed for cube: %s, DIT=%d, bg_frame=%s",
            filename, dit_index, bg_frame,
            extra={"step": "extract_cubes", "status": "failed"}
        )
        return False
    except Exception:
        logger.exception(
            "[Worker] Unexpected exception in %s (DIT=%d)",
            filename, dit_index,
            extra={"step": "extract_cubes", "status": "failed"}
        )
        raise
