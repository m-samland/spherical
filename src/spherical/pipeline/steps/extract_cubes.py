"""
Step: Extract IFS data cubes with multiprocessing support.

This module provides the extract_cubes_with_multiprocessing function and its helper for use in the IFS pipeline.
"""

import logging
import os
from multiprocessing import Pool, cpu_count
from typing import Any, Collection, Dict, List, Tuple

import charis
from astropy.io import fits
from tqdm import tqdm

from spherical.database.database_utils import find_nearest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_cubes_with_multiprocessing(
    observation: Any,
    frame_types_to_extract: Collection[str],
    extraction_parameters: Dict[str, Any],
    reduction_parameters: Dict[str, Any],
    wavecal_outputdir: str,
    cube_outputdir: str,
    non_least_square_methods: Collection[str] = ('optext', 'apphot3', 'apphot5'),
    *,
    extract_cubes: bool = True,
) -> None:
    r"""
    Extract data cubes from observations, optionally in parallel.

    This function processes specified frame types (e.g., 'CORO', 'CENTER', 'FLUX')
    from the given ``observation`` object and extracts data cubes based on
    the provided parameters. The extraction can run in parallel unless
    ``ncpu_cubebuilding == 1``, in which case it runs serially.

    Parameters
    ----------
    observation : object
        Object containing frames and background data (e.g., ``observation.frames[key]``).
        Typically includes relevant file paths, headers, and observational metadata.
    frame_types_to_extract : list of str
        List of frame types to be reduced (e.g. ['CORO', 'CENTER', 'FLUX']).
    extraction_parameters : dict
        Dictionary of parameters controlling extraction details such as
        background subtraction (``bgsub``) or fitting (``fitbkgnd``), among others.
    reduction_parameters : dict
        Dictionary with additional settings controlling the reduction, e.g.,
        number of CPUs for parallel processing (``ncpu_cubebuilding``),
        whether PCA background subtraction is used (``bg_pca``),
        and whether to subtract CORO frames from CENTER (``subtract_coro_from_center``).
    wavecal_outputdir : str
        Path to the directory where wavelength calibration outputs are located.
    cube_outputdir : str
        Top-level directory where extracted cubes will be written.
    non_least_square_methods : set or list
        Methods indicating use of a non-linear approach that requires special
        wavelength handling.
    extract_cubes : bool, optional
        If ``False``, extraction is skipped, though the pipeline logic is maintained.
        Defaults to ``True``.

    Returns
    -------
    None
        This function does not return anything. It writes extracted cubes to
        the specified output directory.

    Notes
    -----
    - If both ``bgsub`` and ``fitbkgnd`` are set in ``extraction_parameters``,
      a ValueError is raised.
    - For frame type ``CENTER``, if ``subtract_coro_from_center`` is ``True``
      and CORO frames exist, those frames are used as background instead of PCA.
    - For frame type ``FLUX``, the ``fitshift`` parameter is forced to ``False``.
    - If parallelization is used (``ncpu_cubebuilding`` > 1), tasks are collected
      and passed to a multiprocessing pool. If a task raises a ``ValueError`` inside
      ``_parallel_extraction_worker``, the error is logged and the pool continues
      with remaining tasks.
    - In single-threaded mode (``ncpu_cubebuilding == 1``), each task is processed
      immediately in a for-loop.

    Examples
    --------
    >>> extract_cubes_with_multiprocessing(
    ...     observation=my_obs,
    ...     frame_types_to_extract=["CORO", "CENTER", "FLUX"],
    ...     extraction_parameters={"bgsub": True, "fitbkgnd": False, ...},
    ...     reduction_parameters={"ncpu_cubebuilding": 8, "bg_pca": False, ...},
    ...     instrument=my_instrument,
    ...     wavecal_outputdir="/path/to/wavecal",
    ...     cube_outputdir="/path/to/output",
    ...     non_least_square_methods={"some_method"},
    ...     extract_cubes=True
    ... )

    """

    # Ensure the output directory exists.
    os.makedirs(cube_outputdir, exist_ok=True)

    if not extract_cubes:
        logger.info("Extraction disabled by caller – returning early.")
        return

    if extraction_parameters.get('bgsub') and extraction_parameters.get('fitbkgnd'):
        raise ValueError("Cannot enable both 'bgsub' and 'fitbkgnd'.")

    # If PCA background is requested we do *not* pass a file path to getcube.
    if reduction_parameters.get('bg_pca'):
        bgpath_global: str | None = None
    else:
        try:
            bgpath_global = observation.frames['BG_SCIENCE']['FILE'].iloc[-1]
        except (KeyError, IndexError):
            logger.info("No BG_SCIENCE frame found – falling back to PCA fit.")
            bgpath_global = None

    fitshift_original: bool = extraction_parameters.get('fitshift', False)

    # ---------------------------------------------------------------------
    # Collect *every* extraction task first
    # ---------------------------------------------------------------------
    tasks: List[Tuple[str, int, str | None, str, str, Dict[str, Any]]] = []

    for key in frame_types_to_extract:
        if len(observation.frames.get(key, [])) == 0:
            logger.info("No files to reduce for frame‑type %s.", key)
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
                    logger.debug("CENTER [%d] uses CORO background: %s", idx, bg_frame)
                else:
                    extraction_parameters['bg_scaling_without_mask'] = False
                    bg_frame = None
                    logger.debug("CENTER [%d] uses PCA background.", idx)
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
                    charis.extractcube.getcube(
                        filename=filename,
                        dit=dit_index,
                        bgpath=bg_frame,
                        calibdir=wavecal_outputdir,
                        outdir=cube_type_outputdir,
                        **extraction_parameters,
                    )
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
            len(tasks),
            ncpus,
        )
        try:
            with Pool(processes=ncpus) as pool:
                results = pool.map(_parallel_extraction_worker, tasks)
            num_failed = sum(not ok for ok in results)
            if num_failed:
                logger.warning(
                    "%d extraction task(s) failed with ValueError and were skipped.",
                    num_failed,
                )
        except Exception:
            logger.exception("Unexpected exception during parallel extraction.")
            raise


def _parallel_extraction_worker(
    task: Tuple[str, int, str | None, str, str, Dict[str, Any]],
) -> bool:
    """
    Worker helper for :func:`extract_cubes_with_multiprocessing`.

    Parameters
    ----------
    task
        ``(filename, dit_index, bg_frame, wavecal_outputdir, outputdir,
        extraction_params)``
    Returns
    -------
    bool
        ``True`` on success, ``False`` if ``charis.extractcube.getcube`` raised
        *ValueError* (all other exceptions propagate).
    """
    (
        filename,
        dit_index,
        bg_frame,
        wavecal_outputdir,
        outputdir,
        extraction_params,
    ) = task

    try:
        charis.extractcube.getcube(
            filename=filename,
            dit=dit_index,
            bgpath=bg_frame,
            calibdir=wavecal_outputdir,
            outdir=outputdir,
            **extraction_params,
        )
        return True
    except ValueError as err:
        logger.warning(
            "[Worker] ValueError in %s (DIT=%d): %s", filename, dit_index, err
        )
        return False
