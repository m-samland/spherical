"""
FITS header update step for SPHERE IFS data reduction pipeline.

This module updates the FITS headers of reduced data cubes with pipeline metadata
including software versions, processing parameters, and provenance information.
This step runs after bundle_output to add comprehensive metadata to the final
data products.
"""

import os

from astropy.io import fits

from spherical.pipeline.fits.headers import update_cube_fits_header_after_reduction
from spherical.pipeline.logging_utils import optional_logger
from spherical.pipeline.science_frames import WAFFLE_KEYWORD


def _stamp_waffle_mode(converted_dir, frame_types, continuous_satellite_spots, logger):
    """Record WAFFLE_MODE on every cube that exists.

    The science frame type follows from this flag (see
    ``science_frames.science_frame_type``), and the cubes are the only place a
    consumer outside the pipeline can learn it. Written after the metadata pass
    so nothing can clobber it.

    Returns:
        True when every cube that exists was stamped. A cube that is not there
        is not a failure; a cube that could not be written is.
    """
    value = bool(continuous_satellite_spots)
    stamped_all = True
    for frame_type in frame_types:
        path = os.path.join(converted_dir, f"{frame_type.lower()}_cube.fits")
        if not os.path.exists(path):
            continue
        try:
            with fits.open(path, mode="update") as hdulist:
                # No comment: the HIERARCH key already fills most of the card,
                # and astropy truncates anything appended to it.
                hdulist[0].header[WAFFLE_KEYWORD] = value
        except Exception:
            stamped_all = False
            logger.exception(
                f"Failed to stamp {WAFFLE_KEYWORD} on {path}.",
                extra={"step": "cube_header_update", "status": "failed"},
            )
    return stamped_all


@optional_logger
def run_cube_header_update(
    frame_types_to_extract,
    converted_dir,
    override_mode_file="update",
    override_mode_header="update",
    continuous_satellite_spots=None,
    logger=None,
):
    """Update FITS headers of reduced data cubes with pipeline metadata.

    This is the fifth step in the SPHERE/IFS data reduction pipeline. It updates
    the FITS headers of bundled data cubes with comprehensive metadata about the
    reduction pipeline, including software versions, processing parameters, and
    provenance information.

    Required Input Files
    -------------------
    From previous step (bundle_output):
    - converted_dir/coro_cube.fits (if CORO in frame_types_to_extract)
    - converted_dir/center_cube.fits (if CENTER in frame_types_to_extract)  
    - converted_dir/flux_cube.fits (if FLUX in frame_types_to_extract)
    - converted_dir/frames_info_coro.csv (if CORO in frame_types_to_extract)
    - converted_dir/frames_info_center.csv (if CENTER in frame_types_to_extract)
    - converted_dir/frames_info_flux.csv (if FLUX in frame_types_to_extract)

    Modified Output Files
    --------------------
    In converted_dir:
    - Updated FITS headers in all *_cube.fits files with metadata including:
        - SPHERICAL pipeline version and git information
        - Processing timestamp and hostname
        - Constant values from frames_info CSV files
        - Pipeline step provenance information

    Parameters
    ----------
    frame_types_to_extract : list of str
        Frame types to update headers for (e.g., ["FLUX", "CENTER", "CORO"]).
    converted_dir : str
        Directory containing bundled cube files from previous step.
    override_mode_file : {"copy", "update"}, default "update"
        How to handle file overrides. Currently only "update" is supported.
    override_mode_header : {"keep", "update"}, default "update"
        How to handle header overrides. "update" overwrites existing keys,
        "keep" preserves existing metadata.
    continuous_satellite_spots : bool or None, default None
        The observation's WAFFLE_MODE flag, recorded on every cube as
        ``HIERARCH SPHERICAL WAFFLE MODE``. It is the only on-disk record of
        which frame type carries the science, and the standalone frame-alignment
        re-run reads it. None writes no keyword.
    logger : logging.Logger
        Logger instance to use for logging messages.

    Returns
    -------
    None
        This function modifies FITS files in place and does not return a value.

    Notes
    -----
    - Updates headers in place for all frame types that were processed
    - Adds comprehensive provenance metadata to support reproducibility
    - Includes constant values from frames_info CSV files as FITS keywords
    - Error handling ensures pipeline continues even if header updates fail
    - Uses HIERARCH keywords for compatibility with long keyword names

    Examples
    --------
    >>> run_cube_header_update(
    ...     frame_types_to_extract=["CORO", "CENTER", "FLUX"],
    ...     converted_dir="/path/to/converted",
    ...     override_mode_header="update"
    ... )
    """

    logger.info("Starting cube header update step.", extra={"step": "cube_header_update", "status": "started"})
    logger.debug(f"Parameters: frame_types={frame_types_to_extract}, "
                 f"converted_dir={converted_dir}, "
                 f"override_mode_file={override_mode_file}, "
                 f"override_mode_header={override_mode_header}")

    logger.info("Updating FITS headers with pipeline metadata...")
    for frame_type in frame_types_to_extract:
        try:
            update_cube_fits_header_after_reduction(
                path=converted_dir,
                target=frame_type.lower(),
                override_mode_file=override_mode_file,
                override_mode_header=override_mode_header,
                logger=logger
            )
            logger.debug(f"FITS header updated successfully for {frame_type}.")
        except Exception:
            logger.exception(f"Failed to update FITS header for {frame_type}.", extra={"step": f"cube_header_update_{frame_type.lower()}", "status": "failed"})
    
    stamped_all = True
    if continuous_satellite_spots is not None:
        stamped_all = _stamp_waffle_mode(
            converted_dir, frame_types_to_extract, continuous_satellite_spots, logger
        )

    logger.info("FITS headers updated for all processed frame types.")
    if not stamped_all:
        # Reporting success here would send a later standalone frame-alignment
        # re-run to the "re-run cube_header_update" message, naming the step
        # that had just claimed to succeed.
        logger.error(
            f"Finished cube header update step, but {WAFFLE_KEYWORD} is missing from at "
            "least one cube. A standalone frame-alignment re-run cannot resolve the "
            "science frame type from these cubes.",
            extra={"step": "cube_header_update", "status": "failed"},
        )
        return
    logger.info("Finished cube header update step.", extra={"step": "cube_header_update", "status": "success"})
