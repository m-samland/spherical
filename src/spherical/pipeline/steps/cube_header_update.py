"""
FITS header update step for SPHERE IFS data reduction pipeline.

This module updates the FITS headers of reduced data cubes with pipeline metadata
including software versions, processing parameters, and provenance information.
This step runs after bundle_output to add comprehensive metadata to the final
data products.
"""

from spherical.pipeline.fits.headers import update_cube_fits_header_after_reduction
from spherical.pipeline.logging_utils import optional_logger


@optional_logger
def run_cube_header_update(
    frame_types_to_extract,
    converted_dir,
    override_mode_file="update",
    override_mode_header="update",
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
    
    logger.info("FITS headers updated for all processed frame types.")
    logger.info("Finished cube header update step.", extra={"step": "cube_header_update", "status": "success"})
