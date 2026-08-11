# Step: Frame Info Computation
# This module will contain the frame info computation logic refactored from ifs_reduction.py.

from spherical.pipeline.logging_utils import optional_logger


@optional_logger
def run_frame_info_computation(observation, converted_dir, logger):
    """Compute and save frame information tables for SPHERE/IFS data.

    This is the fifth step in the SPHERE/IFS data reduction pipeline. It processes
    observation metadata to create comprehensive information tables for each frame
    type, including computed times and angles. These tables are essential for
    subsequent analysis and visualization steps.

    Required Input Files
    -------------------
    From previous step (bundle_output):
    - converted_dir/coro_cube.fits
    - converted_dir/center_cube.fits
    - converted_dir/flux_cube.fits
        Master cubes containing the data for each frame type

    Generated Output Files
    ---------------------
    In converted_dir:
    - frames_info_coro.csv
        Table containing metadata for coronagraphic frames:
        - Observation times
        - Parallactic angles
        - Other frame-specific parameters
    - frames_info_center.csv
        Table containing metadata for center frames
    - frames_info_flux.csv
        Table containing metadata for flux frames

    Parameters
    ----------
    observation : Observation
        Observation object containing frame data. Must have:
        - frames[key]: DataFrame
            Frame data for each type ('FLUX', 'CORO', 'CENTER')
            with required metadata columns
    converted_dir : str
        Directory where the output CSV files will be saved.
        Must be the same directory containing the bundled cubes.

    Returns
    -------
    dict
        Dictionary containing DataFrames for each frame type:
        - 'FLUX': DataFrame
            Processed metadata for flux frames
        - 'CORO': DataFrame
            Processed metadata for coronagraphic frames
        - 'CENTER': DataFrame
            Processed metadata for center frames

    Notes
    -----
    - Skips frame types that have no data in observation.frames
    - Computes additional time and angle information from raw metadata
    - Creates CSV files in the same directory as the bundled cubes
    - Uses spherical.database.metadata for data processing
    - Preserves original observation data by working on copies

    Examples
    --------
    >>> frames_info = run_frame_info_computation(
    ...     observation=obs,
    ...     converted_dir="/path/to/converted"
    ... )
    >>> print(frames_info['CORO'].columns)
    ['MJD_OBS', 'PARANG', 'EXPTIME', ...]
    """
    import copy
    import os

    from spherical.database import metadata

    logger.info("Starting frame info computation step.", extra={"step": "frame_info_computation", "status": "started"})
    logger.debug(f"Parameters: converted_dir={converted_dir}, frame_types={list(observation.frames.keys())}")

    frames_info = {}
    for key in ['FLUX', 'CORO', 'CENTER']:
        if len(observation.frames[key]) == 0:
            logger.warning(f"No data for frame type: {key}", extra={"step": "frame_info_computation", "status": "failed"})
            continue
        try:
            frames_table = copy.copy(observation.frames[key])
            frames_info[key] = metadata.prepare_dataframe(frames_table)
            metadata.compute_times(frames_info[key])
            metadata.compute_angles(frames_info[key])
            output_path = os.path.join(converted_dir, f'frames_info_{key.lower()}.csv')
            frames_info[key].to_csv(output_path)
            logger.info(f"Frame info for {key} written to: {output_path}")
        except Exception:
            logger.exception(f"Failed to process frame type: {key}", extra={"step": "frame_info_computation", "status": "failed"})
    logger.info("Finished frame info computation step.", extra={"step": "frame_info_computation", "status": "success"})
    return frames_info
