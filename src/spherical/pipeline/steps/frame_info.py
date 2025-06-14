# Step: Frame Info Computation
# This module will contain the frame info computation logic refactored from ifs_reduction.py.

def run_frame_info_computation(observation, converted_dir):
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

    frames_info = {}
    for key in ['FLUX', 'CORO', 'CENTER']:
        if len(observation.frames[key]) == 0:
            continue
        frames_table = copy.copy(observation.frames[key])
        frames_info[key] = metadata.prepare_dataframe(frames_table)
        metadata.compute_times(frames_info[key])
        metadata.compute_angles(frames_info[key])
        frames_info[key].to_csv(
            os.path.join(converted_dir, f'frames_info_{key.lower()}.csv'))
    return frames_info
