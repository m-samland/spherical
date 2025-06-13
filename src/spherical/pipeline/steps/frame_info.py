# Step: Frame Info Computation
# This module will contain the frame info computation logic refactored from ifs_reduction.py.

def run_frame_info_computation(observation, converted_dir):
    """
    Compute and save frame information tables for each frame type.

    Parameters
    ----------
    observation : Observation
        The observation object containing frame data.
    converted_dir : str
        Directory where the output CSV files will be saved.

    Returns
    -------
    dict
        Dictionary of DataFrames for each frame type ('FLUX', 'CORO', 'CENTER').
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
