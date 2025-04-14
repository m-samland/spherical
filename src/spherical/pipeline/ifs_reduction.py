import copy
import logging
import os
import shutil

# import warnings
import time
from glob import glob
from multiprocessing import Pool, cpu_count
from os import path

import charis
import dill as pickle
import matplotlib

matplotlib.use(backend='Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip
from astropy.table import Table, setdiff, vstack
from astroquery.eso import Eso
from natsort import natsorted
from photutils.aperture import CircularAnnulus, CircularAperture
from tqdm import tqdm

from spherical.pipeline import flux_calibration, toolbox, transmission
from spherical.pipeline.find_star import process_center_frames_in_parallel
from spherical.pipeline.toolbox import make_target_folder_string
from spherical.sphere_database.database_utils import find_nearest

# Create a module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_paths_to_filenames(full_paths):
    filenames = []
    for full_path in full_paths:
        extension = os.path.splitext(os.path.split(full_path)[-1])[1]
        if extension == '.Z':
            file_id = os.path.splitext(os.path.split(full_path)[-1])[0][:-5]
        else:
            file_id = os.path.splitext(os.path.split(full_path)[-1])[0]
        filenames.append(file_id)

    return filenames


def download_data_for_observation(raw_directory, observation, eso_username=None):
    if not os.path.exists(raw_directory):
        os.makedirs(raw_directory)

    if observation.filter == 'OBS_H' or observation.filter == 'OBS_YJ':
        raw_directory = os.path.join(raw_directory, 'IFS')
        print('Download attempted for IFS')
        science_keys = ['CORO', 'CENTER', 'FLUX']
        calibration_keys = [
            'WAVECAL']#, 'SPECPOS', 'BG_WAVECAL', 'BG_SPECPOS', 'FLAT']
    else:
        print('Download attempted for IRDIS')
        raw_directory = os.path.join(raw_directory, 'IRDIS')
        science_keys = ['CORO', 'CENTER', 'FLUX', 'BG_SCIENCE', 'BG_FLUX']
        calibration_keys = ['FLAT']  # 'DISTORTION'

    science_directory = os.path.join(raw_directory, 'science')
    if not os.path.exists(science_directory):
        os.makedirs(science_directory)

    target_name = observation.observation['MAIN_ID'][0]
    target_name = " ".join(target_name.split())
    target_name = target_name.replace(" ", "_")

    obs_band = observation.filter
    date = observation.observation['NIGHT_START'][0]

    target_directory = path.join(
        science_directory, target_name + '/' + obs_band + '/' + date)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    calibration_directory = os.path.join(raw_directory, 'calibration', obs_band)
    if not os.path.exists(calibration_directory):
        os.makedirs(calibration_directory)

    filenames = vstack(list(observation.frames.values()))['DP.ID']

    # Check for existing files
    existing_files = glob(os.path.join(raw_directory, '**', 'SPHER.*'), recursive=True)
    existing_files = convert_paths_to_filenames(existing_files)

    filename_table = Table({'DP.ID': filenames})
    existing_file_table = Table({'DP.ID': existing_files})
    if len(existing_file_table) > 0:
        download_list = setdiff(filename_table, existing_file_table, keys=['DP.ID'])
    else:
        download_list = filename_table

    if len(download_list) > 0:
        eso = Eso()
        if eso_username is not None:
            eso.login(username=eso_username)
        
        _ = eso.retrieve_data(
            datasets=list(download_list['DP.ID'].data),
            destination=raw_directory,
            with_calib=None,
            unzip=True)
    else:
        "No files to download."

    time.sleep(3)
    # Move downloaded files to proper directory
    for science_key in science_keys:
        files = list(observation.frames[science_key]['DP.ID']) #convert_paths_to_filenames(observation.frames[science_key]['FILE'])
        if len(files) > 0:
            destination_path = os.path.join(target_directory, science_key)
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            for file in files:
                origin_path = os.path.join(raw_directory, file + '.fits')
                shutil.move(origin_path, destination_path)

    for calibration_key in calibration_keys:
        files = list(observation.frames[calibration_key]['DP.ID']) #convert_paths_to_filenames(observation.frames[calibration_key]['FILE'])
        if len(files) > 0:
            destination_path = os.path.join(calibration_directory, calibration_key)
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            for file in files:
                origin_path = os.path.join(raw_directory, file + '.fits')
                shutil.move(origin_path, destination_path)
    return None


def bundle_output_into_cubes(key, cube_outputdir, output_type='resampled', overwrite=True):

    extracted_dir = path.join(cube_outputdir, key) + '/'
    converted_dir = path.join(cube_outputdir, 'converted') + '/'
    if not path.exists(converted_dir):
        os.makedirs(converted_dir)

    if output_type == 'resampled':
        glob_pattern = 'SPHER.*cube_resampled_DIT*.fits'
        name_suffix = ''
    elif output_type == 'hexagons':
        glob_pattern = 'SPHER.*cube_DIT*.fits'
        name_suffix = 'hexagons_'
    elif output_type == 'residuals':
        glob_pattern = 'SPHER.*cube_residuals_DIT*.fits'
        name_suffix = 'residuals_'
    else:
        raise ValueError('Invalid output_type selected.')

    science_files = natsorted(
        glob(os.path.join(extracted_dir, glob_pattern), recursive=False))

    if len(science_files) == 0:
        print('No output found in: {}'.format(extracted_dir))
        return None

    data_cube = []
    inverse_variance_cube = []
    parallactic_angles = []

    for file in science_files:
        hdus = fits.open(file)
        data_cube.append(hdus[1].data.astype('float32'))
        inverse_variance_cube.append(hdus[2].data.astype('float32'))
        try:
            parallactic_angles.append(
                hdus[0].header['HIERARCH DEROT ANGLE'])
        except Exception:
            print("Error retrieving parallactic angles for {}".format("CORO"))
        hdus.close()

    data_cube = np.array(data_cube, dtype='float32')
    inverse_variance_cube = np.array(inverse_variance_cube, dtype='float32')
    parallactic_angles = np.array(parallactic_angles)

    data_cube = np.swapaxes(data_cube, 0, 1)
    inverse_variance_cube = np.swapaxes(inverse_variance_cube, 0, 1)

    fits.writeto(os.path.join(converted_dir, '{}_{}cube.fits'.format(
        key, name_suffix).lower()), data_cube.astype('float32'), overwrite=overwrite)
    fits.writeto(os.path.join(converted_dir, '{}_{}ivar_cube.fits'.format(
        key, name_suffix).lower()), inverse_variance_cube.astype('float32'), overwrite=overwrite)
    if len(parallactic_angles) > 0:
        fits.writeto(os.path.join(converted_dir, '{}_parallactic_angles.fits'.format(
            key).lower()), parallactic_angles, overwrite=overwrite)


def update_observation_file_paths(existing_file_paths, observation, used_keys):
    """Update observation frames with actual file paths based on filenames."""
    
    # Convert file paths to filenames
    existing_file_names = convert_paths_to_filenames(existing_file_paths)
    existing_files = pd.DataFrame({'name': existing_file_names})
    
    for key in used_keys:
        try:
            # Ensure frame is a DataFrame and decode FILE column if needed
            observation.frames[key] = observation.frames[key].to_pandas()
            observation.frames[key]['FILE'] = observation.frames[key]['FILE'].str.decode('UTF-8')
        except Exception:
            pass

        # Match filenames to actual file paths
        filepaths = []
        names_of_header_files = list(observation.frames[key]['DP.ID'])
        for name in names_of_header_files:
            index_of_file = existing_files[existing_files['name'] == name].index.values[0]
            real_path_of_file = existing_file_paths[index_of_file]
            filepaths.append(real_path_of_file)

        # Update FILE column with actual paths
        observation.frames[key]['FILE'] = filepaths


def extract_cubes_with_multiprocessing(
    observation,
    frame_types_to_extract,
    extraction_parameters,
    reduction_parameters,
    instrument,
    wavecal_outputdir,
    cube_outputdir,
    non_least_square_methods=['optext', 'apphot3', 'apphot5'],
    extract_cubes=True
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
    instrument : object
        Instrument descriptor, which can include wavelength range, midpoint
        arrays, or other instrument-specific configurations.
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
        return  # If extraction is disabled, return early.

    # Check for conflicts in background subtraction parameters.
    if extraction_parameters['bgsub'] and extraction_parameters['fitbkgnd']:
        raise ValueError('Background subtraction and fitting should not be used together.')

    # Unpack parameters we might modify below.
    bgsub = extraction_parameters['bgsub']
    fitbkgnd = extraction_parameters['fitbkgnd']
    extraction_parameters['bgsub'] = bgsub
    extraction_parameters['fitbkgnd'] = fitbkgnd

    # If a user requests PCA for background, don't use a direct file path.
    if reduction_parameters['bg_pca']:
        bgpath = None
    else:
        # Attempt to find a "BG_SCIENCE" frame; if none found, fall back on PCA.
        try:
            bgpath_candidate = observation.frames['BG_SCIENCE']['FILE'].iloc[-1]
            bgpath = bgpath_candidate
        except (KeyError, IndexError):
            print("No BG frame found to subtract. Falling back on PCA fit.")
            bgpath = None

    fitshift_original = extraction_parameters['fitshift']

    # Main loop over each requested frame type.
    for key in frame_types_to_extract:
        if len(observation.frames[key]) == 0:
            logger.info(f"No files to reduce for key: {key}")
            continue

        cube_type_outputdir = os.path.join(cube_outputdir, key)
        os.makedirs(cube_type_outputdir, exist_ok=True)

        tasks = []

        for idx, file in tqdm(enumerate(observation.frames[key]['FILE']),
                              desc=f"Collect tasks for {key}", unit="file"):
            hdr = fits.getheader(file)
            ndit = hdr['HIERARCH ESO DET NDIT']

            # Determine the background frame logic for each file.
            if key == 'CENTER':
                if (len(observation.frames.get('CORO', [])) > 0
                        and reduction_parameters.get('subtract_coro_from_center', False)):
                    extraction_parameters['bg_scaling_without_mask'] = True
                    logger.info(f'CENTER file uses CORO background (file index {idx}).')
                    idx_nearest = find_nearest(
                        observation.frames['CORO']['MJD_OBS'].values,
                        observation.frames['CENTER'].iloc[idx]['MJD_OBS']
                    )
                    bg_frame = observation.frames['CORO']['FILE'].iloc[idx_nearest]
                else:
                    extraction_parameters['bg_scaling_without_mask'] = False
                    bg_frame = None
                    logger.info(f'CENTER file uses PCA background (file index {idx}).')
            else:
                extraction_parameters['bg_scaling_without_mask'] = False
                bg_frame = bgpath

            # Decide on fitshift usage.
            if key == 'FLUX':
                extraction_parameters['fitshift'] = False
            else:
                extraction_parameters['fitshift'] = fitshift_original

            # If single-threaded, process each DIT directly.
            if reduction_parameters['ncpu_cubebuilding'] == 1:
                for dit_index in tqdm(range(ndit), desc=f"Extract {key} (1 CPU)", unit="DIT"):
                    charis.extractcube.getcube(
                        filename=file,
                        dit=dit_index,
                        bgpath=bg_frame,
                        calibdir=wavecal_outputdir,
                        outdir=cube_type_outputdir,
                        **extraction_parameters
                    )
            else:
                # Collect tasks for parallel processing.
                for dit_index in range(ndit):
                    tasks.append(
                        (
                            file,
                            dit_index,
                            bg_frame,
                            wavecal_outputdir,
                            cube_type_outputdir,
                            extraction_parameters.copy()
                        )
                    )

            # Restore state modified for this file.
            extraction_parameters['bg_scaling_without_mask'] = False
            extraction_parameters['fitshift'] = fitshift_original

        # Parallel map if we have tasks and multiple CPUs.
        if tasks and reduction_parameters['ncpu_cubebuilding'] != 1:
            ncpus = min(reduction_parameters['ncpu_cubebuilding'], cpu_count())
            logger.info(f"Starting parallel extraction for {len(tasks)} tasks using up to {ncpus} processes.")

            try:
                with Pool(processes=ncpus) as pool:
                    results = pool.map(_parallel_extraction_worker, tasks)
                # Count and log failures due to ValueError
                num_failed = sum(not r for r in results)
                if num_failed > 0:
                    logger.warning(f"{num_failed} extraction task(s) failed due to ValueError and were skipped.")
            except Exception as e:
                logger.exception(f"Unexpected exception during parallel extraction for frame type '{key}': {e}")
                raise  # Stop pipeline on any other error


def _parallel_extraction_worker(task):
    r"""
    Worker function to extract a single DIT from a CHARIS file.

    Parameters
    ----------
    task : tuple
        (filename, dit_index, bg_frame, wavecal_outputdir, outputdir, extraction_params)

    Returns
    -------
    bool
        True if extraction succeeded, False if it failed due to ValueError.

    Raises
    ------
    Any exception other than ValueError will propagate and stop the pipeline.
    """
    (filename,
     dit_index,
     bg_frame,
     wavecal_outputdir,
     outputdir,
     extraction_params) = task

    try:
        charis.extractcube.getcube(
            filename=filename,
            dit=dit_index,
            bgpath=bg_frame,
            calibdir=wavecal_outputdir,
            outdir=outputdir,
            **extraction_params
        )
        return True  # Successful extraction
    except ValueError as err:
        logger.warning(f"[Worker] ValueError in {filename} (DIT={dit_index}): {err}")
        return False  # Expected recoverable failure


def execute_IFS_target(
        observation,
        calibration_parameters,
        extraction_parameters,
        reduction_parameters,
        reduction_directory,
        raw_directory=None,
        download_data=True,
        reduce_calibration=True,
        extract_cubes=False,
        frame_types_to_extract=['FLUX', 'CENTER', 'CORO'],
        bundle_output=True,
        bundle_hexagons=True,
        bundle_residuals=True,
        compute_frames_info=False,
        find_centers=False,
        plot_image_center_evolution=False,
        process_extracted_centers=False,
        calibrate_spot_photometry=False,
        calibrate_flux_psf=False,
        spot_to_flux=True,
        eso_username=None,
        overwrite_calibration=False,
        overwrite_bundle=False,
        overwrite_preprocessing=False,
        save_plots=True,
        verbose=True):

    start = time.time()

    observation = copy.copy(observation)


    # name_mode_date = make_target_folder_string(observation)
 
    target_name = observation.observation['MAIN_ID'][0]
    target_name = " ".join(target_name.split())
    target_name = target_name.replace(" ", "_")
    obs_band = observation.observation['IFS_MODE'][0]
    date = observation.observation['NIGHT_START'][0]
    name_mode_date = target_name + '/' + obs_band + '/' + date
        
    outputdir = path.join(
        reduction_directory, 'IFS/observation', name_mode_date)
    cube_outputdir = path.join(outputdir, '{}'.format(
        extraction_parameters['method']))
    converted_dir = path.join(cube_outputdir, 'converted') + '/'

    if verbose:
        print(f"Start reduction of: {name_mode_date}")

    if obs_band == 'OBS_YJ':
        instrument = charis.instruments.SPHERE('YJ')
        extraction_parameters['R'] = 55
    elif obs_band == 'OBS_H':
        instrument = charis.instruments.SPHERE('YH')
        extraction_parameters['R'] = 35

    # log_dir = outputdir + '/logs'
    # log_file = os.path.join(log_dir, "reduction.log")

    # # Make sure directory exists
    # os.makedirs(log_dir, exist_ok=True)

    # # Set up logger
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)

    # # Create file handler
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.INFO)

    # # Create formatter and attach it
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)

    # # Avoid duplicate handlers
    # if not logger.hasHandlers():
    #     logger.addHandler(file_handler)

    if download_data:
        try:
            _ = download_data_for_observation(raw_directory=raw_directory, observation=observation, eso_username=eso_username)
        except:
            print('Download failed for observation: {}'.format(observation.observation['MAIN_ID'][0]))

    existing_file_paths = glob(os.path.join(raw_directory, '**', 'SPHER.*.fits'), recursive=True)
    used_keys = ['CORO', 'CENTER', 'FLUX']
    if reduce_calibration:
        used_keys += ['WAVECAL']

    non_least_square_methods = ['optext', 'apphot3', 'apphot5']

    if reduce_calibration or extract_cubes or bundle_output:
        update_observation_file_paths(existing_file_paths, observation, used_keys)

    calibration_time_name = observation.frames['WAVECAL']['DP.ID'][0][6:]
    wavecal_outputdir = os.path.join(reduction_directory, 'IFS/calibration', obs_band, calibration_time_name)#, date)

    if reduce_calibration:
        if not path.exists(outputdir):
            os.makedirs(outputdir)

        # -------------------------------------------------------------------#
        # --------------------Wavelength calibration-------------------------#
        # -------------------------------------------------------------------#


        # wavecal_outputdir = path.join(calibration_directory, 'calibration') + '/'
        if not path.exists(wavecal_outputdir):
            os.makedirs(wavecal_outputdir)

        files_in_calibration_folder = glob(os.path.join(wavecal_outputdir, '*key*.fits')) # TODO: This should check if oversanpling is used or not, different filename
        if len(files_in_calibration_folder) == 0 or overwrite_calibration:
            # check if reduction exists
            calibration_wavelength = instrument.calibration_wavelength
            wavecal_file = observation.frames['WAVECAL']['FILE'][0]
            inImage, hdr = charis.buildcalibrations.read_in_file(
                wavecal_file, instrument, calibration_wavelength,
                ncpus=calibration_parameters['ncpus'])
            # outdir=os.path.join(outputdir, 'calibration/')
            charis.buildcalibrations.buildcalibrations(
                inImage=inImage, instrument=instrument,
                inLam=calibration_wavelength.value,
                outdir=wavecal_outputdir,
                header=hdr,
                **calibration_parameters)

    # -------------------------------------------------------------------#
    # --------------------Cube extraction--------------------------------#
    # -------------------------------------------------------------------#
    if extract_cubes:
        # if not path.exists(cube_outputdir):
        #     os.makedirs(cube_outputdir)

        # if extract_cubes:
        #     if (extraction_parameters['method'] in non_least_square_methods) \
        #             and extraction_parameters['linear_wavelength']:
        #         wavelengths = np.linspace(
        #             instrument.wavelength_range[0].value,
        #             instrument.wavelength_range[1].value,
        #             39)
        #     else:
        #         wavelengths = instrument.lam_midpts

        #     if extraction_parameters['bgsub'] and extraction_parameters['fitbkgnd']:
        #         raise ValueError('Background subtraction and fitting should not be used together.')

        #     # if observation.observation['WAFFLE_MODE'][0] == 'True':
        #     bgsub = extraction_parameters['bgsub']
        #     fitbkgnd = extraction_parameters['fitbkgnd']

        #     extraction_parameters['bgsub'] = bgsub
        #     extraction_parameters['fitbkgnd'] = fitbkgnd

        #     frame_type_to_dark_mapping = {
        #         'FLUX': 'SCIENCE',  # Don't use flux exposure time for background, because lower SNR for background compared to science frames
        #         'CORO': 'SCIENCE',
        #         'CENTER': 'SCIENCE',
        #         'WAVECAL': 'WAVECAL',
        #         'SPECPOS': 'SPECPOS'
        #     }

        #     if reduction_parameters['bg_pca']:
        #         bgpath = None
        #     else:
        #         if len(observation.background[frame_type_to_dark_mapping[key]]['SKY']['FILE']) > 0:
        #             bgpath = observation.frames['BG_SCIENCE']['FILE'].iloc[-1]
        #         else:
        #             print("No BG frame found to subtract. Falling back on PCA fit.")
        #             bgpath = None

        #     fitshift = extraction_parameters['fitshift']
        #     for key in frame_types_to_extract:
        #         if len(observation.frames[key]) == 0:
        #             print("No files to reduce for key: {}".format(key))
        #             continue
        #         cube_type_outputdir = path.join(cube_outputdir, key) + '/'
        #         if not path.exists(cube_type_outputdir):
        #             os.makedirs(cube_type_outputdir)

        #         for idx, file in tqdm(enumerate(observation.frames[key]['FILE'])):
        #             hdr = fits.getheader(file)
        #             ndit = hdr['HIERARCH ESO DET NDIT']

        #             if key == 'CENTER':
        #                 if len(observation.frames['CORO']) > 0 and reduction_parameters['subtract_coro_from_center']:
        #                     extraction_parameters['bg_scaling_without_mask'] = True
        #                     print('center file!')
        #                     idx_nearest = find_nearest(
        #                         observation.frames['CORO'].iloc[:]['MJD_OBS'].values,
        #                         observation.frames['CENTER'].iloc[idx]['MJD_OBS'])
        #                     bg_frame = observation.frames['CORO']['FILE'].iloc[idx_nearest]
        #                 else:
        #                     extraction_parameters['bg_scaling_without_mask'] = False
        #                     bg_frame = None
        #                     print("PCA subtracting center frame BG.")
        #             else:
        #                 extraction_parameters['bg_scaling_without_mask'] = False
        #                 bg_frame = bgpath

        #             if key == 'FLUX':
        #                 # fitshift operates on spectra in the whole FoV
        #                 extraction_parameters['fitshift'] = False
        #             else:
        #                 extraction_parameters['fitshift'] = fitshift

        #             # if len(reduced_files) != ndit:
        #             if reduction_parameters['dit_cpus_max'] == 1:
        #                 for dit in tqdm(range(ndit)):
        #                     charis.extractcube.getcube(
        #                         filename=file,
        #                         dit=dit,
        #                         bgpath=bg_frame,
        #                         calibdir=wavecal_outputdir,
        #                         outdir=cube_type_outputdir,
        #                         **extraction_parameters
        #                     )
        #             else:
        #                 if ndit <= reduction_parameters['dit_cpus_max']:
        #                     ncpus = ndit
        #                 else:
        #                     ncpus = reduction_parameters['dit_cpus_max']

        #                 multiprocess_charis_ifs = partial(
        #                     charis.extractcube.getcube,
        #                     filename=file,
        #                     bgpath=bg_frame,
        #                     calibdir=wavecal_outputdir,
        #                     outdir=cube_type_outputdir,
        #                     **extraction_parameters)
        #                 # indices = range(ndit)           
        #                 # with Pool(processes=ncpus) as pool:  
        #                 #     for _ in tqdm(pool.imap(func=multiprocess_charis_ifs, iterable=indices), total=len(indices)):
        #                 #         pass
        #                 # Create a pool of workers
        #                 max_retries = 3
        #                 for i in range(max_retries):
        #                     try:
        #                         with Pool(processes=min(ncpus, cpu_count())) as pool:  
        #                             pool.map(func=multiprocess_charis_ifs, iterable=range(ndit))
        #                         break  # If the map call succeeds, break the loop
        #                     except BrokenPipeError:
        #                         print("A child process terminated abruptly, causing a BrokenPipeError.")
        #                         if i < max_retries - 1:  # No need to sleep on the last iteration
        #                             sleep(10)  # Wait for 10 seconds before retrying
        #                     except Exception as e:
        #                         print(f"An unexpected error occurred: {e}")
        #                         raise  # If an unexpected error occurs, break the loop

        #             extraction_parameters['bg_scaling_without_mask'] = False

        extract_cubes_with_multiprocessing(
            observation=observation,
            frame_types_to_extract=['CORO', 'CENTER', 'FLUX'],
            extraction_parameters=extraction_parameters,
            reduction_parameters=reduction_parameters,
            instrument=instrument,
            wavecal_outputdir=wavecal_outputdir,
            cube_outputdir=cube_outputdir,
            non_least_square_methods=non_least_square_methods,
            extract_cubes=extract_cubes)


    if bundle_output:
        for key in frame_types_to_extract:
            if bundle_hexagons:
                bundle_output_into_cubes(
                    key, cube_outputdir, output_type='hexagons', overwrite=overwrite_bundle)
            if bundle_residuals:
                bundle_output_into_cubes(
                    key, cube_outputdir, output_type='residuals', overwrite=overwrite_bundle)
            bundle_output_into_cubes(
                key, cube_outputdir, output_type='resampled', overwrite=overwrite_bundle)

        # If we are using certain non-least-square methods, and linear_wavelength is set,
        # create a special wavelength array; otherwise use lam_midpts from instrument.
        if (extraction_parameters['method'] in non_least_square_methods) \
                and extraction_parameters['linear_wavelength']:
            wavelengths = np.linspace(
                instrument.wavelength_range[0].value,
                instrument.wavelength_range[1].value,
                39
            )
        else:
            wavelengths = instrument.lam_midpts

        fits.writeto(os.path.join(converted_dir, 'wavelengths.fits'),
                     wavelengths, overwrite=overwrite_bundle)

    """ FRAME INFO COMPUTATION """
    if compute_frames_info:
        frames_info = {}
        for key in ['FLUX', 'CORO', 'CENTER']:
            if len(observation.frames[key]) == 0:
                continue
            frames_table = copy.copy(observation.frames[key])
            frames_info[key] = toolbox.prepare_dataframe(frames_table)
            toolbox.compute_times(frames_info[key])
            toolbox.compute_angles(frames_info[key])
            frames_info[key].to_csv(
                os.path.join(converted_dir, 'frames_info_{}.csv'.format(key.lower())))
            # except:
            #     print("Failed to compute angles for key: {}".format(key))

    """ DETERMINE CENTERS """
    if find_centers:
        process_center_frames_in_parallel(
            converted_dir=converted_dir,
            observation=observation,
            overwrite=overwrite_preprocessing,
            ncpu=reduction_parameters['ncpu_find_center'])

    if process_extracted_centers:
        plot_dir = os.path.join(converted_dir, 'center_plots/')
        if not path.exists(plot_dir):
            os.makedirs(plot_dir)
        spot_centers = fits.getdata(os.path.join(converted_dir, 'spot_centers.fits'))
        # spot_distances = fits.getdata(os.path.join(converted_dir, 'spot_distances.fits'))
        image_centers = fits.getdata(os.path.join(converted_dir, 'image_centers.fits'))
        # spot_amplitudes = fits.getdata(os.path.join(converted_dir, 'spot_fit_amplitudes.fits'))
        center_cube = fits.getdata(os.path.join(converted_dir, 'center_cube.fits'))
        wavelengths = fits.getdata(os.path.join(converted_dir, 'wavelengths.fits'))

        satellite_psf_stamps = toolbox.extract_satellite_spot_stamps(center_cube, spot_centers, stamp_size=57,
                                                                     shift_order=3, plot=False)
        master_satellite_psf_stamps = np.nanmean(np.nanmean(satellite_psf_stamps, axis=2), axis=1)
        fits.writeto(
            os.path.join(converted_dir, 'satellite_psf_stamps.fits'),
            satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)
        fits.writeto(
            os.path.join(converted_dir, 'master_satellite_psf_stamps.fits'),
            master_satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)

        # mean_spot_stamps = np.sum(satellite_psf_stamps, axis=2)
        # aperture_size = 3

        # stamp_size = [mean_spot_stamps.shape[-1], mean_spot_stamps.shape[-2]]
        # stamp_center = [mean_spot_stamps.shape[-1] // 2, mean_spot_stamps.shape[-2] // 2]
        # aperture = CircularAperture(stamp_center, aperture_size)
        #
        # psf_mask = aperture.to_mask(method='center')
        # # Make sure only pixels are used for which data exists
        # psf_mask = psf_mask.to_image(stamp_size) > 0
        # flux_sum_with_bg = np.sum(mean_spot_stamps[:, :, psf_mask], axis=2)
        #
        # bg_aperture = CircularAnnulus(stamp_center, r_in=aperture_size, r_out=aperture_size+2)
        # bg_mask = bg_aperture.to_mask(method='center')
        # bg_mask = bg_mask.to_image(stamp_size) > 0
        # area = np.pi*aperture_size**2
        # bg_flux = np.mean(mean_spot_stamps[:, :, bg_mask], axis=2) * area
        #
        # flux_sum_without_bg = flux_sum_with_bg - bg_flux

        if (extraction_parameters['method'] in non_least_square_methods) \
                and extraction_parameters['linear_wavelength'] is True:
            remove_indices = [0, 1, 21, 38]
        else:
            remove_indices = [0, 13, 19, 20]

        anomalous_centers_mask = np.zeros_like(wavelengths).astype('bool')
        anomalous_centers_mask[remove_indices] = True

        image_centers_fitted = np.zeros_like(image_centers)
        image_centers_fitted2 = np.zeros_like(image_centers)

        # wavelengths_clean = np.delete(wavelengths, remove_indices)
        # image_centers_clean = np.delete(image_centers, remove_indices, axis=0)

        coefficients_x_list = []
        coefficients_y_list = []

        for frame_idx in range(image_centers.shape[1]):
            good_wavelengths = ~anomalous_centers_mask & np.all(
                np.isfinite(image_centers[:, frame_idx]), axis=1)
            try:
                coefficients_x = np.polyfit(
                    wavelengths[good_wavelengths], image_centers[good_wavelengths, frame_idx, 0], deg=2)
                coefficients_y = np.polyfit(
                    wavelengths[good_wavelengths], image_centers[good_wavelengths, frame_idx, 1], deg=3)

                coefficients_x_list.append(coefficients_x)
                coefficients_y_list.append(coefficients_y)

                image_centers_fitted[:, frame_idx, 0] = np.poly1d(coefficients_x)(wavelengths)
                image_centers_fitted[:, frame_idx, 1] = np.poly1d(coefficients_y)(wavelengths)
            except:
                print("Failed first iteration polyfit for frame {}".format(frame_idx))
                image_centers_fitted[:, frame_idx, 0] = np.nan
                image_centers_fitted[:, frame_idx, 1] = np.nan

        for frame_idx in range(image_centers.shape[1]):
            if np.all(~np.isfinite(image_centers_fitted[:, frame_idx])):
                image_centers_fitted2[:, frame_idx, 0] = np.nan
                image_centers_fitted2[:, frame_idx, 1] = np.nan
            else:
                deviation = image_centers - image_centers_fitted
                filtered_data = sigma_clip(
                    deviation, axis=0, sigma=3, maxiters=None, cenfunc='median', stdfunc=mad_std, masked=True, copy=True)
                anomalous_centers_mask = np.any(filtered_data.mask, axis=2)
                anomalous_centers_mask[remove_indices] = True
                try:
                    coefficients_x = np.polyfit(
                        wavelengths[~anomalous_centers_mask[:, 1]],
                        image_centers[~anomalous_centers_mask[:, 1], frame_idx, 0], deg=2)
                    coefficients_y = np.polyfit(
                        wavelengths[~anomalous_centers_mask[:, 1]],
                        image_centers[~anomalous_centers_mask[:, 1], frame_idx, 1], deg=3)

                    coefficients_x_list.append(coefficients_x)
                    coefficients_y_list.append(coefficients_y)

                    image_centers_fitted2[:, frame_idx, 0] = np.poly1d(coefficients_x)(wavelengths)
                    image_centers_fitted2[:, frame_idx, 1] = np.poly1d(coefficients_y)(wavelengths)
                except:
                    print("Failed second iteration polyfit for frame {}".format(frame_idx))
                    image_centers_fitted2[:, frame_idx, 0] = np.nan
                    image_centers_fitted2[:, frame_idx, 1] = np.nan

        fits.writeto(os.path.join(converted_dir, 'image_centers_fitted.fits'),
                     image_centers_fitted, overwrite=overwrite_preprocessing)
        fits.writeto(os.path.join(converted_dir, 'image_centers_fitted_robust.fits'),
                     image_centers_fitted2, overwrite=overwrite_preprocessing)


    if plot_image_center_evolution:
        image_centers = fits.getdata(os.path.join(converted_dir, 'image_centers.fits'))
        image_centers_fitted = fits.getdata(os.path.join(converted_dir, 'image_centers_fitted.fits'))
        image_centers_fitted2 = fits.getdata(os.path.join(converted_dir, 'image_centers_fitted_robust.fits'))

        plot_dir = os.path.join(converted_dir, 'center_plots/')
        if not path.exists(plot_dir):
            os.makedirs(plot_dir)

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D
        # --- Step 1: Load time info ---
        frame_info_center = pd.read_csv(os.path.join(converted_dir, 'frames_info_center.csv'))
        time_strings = frame_info_center["TIME"]
        times = pd.to_datetime(time_strings)

        # --- Step 2: Compute elapsed minutes ---
        start_time = times.min()
        elapsed_minutes = (times - start_time).dt.total_seconds() / 60.0

        # --- Step 3: Normalize elapsed minutes for colormap ---
        norm = Normalize(vmin=elapsed_minutes.min(), vmax=elapsed_minutes.max())
        cmap = plt.cm.PiYG
        colors = cmap(norm(elapsed_minutes))

        # --- Step 4: Plot using elapsed-time-based colors ---
        plt.close()
        n_wavelengths = image_centers.shape[0]
        n_frames = image_centers.shape[1]
        sizes = np.linspace(20, 300, n_wavelengths)

        fig, ax = plt.subplots(figsize=(8, 6))

        for frame_idx in range(n_frames):
            color = colors[frame_idx]
            ax.scatter(image_centers_fitted[:, frame_idx, 0], image_centers_fitted[:, frame_idx, 1],
                    s=sizes, marker='o', color=color, alpha=0.6)
            ax.scatter(image_centers_fitted2[:, frame_idx, 0], image_centers_fitted2[:, frame_idx, 1],
                    s=sizes, marker='x', color=color, alpha=0.9)
            ax.scatter(image_centers[:, frame_idx, 0], image_centers[:, frame_idx, 1],
                    s=sizes, marker='+', color=color, alpha=0.6)

        # --- Step 5: Add legend ---
        # --- Updated: Move legend outside the plot ---
        # --- Updated: Move legend below the plot ---
        legend_elements = [
            Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=10, label='1st Fit (fitted)'),
            Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=10, label='2nd Fit (robust)'),
            Line2D([0], [0], marker='+', color='gray', linestyle='None', markersize=10, label='Original Data'),
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            title='Marker Meaning',
            frameon=False
        )

        # --- Step 6: Add colorbar ---
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # required for colorbar
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Elapsed Time (minutes)')

        # --- Final plot adjustments ---
        ax.set_xlabel('X Center Position')
        ax.set_ylabel('Y Center Position')
        ax.set_title('Center Position Evolution per Wavelength and Time')
        ax.set_aspect('equal')

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)  # Makes space for the legend

        plt.savefig(os.path.join(plot_dir, 'center_evolution_time_colorbar.pdf'), bbox_inches='tight')



    if calibrate_spot_photometry:
        satellite_psf_stamps = fits.getdata(os.path.join(
            converted_dir, 'satellite_psf_stamps.fits')).astype('float64')
        # flux_variance = fits.getdata(os.path.join(converted_dir, 'flux_cube.fits'), 1)

        stamp_size = [satellite_psf_stamps.shape[-1], satellite_psf_stamps.shape[-2]]
        stamp_center = [satellite_psf_stamps.shape[-1] // 2, satellite_psf_stamps.shape[-2] // 2]

        # BG measurement
        bg_aperture = CircularAnnulus(stamp_center, r_in=15, r_out=18)
        bg_mask = bg_aperture.to_mask(method='center')
        bg_mask = bg_mask.to_image(stamp_size) > 0

        mask = np.ones_like(satellite_psf_stamps)
        mask[:, :, :, bg_mask] = 0
        ma_spot_stamps = np.ma.array(
            data=satellite_psf_stamps.reshape(
                satellite_psf_stamps.shape[0], satellite_psf_stamps.shape[1], satellite_psf_stamps.shape[2], -1),
            mask=mask.reshape(
                satellite_psf_stamps.shape[0], satellite_psf_stamps.shape[1], satellite_psf_stamps.shape[2], -1))

        sigma_clipped_array = sigma_clip(
            ma_spot_stamps,  # satellite_psf_stamps[:, :, :, bg_mask],
            sigma=3, maxiters=5, cenfunc=np.nanmedian, stdfunc=np.nanstd,
            axis=3, masked=True, return_bounds=False)

        bg_counts = np.ma.median(sigma_clipped_array, axis=3).data
        bg_std = np.ma.std(sigma_clipped_array, axis=3).data

        # BG correction of stamps
        bg_corr_satellite_psf_stamps = satellite_psf_stamps - bg_counts[:, :, :, None, None]

        fits.writeto(
            os.path.join(converted_dir, 'satellite_psf_stamps_bg_corrected.fits'),
            bg_corr_satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)

        aperture = CircularAperture(stamp_center, 3)
        psf_mask = aperture.to_mask(method='center')
        # Make sure only pixels are used for which data exists
        psf_mask = psf_mask.to_image(stamp_size) > 0

        flux_sum = np.nansum(bg_corr_satellite_psf_stamps[:, :, :, psf_mask], axis=3)

        spot_snr = flux_sum / (bg_std * np.sum(psf_mask))

        master_satellite_psf_stamps_bg_corr = np.nanmean(
            np.nansum(bg_corr_satellite_psf_stamps, axis=2), axis=1)

        fits.writeto(
            os.path.join(converted_dir, 'spot_amplitudes.fits'),
            flux_sum, overwrite=overwrite_preprocessing)

        fits.writeto(
            os.path.join(converted_dir, 'spot_snr.fits'),
            spot_snr, overwrite=overwrite_preprocessing)

        fits.writeto(
            os.path.join(converted_dir, 'master_satellite_psf_stamps_bg_corr.fits'),
            master_satellite_psf_stamps_bg_corr.astype('float32'), overwrite=overwrite_preprocessing)

    """ FLUX FRAMES """
    if calibrate_flux_psf:
        wavelengths = fits.getdata(
            os.path.join(converted_dir, 'wavelengths.fits'))
        flux_cube = fits.getdata(os.path.join(converted_dir, 'flux_cube.fits')).astype('float64')
        # flux_variance = fits.getdata(os.path.join(converted_dir, 'flux_cube.fits'), 1)
        frames_info = {}
        frames_info['CENTER'] = pd.read_csv(os.path.join(converted_dir, 'frames_info_center.csv'))
        frames_info['FLUX'] = pd.read_csv(os.path.join(converted_dir, 'frames_info_flux.csv'))
        
        plot_dir = os.path.join(converted_dir, 'flux_plots/')
        if not path.exists(plot_dir):
            os.makedirs(plot_dir)

        flux_centers = []
        flux_amplitudes = []
        guess_center_yx = []
        wave_median_flux_image = np.nanmedian(flux_cube[1:-1], axis=0)
        median_flux_image = np.nanmedian(wave_median_flux_image, axis=0)

        guess_center_yx = np.unravel_index(
            np.nanargmax(median_flux_image), median_flux_image.shape)
        for frame_number in range(flux_cube.shape[1]):
            data = flux_cube[:, frame_number]
            flux_center, flux_amplitude = toolbox.star_centers_from_PSF_img_cube(
                cube=data,
                wave=wavelengths,
                pixel=7.46,
                guess_center_yx=guess_center_yx,  # [frame_number],
                fit_background=False,
                fit_symmetric_gaussian=True,
                mask_deviating=False,
                deviation_threshold=0.8,
                mask=None,
                save_path=None)
            flux_centers.append(flux_center)
            flux_amplitudes.append(flux_amplitude)

        flux_centers = np.expand_dims(
            np.swapaxes(np.array(flux_centers), 0, 1),
            axis=2)
        flux_amplitudes = np.swapaxes(np.array(flux_amplitudes), 0, 1)
        fits.writeto(
            os.path.join(converted_dir, 'flux_centers.fits'), flux_centers, overwrite=overwrite_preprocessing)
        fits.writeto(
            os.path.join(converted_dir, 'flux_gauss_amplitudes.fits'), flux_amplitudes, overwrite=overwrite_preprocessing)

        flux_stamps = toolbox.extract_satellite_spot_stamps(
            flux_cube, flux_centers, stamp_size=57,
            shift_order=3, plot=False)
        # flux_variance_stamps = toolbox.extract_satellite_spot_stamps(
        #     flux_cube, flux_centers, stamp_size=57,
        #     shift_order=3, plot=False)
        fits.writeto(os.path.join(converted_dir, 'flux_stamps_uncalibrated.fits'),
                     flux_stamps.astype('float32'), overwrite=overwrite_preprocessing)

        # Adjust for exposure time and ND filter, put all frames to 1 second exposure
        # wave, bandwidth = transmission.wavelength_bandwidth_filter(filter_comb)
        if len(frames_info['FLUX']['INS4 FILT2 NAME'].unique()) > 1:
            raise ValueError('Non-unique ND filters in sequence.')
        else:
            ND = frames_info['FLUX']['INS4 FILT2 NAME'].unique()[0]

        _, attenuation = transmission.transmission_nd(ND, wave=wavelengths)
        fits.writeto(os.path.join(converted_dir, 'nd_attenuation.fits'),
                     attenuation, overwrite=overwrite_preprocessing)
        dits_flux = np.array(frames_info['FLUX']['DET SEQ1 DIT'])
        dits_center = np.array(frames_info['CENTER']['DET SEQ1 DIT'])
        # dits_coro = np.array(frames_info['CORO']['DET SEQ1 DIT'])

        unique_dits_center, unique_dits_center_counts = np.unique(dits_center, return_counts=True)

        # Normalize coronagraphic sequence to DIT that is most common
        if len(unique_dits_center) == 1:
            dits_factor = unique_dits_center[0] / dits_flux
            most_common_dit_center = unique_dits_center[0]
        else:
            most_common_dit_center = unique_dits_center[np.argmax(unique_dits_center_counts)]
            dits_factor = most_common_dit_center / dits_flux

        dit_factor_center = most_common_dit_center / dits_center

        fits.writeto(os.path.join(converted_dir, 'center_frame_dit_adjustment_factors.fits'),
                     dit_factor_center, overwrite=overwrite_preprocessing)

        print("Attenuation: {}".format(attenuation))
        # if adjust_dit:
        flux_stamps_calibrated = flux_stamps * dits_factor[None, :, None, None]

        flux_stamps_calibrated = flux_stamps_calibrated / \
            attenuation[:, np.newaxis, np.newaxis, np.newaxis]
        fits.writeto(os.path.join(converted_dir, 'flux_stamps_dit_nd_calibrated.fits'),
                     flux_stamps_calibrated, overwrite=overwrite_preprocessing)

        # fwhm_angle = ((wavelengths * u.nm) / (7.99 * u.m)).to(
        #     u.mas, equivalencies=u.dimensionless_angles())
        # fwhm = fwhm_angle.to(u.pixel, u.pixel_scale(0.00746 * u.arcsec / u.pixel)).value
        # aperture_sizes = np.round(fwhm * 2.1)

        flux_photometry = flux_calibration.get_aperture_photometry(
            flux_stamps_calibrated, aperture_radius_range=[1, 15],
            bg_aperture_inner_radius=15,
            bg_aperture_outer_radius=18)

        filehandler = open(os.path.join(converted_dir, 'flux_photometry.obj'), 'wb')
        pickle.dump(flux_photometry, filehandler)
        filehandler.close()

        fits.writeto(os.path.join(converted_dir, 'flux_amplitude_calibrated.fits'),
                     flux_photometry['psf_flux_bg_corr_all'], overwrite=overwrite_preprocessing)

        fits.writeto(os.path.join(converted_dir, 'flux_snr.fits'),
                     flux_photometry['snr_all'], overwrite=overwrite_preprocessing)

        plt.close()
        plt.plot(flux_photometry['aperture_sizes'], flux_photometry['snr_all'][:, :, 0])
        plt.xlabel('Aperture Size (pix)')
        plt.ylabel('BG limited SNR')
        plt.savefig(os.path.join(plot_dir, 'Flux_PSF_aperture_SNR.png'))
        plt.close()
        bg_sub_flux_stamps_calibrated = flux_stamps_calibrated - \
            flux_photometry['psf_bg_counts_all'][:, :, None, None]

        fits.writeto(os.path.join(converted_dir, 'flux_stamps_calibrated_bg_corrected.fits'),
                     bg_sub_flux_stamps_calibrated.astype('float32'), overwrite=overwrite_preprocessing)

        # bg_sub_flux_phot = get_aperture_photometry(
        #     bg_sub_flux_stamps_calibrated, aperture_radius_range=[1, 15],
        #     bg_aperture_inner_radius=15,
        #     bg_aperture_outer_radius=17)

        # Make master PSF
        flux_calibration_indices, indices_of_discontinuity = flux_calibration.get_flux_calibration_indices(
            frames_info['CENTER'], frames_info['FLUX'])
        flux_calibration_indices.to_csv(os.path.join(converted_dir, 'flux_calibration_indices.csv'))
        indices_of_discontinuity.tofile(os.path.join(
            converted_dir, 'indices_of_discontinuity.csv'), sep=',')

        # Normalize all PSF frames to respective calibration flux index based on aperture size 3 pixel
        # (index 2)

        number_of_flux_frames = flux_stamps.shape[1]
        flux_calibration_frames = []

        if reduction_parameters['flux_combination_method'] == 'mean':
            comb_func = np.nanmean
        elif reduction_parameters['flux_combination_method'] == 'median':
            comb_func = np.nanmedian
        else:
            raise ValueError('Unknown flux combination method.')


        for idx in range(len(flux_calibration_indices)):
            try:
                upper_range = flux_calibration_indices['flux_idx'].iloc[idx+1]
            except IndexError:
                upper_range = number_of_flux_frames

            if idx == 0:
                lower_index = 0
                lower_index_frame_combine = 0
                number_of_frames_to_combine = upper_range - lower_index
                if reduction_parameters['exclude_first_flux_frame'] and number_of_frames_to_combine > 1:
                    lower_index_frame_combine = 1               
            else:
                lower_index = flux_calibration_indices['flux_idx'].iloc[idx]
                lower_index_frame_combine = 0
                number_of_frames_to_combine = upper_range - lower_index
                if reduction_parameters['exclude_first_flux_frame_all'] and number_of_frames_to_combine > 1:
                    lower_index_frame_combine = 1

            phot_values = flux_photometry['psf_flux_bg_corr_all'][2][:, lower_index: upper_range]
            # Old way: pick closest in time
            # reference_value_old = flux_photometry['psf_flux_bg_corr_all'][2][:, flux_calibration_indices['flux_idx'].iloc[idx]]
            # New way do mean
            reference_value = np.nanmean(
                flux_photometry['psf_flux_bg_corr_all'][2][:, lower_index_frame_combine:upper_range], axis=1)


            normalization_values = phot_values / reference_value[:, None]

            flux_calibration_frame = bg_sub_flux_stamps_calibrated[:,
                                                                   lower_index:upper_range] / normalization_values[:, :, None, None]
            # Could weight by snr, but let's not do that yet
            # snr_values = flux_photometry['snr_all'][2][:, lower_index:upper_range]
            # if reduction_parameters['exclude_first_flux_frame']:
            #     flux_calibration_frame = np.mean(flux_calibration_frame[:, 1:], axis=1)
            # else:
            flux_calibration_frame = comb_func(flux_calibration_frame[:, lower_index_frame_combine:], axis=1)
            flux_calibration_frames.append(flux_calibration_frame)

        flux_calibration_frames = np.array(flux_calibration_frames)
        flux_calibration_frames = np.swapaxes(flux_calibration_frames, 0, 1)

        fits.writeto(os.path.join(converted_dir, 'master_flux_calibrated_psf_frames.fits'),
                     flux_calibration_frames.astype('float32'), overwrite=overwrite_preprocessing)

    if spot_to_flux:
        plot_dir = os.path.join(converted_dir, 'flux_plots/')
        if not path.exists(plot_dir):
            os.makedirs(plot_dir)

        wavelengths = fits.getdata(
            os.path.join(converted_dir, 'wavelengths.fits')) * u.nm
        wavelengths = wavelengths.to(u.micron)
        flux_amplitude = fits.getdata(
            os.path.join(converted_dir, 'flux_amplitude_calibrated.fits'))[2]

        spot_amplitude = fits.getdata(
            os.path.join(converted_dir, 'spot_amplitudes.fits'))
        master_spot_amplitude = np.mean(spot_amplitude, axis=2)

        psf_flux = flux_calibration.SimpleSpectrum(
            wavelength=wavelengths,
            flux=flux_amplitude,
            norm_wavelength_range=[1.0, 1.3] * u.micron,
            metadata=frames_info['FLUX'],
            rescale=False,
            normalize=False)

        # psf_flux_norm = flux_calibration.SimpleSpectrum(
        #     wavelength=wavelengths,
        #     flux=flux_amplitude,
        #     norm_wavelength_range=[1.0, 1.3] * u.micron,
        #     metadata=frames_info['FLUX'],
        #     rescale=False,
        #     normalize=True)

        spot_flux = flux_calibration.SimpleSpectrum(
            wavelength=wavelengths,
            flux=master_spot_amplitude,  # flux_sum_with_bg,
            norm_wavelength_range=[1.0, 1.3] * u.micron,
            metadata=frames_info['CENTER'],
            rescale=True,
            normalize=False)

        # spot_flux_norm = flux_calibration.SimpleSpectrum(
        #     wavelength=wavelengths,
        #     flux=master_spot_amplitude,  # flux_sum_with_bg,
        #     norm_wavelength_range=[1.0, 1.3] * u.micron,
        #     metadata=frames_info['CENTER'],
        #     rescale=True,
        #     normalize=True)

        psf_flux.plot_flux(plot_original=False, autocolor=True, cmap=plt.cm.cool,
                           savefig=True, savedir=plot_dir, filename='psf_flux.png',
                           )
        spot_flux.plot_flux(plot_original=False, autocolor=True, cmap=plt.cm.cool,
                            savefig=True, savedir=plot_dir, filename='spot_flux_rescaled.png',
                            )

        flux_calibration_indices = pd.read_csv(os.path.join(
            converted_dir, 'flux_calibration_indices.csv'))

        normalization_factors, averaged_normalization, std_dev_normalization = flux_calibration.compute_flux_normalization_factors(
            flux_calibration_indices, psf_flux, spot_flux)

        flux_calibration.plot_flux_normalization_factors(
            flux_calibration_indices, normalization_factors[:, 1:-1],
            wavelengths=wavelengths[1:-1], cmap=plt.cm.cool,
            savefig=True, savedir=plot_dir)

        fits.writeto(os.path.join(
            converted_dir, 'spot_normalization_factors.fits'), normalization_factors,
            overwrite=True)

        fits.writeto(os.path.join(
            converted_dir, 'spot_normalization_factors_average.fits'), averaged_normalization,
            overwrite=True)

        fits.writeto(os.path.join(
            converted_dir, 'spot_normalization_factors_stddev.fits'), std_dev_normalization,
            overwrite=True)

        flux_calibration.plot_timeseries(
            frames_info['FLUX'], frames_info['CENTER'], psf_flux, spot_flux, averaged_normalization,
            x_axis_quantity='HOUR ANGLE', wavelength_channels=np.arange(len(wavelengths))[1:-1],
            savefig=True, savedir=plot_dir)

        scaled_spot_flux = spot_flux.flux * averaged_normalization[:, None]
        temporal_mean = np.nanmean(scaled_spot_flux, axis=1)
        amplitude_variation = scaled_spot_flux / temporal_mean[:, None]

        fits.writeto(os.path.join(
            converted_dir, 'spot_amplitude_variation.fits'), amplitude_variation,
            overwrite=True)

    end = time.time()
    print((end - start) / 60.)

    return None

def output_directory_path(reduction_directory, observation, method='optext'):
    """
    Create the path for the final converted file directory.

    Parameters:
    reduction_directory (str): The path to the reduction directory.
    observation (Observation): An observation object.
    method (str, optional): The reduction method. Defaults to 'optext'.

    Returns:
    str: The path for the final converted file directory.
    """

    name_mode_date = make_target_folder_string(observation)
    outputdir = path.join(
        reduction_directory, 'IFS/observation', name_mode_date, f'{method}/converted/')

    return outputdir


def check_output(reduction_directory, observation_object_list, method='optext'):
    """
    Check if all required files are present in the output directory.

    Parameters:
    reduction_directory (str): The path to the reduction directory.
    observation_object_list (list): A list of observation objects.
    method (str, optional): The reduction method. Defaults to 'optext'.

    Returns:
    tuple: A tuple containing two lists. The first list contains boolean values indicating if the required files are present for each observation. The second list contains the missing files for each observation.
    """

    reduced = []
    missing_files_reduction = []

    for observation in observation_object_list:
        outputdir = output_directory_path(
            reduction_directory, 
            observation,
            method)
        
        files_to_check = [
            'wavelengths.fits',
            'coro_cube.fits',
            'center_cube.fits',
            'flux_stamps_calibrated_bg_corrected.fits',
            'frames_info_flux.csv',
            'frames_info_center.csv',
            'frames_info_coro.csv',
            'image_centers_fitted_robust.fits',
            'spot_amplitudes.fits',
        ]

        missing_files = []
        for file in files_to_check:
            if not path.isfile(path.join(outputdir, file)):
                missing_files.append(file)
        if len(missing_files) > 0:
            reduced.append(False)
        else:
            reduced.append(True)
        missing_files_reduction.append(missing_files)
    
    return reduced, missing_files_reduction