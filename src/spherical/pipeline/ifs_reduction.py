from __future__ import annotations

import copy
import logging
import os

# import warnings
import time
from glob import glob
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
from astropy.stats import sigma_clip
from photutils.aperture import CircularAnnulus, CircularAperture

from spherical.pipeline import flux_calibration, toolbox, transmission
from spherical.pipeline.steps.bundle_output import run_bundle_output
from spherical.pipeline.steps.download_data import download_data_for_observation, update_observation_file_paths
from spherical.pipeline.steps.extract_cubes import extract_cubes_with_multiprocessing
from spherical.pipeline.steps.find_star import fit_centers_in_parallel, star_centers_from_PSF_img_cube
from spherical.pipeline.steps.frame_info import run_frame_info_computation
from spherical.pipeline.steps.wavelength_calibration import run_wavelength_calibration
from spherical.pipeline.toolbox import make_target_folder_string

# Create a module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def execute_target(
        observation,
        calibration_parameters,
        extraction_parameters,
        reduction_parameters,
        reduction_directory,
        instrument='ifs',
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

    target_name = observation.observation['MAIN_ID'][0]
    target_name = " ".join(target_name.split())
    target_name = target_name.replace(" ", "_")
    obs_band = observation.observation['FILTER'][0]
    date = observation.observation['NIGHT_START'][0]
    name_mode_date = target_name + '/' + obs_band + '/' + date
        
    outputdir = path.join(
        reduction_directory, f'{instrument.upper()}/observation', name_mode_date)
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

    if download_data:
        _ = download_data_for_observation(raw_directory=raw_directory, observation=observation, eso_username=eso_username)

    existing_file_paths = glob(os.path.join(raw_directory, '**', 'SPHER.*.fits'), recursive=True)
    used_keys = ['CORO', 'CENTER', 'FLUX', 'WAVECAL']
    if reduce_calibration:
        used_keys += ['WAVECAL']

    non_least_square_methods = ['optext', 'apphot3', 'apphot5']

    if reduce_calibration or extract_cubes or bundle_output:
        update_observation_file_paths(existing_file_paths, observation)

    calibration_time_name = observation.frames['WAVECAL']['DP.ID'][0][6:]
    wavecal_outputdir = os.path.join(reduction_directory, 'IFS/calibration', obs_band, calibration_time_name)
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # TODO: Move default calibration parameters to a config file or constant module, only overwrite keywords if provided
    if reduce_calibration:
        run_wavelength_calibration(
            observation=observation,
            instrument=instrument,
            calibration_parameters=calibration_parameters,
            wavecal_outputdir=wavecal_outputdir,
            overwrite_calibration=overwrite_calibration
        )

    if extract_cubes:
        extract_cubes_with_multiprocessing(
            observation=observation,
            frame_types_to_extract=['CORO', 'CENTER', 'FLUX'],
            extraction_parameters=extraction_parameters,
            reduction_parameters=reduction_parameters,
            wavecal_outputdir=wavecal_outputdir,
            cube_outputdir=cube_outputdir,
            non_least_square_methods=non_least_square_methods,
            extract_cubes=extract_cubes)


    if bundle_output:
        run_bundle_output(
            frame_types_to_extract=frame_types_to_extract,
            cube_outputdir=cube_outputdir,
            converted_dir=converted_dir,
            extraction_parameters=extraction_parameters,
            instrument=instrument,
            non_least_square_methods=non_least_square_methods,
            overwrite_bundle=overwrite_bundle,
            bundle_hexagons=bundle_hexagons,
            bundle_residuals=bundle_residuals
        )

    if compute_frames_info:
        run_frame_info_computation(observation, converted_dir)

    if find_centers:
        fit_centers_in_parallel(
            converted_dir=converted_dir,
            observation=observation,
            overwrite=overwrite_preprocessing,
            ncpu=reduction_parameters['ncpu_find_center'])

    if process_extracted_centers:
        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit
        run_polynomial_center_fit(
            converted_dir=converted_dir,
            extraction_parameters=extraction_parameters,
            non_least_square_methods=non_least_square_methods,
            overwrite_preprocessing=overwrite_preprocessing
        )

    if plot_image_center_evolution:
        from spherical.pipeline.steps.plot_center_evolution import run_image_center_evolution_plot
        run_image_center_evolution_plot(converted_dir)

    if calibrate_spot_photometry:
        from spherical.pipeline.steps.spot_photometry import run_spot_photometry_calibration
        run_spot_photometry_calibration(converted_dir, overwrite_preprocessing)

    if calibrate_flux_psf:
        from spherical.pipeline.steps.flux_psf_calibration import run_flux_psf_calibration
        run_flux_psf_calibration(converted_dir, overwrite_preprocessing, reduction_parameters)
        
    if spot_to_flux:
        from spherical.pipeline.steps.spot_to_flux import run_spot_to_flux_normalization
        run_spot_to_flux_normalization(converted_dir, reduction_parameters)
    
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