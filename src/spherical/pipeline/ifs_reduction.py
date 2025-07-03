"""
VLT/SPHERE IFS Data Reduction Pipeline.

This module implements the complete end-to-end data reduction pipeline for
ESO VLT/SPHERE Integral Field Spectrograph (IFS) observations, optimized for
high-contrast imaging and exoplanet detection. The pipeline adapts the CHARIS
reduction algorithms for SPHERE data, providing automated wavelength calibration,
optimal spectral extraction, and comprehensive astrometric and photometric
calibrations.

The reduction methodology follows Samland et al. (2022), implementing robust
algorithms for:
- Wavelength calibration using arc lamp exposures
- Optimal extraction of spectral cubes from raw detector images  
- Satellite spot astrometry for precise registration
- Flux calibration using unsaturated stellar PSF references
- Background subtraction and bad pixel correction
- Quality assessment and provenance tracking

Key Features
------------
- Automated configuration based on observing mode (YJ/H band)
- Parallel processing for computational efficiency
- Comprehensive error handling and crash recovery
- Integration with post-processing tools (TRAP algorithm)
- Standardized output formats compatible with scientific analysis tools

Scientific Context
------------------
SPHERE IFS operates in the near-infrared (1.0-2.3 μm) with moderate spectral
resolution (R~35-55) optimized for detecting and characterizing exoplanets and
circumstellar disks. The pipeline handles both coronagraphic observations for
direct imaging and satellite spot observations for precise astrometry.

Wavelength coverage and resolution:
- YJ band (OBS_YJ): 0.95-1.35 μm, R~55
- H band (OBS_H): 1.45-1.85 μm, R~35  
- YJH band: 0.95-1.85 μm (dual-band mode)

All wavelengths are vacuum wavelengths following IAU standards, with
astrometric solutions referenced to the ICRS coordinate system.

Examples
--------
Basic reduction with default configuration:

>>> from spherical.pipeline.ifs_reduction import execute_targets
>>> from spherical.database import Sphere_database
>>> database = Sphere_database(table_obs, table_files)
>>> observations = database.get_observation_SIMBAD("HD 95086") 
>>> execute_targets(observations)

Custom high-performance configuration:

>>> from spherical.pipeline.pipeline_config import IFSReductionConfig
>>> config = IFSReductionConfig()
>>> config.resources.ncpu_extract = 32
>>> config.extraction.method = "optimal"
>>> execute_targets(observations, config=config)

See Also
--------
spherical.pipeline.pipeline_config : Configuration management
spherical.pipeline.steps : Individual processing steps
spherical.database : Observation database interface

References
----------
.. [1] Samland et al. (2022), "SPHERE Data Reduction and Analysis Pipeline",
   A&A, 668, A84
.. [2] Samland et al. (2021), "TRAP: Temporal Reference Analysis of Planets",
   ApJ, 919, 15
"""
from __future__ import annotations

import copy
import os
import time
import traceback
from glob import glob
from os import path
from pathlib import Path
from typing import Union

# Third-party imports
import charis
import matplotlib

from spherical.database.ifs_observation import IFSObservation
from spherical.database.irdis_observation import IRDISObservation
from spherical.pipeline.logging_utils import (
    PipelineLoggerAdapter,
    get_pipeline_log_context,
    get_pipeline_logger,
    remove_queue_listener,
)

matplotlib.use(backend='Agg')  # Must be set before any matplotlib imports

# Local imports
from spherical.pipeline.pipeline_config import IFSReductionConfig, defaultIFSReduction
from spherical.pipeline.steps.bundle_output import run_bundle_output
from spherical.pipeline.steps.download_data import download_data_for_observation, update_observation_file_paths
from spherical.pipeline.steps.extract_cubes import extract_cubes_with_multiprocessing
from spherical.pipeline.steps.find_star import fit_centers_in_parallel
from spherical.pipeline.steps.flux_psf_calibration import run_flux_psf_calibration
from spherical.pipeline.steps.frame_info import run_frame_info_computation
from spherical.pipeline.steps.plot_center_evolution import run_image_center_evolution_plot
from spherical.pipeline.steps.process_centers import run_polynomial_center_fit
from spherical.pipeline.steps.spot_photometry import run_spot_photometry_calibration
from spherical.pipeline.steps.spot_to_flux import run_spot_to_flux_normalization
from spherical.pipeline.steps.wavelength_calibration import run_wavelength_calibration
from spherical.pipeline.toolbox import make_target_folder_string


def execute_targets(
        observations: Union[IFSObservation, IRDISObservation, list[Union[IFSObservation, IRDISObservation]]],
        config: IFSReductionConfig | None = None):
    """
    Execute VLT/SPHERE IFS reduction pipeline for multiple observations.

    This function provides a convenient wrapper to process multiple SPHERE
    observations sequentially using the same configuration. It handles both
    single observations and lists of observations, applying the full end-to-end
    IFS data reduction pipeline including wavelength calibration, cube
    extraction, astrometric calibration, and photometric calibration.

    Parameters
    ----------
    observations : IFSObservation, IRDISObservation, or list thereof
        Single observation object or list of observation objects to process.
        Each observation must contain valid SPHERE data with the required
        frame types (CORO, CENTER, FLUX) and calibration data.
    config : IFSReductionConfig, optional
        Configuration object containing all pipeline parameters including
        directory paths, processing steps, resource allocation, and algorithm
        settings. If None, a default IFS configuration will be used for IFS
        observations. Default is None.

    Returns
    -------
    None
        This function does not return values but writes reduced data products
        to disk in the configured reduction directory.

    See Also
    --------
    execute_target : Process a single observation
    IFSReductionConfig : Configuration class for pipeline parameters

    Notes
    -----
    The function automatically applies resource configuration to ensure
    consistent CPU usage across all pipeline steps. Each observation is
    processed independently, allowing partial completion if errors occur
    with individual targets.

    Examples
    --------
    Process multiple observations with default configuration:

    >>> from spherical.pipeline.ifs_reduction import execute_targets
    >>> execute_targets(observation_list)

    Process with custom configuration:

    >>> from spherical.pipeline.pipeline_config import IFSReductionConfig
    >>> config = IFSReductionConfig()
    >>> config.resources.ncpu_extract = 8
    >>> execute_targets(observation_list, config=config)
    """
    # Handle both single observation and list of observations
    if not isinstance(observations, list):
        observations = [observations]
    
    for observation in observations:
        execute_target(
            observation=observation,
            config=config
        )


def execute_target(
        observation: Union[IFSObservation, IRDISObservation],
        config: IFSReductionConfig | None = None):
    """
    Execute complete VLT/SPHERE IFS data reduction pipeline for single observation.

    This function implements the full end-to-end reduction workflow for SPHERE
    IFS observations, adapting the CHARIS pipeline for SPHERE data. The pipeline
    includes data download, wavelength calibration, spectral cube extraction,
    astrometric and photometric calibration, and post-processing steps optimized
    for high-contrast imaging and exoplanet detection.

    The reduction follows the methodology described in Samland et al. (2022),
    with automatic configuration based on observing band (YJ: R~55, H: R~35)
    and robust handling of different observing modes including coronagraphic
    and satellite spot observations.

    Parameters
    ----------
    observation : IFSObservation or IRDISObservation
        SPHERE observation object containing metadata, file paths, and frame
        information. Must include valid CORO/CENTER/FLUX frames and associated
        calibration data (wavelength calibration, flats, darks).
    config : IFSReductionConfig, optional
        Configuration object specifying all pipeline parameters including:
        - Directory paths for raw data and reduction outputs
        - Processing steps to execute (calibration, extraction, etc.)
        - Algorithm parameters (extraction method, background subtraction)
        - Resource allocation (CPU cores for parallel processing)
        If None, default IFS configuration is used for IFS instruments.
        Default is None.

    Returns
    -------
    None
        Function writes reduced data products to disk but returns None.
        Output files are saved in a structured directory hierarchy:
        {reduction_dir}/IFS/observation/{target}/{filter}/{date}/

    Raises
    ------
    ValueError
        If no configuration is provided for non-IFS instruments, or if no
        valid frame types are found in the observation.
    FileNotFoundError
        If required calibration files or raw data are missing.

    See Also
    --------
    execute_targets : Process multiple observations
    IFSReductionConfig : Configuration class
    spherical.pipeline.steps : Individual pipeline step modules

    Notes
    -----
    The pipeline automatically configures spectral resolution based on filter:
    - OBS_YJ: R = 55 (1.1-1.35 μm)
    - OBS_H: R = 35 (1.45-1.85 μm)

    Processing steps include:
    1. Data download from ESO archive (if enabled)
    2. Wavelength calibration using WAVECAL frames
    3. Spectral cube extraction with optimal extraction algorithm
    4. Astrometric calibration using satellite spots or unsaturated PSF
    5. Photometric calibration using flux reference frames
    6. Quality assessment and metadata generation

    All intermediate and final data products preserve FITS header information
    and include comprehensive provenance tracking.

    Examples
    --------
    Basic usage with default configuration:

    >>> from spherical.database import Sphere_database
    >>> from spherical.pipeline.ifs_reduction import execute_target
    >>> observation = database.get_observation_by_id("target_name")
    >>> execute_target(observation)

    Custom configuration for high-performance computing:

    >>> from spherical.pipeline.pipeline_config import IFSReductionConfig
    >>> config = IFSReductionConfig()
    >>> config.resources.ncpu_extract = 32
    >>> config.extraction.method = "optimal"
    >>> execute_target(observation, config=config)
    """
    
    # Get instrument from observation object
    instrument = str(observation.observation['INSTRUMENT'][0]).lower()

    # Only instantiate default IFS config if instrument is IFS and no config provided
    if config is None:
        if instrument == 'ifs':
            config = defaultIFSReduction()
        else:
            raise ValueError(f"No configuration provided and instrument '{instrument}' is not supported. Only 'ifs' is currently supported.")
    
    # Apply resource configuration to all sub-configs
    config.apply_resources()
    
    # Extract step configuration for cleaner code
    steps = config.steps    
    
    # Get directory paths from config
    raw_directory = config.directories.raw_directory
    reduction_directory = config.directories.reduction_directory
    
    calibration_parameters, extraction_parameters, reduction_parameters, directories_parameters = config.as_plain_dicts()

    # Get frame_types_to_extract from config and filter to only include types that have files
    # This automatically handles the case where users specify frame types that don't exist
    available_frame_types = []
    for frame_type in config.preprocessing.frame_types_to_extract:
        if (frame_type in observation.frames and 
            observation.frames[frame_type] is not None and 
            len(observation.frames[frame_type]) > 0):
            available_frame_types.append(frame_type)
    
    frame_types_to_extract = available_frame_types
    if not frame_types_to_extract:
        raise ValueError("No frame types with available files found in observation")

    start = time.time()

    observation = copy.copy(observation)

    target_name = str(observation.observation['MAIN_ID'][0])
    target_name = " ".join(target_name.split())
    target_name = target_name.replace(" ", "_")
    obs_band = str(observation.observation['FILTER'][0])
    date = str(observation.observation['NIGHT_START'][0])
    observation.target_name = target_name  # type: ignore
    observation.obs_band = obs_band  # type: ignore
    observation.date = date  # type: ignore
    
    name_mode_date = target_name + '/' + obs_band + '/' + date
    
    outputdir = Path(str(reduction_directory)) / f"{instrument.upper()}/observation" / name_mode_date
    outputdir.mkdir(parents=True, exist_ok=True)

    context = get_pipeline_log_context(observation)

    logger = get_pipeline_logger(name_mode_date, outputdir, verbose=config.extraction.verbose)
    logger = PipelineLoggerAdapter(logger, context)

    logger.info("Pipeline session started", extra={"step": "session_start", "status": "started"})

    # outputdir = path.join(
    #     reduction_directory, f'{instrument.upper()}/observation', name_mode_date)
    cube_outputdir = path.join(outputdir, '{}'.format(
        extraction_parameters['method']))
    converted_dir = path.join(cube_outputdir, 'converted') + '/'

    try:
        logger.info(f"Start reduction of: {name_mode_date}")
        logger.info(f"Processing frame types: {', '.join(frame_types_to_extract)}")

        if obs_band == 'OBS_YJ':
            instrument = charis.instruments.SPHERE('YJ')
            config.extraction.R = 55
        elif obs_band == 'OBS_H':
            instrument = charis.instruments.SPHERE('YH')
            config.extraction.R = 35
        
        # Regenerate parameter dictionaries after config updates
        calibration_parameters, extraction_parameters, reduction_parameters, directories_parameters = config.as_plain_dicts()

        if steps.download_data:
            _ = download_data_for_observation(raw_directory=str(raw_directory), observation=observation, eso_username=config.preprocessing.eso_username, logger=logger)

        existing_file_paths = glob(os.path.join(str(raw_directory) or '', '**', 'SPHER.*.fits'), recursive=True)
        used_keys = ['CORO', 'CENTER', 'FLUX', 'WAVECAL']
        if steps.reduce_calibration:
            used_keys += ['WAVECAL']

        non_least_square_methods = ['optext', 'apphot3', 'apphot5']

        if steps.reduce_calibration or steps.extract_cubes or steps.bundle_output:
            update_observation_file_paths(existing_file_paths, observation, logger=logger)

        calibration_time_name = str(observation.frames['WAVECAL']['DP.ID'][0][6:])  # type: ignore
        wavecal_outputdir = os.path.join(str(reduction_directory), 'IFS/calibration', obs_band, calibration_time_name)
        
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        # TODO: Move default calibration parameters to a config file or constant module, only overwrite keywords if provided
        if steps.reduce_calibration:
            run_wavelength_calibration(
                observation=observation,
                instrument=instrument,
                calibration_parameters=calibration_parameters,
                wavecal_outputdir=wavecal_outputdir,
                overwrite_calibration=steps.overwrite_calibration,
                logger=logger,
            )

        if steps.extract_cubes:
            extract_cubes_with_multiprocessing(
                observation=observation,
                frame_types_to_extract=['CORO', 'CENTER', 'FLUX'],
                extraction_parameters=extraction_parameters,
                reduction_parameters=reduction_parameters,
                wavecal_outputdir=wavecal_outputdir,
                cube_outputdir=cube_outputdir,
                logger=logger,
                non_least_square_methods=non_least_square_methods,
                extract_cubes=steps.extract_cubes,
            )

        if steps.bundle_output:
            run_bundle_output(
                frame_types_to_extract=frame_types_to_extract,
                cube_outputdir=cube_outputdir,
                converted_dir=converted_dir,
                extraction_parameters=extraction_parameters,
                instrument=instrument,
                non_least_square_methods=non_least_square_methods,
                overwrite_bundle=steps.overwrite_bundle,
                bundle_hexagons=steps.bundle_hexagons,
                bundle_residuals=steps.bundle_residuals,
                logger=logger,
            )

        if steps.compute_frames_info:
            run_frame_info_computation(observation, converted_dir, logger=logger)

        if steps.find_centers:
            fit_centers_in_parallel(
                converted_dir=converted_dir,
                observation=observation,
                overwrite=steps.overwrite_preprocessing,
                ncpu=reduction_parameters['ncpu_find_center'],
                logger=logger,
                )

        if steps.process_extracted_centers:
            run_polynomial_center_fit(
                converted_dir=converted_dir,
                extraction_parameters=extraction_parameters,
                non_least_square_methods=non_least_square_methods,
                overwrite_preprocessing=steps.overwrite_preprocessing,
                logger=logger,
            )

        if steps.plot_image_center_evolution:
            run_image_center_evolution_plot(converted_dir, logger=logger)

        if steps.calibrate_spot_photometry:
            run_spot_photometry_calibration(converted_dir, steps.overwrite_preprocessing, logger=logger)

        if steps.calibrate_flux_psf:
            run_flux_psf_calibration(converted_dir, steps.overwrite_preprocessing, reduction_parameters, logger=logger)
            
        if steps.spot_to_flux:
            run_spot_to_flux_normalization(converted_dir, reduction_parameters, logger=logger)
        
        end = time.time()
        logger.info(f"Reduction finished in {(end - start) / 60.:.2f} minutes.")

        return None

    except Exception:
        logger.exception("Pipeline execution failed.")
        crash_report_path = outputdir / 'crash_report.txt'
        with open(crash_report_path, 'w') as f:
            f.write(f"An error occurred during the reduction of {name_mode_date}.\n\n")
            traceback.print_exc(file=f)
        logger.info(f"Crash report saved to {crash_report_path}")
        return None

    finally:
        remove_queue_listener()


def output_directory_path(reduction_directory, observation: Union[IFSObservation, IRDISObservation], method='optext'):
    """
    Generate standardized output directory path for reduced SPHERE data products.

    Creates the canonical directory structure used by the spherical pipeline
    for organizing reduced data products. The path follows the convention:
    {reduction_directory}/IFS/observation/{target_name}/{filter}/{date}/{method}/converted/

    This standardized structure ensures consistent data organization and
    facilitates automated analysis workflows and data discovery.

    Parameters
    ----------
    reduction_directory : str or Path
        Root directory for all reduced data products. This should be a
        persistent storage location with sufficient space for large
        spectral cubes and intermediate data products.
    observation : IFSObservation or IRDISObservation
        SPHERE observation object containing target metadata including
        MAIN_ID (target name), FILTER (observing band), and NIGHT_START
        (observation date). These fields are used to construct the
        hierarchical directory structure.
    method : str, optional
        Spectral extraction method identifier. Common values include:
        - 'optext': Optimal extraction (default)
        - 'apphot3': 3-pixel aperture photometry
        - 'apphot5': 5-pixel aperture photometry
        Default is 'optext'.

    Returns
    -------
    str
        Complete absolute path to the converted data directory where
        final reduced data products are stored. The directory includes
        spectral cubes, wavelength solutions, astrometric solutions,
        and photometric calibrations.

    See Also
    --------
    check_output : Verify presence of expected output files
    make_target_folder_string : Generate target folder string

    Notes
    -----
    The directory structure is designed to support:
    - Multiple extraction methods for the same observation
    - Easy identification of data by target, filter, and date
    - Separation of different processing stages (raw cubes vs converted)
    - Integration with post-processing tools like TRAP

    The 'converted' subdirectory contains final data products ready for
    scientific analysis, while parent directories may contain intermediate
    processing stages.

    Examples
    --------
    Generate output path for default optimal extraction:

    >>> from spherical.pipeline.ifs_reduction import output_directory_path
    >>> output_path = output_directory_path("/data/sphere/reduced", observation)
    >>> print(output_path)
    /data/sphere/reduced/IFS/observation/HD_95086/OBS_H/2023-04-15/optext/converted/

    Specify custom extraction method:

    >>> output_path = output_directory_path(
    ...     "/data/sphere/reduced", observation, method="apphot5"
    ... )
    """

    name_mode_date = make_target_folder_string(observation)
    outputdir = path.join(
        reduction_directory, 'IFS/observation', name_mode_date, f'{method}/converted/')

    return outputdir


def check_output(reduction_directory, observation_object_list: list[Union[IFSObservation, IRDISObservation]], method='optext'):
    """
    Verify completeness of SPHERE IFS reduction pipeline output files.

    Performs systematic verification that all expected data products from the
    IFS reduction pipeline are present and accessible. This function is
    essential for quality assurance and identifying incomplete reductions
    before proceeding to post-processing or scientific analysis.

    The verification checks for critical files including spectral cubes,
    wavelength calibrations, astrometric solutions, and photometric
    calibrations required for high-contrast imaging analysis.

    Parameters
    ----------
    reduction_directory : str or Path
        Root directory containing reduced data products. Should match the
        reduction_directory used in the original pipeline execution.
    observation_object_list : list of IFSObservation or IRDISObservation
        List of SPHERE observation objects to verify. Each observation
        should have completed the reduction pipeline.
    method : str, optional
        Spectral extraction method to verify. Must match the method used
        during reduction. Common values:
        - 'optext': Optimal extraction (default)
        - 'apphot3': 3-pixel aperture photometry  
        - 'apphot5': 5-pixel aperture photometry
        Default is 'optext'.

    Returns
    -------
    reduced : list of bool
        Boolean list indicating completion status for each observation.
        True indicates all required files are present, False indicates
        missing files or incomplete reduction.
    missing_files_reduction : list of list of str
        Nested list containing names of missing files for each observation.
        Empty lists indicate complete reductions. File names correspond to
        expected SPHERE IFS data products.

    See Also
    --------
    output_directory_path : Generate output directory paths
    execute_target : Main reduction pipeline function

    Notes
    -----
    Required files checked include:
    - wavelengths.fits: Wavelength calibration solution (nm units)
    - coro_cube.fits: Coronagraphic science data cube
    - center_cube.fits: Unsaturated PSF reference data cube  
    - flux_stamps_calibrated_bg_corrected.fits: Photometric reference data
    - frames_info_*.csv: Frame metadata and quality flags
    - image_centers_fitted_robust.fits: Astrometric solution
    - spot_amplitudes.fits: Satellite spot photometry

    Missing files may indicate:
    - Incomplete pipeline execution
    - Insufficient observing time for specific frame types
    - Calibration data unavailability
    - Processing errors requiring investigation

    Examples
    --------
    Check completion for multiple observations:

    >>> from spherical.pipeline.ifs_reduction import check_output
    >>> reduced, missing = check_output(
    ...     "/data/sphere/reduced", observation_list
    ... )
    >>> incomplete = [obs for i, obs in enumerate(observation_list) 
    ...               if not reduced[i]]

    Identify specific missing files:

    >>> for i, obs in enumerate(observation_list):
    ...     if missing[i]:
    ...         print(f"{obs}: Missing {missing[i]}")
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