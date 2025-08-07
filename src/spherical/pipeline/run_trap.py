"""
TRAP Post-Processing Pipeline for SPHERE IFS Observations.

This module provides wrapper functions for running TRAP (Temporal Reference 
Analysis of Planets) post-processing on reduced SPHERE IFS data. TRAP is a 
sophisticated algorithm for high-contrast imaging that uses temporal reference 
differential imaging to detect and characterize exoplanets in spectroscopic data.

The functions in this module wrap the existing TRAP functionality to provide
a consistent interface similar to the main IFS reduction pipeline, enabling
seamless integration into automated data processing workflows.
"""

import copy
import os
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import ray
from astropy import units as u
from astropy.io import fits
from trap.detection import DetectionAnalysis
from trap.reduction_wrapper import run_complete_reduction

from spherical.database.ifs_observation import IFSObservation
from spherical.database.irdis_observation import IRDISObservation
from spherical.pipeline import ifs_reduction
from spherical.pipeline.logging_utils import (
    PipelineLoggerAdapter,
    get_pipeline_log_context,
    get_pipeline_logger,
    remove_queue_listener,
)
from spherical.pipeline.pipeline_config import IFSReductionConfig
from spherical.pipeline.toolbox import make_target_folder_string


def run_trap_on_observation(
    observation: Union[IFSObservation, IRDISObservation],
    trap_config,
    reduction_config: IFSReductionConfig,
    species_database_directory: Union[str, Path],
) -> None:
    """
    Run TRAP post-processing on a single SPHERE IFS observation.
    
    This function performs TRAP (Temporal Reference Analysis of Planets) 
    post-processing on reduced SPHERE IFS data, including both the reduction
    and detection phases. It processes a single observation and applies the
    TRAP algorithm for high-contrast imaging and exoplanet detection.
    
    Parameters
    ----------
    observation : IFSObservation or IRDISObservation
        SPHERE observation object containing metadata and file information
        for the observation to process.
    trap_config : TrapConfig
        TRAP configuration object containing all parameters for the TRAP
        algorithm including temporal components fraction, detection thresholds,
        and stellar parameters.
    reduction_config : IFSReductionConfig
        IFS reduction configuration object containing directory paths and
        extraction method settings needed to locate the reduced data.
    species_database_directory : str or Path
        Path to the species database directory used for stellar template
        matching during detection and characterization.
        
    Returns
    -------
    None
        Function writes TRAP outputs to disk but returns None.
        
    Raises
    ------
    AssertionError
        If observation is not done with IFS instrument (OBS_YJ or OBS_H).
        
    Notes
    -----
    This function expects that the IFS reduction pipeline has already been
    run and that the following files exist in the data directory:
    - wavelengths.fits
    - frames_info_{file_identifier}.csv  
    - {file_identifier}_cube.fits
    - psf_cube_for_postprocessing.fits
    - image_centers_fitted_robust.fits
    
    The TRAP reduction and detection steps are controlled by the settings
    in `reduction_config.steps.run_trap_reduction` and 
    `reduction_config.steps.run_trap_detection` respectively.
    
    The function automatically determines the file identifier based on
    the WAFFLE_MODE: "center" for continuous satellite spots, "coro" otherwise.
    """
    if not reduction_config.steps.run_trap_reduction and not reduction_config.steps.run_trap_detection:
        print("No TRAP processing selected to run. Skipping observation.")
        return
        
    obs_mode = observation.observation['FILTER'][0]
    assert obs_mode in ['OBS_YJ', 'OBS_H'], "Observation has to be done with IFS."

    continuous_satellite_spots = observation.observation['WAFFLE_MODE'][0]

    # Create instrument from TRAP configuration
    used_instrument = trap_config.get_instrument(obs_mode)

    data_directory = ifs_reduction.output_directory_path(
        str(reduction_config.directories.reduction_directory),
        observation,
        method=reduction_config.extraction.method)
    
    # Extract and set observation attributes (following IFS reduction pattern)
    observation = copy.deepcopy(observation)
    target_name = str(observation.observation['MAIN_ID'][0])
    target_name = " ".join(target_name.split())
    target_name = target_name.replace(" ", "_")
    obs_band = str(observation.observation['FILTER'][0])
    date = str(observation.observation['NIGHT_START'][0])
    observation.target_name = target_name  # type: ignore
    observation.obs_band = obs_band  # type: ignore
    observation.date = date  # type: ignore
    
    name_mode_date = make_target_folder_string(observation)
    result_folder = os.path.join(str(reduction_config.directories.reduction_directory), 'IFS/trap', name_mode_date)
    
    # Create TRAP result folder
    os.makedirs(result_folder, exist_ok=True)

    # Initialize logging for TRAP session with trap_ prefix for log files
    context = get_pipeline_log_context(observation)
    logger = get_pipeline_logger(
        f"trap_{name_mode_date}", 
        Path(result_folder), 
        verbose=trap_config.processing.verbose,
        log_prefix="trap_reduction"
    )
    logger = PipelineLoggerAdapter(logger, context)

    start_time = time.time()
    logger.info("TRAP session started", extra={"step": "trap_session_start", "status": "started"})

    try:
        # Early exit logging
        if not reduction_config.steps.run_trap_reduction and not reduction_config.steps.run_trap_detection:
            logger.info("No TRAP processing steps enabled", extra={"step": "trap_session", "status": "skipped"})
            return

        # Log observation details
        logger.info(f"Starting TRAP processing for {name_mode_date}")
        logger.info(f"TRAP steps enabled - Reduction: {reduction_config.steps.run_trap_reduction}, Detection: {reduction_config.steps.run_trap_detection}")
        logger.debug(f"TRAP result folder: {result_folder}")
    
        trap_parameters = trap_config.get_reduction_parameters()
        trap_parameters.result_folder = result_folder

        if continuous_satellite_spots:
            file_identifier = "center"
        else:
            file_identifier = "coro"

        logger.debug(f"File identifier: {file_identifier}")
        logger.debug(f"Temporal components fraction: {trap_config.processing.temporal_components_fraction}")

        wavelengths = (
            fits.getdata(os.path.join(data_directory, "wavelengths.fits")) * u.nm
        ).to(u.micron)
        used_instrument.wavelengths = wavelengths

        pa = pd.read_csv(
            os.path.join(data_directory, f"frames_info_{file_identifier}.csv")
        )['DEROT ANGLE'].values

        data_full = fits.getdata(
            os.path.join(data_directory, f"{file_identifier}_cube.fits")
        )
        try:
            flux_psf_full = fits.getdata(
                os.path.join(data_directory, "psf_cube_for_postprocessing.fits")
            )
        except FileNotFoundError:
            flux_psf_full = fits.getdata(
                os.path.join(data_directory, "master_flux_calibrated_psf_frames.fits")
            )
        flux_psf_full = np.nanmean(flux_psf_full, axis=1)

        xy_image_centers = fits.getdata(
            os.path.join(data_directory, "image_centers_fitted_robust.fits")
        )
        if not continuous_satellite_spots:
            xy_image_centers = np.nanmean(xy_image_centers, axis=1)
            xy_image_centers = xy_image_centers[:, None, :].repeat(len(pa), axis=1)

        # Waffle amplitudes
        amplitude_modulation_full = None
        inverse_variance_full = None
        bad_pixel_mask_full = None
        bad_frames = None

        # Get configured parameters
        wavelength_indices = np.array(trap_config.processing.wavelength_indices)
        temporal_components_fraction = trap_config.processing.temporal_components_fraction

        if reduction_config.steps.run_trap_reduction:
            logger.info("Starting TRAP reduction", extra={"step": "trap_reduction", "status": "started"})
            logger.debug(f"Wavelength indices: {wavelength_indices}")
            logger.debug(f"Data cube shape: {data_full.shape}")
            
            try:
                _ = run_complete_reduction(
                    data_full=data_full,
                    flux_psf_full=flux_psf_full,
                    pa=pa,
                    instrument=used_instrument,
                    reduction_parameters=deepcopy(trap_parameters),
                    temporal_components_fraction=temporal_components_fraction,
                    wavelength_indices=wavelength_indices,
                    inverse_variance_full=inverse_variance_full,
                    bad_frames=bad_frames,
                    amplitude_modulation_full=amplitude_modulation_full,
                    xy_image_centers=xy_image_centers,
                    overwrite=trap_config.processing.overwrite_reduction,
                    verbose=trap_config.processing.verbose,
                )
                
                logger.info("TRAP reduction completed", extra={"step": "trap_reduction", "status": "success"})
                
            except Exception as e:
                logger.error(f"TRAP reduction failed: {str(e)}", extra={"step": "trap_reduction", "status": "failed"})
                
                # Ensure Ray server is properly shut down
                try:
                    logger.info("Shutting down Ray server due to TRAP reduction error")
                    ray.shutdown()
                    
                    # Wait a bit to ensure proper shutdown
                    time.sleep(2)
                    logger.info("Ray server shutdown completed")
                    
                except Exception as ray_error:
                    logger.warning(f"Error during Ray shutdown: {str(ray_error)}")
                
                # Re-raise the original exception
                raise

        if reduction_config.steps.run_trap_detection:
            logger.info("Starting TRAP detection", extra={"step": "trap_detection", "status": "started"})
            
            # Fix B: Enhanced diagnostic logging for TRAP parameters
            logger.debug("TRAP detection parameters:")
            logger.debug(f"  - Detection threshold: {trap_config.detection.detection_threshold}")
            logger.debug(f"  - Candidate threshold: {trap_config.detection.candidate_threshold}")
            logger.debug(f"  - Search radius: {trap_config.detection.search_radius} pixels")
            logger.debug(f"  - Use spectral correlation: {trap_config.detection.use_spectral_correlation}")
            logger.debug(f"  - Inner mask radius: {trap_config.detection.inner_mask_radius}")
            logger.debug(f"  - Good fraction threshold: {trap_config.detection.good_fraction_threshold}")
            logger.debug(f"  - Theta deviation threshold: {trap_config.detection.theta_deviation_threshold}")
            logger.debug(f"  - YX FWHM ratio threshold: {trap_config.detection.yx_fwhm_ratio_threshold}")
            
            # Stellar parameters diagnostic
            stellar_params = trap_config.get_stellar_parameters()
            logger.debug("Stellar parameters:")
            logger.debug(f"  - Effective temperature: {stellar_params.get('teff', 'Not set')} K")
            logger.debug(f"  - Surface gravity: {stellar_params.get('logg', 'Not set')}")
            logger.debug(f"  - Metallicity: {stellar_params.get('feh', 'Not set')}")
            
            # Processing parameters diagnostic
            logger.debug("Processing parameters:")
            logger.debug(f"  - Temporal components fraction: {trap_config.processing.temporal_components_fraction}")
            logger.debug(f"  - Species database directory: {species_database_directory}")
            
            analysis = DetectionAnalysis(
                reduction_parameters=None,
                instrument=None,
            )
            
            logger.debug("Reading TRAP reduction outputs")
            analysis.read_output(
                trap_config.processing.temporal_components_fraction[0],
                result_folder=trap_parameters.result_folder,
                reduction_type="temporal",
                correlated_residuals=False,
                read_parameters=True,
                read_instrument=True,
            )

            # Update result folder in case data was copied after reduction phase
            # The saved parameters don't know about potential folder moves
            analysis.reduction_parameters.result_folder = trap_parameters.result_folder

            if trap_config.processing.verbose:
                logger.debug("Parameter consistency check:")
                logger.debug(f"Config temporal_components_fraction: {trap_config.processing.temporal_components_fraction}")
                logger.debug(f"Config annulus_width: {trap_config.reduction.annulus_width}")
                logger.debug(f"Loaded annulus_width: {analysis.reduction_parameters.annulus_width}")
                logger.debug(f"Config companion_mask_radius: {trap_config.reduction.companion_mask_radius}")
                logger.debug(f"Loaded companion_mask_radius: {analysis.reduction_parameters.companion_mask_radius}")

            # Additional pre-detection diagnostics for debugging the TypeError
            logger.debug("Pre-detection diagnostics:")
            if hasattr(analysis, 'reduction_parameters') and analysis.reduction_parameters:
                logger.debug("  - Reduction parameters loaded: True")
                logger.debug(f"  - Result folder: {analysis.reduction_parameters.result_folder}")
            else:
                logger.warning("  - Reduction parameters: None or missing")
            
            if hasattr(analysis, 'instrument') and analysis.instrument:
                logger.debug("  - Instrument loaded: True")
            else:
                logger.warning("  - Instrument: None or missing")
            
            logger.debug(f"  - Data cube shape: {data_full.shape if data_full is not None else 'None'}")
            logger.debug(f"  - Flux PSF shape: {flux_psf_full.shape if flux_psf_full is not None else 'None'}")
            logger.debug(f"  - Wavelength indices length: {len(wavelength_indices) if wavelength_indices is not None else 'None'}")
            logger.debug(f"  - Species database exists: {Path(species_database_directory).exists()}")

            logger.debug("Starting template matching and characterization")
            analysis.detection_and_characterization_with_template_matching(
                reduction_parameters=deepcopy(analysis.reduction_parameters),
                instrument=analysis.instrument, 
                species_database_directory=species_database_directory,
                stellar_parameters=trap_config.get_stellar_parameters(),
                data_full=data_full,
                flux_psf_full=flux_psf_full,
                pa=pa,
                temporal_components_fraction=trap_config.processing.temporal_components_fraction[0],
                wavelength_indices=wavelength_indices,
                xy_image_centers=xy_image_centers, 
                inverse_variance_full=inverse_variance_full,
                bad_frames=bad_frames,
                bad_pixel_mask_full=bad_pixel_mask_full, 
                amplitude_modulation_full=amplitude_modulation_full, 
                detection_threshold=trap_config.detection.detection_threshold,
                candidate_threshold=trap_config.detection.candidate_threshold,
                use_spectral_correlation=trap_config.detection.use_spectral_correlation,
                search_radius=trap_config.detection.search_radius,
                inner_mask_radius=trap_config.detection.inner_mask_radius,
                good_fraction_threshold=trap_config.detection.good_fraction_threshold,
                theta_deviation_threshold=trap_config.detection.theta_deviation_threshold,
                yx_fwhm_ratio_threshold=trap_config.detection.yx_fwhm_ratio_threshold
            )
            
            logger.info("TRAP detection completed", extra={"step": "trap_detection", "status": "success"})

        # Session completion logging
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60.0
        logger.info(f"TRAP processing completed in {elapsed_minutes:.2f} minutes", 
                    extra={"step": "trap_session", "status": "success"})

    except Exception as e:
        logger.exception("TRAP processing failed", extra={"step": "trap_session", "status": "failed"})
        # Create crash report in TRAP folder
        crash_report_path = Path(result_folder) / 'trap_crash_report.txt'
        with open(crash_report_path, 'w', encoding='utf-8') as f:
            f.write(f"TRAP processing error for {name_mode_date}\n\n")
            f.write(f"Error: {str(e)}\n\n")
            traceback.print_exc(file=f)
        logger.info(f"TRAP crash report saved to {crash_report_path}")
        
        # Don't re-raise the exception to allow processing to continue with next observation
        logger.info("Continuing with next observation despite TRAP processing failure")

    finally:
        remove_queue_listener()


def run_trap_on_observations(
    observations: Union[IFSObservation, IRDISObservation, list[Union[IFSObservation, IRDISObservation]]],
    trap_config,
    reduction_config: IFSReductionConfig,
    species_database_directory: Union[str, Path],
) -> None:
    """
    Run TRAP post-processing on multiple SPHERE IFS observations.
    
    This function provides a convenient wrapper to process multiple SPHERE
    observations sequentially using the same TRAP configuration. It handles both
    single observations and lists of observations, applying TRAP post-processing
    including reduction and detection phases for high-contrast imaging and
    exoplanet detection.
    
    Parameters
    ----------
    observations : IFSObservation, IRDISObservation, or list thereof
        Single observation object or list of observation objects to process
        with TRAP. Each observation must have already been processed through
        the IFS reduction pipeline.
    trap_config : TrapConfig
        TRAP configuration object containing all parameters for the TRAP
        algorithm including temporal components fraction, detection thresholds,
        and stellar parameters.
    reduction_config : IFSReductionConfig
        IFS reduction configuration object containing directory paths and
        extraction method settings needed to locate the reduced data.
    species_database_directory : str or Path
        Path to the species database directory used for stellar template
        matching during detection and characterization.
        
    Returns
    -------
    None
        Function writes TRAP outputs to disk but returns None.
        
    Notes
    -----
    This function automatically handles both single observation and list
    of observations. Each observation is processed independently, allowing
    partial completion if errors occur with individual targets.
    
    The TRAP reduction and detection steps are controlled by the settings
    in `reduction_config.steps.run_trap_reduction` and 
    `reduction_config.steps.run_trap_detection` respectively.
    
    The function expects that the IFS reduction pipeline has already been
    run for all observations and that the required data products exist
    in the expected directory structure.
    
    Examples
    --------
    Process multiple observations with TRAP:
    
    >>> from spherical.pipeline.run_trap import run_trap_on_observations
    >>> run_trap_on_observations(
    ...     observations=observation_list,
    ...     trap_config=trap_config,
    ...     reduction_config=config,
    ...     species_database_directory=species_db_path
    ... )
    
    Process single observation with custom steps configuration:
    
    >>> # Configure which TRAP steps to run
    >>> config.steps.run_trap_reduction = True
    >>> config.steps.run_trap_detection = False
    >>> run_trap_on_observations(
    ...     observations=single_observation,
    ...     trap_config=trap_config,
    ...     reduction_config=config,
    ...     species_database_directory=species_db_path
    ... )
    """
    # Handle both single observation and list of observations
    if not isinstance(observations, list):
        observations = [observations]
    
    for observation in observations:
        run_trap_on_observation(
            observation=observation,
            trap_config=trap_config,
            reduction_config=reduction_config,
            species_database_directory=species_database_directory,
        )
