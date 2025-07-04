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

import os
from copy import copy
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from trap.detection import DetectionAnalysis
from trap.parameters import Instrument
from trap.reduction_wrapper import run_complete_reduction

from spherical.database.ifs_observation import IFSObservation
from spherical.database.irdis_observation import IRDISObservation
from spherical.pipeline import ifs_reduction
from spherical.pipeline.pipeline_config import IFSReductionConfig
from spherical.pipeline.toolbox import make_target_folder_string


def run_trap_on_observation(
    observation: Union[IFSObservation, IRDISObservation],
    trap_config,
    reduction_config: IFSReductionConfig,
    species_database_directory: Union[str, Path],
    run_trap_reduction: bool = True,
    run_trap_detection: bool = True,
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
    run_trap_reduction : bool, optional
        Whether to run the TRAP reduction phase. Default is True.
    run_trap_detection : bool, optional
        Whether to run the TRAP detection and characterization phase.
        Default is True.
        
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
    - flux_stamps_calibrated_bg_corrected.fits
    - image_centers_fitted_robust.fits
    
    The function automatically determines the file identifier based on
    the WAFFLE_MODE: "center" for continuous satellite spots, "coro" otherwise.
    """
    if not run_trap_reduction and not run_trap_detection:
        print("No TRAP processing selected to run. Skipping observation.")
        return
        
    obs_mode = observation.observation['FILTER'][0]
    assert obs_mode in ['OBS_YJ', 'OBS_H'], "Observation has to be done with IFS."

    if obs_mode == 'OBS_YJ':
        spectral_resolution = 55
    elif obs_mode == 'OBS_H':
        spectral_resolution = 35

    continuous_satellite_spots = observation.observation['WAFFLE_MODE'][0]

    used_instrument = Instrument(
        name="IFS",
        pixel_scale=u.pixel_scale(0.00746 * u.arcsec / u.pixel),
        telescope_diameter=7.99 * u.m,
        detector_gain=1.0,
        readnoise=0.0,
        instrument_type="ifu",
        wavelengths=None,
        spectral_resolution=spectral_resolution,
        filters=None,
        transmission=None,
    )

    data_directory = ifs_reduction.output_directory_path(
        str(reduction_config.directories.reduction_directory),
        observation,
        method=reduction_config.extraction.method)
    
    name_mode_date = make_target_folder_string(observation)
    result_folder = os.path.join(str(reduction_config.directories.reduction_directory), 'IFS/trap', name_mode_date)
    
    # Configure TRAP result folder
    trap_config.reduction.result_folder = result_folder
    
    # Get the configured TRAP parameters
    trap_parameters = trap_config.get_reduction_parameters()

    if continuous_satellite_spots:
        file_identifier = "center"
    else:
        file_identifier = "coro"

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
    flux_psf_full = fits.getdata(
        os.path.join(data_directory, "flux_stamps_calibrated_bg_corrected.fits")
    )
    flux_psf_full = np.mean(flux_psf_full, axis=1)
    # flux_psf_full = flux_psf_full[:, 0]

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

    if run_trap_reduction:
        _ = run_complete_reduction(
            data_full=data_full,
            flux_psf_full=flux_psf_full,
            pa=pa,
            instrument=used_instrument,
            reduction_parameters=copy(trap_parameters),
            temporal_components_fraction=temporal_components_fraction,
            wavelength_indices=wavelength_indices,
            inverse_variance_full=inverse_variance_full,
            bad_frames=bad_frames,
            amplitude_modulation_full=amplitude_modulation_full,
            xy_image_centers=xy_image_centers,
            overwrite=trap_config.processing.overwrite_reduction,
            verbose=trap_config.processing.verbose,
        )

    if run_trap_detection:
        analysis = DetectionAnalysis(
            reduction_parameters=trap_parameters, instrument=used_instrument
        )
        analysis.read_output(
            trap_config.processing.temporal_components_fraction[0],
            result_folder=trap_parameters.result_folder,
            reduction_type="temporal",
            correlated_residuals=False,
            read_parameters=True,
        )

        analysis.reduction_parameters.result_folder = result_folder

        analysis.detection_and_characterization_with_template_matching(
            reduction_parameters=copy(trap_parameters),
            instrument=used_instrument, 
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
            yx_fwhm_ratio_threshold=trap_config.detection.yx_fwhm_ratio
        )


def run_trap_on_observations(
    observations: Union[IFSObservation, IRDISObservation, list[Union[IFSObservation, IRDISObservation]]],
    trap_config,
    reduction_config: IFSReductionConfig,
    species_database_directory: Union[str, Path],
    run_trap_reduction: bool = True,
    run_trap_detection: bool = True,
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
    run_trap_reduction : bool, optional
        Whether to run the TRAP reduction phase for all observations.
        Default is True.
    run_trap_detection : bool, optional
        Whether to run the TRAP detection and characterization phase
        for all observations. Default is True.
        
    Returns
    -------
    None
        Function writes TRAP outputs to disk but returns None.
        
    Notes
    -----
    This function automatically handles both single observation and list
    of observations. Each observation is processed independently, allowing
    partial completion if errors occur with individual targets.
    
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
    
    Process single observation:
    
    >>> run_trap_on_observations(
    ...     observations=single_observation,
    ...     trap_config=trap_config,
    ...     reduction_config=config,
    ...     species_database_directory=species_db_path,
    ...     run_trap_reduction=True,
    ...     run_trap_detection=False
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
            run_trap_reduction=run_trap_reduction,
            run_trap_detection=run_trap_detection,
        )
