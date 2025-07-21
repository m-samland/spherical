"""
Flux PSF Calibration Step

Parameters
----------
converted_dir : str
    Directory where the output files are stored and written.
overwrite_preprocessing : bool
    Whether to overwrite existing files.
reduction_parameters : dict
    Reduction parameters dict, must contain 'flux_combination_method', 'exclude_first_flux_frame', and 'exclude_first_flux_frame_all'.
"""
import os
from pathlib import Path
from typing import Dict

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from spherical.pipeline import flux_calibration, toolbox, transmission
from spherical.pipeline.logging_utils import optional_logger
from spherical.pipeline.steps.find_star import guess_position_psf, star_centers_from_PSF_img_cube


@optional_logger
def run_flux_psf_calibration(
    converted_dir: str,
    overwrite_preprocessing: bool,
    reduction_parameters: Dict[str, str | bool],
    logger
) -> None:
    """
    Calibrate flux PSF measurements in SPHERE/IFS data.

    This is the tenth step in the SPHERE/IFS data reduction pipeline. It performs
    flux calibration of PSF measurements, including DIT normalization, ND filter
    correction, and frame combination. This step is crucial for absolute flux
    calibration and photometric accuracy.

    Required Input Files
    -------------------
    From previous steps:
    - converted_dir/wavelengths.fits
        Wavelength array for the data cube
    - converted_dir/flux_cube.fits
        Master cube of flux data
    - converted_dir/frames_info_center.csv
        Frame information for center data
    - converted_dir/frames_info_flux.csv
        Frame information for flux data

    Generated Output Files
    ---------------------
    In converted_dir:
    - flux_amplitude_calibrated.fits
        Calibrated flux amplitudes
    - flux_calibration_indices.csv
        Frame indices for flux calibration
    - psf_cube_for_postprocessing.fits
        Combined calibrated flux PSF frames

    In converted_dir/additional_outputs/:
    - flux_centers.fits
        Fitted centers of flux PSFs
    - flux_gauss_amplitudes.fits
        Gaussian fit amplitudes for flux PSFs
    - flux_stamps_uncalibrated.fits
        Raw extracted flux PSF stamps
    - nd_attenuation.fits
        ND filter transmission correction
    - center_frame_dit_adjustment_factors.fits
        DIT normalization factors for center frames
    - flux_stamps_dit_nd_calibrated.fits
        DIT and ND-corrected flux stamps
    - flux_photometry.obj
        Pickled photometry results
    - flux_snr.fits
        Signal-to-noise ratios
    - flux_stamps_calibrated_bg_corrected.fits
        Background-subtracted calibrated stamps
    - indices_of_discontinuity.csv
        Indices where flux calibration changes
    - Flux_PSF_aperture_SNR.png
        Plot of SNR vs aperture size

    Parameters
    ----------
    converted_dir : str
        Directory containing the input files and where outputs will be written.
    overwrite_preprocessing : bool
        Whether to overwrite existing output files.
    reduction_parameters : dict
        Reduction parameters dict, must contain:
        - flux_combination_method: str
            Method to combine flux frames ('mean' or 'median')
        - exclude_first_flux_frame: bool
            Whether to exclude first frame in first sequence
        - exclude_first_flux_frame_all: bool
            Whether to exclude first frame in all sequences
    logger : logging.Logger
        Logger instance injected by @optional_logger for structured logging.

    Returns
    -------
    None
        This function writes calibrated flux data to disk and does not return
        a value.

    Notes
    -----
    - Extracts 57x57 pixel stamps centered on each flux PSF
    - Performs multiple calibration steps:
        * DIT normalization using most common DIT from center frames
        * ND filter transmission correction
        * Background subtraction using annulus photometry
    - Uses aperture photometry with:
        * Aperture radius range: 1-15 pixels
        * Background annulus: 15-18 pixels
    - Handles frame combination with:
        * Configurable combination method (mean/median)
        * Optional first frame exclusion
        * Frame sequence detection
    - Creates visualization of SNR vs aperture size
    - All output arrays are saved as float32 for efficiency
    
    Master Flux Calibrated PSF Frames Generation
    -------------------------------------------
    The psf_cube_for_postprocessing.fits file is generated through a sophisticated 
    flux calibration and frame combination process:
    
    1. **Sequence Detection**: Uses flux_calibration.get_flux_calibration_indices() to 
       identify temporal segments where observing conditions are stable (e.g., same ND 
       filter, continuous observing).
    
    2. **Frame Normalization**: Within each sequence, frames are normalized using 
       3-pixel aperture photometry results. Each frame is divided by the sequence 
       mean to correct for temporal variations in flux.
    
    3. **Frame Combination**: Frames within each sequence are combined using the 
       specified method (mean or median). Optional first-frame exclusion handles 
       potential settling effects after instrument changes.
    
    4. **Final Assembly**: Results from all sequences are assembled into a single 
       array with dimensions (wavelengths, sequences, 57, 57).
    
    Output Dimensions
    ----------------
    psf_cube_for_postprocessing.fits: (wavelengths, sequences, y_pixels, x_pixels)
    - wavelengths: Number of IFS wavelength channels
    - sequences: Number of flux calibration sequences (temporal segments)
    - y_pixels: 57 (PSF stamp height)
    - x_pixels: 57 (PSF stamp width)
    
    Each sequence represents a temporally stable observing period with improved 
    SNR through frame combination and normalized flux calibration.

    Examples
    --------
    >>> run_flux_psf_calibration(
    ...     converted_dir="/path/to/converted",
    ...     overwrite_preprocessing=True,
    ...     reduction_parameters={
    ...         "flux_combination_method": "mean",
    ...         "exclude_first_flux_frame": True,
    ...         "exclude_first_flux_frame_all": False,
    ...         "logger": logger
    ...     }
    ... )
    """
    logger.info("Starting flux PSF calibration", extra={"step": "flux_psf_calibration", "status": "started"})
    
    # Create directories for outputs
    plot_dir = os.path.join(converted_dir, 'flux_plots/')
    additional_outputs_dir = Path(converted_dir) / 'additional_outputs'
    additional_outputs_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        logger.debug(f"Created plot directory: {plot_dir}")
    logger.debug(f"Created additional outputs directory: {additional_outputs_dir}")
    # Load required files and log their shapes
    wavelengths_path = os.path.join(converted_dir, 'wavelengths.fits')
    flux_cube_path = os.path.join(converted_dir, 'flux_cube.fits')
    frames_info_center_path = os.path.join(converted_dir, 'frames_info_center.csv')
    frames_info_flux_path = os.path.join(converted_dir, 'frames_info_flux.csv')
    for fpath in [wavelengths_path, flux_cube_path, frames_info_center_path, frames_info_flux_path]:
        if not os.path.exists(fpath):
            logger.warning(f"Missing required file: {fpath}", extra={"step": "flux_psf_calibration", "status": "failed"})
    wavelengths = fits.getdata(wavelengths_path)
    flux_cube = fits.getdata(flux_cube_path).astype('float64')
    logger.debug(f"Loaded wavelengths shape: {wavelengths.shape}, flux_cube shape: {flux_cube.shape}")
    frames_info = {}
    frames_info['CENTER'] = pd.read_csv(frames_info_center_path)
    frames_info['FLUX'] = pd.read_csv(frames_info_flux_path)
    logger.debug(f"Loaded frames_info: CENTER shape {frames_info['CENTER'].shape}, FLUX shape {frames_info['FLUX'].shape}")
    
    # First, compute guess positions for all frames
    logger.debug("Computing initial guess positions for all flux frames")
    guess_positions_yx = []
    for frame_number in range(flux_cube.shape[1]):
        data = flux_cube[:, frame_number]
        cy, cx = guess_position_psf(
            cube=data,
            exclude_edge_pixels=30,
            mask_coronagraph_center=True,
            coronagraph_mask_x=126,
            coronagraph_mask_y=131,
            coronagraph_mask_radius=30
        )
        guess_positions_yx.append((cy, cx))
    
    # Replace unreliable first frame guess with second frame guess
    if len(guess_positions_yx) >= 2:
        logger.debug(f"Replacing unreliable first frame guess position {guess_positions_yx[0]} with second frame position {guess_positions_yx[1]}")
        guess_positions_yx[0] = guess_positions_yx[1]
    else:
        logger.warning("Less than 2 flux frames available, cannot replace first frame guess position")
    
    logger.debug(f"Computed guess positions for {len(guess_positions_yx)} frames")
    
    # Now compute flux centers using pre-computed guess positions
    flux_centers = []
    flux_amplitudes = []
    for frame_number in range(flux_cube.shape[1]):
        data = flux_cube[:, frame_number]
        flux_center, flux_amplitude = star_centers_from_PSF_img_cube(
            cube=data,
            wave=wavelengths,
            pixel=7.46,
            guess_center_yx=guess_positions_yx[frame_number],  # Use pre-computed guess
            fit_background=True,
            fit_symmetric_gaussian=True,
            mask_deviating=False,
            deviation_threshold=0.8,
            exclude_edge_pixels=30,
            mask_coronagraph_center=True,
            coronagraph_mask_x=126,
            coronagraph_mask_y=131,
            coronagraph_mask_radius=30,
            mask=None,
            save_path=None,
            verbose=False,
            logger=logger,
            frame_number=frame_number,
        )
        flux_centers.append(flux_center)
        flux_amplitudes.append(flux_amplitude)

    # Replace unreliable first frame centers with second frame centers
    if len(flux_centers) >= 2:
        logger.debug("Replacing unreliable first flux frame centers with second frame centers")
        # flux_centers[0] contains centers for all wavelengths in first frame
        # flux_centers[1] contains centers for all wavelengths in second frame
        flux_centers[0] = flux_centers[1].copy()
        logger.debug("Successfully replaced first frame centers")
    else:
        logger.warning("Less than 2 flux frames available, cannot replace first frame centers")

    # Continue with existing array transformations
    flux_centers = np.expand_dims(np.swapaxes(np.array(flux_centers), 0, 1), axis=2)
    flux_amplitudes = np.swapaxes(np.array(flux_amplitudes), 0, 1)
    logger.debug(f"Extracted flux_centers shape: {flux_centers.shape}, flux_amplitudes shape: {flux_amplitudes.shape}")
    fits.writeto(additional_outputs_dir / 'flux_centers.fits', flux_centers, overwrite=overwrite_preprocessing)
    fits.writeto(additional_outputs_dir / 'flux_gauss_amplitudes.fits', flux_amplitudes, overwrite=overwrite_preprocessing)
    flux_stamps = toolbox.extract_satellite_spot_stamps(
        flux_cube, flux_centers, stamp_size=57, shift_order=3, plot=False)
    logger.debug(f"Extracted flux_stamps shape: {flux_stamps.shape}")
    fits.writeto(additional_outputs_dir / 'flux_stamps_uncalibrated.fits',
                 flux_stamps.astype('float32'), overwrite=overwrite_preprocessing)
    if len(frames_info['FLUX']['INS4 FILT2 NAME'].unique()) > 1:
        logger.warning('Non-unique ND filters in sequence.', extra={"step": "flux_psf_calibration", "status": "failed"})
        raise ValueError('Non-unique ND filters in sequence.')
    else:
        ND = frames_info['FLUX']['INS4 FILT2 NAME'].unique()[0]
    _, attenuation = transmission.transmission_nd(ND, wave=wavelengths)
    fits.writeto(additional_outputs_dir / 'nd_attenuation.fits', attenuation, overwrite=overwrite_preprocessing)
    dits_flux = np.array(frames_info['FLUX']['DET SEQ1 DIT'])
    dits_center = np.array(frames_info['CENTER']['DET SEQ1 DIT'])
    unique_dits_center, unique_dits_center_counts = np.unique(dits_center, return_counts=True)
    if len(unique_dits_center) == 1:
        dits_factor = unique_dits_center[0] / dits_flux
        most_common_dit_center = unique_dits_center[0]
    else:
        most_common_dit_center = unique_dits_center[np.argmax(unique_dits_center_counts)]
        dits_factor = most_common_dit_center / dits_flux
    dit_factor_center = most_common_dit_center / dits_center
    fits.writeto(additional_outputs_dir / 'center_frame_dit_adjustment_factors.fits',
                 dit_factor_center, overwrite=overwrite_preprocessing)
    flux_stamps_calibrated = flux_stamps * dits_factor[None, :, None, None]
    flux_stamps_calibrated = flux_stamps_calibrated / attenuation[:, np.newaxis, np.newaxis, np.newaxis]
    fits.writeto(additional_outputs_dir / 'flux_stamps_dit_nd_calibrated.fits',
                 flux_stamps_calibrated, overwrite=overwrite_preprocessing)
    flux_photometry = flux_calibration.get_aperture_photometry(
        flux_stamps_calibrated, aperture_radius_range=[1, 15],
        bg_aperture_inner_radius=15, bg_aperture_outer_radius=18)
    filehandler = open(additional_outputs_dir / 'flux_photometry.obj', 'wb')
    pickle.dump(flux_photometry, filehandler)
    filehandler.close()
    fits.writeto(os.path.join(converted_dir, 'flux_amplitude_calibrated.fits'),
                 flux_photometry['psf_flux_bg_corr_all'], overwrite=overwrite_preprocessing)
    fits.writeto(additional_outputs_dir / 'flux_snr.fits',
                 flux_photometry['snr_all'], overwrite=overwrite_preprocessing)
    
    plt.close()
    plt.plot(flux_photometry['aperture_sizes'], flux_photometry['snr_all'][:, :, 0])
    plt.xlabel('Aperture Size (pix)')
    plt.ylabel('BG limited SNR')
    plt.savefig(additional_outputs_dir / 'Flux_PSF_aperture_SNR.png')
    plt.close()
    bg_sub_flux_stamps_calibrated = flux_stamps_calibrated - flux_photometry['psf_bg_counts_all'][:, :, None, None]
    fits.writeto(additional_outputs_dir / 'flux_stamps_calibrated_bg_corrected.fits',
                 bg_sub_flux_stamps_calibrated.astype('float32'), overwrite=overwrite_preprocessing)
    flux_calibration_indices, indices_of_discontinuity = flux_calibration.get_flux_calibration_indices(
        frames_info['CENTER'], frames_info['FLUX'])
    flux_calibration_indices.to_csv(os.path.join(converted_dir, 'flux_calibration_indices.csv'))
    indices_of_discontinuity.tofile(additional_outputs_dir / 'indices_of_discontinuity.csv', sep=',')
    number_of_flux_frames = flux_stamps.shape[1]
    flux_calibration_frames = []
    if reduction_parameters['flux_combination_method'] == 'mean':
        comb_func = np.nanmean
    elif reduction_parameters['flux_combination_method'] == 'median':
        comb_func = np.nanmedian
    else:
        logger.warning('Unknown flux combination method.', extra={"step": "flux_psf_calibration", "status": "failed"})
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
        reference_value = np.nanmean(
            flux_photometry['psf_flux_bg_corr_all'][2][:, lower_index_frame_combine:upper_range], axis=1)
        normalization_values = phot_values / reference_value[:, None]
        flux_calibration_frame = bg_sub_flux_stamps_calibrated[:, lower_index:upper_range] / normalization_values[:, :, None, None]
        flux_calibration_frame = comb_func(flux_calibration_frame[:, lower_index_frame_combine:], axis=1)
        flux_calibration_frames.append(flux_calibration_frame)
    flux_calibration_frames = np.array(flux_calibration_frames)
    flux_calibration_frames = np.swapaxes(flux_calibration_frames, 0, 1)
    fits.writeto(os.path.join(converted_dir, 'psf_cube_for_postprocessing.fits'),
                 flux_calibration_frames.astype('float32'), overwrite=overwrite_preprocessing)
    logger.info("Step finished", extra={"step": "flux_psf_calibration", "status": "success"})
