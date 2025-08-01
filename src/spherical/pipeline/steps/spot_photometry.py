"""
Spot Photometry Calibration Step

Parameters
----------
converted_dir : str
    Directory where the output files are stored and written.
overwrite_preprocessing : bool
    Whether to overwrite existing files.
"""
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from photutils.aperture import CircularAnnulus, CircularAperture

from spherical.pipeline import toolbox
from spherical.pipeline.logging_utils import optional_logger


@optional_logger
def run_spot_photometry_calibration(converted_dir: str, overwrite_preprocessing: bool, logger) -> None:
    """
    Calibrate photometry of satellite spots in SPHERE/IFS data.

    This is the ninth step in the SPHERE/IFS data reduction pipeline. It extracts
    satellite spot PSF stamps, performs background subtraction, and calculates
    spot fluxes and signal-to-noise ratios. This calibration is essential for
    flux normalization and absolute photometry.

    Required Input Files
    -------------------
    From previous steps:
    - converted_dir/additional_outputs/spot_centers.fits
        Positions of satellite spots
    - converted_dir/center_cube.fits
        Master cube of center data containing satellite spots

    Generated Output Files
    ---------------------
    In converted_dir/additional_outputs/:
    - satellite_psf_stamps.fits
        Extracted PSF stamps for each satellite spot
    - master_satellite_psf_stamps.fits
        Mean PSF stamps across all frames
    - satellite_psf_stamps_bg_corrected.fits
        Background-subtracted PSF stamps
    - spot_amplitudes.fits
        Integrated fluxes for each spot
    - spot_snr.fits
        Signal-to-noise ratios for each spot
    - master_satellite_psf_stamps_bg_corrected.fits
        Mean background-subtracted PSF stamps

    Parameters
    ----------
    converted_dir : str
        Directory containing the input files and where outputs will be written.
    overwrite_preprocessing : bool
        Whether to overwrite existing output files.
    logger : logging.Logger
        Logger instance injected by @optional_logger for structured logging.

    Returns
    -------
    None
        This function writes calibrated photometry data to disk and does not
        return a value.

    Notes
    -----
    - Extracts 57x57 pixel stamps centered on each satellite spot
    - Uses 3rd order shift interpolation for accurate centering
    - Performs background subtraction using:
        * Circular annulus (r_in=15, r_out=18 pixels)
        * 3-sigma clipping for robust background estimation
    - Calculates spot fluxes using:
        * Circular aperture (r=3 pixels)
        * Background-subtracted data
    - Computes SNR using:
        * Integrated flux in aperture
        * Background standard deviation
        * Aperture area
    - Creates master stamps by averaging across frames
    - All output arrays are saved as float32 for efficiency

    Examples
    --------
    >>> run_spot_photometry_calibration(
    ...     converted_dir="/path/to/converted",
    ...     overwrite_preprocessing=True,
    ...     logger=logger
    ... )
    """
    logger.info("Starting spot photometry calibration", extra={"step": "spot_photometry_calibration", "status": "started"})
    additional_outputs_dir = Path(converted_dir) / 'additional_outputs'
    spot_centers_path = additional_outputs_dir / 'spot_centers.fits'
    center_cube_path = os.path.join(converted_dir, 'center_cube.fits')
    if not os.path.exists(spot_centers_path):
        logger.warning(f"Missing spot_centers.fits at {spot_centers_path}", extra={"step": "spot_photometry_calibration", "status": "failed"})
    if not os.path.exists(center_cube_path):
        logger.warning(f"Missing center_cube.fits at {center_cube_path}", extra={"step": "spot_photometry_calibration", "status": "failed"})
    spot_centers = fits.getdata(spot_centers_path)
    center_cube = fits.getdata(center_cube_path)
    logger.debug(f"Loaded spot_centers shape: {spot_centers.shape}, center_cube shape: {center_cube.shape}")
    
    # Create additional outputs directory
    additional_outputs_dir.mkdir(exist_ok=True)
    
    satellite_psf_stamps = toolbox.extract_satellite_spot_stamps(center_cube, spot_centers, stamp_size=57, shift_order=3, plot=False)
    logger.debug(f"Extracted satellite_psf_stamps shape: {satellite_psf_stamps.shape}")
    master_satellite_psf_stamps = np.nanmean(np.nanmean(satellite_psf_stamps, axis=2), axis=1)
    fits.writeto(additional_outputs_dir / 'satellite_psf_stamps.fits', satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)
    fits.writeto(additional_outputs_dir / 'master_satellite_psf_stamps.fits', master_satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)
    stamp_size = [satellite_psf_stamps.shape[-1], satellite_psf_stamps.shape[-2]]
    stamp_center = [satellite_psf_stamps.shape[-1] // 2, satellite_psf_stamps.shape[-2] // 2]
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
        ma_spot_stamps,
        sigma=3, maxiters=5, cenfunc=np.nanmedian, stdfunc=np.nanstd,
        axis=3, masked=True, return_bounds=False)
    bg_counts = np.ma.median(sigma_clipped_array, axis=3).data
    bg_std = np.ma.std(sigma_clipped_array, axis=3).data
    bg_corr_satellite_psf_stamps = satellite_psf_stamps - bg_counts[:, :, :, None, None]
    fits.writeto(
        additional_outputs_dir / 'satellite_psf_stamps_bg_corrected.fits',
        bg_corr_satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)
    aperture = CircularAperture(stamp_center, 3)
    psf_mask = aperture.to_mask(method='center')
    psf_mask = psf_mask.to_image(stamp_size) > 0
    flux_sum = np.nansum(bg_corr_satellite_psf_stamps[:, :, :, psf_mask], axis=3)
    spot_snr = flux_sum / (bg_std * np.sum(psf_mask))
    master_satellite_psf_stamps_bg_corr = np.nanmean(
        np.nansum(bg_corr_satellite_psf_stamps, axis=2), axis=1)
    fits.writeto(
        additional_outputs_dir / 'spot_amplitudes.fits',
        flux_sum, overwrite=overwrite_preprocessing)
    fits.writeto(
        additional_outputs_dir / 'spot_snr.fits',
        spot_snr, overwrite=overwrite_preprocessing)
    fits.writeto(
        additional_outputs_dir / 'master_satellite_psf_stamps_bg_corrected.fits',
        master_satellite_psf_stamps_bg_corr.astype('float32'), overwrite=overwrite_preprocessing)
    logger.info("Step finished", extra={"step": "spot_photometry_calibration", "status": "success"})
