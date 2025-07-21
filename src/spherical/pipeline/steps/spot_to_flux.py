"""
Spot-to-Flux Normalization Step

Parameters
----------
converted_dir : str
    Directory where the output files are stored and written.
reduction_parameters : dict
    Reduction parameters dict (not used in this step, but included for API consistency).
"""
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits

from spherical.pipeline import flux_calibration
from spherical.pipeline.logging_utils import optional_logger


@optional_logger
def run_spot_to_flux_normalization(
    converted_dir: str,
    reduction_parameters: Dict[str, str | bool],
    logger
) -> None:
    """
    Normalize satellite spot fluxes to absolute flux scale.

    This is the eleventh step in the SPHERE/IFS data reduction pipeline. It
    computes normalization factors to convert satellite spot fluxes to absolute
    flux units by comparing them with calibrated PSF flux measurements. This
    step is essential for absolute photometry and flux calibration.

    Required Input Files
    -------------------
    From previous steps:
    - converted_dir/wavelengths.fits
        Wavelength array for the data cube
    - converted_dir/flux_amplitude_calibrated.fits
        Calibrated flux amplitudes
    - converted_dir/additional_outputs/spot_amplitudes.fits
        Satellite spot amplitudes
    - converted_dir/frames_info_flux.csv
        Frame information for flux data
    - converted_dir/frames_info_center.csv
        Frame information for center data
    - converted_dir/flux_calibration_indices.csv
        Frame indices for flux calibration

    Generated Output Files
    ---------------------
    In converted_dir:
    - spot_amplitude_variation.fits
        Temporal variation of normalized spot amplitudes
    
    In converted_dir/additional_outputs/:
    - spot_normalization_factors.fits
        Wavelength-dependent normalization factors
    - spot_normalization_factors_average.fits
        Averaged normalization factors
    - spot_normalization_factors_stddev.fits
        Standard deviation of normalization factors

    In converted_dir/flux_plots/:
    - psf_flux.png
        Plot of PSF flux spectrum
    - spot_flux_rescaled.png
        Plot of rescaled spot flux spectrum
    - flux_normalization_factors.png
        Plot of wavelength-dependent normalization factors
    - flux_timeseries.png
        Plot of flux variation with hour angle

    Parameters
    ----------
    converted_dir : str
        Directory containing the input files and where outputs will be written.
    reduction_parameters : dict
        Reduction parameters dict (not used in this step, but included for API
        consistency).
    logger : logging.Logger
        Logger instance injected by @optional_logger for structured logging.

    Returns
    -------
    None
        This function writes normalization factors and plots to disk and does not
        return a value.

    Notes
    -----
    - Converts wavelengths to microns for internal calculations
    - Normalizes spectra in wavelength range 1.0-1.3 microns
    - Computes normalization factors by comparing:
        * Calibrated PSF flux measurements
        * Mean satellite spot amplitudes
    - Handles temporal variations by:
        * Computing per-frame normalization factors
        * Averaging across frames
        * Calculating standard deviations
    - Creates multiple visualizations:
        * Flux spectra for PSF and spots
        * Wavelength-dependent normalization factors
        * Temporal variation with hour angle
    - All output arrays are saved as float32 for efficiency

    Examples
    --------
    >>> run_spot_to_flux_normalization(
    ...     converted_dir="/path/to/converted",
    ...     reduction_parameters={},
    ...     logger=logger
    ... )
    """
    logger.info("Starting spot-to-flux normalization", extra={"step": "spot_to_flux_normalization", "status": "started"})
    plot_dir = os.path.join(converted_dir, 'flux_plots/')
    additional_outputs_dir = Path(converted_dir) / 'additional_outputs'
    additional_outputs_dir.mkdir(exist_ok=True)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        logger.debug(f"Created plot directory: {plot_dir}")
    # Load required files and log their shapes
    wavelengths_path = os.path.join(converted_dir, 'wavelengths.fits')
    flux_amplitude_path = os.path.join(converted_dir, 'flux_amplitude_calibrated.fits')
    additional_outputs_dir = Path(converted_dir) / 'additional_outputs'
    spot_amplitude_path = additional_outputs_dir / 'spot_amplitudes.fits'
    frames_info_flux_path = os.path.join(converted_dir, 'frames_info_flux.csv')
    frames_info_center_path = os.path.join(converted_dir, 'frames_info_center.csv')
    flux_calibration_indices_path = os.path.join(converted_dir, 'flux_calibration_indices.csv')
    for fpath in [wavelengths_path, flux_amplitude_path, spot_amplitude_path, frames_info_flux_path, frames_info_center_path, flux_calibration_indices_path]:
        if not os.path.exists(fpath):
            logger.warning(f"Missing required file: {fpath}", extra={"step": "spot_to_flux_normalization", "status": "failed"})
    wavelengths = fits.getdata(wavelengths_path) * u.nm
    wavelengths = wavelengths.to(u.micron)
    flux_amplitude = fits.getdata(flux_amplitude_path)[2]
    spot_amplitude = fits.getdata(spot_amplitude_path)
    logger.debug(f"Loaded wavelengths shape: {wavelengths.shape}, flux_amplitude shape: {flux_amplitude.shape}, spot_amplitude shape: {spot_amplitude.shape}")
    master_spot_amplitude = np.mean(spot_amplitude, axis=2)
    frames_info = {}
    frames_info['FLUX'] = pd.read_csv(frames_info_flux_path)
    frames_info['CENTER'] = pd.read_csv(frames_info_center_path)
    logger.debug(f"Loaded frames_info: FLUX shape {frames_info['FLUX'].shape}, CENTER shape {frames_info['CENTER'].shape}")
    psf_flux = flux_calibration.SimpleSpectrum(
        wavelength=wavelengths,
        flux=flux_amplitude,
        norm_wavelength_range=[1.0, 1.3] * u.micron,
        metadata=frames_info['FLUX'],
        rescale=False,
        normalize=False)
    spot_flux = flux_calibration.SimpleSpectrum(
        wavelength=wavelengths,
        flux=master_spot_amplitude,
        norm_wavelength_range=[1.0, 1.3] * u.micron,
        metadata=frames_info['CENTER'],
        rescale=True,
        normalize=False)
    logger.debug("Constructed SimpleSpectrum objects for PSF and spot flux.")
    psf_flux.plot_flux(plot_original=False, autocolor=True, cmap=plt.cm.cool,
                       savefig=True, savedir=plot_dir, filename='psf_flux.png')
    spot_flux.plot_flux(plot_original=False, autocolor=True, cmap=plt.cm.cool,
                        savefig=True, savedir=plot_dir, filename='spot_flux_rescaled.png')
    flux_calibration_indices = pd.read_csv(flux_calibration_indices_path)
    normalization_factors, averaged_normalization, std_dev_normalization = flux_calibration.compute_flux_normalization_factors(
        flux_calibration_indices, psf_flux, spot_flux)
    logger.debug(f"Computed normalization factors: shape {normalization_factors.shape}")
    flux_calibration.plot_flux_normalization_factors(
        flux_calibration_indices, normalization_factors[:, 1:-1],
        wavelengths=wavelengths[1:-1], cmap=plt.cm.cool,
        savefig=True, savedir=plot_dir)
    fits.writeto(additional_outputs_dir / 'spot_normalization_factors.fits', normalization_factors, overwrite=True)
    fits.writeto(additional_outputs_dir / 'spot_normalization_factors_average.fits', averaged_normalization, overwrite=True)
    fits.writeto(additional_outputs_dir / 'spot_normalization_factors_stddev.fits', std_dev_normalization, overwrite=True)
    flux_calibration.plot_timeseries(
        frames_info['FLUX'], frames_info['CENTER'], psf_flux, spot_flux, averaged_normalization,
        x_axis_quantity='HOUR ANGLE', wavelength_channels=np.arange(len(wavelengths))[1:-1],
        savefig=True, savedir=plot_dir)
    scaled_spot_flux = spot_flux.flux * averaged_normalization[:, None]
    temporal_mean = np.nanmean(scaled_spot_flux, axis=1)
    amplitude_variation = scaled_spot_flux / temporal_mean[:, None]
    fits.writeto(os.path.join(converted_dir, 'spot_amplitude_variation.fits'), amplitude_variation, overwrite=True)
    logger.info("Step finished", extra={"step": "spot_to_flux_normalization", "status": "success"})
