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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits

from spherical.pipeline import flux_calibration


def run_spot_to_flux_normalization(converted_dir, reduction_parameters):
    plot_dir = os.path.join(converted_dir, 'flux_plots/')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    wavelengths = fits.getdata(os.path.join(converted_dir, 'wavelengths.fits')) * u.nm
    wavelengths = wavelengths.to(u.micron)
    flux_amplitude = fits.getdata(os.path.join(converted_dir, 'flux_amplitude_calibrated.fits'))[2]
    spot_amplitude = fits.getdata(os.path.join(converted_dir, 'spot_amplitudes.fits'))
    master_spot_amplitude = np.mean(spot_amplitude, axis=2)
    frames_info = {}
    frames_info['FLUX'] = pd.read_csv(os.path.join(converted_dir, 'frames_info_flux.csv'))
    frames_info['CENTER'] = pd.read_csv(os.path.join(converted_dir, 'frames_info_center.csv'))
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
    psf_flux.plot_flux(plot_original=False, autocolor=True, cmap=plt.cm.cool,
                       savefig=True, savedir=plot_dir, filename='psf_flux.png')
    spot_flux.plot_flux(plot_original=False, autocolor=True, cmap=plt.cm.cool,
                        savefig=True, savedir=plot_dir, filename='spot_flux_rescaled.png')
    flux_calibration_indices = pd.read_csv(os.path.join(converted_dir, 'flux_calibration_indices.csv'))
    normalization_factors, averaged_normalization, std_dev_normalization = flux_calibration.compute_flux_normalization_factors(
        flux_calibration_indices, psf_flux, spot_flux)
    flux_calibration.plot_flux_normalization_factors(
        flux_calibration_indices, normalization_factors[:, 1:-1],
        wavelengths=wavelengths[1:-1], cmap=plt.cm.cool,
        savefig=True, savedir=plot_dir)
    fits.writeto(os.path.join(converted_dir, 'spot_normalization_factors.fits'), normalization_factors, overwrite=True)
    fits.writeto(os.path.join(converted_dir, 'spot_normalization_factors_average.fits'), averaged_normalization, overwrite=True)
    fits.writeto(os.path.join(converted_dir, 'spot_normalization_factors_stddev.fits'), std_dev_normalization, overwrite=True)
    flux_calibration.plot_timeseries(
        frames_info['FLUX'], frames_info['CENTER'], psf_flux, spot_flux, averaged_normalization,
        x_axis_quantity='HOUR ANGLE', wavelength_channels=np.arange(len(wavelengths))[1:-1],
        savefig=True, savedir=plot_dir)
    scaled_spot_flux = spot_flux.flux * averaged_normalization[:, None]
    temporal_mean = np.nanmean(scaled_spot_flux, axis=1)
    amplitude_variation = scaled_spot_flux / temporal_mean[:, None]
    fits.writeto(os.path.join(converted_dir, 'spot_amplitude_variation.fits'), amplitude_variation, overwrite=True)
