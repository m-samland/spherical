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

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from photutils.aperture import CircularAnnulus, CircularAperture
from spherical.pipeline import toolbox

def run_spot_photometry_calibration(converted_dir, overwrite_preprocessing):
    # Extract and write satellite PSF stamps if not already present
    spot_centers = fits.getdata(os.path.join(converted_dir, 'spot_centers.fits'))
    # TODO: Seems out of place, should be moved to a more appropriate step or function to extract satellite PSF stamps
    center_cube = fits.getdata(os.path.join(converted_dir, 'center_cube.fits'))
    satellite_psf_stamps = toolbox.extract_satellite_spot_stamps(center_cube, spot_centers, stamp_size=57, shift_order=3, plot=False)
    master_satellite_psf_stamps = np.nanmean(np.nanmean(satellite_psf_stamps, axis=2), axis=1)
    fits.writeto(os.path.join(converted_dir, 'satellite_psf_stamps.fits'), satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)
    fits.writeto(os.path.join(converted_dir, 'master_satellite_psf_stamps.fits'), master_satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)
    
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
        os.path.join(converted_dir, 'satellite_psf_stamps_bg_corrected.fits'),
        bg_corr_satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)
    aperture = CircularAperture(stamp_center, 3)
    psf_mask = aperture.to_mask(method='center')
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
        os.path.join(converted_dir, 'master_satellite_psf_stamps_bg_corrected.fits'),
        master_satellite_psf_stamps_bg_corr.astype('float32'), overwrite=overwrite_preprocessing)
