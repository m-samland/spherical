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

import dill as pickle
import numpy as np
import pandas as pd
from astropy.io import fits

from spherical.pipeline import flux_calibration, toolbox, transmission
from spherical.pipeline.find_star import star_centers_from_PSF_img_cube


def calibrate_flux_psf(converted_dir, overwrite_preprocessing, reduction_parameters):
    wavelengths = fits.getdata(os.path.join(converted_dir, 'wavelengths.fits'))
    flux_cube = fits.getdata(os.path.join(converted_dir, 'flux_cube.fits')).astype('float64')
    frames_info = {}
    frames_info['CENTER'] = pd.read_csv(os.path.join(converted_dir, 'frames_info_center.csv'))
    frames_info['FLUX'] = pd.read_csv(os.path.join(converted_dir, 'frames_info_flux.csv'))
    plot_dir = os.path.join(converted_dir, 'flux_plots/')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    flux_centers = []
    flux_amplitudes = []
    for frame_number in range(flux_cube.shape[1]):
        data = flux_cube[:, frame_number]
        flux_center, flux_amplitude = star_centers_from_PSF_img_cube(
            cube=data,
            wave=wavelengths,
            pixel=7.46,
            guess_center_yx=None,
            fit_background=False,
            fit_symmetric_gaussian=True,
            mask_deviating=False,
            deviation_threshold=0.8,
            mask=None,
            save_path=None,
            verbose=False)
        flux_centers.append(flux_center)
        flux_amplitudes.append(flux_amplitude)
    flux_centers = np.expand_dims(np.swapaxes(np.array(flux_centers), 0, 1), axis=2)
    flux_amplitudes = np.swapaxes(np.array(flux_amplitudes), 0, 1)
    fits.writeto(os.path.join(converted_dir, 'flux_centers.fits'), flux_centers, overwrite=overwrite_preprocessing)
    fits.writeto(os.path.join(converted_dir, 'flux_gauss_amplitudes.fits'), flux_amplitudes, overwrite=overwrite_preprocessing)
    flux_stamps = toolbox.extract_satellite_spot_stamps(
        flux_cube, flux_centers, stamp_size=57, shift_order=3, plot=False)
    fits.writeto(os.path.join(converted_dir, 'flux_stamps_uncalibrated.fits'),
                 flux_stamps.astype('float32'), overwrite=overwrite_preprocessing)
    if len(frames_info['FLUX']['INS4 FILT2 NAME'].unique()) > 1:
        raise ValueError('Non-unique ND filters in sequence.')
    else:
        ND = frames_info['FLUX']['INS4 FILT2 NAME'].unique()[0]
    _, attenuation = transmission.transmission_nd(ND, wave=wavelengths)
    fits.writeto(os.path.join(converted_dir, 'nd_attenuation.fits'), attenuation, overwrite=overwrite_preprocessing)
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
    fits.writeto(os.path.join(converted_dir, 'center_frame_dit_adjustment_factors.fits'),
                 dit_factor_center, overwrite=overwrite_preprocessing)
    flux_stamps_calibrated = flux_stamps * dits_factor[None, :, None, None]
    flux_stamps_calibrated = flux_stamps_calibrated / attenuation[:, np.newaxis, np.newaxis, np.newaxis]
    fits.writeto(os.path.join(converted_dir, 'flux_stamps_dit_nd_calibrated.fits'),
                 flux_stamps_calibrated, overwrite=overwrite_preprocessing)
    flux_photometry = flux_calibration.get_aperture_photometry(
        flux_stamps_calibrated, aperture_radius_range=[1, 15],
        bg_aperture_inner_radius=15, bg_aperture_outer_radius=18)
    filehandler = open(os.path.join(converted_dir, 'flux_photometry.obj'), 'wb')
    pickle.dump(flux_photometry, filehandler)
    filehandler.close()
    fits.writeto(os.path.join(converted_dir, 'flux_amplitude_calibrated.fits'),
                 flux_photometry['psf_flux_bg_corr_all'], overwrite=overwrite_preprocessing)
    fits.writeto(os.path.join(converted_dir, 'flux_snr.fits'),
                 flux_photometry['snr_all'], overwrite=overwrite_preprocessing)
    import matplotlib.pyplot as plt
    plt.close()
    plt.plot(flux_photometry['aperture_sizes'], flux_photometry['snr_all'][:, :, 0])
    plt.xlabel('Aperture Size (pix)')
    plt.ylabel('BG limited SNR')
    plt.savefig(os.path.join(plot_dir, 'Flux_PSF_aperture_SNR.png'))
    plt.close()
    bg_sub_flux_stamps_calibrated = flux_stamps_calibrated - flux_photometry['psf_bg_counts_all'][:, :, None, None]
    fits.writeto(os.path.join(converted_dir, 'flux_stamps_calibrated_bg_corrected.fits'),
                 bg_sub_flux_stamps_calibrated.astype('float32'), overwrite=overwrite_preprocessing)
    flux_calibration_indices, indices_of_discontinuity = flux_calibration.get_flux_calibration_indices(
        frames_info['CENTER'], frames_info['FLUX'])
    flux_calibration_indices.to_csv(os.path.join(converted_dir, 'flux_calibration_indices.csv'))
    indices_of_discontinuity.tofile(os.path.join(converted_dir, 'indices_of_discontinuity.csv'), sep=',')
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
        reference_value = np.nanmean(
            flux_photometry['psf_flux_bg_corr_all'][2][:, lower_index_frame_combine:upper_range], axis=1)
        normalization_values = phot_values / reference_value[:, None]
        flux_calibration_frame = bg_sub_flux_stamps_calibrated[:, lower_index:upper_range] / normalization_values[:, :, None, None]
        flux_calibration_frame = comb_func(flux_calibration_frame[:, lower_index_frame_combine:], axis=1)
        flux_calibration_frames.append(flux_calibration_frame)
    flux_calibration_frames = np.array(flux_calibration_frames)
    flux_calibration_frames = np.swapaxes(flux_calibration_frames, 0, 1)
    fits.writeto(os.path.join(converted_dir, 'master_flux_calibrated_psf_frames.fits'),
                 flux_calibration_frames.astype('float32'), overwrite=overwrite_preprocessing)
