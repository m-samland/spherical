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

    """ FLUX FRAMES """
    if calibrate_flux_psf:
        wavelengths = fits.getdata(
            os.path.join(converted_dir, 'wavelengths.fits'))
        flux_cube = fits.getdata(os.path.join(converted_dir, 'flux_cube.fits')).astype('float64')
        # flux_variance = fits.getdata(os.path.join(converted_dir, 'flux_cube.fits'), 1)
        frames_info = {}
        frames_info['CENTER'] = pd.read_csv(os.path.join(converted_dir, 'frames_info_center.csv'))
        frames_info['FLUX'] = pd.read_csv(os.path.join(converted_dir, 'frames_info_flux.csv'))
        
        plot_dir = os.path.join(converted_dir, 'flux_plots/')
        if not path.exists(plot_dir):
            os.makedirs(plot_dir)

        flux_centers = []
        flux_amplitudes = []
        # guess_center_yx = []
        # wave_median_flux_image = np.nanmedian(flux_cube[1:-1], axis=0)
        # median_flux_image = np.nanmedian(wave_median_flux_image, axis=0)

        # guess_center_yx = np.unravel_index(
        #     np.nanargmax(median_flux_image), median_flux_image.shape)
        for frame_number in range(flux_cube.shape[1]):
            data = flux_cube[:, frame_number]
            flux_center, flux_amplitude = star_centers_from_PSF_img_cube(
                cube=data,
                wave=wavelengths,
                pixel=7.46,
                guess_center_yx=None,  # [frame_number],
                fit_background=False,
                fit_symmetric_gaussian=True,
                mask_deviating=False,
                deviation_threshold=0.8,
                mask=None,
                save_path=None,
                verbose=False)
            flux_centers.append(flux_center)
            flux_amplitudes.append(flux_amplitude)

        flux_centers = np.expand_dims(
            np.swapaxes(np.array(flux_centers), 0, 1),
            axis=2)
        flux_amplitudes = np.swapaxes(np.array(flux_amplitudes), 0, 1)
        fits.writeto(
            os.path.join(converted_dir, 'flux_centers.fits'), flux_centers, overwrite=overwrite_preprocessing)
        fits.writeto(
            os.path.join(converted_dir, 'flux_gauss_amplitudes.fits'), flux_amplitudes, overwrite=overwrite_preprocessing)

        flux_stamps = toolbox.extract_satellite_spot_stamps(
            flux_cube, flux_centers, stamp_size=57,
            shift_order=3, plot=False)
        # flux_variance_stamps = toolbox.extract_satellite_spot_stamps(
        #     flux_cube, flux_centers, stamp_size=57,
        #     shift_order=3, plot=False)
        fits.writeto(os.path.join(converted_dir, 'flux_stamps_uncalibrated.fits'),
                     flux_stamps.astype('float32'), overwrite=overwrite_preprocessing)

        # Adjust for exposure time and ND filter, put all frames to 1 second exposure
        # wave, bandwidth = transmission.wavelength_bandwidth_filter(filter_comb)
        if len(frames_info['FLUX']['INS4 FILT2 NAME'].unique()) > 1:
            raise ValueError('Non-unique ND filters in sequence.')
        else:
            ND = frames_info['FLUX']['INS4 FILT2 NAME'].unique()[0]

        _, attenuation = transmission.transmission_nd(ND, wave=wavelengths)
        fits.writeto(os.path.join(converted_dir, 'nd_attenuation.fits'),
                     attenuation, overwrite=overwrite_preprocessing)
        dits_flux = np.array(frames_info['FLUX']['DET SEQ1 DIT'])
        dits_center = np.array(frames_info['CENTER']['DET SEQ1 DIT'])
        # dits_coro = np.array(frames_info['CORO']['DET SEQ1 DIT'])

        unique_dits_center, unique_dits_center_counts = np.unique(dits_center, return_counts=True)

        # Normalize coronagraphic sequence to DIT that is most common
        if len(unique_dits_center) == 1:
            dits_factor = unique_dits_center[0] / dits_flux
            most_common_dit_center = unique_dits_center[0]
        else:
            most_common_dit_center = unique_dits_center[np.argmax(unique_dits_center_counts)]
            dits_factor = most_common_dit_center / dits_flux

        dit_factor_center = most_common_dit_center / dits_center

        fits.writeto(os.path.join(converted_dir, 'center_frame_dit_adjustment_factors.fits'),
                     dit_factor_center, overwrite=overwrite_preprocessing)

        print("Attenuation: {}".format(attenuation))
        # if adjust_dit:
        flux_stamps_calibrated = flux_stamps * dits_factor[None, :, None, None]

        flux_stamps_calibrated = flux_stamps_calibrated / \
            attenuation[:, np.newaxis, np.newaxis, np.newaxis]
        fits.writeto(os.path.join(converted_dir, 'flux_stamps_dit_nd_calibrated.fits'),
                     flux_stamps_calibrated, overwrite=overwrite_preprocessing)

        # fwhm_angle = ((wavelengths * u.nm) / (7.99 * u.m)).to(
        #     u.mas, equivalencies=u.dimensionless_angles())
        # fwhm = fwhm_angle.to(u.pixel, u.pixel_scale(0.00746 * u.arcsec / u.pixel)).value
        # aperture_sizes = np.round(fwhm * 2.1)

        flux_photometry = flux_calibration.get_aperture_photometry(
            flux_stamps_calibrated, aperture_radius_range=[1, 15],
            bg_aperture_inner_radius=15,
            bg_aperture_outer_radius=18)

        filehandler = open(os.path.join(converted_dir, 'flux_photometry.obj'), 'wb')
        pickle.dump(flux_photometry, filehandler)
        filehandler.close()

        fits.writeto(os.path.join(converted_dir, 'flux_amplitude_calibrated.fits'),
                     flux_photometry['psf_flux_bg_corr_all'], overwrite=overwrite_preprocessing)

        fits.writeto(os.path.join(converted_dir, 'flux_snr.fits'),
                     flux_photometry['snr_all'], overwrite=overwrite_preprocessing)

        plt.close()
        plt.plot(flux_photometry['aperture_sizes'], flux_photometry['snr_all'][:, :, 0])
        plt.xlabel('Aperture Size (pix)')
        plt.ylabel('BG limited SNR')
        plt.savefig(os.path.join(plot_dir, 'Flux_PSF_aperture_SNR.png'))
        plt.close()
        bg_sub_flux_stamps_calibrated = flux_stamps_calibrated - \
            flux_photometry['psf_bg_counts_all'][:, :, None, None]

        fits.writeto(os.path.join(converted_dir, 'flux_stamps_calibrated_bg_corrected.fits'),
                     bg_sub_flux_stamps_calibrated.astype('float32'), overwrite=overwrite_preprocessing)

        # bg_sub_flux_phot = get_aperture_photometry(
        #     bg_sub_flux_stamps_calibrated, aperture_radius_range=[1, 15],
        #     bg_aperture_inner_radius=15,
        #     bg_aperture_outer_radius=17)

        # Make master PSF
        flux_calibration_indices, indices_of_discontinuity = flux_calibration.get_flux_calibration_indices(
            frames_info['CENTER'], frames_info['FLUX'])
        flux_calibration_indices.to_csv(os.path.join(converted_dir, 'flux_calibration_indices.csv'))
        indices_of_discontinuity.tofile(os.path.join(
            converted_dir, 'indices_of_discontinuity.csv'), sep=',')

        # Normalize all PSF frames to respective calibration flux index based on aperture size 3 pixel
        # (index 2)

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
            # Old way: pick closest in time
            # reference_value_old = flux_photometry['psf_flux_bg_corr_all'][2][:, flux_calibration_indices['flux_idx'].iloc[idx]]
            # New way do mean
            reference_value = np.nanmean(
                flux_photometry['psf_flux_bg_corr_all'][2][:, lower_index_frame_combine:upper_range], axis=1)


            normalization_values = phot_values / reference_value[:, None]

            flux_calibration_frame = bg_sub_flux_stamps_calibrated[:,
                                                                   lower_index:upper_range] / normalization_values[:, :, None, None]
            # Could weight by snr, but let's not do that yet
            # snr_values = flux_photometry['snr_all'][2][:, lower_index:upper_range]
            # if reduction_parameters['exclude_first_flux_frame']:
            #     flux_calibration_frame = np.mean(flux_calibration_frame[:, 1:], axis=1)
            # else:
            flux_calibration_frame = comb_func(flux_calibration_frame[:, lower_index_frame_combine:], axis=1)
            flux_calibration_frames.append(flux_calibration_frame)

        flux_calibration_frames = np.array(flux_calibration_frames)
        flux_calibration_frames = np.swapaxes(flux_calibration_frames, 0, 1)

        fits.writeto(os.path.join(converted_dir, 'master_flux_calibrated_psf_frames.fits'),
                     flux_calibration_frames.astype('float32'), overwrite=overwrite_preprocessing)

    if spot_to_flux:
        plot_dir = os.path.join(converted_dir, 'flux_plots/')
        if not path.exists(plot_dir):
            os.makedirs(plot_dir)

        wavelengths = fits.getdata(
            os.path.join(converted_dir, 'wavelengths.fits')) * u.nm
        wavelengths = wavelengths.to(u.micron)
        flux_amplitude = fits.getdata(
            os.path.join(converted_dir, 'flux_amplitude_calibrated.fits'))[2]

        spot_amplitude = fits.getdata(
            os.path.join(converted_dir, 'spot_amplitudes.fits'))
        master_spot_amplitude = np.mean(spot_amplitude, axis=2)

        psf_flux = flux_calibration.SimpleSpectrum(
            wavelength=wavelengths,
            flux=flux_amplitude,
            norm_wavelength_range=[1.0, 1.3] * u.micron,
            metadata=frames_info['FLUX'],
            rescale=False,
            normalize=False)

        # psf_flux_norm = flux_calibration.SimpleSpectrum(
        #     wavelength=wavelengths,
        #     flux=flux_amplitude,
        #     norm_wavelength_range=[1.0, 1.3] * u.micron,
        #     metadata=frames_info['FLUX'],
        #     rescale=False,
        #     normalize=True)

        spot_flux = flux_calibration.SimpleSpectrum(
            wavelength=wavelengths,
            flux=master_spot_amplitude,  # flux_sum_with_bg,
            norm_wavelength_range=[1.0, 1.3] * u.micron,
            metadata=frames_info['CENTER'],
            rescale=True,
            normalize=False)

        # spot_flux_norm = flux_calibration.SimpleSpectrum(
        #     wavelength=wavelengths,
        #     flux=master_spot_amplitude,  # flux_sum_with_bg,
        #     norm_wavelength_range=[1.0, 1.3] * u.micron,
        #     metadata=frames_info['CENTER'],
        #     rescale=True,
        #     normalize=True)

        psf_flux.plot_flux(plot_original=False, autocolor=True, cmap=plt.cm.cool,
                           savefig=True, savedir=plot_dir, filename='psf_flux.png',
                           )
        spot_flux.plot_flux(plot_original=False, autocolor=True, cmap=plt.cm.cool,
                            savefig=True, savedir=plot_dir, filename='spot_flux_rescaled.png',
                            )

        flux_calibration_indices = pd.read_csv(os.path.join(
            converted_dir, 'flux_calibration_indices.csv'))

        normalization_factors, averaged_normalization, std_dev_normalization = flux_calibration.compute_flux_normalization_factors(
            flux_calibration_indices, psf_flux, spot_flux)

        flux_calibration.plot_flux_normalization_factors(
            flux_calibration_indices, normalization_factors[:, 1:-1],
            wavelengths=wavelengths[1:-1], cmap=plt.cm.cool,
            savefig=True, savedir=plot_dir)

        fits.writeto(os.path.join(
            converted_dir, 'spot_normalization_factors.fits'), normalization_factors,
            overwrite=True)

        fits.writeto(os.path.join(
            converted_dir, 'spot_normalization_factors_average.fits'), averaged_normalization,
            overwrite=True)

        fits.writeto(os.path.join(
            converted_dir, 'spot_normalization_factors_stddev.fits'), std_dev_normalization,
            overwrite=True)

        flux_calibration.plot_timeseries(
            frames_info['FLUX'], frames_info['CENTER'], psf_flux, spot_flux, averaged_normalization,
            x_axis_quantity='HOUR ANGLE', wavelength_channels=np.arange(len(wavelengths))[1:-1],
            savefig=True, savedir=plot_dir)

        scaled_spot_flux = spot_flux.flux * averaged_normalization[:, None]
        temporal_mean = np.nanmean(scaled_spot_flux, axis=1)
        amplitude_variation = scaled_spot_flux / temporal_mean[:, None]

        fits.writeto(os.path.join(
            converted_dir, 'spot_amplitude_variation.fits'), amplitude_variation,
            overwrite=True)

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