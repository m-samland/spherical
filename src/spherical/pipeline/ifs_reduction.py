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
from astropy.stats import mad_std, sigma_clip
from photutils.aperture import CircularAnnulus, CircularAperture

from spherical.database import metadata
from spherical.pipeline import flux_calibration, toolbox, transmission
from spherical.pipeline.steps.download_data import download_data_for_observation, update_observation_file_paths
from spherical.pipeline.steps.bundle_output import bundle_IFS_output_into_cubes
from spherical.pipeline.steps.extract_cubes import extract_cubes_with_multiprocessing
from spherical.pipeline.steps.find_star import fit_centers_in_parallel
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
        for key in frame_types_to_extract:
            if bundle_hexagons:
                bundle_IFS_output_into_cubes(
                    key, cube_outputdir, output_type='hexagons', overwrite=overwrite_bundle)
            if bundle_residuals:
                bundle_IFS_output_into_cubes(
                    key, cube_outputdir, output_type='residuals', overwrite=overwrite_bundle)
            bundle_IFS_output_into_cubes(
                key, cube_outputdir, output_type='resampled', overwrite=overwrite_bundle)

        # If we are using certain non-least-square methods, and linear_wavelength is set,
        # create a special wavelength array; otherwise use lam_midpts from instrument.
        if (extraction_parameters['method'] in non_least_square_methods) \
                and extraction_parameters['linear_wavelength']:
            wavelengths = np.linspace(
                instrument.wavelength_range[0].value,
                instrument.wavelength_range[1].value,
                39
            )
        else:
            wavelengths = instrument.lam_midpts

        fits.writeto(os.path.join(converted_dir, 'wavelengths.fits'),
                     wavelengths, overwrite=overwrite_bundle)

    """ FRAME INFO COMPUTATION """
    if compute_frames_info:
        frames_info = {}
        for key in ['FLUX', 'CORO', 'CENTER']:
            if len(observation.frames[key]) == 0:
                continue
            frames_table = copy.copy(observation.frames[key])
            frames_info[key] = metadata.prepare_dataframe(frames_table)
            metadata.compute_times(frames_info[key])
            metadata.compute_angles(frames_info[key])
            frames_info[key].to_csv(
                os.path.join(converted_dir, 'frames_info_{}.csv'.format(key.lower())))
            # except:
            #     print("Failed to compute angles for key: {}".format(key))

    """ DETERMINE CENTERS """
    if find_centers:
        fit_centers_in_parallel(
            converted_dir=converted_dir,
            observation=observation,
            overwrite=overwrite_preprocessing,
            ncpu=reduction_parameters['ncpu_find_center'])

    if process_extracted_centers:
        plot_dir = os.path.join(converted_dir, 'center_plots/')
        if not path.exists(plot_dir):
            os.makedirs(plot_dir)
        spot_centers = fits.getdata(os.path.join(converted_dir, 'spot_centers.fits'))
        # spot_distances = fits.getdata(os.path.join(converted_dir, 'spot_distances.fits'))
        image_centers = fits.getdata(os.path.join(converted_dir, 'image_centers.fits'))
        # spot_amplitudes = fits.getdata(os.path.join(converted_dir, 'spot_fit_amplitudes.fits'))
        center_cube = fits.getdata(os.path.join(converted_dir, 'center_cube.fits'))
        wavelengths = fits.getdata(os.path.join(converted_dir, 'wavelengths.fits'))

        satellite_psf_stamps = toolbox.extract_satellite_spot_stamps(center_cube, spot_centers, stamp_size=57,
                                                                     shift_order=3, plot=False)
        master_satellite_psf_stamps = np.nanmean(np.nanmean(satellite_psf_stamps, axis=2), axis=1)
        fits.writeto(
            os.path.join(converted_dir, 'satellite_psf_stamps.fits'),
            satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)
        fits.writeto(
            os.path.join(converted_dir, 'master_satellite_psf_stamps.fits'),
            master_satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)

        # mean_spot_stamps = np.sum(satellite_psf_stamps, axis=2)
        # aperture_size = 3

        # stamp_size = [mean_spot_stamps.shape[-1], mean_spot_stamps.shape[-2]]
        # stamp_center = [mean_spot_stamps.shape[-1] // 2, mean_spot_stamps.shape[-2] // 2]
        # aperture = CircularAperture(stamp_center, aperture_size)
        #
        # psf_mask = aperture.to_mask(method='center')
        # # Make sure only pixels are used for which data exists
        # psf_mask = psf_mask.to_image(stamp_size) > 0
        # flux_sum_with_bg = np.sum(mean_spot_stamps[:, :, psf_mask], axis=2)
        #
        # bg_aperture = CircularAnnulus(stamp_center, r_in=aperture_size, r_out=aperture_size+2)
        # bg_mask = bg_aperture.to_mask(method='center')
        # bg_mask = bg_mask.to_image(stamp_size) > 0
        # area = np.pi*aperture_size**2
        # bg_flux = np.mean(mean_spot_stamps[:, :, bg_mask], axis=2) * area
        #
        # flux_sum_without_bg = flux_sum_with_bg - bg_flux

        if (extraction_parameters['method'] in non_least_square_methods) \
                and extraction_parameters['linear_wavelength'] is True:
            remove_indices = [0, 1, 21, 38]
        else:
            remove_indices = [0, 13, 19, 20]

        anomalous_centers_mask = np.zeros_like(wavelengths).astype('bool')
        anomalous_centers_mask[remove_indices] = True

        image_centers_fitted = np.zeros_like(image_centers)
        image_centers_fitted2 = np.zeros_like(image_centers)

        # wavelengths_clean = np.delete(wavelengths, remove_indices)
        # image_centers_clean = np.delete(image_centers, remove_indices, axis=0)

        coefficients_x_list = []
        coefficients_y_list = []

        for frame_idx in range(image_centers.shape[1]):
            good_wavelengths = ~anomalous_centers_mask & np.all(
                np.isfinite(image_centers[:, frame_idx]), axis=1)
            try:
                coefficients_x = np.polyfit(
                    wavelengths[good_wavelengths], image_centers[good_wavelengths, frame_idx, 0], deg=2)
                coefficients_y = np.polyfit(
                    wavelengths[good_wavelengths], image_centers[good_wavelengths, frame_idx, 1], deg=3)

                coefficients_x_list.append(coefficients_x)
                coefficients_y_list.append(coefficients_y)

                image_centers_fitted[:, frame_idx, 0] = np.poly1d(coefficients_x)(wavelengths)
                image_centers_fitted[:, frame_idx, 1] = np.poly1d(coefficients_y)(wavelengths)
            except:
                print("Failed first iteration polyfit for frame {}".format(frame_idx))
                image_centers_fitted[:, frame_idx, 0] = np.nan
                image_centers_fitted[:, frame_idx, 1] = np.nan

        for frame_idx in range(image_centers.shape[1]):
            if np.all(~np.isfinite(image_centers_fitted[:, frame_idx])):
                image_centers_fitted2[:, frame_idx, 0] = np.nan
                image_centers_fitted2[:, frame_idx, 1] = np.nan
            else:
                deviation = image_centers - image_centers_fitted
                filtered_data = sigma_clip(
                    deviation, axis=0, sigma=3, maxiters=None, cenfunc='median', stdfunc=mad_std, masked=True, copy=True)
                anomalous_centers_mask = np.any(filtered_data.mask, axis=2)
                anomalous_centers_mask[remove_indices] = True
                try:
                    coefficients_x = np.polyfit(
                        wavelengths[~anomalous_centers_mask[:, 1]],
                        image_centers[~anomalous_centers_mask[:, 1], frame_idx, 0], deg=2)
                    coefficients_y = np.polyfit(
                        wavelengths[~anomalous_centers_mask[:, 1]],
                        image_centers[~anomalous_centers_mask[:, 1], frame_idx, 1], deg=3)

                    coefficients_x_list.append(coefficients_x)
                    coefficients_y_list.append(coefficients_y)

                    image_centers_fitted2[:, frame_idx, 0] = np.poly1d(coefficients_x)(wavelengths)
                    image_centers_fitted2[:, frame_idx, 1] = np.poly1d(coefficients_y)(wavelengths)
                except:
                    print("Failed second iteration polyfit for frame {}".format(frame_idx))
                    image_centers_fitted2[:, frame_idx, 0] = np.nan
                    image_centers_fitted2[:, frame_idx, 1] = np.nan

        fits.writeto(os.path.join(converted_dir, 'image_centers_fitted.fits'),
                     image_centers_fitted, overwrite=overwrite_preprocessing)
        fits.writeto(os.path.join(converted_dir, 'image_centers_fitted_robust.fits'),
                     image_centers_fitted2, overwrite=overwrite_preprocessing)


    if plot_image_center_evolution:
        image_centers = fits.getdata(os.path.join(converted_dir, 'image_centers.fits'))
        image_centers_fitted = fits.getdata(os.path.join(converted_dir, 'image_centers_fitted.fits'))
        image_centers_fitted2 = fits.getdata(os.path.join(converted_dir, 'image_centers_fitted_robust.fits'))

        plot_dir = os.path.join(converted_dir, 'center_plots/')
        if not path.exists(plot_dir):
            os.makedirs(plot_dir)

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D
        # --- Step 1: Load time info ---
        frame_info_center = pd.read_csv(os.path.join(converted_dir, 'frames_info_center.csv'))
        time_strings = frame_info_center["TIME"]
        times = pd.to_datetime(time_strings)

        # --- Step 2: Compute elapsed minutes ---
        start_time = times.min()
        elapsed_minutes = (times - start_time).dt.total_seconds() / 60.0

        # --- Step 3: Normalize elapsed minutes for colormap ---
        norm = Normalize(vmin=elapsed_minutes.min(), vmax=elapsed_minutes.max())
        cmap = plt.cm.PiYG
        colors = cmap(norm(elapsed_minutes))

        # --- Step 4: Plot using elapsed-time-based colors ---
        plt.close()
        n_wavelengths = image_centers.shape[0]
        n_frames = image_centers.shape[1]
        sizes = np.linspace(20, 300, n_wavelengths)

        fig, ax = plt.subplots(figsize=(8, 6))

        for frame_idx in range(n_frames):
            color = colors[frame_idx]
            ax.scatter(image_centers_fitted[:, frame_idx, 0], image_centers_fitted[:, frame_idx, 1],
                    s=sizes, marker='o', color=color, alpha=0.6)
            ax.scatter(image_centers_fitted2[:, frame_idx, 0], image_centers_fitted2[:, frame_idx, 1],
                    s=sizes, marker='x', color=color, alpha=0.9)
            ax.scatter(image_centers[:, frame_idx, 0], image_centers[:, frame_idx, 1],
                    s=sizes, marker='+', color=color, alpha=0.6)

        # --- Step 5: Add legend ---
        # --- Updated: Move legend outside the plot ---
        # --- Updated: Move legend below the plot ---
        legend_elements = [
            Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=10, label='1st Fit (fitted)'),
            Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=10, label='2nd Fit (robust)'),
            Line2D([0], [0], marker='+', color='gray', linestyle='None', markersize=10, label='Original Data'),
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            title='Marker Meaning',
            frameon=False
        )

        # --- Step 6: Add colorbar ---
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # required for colorbar
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Elapsed Time (minutes)')

        # --- Final plot adjustments ---
        ax.set_xlabel('X Center Position')
        ax.set_ylabel('Y Center Position')
        ax.set_title('Center Position Evolution per Wavelength and Time')
        ax.set_aspect('equal')

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)  # Makes space for the legend

        plt.savefig(os.path.join(plot_dir, 'center_evolution_time_colorbar.pdf'), bbox_inches='tight')



    if calibrate_spot_photometry:
        satellite_psf_stamps = fits.getdata(os.path.join(
            converted_dir, 'satellite_psf_stamps.fits')).astype('float64')
        # flux_variance = fits.getdata(os.path.join(converted_dir, 'flux_cube.fits'), 1)

        stamp_size = [satellite_psf_stamps.shape[-1], satellite_psf_stamps.shape[-2]]
        stamp_center = [satellite_psf_stamps.shape[-1] // 2, satellite_psf_stamps.shape[-2] // 2]

        # BG measurement
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
            ma_spot_stamps,  # satellite_psf_stamps[:, :, :, bg_mask],
            sigma=3, maxiters=5, cenfunc=np.nanmedian, stdfunc=np.nanstd,
            axis=3, masked=True, return_bounds=False)

        bg_counts = np.ma.median(sigma_clipped_array, axis=3).data
        bg_std = np.ma.std(sigma_clipped_array, axis=3).data

        # BG correction of stamps
        bg_corr_satellite_psf_stamps = satellite_psf_stamps - bg_counts[:, :, :, None, None]

        fits.writeto(
            os.path.join(converted_dir, 'satellite_psf_stamps_bg_corrected.fits'),
            bg_corr_satellite_psf_stamps.astype('float32'), overwrite=overwrite_preprocessing)

        aperture = CircularAperture(stamp_center, 3)
        psf_mask = aperture.to_mask(method='center')
        # Make sure only pixels are used for which data exists
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
            os.path.join(converted_dir, 'master_satellite_psf_stamps_bg_corr.fits'),
            master_satellite_psf_stamps_bg_corr.astype('float32'), overwrite=overwrite_preprocessing)

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
            flux_center, flux_amplitude = find_star.star_centers_from_PSF_img_cube(
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