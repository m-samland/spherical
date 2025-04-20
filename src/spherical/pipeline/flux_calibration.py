import os

import matplotlib

matplotlib.use(backend='Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from photutils.aperture import CircularAnnulus, CircularAperture

from spherical.database.database_utils import find_nearest


# def find_nearest(array, value):
#     """Return index of array value closest to specified value.
#     """
#     idx = (np.abs(array - value)).argmin()
#     return idx


class SimpleSpectrum(object):

    def __init__(self, wavelength, flux, norm_wavelength_range=None, metadata=None,
                 rescale=False, rescale_exponent=2., normalize=False):
        self.wavelength = wavelength
        self.flux = flux
        self.original_flux = flux
        self.norm_wavelength_range = norm_wavelength_range
        self.metadata = metadata
        if rescale:
            self.rescale_flux(exponent=rescale_exponent)
        if normalize:
            self.normalize_flux()


#         self.init_plot()


    def normalize_flux(self):
        if self.norm_wavelength_range is None:
            self.norm_factor = np.sum(self.flux, axis=0)
        else:
            self.wavelength_mask = np.logical_and(
                self.wavelength >= self.norm_wavelength_range[0],
                self.wavelength <= self.norm_wavelength_range[1])
            self.norm_factor = np.sum(self.flux[self.wavelength_mask], axis=0)
        self.flux = self.flux / self.norm_factor

    def rescale_flux(self, wavelength=None, exponent=2):
        if wavelength is None:
            wavelength = self.wavelength
        self.flux = self.flux * (wavelength.value[:, None])**exponent

#     def init_plot(self):
#         fig = plt.figure(9)
#         ax = fig.add_subplot(111)

    def plot_flux(self, plot_original=False, autocolor=True, cmap=plt.cm.PiYG,
                  savefig=False, savedir=None, filename=None):
        plt.close()
        fig = plt.figure(9)
        ax = fig.add_subplot(111)

        if self.flux.ndim == 1:
            n_frames = 1
        else:
            n_frames = self.flux.shape[-1]
        if self.metadata is not None:
            metadata = self.metadata
            mjd_range = np.max(metadata['MJD'] - np.min(metadata['MJD']))
            scaled_values = (metadata['MJD'].values - np.min(metadata['MJD'])) / mjd_range
            colors = cmap(scaled_values)
        else:
            colors = cmap(np.linspace(0, 1, n_frames))

        if plot_original:
            plotted_quantity = self.original_flux
        else:
            plotted_quantity = self.flux
#         if rescaled:
#             if normalized:
#                 plotted_quantity = self.rescaled_norm_flux
#                 plt.ylabel('Normalized flux')
#             else:
#                 plotted_quantity = self.rescaled_flux
#                 plt.ylabel('Flux')
#         else:
#             if normalized:
#                 plotted_quantity = self.norm_flux
#                 plt.ylabel('Normalized flux')
#             else:
#                 plotted_quantity = self.flux
#                 plt.ylabel('Flux')

        for frame_idx in range(n_frames):
            if frame_idx == 0 or frame_idx == n_frames - 1:
                label = metadata['TIME'][frame_idx]
            else:
                label = None
            ax.plot(self.wavelength, plotted_quantity[...,
                    frame_idx], c=colors[frame_idx], label=label)
        plt.legend()
        plt.xlabel(f'Wavelength ({self.wavelength.unit})')
        if savefig and savedir and filename is not None:
            plt.savefig(os.path.join(savedir, f'{filename}'))
            plt.close()
        else:
            plt.show()


def get_aperture_photometry(flux_stamps, aperture_radius_range=[1, 15],
                            bg_aperture_inner_radius=15,
                            bg_aperture_outer_radius=18):

    flux_stamps_flattened = flux_stamps.reshape(
        flux_stamps.shape[0], flux_stamps.shape[1], -1)
    stamp_size = [flux_stamps.shape[-1], flux_stamps.shape[-2]]
    stamp_center = [flux_stamps.shape[-1] // 2, flux_stamps.shape[-2] // 2]

    aperture_sizes = np.arange(aperture_radius_range[0], aperture_radius_range[1])
    # psf_area = np.pi * aperture_sizes**2
    photometry = {}
    photometry['aperture_sizes'] = aperture_sizes

    # Compute background statistics
    bg_aperture = CircularAnnulus(
        stamp_center,
        r_in=bg_aperture_inner_radius,
        r_out=bg_aperture_outer_radius)  # aperture_size+2, r_out=aperture_size+5)
    bg_mask = bg_aperture.to_mask(method='center')
    bg_mask = bg_mask.to_image(stamp_size) > 0

    mask = np.ones_like(flux_stamps)
    mask[:, :, bg_mask] = 0
    ma_flux_stamps = np.ma.array(
        data=flux_stamps_flattened,
        mask=mask.reshape(
            flux_stamps.shape[0], flux_stamps.shape[1], -1))

    sigma_clipped_array = sigma_clip(
        ma_flux_stamps,  # satellite_psf_stamps[:, :, :, bg_mask],
        sigma=3, maxiters=5, cenfunc=np.nanmedian, stdfunc=np.nanstd,
        axis=2, masked=True, return_bounds=False)

    photometry['psf_bg_counts_all'] = np.ma.median(sigma_clipped_array, axis=2).data
    photometry['psf_bg_std_all'] = np.ma.std(sigma_clipped_array, axis=2).data

    psf_flux_with_bg_all = []
    psf_flux_bg_corr_all = []
    photometry['psf_area'] = []

    for aperture_size in aperture_sizes:
        aperture = CircularAperture(stamp_center, aperture_size)
        psf_mask = aperture.to_mask(method='center')
        psf_mask = psf_mask.to_image(stamp_size) > 0
        photometry['psf_area'].append(np.sum(psf_mask))

        mask = np.ones_like(flux_stamps)
        mask[:, :, psf_mask] = 0
        mask = mask.reshape(
            flux_stamps.shape[0], flux_stamps.shape[1], -1)
        ma_flux_stamps.mask = mask
        psf_flux_with_bg_all.append(np.ma.sum(ma_flux_stamps, axis=2).data)
        psf_flux_bg_corr_all.append(np.ma.sum(
            ma_flux_stamps - photometry['psf_bg_counts_all'][:, :, None], axis=2).data)

    photometry['psf_area'] = np.array(photometry['psf_area'])
    photometry['psf_flux_sum_all'] = np.array(psf_flux_with_bg_all)
    photometry['psf_flux_bg_corr_all'] = np.array(psf_flux_bg_corr_all)
    photometry['psf_bg_noise_all'] = photometry['psf_bg_std_all'] * \
        photometry['psf_area'][:, None, None]

    # The following snr is only valid if BG is subtracted first
    photometry['snr_all'] = photometry['psf_flux_bg_corr_all'] / \
        photometry['psf_bg_noise_all']
    photometry['normalized_psf_aperture_flux'] = photometry['psf_flux_bg_corr_all'] / \
        np.nanmax(photometry['psf_flux_bg_corr_all'], axis=0)

    # max_ind = np.argmax(snr_all, axis=0)
    return photometry


def get_flux_calibration_indices(frames_info_center, frames_info_flux, number_excluded_after_coro=None):
    flux_calibration_indices = []
    number_of_flux_frames = len(frames_info_flux)
    if number_of_flux_frames == 1:
        science_idx = find_nearest(
            array=frames_info_center['LST'], value=frames_info_flux['LST'].iloc[0])
        flux_calibration_indices.append({'flux_idx': 0, 'science_idx': science_idx})
    else:
        # assert np.all(
        #     frames_info_flux['EXPTIME'] == frames_info_flux['EXPTIME'].iloc[0]), "Different exposure times for flux frames"
        # flux_time_diff = np.diff(frames_info_flux['LST'])
        flux_time_diff_in_dit = np.diff(
            frames_info_flux['LST'] * 60 * 60) / np.max(frames_info_flux['EXPTIME'].iloc[0])

        # center_time_diff = np.diff(frames_info_center['LST'])
        # center_time_diff_in_dit = np.diff(
        #     frames_info_center['LST'] * 60 * 60) / np.max(frames_info_center['EXPTIME'].iloc[0])

        # At least large than 15 times the maximum flux integration time
        indices_of_discontinuity = np.where(flux_time_diff_in_dit > 15.)[0]
        # for i, index in enumerate(indices_of_discontinuity):
        #     if index + number_excluded_after_coro < number_of_flux_frames:
        #         print(
        #             f"Excluded {number_excluded_after_coro} flux frames after science seqeunce from normalization.")
        #         indices_of_discontinuity[i] += number_excluded_after_coro

        science_idx_first = find_nearest(
            array=frames_info_center['LST'], value=frames_info_flux['LST'].iloc[0])
        science_idx_last = find_nearest(
            array=frames_info_center['LST'], value=frames_info_flux['LST'].iloc[-1])
        flux_calibration_indices.append(
            {'flux_idx': 0,
             'flux_lst': frames_info_flux['LST'].iloc[0],
             'science_idx': science_idx_first,
             'science_lst': frames_info_center['LST'].iloc[science_idx_first]})
        index_of_last_flux_frame = number_of_flux_frames - 1
        flux_calibration_indices.append(
            {'flux_idx': index_of_last_flux_frame,
             'flux_lst': frames_info_flux['LST'].iloc[index_of_last_flux_frame],
             'science_idx': science_idx_last,
             'science_lst': frames_info_center['LST'].iloc[science_idx_last]})

        for idx in indices_of_discontinuity:
            science_idx_1 = find_nearest(
                array=frames_info_center['LST'], value=frames_info_flux['LST'].iloc[idx])
            science_idx_2 = find_nearest(
                array=frames_info_center['LST'], value=frames_info_flux['LST'].iloc[idx+1])
            flux_calibration_indices.append(
                {'flux_idx': idx, 'flux_lst': frames_info_flux['LST'].iloc[idx],
                 'science_idx': science_idx_1, 'science_lst': frames_info_center['LST'].iloc[science_idx_1]})
            flux_calibration_indices.append(
                {'flux_idx': idx+1, 'flux_lst': frames_info_flux['LST'].iloc[idx+1],
                 'science_idx': science_idx_2, 'science_lst': frames_info_center['LST'].iloc[science_idx_2]})
    #         flux_calibration_indices.append(
    #             {'flux_idx': idx+1, 'flux_lst': frames_info_flux['LST'].iloc[idx+1],
    #              'science_idx': science_idx_2, 'science_lst': frames_info_center['LST'].iloc[science_idx_2]})
    #         flux_calibration_indices.append({'flux_idx': idx+1, 'science_idx': science_idx_2})
    flux_calibration_indices = pd.DataFrame(flux_calibration_indices)
    flux_calibration_indices['lst_diff'] = np.abs(
        flux_calibration_indices['flux_lst'] - flux_calibration_indices['science_lst'])

    flux_calibration_indices = flux_calibration_indices.sort_values(['science_lst', 'lst_diff'])
    flux_calibration_indices = flux_calibration_indices.drop_duplicates(
        subset=['science_idx'], keep='first')

    return flux_calibration_indices, indices_of_discontinuity


def compute_flux_normalization_factors(
        flux_calibration_indices, psf_flux, spot_flux,
        moving_wavelength_average=3):
    normalization_factors = []
    for idx, row in enumerate(flux_calibration_indices.iterrows()):
        flux_arr = psf_flux.flux[:, int(row[1]['flux_idx'])]
        spot_arr = spot_flux.flux[:, int(row[1]['science_idx'])]

        normalization_factors.append(flux_arr / spot_arr)
    normalization_factors = np.array(normalization_factors)
    averaged_normalization = np.nanmean(normalization_factors, axis=0)
    std_dev_normalization = np.nanstd(normalization_factors, axis=0)

    return normalization_factors, averaged_normalization, std_dev_normalization


def plot_flux_normalization_factors(
        flux_calibration_indices, normalization_factors, wavelengths=None,
        cmap=plt.cm.cool,
        savefig=False, savedir=None):

    plt.close()
    mjd_range = np.max(flux_calibration_indices['flux_lst']) - \
        np.min(flux_calibration_indices['flux_lst'])
    scaled_values = (flux_calibration_indices['flux_lst'].values -
                     np.min(flux_calibration_indices['flux_lst'])) / mjd_range
    colors = cmap(scaled_values)

    if wavelengths is None:
        wavelengths = np.arange(normalization_factors.shape[-1])
        plt.xlabel('Wavelength Channel')
    else:
        plt.xlabel(f'Wavelength ({wavelengths.unit})')

    for idx, normalization in enumerate(normalization_factors):
        lst = np.mean([flux_calibration_indices['flux_lst'][idx],
                       flux_calibration_indices['science_lst'][idx]])
        plt.plot(
            wavelengths,
            normalization,
            color=colors[idx],
            label=f'LST: {lst:.3f}h')
    plt.legend()
    plt.ylabel('Normalization factors')
    if savefig and savedir is not None:
        plt.savefig(os.path.join(savedir, 'normalization_factors.png'))
        plt.close()
    else:
        plt.show()


def plot_timeseries(frames_info_flux, frames_info_center, psf_flux, spot_flux, averaged_normalization,
                    x_axis_quantity='HOUR ANGLE', wavelength_channels=None,
                    savefig=False, savedir=None):

    plt.close()
    n_channels = psf_flux.flux.shape[0]
    colors = plt.cm.PiYG(np.linspace(0, 1, n_channels))
    # Dark Purple -> light purple -> light green -> dark green

    if wavelength_channels is None:
        wavelength_channels = range(n_channels)

    for i in wavelength_channels:  # range(n_channels):
        plt.plot(frames_info_flux[x_axis_quantity], psf_flux.flux[i, :],
                 'o', color=colors[i], alpha=1, label=f'{i}')
        plt.plot(frames_info_center[x_axis_quantity], spot_flux.flux[i, :] * averaged_normalization[i],
                 'x', color=colors[i], alpha=0.6, label=f'{i}')
    # plt.legend()

    plt.xlabel(x_axis_quantity)
    plt.ylabel('Aperture flux')
    if savefig and savedir is not None:
        plt.savefig(os.path.join(savedir, 'photometric_timeseries.png'))
        plt.close()
    else:
        plt.show()
