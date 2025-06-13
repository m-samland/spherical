"""
Process extracted centers: fits, robust fitting, and output of center positions.

Parameters
----------
converted_dir : str
    Directory where the output files are stored and written.
extraction_parameters : dict
    Extraction parameters dict, must contain 'method' and 'linear_wavelength'.
non_least_square_methods : list
    List of method names that are not least-square based.
overwrite_preprocessing : bool
    Whether to overwrite existing files.
"""

import os

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from spherical.pipeline import toolbox


def run_polynomial_center_fit(converted_dir, extraction_parameters, non_least_square_methods, overwrite_preprocessing):
    """
    Process extracted centers: fits, robust fitting, and output of center positions.
    
    Parameters
    ----------
    converted_dir : str
        Directory where the output files are stored and written.
    extraction_parameters : dict
        Extraction parameters dict, must contain 'method' and 'linear_wavelength'.
    non_least_square_methods : list
        List of method names that are not least-square based.
    overwrite_preprocessing : bool
        Whether to overwrite existing files.
    """

    plot_dir = os.path.join(converted_dir, 'center_plots/')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    image_centers = fits.getdata(os.path.join(converted_dir, 'image_centers.fits'))
    wavelengths = fits.getdata(os.path.join(converted_dir, 'wavelengths.fits'))

    # Defensive: ensure wavelengths is a numpy array
    wavelengths = np.array(wavelengths).flatten()
    image_centers = np.array(image_centers)

    if (extraction_parameters['method'] in non_least_square_methods) and extraction_parameters['linear_wavelength'] is True:
        remove_indices = [0, 1, 21, 38]
    else:
        remove_indices = [0, 13, 19, 20]

    anomalous_centers_mask = np.zeros(wavelengths.shape, dtype=bool)
    anomalous_centers_mask[remove_indices] = True

    image_centers_fitted = np.zeros_like(image_centers)
    image_centers_fitted2 = np.zeros_like(image_centers)

    coefficients_x_list = []
    coefficients_y_list = []

    for frame_idx in range(image_centers.shape[1]):
        good_wavelengths = ~anomalous_centers_mask & np.all(np.isfinite(image_centers[:, frame_idx]), axis=1)
        try:
            coefficients_x = np.polyfit(wavelengths[good_wavelengths], image_centers[good_wavelengths, frame_idx, 0], deg=2)
            coefficients_y = np.polyfit(wavelengths[good_wavelengths], image_centers[good_wavelengths, frame_idx, 1], deg=3)
            coefficients_x_list.append(coefficients_x)
            coefficients_y_list.append(coefficients_y)
            image_centers_fitted[:, frame_idx, 0] = np.poly1d(coefficients_x)(wavelengths)
            image_centers_fitted[:, frame_idx, 1] = np.poly1d(coefficients_y)(wavelengths)
        except Exception:
            image_centers_fitted[:, frame_idx, 0] = np.nan
            image_centers_fitted[:, frame_idx, 1] = np.nan

    for frame_idx in range(image_centers.shape[1]):
        if np.all(~np.isfinite(image_centers_fitted[:, frame_idx])):
            image_centers_fitted2[:, frame_idx, 0] = np.nan
            image_centers_fitted2[:, frame_idx, 1] = np.nan
        else:
            deviation = image_centers - image_centers_fitted
            filtered_data = sigma_clip(deviation[:, frame_idx, :], sigma=3, cenfunc='median', stdfunc='std', masked=True, copy=True)
            # Robustly extract mask from filtered_data
            mask = getattr(filtered_data, 'mask', None)
            if mask is None and isinstance(filtered_data, tuple) and len(filtered_data) > 1:
                mask = filtered_data[1]
            mask_any = np.any(mask, axis=1) if mask is not None else np.zeros(wavelengths.shape, dtype=bool)
            mask_any = np.array(mask_any, dtype=bool)  # Ensure always an array
            mask_any[remove_indices] = True
            try:
                good_idx = ~mask_any
                coefficients_x = np.polyfit(wavelengths[good_idx], image_centers[good_idx, frame_idx, 0], deg=2)
                coefficients_y = np.polyfit(wavelengths[good_idx], image_centers[good_idx, frame_idx, 1], deg=3)
                coefficients_x_list.append(coefficients_x)
                coefficients_y_list.append(coefficients_y)
                image_centers_fitted2[:, frame_idx, 0] = np.poly1d(coefficients_x)(wavelengths)
                image_centers_fitted2[:, frame_idx, 1] = np.poly1d(coefficients_y)(wavelengths)
            except Exception:
                image_centers_fitted2[:, frame_idx, 0] = np.nan
                image_centers_fitted2[:, frame_idx, 1] = np.nan

    fits.writeto(os.path.join(converted_dir, 'image_centers_fitted.fits'), image_centers_fitted, overwrite=overwrite_preprocessing)
    fits.writeto(os.path.join(converted_dir, 'image_centers_fitted_robust.fits'), image_centers_fitted2, overwrite=overwrite_preprocessing)
