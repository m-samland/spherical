"""Tools for detecting bad frames."""

__all__ = ['Rejection_statistics', 'select_using_robust_standard_deviation',
           'detect_bad_frames_sortsym', 'remove_bad_frames']

from os import path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std


def mask_for_center(radius, yx_size, yx_center=None):
    x = np.arange(yx_size[1])
    y = np.arange(yx_size[0]).reshape(-1, 1)

    if yx_center is None:
        yx_center = (yx_size[0] // 2., yx_size[1] // 2.)

    dist = np.sqrt((x - yx_center[1])**2 + (y - yx_center[0])**2)

    center_mask = dist < radius

    return center_mask


class Rejection_statistics(object):
    """Container class for rejection thresholds

    Contains median of distribution and cut-off thresholds
    selected by algorithms, as well as the fraction of
    rejections.

    Could be extended to actually contain the distribution
    and all related rejection as well as plotting algos.
    """

    def __init__(self, median, reject_below, reject_above,
                 fraction_rejected):
        self.median = median,
        self.reject_below = reject_below
        self.reject_above = reject_above
        self.fraction_rejected = fraction_rejected


def select_using_robust_standard_deviation(statistic, robust_sigma=2.0):
    """Select outliers using robust sigma clipping.

    Outliers of a 1d point distribution are identified
    using the deviation from the median by sigma times
    the robust standard deviation (astropy.stats.mad_std).

    Parameters
    ----------
    statistic : array
        contains the 1d distribution.
    robust_sigma : float
        Number of robust standard deviations points can
        deviate from the median.

    Returns
    -------
    type
        Rejection_statistics object containing summary
        statistics.

    """
    median_statistic = np.median(statistic)
    robust_deviation = mad_std(statistic)
    reject_above = median_statistic + robust_sigma * robust_deviation
    reject_below = median_statistic - robust_sigma * robust_deviation
    bad_frame_mask = np.logical_or(
        statistic > reject_above,
        statistic < reject_below)
    bad_frame_indices = np.where(bad_frame_mask)[0]
    number_rejected = np.sum(bad_frame_mask)
    fraction_rejected = float(number_rejected) / len(statistic)

    rejection_statistics = Rejection_statistics(
        median=median_statistic,
        reject_below=reject_below,
        reject_above=reject_above,
        fraction_rejected=fraction_rejected
    )

    return bad_frame_indices, rejection_statistics


def detect_bad_frames_sortsym(data_cube, channel=0, radius=6., sigma=2.0,
                              output_plot_path=None):
    """Short summary.

    Parameters
    ----------
    data_cube : type
        Description of parameter `data_cube`.
    channel : type
        Description of parameter `channel`.
    radius : type
        Description of parameter `radius`.
    sigma : type
        Description of parameter `sigma`.
    output_plot_path : type
        Description of parameter `output_plot_path`.

    Returns
    -------
    type
        Description of returned object.

    """
    if channel is not None:
        data_cube = data_cube[channel]

    yx_size = (data_cube.shape[-2], data_cube.shape[-1])
    center_mask = mask_for_center(radius=radius, yx_size=yx_size)

    statistic = np.sum(data_cube[:, center_mask], axis=1)

    bad_frame_indices, rejection_statistics = \
        select_using_robust_standard_deviation(statistic=statistic, robust_sigma=sigma)

    if output_plot_path is not None:
        plt.close()
        plt.hlines(rejection_statistics.reject_below, xmin=0, xmax=len(statistic))
        plt.hlines(rejection_statistics.median, xmin=0, xmax=len(statistic), linestyles='dashed')
        plt.hlines(rejection_statistics.reject_above, xmin=0, xmax=len(statistic))
        plt.plot(statistic)
        plt.xlabel("Frame")
        plt.ylabel("Central Flux (radius: {} pix)".format(str(radius)))
        plt.title("Sigma: {} Rejection fraction: {:.02f}".format(
            sigma, rejection_statistics.fraction_rejected))
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)

    return bad_frame_indices, rejection_statistics.fraction_rejected


def remove_bad_frames(data_cube_path,
                      channel=0, radius=6., sigma=2.0,
                      make_plot=True):
    """Short summary.

    Parameters
    ----------
    data_cube_path : type
        Description of parameter `data_cube_path`.
    parallactic_angle_path : type
        Description of parameter `parallactic_angle_path`.
    channel : type
        Description of parameter `channel`.
    radius : type
        Description of parameter `radius`.
    sigma : type
        Description of parameter `sigma`.
    make_plot : type
        Description of parameter `make_plot`.

    Returns
    -------
    type
        Description of returned object.

    """
    data_cube = fits.getdata(data_cube_path)

    data_directory = path.dirname(data_cube_path)

    if make_plot:
        output_plot_path = path.join(data_directory, 'frame_selection.png')
    else:
        output_plot_path = None

    bad_frame_indices, fraction_rejected = detect_bad_frames_sortsym(
        data_cube, channel=channel, radius=radius, sigma=sigma,
        output_plot_path=output_plot_path)

    # data_output_path = path.join(data_directory, 'center_im_sorted.fits')
    # pa_output_path = path.join(data_directory, 'rotnth_sorted.fits')
    bad_frame_indices_path = path.join(data_directory, 'bad_frame_indices.fits')
    # data_cube = np.delete(data_cube, bad_frame_indices, axis=1)
    # pa = np.delete(pa, bad_frame_indices, axis=0)
    # fits.writeto(data_output_path, data_cube, header, overwrite=True)
    # fits.writeto(pa_output_path, pa, overwrite=True)

    fits.writeto(bad_frame_indices_path, bad_frame_indices, overwrite=True)

    return bad_frame_indices, fraction_rejected

# bad_frame_indices, fraction_rejected = remove_bad_frames(
#     data_cube_path, parallactic_angle_path,
#     channel=0, radius=6., sigma=2.0,
#     make_plot=True)
