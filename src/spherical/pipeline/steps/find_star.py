import os
from typing import Tuple

import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from astropy.io import fits
from astropy.modeling import fitting, models
from matplotlib.backends.backend_pdf import PdfPages

from spherical.pipeline import transmission
from spherical.pipeline.parallel import parallel_map_ordered

global_cmap = 'inferno'


def lines_intersect(a1, a2, b1, b2):
    '''
    Determines the intersection point of two lines passing by points
    (a1,a2) and (b1,b2).

    See https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    Parameters
    ----------

    a, b : 2D tuples
        Coordinates of points on line 1

    c, d : 2D tuples
        Coordinates of points on line 2

    Returns
    -------
    val
        Returns None is lines are parallel, (cx,cy) otherwise.
    '''

    # make sure we have arrays
    a1 = np.array(a1)
    a2 = np.array(a2)
    b1 = np.array(b1)
    b2 = np.array(b2)

    # test lines
    da = a2 - a1                # vector from A1 to A2
    db = b2 - b1                # vector from B1 to B2
    dp = a1 - b1
    pda = [-da[1], da[0]]       # perpendicular to A1-A2 vector

    # parallel lines
    if (pda * db).sum() == 0:
        return None

    # find intersection
    denom = pda @ db
    num = pda @ dp

    return (num / denom) * db + b1


def star_centers_from_waffle_img_cube(cube_cen, wave, waffle_orientation, center_guess, pixel,
                                      orientation_offset=0., mask=None, fit_background=True,
                                      fit_symmetric_gaussian=True,
                                      mask_deviating=True,
                                      deviation_threshold=0.8, high_pass=False,
                                      center_offset=(0, 0), smooth=0, coro=True,
                                      save_plot=True, save_path=None,
                                      verbose=False):
    '''
    Compute star center from waffle images (IRDIS CI, IRDIS DBI, IFS)

    Parameters
    ----------
    cube_cen : array_like
        IRDIFS waffle cube

    wave : array_like
        Wavelength values, in nanometers

    waffle_orientation : str
        String giving the waffle orientation '+' or 'x'

    mask : array_like
        Boolean bad pixel mask (True is bad pixel)

    center_guess : array
        Estimation of the image center as a function of wavelength.
        This should be an array of shape nwave*2.

    pixel : float
        Pixel scale, in mas/pixel

    orientation_offset : float
        Field orientation offset, in degrees

    high_pass : bool
        Apply high-pass filter to the image before searching for the
        satelitte spots. Default is False

    smooth : int
        Apply a gaussian smoothing to the images to reduce noise. The
        value is the sigma of the gaussian in pixel.  Default is no
        smoothing

    center_offset : tuple
        Apply an (x,y) offset to the default center position. The offset
        will move the search box of the waffle spots by the amount of
        specified pixels in each direction. Default is no offset

    coro : bool
        Observation was performed with a coronagraph. Default is True

    save_path : str
        Path where to save the fit images. Default is None, which means
        that the plot is not produced

    logger : logHandler object
        Log handler for the reduction. Default is root logger

    Returns
    -------
    spot_centers : array_like
        Centers of each individual spot in each frame of the cube

    spot_dist : array_like
        The 6 possible distances between the different spots

    img_centers : array_like
        The star center in each frame of the cube

    '''

    # standard parameters
    nwave = wave.size
    loD = wave*1e-9/7.99 * 180/np.pi * 3600*1000/pixel

    # waffle parameters
    freq = 10 * np.sqrt(2) * 0.97
    # freq = 10 * np.sqrt(2) * 1.02
    box = 8

    if waffle_orientation == '+':
        orient = orientation_offset * np.pi / 180
    elif waffle_orientation == 'x':
        orient = orientation_offset * np.pi / 180 + np.pi / 4

    # spot fitting
    xx, yy = np.meshgrid(np.arange(2*box), np.arange(2*box))

    # multi-page PDF to save result
    if save_plot and save_path is not None:
        pdf = PdfPages(save_path)

    # loop over images
    spot_centers = np.empty((nwave, 4, 2))
    spot_dist = np.empty((nwave, 6))
    img_centers = np.empty((nwave, 2))
    spot_amplitudes = np.empty((nwave, 4))
    spot_centers[:] = np.nan
    spot_dist[:] = np.nan
    img_centers[:] = np.nan
    spot_amplitudes[:] = np.nan
    for idx, (wave, img) in enumerate(zip(wave, cube_cen)):
        if verbose:
            print('   ==> wave {0:2d}/{1:2d} ({2:4.0f} nm)'.format(idx+1, nwave, wave))

        # remove any NaN
        if mask is not None:
            mask = mask.astype('bool')
            img[mask[idx]] = np.nan

        img = np.nan_to_num(img)

        # center guess (+offset)
        cx_int = int(center_guess[idx, 0]) + center_offset[0]
        cy_int = int(center_guess[idx, 1]) + center_offset[1]

        # optional high-pass filter
        if high_pass:
            img = img - ndimage.median_filter(img, 15, mode='mirror')

        # optional smoothing
        if smooth > 0:
            img = ndimage.gaussian_filter(img, smooth)

        # create plot if needed
        if save_path is not None and save_plot:
            fig = plt.figure('Waffle center - imaging', figsize=(8.3, 8))
            plt.clf()

            # if high_pass:
            norm = colors.PowerNorm(gamma=1, vmin=-1e-1, vmax=1e-1)
            # else:
            #     norm = colors.LogNorm(vmin=1e-2, vmax=1)

            col = ['green', 'blue', 'deepskyblue', 'purple']
            ax = fig.add_subplot(111)
            ax.imshow(img/img.max(), aspect='equal', norm=norm, interpolation='nearest',
                      cmap=global_cmap)
            ax.set_title(r'Image #{0} - {1:.0f} nm'.format(idx+1, wave))
            ax.set_xlabel('x position [pix]')
            ax.set_ylabel('y position [pix]')

        # satelitte spots
        for s in range(4):
            cx = int(cx_int + freq*loD[idx] * np.cos(orient + np.pi/2*s))
            cy = int(cy_int + freq*loD[idx] * np.sin(orient + np.pi/2*s))

            sub = img[cy - box:cy + box, cx - box:cx + box].copy()
            if mask is not None:
                sub_mask = mask[idx][cy - box:cy + box, cx - box:cx + box]
            else:
                sub_mask = np.zeros_like(sub, dtype='bool')

            # bounds for fitting: spots slightly outside of the box are allowed
            gbounds = {
                'amplitude': (0.0, None),
                'x_mean': (-2.0, box*2+2),
                'y_mean': (-2.0, box*2+2),
                'x_stddev': (1.0, 20.0),
                'y_stddev': (1.0, 20.0)
            }

            # fit: Gaussian + constant
            # center_estimate = np.round(center_of_mass(sub)).astype('int')
            center_estimate = np.array(np.unravel_index(np.argmax(sub), sub.shape))
            # Check if estimated center is inside of box at all
            # if np.all(center_estimate > 0) and np.all(center_estimate < 2*box - 1):
            amplitude_estimate = sub[center_estimate[0], center_estimate[1]]
            cutout_median_flux_threshold = np.median(sub)
            if amplitude_estimate > cutout_median_flux_threshold:
                if fit_background:
                    g_init = models.Gaussian2D(amplitude=amplitude_estimate,
                                               x_mean=center_estimate[1],
                                               y_mean=center_estimate[0],
                                               x_stddev=loD[idx]/2.355,
                                               y_stddev=loD[idx]/2.355,
                                               theta=None, bounds=gbounds) + \
                        models.Const2D(amplitude=sub[~sub_mask].min())
                    if fit_symmetric_gaussian:
                        g_init.x_stddev_0.fixed = True
                        g_init.y_stddev_0.fixed = True
                        g_init.theta_0.fixed = True

                else:
                    g_init = models.Gaussian2D(amplitude=amplitude_estimate,
                                               x_mean=center_estimate[1],
                                               y_mean=center_estimate[0],
                                               x_stddev=loD[idx]/2.355,
                                               y_stddev=loD[idx]/2.355,
                                               theta=None, bounds=gbounds)
                    if fit_symmetric_gaussian:
                        g_init.x_stddev.fixed = True
                        g_init.y_stddev.fixed = True
                        g_init.theta.fixed = True
                fitter = fitting.LevMarLSQFitter()
                par = fitter(g_init, xx[~sub_mask], yy[~sub_mask], sub[~sub_mask])
                model = par(xx, yy)

                if fit_background:
                    par = par[0]

                non_deviating_mask = abs(
                    (sub - model) / sub) < deviation_threshold  # Filter out
                non_deviating_mask = np.logical_and(non_deviating_mask, ~sub_mask)

                if np.sum(non_deviating_mask) < 6:
                    spot_centers[idx, s, 0] = np.nan
                    spot_centers[idx, s, 1] = np.nan
                    spot_amplitudes[idx, s] = np.nan
                    print("Not enough pixel left after masking deviating pixels for spot: {}.".format(
                        s))
                    continue
                if mask_deviating:
                    g_init = models.Gaussian2D(
                        amplitude=par.amplitude,
                        x_mean=par.x_mean.value,
                        y_mean=par.y_mean.value,
                        x_stddev=par.x_stddev.value,
                        y_stddev=par.y_stddev.value)
                    g_init.x_stddev.fixed = True
                    g_init.y_stddev.fixed = True
                    g_init.theta.fixed = True
                    # ipsh()
                    par = fitter(g_init, xx[non_deviating_mask],
                                 yy[non_deviating_mask], sub[non_deviating_mask])
                    model = par(xx, yy)

                if idx == 1:
                    if verbose:
                        print(par)
                if fit_symmetric_gaussian:
                    par_gaussian = par
                    # par_gaussian.x_mean = par_gaussian.x_0
                    # par_gaussian.y_mean = par_gaussian.y_0
                else:
                    par_gaussian = par[0]

                cx_final = cx - box + par_gaussian.x_mean
                cy_final = cy - box + par_gaussian.y_mean

                spot_centers[idx, s, 0] = cx_final
                spot_centers[idx, s, 1] = cy_final
                spot_amplitudes[idx, s] = par.amplitude[0]

                # plot sattelite spots and fit
                if save_path is not None and save_plot:
                    ax.plot([cx_final], [cy_final], marker='D', color=col[s], zorder=1000)
                    ax.add_patch(patches.Rectangle((cx-box, cy-box),
                                 2*box, 2*box, ec='white', fc='none'))

                    axs = fig.add_axes((0.17+s*0.2, 0.17, 0.1, 0.1))
                    axs.imshow(sub, aspect='equal', vmin=0, vmax=sub.max(), interpolation='nearest',
                               cmap=global_cmap)
                    axs.plot([par_gaussian.x_mean.value], [
                             par_gaussian.y_mean.value], marker='D', color=col[s])
                    axs.set_xticks([])
                    axs.set_yticks([])

                    axs = fig.add_axes((0.17+s*0.2, 0.06, 0.1, 0.1))
                    axs.imshow(model, aspect='equal', vmin=0, vmax=sub.max(), interpolation='nearest',
                               cmap=global_cmap)
                    axs.set_xticks([])
                    axs.set_yticks([])
            else:
                spot_centers[idx, s, 0] = np.nan
                spot_centers[idx, s, 1] = np.nan
                spot_amplitudes[idx, s] = np.nan
                print("Center of light outside of sub-image and/or too small value at estimated center position.")
        # lines intersection
        intersect = lines_intersect(spot_centers[idx, 0, :], spot_centers[idx, 2, :],
                                    spot_centers[idx, 1, :], spot_centers[idx, 3, :])
        img_centers[idx] = intersect

        # scaling
        spot_dist[idx, 0] = np.sqrt(np.sum((spot_centers[idx, 0, :] - spot_centers[idx, 2, :])**2))
        spot_dist[idx, 1] = np.sqrt(np.sum((spot_centers[idx, 1, :] - spot_centers[idx, 3, :])**2))
        spot_dist[idx, 2] = np.sqrt(np.sum((spot_centers[idx, 0, :] - spot_centers[idx, 1, :])**2))
        spot_dist[idx, 3] = np.sqrt(np.sum((spot_centers[idx, 0, :] - spot_centers[idx, 3, :])**2))
        spot_dist[idx, 4] = np.sqrt(np.sum((spot_centers[idx, 1, :] - spot_centers[idx, 2, :])**2))
        spot_dist[idx, 5] = np.sqrt(np.sum((spot_centers[idx, 2, :] - spot_centers[idx, 3, :])**2))

        # finalize plot
        if save_path is not None and save_plot and np.all(np.isfinite(intersect)):
            ax.plot([spot_centers[idx, 0, 0], spot_centers[idx, 2, 0]],
                    [spot_centers[idx, 0, 1], spot_centers[idx, 2, 1]],
                    color='w', linestyle='dashed', zorder=900)
            ax.plot([spot_centers[idx, 1, 0], spot_centers[idx, 3, 0]],
                    [spot_centers[idx, 1, 1], spot_centers[idx, 3, 1]],
                    color='w', linestyle='dashed', zorder=900)

            ax.plot([intersect[0]], [intersect[1]], marker='+', color='w', ms=15)

            ext = 1000 / pixel
            ax.set_xlim(intersect[0]-ext, intersect[0]+ext)
            ax.set_ylim(intersect[1]-ext, intersect[1]+ext)

            plt.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.95)
            if save_plot and save_path is not None:
                pdf.savefig()
            # plt.savefig(os.path.splitext(save_path)[0]+'.png', dpi=300)

    if save_plot and save_path is not None:
        pdf.close()

    return spot_centers, spot_dist, img_centers, spot_amplitudes


def measure_center_waffle(cube, outputdir, instrument,
                          bpm_cube=None, wavelengths=None,
                          waffle_orientation=None,
                          frames_info=None,
                          center_guess=None,
                          crop=False,
                          crop_center=((480, 525), (483, 511)),
                          fit_background=True,
                          fit_symmetric_gaussian=False,
                          mask_deviating=True,
                          deviation_threshold=0.8,
                          high_pass=False,
                          save_plot=True,
                          save_path=None,
                          verbose=False):
    
    # Detect if in single-frame mode and reshape
    single_frame_mode = cube.ndim == 3
    if single_frame_mode:
        cube = cube[:, np.newaxis, :, :]
        if frames_info is not None:
            frames_info = frames_info[:1]
    
    spot_centers = []
    spot_distances = []
    image_centers = []
    spot_amplitudes = []

    for i in range(cube.shape[1]):
        if verbose:
            print("Frame: {}".format(i))
        if waffle_orientation is None and frames_info is not None:
            row = frames_info.iloc[i]
            waffle_orientation = row['OCS WAFFLE ORIENT']
        data = cube[:, i]  # fits.getdata(frames_info['FILE'][i])

        if save_path is not None:
            # Use external file path if provided
            plot_path = save_path
        else:
            plot_path = os.path.join(outputdir, f'CENTER_img_{i:03d}.pdf')

        if instrument == 'IRDIS':
            if wavelengths is None and frames_info is not None:
                wavelengths = np.array(transmission.wavelength_bandwidth_filter(
                    frames_info['INS COMB IFLT'][i])[0])
            pixel = 12.25
            orientation_offset = 0
            if center_guess is None:
                K_band_guess = np.array(((480, 524.7), (482.5, 511.4)))
                H_band_guess = np.array(((485.81, 523.54), (487.95, 514.36)))
                if np.max(wavelengths) > 2000:  # K band center
                    center_guess = K_band_guess  # DB_K12
                else:  # H band center
                    center_guess = H_band_guess  # DB_H23
        if instrument == 'IFS':
            pixel = 7.46
            orientation_offset = 102
            if center_guess is None:
                center_guess = np.array([128, 128])[None, :].repeat(cube.shape[0], axis=0)
        else:
            raise ValueError('Only IRDIS and IFS instruments known.')

        if crop:
            crop_center_orig = np.array(crop_center)
            box_center = np.array((data.shape[-2] // 2, data.shape[-1] // 2))
            center_offset = center_guess - crop_center_orig
            center_guess = box_center + center_offset

        spot_center, spot_distance, img_center, spot_amplitude = star_centers_from_waffle_img_cube(
            data, wave=wavelengths,
            waffle_orientation=waffle_orientation,
            mask=bpm_cube,  # TO BE ADDED TO FUNCTION
            center_guess=center_guess,
            pixel=pixel,
            orientation_offset=orientation_offset,  # CHECK IF THIS IS NONZERO FOR IRDIS
            fit_background=fit_background,
            fit_symmetric_gaussian=fit_symmetric_gaussian,
            mask_deviating=mask_deviating,
            deviation_threshold=deviation_threshold,
            high_pass=high_pass,
            center_offset=(0, 0),
            smooth=0, coro=True,
            save_plot=save_plot,
            save_path=plot_path,
            verbose=verbose)

        spot_centers.append(spot_center)
        spot_distances.append(spot_distance)
        image_centers.append(img_center)
        spot_amplitudes.append(spot_amplitude)

        plt.close()

    spot_centers = np.swapaxes(np.array(spot_centers), 0, 1)
    spot_distances = np.swapaxes(np.array(spot_distances), 0, 1)
    image_centers = np.swapaxes(np.array(image_centers), 0, 1)
    spot_amplitudes = np.swapaxes(np.array(spot_amplitudes), 0, 1)

    return spot_centers, spot_distances, \
        image_centers, spot_amplitudes


def _fit_center_for_cube(args) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Worker function to process a single frame and extract waffle spot centers.
    Returns:
        index (int): original frame index
        spot_centers, spot_distances, image_centers, spot_amplitudes (arrays)
    """
    index, frame_cube, frame_info, wavelengths, plot_dir, fit_background, instrument = args
    assert frame_cube.ndim == 3, f"Expected (n_waves, H, W), got {frame_cube.shape}"
    
    plot_path = os.path.join(plot_dir, f'CENTER_img_{index:03d}.pdf')
    frame_cube = frame_cube[:, np.newaxis, :, :]  # shape: (n_wavelengths, 1, H, W)

    spot_centers, spot_distances, image_centers, spot_amplitudes = measure_center_waffle(
        cube=frame_cube,
        wavelengths=wavelengths,
        waffle_orientation=None,
        frames_info=frame_info,
        bpm_cube=None,
        outputdir=plot_dir,
        instrument=instrument,
        crop=False,
        crop_center=None,
        fit_background=fit_background,
        fit_symmetric_gaussian=True,
        high_pass=False,
        save_plot=True,
        save_path=plot_path,
    )

    return index, (spot_centers, spot_distances, image_centers, spot_amplitudes)


def fit_centers_in_parallel(converted_dir: str, observation, overwrite: bool = True, ncpu: int = 4):
    """
    Measures waffle spot centers across an IFS cube in parallel using multiprocessing.
    
    Saves results as FITS files:
        - spot_centers.fits
        - spot_distances.fits
        - image_centers.fits
        - spot_fit_amplitudes.fits
    """
    center_cube = fits.getdata(os.path.join(converted_dir, 'center_cube.fits'))
    wavelengths = fits.getdata(os.path.join(converted_dir, 'wavelengths.fits'))
    frame_info_center = pd.read_csv(os.path.join(converted_dir, 'frames_info_center.csv'))

    plot_dir = os.path.join(converted_dir, 'center_plots/')
    os.makedirs(plot_dir, exist_ok=True)

    fit_background = len(observation.frames['CORO']) == 0
    n_frames = center_cube.shape[1]

    # Prepare args for each frame
    args_list = [
        (
            i,
            center_cube[:, i, :, :],               # single frame, shape: (n_waves, H, W)
            frame_info_center[i:i+1],            # Single-row Table
            wavelengths,
            plot_dir,
            fit_background,
            'IFS'
        )
        for i in range(n_frames)
    ]

    # Process each frame in parallel and preserve order
    results = parallel_map_ordered(
        func=_fit_center_for_cube,
        args_list=args_list,
        ncpu=ncpu,
        desc="Measuring centers"
    )

    # Unpack results
    spot_centers_list, spot_distances_list, image_centers_list, spot_amplitudes_list = zip(*results)

    spot_centers = np.concatenate(spot_centers_list, axis=1)
    spot_distances = np.concatenate(spot_distances_list, axis=1)
    image_centers = np.concatenate(image_centers_list, axis=1)
    spot_fit_amplitudes = np.concatenate(spot_amplitudes_list, axis=1)

    # Write outputs
    fits.writeto(os.path.join(converted_dir, 'spot_centers.fits'), spot_centers, overwrite=overwrite)
    fits.writeto(os.path.join(converted_dir, 'spot_distances.fits'), spot_distances, overwrite=overwrite)
    fits.writeto(os.path.join(converted_dir, 'image_centers.fits'), image_centers, overwrite=overwrite)
    fits.writeto(os.path.join(converted_dir, 'spot_fit_amplitudes.fits'), spot_fit_amplitudes, overwrite=overwrite)


def star_centers_from_PSF_img_cube(cube, wave, pixel, guess_center_yx=None,
                                   box_size=30,
                                   fit_background=False, fit_symmetric_gaussian=True,
                                   mask_deviating=True, deviation_threshold=0.8,
                                   edge_exclude_fraction=0.1,
                                   mask=None, save_path=None,
                                   verbose=False):
    """
    Compute the star center in each frame of a PSF image cube using 2D Gaussian fitting.

    This function fits a 2D Gaussian (with optional constant background) to estimate
    the centroid of a stellar PSF in each frame of a spectral image cube (e.g., IRDIS CI, DBI, or IFS data).
    It supports masking bad pixels, removing outlier pixels during fitting, and handling cases
    where the PSF peak is near the edge of the image.

    Parameters
    ----------
    cube : array_like, shape (nwave, ny, nx)
        PSF image cube, with one image per wavelength channel.

    wave : array_like, shape (nwave,)
        Wavelength values for each frame, in nanometers.

    pixel : float
        Pixel scale in milliarcseconds per pixel (mas/pixel).

    guess_center_yx : tuple of int, optional
        (y, x) coordinates of the initial guess for the PSF center. If None, the center is
        automatically estimated by locating the brightest pixel while avoiding the image edges
        (see `edge_exclude_fraction`).

    box_size : int, optional
        Half-size of the square sub-image used for fitting (default is 30, resulting in a 60×60 cutout).

    fit_background : bool, optional
        Whether to include a constant background level in the Gaussian fit.

    fit_symmetric_gaussian : bool, optional
        If True, the Gaussian fit is constrained to be circular (equal stddev in x and y, and no rotation).

    mask_deviating : bool, optional
        If True, pixels that deviate significantly from the model in the first fit are masked and
        the fit is repeated.

    deviation_threshold : float, optional
        Threshold on relative deviation (|residual/model|) used for masking deviating pixels.

    edge_exclude_fraction : float, optional
        Fraction of the image borders to exclude when guessing the center position
        in the absence of a user-provided guess (default is 0.1 = 10%).

    mask : array_like of bool, optional
        Boolean mask array with same shape as `cube`, where True indicates bad pixels to exclude from fitting.

    save_path : str, optional
        Path to save a multi-page PDF with diagnostic plots. If None, no plots are saved.

    Returns
    -------
    image_centers : ndarray, shape (nwave, 2)
        Array of fitted star center positions for each frame, in (x, y) pixel coordinates.

    amplitudes : ndarray, shape (nwave,)
        Fitted peak amplitude of the Gaussian for each frame.

    Notes
    -----
    - The function uses a Levenberg-Marquardt least squares fitter.
    - If `fit_symmetric_gaussian` is True, standard deviations and angle of the Gaussian are fixed.
    - If the fitting fails or the mask removes too many pixels, NaNs are returned for that frame.
    """

    # standard parameters
    nwave = wave.size
    loD = wave*1e-9/7.99 * 180/np.pi * 3600*1000/pixel
    box = box_size

    # spot fitting
    xx, yy = np.meshgrid(np.arange(2 * box), np.arange(2 * box))

    # multi-page PDF to save result
    if save_path is not None:
        pdf = PdfPages(save_path)

    # loop over wavelengths
    image_centers = np.empty((nwave, 2))
    amplitudes = np.empty(nwave)
    image_centers[:] = np.nan
    amplitudes[:] = np.nan

    for idx, (wave, img) in enumerate(zip(wave, cube)):
        if verbose:
            print('   ==> wave {0:2d}/{1:2d} ({2:4.0f} nm)'.format(idx+1, nwave, wave))

        # remove any NaN

        if mask is not None:
            mask = mask.astype('bool')
            img[mask[idx]] = np.nan

        bad_mask = np.logical_or(~np.isfinite(img), img == 0.)

        img = np.nan_to_num(img)

        # center guess
        # center guess
        if guess_center_yx is None:
            cy, cx = np.unravel_index(np.argmax(img), img.shape)

            # check if we are really too close to the edge
            dim = img.shape
            lf = edge_exclude_fraction
            hf = 1 - edge_exclude_fraction
            if (cx <= lf*dim[-1]) or (cx >= hf*dim[-1]) or \
            (cy <= lf*dim[0]) or (cy >= hf*dim[0]):
                nimg = img.copy()
                nimg[:, :int(lf*dim[-1])] = 0
                nimg[:, int(hf*dim[-1]):] = 0
                nimg[:int(lf*dim[0]), :] = 0
                nimg[int(hf*dim[0]):, :] = 0

                cy, cx = np.unravel_index(np.argmax(nimg), img.shape)
        else:
            cy, cx = guess_center_yx

        sub = img[cy - box:cy + box, cx - box:cx + box].copy()
        if mask is not None:
            sub_mask = mask[idx][cy - box:cy + box, cx - box:cx + box]
            sub_mask = np.logical_or(sub_mask, bad_mask)
        else:
            sub_mask = bad_mask[cy - box:cy + box, cx - box:cx + box]
            sub = img[cy - box:cy + box, cx - box:cx + box].copy()

            # bounds for fitting: spots slightly outside of the box are allowed
            gbounds = {
                'amplitude': (0.0, None),
                'x_mean': (-2.0, box*2+2),
                'y_mean': (-2.0, box*2+2),
                'x_stddev': (0.3, 20.0),
                'y_stddev': (0.3, 20.0)
            }

            # fit: Gaussian + constant
            # center_estimate2 = np.round(center_of_mass(sub)).astype('int')
            center_estimate = np.array(np.unravel_index(np.argmax(sub), sub.shape))
            if np.all(center_estimate > 0) and np.all(center_estimate < 2*box - 1):
                amplitude_estimate = sub[center_estimate[0], center_estimate[1]]
                cutout_median_flux_threshold = np.median(sub)
                if amplitude_estimate > cutout_median_flux_threshold:
                    if fit_background:
                        g_init = models.Gaussian2D(amplitude=amplitude_estimate,
                                                   x_mean=center_estimate[1],
                                                   y_mean=center_estimate[0],
                                                   x_stddev=loD[idx]/2.355,
                                                   y_stddev=loD[idx]/2.355,
                                                   theta=None, bounds=gbounds) + \
                            models.Const2D(amplitude=sub[~sub_mask].min())
                        if fit_symmetric_gaussian:
                            g_init.x_stddev_0.fixed = True
                            g_init.y_stddev_0.fixed = True
                            g_init.theta_0.fixed = True

                    else:
                        g_init = models.Gaussian2D(amplitude=amplitude_estimate,
                                                   x_mean=center_estimate[1],
                                                   y_mean=center_estimate[0],
                                                   x_stddev=loD[idx]/2.355,
                                                   y_stddev=loD[idx]/2.355,
                                                   theta=None, bounds=gbounds)
                        # g_init = models.Moffat2D(amplitude=sub.max(),
                        #                          x_0=imax[1],
                        #                          y_0=imax[0],
                        #                          gamma=loD[idx]/2.355,
                        #                          alpha=1
                        #                          )
                        if fit_symmetric_gaussian:
                            g_init.x_stddev.fixed = True
                            g_init.y_stddev.fixed = True
                            g_init.theta.fixed = True
                    fitter = fitting.LevMarLSQFitter()
                    par = fitter(g_init, xx[~sub_mask], yy[~sub_mask], sub[~sub_mask])
                    model = par(xx, yy)

                    non_deviating_mask = abs(
                        (sub - model) / model) < deviation_threshold  # Filter out
                    non_deviating_mask = np.logical_and(non_deviating_mask, ~sub_mask)
                    if np.sum(non_deviating_mask) < 6:
                        image_centers[idx, 0] = np.nan
                        image_centers[idx, 1] = np.nan
                        amplitudes[idx] = np.nan
                        print("Not enough pixel left after masking deviating pixels for PSF: {}.")
                        continue

                    if mask_deviating:
                        if fit_background:
                            g_init = models.Gaussian2D(amplitude=par[0].amplitude.value,
                                                       x_mean=par[0].x_mean.value,
                                                       y_mean=par[0].y_mean.value,
                                                       x_stddev=par[0].x_stddev.value,
                                                       y_stddev=par[0].y_stddev.value,
                                                       theta=None, bounds=gbounds) + \
                                models.Const2D(amplitude=par[1].amplitude.value)
                            if fit_symmetric_gaussian:
                                g_init.x_stddev_0.fixed = True
                                g_init.y_stddev_0.fixed = True
                                g_init.theta_0.fixed = True
                        else:
                            g_init = models.Gaussian2D(
                                amplitude=par.amplitude.value,
                                x_mean=par.x_mean.value,
                                y_mean=par.y_mean.value,
                                x_stddev=par.x_stddev.value,
                                y_stddev=par.y_stddev.value)
                            if fit_symmetric_gaussian:
                                g_init.x_stddev.fixed = True
                                g_init.y_stddev.fixed = True
                                g_init.theta.fixed = True

                        par = fitter(g_init, xx[non_deviating_mask],
                                     yy[non_deviating_mask], sub[non_deviating_mask])
                        model = par(xx, yy)
                    if idx == 1:
                        if verbose:
                            print(par)

                    if fit_symmetric_gaussian:
                        par_gaussian = par
                        # par_gaussian.x_mean = par_gaussian.x_0
                        # par_gaussian.y_mean = par_gaussian.y_0
                    else:
                        if fit_background:
                            par_gaussian = par[0]
                        else:
                            par_gaussian = par
                    cx_final = cx - box + par_gaussian.x_mean
                    cy_final = cy - box + par_gaussian.y_mean

                    image_centers[idx, 0] = cx_final
                    image_centers[idx, 1] = cy_final
                    amplitudes[idx] = par_gaussian.amplitude[0]
            else:
                image_centers[idx, 0] = np.nan
                image_centers[idx, 1] = np.nan
                amplitudes[idx] = np.nan

        if save_path:
            plt.figure('PSF center - imaging', figsize=(8.3, 8))
            plt.clf()

            plt.subplot(111)
            plt.imshow(img/img.max(), aspect='equal', vmin=1e-6, vmax=1, norm=colors.LogNorm(),
                       interpolation='nearest', cmap=global_cmap)
            plt.plot([cx_final], [cy_final], marker='D', color='blue')
            plt.gca().add_patch(patches.Rectangle((cx-box, cy-box), 2*box, 2*box, ec='white', fc='none'))
            plt.title(r'Image #{0} - {1:.0f} nm'.format(idx+1, wave))

            ext = 1000 / pixel
            plt.xlim(cx_final-ext, cx_final+ext)
            plt.xlabel('x position [pix]')
            plt.ylim(cy_final-ext, cy_final+ext)
            plt.ylabel('y position [pix]')

            plt.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.95)

            pdf.savefig()

    if save_path:
        pdf.close()

    return image_centers, amplitudes