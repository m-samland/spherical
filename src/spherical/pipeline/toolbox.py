import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.nddata import Cutout2D
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import shift

global_cmap = 'inferno'


def check_recipe_execution(recipe_execution, recipe_name, recipe_requirements):
    '''
    Check execution of previous recipes for a given recipe.

    Parameters
    ----------
    recipe_execution : dict
        Status of executed recipes

    recipe_name : str
        Name of the current recipe

    recipe_requirements : dict
        Dictionary providing the recipe requirements

    Returns
    -------
    execute_recipe : bool
        Current recipe can be executed safely
    '''
    requirements = recipe_requirements[recipe_name]

    execute_recipe = True
    missing = []
    for r in requirements:
        if not recipe_execution[r]:
            execute_recipe = False
            missing.append(r)

    if not execute_recipe:
        raise ValueError('{0} cannot executed because some files have been '.format(recipe_name) +
                         'removed from the reduction directory ' +
                         'or the following recipes have not been executed: {0}. '.format(missing))

    return execute_recipe


def compute_bad_pixel_map(bpm_files, dtype=np.uint8):
    '''
    Compute a combined bad pixel map provided a list of files

    Parameters
    ----------
    bpm_files : list
        List of names for the bpm files

    dtype : data type
        Data type for the final bpm

    Returns
    bpm : array_like
        Combined bad pixel map
    '''

    # check that we have files
    if len(bpm_files) == 0:
        raise ValueError('No bad pixel map files provided')

    # get shape
    shape = fits.getdata(bpm_files[0]).shape

    # star with empty bpm
    bpm = np.zeros((shape[-2], shape[-1]), dtype=np.uint8)

    # fill if files are provided
    for f in bpm_files:
        data = fits.getdata(f)
        bpm = np.logical_or(bpm, data)

    bpm = bpm.astype(dtype)

    return bpm


def collapse_frames_info(finfo, fname, collapse_type, coadd_value=2):
    '''
    Collapse frame info to match the collapse operated on the data

    Parameters
    ----------
    finfo : dataframe
        The data frame with all the information on science frames

    fname : str
       The name of the current file

    collapse_type : str
        Type of collapse. Possible values are mean or coadd. Default
        is mean.

    coadd_value : int
        Number of consecutive frames to be coadded when collapse_type
        is coadd. Default is 2

    Returns
    -------
    nfinfo : dataframe
        Collapsed data frame, or None in case of error
    '''

    # logger.info('   ==> collapse frames information')
    print('   ==> collapse frames information')

    nfinfo = None
    if collapse_type == 'none':
        nfinfo = finfo
        # logger.debug('> type=none: copy input data frame')
        print('> type=none: copy input data frame')
    elif collapse_type == 'mean':
        index = pd.MultiIndex.from_arrays([[fname], [0]], names=['FILE', 'IMG'])
        nfinfo = pd.DataFrame(columns=finfo.columns, index=index, dtype='float')

        # logger.debug('> type=mean: extract min/max values')
        print('> type=mean: extract min/max values')

        # get min/max indices
        imin = finfo.index.get_level_values(1).min()
        imax = finfo.index.get_level_values(1).max()

        # copy data
        nfinfo.loc[(fname, 0)] = finfo.loc[(fname, imin)]

        # update time values
        nfinfo.loc[(fname, 0), 'DET NDIT'] = 1
        nfinfo.loc[(fname, 0), 'TIME START'] = finfo.loc[(fname, imin), 'TIME START']
        nfinfo.loc[(fname, 0), 'TIME END'] = finfo.loc[(fname, imax), 'TIME END']
        nfinfo.loc[(fname, 0), 'TIME'] = finfo.loc[(fname, imin), 'TIME START'] + \
            (finfo.loc[(fname, imax), 'TIME END'] - finfo.loc[(fname, imin), 'TIME START']) / 2

        # recompute angles
        # ret = compute_angles(nfinfo)
        # if ret == vltpf.ERROR:
        #     return None
    elif collapse_type == 'coadd':
        coadd_value = int(coadd_value)
        NDIT = len(finfo)
        NDIT_new = NDIT // coadd_value

        # logger.debug('> type=coadd: extract sub-groups of {} frames'.format(coadd_value))
        print('> type=coadd: extract sub-groups of {} frames'.format(coadd_value))

        index = pd.MultiIndex.from_arrays(
            [np.full(NDIT_new, fname), np.arange(NDIT_new)], names=['FILE', 'IMG'])
        nfinfo = pd.DataFrame(columns=finfo.columns, index=index, dtype='float')

        for f in range(NDIT_new):
            # get min/max indices
            imin = int(f*coadd_value)
            imax = int((f+1)*coadd_value-1)

            # copy data
            nfinfo.loc[(fname, f)] = finfo.loc[(fname, imin)]

            # update time values
            nfinfo.loc[(fname, f), 'DET NDIT'] = 1
            nfinfo.loc[(fname, f), 'TIME START'] = finfo.loc[(fname, imin), 'TIME START']
            nfinfo.loc[(fname, f), 'TIME END'] = finfo.loc[(fname, imax), 'TIME END']
            nfinfo.loc[(fname, f), 'TIME'] = finfo.loc[(fname, imin), 'TIME START'] + \
                (finfo.loc[(fname, imax), 'TIME END'] - finfo.loc[(fname, imin), 'TIME START']) / 2

        # recompute angles
        # ret = compute_angles(nfinfo)
        # if ret == vltpf.ERROR:
        #     return None
    else:
        # logger.error('Unknown collapse type {0}'.format(collapse_type))
        print('Unknown collapse type {0}'.format(collapse_type))
        return None

    return nfinfo


def collapse_frames_info_spherical(finfo, fname, collapse_type, coadd_value=2):
    '''
    Collapse frame info to match the collapse operated on the data

    Parameters
    ----------
    finfo : dataframe
        The data frame with all the information on science frames

    fname : str
       The name of the current file

    collapse_type : str
        Type of collapse. Possible values are mean or coadd. Default
        is mean.

    coadd_value : int
        Number of consecutive frames to be coadded when collapse_type
        is coadd. Default is 2

    Returns
    -------
    nfinfo : dataframe
        Collapsed data frame, or None in case of error
    '''

    # logger.info('   ==> collapse frames information')
    print('   ==> collapse frames information')

    nfinfo = None
    if collapse_type == 'none':
        nfinfo = finfo
        # logger.debug('> type=none: copy input data frame')
        print('> type=none: copy input data frame')
    elif collapse_type == 'mean':
        index = pd.MultiIndex.from_arrays([[fname], [0]], names=['FILE', 'IMG'])
        nfinfo = pd.DataFrame(columns=finfo.columns, index=index, dtype='float')

        # logger.debug('> type=mean: extract min/max values')
        print('> type=mean: extract min/max values')

        # get min/max indices
        imin = finfo.index.get_level_values(1).min()
        imax = finfo.index.get_level_values(1).max()

        # copy data
        nfinfo.loc[(fname, 0)] = finfo.loc[(fname, imin)]

        # update time values
        nfinfo.loc[(fname, 0), 'DET NDIT'] = 1
        nfinfo.loc[(fname, 0), 'TIME START'] = finfo.loc[(fname, imin), 'TIME START']
        nfinfo.loc[(fname, 0), 'TIME END'] = finfo.loc[(fname, imax), 'TIME END']
        nfinfo.loc[(fname, 0), 'TIME'] = finfo.loc[(fname, imin), 'TIME START'] + \
            (finfo.loc[(fname, imax), 'TIME END'] - finfo.loc[(fname, imin), 'TIME START']) / 2

        # recompute angles
        # ret = compute_angles(nfinfo)
        # if ret == vltpf.ERROR:
        #     return None
    elif collapse_type == 'coadd':
        coadd_value = int(coadd_value)
        NDIT = len(finfo)
        NDIT_new = NDIT // coadd_value

        # logger.debug('> type=coadd: extract sub-groups of {} frames'.format(coadd_value))
        print('> type=coadd: extract sub-groups of {} frames'.format(coadd_value))

        index = pd.MultiIndex.from_arrays(
            [np.full(NDIT_new, fname), np.arange(NDIT_new)], names=['FILE', 'IMG'])
        nfinfo = pd.DataFrame(columns=finfo.columns, index=index, dtype='float')

        for f in range(NDIT_new):
            # get min/max indices
            imin = int(f*coadd_value)
            imax = int((f+1)*coadd_value-1)

            # copy data
            nfinfo.loc[(fname, f)] = finfo.loc[(fname, imin)]

            # update time values
            nfinfo.loc[(fname, f), 'DET NDIT'] = 1
            nfinfo.loc[(fname, f), 'TIME START'] = finfo.loc[(fname, imin), 'TIME START']
            nfinfo.loc[(fname, f), 'TIME END'] = finfo.loc[(fname, imax), 'TIME END']
            nfinfo.loc[(fname, f), 'TIME'] = finfo.loc[(fname, imin), 'TIME START'] + \
                (finfo.loc[(fname, imax), 'TIME END'] - finfo.loc[(fname, imin), 'TIME START']) / 2

        # recompute angles
        # ret = compute_angles(nfinfo)
        # if ret == vltpf.ERROR:
        #     return None
    else:
        # logger.error('Unknown collapse type {0}'.format(collapse_type))
        print('Unknown collapse type {0}'.format(collapse_type))
        return None

    return nfinfo





def star_centers_from_PSF_img_cube(cube, wave, pixel, guess_center_yx=None,
                                   box_size=30,
                                   fit_background=False, fit_symmetric_gaussian=True,
                                   mask_deviating=True, deviation_threshold=0.8,
                                   edge_exclude_fraction=0.1,
                                   mask=None, save_path=None):
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
        # sub-image
        # ipsh()
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


# def star_centers_from_waffle_cube(cube, wave, instrument, waffle_orientation,
#                                   center_guess=None,
#                                   mask=None, high_pass=False, center_offset=(0, 0),
#                                   smooth=0, coro=True, display=False, save_path=None):
#     '''
#     Compute star center from waffle images
#
#     Parameters
#     ----------
#     cube : array_like
#         Waffle IRDIS cube
#
#     wave : array_like
#         Wavelength values, in nanometers
#
#     instrument : str
#         Instrument, IFS or IRDIS
#
#     waffle_orientation : str
#         String giving the waffle orientation '+' or 'x'
#
#     high_pass : bool
#         Apply high-pass filter to the image before searching for the
#         satelitte spots
#
#     smooth : int
#         Apply a gaussian smoothing to the images to reduce noise. The
#         value is the sigma of the gaussian in pixel.  Default is no
#         smoothing
#
#     center_offset : tuple
#         Apply an (x,y) offset to the default center position. Default is no offset
#
#     coro : bool
#         Observation was performed with a coronagraph. Default is True
#
#     display : bool
#         Display the fit of the satelitte spots
#
#     save_path : str
#         Path where to save the fit images
#
#     Returns
#     -------
#     spot_center : array_like
#         Centers of each individual spot in each frame of the cube
#
#     spot_dist : array_like
#         The 6 possible distances between the different spots
#
#     img_center : array_like
#         The star center in each frame of the cube
#
#     '''
#
#     # instrument
#     if instrument == 'IFS':
#         pixel = 7.46
#         offset = 102
#     elif instrument == 'IRDIS':
#         pixel = 12.25
#         offset = 0
#     else:
#         raise ValueError('Unknown instrument {0}'.format(instrument))
#
#     # standard parameters
#     dim = cube.shape[-1]
#     nwave = wave.size
#     loD = wave * 1e-6 / 8.0 * 180 / np.pi * 3600 * 1000 / pixel
#
#     # waffle parameters
#     # freq = 10 * np.sqrt(2) * 0.97
#     freq = 10 * np.sqrt(2) * 1.02
#
#     box = 8
#     if waffle_orientation == '+':
#         orient = offset * np.pi / 180
#     elif waffle_orientation == 'x':
#         orient = offset * np.pi / 180 + np.pi / 4
#
#     # spot fitting
#     xx, yy = np.meshgrid(np.arange(2 * box), np.arange(2 * box))
#
#     # multi-page PDF to save result
#     if save_path is not None:
#         pdf = PdfPages(save_path)
#
#     # center guess
#     if instrument == 'IFS':
#         center_guess = np.full((nwave, 2), ((dim // 2) + 3, (dim // 2) - 1))
#     elif instrument == 'IRDIS':
#         #     # minus coro frame
#         #     # minus_coro = np.array([[479.9350693, 524.68582873], [482.48139022, 511.34742632]])
#         #     # high pass
#         #     # high_pass = np.array([[479.92802042, 524.69353534], [482.47134037, 511.34879875]])
#         #     # without high pass
#         #     # wo_high_pass = np.array([[479.94126589, 524.69235863], [482.48647687, 511.35681847]])
#         #     # center_guess = np.array(((485, 520), (486, 508)))
#         #     # center_guess = np.array(((480, 525), (483, 511)))
#         if center_guess is None:
#             if np.max(wave) > 2:  # K band center
#                 center_guess = np.array(((481.5, 524.7), (482.5, 511.4)))  # DB_K12
#             else:  # H band center
#                 center_guess = np.array([[485.81, 523.54], [487.95, 514.36]])  # DB_H23
#
#         # loop over images
#     spot_center = np.zeros((nwave, 4, 2))
#     spot_dist = np.zeros((nwave, 6))
#     img_center = np.zeros((nwave, 2))
#     spot_amplitude = np.zeros((nwave, 4))
#
#     for idx, (wave, img) in enumerate(zip(wave, cube)):
#         print('  wave {0:2d}/{1:2d} ({2:.3f} micron)'.format(idx + 1, nwave, wave))
#
#         # remove any NaN
#         if mask is not None:
#             mask = mask.astype('bool')
#             img[mask[idx]] = np.nan
#
#         img = np.nan_to_num(img)
#
#         # center guess (+offset)
#         cx_int = int(center_guess[idx, 0]) + center_offset[0]
#         cy_int = int(center_guess[idx, 1]) + center_offset[1]
#
#         # optional high-pass filter
#         if high_pass:
#             img = img - ndimage.median_filter(img, 15, mode='mirror')
#
#         # optional smoothing
#         if smooth > 0:
#             img = ndimage.gaussian_filter(img, smooth)
#
#         # mask for non-coronagraphic observations
#         # if not coro:
#         #     mask = aperture.disc(cube[0].shape[-1], 5 * loD[idx], diameter=False,
#         #                          center=(cx_int, cy_int), invert=True)
#         #     img *= mask
#
#         # create plot if needed
#         if save_path or display:
#             fig = plt.figure(0, figsize=(8, 8))
#             plt.clf()
#             col = ['red', 'blue', 'magenta', 'purple']
#             ax = fig.add_subplot(111)
#             ax.imshow(img / img.max(), aspect='equal', vmin=1e-2, vmax=1, norm=colors.LogNorm())
#             ax.set_title(r'Image #{0} - {1:.3f} $\mu$m'.format(idx + 1, wave))
#
#         # satelitte spots
#         for s in range(4):
#             cx = int(cx_int + freq * loD[idx] * np.cos(orient + np.pi / 2 * s))
#             cy = int(cy_int + freq * loD[idx] * np.sin(orient + np.pi / 2 * s))
#
#             spot_angle = orient + np.pi / 2 * s
#             sub = img[cy - box:cy + box, cx - box:cx + box].copy()
#
#             # bounds for fitting: spots slightly outside of the box are allowed
#             gbounds = {
#                 'amplitude': (0.0, None),
#                 'x_mean': (-2.0, box*2+2),
#                 'y_mean': (-2.0, box*2+2),
#                 'x_stddev': (1.0, 20.0),
#                 'y_stddev': (1.0, 20.0)
#             }
#
#             if mask is not None:
#                 sub_mask = mask[idx][cy - box:cy + box, cx - box:cx + box]
#             else:
#                 sub_mask = np.zeros_like(sub, dtype='bool')
#             # fit: Gaussian + constant
#
#             imax = np.unravel_index(np.argmax(sub), sub.shape)
#             g_init = models.Gaussian2D(amplitude=sub.max(), x_mean=imax[1], y_mean=imax[0],
#                                        x_stddev=loD[idx], y_stddev=loD[idx],
#                                        theta=spot_angle, bounds=gbounds) + \
#                 models.Const2D(amplitude=sub.min())
#             fitter = fitting.LevMarLSQFitter()
#             par = fitter(g_init, xx[~sub_mask], yy[~sub_mask], sub[~sub_mask])
#             fit = par(xx, yy)
#             if idx == 0:
#                 print(par)
#             cx_final = cx - box + par[0].x_mean
#             cy_final = cy - box + par[0].y_mean
#
#             spot_center[idx, s, 0] = cx_final
#             spot_center[idx, s, 1] = cy_final
#             spot_amplitude[idx, s] = par[0].amplitude[0]
#
#             # plot sattelite spots and fit
#             if save_path or display:
#                 ax.plot([cx_final], [cy_final], marker='D', color=col[s])
#                 ax.add_patch(patches.Rectangle((cx - box, cy - box), 2 * box, 2 * box, ec='white', fc='none'))
#
#                 axs = fig.add_axes((0.17 + s * 0.2, 0.17, 0.1, 0.1))
#                 axs.imshow(sub, origin='bottom', aspect='equal', vmin=0, vmax=sub.max())
#                 axs.plot([par[0].x_mean], [par[0].y_mean], marker='D', color=col[s])
#                 axs.set_xticks([])
#                 axs.set_yticks([])
#
#                 axs = fig.add_axes((0.17 + s * 0.2, 0.06, 0.1, 0.1))
#                 axs.imshow(fit, origin='bottom', aspect='equal', vmin=0, vmax=sub.max())
#                 axs.set_xticks([])
#                 axs.set_yticks([])
#
#         # lines intersection
#         intersect = lines_intersect(spot_center[idx, 0, :], spot_center[idx, 2, :],
#                                     spot_center[idx, 1, :], spot_center[idx, 3, :])
#         img_center[idx] = intersect
#
#         # scaling
#         spot_dist[idx, 0] = np.sqrt(np.sum((spot_center[idx, 0, :] - spot_center[idx, 2, :])**2))
#         spot_dist[idx, 1] = np.sqrt(np.sum((spot_center[idx, 1, :] - spot_center[idx, 3, :])**2))
#         spot_dist[idx, 2] = np.sqrt(np.sum((spot_center[idx, 0, :] - spot_center[idx, 1, :])**2))
#         spot_dist[idx, 3] = np.sqrt(np.sum((spot_center[idx, 0, :] - spot_center[idx, 3, :])**2))
#         spot_dist[idx, 4] = np.sqrt(np.sum((spot_center[idx, 1, :] - spot_center[idx, 2, :])**2))
#         spot_dist[idx, 5] = np.sqrt(np.sum((spot_center[idx, 2, :] - spot_center[idx, 3, :])**2))
#
#         # finalize plot
#         if save_path or display:
#             ax.plot([spot_center[idx, 0, 0], spot_center[idx, 2, 0]],
#                     [spot_center[idx, 0, 1], spot_center[idx, 2, 1]],
#                     color='w', linestyle='dashed')
#             ax.plot([spot_center[idx, 1, 0], spot_center[idx, 3, 0]],
#                     [spot_center[idx, 1, 1], spot_center[idx, 3, 1]],
#                     color='w', linestyle='dashed')
#
#             ax.plot([intersect[0]], [intersect[1]], marker='+', color='w', ms=15)
#
#             ext = 1000 / pixel
#             ax.set_xlim(intersect[0] - ext, intersect[0] + ext)
#             ax.set_ylim(intersect[1] - ext, intersect[1] + ext)
#
#             # plt.tight_layout()
#
#             if save_path:
#                 pdf.savefig()
#                 plt.savefig(save_path+'.png', dpi=300)
#
#             if display:
#                 plt.pause(1e-3)
#
#     if save_path:
#         pdf.close()
#
#     return spot_center, spot_dist, img_center, spot_amplitude


def extract_satellite_spot_stamps(center_cube, xy_positions, stamp_size=23,
                                  shift_order=3, plot=False):
    """Short summary.

    Parameters
    ----------
    flux_arr : array
        ADI sequence image cube.
    stamp_size : tuple
        Size of stamp to be extracted.
    plot : bool
        Show extracted stamps.

    Returns
    -------
    tuple
        Array of stamp images and subpixel shifts.

    """

    # yx_position = yx_position_in_cube((flux_arr.shape[-2], flux_arr.shape[-1]),
    #                                   pos, pa, image_center, yx_anamorphism,
    #                                   right_handed)

    yx_positions = xy_positions[..., ::-1]
    # yx_positions = np.swapaxes(yx_positions, 0, 1)
    stamps = np.empty(
        [center_cube.shape[0], center_cube.shape[1],
         yx_positions.shape[2], stamp_size, stamp_size])
    stamps[:] = np.nan
    # shifts = np.zeros(
    #     [center_cube.shape[0], center_cube.shape[1],
    #      yx_positions.shape[2], 2])

    # stamps = []
    # shifts = []
    for wave_idx, wave_slice in enumerate(center_cube):
        for time_idx, frame in enumerate(wave_slice):
            for spot_idx, position in enumerate(yx_positions[wave_idx, time_idx]):
                if np.any(~np.isfinite(position)):
                    continue
                else:
                    cutout = Cutout2D(frame, (position[-1], position[-2]), stamp_size, copy=True)
                    if plot:
                        plt.imshow(frame, origin='lower')
                        cutout.plot_on_original(color='white')
                        plt.show()
                    subpixel_shift = np.array(cutout.position_original) - \
                        np.array(cutout.input_position_original)
                    # ipsh()
                    # shifts[wave_idx, time_idx, spot_idx] = subpixel_shift
                    stamps[wave_idx, time_idx, spot_idx] = shift(
                        cutout.data, (subpixel_shift[-1], subpixel_shift[-2]), output=None,
                        order=shift_order, mode='constant', cval=0.0, prefilter=True)

    # stamps=np.array(stamps)
    return np.squeeze(stamps)  # , np.squeeze(shifts)


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t = linspace(-2, 2, 0.1)
    x = sin(t)+randn(len(t))*0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def make_target_folder_string(observation):
    target_name = observation.observation['MAIN_ID'][0]
    target_name = " ".join(target_name.split())
    target_name = target_name.replace(" ", "_")
    obs_band = observation.observation['IFS_MODE'][0]
    date = observation.observation['NIGHT_START'][0]
    return target_name + '/' + obs_band + '/' + date