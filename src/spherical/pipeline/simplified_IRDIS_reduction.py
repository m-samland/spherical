import collections
import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.table import Column, Table, vstack
from skimage.feature import register_translation
from tqdm import tqdm
from trap.utils import crop_box_from_3D_cube, crop_box_from_image

from spherical.pipeline import imutils, toolbox, transmission
from spherical.pipeline.toolbox import (
    # measure_center_waffle,
    # prepare_dataframe,
    star_centers_from_waffle_img_cube,
)


def reshape_IRDIS_cube_loop(img):
    nframes = img.shape[0]
    nimg = np.zeros((nframes, 2, 1024, 1024))
    for f in range(len(img)):
        nimg[f, 0] = img[f, :, 0:1024]
        nimg[f, 1] = img[f, :, 1024:]
    return nimg


def reshape_IRDIS_cube(cube):
    nframes = cube.shape[0]
    cube_reshaped = np.zeros((nframes, 2, 1024, 1024))
    cube_reshaped[:, 0] = cube[:, :, 0:1024]
    cube_reshaped[:, 1] = cube[:, :, 1024:]
    return cube_reshaped


def reshape_IRDIS_image(img):
    nimg = np.zeros((2, 1024, 1024))
    nimg[0] = img[:, 0:1024]
    nimg[1] = img[:, 1024:]
    return nimg


def prepare_dataframe(file_table):
    header_keys = collections.OrderedDict([
        ('TARG_ALPHA', 'TEL TARG ALPHA'),
        ('TARG_DELTA', 'TEL TARG DELTA'),
        ('TARG_PMA', 'TEL TARG PMA'),
        ('TARG_PMD', 'TEL TARG PMD'),
        ('DROT2_RA', 'INS4 DROT2 RA'),
        ('DROT2_DEC', 'INS4 DROT2 DEC'),
        ('INS4 DROT2 MODE', 'INS4 DROT2 MODE'),
        ('DROT2_BEGIN', 'INS4 DROT2 BEGIN'),
        ('DROT2_END', 'INS4 DROT2 END'),
        ('GEOLAT', 'TEL GEOLAT'),
        ('GEOLON', 'TEL GEOLON'),
        ('ALT_BEGIN', 'TEL ALT'),
        ('EXPTIME', 'EXPTIME'),
        ('SEQ1_DIT', 'DET SEQ1 DIT'),
        ('DIT_DELAY', 'DET DITDELAY'),
        ('NDIT', 'DET NDIT'),
        ('DB_FILTER', 'INS COMB IFLT'),
        ('CORO', 'INS COMB ICOR'),
        ('ND_FILTER', 'INS4 FILT2 NAME'),
        ('SHUTTER', 'INS4 SHUT ST'),
        ('IFS_MODE', 'INS2 COMB IFS'),
        ('PRISM', 'INS2 OPTI2 ID'),
        ('LAMP', 'INS2 CAL'),
        ('DPR_TYPE', 'DPR TYPE'),
        ('DPR_TECH', 'DPR TECH'),
        ('DET_ID', 'DET ID'),
        ('SEQ_ARM', 'SEQ ARM'),
        ('DEROTATOR_MODE', 'INS4 COMB ROT'),
        ('PAC_X', 'INS1 PAC X'),
        ('PAC_Y', 'INS1 PAC Y'),
        ('READOUT_MODE', 'DET READ CURNAME'),
        ('WAFFLE_AMP', 'OCS WAFFLE AMPL'),
        ('WAFFLE_ORIENT', 'OCS WAFFLE ORIENT'),
        ('SPATIAL_FILTER', 'INS4 OPTI22 NAME'),
        ('AMBI_TAU', 'TEL AMBI TAU0'),
        ('AMBI_FWHM_START', 'TEL AMBI FWHM START'),
        ('AMBI_FWHM_END', 'TEL AMBI FWHM END'),
        ('AIRMASS', 'TEL AIRM START'),
        ('OBS_PROG_ID', 'OBS PROG ID'),
        ('OBS_PI_COI', 'OBS PI-COI NAME'),
        ('OBS_ID', 'OBS ID'),
        ('ORIGFILE', 'ORIGFILE'),
        ('DATE', 'DATE'),
        ('DATE_OBS', 'DATE-OBS'),
        ('SEQ_UTC', 'DET SEQ UTC'),
        ('FRAM_UTC', 'DET FRAM UTC'),
        ('MJD_OBS', 'MJD-OBS')])

    for key in header_keys:
        try:
            file_table.rename_column(key, header_keys[key])
        except KeyError:
            pass

    # files = []
    # img = []
    # for finfo in file_table:
    #     NDIT = int(finfo['DET NDIT'])
    #
    #     files.extend(np.repeat(finfo['FILE'], NDIT))
    #     img.extend(list(np.arange(NDIT)))
    #
    # # create new dataframe
    frames_info = file_table.copy()
    for frame_index, _ in enumerate(file_table):
        for dit in range(file_table['DET NDIT'][frame_index]):
            frames_info = vstack([frames_info, Table(file_table[frame_index])])
    frames_info = frames_info[len(file_table):]

    dit_index = []
    for frame_index, _ in enumerate(file_table):
        dit_index.extend(np.arange(file_table['DET NDIT'][frame_index]))
    dit_index = Column(name='DIT INDEX', data=dit_index)
    frames_info.add_column(dit_index, 1)
    frames_info = frames_info.to_pandas()

    for key in frames_info:
        try:
            frames_info[key] = frames_info[key].str.decode("utf-8")
        except AttributeError:
            pass
    # column_names = file_table.copy()
    # del column_names['FILE']
    #
    # frames_info = pd.DataFrame(columns=column_names.columns,
    #                            index=pd.MultiIndex.from_arrays([files, img], names=['FILE', 'IMG']))
    #
    # # expand files_info into frames_info
    # frames_info = frames_info.align(file_table, level=0)[1]

    # frames_info = frames_info.to_pandas()
    frames_info['DATE-OBS'] = pd.to_datetime(frames_info['DATE-OBS'], utc=True)
    frames_info['DATE'] = pd.to_datetime(frames_info['DATE'], utc=True)
    frames_info['DET FRAM UTC'] = pd.to_datetime(frames_info['DET FRAM UTC'], utc=True)

    return frames_info


def collapse_data_frame(dframe):
    return dframe.groupby('FILE').mean()


def preprocess_data(observation, frame_type, outputdir, bpm=None,
                    dark_correction=True,
                    flat_correction=True,
                    crop=False,
                    crop_center=((480, 525), (483, 511)),
                    crop_size=256,
                    fix_badpix=False,
                    adjust_dit=True,
                    adjust_attenuation=True,
                    verbose=True):
    """Requires bpm right now
    EDIT: NEED TO CHANGE THIS TO ADJUST FLUX DIT TOWARDS CORO DIT
    DONT SCALE CORO DIT."""

    if verbose:
        print('Reduction of: {}'.format(frame_type))

    files_path = path.join(outputdir, frame_type)
    if not os.path.exists(files_path):
        os.makedirs(files_path)

    flat = fits.getdata(
        '/home/masa4294/science/python/projects/esorexer/static_calibration/irdis_flat_20150925.fits')
    # '/data/beegfs/astro-storage/groups/feldt/sphere_shared/automatic_reduction/ifs_esorexer/esorexer/static_calibration/irdis_flat_20150925.fits')

    frames_info = prepare_dataframe(observation.frames[frame_type])
    # frames_info = prepare_dataframe(a.frames['CENTER'])
    toolbox.compute_times(frames_info)
    toolbox.compute_angles(frames_info)

    cube = np.zeros((len(frames_info), 1024, 2048))
    frame_counter = 0
    for cube_idx, frame_info in enumerate(tqdm(observation.frames[frame_type])):
        img = fits.getdata(frame_info['FILE'])
        # add extra dimension to single images to make cubes
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        counter_init = frame_counter
        frame_counter += frame_info['DET NDIT']
        cube[counter_init:frame_counter] = img
    # cube = np.array(cube)

    # mask dead regions
    # if frame_type != 'FLUX':
    cube[:, :15, :] = np.nan
    cube[:, 1013:, :] = np.nan
    cube[:, :, :50] = np.nan
    cube[:, :, 941:1078] = np.nan
    cube[:, :, 1966:] = np.nan

    if bpm is not None:
        bpm_cube = reshape_IRDIS_image(bpm)

    if dark_correction:
        if frame_type == 'CORO' or frame_type == 'CENTER':
            dark = fits.getdata(observation.frames['BG_SCIENCE']['FILE'][-1])
        elif frame_type == 'FLUX':
            dark = fits.getdata(observation.frames['BG_FLUX']['FILE'][-1])
            # dark_sky = fits.getdata(observation.frames['BG_SCIENCE']['FILE'][-1])
            # y_dark_region = 542
            # dy_dark_region = 311
            # x_dark_region = 1044
            # dx_dark_region = 180
            # yx_scaling_box = np.array((
            #     (y_dark_region-dy_dark_region, y_dark_region+dy_dark_region),
            #     (x_dark_region-dx_dark_region, x_dark_region+dx_dark_region)))
            #
            # dark_box_mask = np.zeros([dark_sky.shape[-2], dark_sky.shape[-1]], dtype='bool')
            # dark_box_mask[
            #     yx_scaling_box[0, 0]:yx_scaling_box[0, 1],
            #     yx_scaling_box[1, 0]:yx_scaling_box[1, 1]] = True
            # dark_box_bpm = bpm[dark_box_mask]
            #
            # dark_box_median = np.median(dark_sky[:, dark_box_mask]) #[:, ~dark_box_bpm])
            #
            # # cube_scalings = np.nanmedian(cube[:, dark_box_mask][:, ~dark_box_bpm], axis=1)
            # cube_scalings = np.nanmedian(cube[:, dark_box_mask], axis=1)
            # cube_scalings /= dark_box_median
            #
            # cube -= cube_scalings[:, None, None] * dark_sky
            # bpm_dark_box = bpm[yx_scaling_box[0, 0]:yx_scaling_box[0, 1],
            #     yx_scaling_box[1, 0]:yx_scaling_box[1, 1]]

        else:
            raise ValueError('Cannot process frame type: {}. Only "CORO", "CENTER", and "FLUX"'.format(frame_type))
        if dark.ndim == 3 and dark.shape[0] > 1:
            dark = np.median(dark, axis=0)
        dark = dark.squeeze()
        if verbose:
            print('   ==> subtract dark')
        cube -= dark

    if flat_correction:
        if verbose:
            print('   ==> divide by flat field')
        cube /= flat

    # reshape data
    if verbose:
        print('   ==> reshape data')
    cube = reshape_IRDIS_cube(cube)
    cube = np.swapaxes(cube, 0, 1)

    # Subtract CORO if available?
    if crop:
        crop_center = np.array(crop_center)
        cube_cropped = np.zeros((cube.shape[0], cube.shape[1], crop_size, crop_size))
        bpm_cube_cropped = np.zeros((cube.shape[0], crop_size, crop_size))
        for idx, monochromatic_cube in enumerate(cube):
            cube_cropped[idx] = crop_box_from_3D_cube(
                flux_arr=monochromatic_cube,
                boxsize=crop_size,
                center_yx=crop_center[idx][::-1])
            bpm_cube_cropped[idx] = crop_box_from_image(
                flux_arr=bpm_cube[idx],
                boxsize=crop_size,
                center_yx=crop_center[idx][::-1])
        cube = cube_cropped
        bpm_cube = bpm_cube_cropped

    # bad pixels correction
    # Change to work on reshaped cube

    if fix_badpix:
        number_of_pixels = np.prod(np.array((cube.shape[-2], cube.shape[-1])))
        threshhold_bad_warning = round(0.005 * number_of_pixels)
        if verbose:
            print('   ==> correct bad pixels')
        for wave_idx, wave_cube in enumerate(cube):
            # ipsh()
            for frame_idx, frame in enumerate(tqdm(wave_cube)):
                frame = imutils.fix_badpix(frame, bpm_cube[wave_idx], npix=12, weight=True)

                # additional sigma clipping to remove transitory bad pixels
                # not done for OBJECT,FLUX because PSF peak can be clipped
                if (frame_type != 'FLUX'):
                    frame_temp, cosmic_mask = imutils.sigma_filter(
                        frame, box=7, nsigma=4, return_mask=True, iterate=False)
                    if np.sum(cosmic_mask) < threshhold_bad_warning:
                        frame = frame_temp
                    else:
                        print("Warning: transient bad pixel number above 1 perc: {}. On frame {} wavelength {}. No interpolation performed".format(
                            np.sum(cosmic_mask), frame_idx, wave_idx))

                cube[wave_idx, frame_idx] = frame

    # Adjust for exposure time and ND filter, put all frames to 1 second exposure
    if len(frames_info['INS COMB IFLT'].unique()) > 1:
        raise ValueError('Non-unique filters in sequence.')
    else:
        filter_comb = frames_info['INS COMB IFLT'].unique()[0]
    wave, bandwidth = transmission.wavelength_bandwidth_filter(filter_comb)
    wave = np.array(wave)
    if len(frames_info['INS4 FILT2 NAME'].unique()) > 1:
        raise ValueError('Non-unique ND filters in sequence.')
    else:
        ND = frames_info['INS4 FILT2 NAME'].unique()[0]
    w, attenuation = transmission.transmission_nd(ND, wave=wave)
    dits = np.array(frames_info['DET SEQ1 DIT'])
    print("Attenuation: {}".format(attenuation))
    print("DITs: {}".format(dits))
    if adjust_dit:
        cube /= dits[np.newaxis, :, np.newaxis, np.newaxis]
    if adjust_attenuation:
        cube /= attenuation[:, np.newaxis, np.newaxis, np.newaxis]

    # filenames = []
    # for f in range(len(frames_info)):
    #     #     frame = img[f, ...].squeeze()
    #     #     hdr['DET NDIT'] = 1
    #     filename = os.path.join(files_path, path.splitext(path.basename(
    #         frame_info['FILE']))[0] + '_DIT{0:03d}_preproc.fits'.format(f))
    #     filenames.append(filename)
    #     fits.writeto(filename, frame, hdr, overwrite=True, output_verify='silentfix')

    # Should probably set NDIT to 1. But check if this would cause problems later
    # frames_info['FILE'] = filenames
    fits.writeto(os.path.join(files_path, 'cube.fits'), cube.astype('float32'), overwrite=True)
    # frames_info_astropy = Table.from_pandas(frames_info)
    frames_info.to_csv(os.path.join(files_path, 'frames_preproc.csv'))
    # frames_info_astropy.write(os.path.join(files_path, 'frames_info.fits'))
    return cube, frames_info, bpm_cube


# def create_psf_model_from_flux_frames():


def execute_IRDIS_target(
        observation,
        reduction_folder,
        dark_correction=False,
        flat_correction=True,
        crop=False,
        crop_center=((480, 525), (483, 511)),
        crop_size=256,
        fix_badpix=False,
        adjust_dit=True,
        adjust_attenuation=True,
        center_highpass=False,
        verbose=True):

    target_name = observation.observation['MAIN_ID'][0]
    target_name = " ".join(target_name.split())
    target_name = target_name.replace(" ", "_")

    obs_band = observation.observation['DB_FILTER'][0]
    date = observation.observation['DATE_SHORT'][0]
    # outputdir = path.join(root,target_name,obs_band,date,'/')
    outputdir = path.join(
        reduction_folder, target_name + '/' + obs_band + '/' + date)

    if verbose:
        print("Start reduction of: {} {} {}".format(target_name, obs_band, date))

    if not path.exists(outputdir):
        os.makedirs(outputdir)

    # bad pixel map
    bpm = fits.getdata(
        '/home/masa4294/science/python/projects/esorexer/static_calibration/badpixel_mask_irdis_20150925.fits').astype('bool')
    # Mask dead regions as good pixel to prevent them from being interpolated later

    bpm[:15, :] = False
    bpm[1013:, :] = False
    bpm[:, :50] = False
    bpm[:, 941:1078] = False
    bpm[:, 1966:] = False

    bpm_cube = reshape_IRDIS_image(bpm)
    fits.writeto(os.path.join(outputdir, 'badpixelmask.fits'),
                 bpm_cube, overwrite=True)

    center_cube, frames_info_center, bpm_cube_center = preprocess_data(
        observation, frame_type='CENTER',
        outputdir=outputdir,
        bpm=bpm, dark_correction=dark_correction,
        flat_correction=flat_correction,
        crop=crop,
        crop_center=crop_center,
        crop_size=crop_size,
        fix_badpix=fix_badpix,
        adjust_dit=adjust_dit,
        adjust_attenuation=adjust_attenuation,
        verbose=verbose)

    if not observation.observation['WAFFLE_MODE'][0]:
        coro_cube, frames_info_coro, bpm_cube_coro = preprocess_data(
            observation, frame_type='CORO',
            outputdir=outputdir,
            bpm=bpm, dark_correction=dark_correction,
            flat_correction=flat_correction,
            crop=crop,
            crop_center=crop_center,
            crop_size=crop_size,
            fix_badpix=fix_badpix,
            adjust_dit=adjust_dit,
            adjust_attenuation=adjust_attenuation,
            verbose=verbose)

    flux_cube, frames_info_flux, bpm_cube_flux = preprocess_data(
        observation, frame_type='FLUX',
        outputdir=outputdir,
        bpm=bpm, dark_correction=dark_correction,
        flat_correction=flat_correction,
        crop=True,
        crop_center=((511, 493), (514, 480)),
        crop_size=128,
        fix_badpix=fix_badpix,
        adjust_dit=adjust_dit,
        adjust_attenuation=adjust_attenuation,
        verbose=verbose)

    # ipsh()
    # for frame_type in ['CENTER']:  # , 'CORO', 'FLUX']:
    #     # create new dataframe
    # ipsh()
    # centers_xy: (time, wave, (x, y))
    # ipsh()

    spot_centers, spot_distances, image_centers, spot_amplitudes = measure_center_waffle(
        frames_info=frames_info_center,
        cube=center_cube,
        bpm_cube=bpm_cube_center,
        outputdir=outputdir,
        instrument='IRDIS',
        crop=crop,
        crop_center=crop_center,
        high_pass=center_highpass)

    fits.writeto(os.path.join(outputdir, 'CENTER', 'center_frame_centers.fits'), image_centers, overwrite=True)

    # NDIT combine center measurements and times
    if not observation.observation['WAFFLE_MODE'][0]:
        # mean_cube = np.mean(coro_cube[0].reshape(-1, 16, 1024, 1024), axis=1)

        # Determine shift based on cross-correlation
        shifts = []
        # shifts_spider = []
        errors = []
        diffphases = []

        ref_image = np.mean(center_cube[0, :4], axis=0)

        for image in tqdm(coro_cube[0]):
            shift, error, diffphase = register_translation(
                # image[519:527, 475:488], ref_image[519:527, 475:488], 100)
                image[521:529, 477:485], ref_image[521:529, 477:485], 100)
            # shift_spider, _, _ = register_translation(
            #     # image[519:527, 475:488], ref_image[519:527, 475:488], 100)
            #     image[648:666, 629:647], ref_image[648:666, 629:647], 100)
            shifts.append(shift)
            # shifts_spider.append(shift_spider)
            errors.append(error)
            diffphases.append(diffphases)
        shifts = np.array(shifts)
        # shifts_spider = np.array(shifts_spider)
        shift_x = shifts[:, 1]
        shift_y = shifts[:, 0]

        # shift_x2 = shifts_spider[:, 1]
        # shift_y2 = shifts_spider[:, 0]
        # centers = image_centers[:, 0]
        # cube_centers = np.mean(centers.reshape(-1, 32, 2), axis=1)

        # Gauss fit of center
        threshold = 0.5
        # Reference
        ref_image = np.mean(center_cube[0, 3:4], axis=0)
        sub = ref_image[521:529, 477:485].copy()
        xx, yy = np.meshgrid(np.arange(sub.shape[0]), np.arange(sub.shape[1]))

        # for image in tqdm(coro_cube[0]):
        # imax = np.unravel_index(np.argmax(sub), sub.shape)
        # g_init = models.Moffat2D(amplitude=sub.max() * 1.3, x_0=3, y_0=4,
        #                            gamma=2.13 * 2.355, alpha=1) - \
        #     models.Gaussian2D(amplitude=sub.max()/1.5, x_mean=3, y_mean=4,
        #                            x_stddev=1.23, y_stddev=1.23)
        g_init = models.Gaussian2D(amplitude=sub.max() * 1.3, x_mean=3, y_mean=4,
                                   x_stddev=2., y_stddev=2., theta=1.) - \
            models.Gaussian2D(amplitude=sub.max() / 1.5, x_mean=3, y_mean=4,
                              x_stddev=1.3, y_stddev=1.3, theta=1.)
        g_init.y_stddev_0.fixed = True
        g_init.x_stddev_0.fixed = True
        g_init.y_stddev_1.fixed = True
        g_init.x_stddev_1.fixed = True
        g_init.theta_0.fixed = True
        g_init.theta_1.fixed = True
        # g_init = models.Gaussian2D(amplitude=sub.max(), x_mean=imax[1], y_mean=imax[0],
        #                            x_stddev=1., y_stddev=1.) + \
        #     models.Const2D(amplitude=sub.min())
        fitter = fitting.LevMarLSQFitter()
        # par = fitter(g_init, xx[~sub_mask], yy[~sub_mask], sub[~sub_mask])
        par = fitter(g_init, xx, yy, sub)
        gauss_model = par(xx, yy)
        residual_mask = np.abs(sub - gauss_model) / sub < threshold
        par = fitter(par, xx[residual_mask], yy[residual_mask], sub[residual_mask])
        # x_ref = np.mean([par.x_mean_0.value, par.x_mean_1.value])
        # g_big = models.Gaussian2D(
        #     amplitude=par.amplitude_0.value,
        #     x_mean=par.x_mean_0.value,
        #     y_mean=par.y_mean_0.value,
        #     x_stddev=1.23,
        #     y_stddev=1.23)
        x_ref = par.x_mean_1.value
        y_ref = par.y_mean_1.value
        # Science sequence

        ref_image = np.mean(center_cube[0, 4:], axis=0)
        sub = ref_image[521:529, 477:485].copy()
        xx, yy = np.meshgrid(np.arange(sub.shape[0]), np.arange(sub.shape[1]))

        # for image in tqdm(coro_cube[0]):
        # imax = np.unravel_index(np.argmax(sub), sub.shape)
        # g_init = models.Moffat2D(amplitude=sub.max() * 1.3, x_0=3, y_0=4,
        #                            gamma=2.13 * 2.355, alpha=1) - \
        #     models.Gaussian2D(amplitude=sub.max()/1.5, x_mean=3, y_mean=4,
        #                            x_stddev=1.23, y_stddev=1.23)
        g_init = models.Gaussian2D(amplitude=sub.max() * 1.3, x_mean=3, y_mean=4,
                                   x_stddev=2., y_stddev=2., theta=1) - \
            models.Gaussian2D(amplitude=sub.max() / 1.5, x_mean=3, y_mean=4,
                              x_stddev=1.3, y_stddev=1.3, theta=1)
        g_init.y_stddev_0.fixed = True
        g_init.x_stddev_0.fixed = True
        g_init.y_stddev_1.fixed = True
        g_init.x_stddev_1.fixed = True
        g_init.theta_0.fixed = True
        g_init.theta_1.fixed = True
        # g_init = models.Gaussian2D(amplitude=sub.max(), x_mean=imax[1], y_mean=imax[0],
        #                            x_stddev=1., y_stddev=1.) + \
        #     models.Const2D(amplitude=sub.min())
        fitter = fitting.LevMarLSQFitter()
        # par = fitter(g_init, xx[~sub_mask], yy[~sub_mask], sub[~sub_mask])
        par = fitter(g_init, xx, yy, sub)
        gauss_model = par(xx, yy)
        residual_mask = np.abs(sub - gauss_model) / sub < threshold
        par = fitter(par, xx[residual_mask], yy[residual_mask], sub[residual_mask])
        # x_ref = np.mean([par.x_mean_0.value, par.x_mean_1.value])
        # g_big = models.Gaussian2D(
        #     amplitude=par.amplitude_0.value,
        #     x_mean=par.x_mean_0.value,
        #     y_mean=par.y_mean_0.value,
        #     x_stddev=1.23,
        #     y_stddev=1.23)
        x_ref2 = par.x_mean_1.value
        y_ref2 = par.y_mean_1.value

        shifts_gauss_x = []
        shifts_gauss_y = []
        # ipsh()
        for image in tqdm(coro_cube[0]):
            def tie_x(model):
                x = model.x_mean_0
                return x
            sub = image[521:529, 477:485].copy()
            xx, yy = np.meshgrid(np.arange(sub.shape[0]), np.arange(sub.shape[1]))
            # imax = np.unravel_index(np.argmax(sub2), sub2.shape)
            # g_init = models.Gaussian2D(amplitude=sub.max(), x_mean=imax[1], y_mean=imax[0],
            #                            x_stddev=1., y_stddev=1.) + \
            #     models.Const2D(amplitude=sub.min())
            g_init = models.Gaussian2D(amplitude=sub.max() * 1.3, x_mean=3, y_mean=4,
                                       x_stddev=2., y_stddev=2., theta=1) - \
                models.Gaussian2D(amplitude=sub.max() / 1.5, x_mean=3, y_mean=4,
                                  x_stddev=1.3, y_stddev=1.3, theta=1)
            g_init.y_stddev_0.fixed = True
            g_init.x_stddev_0.fixed = True
            g_init.y_stddev_1.fixed = True
            g_init.x_stddev_1.fixed = True
            g_init.theta_0.fixed = True
            g_init.theta_1.fixed = True
            fitter = fitting.LevMarLSQFitter()
            # par = fitter(g_init, xx[~sub_mask], yy[~sub_mask], sub[~sub_mask])
            par = fitter(g_init, xx, yy, sub)
            gauss_model = par(xx, yy)
            residual_mask = np.abs(sub - gauss_model) / sub < threshold
            par = fitter(par, xx[residual_mask], yy[residual_mask], sub[residual_mask])
            # shifts_gauss_x.append(np.mean([par.x_mean_0.value, par.x_mean_1.value]))
            # shifts_gauss_y.append(np.mean([par.y_mean_0.value, par.y_mean_1.value]))
            shifts_gauss_x.append(par.x_mean_1.value)
            shifts_gauss_y.append(par.y_mean_1.value)
        shifts_gauss_x = np.array(shifts_gauss_x)
        shifts_gauss_y = np.array(shifts_gauss_y)

        shifts_gauss_x -= x_ref
        shifts_gauss_y -= y_ref

        x_ref_diff = x_ref - x_ref2
        y_ref_diff = y_ref - y_ref2

        # Header based shift
        dms_dx_ref = frames_info_center['INS1 PAC X'][0] / 18
        dms_dy_ref = frames_info_center['INS1 PAC Y'][0] / 18
        # dms_dx_ref = np.array(frames_info_center['INS1 PAC X'])[-1] / 18
        # dms_dy_ref = np.array(frames_info_center['INS1 PAC Y'])[-1] / 18

        # dms_dx = frames_info_center['INS1 PAC X'] / 18
        # dms_dy = frames_info_center['INS1 PAC Y'] / 18

        dms_dx = np.array(frames_info_coro['INS1 PAC X'] / 18)
        dms_dy = np.array(frames_info_coro['INS1 PAC Y'] / 18)

        dms_offset_x = dms_dx_ref - dms_dx
        dms_offset_y = dms_dy_ref - dms_dy

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 9))
        axes[0].plot(shift_x, label='shift x')
        # axes[0].plot(shift_x2, label='shift x spider')
        axes[0].plot(shifts_gauss_x, label='gauss x')
        axes[0].plot(-1 * dms_offset_x, label='dms shift x')
        axes[0].plot(len(shift_x) + 1, x_ref_diff, 'o', label='pos ref after')
        axes[0].axhline(y=x_ref_diff, xmin=0, xmax=len(shift_x) + 1)
        axes[0].set_ylabel('x shift')
        axes[0].set_ylim(-3, 3)
        # axes[1].yaxis.set_major_locator(MaxNLocator(5))
        axes[1].plot(shift_y, label='shift y')
        # axes[1].plot(shift_y2, label='shift y spider')
        axes[1].plot(shifts_gauss_y, label='gauss x')
        axes[1].plot(-1 * dms_offset_y, label='dms shift y')
        axes[1].plot(len(shift_y) + 1, y_ref_diff, 'o', label='pos ref after')
        axes[1].axhline(y=y_ref_diff, xmin=0, xmax=len(shift_y) + 1)
        axes[1].set_ylabel('y shift')
        axes[1].set_ylim(-3, 2)
        fig.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outputdir, "shift_sequence.png"))
        # plt.plot(shift_x, label='shift x')
        # plt.plot(-1 * dms_offset_x, label='dms shift x')
        # plt.show()

        plt.close()
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 9))
        axes[0].plot(shifts_gauss_x + dms_offset_x, label='diff x')
        axes[0].set_ylabel('x shift')
        # axes[1].plot(shift_y2, label='shift y spider')
        axes[1].plot(shifts_gauss_y + dms_offset_y, label='diff y')
        axes[1].set_ylabel('y shift')
        # axes[1].set_ylim(-3, 2)
        fig.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outputdir, "shift_sequence_header_vs_gauss.png"))
        # plt.plot(shift_x, label='shift x')
        # plt.plot(-1 * dms_offset_x, label='dms shift x')
        # plt.show()

        np.std(shifts_gauss_y + dms_offset_y)

        reference_center = np.mean(
            image_centers[: frames_info_center['DET NDIT'][0]], axis=0)

        centers_corrected = np.repeat(
            np.expand_dims(reference_center, axis=0), len(frames_info_coro), axis=0)
        centers_corrected[:, :, 0] += dms_offset_x[:, None]
        centers_corrected[:, :, 1] += dms_offset_y[:, None]

        centers_corrected_minus = np.repeat(
            np.expand_dims(reference_center, axis=0), len(frames_info_coro), axis=0)
        centers_corrected_minus[:, :, 0] -= dms_offset_x[:, None]
        centers_corrected_minus[:, :, 1] -= dms_offset_y[:, None]

        fits.writeto(os.path.join(outputdir, 'image_centers_plus.fits'), centers_corrected, overwrite=True)
        fits.writeto(os.path.join(outputdir, 'image_centers_minus.fits'), centers_corrected_minus, overwrite=True)

    else:
        fits.writeto(os.path.join(outputdir, 'image_centers.fits'), image_centers, overwrite=True)

    fits.writeto(os.path.join(outputdir, 'waffle_amplitudes.fits'), spot_amplitudes, overwrite=True)
    # ipsh()
    # cube_dms_dx = np.mean(np.array(dms_dx).reshape(-1, 16), axis=1)
    # cube_dms_dy = np.mean(np.array(dms_dy).reshape(-1, 16), axis=1)

    # plt.scatter(cube_centers[:, 0], cube_centers[:, 1], label='uncorrected')
    # plt.scatter(
    #     cube_centers[:, 0] - (cube_dms_dx - dms_dx_ref),
    #     cube_centers[:, 1] - (cube_dms_dy - dms_dy_ref), label='corrected')
    # plt.legend()
    # plt.show()

    # dms_dx = frames_info_coro['INS1 PAC X'] / 18
    # dms_dy = frames_info_coro['INS1 PAC Y'] / 18

    # center frames
    # centers = []
    # # cx, cy image center of reference frame
    # # for wave_idx, img in enumerate(cube):
    # for wave_idx, img in enumerate(cube):
    #     cx, cy = image_centers[wave_idx, :]
    #
    #     # DMS contribution
    #     cx = cx + dms_dx_ref + dms_dx
    #     cy = cy + dms_dy_ref + dms_dy

    # filter_comb = frames_info['INS COMB IFLT'].unique()[0]
    # wave, bandwidth = transmission.wavelength_bandwidth_filter(filter_comb)
    # wave = np.array(wave) / 1000.
    #
    # ND = frames_info.loc[(file, idx), 'INS4 FILT2 NAME']
    # w, attenuation = transmission.transmission_nd(ND, wave=wave * 1000)
    # nimg = nimg / DIT / attenuation[wave_idx]
    # centers_highpass_xy = measure_center_waffle(
    #     frames_info_center, bpm_cube, outputdir, instrument='IRDIS',
    #     high_pass=True)
    #
    # centers_adj = centers_xy.copy()
    # dpac_x = 0.007
    # dpac_y = 0.001
    #
    # centers_adj[4:, :, 0] = centers_xy[4:, :, 0].copy() + dpac_x
    # centers_adj[4:, :, 1] = centers_xy[4:, :, 1].copy() + dpac_y
    # plt.scatter(x=np.mean(centers_xy[:4, 0, 0]), y=np.mean(centers_xy[:4, 0, 1]), label='1st mean')
    # plt.scatter(x=np.mean(centers_xy[4:, 0, 0]), y=np.mean(centers_xy[4:, 0, 1]), label='2st mean')
    # plt.plot(centers_xy[:4, 0, 0], centers_xy[:4, 0, 1], label='1st')
    # plt.plot(centers_xy[4:, 0, 0], centers_xy[4:, 0, 1], label='2nd')
    # plt.plot(centers_adj[4:, 0, 0], centers_adj[4:, 0, 1], label='2nd adj')
    # plt.scatter(x=np.mean(centers_adj[4:, 0, 0]), y=np.mean(centers_adj[4:, 0, 1]), label='2nd adj mean')
    # # plt.plot(centers_highpass[:4, 0, 0], centers_highpass[:4, 0, 1], label='1st with filter')
    # # plt.plot(centers_highpass[4:, 0, 0], centers_highpass[4:, 0, 1], label='2nd with filter')
    # plt.legend()
    # plt.show()


def measure_center_waffle(cube, outputdir, instrument,
                          bpm_cube=None, wavelengths=None,
                          waffle_orientation=None,
                          frames_info=None,
                          center_guess=None,
                          crop=False,
                          crop_center=((480, 525), (483, 511)),
                          fit_background=True,
                          fit_symmetric_gaussian=False,
                          high_pass=False):
    spot_centers = []
    spot_distances = []
    image_centers = []
    spot_amplitudes = []

    for i in range(cube.shape[1]):
        print("Frame: {}".format(i))
        if waffle_orientation is None and frames_info is not None:
            waffle_orientation = frames_info['OCS WAFFLE ORIENT'][i]
        data = cube[:, i]  # fits.getdata(frames_info['FILE'][i])

        plot_path = os.path.join(outputdir, 'CENTER_img_{0:03d}.pdf'.format(i))
        # os.path.dirname(frames_info['FILE'][i]),
        # 'img_{0:03d}.pdf'.format(i))

        # spot_center, spot_distance, img_center, spot_amplitude = toolbox.star_centers_from_waffle_cube_old(
        #     data, wave=wave, instrument='IRDIS', waffle_orientation=waffle_orientation,
        #     mask=bpm_cube, high_pass=high_pass,
        #     center_offset=(0, 0), smooth=0,
        #     coro=True, display=False, save_path=plot_path)
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
        elif instrument == 'IFS':
            pixel = 7.46
            orientation_offset = 102
            if center_guess is None:
                center_guess = np.array([129, 126])[None, :].repeat(21, axis=0)
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
            high_pass=high_pass,
            center_offset=(0, 0),
            smooth=0, coro=True,
            save_path=plot_path)

        # ipsh()
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
    # bpm = toolbox.compute_bad_pixel_map(bpm_files)

    # ['CORO', 'CENTER', 'FLUX', 'FLAT', 'DISTORTION', 'CENTER_BEFORE', 'CENTER_AFTER', 'BG_SCIENCE', 'BG_FLUX']

# import pandas as pd
# frames_info = pd.read_csv('frames_preproc.csv')
# properties = frames_info[['OCS WAFFLE AMPL', 'OCS WAFFLE ORIENT', 'MJD']]
# normal_mode_mask = properties['OCS WAFFLE AMPL'] == 0.04
# normal_mode_mask = normal_mode_mask.values
# weak_x = properties['OCS WA']


#
# def sph_ird_star_center(path, pixel, frames_info, high_pass=False, offset=(0, 0), display=False, save=True):
#     '''Determines the star center for all frames where a center can be
#     determined (OBJECT,CENTER and OBJECT,FLUX)
#
#     Parameters
#     ----------
#     high_pass : bool
#         Apply high-pass filter to the image before searching for the satelitte spots
#
#     offset : tuple
#         Apply an (x,y) offset to the default center position, for the waffle centering.
#         Default is no offset
#
#     display : bool
#         Display the fit of the satelitte spots
#
#     save : bool
#         Save the fit of the sattelite spot for quality check. Default is True,
#         although it is a bit slow.
#
#     '''
#
#     # check if recipe can be executed
#
#     print('Star centers determination')
#
#     # parameters
#     # path = self._path
#     # pixel = self._pixel
#     # frames_info = self._frames_info_preproc
#
#     # wavelength
#     filter_comb = frames_info['INS COMB IFLT'].unique()[0]
#     wave, bandwidth = transmission.wavelength_bandwidth_filter(filter_comb)
#     wave = np.array(wave) / 1000.
#
#     # start with OBJECT,FLUX
#     flux_files = frames_info[frames_info['DPR TYPE'] == 'OBJECT,FLUX']
#     if len(flux_files) != 0:
#         # THIS HAS TO BE CHANGED to be compatible with IRDIS observation class!
#         for file, idx in flux_files.index:
#             print('  ==> OBJECT,FLUX: {0}'.format(file))
#
#             # read data
#             fname = '{0}_DIT{1:03d}_preproc'.format(file, idx)
#             files = glob.glob(os.path.join(path.preproc, fname + '.fits'))
#             cube, hdr = fits.getdata(files[0], header=True)
#
#             # centers
#             if save:
#                 save_path = os.path.join(path.products, fname + '_PSF_fitting.pdf')
#             else:
#                 save_path = None
#             img_center = toolbox.star_centers_from_PSF_cube(cube, wave, pixel, display=display, save_path=save_path)
#
#             # save
#             fits.writeto(os.path.join(path.preproc, fname + '_centers.fits'), img_center, overwrite=True)
#             print()
#
#     # then OBJECT,CENTER
#
#     starcen_files = frames_info[frames_info['DPR TYPE'] == 'OBJECT,CENTER']
#     if len(starcen_files) != 0:
#         for file, idx in starcen_files.index:
#             print('  ==> OBJECT,CENTER: {0}'.format(file))
#
#             # read data
#             fname = '{0}_DIT{1:03d}_preproc'.format(file, idx)
#             files = glob.glob(os.path.join(path.preproc, fname + '.fits'))
#             cube, hdr = fits.getdata(files[0], header=True)
#
#             # coronagraph
#             coro_name = starcen_files.loc[(file, idx), 'INS COMB ICOR']
#             if coro_name == 'N_NS_CLEAR':
#                 coro = False
#             else:
#                 coro = True
#
#             # centers
#             waffle_orientation = hdr['OCS WAFFLE ORIENT']
#             if save:
#                 save_path = os.path.join(path.products, fname + '_spots_fitting.pdf')
#             else:
#                 save_path = None
#             spot_center, spot_dist, img_center\
#                 = toolbox.star_centers_from_waffle_cube(cube, wave, 'IRDIS', waffle_orientation,
#                                                         high_pass=high_pass, center_offset=offset,
#                                                         coro=coro, display=display, save_path=save_path)
#
#             # save
#             fits.writeto(os.path.join(path.preproc, fname + '_centers.fits'), img_center, overwrite=True)
#             print()
#
#
# def sph_ird_combine_data(self, cpix=True, psf_dim=80, science_dim=290, correct_anamorphism=True,
#                          shift_method='fft', manual_center=None, skip_center=False, save_scaled=False):
#     '''Combine and save the science data into final cubes
#
#     All types of data are combined independently: PSFs
#     (OBJECT,FLUX), star centers (OBJECT,CENTER) and standard
#     coronagraphic images (OBJECT). For each type of data, the
#     method saves 4 or 5 different files:
#
#       - *_cube: the (x,y,time,lambda) cube
#
#       - *_parang: the parallactic angle vector
#
#       - *_derot: the derotation angles vector. This vector takes
#                  into account the parallactic angle and any
#                  instrumental pupil offset. This is the values
#                  that need to be used for aligning the images with
#                  North up and East left.
#
#       - *_frames: a csv file with all the information for every
#                   frames. There is one line by time step in the
#                   data cube.
#
#       - *_cube_scaled: the (x,y,time,lambda) cube with images
#                        rescaled spectraly. This is useful if you
#                        plan to perform spectral differential
#                        imaging in your analysis.
#
#     Parameters
#     ----------
#     cpix : bool
#         If True the images are centered on the pixel at coordinate
#         (dim//2,dim//2). If False the images are centered between 4
#         pixels, at coordinates ((dim-1)/2,(dim-1)/2). Default is True.
#
#     psf_dim : even int
#         Size of the PSF images. Default is 80x80 pixels
#
#     science_dim : even int
#         Size of the science images (star centers and standard
#         coronagraphic images). Default is 290, 290 pixels
#
#     correct_anamorphism : bool
#         Correct the optical anamorphism of the instrument. Default
#         is True. See user manual for details.
#
#     manual_center : array
#         User provided centers for the OBJECT,CENTER and OBJECT
#         fra@mes. This should be an array of 2x2 values (cx,cy for
#         the 2 wavelengths). If a manual center is provided, the
#         value of skip_center is ignored for the OBJECT,CENTER and
#         OBJECT frames. Default is None
#
#     skip_center : bool
#         Control if images are finely centered or not before being
#         combined. However the images are still roughly centered by
#         shifting them by an integer number of pixel to bring the
#         center of the data close to the center of the images. This
#         option is useful if fine centering must be done
#         afterwards.
#
#     shift_method : str
#         Method to scaling and shifting the images: fft or interp.
#         Default is fft
#
#     save_scaled : bool
#         Also save the wavelength-rescaled cubes. Makes the process
#         much longer. The default is False
#
#     '''
#
#     # check if recipe can be executed
#     toolbox.check_recipe_execution(self._recipe_execution, 'sph_ird_combine_data', self.recipe_requirements)
#
#     print('Combine science data')
#
#     # parameters
#     path = self._path
#     nwave = self._nwave
#     frames_info = self._frames_info_preproc
#
#     # wavelength
#     filter_comb = frames_info['INS COMB IFLT'].unique()[0]
#     wave, bandwidth = transmission.wavelength_bandwidth_filter(filter_comb)
#     wave = np.array(wave) / 1000.
#
#     fits.writeto(os.path.join(path.products, 'wavelength.fits'), wave, overwrite=True)
#
#     # max images size
#     if psf_dim > 1024:
#         print('Warning: psf_dim cannot be larger than 1024 pix. A value of 1024 will be used.')
#         psf_dim = 1024
#
#     if science_dim > 1024:
#         print('Warning: science_dim cannot be larger than 1024 pix. A value of 1024 will be used.')
#         science_dim = 1024
#
#     # centering
#     centers_default = np.array([[484, 517], [486, 508]])
#     if skip_center:
#         print('Warning: images will not be centered. They will just be combined.')
#         shift_method = 'roll'
#
#     if manual_center is not None:
#         manual_center = np.array(manual_center)
#         if manual_center.shape != (2, 2):
#             raise ValueError('manual_center does not have the right number of dimensions.')
#
#         print('Warning: images will be centered at the user-provided values.')
#
#     #
#     # OBJECT,FLUX
#     #
#     flux_files = frames_info[frames_info['DPR TYPE'] == 'OBJECT,FLUX']
#     nfiles = len(flux_files)
#     if nfiles != 0:
#         print(' * OBJECT,FLUX data')
#
#         # final arrays
#         psf_cube = np.zeros((nwave, nfiles, psf_dim, psf_dim))
#         psf_parang = np.zeros(nfiles)
#         psf_derot = np.zeros(nfiles)
#         if save_scaled:
#             psf_cube_scaled = np.zeros((nwave, nfiles, psf_dim, psf_dim))
#
#         # final center
#         if cpix:
#             cc = psf_dim // 2
#         else:
#             cc = (psf_dim - 1) / 2
#
#         # read and combine files
#         for file_idx, (file, idx) in enumerate(flux_files.index):
#             print('  ==> file {0}/{1}: {2}, DIT={3}'.format(file_idx + 1, len(flux_files), file, idx))
#
#             # read data
#             fname = '{0}_DIT{1:03d}_preproc'.format(file, idx)
#             files = glob.glob(os.path.join(path.preproc, fname + '.fits'))
#             cube = fits.getdata(files[0])
#             centers = fits.getdata(os.path.join(path.preproc, fname + '_centers.fits'))
#
#             # neutral density
#             ND = frames_info.loc[(file, idx), 'INS4 FILT2 NAME']
#             w, attenuation = transmission.transmission_nd(ND, wave=wave * 1000)
#
#             # DIT, angles, etc
#             DIT = frames_info.loc[(file, idx), 'DET SEQ1 DIT']
#             psf_parang[file_idx] = frames_info.loc[(file, idx), 'PARANG']
#             psf_derot[file_idx] = frames_info.loc[(file, idx), 'DEROT ANGLE']
#
#             # center frames
#             for wave_idx, img in enumerate(cube):
#                 if skip_center:
#                     cx, cy = centers_default[wave_idx, :]
#                 else:
#                     cx, cy = centers[wave_idx, :]
#
#                 img = img.astype(np.float)
#                 nimg = imutils.shift(img, (cc - cx, cc - cy), method=shift_method)
#                 nimg = nimg / DIT / attenuation[wave_idx]
#
#                 psf_cube[wave_idx, file_idx] = nimg[:psf_dim, :psf_dim]
#
#                 # correct anamorphism
#                 if correct_anamorphism:
#                     nimg = psf_cube[wave_idx, file_idx]
#                     nimg = imutils.scale(nimg, (1.0000, 1.0062), method='interp')
#                     psf_cube[wave_idx, file_idx] = nimg
#
#                 # wavelength-scaled version
#                 if save_scaled:
#                     nimg = psf_cube[wave_idx, file_idx]
#                     psf_cube_scaled[wave_idx, file_idx] = imutils.scale(
#                         nimg, wave[0] / wave[wave_idx], method=shift_method)
#
#         # save final cubes
#         flux_files.to_csv(os.path.join(path.products, 'psf_frames.csv'))
#         fits.writeto(os.path.join(path.products, 'psf_cube.fits'), psf_cube, overwrite=True)
#         fits.writeto(os.path.join(path.products, 'psf_parang.fits'), psf_parang, overwrite=True)
#         fits.writeto(os.path.join(path.products, 'psf_derot.fits'), psf_derot, overwrite=True)
#         if save_scaled:
#             fits.writeto(os.path.join(path.products, 'psf_cube_scaled.fits'), psf_cube_scaled, overwrite=True)
#
#         # delete big cubes
#         del psf_cube
#         if save_scaled:
#             del psf_cube_scaled
#
#         print()
#
#     #
#     # OBJECT,CENTER
#     #
#     starcen_files = frames_info[frames_info['DPR TYPE'] == 'OBJECT,CENTER']
#     nfiles = len(starcen_files)
#     if nfiles != 0:
#         print(' * OBJECT,CENTER data')
#
#         # final arrays
#         cen_cube = np.zeros((nwave, nfiles, science_dim, science_dim))
#         cen_parang = np.zeros(nfiles)
#         cen_derot = np.zeros(nfiles)
#         if save_scaled:
#             cen_cube_scaled = np.zeros((nwave, nfiles, science_dim, science_dim))
#
#         # final center
#         if cpix:
#             cc = science_dim // 2
#         else:
#             cc = (science_dim - 1) / 2
#
#         # read and combine files
#         for file_idx, (file, idx) in enumerate(starcen_files.index):
#             print('  ==> file {0}/{1}: {2}, DIT={3}'.format(file_idx + 1, len(starcen_files), file, idx))
#
#             # read data
#             fname = '{0}_DIT{1:03d}_preproc'.format(file, idx)
#             files = glob.glob(os.path.join(path.preproc, fname + '.fits'))
#             cube = fits.getdata(files[0])
#             centers = fits.getdata(os.path.join(path.preproc, fname + '_centers.fits'))
#
#             # neutral density
#             ND = frames_info.loc[(file, idx), 'INS4 FILT2 NAME']
#             w, attenuation = transmission.transmission_nd(ND, wave=wave * 1000)
#
#             # DIT, angles, etc
#             DIT = frames_info.loc[(file, idx), 'DET SEQ1 DIT']
#             cen_parang[file_idx] = frames_info.loc[(file, idx), 'PARANG']
#             cen_derot[file_idx] = frames_info.loc[(file, idx), 'DEROT ANGLE']
#
#             # center frames
#             for wave_idx, img in enumerate(cube):
#                 if manual_center is not None:
#                     cx, cy = manual_center[wave_idx, :]
#                 else:
#                     if skip_center:
#                         cx, cy = centers_default[wave_idx, :]
#                     else:
#                         cx, cy = centers[wave_idx, :]
#
#                 img = img.astype(np.float)
#                 nimg = imutils.shift(img, (cc - cx, cc - cy), method=shift_method)
#                 nimg = nimg / DIT / attenuation[wave_idx]
#
#                 cen_cube[wave_idx, file_idx] = nimg[:science_dim, :science_dim]
#
#                 # correct anamorphism
#                 if correct_anamorphism:
#                     nimg = cen_cube[wave_idx, file_idx]
#                     nimg = imutils.scale(nimg, (1.0000, 1.0062), method='interp')
#                     cen_cube[wave_idx, file_idx] = nimg
#
#                 # wavelength-scaled version
#                 if save_scaled:
#                     nimg = cen_cube[wave_idx, file_idx]
#                     cen_cube_scaled[wave_idx, file_idx] = imutils.scale(
#                         nimg, wave[0] / wave[wave_idx], method=shift_method)
#
#         # save final cubes
#         starcen_files.to_csv(os.path.join(path.products, 'starcenter_frames.csv'))
#         fits.writeto(os.path.join(path.products, 'starcenter_cube.fits'), cen_cube, overwrite=True)
#         fits.writeto(os.path.join(path.products, 'starcenter_parang.fits'), cen_parang, overwrite=True)
#         fits.writeto(os.path.join(path.products, 'starcenter_derot.fits'), cen_derot, overwrite=True)
#         if save_scaled:
#             fits.writeto(os.path.join(path.products, 'starcenter_cube_scaled.fits'),
#                          cen_cube_scaled, overwrite=True)
#
#         # delete big cubes
#         del cen_cube
#         if save_scaled:
#             del cen_cube_scaled
#
#         print()
#
#     #
#     # OBJECT
#     #
#     object_files = frames_info[frames_info['DPR TYPE'] == 'OBJECT']
#     nfiles = len(object_files)
#     if nfiles != 0:
#         print(' * OBJECT data')
#
#         # get first DIT of first OBJECT,CENTER in the sequence. See issue #12.
#         starcen_files = frames_info[frames_info['DPR TYPE'] == 'OBJECT,CENTER']
#         if (len(starcen_files) == 0) or skip_center or (manual_center is not None):
#             print('Warning: no OBJECT,CENTER file in the data set. Images cannot be accurately centred. ' +
#                   'They will just be combined.')
#
#             # choose between manual center or default centers
#             if manual_center is not None:
#                 centers = manual_center
#             else:
#                 centers = centers_default
#
#             # null value for Dithering Motion Stage
#             dms_dx_ref = 0
#             dms_dy_ref = 0
#         else:
#             fname = '{0}_DIT{1:03d}_preproc_centers.fits'.format(
#                 starcen_files.index.values[0][0], starcen_files.index.values[0][1])
#             centers = fits.getdata(os.path.join(path.preproc, fname))
#
#             # Dithering Motion Stage for star center: value is in micron,
#             # and the pixel size is 18 micron
#             dms_dx_ref = starcen_files['INS1 PAC X'][0] / 18
#             dms_dy_ref = starcen_files['INS1 PAC Y'][0] / 18
#
#         # final center
#         if cpix:
#             cc = science_dim // 2
#         else:
#             cc = (science_dim - 1) / 2
#
#         # final arrays
#         sci_cube = np.zeros((nwave, nfiles, science_dim, science_dim))
#         sci_parang = np.zeros(nfiles)
#         sci_derot = np.zeros(nfiles)
#         if save_scaled:
#             sci_cube_scaled = np.zeros((nwave, nfiles, science_dim, science_dim))
#
#         # read and combine files
#         for file_idx, (file, idx) in enumerate(object_files.index):
#             print('  ==> file {0}/{1}: {2}, DIT={3}'.format(file_idx + 1, len(object_files), file, idx))
#
#             # read data
#             fname = '{0}_DIT{1:03d}_preproc'.format(file, idx)
#             files = glob.glob(os.path.join(path.preproc, fname + '*.fits'))
#             cube = fits.getdata(files[0])
#
#             # neutral density
#             ND = frames_info.loc[(file, idx), 'INS4 FILT2 NAME']
#             w, attenuation = transmission.transmission_nd(ND, wave=wave * 1000)
#
#             # DIT, angles, etc
#             DIT = frames_info.loc[(file, idx), 'DET SEQ1 DIT']
#             sci_parang[file_idx] = frames_info.loc[(file, idx), 'PARANG']
#             sci_derot[file_idx] = frames_info.loc[(file, idx), 'DEROT ANGLE']
#
#             # Dithering Motion Stage for star center: value is in micron,
#             # and the pixel size is 18 micron
#             dms_dx = frames_info.loc[(file, idx), 'INS1 PAC X'] / 18
#             dms_dy = frames_info.loc[(file, idx), 'INS1 PAC Y'] / 18
#
#             # center frames
#             for wave_idx, img in enumerate(cube):
#                 cx, cy = centers[wave_idx, :]
#
#                 # DMS contribution
#                 cx = cx + dms_dx_ref + dms_dx
#                 cy = cy + dms_dy_ref + dms_dy
#
#                 img = img.astype(np.float)
#                 nimg = imutils.shift(img, (cc - cx, cc - cy), method=shift_method)
#                 nimg = nimg / DIT / attenuation[wave_idx]
#
#                 sci_cube[wave_idx, file_idx] = nimg[:science_dim, :science_dim]
#
#                 # correct anamorphism
#                 if correct_anamorphism:
#                     nimg = sci_cube[wave_idx, file_idx]
#                     nimg = imutils.scale(nimg, (1.0000, 1.0062), method='interp')
#                     sci_cube[wave_idx, file_idx] = nimg
#
#                 # wavelength-scaled version
#                 if save_scaled:
#                     nimg = sci_cube[wave_idx, file_idx]
#                     sci_cube_scaled[wave_idx, file_idx] = imutils.scale(
#                         nimg, wave[0] / wave[wave_idx], method=shift_method)
#
#         # save final cubes
#         object_files.to_csv(os.path.join(path.products, 'science_frames.csv'))
#         fits.writeto(os.path.join(path.products, 'science_cube.fits'), sci_cube, overwrite=True)
#         fits.writeto(os.path.join(path.products, 'science_parang.fits'), sci_parang, overwrite=True)
#         fits.writeto(os.path.join(path.products, 'science_derot.fits'), sci_derot, overwrite=True)
#         if save_scaled:
#             fits.writeto(os.path.join(path.products, 'science_cube_scaled.fits'), sci_cube_scaled, overwrite=True)
#
#         # delete big cubes
#         del sci_cube
#         if save_scaled:
#             del sci_cube_scaled
#
#         print()
