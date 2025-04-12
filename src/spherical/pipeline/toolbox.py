from ast import Attribute
import collections
import copy

import astropy.units as units
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.nddata import Cutout2D
from astropy.table import Column, Table, vstack
from astropy.time import Time
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


def ensure_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, Table):
        return data.to_pandas()
    else:
        raise TypeError("Input must be a pandas DataFrame or an astropy Table.")


def expand_frames_info(file_table_df):
    # Add a new column 'DIT INDEX' to the DataFrame for each integration
    # Repeat each row according to DET NDIT
    repeated_df = file_table_df.loc[file_table_df.index.repeat(file_table_df['DET NDIT'])].copy()
    
    # Create DIT INDEX: 0 to NDIT-1 for each original row
    dit_index = np.concatenate([np.arange(ndit) for ndit in file_table_df['DET NDIT']])
    
    # Insert as second column
    repeated_df.insert(1, 'DIT INDEX', dit_index)
    
    return repeated_df


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

    # Check if file_table is a Table or DataFrame
    frames_info_df = ensure_dataframe(copy.copy(file_table))

        
    # for key in header_keys:
    #     try:
    #         file_table.rename_column(key, header_keys[key])
    #     except KeyError:
    #         raise KeyError(f"{key} not found.")
    #     except AttributeError:
    frames_info_df.rename(columns=header_keys, inplace=True)

    frames_info_df = expand_frames_info(frames_info_df)
    # # create new dataframe
    # frames_info = file_table.copy()
    # for frame_index, _ in enumerate(file_table):
    #     for _ in range(file_table['DET NDIT'][frame_index]):
    #         frames_info = vstack([frames_info, Table(file_table[frame_index])])
    # frames_info = frames_info[len(file_table):]

    # dit_index = []
    # for frame_index, _ in enumerate(file_table):
    #     dit_index.extend(np.arange(file_table['DET NDIT'][frame_index]))
    # dit_index = Column(name='DIT INDEX', data=dit_index)
    # frames_info.add_column(dit_index, 1)
    # frames_info = frames_info.to_pandas()

    frames_info_df['DATE-OBS'] = pd.to_datetime(frames_info_df['DATE-OBS'], utc=False)
    frames_info_df['DATE'] = pd.to_datetime(frames_info_df['DATE'], utc=False)
    frames_info_df['DET FRAM UTC'] = pd.to_datetime(frames_info_df['DET FRAM UTC'], utc=False)

    return frames_info_df


def parallactic_angle(ha, dec, geolat):
    '''
    Parallactic angle of a source in degrees

    Parameters
    ----------
    ha : array_like
        Hour angle, in hours

    dec : float
        Declination, in degrees

    geolat : float
        Observatory declination, in degrees

    Returns
    -------
    pa : array_like
        Parallactic angle values
    '''
    pa = -np.arctan2(-np.sin(ha),
                     np.cos(dec) * np.tan(geolat) - np.sin(dec) * np.cos(ha))

    if (dec >= geolat):
        pa[ha < 0] += 360 * units.degree

    return np.degrees(pa)


def compute_times(frames_info: pd.DataFrame) -> None:
    """
    Compute various time-related quantities for science frames.

    Parameters
    ----------
    frames_info : pandas.DataFrame
        DataFrame containing metadata for science frames.
        Required columns:
            - 'DATE-OBS' (datetime64[ns])
            - 'DET FRAM UTC' (datetime64[ns])
            - 'DET NDIT' (int)
            - 'DET SEQ1 DIT' (float, in seconds)
            - 'DET DITDELAY' (float, in seconds)
            - 'DIT INDEX' (int)
            - 'TEL GEOLON', 'TEL GEOLAT' (degrees)
            - 'TEL GEOELEV' (meters)
            - 'SEQ ARM' (string), must include 'IRDIS' or 'IFS'
    Returns
    -------
    None
        The function modifies the `frames_info` DataFrame in-place by adding:
        - 'TIME START', 'TIME', 'TIME END' (datetime64[ns])
        - 'MJD START', 'MJD', 'MJD END' (float)
    """
    
    instrument = frames_info['SEQ ARM'].unique()
    if len(instrument) != 1 or instrument[0] not in ('IRDIS', 'IFS'):
        return  # Instrument not supported for this computation

    instrument = instrument[0]  # safe to treat as scalar now

    # Extract relevant time and index columns
    time_start = frames_info['DATE-OBS'].to_numpy(dtype='datetime64[ns]')
    time_end = frames_info['DET FRAM UTC'].to_numpy(dtype='datetime64[ns]')
    ndit = frames_info['DET NDIT'].to_numpy(dtype=int)
    idx = frames_info['DIT INDEX'].to_numpy()

    # Compute time delta per subintegration
    time_delta = (time_end - time_start) / ndit

    # Convert DIT and DITDELAY from seconds to timedelta64[ms]
    DIT = np.array(frames_info['DET SEQ1 DIT'].to_numpy(dtype=float) * 1000, dtype='timedelta64[ms]')
    DITDELAY = np.array(frames_info['DET DITDELAY'].to_numpy(dtype=float) * 1000, dtype='timedelta64[ms]')

    # Compute start, mid, and end timestamps
    ts_start = time_start + time_delta * idx + DITDELAY
    ts = ts_start + DIT / 2
    ts_end = ts_start + DIT

    # Extract telescope geolocation (only first row used)
    geolon = Angle(frames_info['TEL GEOLON'].iloc[0], unit=u.degree)
    geolat = Angle(frames_info['TEL GEOLAT'].iloc[0], unit=u.degree)
    geoelev = frames_info['TEL GEOELEV'].iloc[0]
    location = (geolon, geolat, geoelev)

    # Compute MJDs
    ts_start_str = ts_start.astype(str).tolist()
    ts_str = ts.astype(str).tolist()
    ts_end_str = ts_end.astype(str).tolist()

    mjd_start = Time(ts_start_str, scale='utc', location=location).mjd
    mjd = Time(ts_str, scale='utc', location=location).mjd
    mjd_end = Time(ts_end_str, scale='utc', location=location).mjd

    # Update the DataFrame in-place
    frames_info['TIME START'] = ts_start
    frames_info['TIME'] = ts
    frames_info['TIME END'] = ts_end

    frames_info['MJD START'] = mjd_start
    frames_info['MJD'] = mjd
    frames_info['MJD END'] = mjd_end


def compute_angles(frames_info, true_north=-1.75):
    '''
    Compute the various angles associated to frames: RA, DEC, parang,
    pupil offset, final derotation angle

    Parameters
    ----------
    frames_info : dataframe
        The data frame with all the information on science frames
    '''

    # derotator drift check and correction
    date_fix = Time('2016-07-12')
    if np.any(frames_info['MJD'].values <= date_fix.mjd):
        try:
            alt = frames_info['TEL ALT'].values.astype('float')
            drot2 = frames_info['INS4 DROT2 BEGIN'].values.astype('float')
            pa_correction = np.degrees(np.arctan(np.tan(np.radians(alt-2.*drot2))))
        except KeyError:
            pa_correction = 0
    else:
        pa_correction = 0

    # RA/DEC
    def convert_drot2_ra_to_deg(ra_drot):
        ra_drot_h = np.floor(ra_drot/1e4)
        ra_drot_m = np.floor((ra_drot - ra_drot_h * 1e4) / 1e2)
        ra_drot_s = ra_drot - ra_drot_h * 1e4 - ra_drot_m * 1e2

        ra_hour_hms_str = []
        for idx, _ in enumerate(ra_drot_h):
            ra_hour_hms_str.append(
                f'{int(ra_drot_h[idx])}h{int(ra_drot_m[idx])}m{ra_drot_s[idx]}s')
        ra_hour_hms_str = np.array(ra_hour_hms_str)
        ra_hour = Angle(angle=ra_hour_hms_str, unit=units.hour)
        ra_deg = ra_hour*15
        return ra_deg, ra_hour
    
    def convert_drot2_dec_to_deg(dec_drot):
        sign = np.sign(dec_drot)
        udec_drot = np.abs(dec_drot)
        dec_drot_d = np.floor(udec_drot / 1e4)
        dec_drot_m = np.floor((udec_drot - dec_drot_d * 1e4) / 1e2)
        dec_drot_s = udec_drot - dec_drot_d * 1e4 - dec_drot_m * 1e2
        dec_drot_d *= sign

        dec_dms_str = []
        for idx, _ in enumerate(dec_drot_d):
            dec_dms_str.append(
                f'{int(dec_drot_d[idx])}d{int(dec_drot_m[idx])}m{dec_drot_s[idx]}s')
        dec_dms_str = np.array(dec_dms_str)
        dec = Angle(dec_dms_str, units.degree)
        return dec

    ra_drot = frames_info['INS4 DROT2 RA'].values.astype('float')
    ra_deg, ra_hour = convert_drot2_ra_to_deg(ra_drot)
    frames_info['RA'] = ra_deg.value

    dec_drot = frames_info['INS4 DROT2 DEC'].values.astype('float')
    dec = convert_drot2_dec_to_deg(dec_drot)
    frames_info['DEC'] = dec.value

    # calculate parallactic angles
    geolon = Angle(frames_info['TEL GEOLON'].values[0], units.degree)
    geolat = Angle(frames_info['TEL GEOLAT'].values[0], units.degree)
    geoelev = frames_info['TEL GEOELEV'].values[0]
    earth_location = EarthLocation.from_geodetic(geolon, geolat, geoelev)

    utc = Time(frames_info['TIME START'].values.astype(str), scale='utc', location=earth_location)
    lst = utc.sidereal_time('apparent')
    ha = lst - ra_hour
    pa = parallactic_angle(ha, dec[0], geolat)
    frames_info['PARANG START'] = pa.value + pa_correction
    frames_info['HOUR ANGLE START'] = ha.value
    frames_info['LST START'] = lst.value

    utc = Time(frames_info['TIME'].values.astype(str), scale='utc', location=earth_location)
    lst = utc.sidereal_time('apparent')
    ha = lst - ra_hour
    pa = parallactic_angle(ha, dec[0], geolat)
    frames_info['PARANG'] = pa.value + pa_correction
    frames_info['HOUR ANGLE'] = ha.value
    frames_info['LST'] = lst.value

    # Altitude and airmass
    j2000 = SkyCoord(ra=ra_hour, dec=dec, frame='icrs', obstime=utc)

    altaz = j2000.transform_to(AltAz(location=earth_location))

    frames_info['ALTITUDE'] = altaz.alt.value
    frames_info['AZIMUTH'] = altaz.az.value
    frames_info['AIRMASS'] = altaz.secz.value

    utc = Time(frames_info['TIME END'].values.astype(str),
               scale='utc', location=(geolon, geolat, geoelev))
    lst = utc.sidereal_time('apparent')
    ha = lst - ra_hour
    pa = parallactic_angle(ha, dec[0], geolat)
    frames_info['PARANG END'] = pa.value + pa_correction
    frames_info['HOUR ANGLE END'] = ha.value
    frames_info['LST END'] = lst.value

    # Derotation angles
    #
    # PA_on-sky = PA_detector + PARANGLE + True_North + PUP_OFFSET + INSTRUMENT_OFFSET + TRUE_NORTH
    #  PUP_OFFSET = -135.99 ± 0.11
    #  INSTRUMENT_OFFSET
    #   IFS = +100.48 ± 0.10
    #   IRD =    0.00 ± 0.00
    #   TRUE_NORTH = -1.75 ± 0.08

    instru = frames_info['SEQ ARM'].unique()
    if len(instru) != 1:
        raise ValueError('Sequence is mixing different instruments: {0}'.format(instru))
    if instru == 'IFS':
        instru_offset = -100.48
    elif instru == 'IRDIS':
        instru_offset = 0.0
    else:
        raise ValueError('Unkown instrument {0}'.format(instru))

    drot_mode = frames_info['INS4 DROT2 MODE'].unique()
    if len(drot_mode) != 1:
        raise ValueError('Derotator mode has several values in the sequence')
    if drot_mode == 'ELEV':
        pupoff = 135.99
    elif drot_mode == 'SKY':
        pupoff = -100.48 + frames_info['INS4 DROT2 POSANG']
    elif drot_mode == 'STAT':
        pupoff = -100.48
    else:
        raise ValueError('Unknown derotator mode {0}'.format(drot_mode))

    frames_info['PUPIL OFFSET'] = pupoff + instru_offset

    # final derotation value
    frames_info['DEROT ANGLE'] = frames_info['PARANG'] + pupoff + instru_offset + true_north


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
                                   mask=None, save_path=None):
    '''
    Compute star center from PSF images (IRDIS CI, IRDIS DBI, IFS)

    Parameters
    ----------
    cube : array_like
        IRDIFS PSF cube

    wave : array_like
        Wavelength values, in nanometers

    pixel : float
        Pixel scale, in mas/pixel

    save_path : str
        Path where to save the fit images. Default is None, which means
        that the plot is not produced

    Returns
    -------
    img_centers : array_like
        The star center in each frame of the cube
    '''

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
        # cy, cx = np.unravel_index(np.argmax(img), img.shape)
        if guess_center_yx is None:
            cy, cx = np.array([188, 170])
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

        # # ipsh()
        # #     sub = np.ma.masked_array(
        # #         sub, mask=mask[idx][cy - box:cy + box, cx - box:cx + box])
        #
        # # fit peak with Gaussian + constant
        # # from astropy.stats import sigma_clip
        # imax = np.unravel_index(np.argmax(sub), sub.shape)
        # g_init = models.Gaussian2D(amplitude=sub.max(), x_mean=imax[1], y_mean=imax[0],
        #                            x_stddev=loD[idx], y_stddev=loD[idx]) + \
        #     models.Const2D(amplitude=sub.min())
        # fitter = fitting.LevMarLSQFitter()
        # par = fitter(g_init, xx[~sub_mask], yy[~sub_mask], sub[~sub_mask])
        #
        # cx_final = cx - box + par[0].x_mean
        # cy_final = cy - box + par[0].y_mean
        #
        # img_centers[idx, 0] = cx_final
        # img_centers[idx, 1] = cy_final

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