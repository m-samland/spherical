import collections
import copy

import astropy.units as units
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time

__all__ = ["prepare_dataframe", "compute_times", "compute_angles"]


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
    """
    Normalize and expand a CHARIS frame table into a row-per-DIT DataFrame.

    If the 'NAXIS3' column exists in the input, its values override 'DET NDIT'
    to define the number of integrations (DITs) per file.

    Parameters
    ----------
    file_table : pd.DataFrame or astropy.table.Table
        Table containing CHARIS file metadata and headers.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per DIT, including parsed and renamed header fields.
    """
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
        ('MJD_OBS', 'MJD-OBS')
    ])

    # Convert input to a DataFrame if needed
    frames_info_df = ensure_dataframe(copy.copy(file_table))

    # Override 'DET NDIT' if 'NAXIS3' exists
    if 'NAXIS3' in frames_info_df.columns:
        frames_info_df['DET NDIT'] = frames_info_df['NAXIS3']

    # Rename columns to standard keys
    frames_info_df.rename(columns=header_keys, inplace=True)

    # Expand to one row per DIT
    frames_info_df = expand_frames_info(frames_info_df)

    # Parse datetime columns
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
    geolon = Angle(frames_info['TEL GEOLON'].iloc[0], unit=units.degree)
    geolat = Angle(frames_info['TEL GEOLAT'].iloc[0], unit=units.degree)
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
