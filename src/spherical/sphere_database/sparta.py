#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'M. Samland @ MPIA (Heidelberg, Germany)'
#__all__ = []

import collections
import copy
import glob
import os
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time, TimeDelta


def unixseconds2mjd(unix_seconds):
    time1970 = Time(0, format='unix', scale='utc')
    delta_sec = TimeDelta(unix_seconds, format='sec')
    mjd_time = time1970 + delta_sec
    mjd_time.format = 'mjd'
    return mjd_time.value


def make_SPARTA_table(sparta_dir, root, new_table_name):
    """ Reads in all fits file containing 'SPHER.*.fits' in the raw_path
    with all subfolder, or single_raw_dir, without subfolder, depending
    on recursive = True or False.
    Creates table and list of failed files in root-folder.
    """

    start = time.time()
    files = [sparta_dir + f for f in os.listdir(sparta_dir) if f.endswith('.fits')]
    files = files[0:2]
    header_list = collections.OrderedDict([
        ('RA', ('RA', -10000)),
        ('DEC', ('DEC', -10000)),
        ('ND_FILTER', ('HIERARCH ESO INS4 FILT2 NAME', 'N/A')),
        ('SHUTTER', ('HIERARCH ESO INS4 SHUT ST', 'N/A')),
        ('IFS_MODE', ('HIERARCH ESO INS2 COMB IFS', 'N/A')),
        ('DPR_TYPE', ('HIERARCH ESO DPR TYPE', 'N/A')),
        ('DPR_TECH', ('HIERARCH ESO DPR TECH', 'N/A')),
        ('DET_ID', ('HIERARCH ESO DET ID', 'ZPL')),
        ('DEROTATOR_MODE', ('HIERARCH ESO INS4 COMB ROT', 'N/A')),
        ('SPATIAL_FILTER', ('HIERARCH ESO INS4 OPTI22 NAME', 'N/A')),
        ('OBS_PROG_ID', ('HIERARCH ESO OBS PROG ID', 'N/A')),
        ('DATE_OBS', ('DATE-OBS', 'N/A')),
        ('MJD_OBS', ('MJD-OBS', -10000))])

    # ORIGFILE=
    # HIERARCH ESO AOS VISWFS MODE
    # Shutter. T: open F: closed

    file_sizes = []
    bad_files = []

    header_result_table = collections.OrderedDict()
    for key in header_list.keys():
        header_result_table[key] = []

    for f in files:
        badflag = 0
        try:
            file_sizes.append(os.path.getsize(f))
            hdulist = fits.open(f)
            header = hdulist[0].header
        except IOError:
            bad_files.append(f)
            badflag = 1
        print('Read in File: {}'.format(f))

        for key in header_list.keys():
            header_result_table[key].append(get_headerval_with_exeption_new(
                header, header_list[key][0], header_list[key][1]))

        if badflag == 0:
            hdulist.close()

    # Add list of files names as first entry to the ordered dictionary
    header_result_table.update({'FILE': files})
    header_result_table.move_to_end('FILE', last=False)

    date_short = copy.deepcopy(header_result_table['DATE_OBS'])
    for i in range(len(date_short)):
        date_short[i] = date_short[i][0:10]
    #t['DATE_SHORT'] = date_short
    header_result_table.update({'DATE_SHORT': date_short})

    file_sizes = np.array(file_sizes) / 1e6
    header_result_table.update({'FILE_SIZE': file_sizes})

    t = Table(header_result_table)

    # Sort frames according to order in which they were taken
    t.sort('MJD_OBS')
    #t = Table([files, objects, ra, dec, exptime, DB_FILTER, coro, nd, ifs_mode, dpr_type, dpr_tech, det_id, derotator_mode, readout_mode, obs_prog_id, date_obs, mjd_obs], names=('FILE', 'OBJECT', 'RA', 'DEC', 'EXPTIME', 'DB_FILTER', 'IFS_mode', 'Coronagraph', 'ND_FILTER', 'DPR_TYPE', 'DPR_TECH', 'DET_ID', 'derotator_mode', 'readout_mode', 'obs_prog_id', 'DATE_OBS', 'MJD_OBS'), meta={'name': 'List of Files'})

    # Make new column containing only date not time

    t.write(new_table_name, format='fits', overwrite=True)
    print("RAW Data that produced errors: ")
    print(bad_files)
    bad_files = np.array(bad_files)
    np.save(root + 'list_of_bad_files.fits', bad_files)
    end = time.time()
    print("Time elapsed (minutes): {0}".format((end - start) / 60.))

    return t


def make_sparta_condition_table(sparta_file_table):
    start = time.time()
    counter = 0
    for idx, sparta_file in enumerate(sparta_file_table):
        dtypes = []
        for i in range(len(sparta_file.dtype)):
            dtypes.append(sparta_file.dtype[i])
        #main_row = query_result['FILE', 'RA', 'DEC'][0]
        print(sparta_file['FILE'])
        hdulist = fits.open(sparta_file['FILE'])
        atmosphere_data = hdulist[1].data
        #atmosphere_data = fits.getdata(sparta_file['FILE'], ext=1)
        for idx_atmos in range(len(atmosphere_data)):
            observation_characteristics = collections.OrderedDict()  # Key and default value
            observation_characteristics['UNIX_SEC'] = [0.]
            observation_characteristics['MJD'] = [0.]
            observation_characteristics['UTC'] = ['                            ']
            observation_characteristics['TIME_DIFF'] = [0.]
            observation_characteristics['R0'] = [0.]
            observation_characteristics['T0'] = [0.]
            observation_characteristics['L0'] = [0.]
            observation_characteristics['WINDSPEED'] = [0.]
            observation_characteristics['STREHL'] = [0.]

            atmo_info_table = Table(observation_characteristics)

            atmo_info_table['UNIX_SEC'][0] = atmosphere_data[idx_atmos][0]
            atmo_info_table['MJD'][0] = unixseconds2mjd(atmosphere_data[idx_atmos][0])
            ut_time = Time(unixseconds2mjd(atmosphere_data[idx_atmos][0]), format='mjd', scale='utc')
            ut_time.format = 'fits'
            atmo_info_table['UTC'][0] = ut_time.value
            atmo_info_table['TIME_DIFF'][0] = sparta_file['MJD_OBS'] * 3600 * 24 - atmo_info_table['MJD'][0] * 3600 * 24
            atmo_info_table['R0'][0] = atmosphere_data[idx_atmos][2]
            atmo_info_table['T0'][0] = atmosphere_data[idx_atmos][3]
            atmo_info_table['L0'][0] = atmosphere_data[idx_atmos][4]
            atmo_info_table['WINDSPEED'][0] = atmosphere_data[idx_atmos][5]
            atmo_info_table['STREHL'][0] = atmosphere_data[idx_atmos][6]

            if counter == 0:  # Create table from one row for first iteration
                table_of_atmo = Table(rows=sparta_file, names=sparta_file.colnames, dtype=dtypes, meta=sparta_file.meta)
                for idx in range(len(atmo_info_table.columns)):
                    table_of_atmo.add_column(atmo_info_table.columns[idx])
                counter += 1
            else:
                new_row_table = Table(rows=sparta_file, names=sparta_file.colnames, dtype=dtypes, meta=sparta_file.meta)
                for idx in range(len(atmo_info_table.columns)):
                    new_row_table.add_column(atmo_info_table.columns[idx])
                #simbad_table.add_row(query_result['MAIN_ID', 'RA', 'DEC', 'COO_ERR_MAJA', 'COO_ERR_MINA'][0])
                table_of_atmo.add_row(new_row_table[0])
        # ipsh()
        hdulist.close()
    end = time.time()
    print("Time elapsed (minutes): {0}".format((end - start) / 60.))
    return table_of_atmo


def plot_sparta_r0(table_of_obs, table_of_atmo, target_name, obs_band, date):

    target_sparta = retrieve_observations_from_name(table_of_atmo, target_name=target_name, show_in_browser=True)
    #ow_mask = np.logical_and.reduce((table_of_obs['MAIN_ID']==target_name, table_of_obs['DATE_SHORT']==date, table_of_obs['DB_FILTER']==obs_band))
    #row = table_of_obs[row_mask]
    #mjd_start = row['OBS_START'][0]
    #mjd_end = row['OBS_END'][0]
    #sparta_mask = np.logical_and(table_of_atmo['MJD_OBS'] > mjd_start, table_of_atmo['MJD_OBS'] < mjd_end)
    #conditions = table_of_atmo[sparta_mask]
    time_obs = Time(target_sparta['MJD'], format='mjd')
    plt.plot_date(time_obs.plot_date, target_sparta['STREHL'])

#sparta = make_SPARTA_table(sparta_directory, root, new_table_name='SPARTA_LIST_2017_02_01.fits')
#sparta = Table.read('SPARTA_LIST_2017_01_31.fits')
#table_of_atmo = make_sparta_condition_table(sparta)
# table_of_atmo.write('ATMOSPHERIC_CONDITIONS_2017_01_31.fits')
#table_of_atmo = Table.read('ATMOSPHERIC_CONDITIONS_2017_01_31.fits')

# N.In <9>: np.min(table_of_atmo['MJD_OBS']*3600*24 - table_of_atmo['MJD']*3600*24)
# N.Out<9>: 27.106158256530762
# all_detections = create_detection_table_andromeda(table_of_reductions)

#plot_sparta_r0(table_of_obs, table_of_atmo, target_name='beta pic', obs_band='DB_K12', date='2015-09-25')

#time_obs = Time(table_of_atmo['MJD'], format='mjd')
#plt.plot_date(time_obs.plot_date, table_of_atmo['STREHL'])
# plt.ylim(0,1)
# plt.show()
