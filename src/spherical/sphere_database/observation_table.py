#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "M. Samland @ MPIA (Heidelberg, Germany)"
__all__ = ["remove_objects_from_simbad_list", "create_observation_table"]

import collections

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Column, Table, hstack, vstack
from spherical.pipeline.embed_shell import ipsh
from spherical.sphere_database.database_utils import add_night_start_date
from spherical.sphere_database.target_table import filter_for_science_frames
from tqdm import tqdm


def remove_objects_from_simbad_list(table_of_targets, list_of_simbad_names_to_exclude):
    for name in list_of_simbad_names_to_exclude:
        mask = table_of_targets["MAIN_ID"] == name
        if np.sum(mask) > 0:
            idx_to_remove = np.where(mask)[0][0]
            table_of_targets.remove_row(idx_to_remove)
        else:
            print("Name: {} not found.".format(name))
    return table_of_targets


def create_science_data_table(
    table_of_files,
    table_of_targets,
    instrument,
    cone_size_science=10.0,
    cone_size_sky=73.0,
):
    try:
        del table_of_targets["NUMBER_OF_OBS"]
    except KeyError:
        pass

    table_of_files = add_night_start_date(table_of_files, key="DATE_OBS")

    if "OBS_NUMBER" not in table_of_files.keys():
        table_of_files["OBS_NUMBER"] = -10000

    t_coro, t_center, t_center_coro, t_science = filter_for_science_frames(
        table_of_files, instrument=instrument, remove_fillers=remove_fillers
    )

    cone_size_science = cone_size_science * u.arcsec
    cone_size_sky = cone_size_sky * u.arcsec
    file_list_coords = SkyCoord(
        ra=t_science["RA"] * u.degree, dec=t_science["DEC"] * u.degree
    )
    number_of_observations_list = []

    science_data_table = []
    counter = 0
    # test for one target with multiple observations
    for _, target in enumerate(tqdm(table_of_targets)):
        # print(target['MAIN_ID'])
        target_coords = SkyCoord(
            ra=target["RA_HEADER"] * u.degree, dec=target["DEC_HEADER"] * u.degree
        )
        # target_coords = SkyCoord(ra=target['RA'] * u.degree, dec=target['DEC'] * u.degree)
        target_mask = target_coords.separation(file_list_coords) < cone_size_science
        target_sky_mask = target_coords.separation(file_list_coords) < cone_size_sky
        t_target = t_science[target_mask]
        t_target_sky = t_science[target_sky_mask]

        t_target = enumerate_observations_by_timediff(t_science, key="DATE_OBS")
        t_target.add_column(target["MAIN_ID"], name="MAIN_ID", index=0)
        # What do I want? A new table of files for all science targets with column
        # Containing a unique ID?

        t_target.append(science_data_table)

    science_data_table = vstack(t_target)

    return science_data_table


def enumerate_observations_by_timediff(table, key="DATE_OBS"):
    times = Time(table[key])
    mjd = times.to_value("mjd")
    # plt.scatter(
    #     times.to_datetime()[:-1],
    #     np.diff(times.to_value('mjd')) * 24)
    jumps = np.diff(mjd * 24) > 1  # consider 1 hour difference as separate obs
    obs_num = 0
    obs_numbers = [0]  # since jumps are on difference array, start with 0 already
    for idx, jump in enumerate(jumps):
        if not jump:
            obs_numbers.append(obs_num)
        else:
            obs_num += 1
            obs_numbers.append(obs_num)
    table["OBS_NUMBER"] = obs_numbers
    return table


# def add_night_start_date(table, key="DATE_OBS"):
#     if "NIGHT_START" not in table.keys():
#         night_start = []
#         times = Time(table[key]).to_datetime()
#         for time in times:
#             if time.hour < 12:
#                 new_time = time - timedelta(1)
#             else:
#                 new_time = time
#             night_start.append(str(new_time.date()))
#         table["NIGHT_START"] = night_start
#     return table


# Loop over targets on science data table
# Look up if PM is corrected?


def create_observation_table(
    table_of_files,
    table_of_targets,
    instrument,
    cone_size_science=15.0,
    cone_size_sky=73.0,
    remove_fillers=True,
):
    # Delete column 'NUMBER_OF_OBS' from observation sequence table if it exists, so that it won't show up for individual observations
    try:
        del table_of_targets["NUMBER_OF_OBS"]
    except KeyError:
        pass

    table_of_files = add_night_start_date(table_of_files, key="DATE_OBS")

    if "OBS_NUMBER" not in table_of_files.keys():
        table_of_files["OBS_NUMBER"] = -10000

    t_coro, t_center, t_center_coro, t_science = filter_for_science_frames(
        table_of_files, instrument=instrument, remove_fillers=remove_fillers
    )

    cone_size_science = cone_size_science * u.arcsec
    cone_size_sky = cone_size_sky * u.arcsec
    file_list_coords = SkyCoord(
        ra=t_science["RA"] * u.degree, dec=t_science["DEC"] * u.degree
    )
    number_of_observations_list = []

    target_science_file_table = []
    counter = 0
    # test for one target with multiple observations
    for idx, target in enumerate(tqdm(table_of_targets)):
        # print(target['MAIN_ID'])
        target_coords = SkyCoord(
            ra=target["RA_HEADER"] * u.degree, dec=target["DEC_HEADER"] * u.degree
        )
        # target_coords = SkyCoord(ra=target['RA'] * u.degree, dec=target['DEC'] * u.degree)
        target_mask = target_coords.separation(file_list_coords) < cone_size_science
        target_sky_mask = target_coords.separation(file_list_coords) < cone_size_sky
        t_target = t_science[target_mask]
        t_target_sky = t_science[target_sky_mask]

        # t_target = enumerate_observations_by_timediff(t_science, key="DATE_OBS")
        # t_target.add_column(target["MAIN_ID"], name="MAIN_ID", index=0)
        # What do I want? A new table of files for all science targets with column
        # Containing a unique ID?
        if len(t_target) == 0:
            print(f"No files found for {target['MAIN_ID']}.")
            continue

        date_keys = t_target.group_by("NIGHT_START").groups.keys
        number_of_dates = len(date_keys)
        number_of_observations = 0
        dtypes = []
        for i in range(len(target.dtype)):
            dtypes.append(target.dtype[i])

        for date in date_keys:
            # print(date[0])
            t_target_date = t_target[t_target["NIGHT_START"] == date[0]]
            t_target_sky_date = t_target_sky[t_target_sky["NIGHT_START"] == date[0]]

            if instrument == "IRDIS":
                observation_modes = t_target_date.group_by("DB_FILTER").groups.keys
            elif instrument == "IFS":
                observation_modes = t_target_date.group_by("IFS_MODE").groups.keys
            number_of_modes = len(observation_modes)

            # print("Number of filter-keys: {}".format(number_of_filters))

            for mode in observation_modes["IFS_MODE"]:
                if instrument == "IRDIS":
                    t_single_obs = t_target_date[t_target_date["DB_FILTER"] == mode]
                elif instrument == "IFS":
                    t_single_obs = t_target_date[t_target_date["IFS_MODE"] == mode]
                t_coro_frames = t_single_obs[t_single_obs["DPR_TYPE"] == "OBJECT"]
                t_center_frames = t_single_obs[
                    t_single_obs["DPR_TYPE"] == "OBJECT,CENTER"
                ]
                t_flux_frames = t_single_obs[t_single_obs["DPR_TYPE"] == "OBJECT,FLUX"]
                t_sky_frames = t_target_sky_date[t_target_sky_date["DPR_TYPE"] == "SKY"]

                if (
                    len(t_coro_frames) == 0
                    and len(t_center_frames) == 0
                    and len(t_flux_frames) == 0
                ):
                    break
                # if len(np.unique(t_single_obs['OBS_ID'])) > 1:
                #     ipsh() # Many observations that belong together have multiple IDs
                # because of changing OBs...

                observation_characteristics = (
                    collections.OrderedDict()
                )  # Key and default value
                # observation_characteristics["DATE_SHORT"] = ["          "]
                observation_characteristics["NIGHT_START"] = ["          "]
                observation_characteristics["OBS_START"] = [t_single_obs["MJD_OBS"][0]]
                observation_characteristics["OBS_END"] = [t_single_obs["MJD_OBS"][-1]]
                observation_characteristics["MJD_MEAN"] = [0.0]
                observation_characteristics["INSTRUMENT"] = ["     "]
                if instrument == "IRDIS":
                    observation_characteristics["DB_FILTER"] = ["             "]
                elif instrument == "IFS":
                    observation_characteristics["IFS_MODE"] = ["      "]
                observation_characteristics["ND_FILTER"] = ["      "]
                observation_characteristics["ND_FILTER_FLUX"] = ["      "]
                observation_characteristics["WAFFLE_MODE"] = [False]
                observation_characteristics["NON_CORO_MODE"] = [False]
                observation_characteristics["FAILED_SEQ"] = [False]
                observation_characteristics["TOTAL_EXPTIME"] = [0.0]  # in minutes
                observation_characteristics["ROTATION"] = [0.0]  # in deg
                observation_characteristics["DIT"] = [0.0]
                observation_characteristics["NDIT"] = [0]
                observation_characteristics["NCUBES"] = [0]
                observation_characteristics["NCENTER"] = [0]
                observation_characteristics["SCI_DIT_FLAG"] = [False]
                observation_characteristics["DIT_FLUX"] = [0.0]
                observation_characteristics["NDIT_FLUX"] = [0]
                observation_characteristics["NFLUX"] = [0]
                observation_characteristics["FLUX_FLAG"] = [False]
                observation_characteristics["NSKY"] = [0]
                observation_characteristics["MEAN_TAU"] = [0.0]
                observation_characteristics["STDDEV_TAU"] = [0.0]
                observation_characteristics["MEAN_FWHM"] = [0.0]
                observation_characteristics["STDDEV_FWHM"] = [0.0]
                observation_characteristics["MEAN_AIRMASS"] = [0.0]
                observation_characteristics["DEROTATOR_MODE"] = ["          "]
                observation_characteristics["WAFFLE_AMP"] = [0.0]
                observation_characteristics["OBS_ID"] = [0]
                observation_characteristics["OBS_PROG_ID"] = ["                "]
                # observation_characteristics["TOTAL_FILE_SIZE"] = [
                #     0.0
                # ]  # Size of science sequence in megabyte

                # Make table with new information about the sequence as a whole
                obs_info_table = Table(observation_characteristics)

                # print()
                # print(obs_info_table.to_pandas())
                # print(list(obs_info_table.to_pandas().columns))
                # print()
                # print(t_flux_frames.to_pandas())
                # print(list(t_flux_frames.to_pandas().columns))
                # print()
                # print(t_coro_frames.to_pandas())
                # print(list(t_coro_frames.to_pandas().columns))
                # print()
                # print(t_center_frames.to_pandas())
                # print(list(t_center_frames.to_pandas().columns))

                obs_info_table["NIGHT_START"][0] = date[0]
                if instrument == "IRDIS":
                    obs_info_table["DB_FILTER"][0] = mode
                elif instrument == "IFS":
                    obs_info_table["IFS_MODE"][0] = mode
                obs_info_table["INSTRUMENT"] = instrument

                number_of_observations += number_of_modes
                number_of_observations_list.append(number_of_observations)

                obs_info_table["NCENTER"] = len(t_center_frames)
                obs_info_table["NFLUX"] = len(t_flux_frames)
                obs_info_table["NSKY"] = len(t_sky_frames)

                if len(t_flux_frames) == 0:
                    obs_info_table["FLUX_FLAG"] = True
                else:
                    if len(t_flux_frames.group_by("EXPTIME").groups.keys) != 1:
                        obs_info_table["FLUX_FLAG"] = True
                    elif len(t_flux_frames) < 2:
                        obs_info_table["FLUX_FLAG"] = True
                    else:
                        obs_info_table["DIT_FLUX"] = t_flux_frames["EXPTIME"][-1]
                        obs_info_table["NDIT_FLUX"] = t_flux_frames["DET NDIT"][-1]

                if len(t_flux_frames) > len(t_coro_frames) and len(t_flux_frames) > len(
                    t_center_frames
                ):
                    obs_info_table["NON_CORO_MODE"] = True

                if len(t_center_frames) > 0:
                    obs_info_table["WAFFLE_AMP"][0] = t_center_frames["WAFFLE_AMP"][-1]


                # Test if waffle mode is used. This is the case if more exposure time is in the center images
                # than in the coro images
                if len(t_center_frames) >= 1:
                    if len(t_coro_frames) == 0:
                        obs_info_table["WAFFLE_MODE"][0] = True
                    elif np.sum(t_center_frames["DET NDIT"]) > np.sum(t_coro_frames["DET NDIT"]):
                        obs_info_table["WAFFLE_MODE"][0] = True
                    else:
                        obs_info_table["WAFFLE_MODE"][0] = False
                else:
                    obs_info_table["WAFFLE_MODE"][0] = False

                # If waffle mode is not used, check if more than 1 cube was taken, if not, mark failed sequence
                if len(t_coro_frames) == 0 and not obs_info_table["WAFFLE_MODE"][0]:
                    obs_info_table["FAILED_SEQ"][0] = True

                # Check if at least one center frame exists
                if len(t_center_frames) == 0 or len(t_flux_frames) == 0:
                    obs_info_table["FAILED_SEQ"][0] = True

                # print(target, date, filt, len(t_coro_frames), len(t_center_frames), len(t_flux_frames))
                if not obs_info_table["WAFFLE_MODE"][0]:
                    active_science_files = t_coro_frames
                else:
                    active_science_files = t_center_frames

                if len(active_science_files) == 0:
                    active_science_files = t_flux_frames

                number_files_in_seq = len(active_science_files)
                middle_index = int(number_files_in_seq // 2.0)
                obs_info_table["TOTAL_EXPTIME"] = (
                    np.sum(
                        active_science_files["EXPTIME"] * active_science_files["DET NDIT"]
                    )
                    / 60.0
                )

                obs_info_table["ROTATION"] = (
                    np.abs(active_science_files["TEL PARANG END"][-1] - active_science_files["TEL PARANG START"][0])
                )

                obs_info_table["DIT"] = active_science_files["EXPTIME"][middle_index]
                obs_info_table["NDIT"] = active_science_files["DET NDIT"][middle_index]

                if len(active_science_files.group_by("EXPTIME").groups.keys) != 1:
                    obs_info_table["SCI_DIT_FLAG"] = True

                obs_info_table["NCUBES"] = number_files_in_seq

                # Compute mean and std dev of Tau and FWHM
                FWHM = (
                    active_science_files["AMBI_FWHM_START"]
                    + active_science_files["AMBI_FWHM_END"]
                ) / 2.0
                obs_info_table["MEAN_TAU"] = np.mean(active_science_files["AMBI_TAU"])
                obs_info_table["STDDEV_TAU"] = np.std(active_science_files["AMBI_TAU"])
                obs_info_table["MEAN_FWHM"] = np.mean(FWHM)
                obs_info_table["STDDEV_FWHM"] = np.std(FWHM)
                obs_info_table["MEAN_AIRMASS"] = np.mean(
                    active_science_files["AIRMASS"]
                )
                obs_info_table["DEROTATOR_MODE"] = active_science_files[
                    "DEROTATOR_MODE"
                ][-1]
                obs_info_table["ND_FILTER"] = active_science_files["ND_FILTER"][
                    middle_index
                ]
                try:
                    obs_info_table["ND_FILTER_FLUX"] = t_flux_frames["ND_FILTER"][-1]
                except IndexError:
                    obs_info_table["ND_FILTER_FLUX"] = "N/A"
                obs_info_table["OBS_PROG_ID"] = active_science_files["OBS_PROG_ID"][
                    middle_index
                ]
                obs_info_table["OBS_ID"] = active_science_files["OBS_ID"][middle_index]
                
                # TODO: Add proper computation of total file size
                # obs_info_table["TOTAL_FILE_SIZE"] = np.sum(
                #     active_science_files["FILE_SIZE"]
                # )
                obs_info_table["MJD_MEAN"] = np.mean(active_science_files["MJD_OBS"])

                if counter == 0:  # Create table from one row for first iteration
                    table_of_obs = Table(
                        rows=target,
                        names=target.colnames,
                        dtype=dtypes,
                        meta=target.meta,
                    )
                    for idx in range(len(obs_info_table.columns)):
                        table_of_obs.add_column(obs_info_table.columns[idx])
                    counter += 1
                else:
                    new_row_table = Table(
                        rows=target,
                        names=target.colnames,
                        dtype=dtypes,
                        meta=target.meta,
                    )
                    for idx in range(len(obs_info_table.columns)):
                        new_row_table.add_column(obs_info_table.columns[idx])
                    # simbad_table.add_row(query_result['MAIN_ID', 'RA', 'DEC', 'COO_ERR_MAJA', 'COO_ERR_MINA'][0])
                    table_of_obs.add_row(new_row_table[0])
    try:
        col_number_of_obs = Table.Column(
            name="NUMBER_OF_OBS", data=number_of_observations_list, dtype="i4"
        )
        table_of_targets.add_column(col_number_of_obs)
    except ValueError:
        pass



    return table_of_obs, table_of_targets


# def plot_sparta_r0(table_of_observation): #, target_name, obs_band, date):
#     target_sparta = retrieve_observations_from_name(table_of_atmo, target_name=target_name, show_in_browser=True)
#     #ow_mask = np.logical_and.reduce((table_of_obs['MAIN_ID']==target_name, table_of_obs['DATE_SHORT']==date, table_of_obs['DB_FILTER']==obs_band))
#     #row = table_of_obs[row_mask]
#     #mjd_start = row['OBS_START'][0]
#     #mjd_end = row['OBS_END'][0]
#     #sparta_mask = np.logical_and(table_of_atmo['MJD_OBS'] > mjd_start, table_of_atmo['MJD_OBS'] < mjd_end)
#     #conditions = table_of_atmo[sparta_mask]
#     time_obs = Time(target_sparta['MJD'], format='mjd')
#     plt.plot_date(time_obs.plot_date, target_sparta['STREHL'])
