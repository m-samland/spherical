from collections.abc import Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.simbad import Simbad
from tqdm import tqdm

from spherical.sphere_database.ifs_observation import IFS_observation
from spherical.sphere_database.irdis_observation import IRDIS_observation
from spherical.sphere_database.target_table import retry_query


class Sphere_database(object):
    """
    
    """

    def __init__(
        self, table_of_observations=None, table_of_files=None, instrument="IRDIS"
    ):
        self.instrument = instrument
        if instrument == "IRDIS":
            self._filter_keyword = "DB_FILTER"
        elif instrument == "IFS":
            self._filter_keyword = "IFS_MODE"
        else:
            raise ValueError("Only IRDIS and IFS instruments implemented.")

        self.table_of_observations = table_of_observations
        self.table_of_files = table_of_files

        if isinstance(table_of_files["SHUTTER"][0], str):
            shutter = []
            for value in table_of_files["SHUTTER"]:
                if value.lower() in ["true", "t", "1"]:
                    shutter.append(True)
                else:
                    shutter.append(False)
            table_of_files["SHUTTER"] = shutter
        # True == 'Open'

        # if table_of_files is not None:
        #     try:
        #         if self.table_of_files.masked is False:
        #             self._mask_bad_values()
        #     except AttributeError:
        #         pass

        self._not_usable_observations_mask = self._mask_not_usable_observations(5.0)

        self._keys_for_summary = [
            "MAIN_ID",
            "HD_ID",
            "HIP_ID",
            "RA",
            "DEC",
            "OTYPE",
            "SP_TYPE",
            "FLUX_H",
            "STARS_IN_CONE",
            "NEAREST_STAR",
            "SEPARATION_NEAREST",
            "NEAREST_OTYPE",
            "NEAREST_STYPE",
            "NEAREST_FLUX_H",
            "COMPANION_NAME",
            "NUMBER_COMPANIONS",
            "NIGHT_START",
            self._filter_keyword,
            "WAFFLE_MODE",
            "FAILED_SEQ",
            "TOTAL_EXPTIME",
            "DEROTATOR_MODE",
            "MEAN_FWHM",
            "OBS_PROG_ID",
        ]
        self._keys_for_obslog_summary = [
            "MAIN_ID",
            self._filter_keyword,
            "NIGHT_START",
            "WAFFLE_MODE",
            "FAILED_SEQ",
            "DEROTATOR_MODE",
            "DIT",
            "NDIT",
            "NCUBES",
            "TOTAL_EXPTIME",
            "MEAN_FWHM",
            "MEAN_TAU",
            "OBS_PROG_ID",
        ]
        self._keys_for_short_summary = [
            "MAIN_ID",
            self._filter_keyword,
            "NIGHT_START",
            "WAFFLE_MODE",
            "FAILED_SEQ",
            "DEROTATOR_MODE",
            "TOTAL_EXPTIME",
            "MEAN_FWHM",
            "OBS_PROG_ID",
        ]
        self._keys_for_medium_summary = [
            "MAIN_ID",
            self._filter_keyword,
            "NIGHT_START",
            "WAFFLE_MODE",
            "FAILED_SEQ",
            "DEROTATOR_MODE",
            "FLUX_H",
            "DIT",
            "NDIT",
            "TOTAL_EXPTIME",
            "MEAN_FWHM",
            "STDDEV_FWHM",
            "OBS_PROG_ID",
            "COMPANION_NAME",
            "NEAREST_STAR",
            "SEPARATION_NEAREST",
        ]

        if table_of_files is not None:
            self._non_instrument_mask = self._mask_non_instrument_files()
            self._non_science_file_mask = self._mask_non_science_files()
            self._calibration = self._get_calibration()

    def _mask_bad_values(self):
        self.table_of_files = Table(self.table_of_files, masked=True)
        for key in self.table_of_files.keys():
            self.table_of_files[key].mask = self.table_of_files[key] == -10000.0

    def _mask_not_usable_observations(self, minimum_total_exposure_time=0.0):
        mask_short_exposure = (
            self.table_of_observations["TOTAL_EXPTIME"] < minimum_total_exposure_time
        )
        mask_failed = self.table_of_observations["FAILED_SEQ"] == 1
        mask_field_stabilized = self.table_of_observations["DEROTATOR_MODE"] == "FIELD"
        not_usable_observation_mask = np.logical_or.reduce(
            [mask_failed, mask_field_stabilized, mask_short_exposure]
        )

        return not_usable_observation_mask

    def _mask_non_instrument_files(self):
        # non_corrupt = self.table_of_files["FILE_SIZE"] > 0.0

        mask = np.logical_and(
            self.table_of_files["DET_ID"] == self.instrument,
            self.table_of_files["READOUT_MODE"] == "Nondest",
        )
        # files = np.logical_and(non_corrupt, mask)
        return ~mask

    def _mask_non_science_files(self):
        science_mask = np.logical_and.reduce(
            (
                self.table_of_files["DEC"] != -10000,
                self.table_of_files["DEC"] != 0.0,
                self.table_of_files["DPR_TYPE"] != "DARK",
                self.table_of_files["DPR_TYPE"] != "FLAT,LAMP",
                self.table_of_files["DPR_TYPE"] != "OBJECT,ASTROMETRY",
                self.table_of_files["DPR_TYPE"] != "STD",
                self.table_of_files["CORO"] != "N/A",
                self.table_of_files["READOUT_MODE"] == "Nondest",
                self.table_of_files["SHUTTER"] == True,
            )
        )

        science_files = np.logical_and(science_mask, ~self._non_instrument_mask)

        return ~science_files

    def _get_calibration(self):
        calibration = {}
        files = self.table_of_files[~self._non_instrument_mask]
        dark_and_background_selection = np.logical_or(
            files["DPR_TYPE"] == "DARK", files["DPR_TYPE"] == "DARK,BACKGROUND"
        )
        dark_and_background = files[dark_and_background_selection]
        calibration["BACKGROUND"] = dark_and_background

        if self.instrument == "IRDIS":
            calibration["FLAT"] = files[files["DPR_TYPE"] == "FLAT,LAMP"]
            distortion = files[files["DPR_TYPE"] == "LAMP,DISTORT"]
            calibration["DISTORTION"] = distortion
        elif self.instrument == "IFS":
            calibration["SPECPOS"] = files[files["DPR_TYPE"] == "SPECPOS,LAMP"]
            calibration["WAVECAL"] = files[files["DPR_TYPE"] == "WAVE,LAMP"]
            flat_selection_yj = np.logical_or.reduce(
                [
                    files["IFS_MODE"] == "CAL_BB_2_YJ",
                    files["IFS_MODE"] == "CAL_NB1_1_YJ",
                    files["IFS_MODE"] == "CAL_NB2_1_YJ",
                    files["IFS_MODE"] == "CAL_NB3_1_YJ",
                    files["IFS_MODE"] == "CAL_NB4_2_YJ",
                ]
            )
            flat_selection_yjh = np.logical_or.reduce(
                [
                    files["IFS_MODE"] == "CAL_BB_2_YJH",
                    files["IFS_MODE"] == "CAL_NB1_1_YJH",
                    files["IFS_MODE"] == "CAL_NB2_1_YJH",
                    files["IFS_MODE"] == "CAL_NB3_1_YJH",
                    files["IFS_MODE"] == "CAL_NB4_2_YJH",
                ]
            )
            calibration["FLAT_YJ"] = files[flat_selection_yj]
            calibration["FLAT_YJH"] = files[flat_selection_yjh]

            # ['CAL_BB_2', 'CAL_NB1_1', 'CAL_NB2_1', 'CAL_NB3_1', 'CAL_NB4_2']

        return calibration

    def return_usable_only(self):
        return self.table_of_observations[~self._not_usable_observations_mask].copy()

    def show_in_browser(self, summary=None, usable_only=False):
        if usable_only:
            table_of_observations = self.table_of_observations[
                ~self._not_usable_observations_mask
            ].copy()
        else:
            table_of_observations = self.table_of_observations.copy()

        if summary == "NORMAL":
            table_of_observations[self._keys_for_summary].show_in_browser(jsviewer=True)
            # table_of_observations[self._keys_for_summary].show_in_notebook()
        elif summary == "SHORT":
            table_of_observations[self._keys_for_short_summary].show_in_browser(jsviewer=True)
            # table_of_observations[self._keys_for_short_summary].show_in_notebook()
        elif summary == "MEDIUM":
            table_of_observations[self._keys_for_medium_summary].show_in_browser(jsviewer=True)
            # table_of_observations[self._keys_for_medium_summary].show_in_notebook()
        elif summary == "OBSLOG":
            table_of_observations[self._keys_for_medium_summary].show_in_browser(jsviewer=True)
            # table_of_observations[self._keys_for_obslog_summary].show_in_notebook()
        else:
            table_of_observations.show_in_browser(jsviewer=True)
            # table_of_observations.show_in_notebook()

    def observations_from_name_SIMBAD(
        self, target_name, summary=None, usable_only=False, query_radius=5.0
    ):
        if usable_only:
            table_of_observations = self.table_of_observations[
                ~self._not_usable_observations_mask
            ].copy()
        else:
            table_of_observations = self.table_of_observations.copy()

        sphere_list_coordinates = SkyCoord(
            ra=table_of_observations["RA_DEG"], dec=table_of_observations["DEC_DEG"], 
            unit=(u.degree, u.degree)
        )

        if not isinstance(target_name, Sequence) or isinstance(target_name, str):
            target_name = [target_name]
        results = []
        for target in target_name:
            query_result = retry_query(
                Simbad.query_object,
                verbose=False,
                number_of_retries=3,
                object_name=target,
            )
            results.append(query_result)
        if len(results) > 1:
            query_result = vstack(results)

        # print(query_result)

        queried_coordinates = SkyCoord(
            ra=query_result["ra"], dec=query_result["dec"], unit=(u.hourangle, u.deg)
        )

        # print(table_of_observations)

        selection_mask = np.zeros(len(table_of_observations), dtype="bool")
        for object_coordinates in queried_coordinates:
            mask = (
                object_coordinates.separation(sphere_list_coordinates)
                < query_radius * u.arcmin
            )
            selection_mask[mask] = True
        table_of_observations = table_of_observations[selection_mask]

        # print(table_of_observations)

        if summary == "NORMAL":
            return table_of_observations[self._keys_for_summary]
        elif summary == "SHORT":
            return table_of_observations[self._keys_for_short_summary]
        elif summary == "MEDIUM":
            return table_of_observations[self._keys_for_medium_summary]
        elif summary == "OBSLOG":
            return table_of_observations[self._keys_for_obslog_summary]
        else:
            return table_of_observations

    def get_observation_SIMBAD(
        self,
        target_name,
        obs_band=None,
        date=None,
        summary=None,
        usable_only=False,
        query_radius=5.0,
    ):
        observations = self.observations_from_name_SIMBAD(
            target_name,
            summary=summary,
            usable_only=usable_only,
            query_radius=query_radius,
        )

        # print(observations)
        # print("-----------------")

        if obs_band is None:
            select_filter = np.ones(len(observations), dtype="bool")
        else:
            select_filter = observations[self._filter_keyword] == obs_band

        if date is None:
            select_date = np.ones(len(observations), dtype="bool")
        else:
            select_date = observations["NIGHT_START"] == date

        select_observation = np.logical_and(select_filter, select_date)

        # print(select_observation)
        # print("=========")
        
        return observations[select_observation].copy()

    def retrieve_observation(self, target_name, obs_band=None, date=None):
        observation = self.get_observation_SIMBAD(target_name, obs_band, date)

        science_files = self.table_of_files[~self._non_science_file_mask]
        file_coordinates = SkyCoord(
            ra=science_files["RA"] * u.degree, dec=science_files["DEC"] * u.degree
        )
        target_coordinates = SkyCoord(
            ra=observation["RA_HEADER"] * u.degree, dec=observation["DEC_HEADER"] * u.degree
        )
        if len(target_coordinates) == 0:
            raise ValueError("No Targets found with the given information.")
        # 1.15 arcmin to include sky frames displaced from the sequence (e.g. 1.12 for HD135344)

        target_selection = (
            target_coordinates.separation(file_coordinates) < 1.2 * u.arcmin
        )
        file_table = science_files[target_selection]

        filter_selection = file_table[self._filter_keyword] == obs_band
        date_selection = file_table["NIGHT_START"] == date
        file_selection = np.logical_and(filter_selection, date_selection)
        file_table = file_table[file_selection]
        file_table.sort("MJD_OBS")

        if self.instrument == "IRDIS":
            obs = IRDIS_observation(observation, file_table, self._calibration)
        elif self.instrument == "IFS":
            obs = IFS_observation(observation, file_table, self._calibration)
        return obs

    def retrieve_observation_object_list(self, table_of_reduction_targets):
        observation_object_list = []
        for observation in tqdm(table_of_reduction_targets):
            observation_object = self.retrieve_observation(
                observation["MAIN_ID"],
                observation[self._filter_keyword],
                observation["NIGHT_START"],
            )
            observation_object_list.append(observation_object)

        return observation_object_list

    def plot_observations(self):
        pass


"""
table_of_observations_gto = Table.read(
    '../../tables/table_of_observations_gto_fixed.fits')
table_of_files = Table.read(
    '../../tables/table_of_files_gto.fits')
# # table_of_observations_non_gto = Table.read(
# #     'table_of_observations_non_gto_2018_02_16.fits')


database_gto = Sphere_database(
    table_of_observations_gto, table_of_files)

a = database_gto.retrieve_IRDIS_observation(
    '51 Eri',
    'DB_K12',
    '2015-09-25')

b = database_gto.retrieve_IRDIS_observation(
    '51 Eri',
    'DB_H23',
    '2016-12-13')

observation_list = [a, b]

static_calibration = {}
static_calib_dir = '/mnt/fhgfs/sphere_shared/automatic_reduction/static_calibration/'
static_calibration['FILTER_TABLE'] = {
    'DB_H23': os.path.join(static_calib_dir, 'filter_table_H23IRD_FILTER_TABLE.fits'),
    'DB_K12': os.path.join(static_calib_dir, 'filter_table_K12IRD_FILTER_TABLE.fits'),
    'DB_Y23': os.path.join(static_calib_dir, 'filter_table_Y23IRD_FILTER_TABLE.fits'),
    'BB_J': os.path.join(static_calib_dir, 'filter_table_H23IRD_FILTER_TABLE.fits'),
    'BB_H': os.path.join(static_calib_dir, 'filter_table_H23IRD_FILTER_TABLE.fits'),
    'BB_Ks': os.path.join(static_calib_dir, 'filter_table_H23IRD_FILTER_TABLE.fits'),
    'DB_NDH23': os.path.join(static_calib_dir, 'filter_table_H23IRD_FILTER_TABLE.fits')}

static_calibration['POINT_PATTERN'] = os.path.join(
    static_calib_dir, 'irdis_distortion_points_DRHIRD_POINT_PATTERN.dat')  # In static_calib folder
static_calibration['CENTERING_MASK'] = os.path.join(
    static_calib_dir, 'waffle_lowmaskgoodIRD_STATIC_BADPIXELMAP.fits')  # In static_calib folder

a.write_sofs('test', static_calibration)
b.write_sofs('test', static_calibration)
"""

# t = database_gto.observations_from_name('51 Eri', summary='SHORT')
# t
#
#
# obs = database_gto.get_observation(
#     target_name='51 Eri',
#     obs_band='DB_K12',
#     date='2015-09-25')
#
# database_gto.table_of_files[database_gto._not_usable_irdis_mask]
#
# obs
# # # database_non_gto = Sphere_database(table_of_observations_non_gto)
# #
# #
# # table_of_files = table_of_files[table_of_files['FILE_SIZE'] > 0.0]
