import glob
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.ascii.core import InconsistentTableError
from astropy.table import Column, Table, hstack, join, vstack
from astropy.time import Time, TimeDelta
from astroquery.simbad import Simbad
from spherical.pipeline.embed_shell import ipsh

from .database_utils import filter_table, find_nearest, make_selection_mask


class IFS_observation(object):
    def __init__(self, observation=None, science_files=None, calibration_table=None):
        self.observation = observation
        self._calibration = calibration_table
        self._science_files = science_files

        self._target_name = " ".join(self.observation["MAIN_ID"][0].split())
        self._target_name = self._target_name.replace(" ", "_")
        self._file_prefix = "{0}_{1}_{2}_".format(
            self._target_name,
            self.observation["NIGHT_START"][0],
            self.observation["IFS_MODE"][0],
        )

        self._object_path_structure = os.path.join(
            self._target_name,
            self.observation["IFS_MODE"][0],
            self.observation["NIGHT_START"][0],
        )
        self.filter = self.observation["IFS_MODE"][0]
        self.frames = self._sort_by_category(science_files)

        try:
            self.frames["FLAT"] = self._get_flats()
        except:
            print("No flats found.")
            self.frames["FLAT"] = None

        if not observation["WAFFLE_MODE"][0]:
            (
                self.frames["CENTER_BEFORE"],
                self.frames["CENTER_AFTER"],
            ) = self._center_before_after_coronagraphy()

        try:
            self.frames["WAVECAL"] = self._get_wavecal()
        except:
            print("No WAVECAL found.")
            self.frames["WAVECAL"] = None

        try:
            self.frames["SPECPOS"] = self._get_specpos()
        except:
            print("No SPECPOS found.")
            self.frames["SPECPOS"] = None

        # There are three types of background frames:
        # SKY, DARK,BACKGROUND, and DARK
        # And two sets of exposure times and filters
        # SCIENCE and FLUX
        self.background = {}
        try:
            self.background["SCIENCE"] = self._get_darks(
                exposure_time=self.frames["CENTER"]["EXPTIME"][-1],
                ND_filter=self.frames["CENTER"]["ND_FILTER"][-1],
            )
            self.frames["BG_SCIENCE"] = self._pick_dark_type(self.background["SCIENCE"])
        except:
            self.background["SCIENCE"] = None
            dtypes = [
                science_files.dtype[i].__str__()
                for i in range(len(science_files.colnames))
            ]
            self.frames["BG_SCIENCE"] = Table(
                names=science_files.colnames, dtype=dtypes
            )

        try:
            self.background["FLUX"] = self._get_darks(
                exposure_time=self.frames["FLUX"]["EXPTIME"][-1],
                ND_filter=self.frames["FLUX"]["ND_FILTER"][-1],
            )
            self.frames["BG_FLUX"] = self._pick_dark_type(self.background["FLUX"])
        except:
            self.background["FLUX"] = None
            dtypes = [
                science_files.dtype[i].__str__()
                for i in range(len(science_files.colnames))
            ]
            self.frames["BG_FLUX"] = Table(names=science_files.colnames, dtype=dtypes)

        self.all_frames = vstack(list(self.frames.values()))
        self.data_quality_flags = self._set_data_quality_flags()

        try:
            self.background["WAVECAL"] = self._get_darks(
                exposure_time=self.frames["WAVECAL"]["EXPTIME"][-1],
                ND_filter=self.frames["WAVECAL"]["ND_FILTER"][-1],
            )
            self.frames["BG_WAVECAL"] = self._pick_dark_type(self.background["WAVECAL"])
        except:
            self.background["WAVECAL"] = None
            dtypes = [
                science_files.dtype[i].__str__()
                for i in range(len(science_files.colnames))
            ]
            self.frames["BG_WAVECAL"] = Table(
                names=science_files.colnames, dtype=dtypes
            )

        try:
            self.background["SPECPOS"] = self._get_darks(
                exposure_time=self.frames["SPECPOS"]["EXPTIME"][-1],
                ND_filter=self.frames["SPECPOS"]["ND_FILTER"][-1],
            )
            self.frames["BG_SPECPOS"] = self._pick_dark_type(self.background["SPECPOS"])
        except:
            self.background["SPECPOS"] = None
            dtypes = [
                science_files.dtype[i].__str__()
                for i in range(len(science_files.colnames))
            ]
            self.frames["BG_SPECPOS"] = Table(
                names=science_files.colnames, dtype=dtypes
            )

    def __repr__(self):
        return "ID: {}\nDATE: {}\nFILTER: {}\n".format(
            self.observation["MAIN_ID"][0],
            self.observation["NIGHT_START"][0],
            self.observation["IFS_MODE"][0],
        )

    def mjd_observation(self):
        middle_index = int(len(self._science_files) // 2.0)
        return Time(
            self._science_files["MJD_OBS"][middle_index], format="mjd", scale="utc"
        )

    def _center_before_after_coronagraphy(self):
        if self.observation["WAFFLE_MODE"][0] == False:
            beginning_of_coro_seq = self.frames["CORO"]["MJD_OBS"][0]
            end_of_coro_seq = self.frames["CORO"]["MJD_OBS"][-1]

            center_frames = self.frames["CENTER"]
            center_before = center_frames[
                center_frames["MJD_OBS"] < beginning_of_coro_seq
            ]
            center_after = center_frames[center_frames["MJD_OBS"] > end_of_coro_seq]
        return center_before, center_after

    def _closest_files_with_single_date(self, table, mjd):
        if len(table) < 2:
            return table
        if len(table) > 1:
            idx = find_nearest(table["MJD_OBS"], mjd)
            date = table[idx]["NIGHT_START"]
            table = filter_table(table, {"NIGHT_START": date})
        return table

    def _sort_by_category(self, science_files):
        frames = {}
        frames["CORO"] = filter_table(science_files, {"DPR_TYPE": "OBJECT"})
        frames["CENTER"] = filter_table(science_files, {"DPR_TYPE": "OBJECT,CENTER"})
        frames["FLUX"] = filter_table(science_files, {"DPR_TYPE": "OBJECT,FLUX"})

        return frames

    def _get_flats(self):
        if self.observation["IFS_MODE"][0] == "OBS_H":
            mode_short = "YJH"
        elif self.observation["IFS_MODE"][0] == "OBS_YJ":
            mode_short = "YJ"

        flats = filter_table(
            self._calibration["FLAT_{}".format(mode_short)],
            {"IFS_MODE": "CAL_BB_2_{}".format(mode_short)},
        )
        flats = self._closest_files_with_single_date(
            flats, self.mjd_observation().value
        )

        return flats

    def _get_wavecal(self):
        wavecal = filter_table(
            self._calibration["WAVECAL"], {"IFS_MODE": self.observation["IFS_MODE"][0]}
        )
        # indices = np.argsort(np.abs(wavecal["MJD_OBS"] - self.mjd_observation().value))
        # values = np.sort(np.abs(wavecal["MJD_OBS"] - self.mjd_observation().value))
        # wavecal[indices]
        # ipsh()
        wavecal = self._closest_files_with_single_date(
            wavecal, self.mjd_observation().value
        )

        return wavecal

    def _get_specpos(self):
        specpos = filter_table(
            self._calibration["SPECPOS"], {"IFS_MODE": self.observation["IFS_MODE"][0]}
        )
        if len(specpos) > 0:
            specpos = self._closest_files_with_single_date(
                specpos, self.mjd_observation().value
            )

        return specpos

    def _get_darks(self, exposure_time, ND_filter):
        # Should information on observation be read from table or files here?
        # Probably table...
        background_files = self._calibration["BACKGROUND"]
        background = {}
        # ipsh()
        background["SKY"] = filter_table(
            table=self._science_files,
            condition_dictionary={
                "DPR_TYPE": "SKY",
                "IFS_MODE": self.observation["IFS_MODE"][0],
                "ND_FILTER": ND_filter,
                "EXPTIME": exposure_time,
            },
        )

        background["BACKGROUND"] = filter_table(
            background_files,
            condition_dictionary={
                "DPR_TYPE": "DARK,BACKGROUND",
                # 'IFS_MODE': self.observation['IFS_MODE'],
                "ND_FILTER": self.frames["CENTER"]["ND_FILTER"][-1],
                "EXPTIME": exposure_time,
            },
        )

        background["BACKGROUND"] = self._closest_files_with_single_date(
            background["BACKGROUND"], self.mjd_observation().value
        )

        background["DARK"] = filter_table(
            background_files,
            condition_dictionary={"DPR_TYPE": "DARK", "EXPTIME": exposure_time},
        )

        background["DARK"] = self._closest_files_with_single_date(
            background["DARK"], self.mjd_observation().value
        )

        return background

    def _pick_dark_type(self, background_dictionary):
        background = background_dictionary
        if len(background["SKY"]) > 0:
            return background["SKY"]
        elif len(background["BACKGROUND"]) > 0:
            return background["BACKGROUND"]
        elif len(background["DARK"]) > 0:
            return background["DARK"]
        else:
            raise FileNotFoundError("No dark found.")

    def _set_data_quality_flags(self):
        data_quality_flags = OrderedDict()
        try:
            data_quality_flags["BG_SCIENCE"] = [
                self.frames["BG_SCIENCE"]["DPR_TYPE"][0]
            ]
            data_quality_flags["BG_SCIENCE_TIMEDIFF"] = [
                (
                    self.mjd_observation()
                    - Time(self.frames["BG_SCIENCE"]["MJD_OBS"][0], format="mjd")
                ).value
            ]
            data_quality_flags["BG_SCIENCE_WARNING"] = [
                data_quality_flags["BG_SCIENCE_TIMEDIFF"][0] > 1
            ]
        except:
            data_quality_flags["BG_SCIENCE"] = ["NA"]
            data_quality_flags["BG_SCIENCE_TIMEDIFF"] = [
                TimeDelta(-10000 * u.day).value
            ]
            data_quality_flags["BG_SCIENCE_WARNING"] = [True]

        try:
            data_quality_flags["BG_FLUX"] = [self.frames["BG_FLUX"]["DPR_TYPE"][0]]
            data_quality_flags["BG_FLUX_TIMEDIFF"] = [
                (
                    self.mjd_observation()
                    - Time(self.frames["BG_FLUX"]["MJD_OBS"][0], format="mjd")
                ).value
            ]
            data_quality_flags["BG_FLUX_WARNING"] = [
                data_quality_flags["BG_FLUX_TIMEDIFF"][0] > 1
            ]
        except:
            data_quality_flags["BG_FLUX"] = ["NA"]
            data_quality_flags["BG_FLUX_TIMEDIFF"] = [TimeDelta(-10000 * u.day).value]
            data_quality_flags["BG_FLUX_WARNING"] = [True]

        try:
            data_quality_flags["FLAT_TIMEDIFF"] = [
                (
                    self.mjd_observation()
                    - Time(self.frames["FLAT"]["MJD_OBS"][0], format="mjd")
                ).value
            ]
            data_quality_flags["FLAT_WARNING"] = [
                data_quality_flags["FLAT_TIMEDIFF"][0] > 1
            ]
        except:
            data_quality_flags["FLAT_TIMEDIFF"] = [TimeDelta(-10000 * u.day).value]
            data_quality_flags["FLAT_WARNING"] = [True]

        try:
            data_quality_flags["SPECPOS_TIMEDIFF"] = [
                (
                    self.mjd_observation()
                    - Time(self.frames["SPECPOS"]["MJD_OBS"][0], format="mjd")
                ).value
            ]
            data_quality_flags["SPECPOS_WARNING"] = [
                data_quality_flags["SPECPOS_TIMEDIFF"][0] > 1
            ]
        except:
            data_quality_flags["SPECPOS_TIMEDIFF"] = [TimeDelta(-10000 * u.day).value]
            data_quality_flags["SPECPOS_WARNING"] = [True]

        try:
            data_quality_flags["WAVECAL_TIMEDIFF"] = [
                (
                    self.mjd_observation()
                    - Time(self.frames["WAVECAL"]["MJD_OBS"][0], format="mjd")
                ).value
            ]
            data_quality_flags["WAVECAL_WARNING"] = [
                data_quality_flags["WAVECAL_TIMEDIFF"][0] > 1
            ]
        except:
            data_quality_flags["WAVECAL_TIMEDIFF"] = [TimeDelta(-10000 * u.day).value]
            data_quality_flags["WAVECAL_WARNING"] = [True]

        return Table(data_quality_flags)

    def check_frames(self):
        keys = ["FLAT", "BG_SCIENCE", "BG_FLUX", "CENTER", "FLUX", "WAVECAL", "SPECPOS"]
        for key in keys:
            if len(self.frames[key]) < 1:
                raise FileNotFoundError(
                    "No {} file for observation {}".format(key, self.__repr__())
                )
        if self.observation["WAFFLE_MODE"][0] == False:
            if len(self.frames["CORO"]) < 1:
                raise FileNotFoundError(
                    "No coronagraphic file for observation {}".format(
                        key, self.__repr__()
                    )
                )

    def get_reduction_info(self, reduction_directory):
        reduction_info_path = os.path.join(
            reduction_directory, self._object_path_structure, "reduction_info.fits"
        )
        try:
            reduction_info = Table.read(reduction_info_path)
        except FileNotFoundError:
            reduction_info = Table()
        return reduction_info
