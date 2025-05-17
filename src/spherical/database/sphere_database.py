from collections.abc import Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.simbad import Simbad
from tqdm import tqdm

from spherical.database.database_utils import convert_table_to_little_endian, retry_query
from spherical.database.ifs_observation import IFSObservation
from spherical.database.irdis_observation import IRDISObservation


def _normalise_filter_column(tbl):
    """Ensure a unified 'FILTER' column exists (alias for DB_FILTER / IFS_MODE)."""
    if "FILTER" in tbl.colnames:
        return                         # already present
    if "DB_FILTER" in tbl.colnames:
        tbl["FILTER"] = tbl["DB_FILTER"]
    elif "IFS_MODE" in tbl.colnames:
        tbl["FILTER"] = tbl["IFS_MODE"]


class Sphere_database(object):
    """
    
    """

    def __init__(
        self, table_of_observations=None, table_of_files=None, instrument="irdis"
    ):
        if table_of_observations is not None:
            _normalise_filter_column(table_of_observations)

        # ---------- 1) basic initialisation (unchanged) ------------------------
        self.instrument = instrument.lower()
        if self.instrument not in ("irdis", "ifs"):
            raise ValueError("Only irdis and ifs instruments implemented.")

        # unified keyword works for *both* tables from here on
        self._filter_keyword = "FILTER"
        
        # ---------- 2) store tables (unchanged helper) -------------------------
        self.table_of_observations = convert_table_to_little_endian(table_of_observations)
        self.table_of_files       = convert_table_to_little_endian(table_of_files)

        # ---------- 3) keep rows only for the chosen instrument ----------------
        if "INSTRUMENT" in self.table_of_observations.colnames:
            mask_instr = self.table_of_observations["INSTRUMENT"] == self.instrument
            self.table_of_observations = self.table_of_observations[mask_instr]

        # ---------- 4) SHUTTER column type fix  --------------------------------
        if isinstance(self.table_of_files["SHUTTER"][0], str):
            self.table_of_files["SHUTTER"] = [s.lower() in ("true", "t", "1")
                                               for s in self.table_of_files["SHUTTER"]]

        # ---------- 5) flag column name (HCI_READY ↔ ~FAILED_SEQ) ---------------
        self._ready_flag = "HCI_READY" if "HCI_READY" in self.table_of_observations.colnames else "FAILED_SEQ"

        self._not_usable_observations_mask = self._mask_not_usable_observations(5.0)

        # ---------- 6) build lists of “summary” keys ---------------------------
        def _keys(base):
            """Replace placeholders with actual column names that exist."""
            out = []
            for item in base:
                if item == "{FILTER}":
                    out.append(self._filter_keyword)
                elif item == "{READY}":
                    out.append(self._ready_flag)
                else:
                    out.append(item)
            # silently drop keys not present in the table
            return [k for k in out if k in self.table_of_observations.colnames]

        base_head = ["MAIN_ID"]
        base_tail = ["NIGHT_START", "{FILTER}", "WAFFLE_MODE", "{READY}", "DEROTATOR_MODE", "PRIMARY_SCIENCE"]

        self._keys_for_summary = _keys(
            base_head
            + [
                "ID_GAIA_DR3", "ID_HIP", "RA", "DEC", "OTYPE", "SP_TYPE", "FLUX_H",
                "STARS_IN_CONE",
            ]
            + base_tail
            + ["TOTAL_EXPTIME_SCI", "ROTATION", "MEAN_FWHM", "OBS_PROG_ID", "TOTAL_FILE_SIZE_MB"]
        )

        self._keys_for_obslog_summary = _keys(
            base_head
            + base_tail
            + ["DIT", "NDIT", "NCUBES", "TOTAL_EXPTIME_SCI", "TOTAL_EXPTIME_FLUX",
               "ROTATION", "MEAN_FWHM", "MEAN_TAU", "OBS_PROG_ID"]
        )

        self._keys_for_short_summary = _keys(
            base_head
            + base_tail
            + ["TOTAL_EXPTIME_SCI", "ROTATION", "MEAN_FWHM", "OBS_PROG_ID"]
        )

        self._keys_for_medium_summary = _keys(
            base_head
            + base_tail
            + ["FLUX_H", "DIT", "NDIT", "TOTAL_EXPTIME_SCI",
               "ROTATION", "MEAN_FWHM", "STDDEV_FWHM", "OBS_PROG_ID"]
        )

        if table_of_files is not None:
            self._non_instrument_mask = self._mask_non_instrument_files()
            self._non_science_file_mask = self._mask_non_science_files()
            self._calibration = self._get_calibration()

    def _mask_bad_values(self):
        self.table_of_files = Table(self.table_of_files, masked=True)
        for key in self.table_of_files.keys():
            self.table_of_files[key].mask = self.table_of_files[key] == -10000.0

    def _mask_not_usable_observations(self, minimum_total_exposure_time=0.0):
        mask_short_exp = self.table_of_observations["TOTAL_EXPTIME_SCI"] < minimum_total_exposure_time

        if self._ready_flag == "HCI_READY":
            mask_bad_flag = ~self.table_of_observations["HCI_READY"]
        else:                                 # legacy
            mask_bad_flag = self.table_of_observations["FAILED_SEQ"] == True

        mask_field_stab = self.table_of_observations["DEROTATOR_MODE"] == "FIELD"

        return np.logical_or.reduce([mask_bad_flag, mask_field_stab, mask_short_exp])

    def _mask_non_instrument_files(self):
        # non_corrupt = self.table_of_files["FILE_SIZE"] > 0.0

        mask = np.logical_and(
            self.table_of_files["DET_ID"] == self.instrument.upper(),
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

        if self.instrument == "irdis":
            calibration["FLAT"] = files[files["DPR_TYPE"] == "FLAT,LAMP"]
            distortion = files[files["DPR_TYPE"] == "LAMP,DISTORT"]
            calibration["DISTORTION"] = distortion
        elif self.instrument == "ifs":
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
        elif summary == "SHORT":
            table_of_observations[self._keys_for_short_summary].show_in_browser(jsviewer=True)
        elif summary == "MEDIUM":
            table_of_observations[self._keys_for_medium_summary].show_in_browser(jsviewer=True)
        elif summary == "OBSLOG":
            table_of_observations[self._keys_for_medium_summary].show_in_browser(jsviewer=True)
        else:
            table_of_observations.show_in_browser(jsviewer=True)

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

        # Check which instrument was used for the observation
        instrument = observation['INSTRUMENT'][0]
        if instrument == "ifs":
            filter_keyword = "IFS_MODE"
        elif instrument == "irdis":
            filter_keyword = "DB_FILTER"

        filter_selection = file_table[filter_keyword] == obs_band
        date_selection = file_table["NIGHT_START"] == date
        file_selection = np.logical_and(filter_selection, date_selection)
        file_table = file_table[file_selection]

        # If irdis, check if polarimetry was used and filter by DPR_TECH
        if instrument == "irdis":
            tech_col = np.char.upper(file_table["DPR_TECH"].astype(str))
            if observation['POLARIMETRY'][0]:
                file_table = file_table[np.char.find(tech_col, "POLARIMETRY") >= 0]
            else:
                file_table = file_table[np.char.find(tech_col, "POLARIMETRY") == -1]

        file_table.sort("MJD_OBS")

        if instrument == "irdis":
            obs = IRDISObservation(observation, file_table, self._calibration)
        elif instrument == "ifs":
            obs = IFSObservation(observation, file_table, self._calibration)
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