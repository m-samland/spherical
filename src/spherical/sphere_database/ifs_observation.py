import os
from collections import OrderedDict
from typing import Dict, Optional

from astropy import units as u
from astropy.table import Table, vstack
from astropy.time import Time, TimeDelta

from .database_utils import filter_table, find_nearest


class IFSObservation:
    def __init__(self, observation: Table, science_files: Table, calibration_table: Dict[str, Table]):
        self.observation = observation
        self._science_files = science_files
        self._calibration = calibration_table

        self._target_name = "_".join(observation["MAIN_ID"][0].split())
        self._file_prefix = f"{self._target_name}_{observation['NIGHT_START'][0]}_{observation['IFS_MODE'][0]}_"

        self._object_path_structure = os.path.join(
            self._target_name,
            observation["IFS_MODE"][0],
            observation["NIGHT_START"][0]
        )

        self.filter = observation["IFS_MODE"][0]
        self.frames: Dict[str, Optional[Table]] = self._sort_by_category(science_files)

        self.frames["FLAT"] = self._safe_call(self._get_flats, default=None, message="No flats found.")

        if not observation["WAFFLE_MODE"][0]:
            self.frames["CENTER_BEFORE"], self.frames["CENTER_AFTER"] = self._center_before_after_coronagraphy()

        self.frames["WAVECAL"] = self._safe_call(self._get_wavecal, default=None, message="No WAVECAL found.")
        self.frames["SPECPOS"] = self._safe_call(self._get_specpos, default=None, message="No SPECPOS found.")

        self.background = {}
        self._set_background_frames("SCIENCE", "CENTER")
        self._set_background_frames("FLUX", "FLUX")
        self._set_background_frames("WAVECAL", "WAVECAL")
        self._set_background_frames("SPECPOS", "SPECPOS")

        self.all_frames = vstack([frame for frame in self.frames.values() if frame is not None and len(frame) > 0])
        self.data_quality_flags = self._set_data_quality_flags()

    def __repr__(self):
        return f"ID: {self.observation['MAIN_ID'][0]}\nDATE: {self.observation['NIGHT_START'][0]}\nFILTER: {self.observation['IFS_MODE'][0]}\n"

    def mjd_observation(self) -> Time:
        middle_index = len(self._science_files) // 2
        return Time(self._science_files["MJD_OBS"][middle_index], format="mjd", scale="utc")

    def _safe_call(self, func, default, message: str):
        try:
            return func()
        except Exception:
            print(message)
            return default

    def _center_before_after_coronagraphy(self):
        coro_start = self.frames["CORO"]["MJD_OBS"][0]
        coro_end = self.frames["CORO"]["MJD_OBS"][-1]
        center_frames = self.frames["CENTER"]

        center_before = center_frames[center_frames["MJD_OBS"] < coro_start]
        center_after = center_frames[center_frames["MJD_OBS"] > coro_end]
        return center_before, center_after

    def _closest_files_with_single_date(self, table: Table, mjd: float) -> Table:
        if len(table) < 2:
            return table
        idx = find_nearest(table["MJD_OBS"], mjd)
        date = table[idx]["NIGHT_START"]
        return filter_table(table, {"NIGHT_START": date})

    def _sort_by_category(self, science_files: Table) -> Dict[str, Table]:
        return {
            "CORO": filter_table(science_files, {"DPR_TYPE": "OBJECT"}),
            "CENTER": filter_table(science_files, {"DPR_TYPE": "OBJECT,CENTER"}),
            "FLUX": filter_table(science_files, {"DPR_TYPE": "OBJECT,FLUX"})
        }

    def _get_flats(self) -> Table:
        mode = self.observation["IFS_MODE"][0]
        mode_short = "YJH" if mode == "OBS_H" else "YJ"
        flats = filter_table(
            self._calibration.get(f"FLAT_{mode_short}", Table()),
            {"IFS_MODE": f"CAL_BB_2_{mode_short}"}
        )
        return self._closest_files_with_single_date(flats, self.mjd_observation().value)

    def _get_wavecal(self) -> Table:
        wavecal = filter_table(
            self._calibration.get("WAVECAL", Table()), {"IFS_MODE": self.observation["IFS_MODE"][0]}
        )
        return self._closest_files_with_single_date(wavecal, self.mjd_observation().value)

    def _get_specpos(self) -> Table:
        specpos = filter_table(
            self._calibration.get("SPECPOS", Table()), {"IFS_MODE": self.observation["IFS_MODE"][0]}
        )
        if len(specpos) > 0:
            specpos = self._closest_files_with_single_date(specpos, self.mjd_observation().value)
        return specpos

    def _get_darks(self, exposure_time, ND_filter) -> Dict[str, Table]:
        bkg_files = self._calibration.get("BACKGROUND", Table())
        mode = self.observation["IFS_MODE"][0]

        background = {
            "SKY": filter_table(self._science_files, {
                "DPR_TYPE": "SKY", "IFS_MODE": mode,
                "ND_FILTER": ND_filter, "EXPTIME": exposure_time
            }),
            "BACKGROUND": self._closest_files_with_single_date(
                filter_table(bkg_files, {
                    "DPR_TYPE": "DARK,BACKGROUND",
                    "ND_FILTER": self.frames["CENTER"]["ND_FILTER"][-1] if len(self.frames["CENTER"]) > 0 else ND_filter,
                    "EXPTIME": exposure_time
                }), self.mjd_observation().value),
            "DARK": self._closest_files_with_single_date(
                filter_table(bkg_files, {
                    "DPR_TYPE": "DARK",
                    "EXPTIME": exposure_time
                }), self.mjd_observation().value)
        }
        return background

    def _pick_dark_type(self, background: Dict[str, Table]) -> Table:
        for key in ["SKY", "BACKGROUND", "DARK"]:
            if key in background and len(background[key]) > 0:
                return background[key]
        raise FileNotFoundError("No dark found.")

    def _set_background_frames(self, label: str, source: str):
        try:
            if source not in self.frames or len(self.frames[source]) == 0:
                raise ValueError(f"Missing source frames for {source}")
            self.background[label] = self._get_darks(
                exposure_time=self.frames[source]["EXPTIME"][-1],
                ND_filter=self.frames[source]["ND_FILTER"][-1]
            )
            self.frames[f"BG_{label}"] = self._pick_dark_type(self.background[label])
        except Exception:
            self.background[label] = None
            colnames = self._science_files.colnames
            dtypes = [self._science_files[col].dtype for col in colnames]
            empty_table = Table(names=colnames, dtype=dtypes)
            self.frames[f"BG_{label}"] = empty_table

    def _set_data_quality_flags(self) -> Table:
        def time_diff(frame_key: str) -> float:
            return (self.mjd_observation() - Time(self.frames[frame_key]["MJD_OBS"][0], format="mjd")).value

        flags = OrderedDict()
        for key in ["SCIENCE", "FLUX"]:
            try:
                bg_key = f"BG_{key}"
                flags[bg_key] = [self.frames[bg_key]["DPR_TYPE"][0]]
                flags[f"{bg_key}_TIMEDIFF"] = [time_diff(bg_key)]
                flags[f"{bg_key}_WARNING"] = [flags[f"{bg_key}_TIMEDIFF"][0] > 1]
            except Exception:
                flags[bg_key] = ["NA"]
                flags[f"{bg_key}_TIMEDIFF"] = [TimeDelta(-10000 * u.day).value]
                flags[f"{bg_key}_WARNING"] = [True]

        for key in ["FLAT", "SPECPOS", "WAVECAL"]:
            try:
                flags[f"{key}_TIMEDIFF"] = [time_diff(key)]
                flags[f"{key}_WARNING"] = [flags[f"{key}_TIMEDIFF"][0] > 1]
            except Exception:
                flags[f"{key}_TIMEDIFF"] = [TimeDelta(-10000 * u.day).value]
                flags[f"{key}_WARNING"] = [True]

        return Table(flags)

    def check_frames(self):
        required_keys = ["FLAT", "BG_SCIENCE", "BG_FLUX", "CENTER", "FLUX", "WAVECAL", "SPECPOS"]
        for key in required_keys:
            if key not in self.frames or len(self.frames[key]) < 1:
                raise FileNotFoundError(f"No {key} file for observation {self!r}")

        if not self.observation["WAFFLE_MODE"][0] and ("CORO" not in self.frames or len(self.frames["CORO"]) < 1):
            raise FileNotFoundError(f"No coronagraphic file for observation {self!r}")

    def get_reduction_info(self, reduction_directory: str) -> Table:
        path = os.path.join(reduction_directory, self._object_path_structure, "reduction_info.fits")
        try:
            return Table.read(path)
        except FileNotFoundError:
            return Table()
