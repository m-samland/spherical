import warnings
from typing import Dict, List, Optional, Union

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.simbad import Simbad
from tqdm import tqdm

from spherical.database.database_utils import convert_table_to_little_endian
from spherical.database.ifs_observation import IFSObservation
from spherical.database.irdis_observation import IRDISObservation


def _normalise_filter_column(tbl: Table) -> None:
    """
    Ensure a unified 'FILTER' column exists in the table.

    This function adds a 'FILTER' column to the input Astropy Table if it does not
    already exist, using either the 'DB_FILTER' or 'IFS_MODE' column as the source.
    This is necessary for consistent downstream processing of both IRDIS and IFS
    observations.

    Parameters
    ----------
    tbl : astropy.table.Table
        Table of observations or files. Must contain either 'DB_FILTER' or 'IFS_MODE'.

    Returns
    -------
    None
        The function modifies the input table in-place.

    Examples
    --------
    >>> from astropy.table import Table
    >>> tbl = Table({'DB_FILTER': ['H', 'J']})
    >>> _normalise_filter_column(tbl)
    >>> 'FILTER' in tbl.colnames
    True
    """
    if "FILTER" in tbl.colnames:
        return                         # already present
    if "DB_FILTER" in tbl.colnames:
        tbl["FILTER"] = tbl["DB_FILTER"]
    elif "IFS_MODE" in tbl.colnames:
        tbl["FILTER"] = tbl["IFS_MODE"]


def _normalize_name(name: str) -> str:
    """
    Normalize a target name for robust string matching.

    Converts the input name to lowercase, strips whitespace, and removes spaces and underscores.
    This is used to ensure consistent matching of target names across different catalogs and tables.

    Parameters
    ----------
    name : str
        Target name to normalize.

    Returns
    -------
    str
        Normalized target name.

    Examples
    --------
    >>> _normalize_name(' Beta_Pic ')
    'betapic'
    """
    return name.strip().lower().replace(" ", "").replace("_", "")


class SphereDatabase(object):
    """
    SPHERE Observation Database Interface.

    Provides methods for loading, filtering, and matching SPHERE/IRDIS and SPHERE/IFS
    observation and file tables. Supports robust target lookup (including SIMBAD fallback),
    filtering for usable observations, and construction of observation objects for
    downstream data reduction and analysis.

    Scientific Context
    -----------------
    - All coordinates are assumed to be in ICRS (J2000) and in degrees unless otherwise noted.
    - Exposure times are in seconds. Fluxes are in standard astronomical units.
    - The database is built from ESO archive headers and cross-matched with Gaia.
    - Designed for high-contrast imaging with VLT/SPHERE.

    Parameters
    ----------
    table_of_observations : astropy.table.Table, optional
        Table of observation metadata (see database documentation for required columns).
    table_of_files : astropy.table.Table, optional
        Table of file-level metadata (see database documentation for required columns).
    instrument : str, default 'irdis'
        Instrument to select ('irdis' or 'ifs').

    Examples
    --------
    >>> from astropy.table import Table
    >>> obs = Table.read('table_of_observations_ifs.fits')
    >>> files = Table.read('table_of_files_ifs.csv')
    >>> db = Sphere_database(obs, files, instrument='ifs')
    >>> db.return_usable_only()
    <Table ...>
    """

    def __init__(
        self,
        table_of_observations: Optional[Table] = None,
        table_of_files: Optional[Table] = None,
        instrument: str = "irdis"
    ) -> None:
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

        # ---------- 6) build lists of "summary" keys ---------------------------
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

        # Precompute normalized ID lookup for fast target search
        self._normalized_id_lookup = self._build_normalized_id_lookup()

    def _mask_bad_values(self) -> None:
        """
        Mask bad values in the file table.

        Sets a mask for all values equal to -10000.0 in the file table, which is used
        as a sentinel for missing or invalid data in the database.

        Returns
        -------
        None
            The function modifies the file table in-place.
        """
        self.table_of_files = Table(self.table_of_files, masked=True)
        for key in self.table_of_files.keys():
            self.table_of_files[key].mask = self.table_of_files[key] == -10000.0

    def _mask_not_usable_observations(self, minimum_total_exposure_time: float = 0.0) -> np.ndarray:
        """
        Compute a mask for observations that are not usable for science analysis.

        Flags observations as unusable if they have insufficient total exposure time,
        are not marked as HCI-ready, or were taken in field-stabilized mode.

        Parameters
        ----------
        minimum_total_exposure_time : float, optional
            Minimum total science exposure time in seconds (default: 0.0).

        Returns
        -------
        np.ndarray
            Boolean mask array (True = not usable, False = usable).
        """
        mask_short_exp = self.table_of_observations["TOTAL_EXPTIME_SCI"] < minimum_total_exposure_time

        if self._ready_flag == "HCI_READY":
            mask_bad_flag = ~self.table_of_observations["HCI_READY"]
        else:                                 # legacy
            mask_bad_flag = self.table_of_observations["FAILED_SEQ"] == True

        mask_field_stab = self.table_of_observations["DEROTATOR_MODE"] == "FIELD"

        return np.logical_or.reduce([mask_bad_flag, mask_field_stab, mask_short_exp])

    def _mask_non_instrument_files(self) -> np.ndarray:
        """
        Compute a mask for files not associated with the selected instrument.

        Returns
        -------
        np.ndarray
            Boolean mask array (True = not instrument files, False = instrument files).
        """
        # non_corrupt = self.table_of_files["FILE_SIZE"] > 0.0

        mask = np.logical_and(
            self.table_of_files["DET_ID"] == self.instrument.upper(),
            self.table_of_files["READOUT_MODE"] == "Nondest",
        )
        # files = np.logical_and(non_corrupt, mask)
        return ~mask

    def _mask_non_science_files(self) -> np.ndarray:
        """
        Compute a mask for files that are not science frames.

        Returns
        -------
        np.ndarray
            Boolean mask array (True = not science files, False = science files).
        """
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

    def _get_calibration(self) -> Dict[str, Table]:
        """
        Build a dictionary of calibration files for the selected instrument.

        Returns
        -------
        dict
            Dictionary mapping calibration type (e.g., 'FLAT', 'DISTORTION', 'BACKGROUND')
            to Astropy Table of calibration files.
        """
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

    def return_usable_only(self) -> Table:
        """
        Return a table of only the usable observations.

        Returns
        -------
        astropy.table.Table
            Table of observations flagged as usable for science analysis.
        """
        return self.table_of_observations[~self._not_usable_observations_mask].copy()

    def show_in_browser(self, summary: Optional[str] = None, usable_only: bool = False) -> None:
        """
        Display the observation table in a web browser using JSViewer.

        Parameters
        ----------
        summary : {'NORMAL', 'SHORT', 'MEDIUM', 'OBSLOG'}, optional
            Selects the summary view to display. If None, shows all columns.
        usable_only : bool, optional
            If True, only show usable observations.

        Returns
        -------
        None
        """
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

    def _build_normalized_id_lookup(self) -> Dict[str, List[int]]:
        """
        Build a lookup dictionary mapping normalized target names to row indices in the observation table.
        """
        lookup: Dict[str, List[int]] = {}
        id_columns = ["MAIN_ID", "ID_HD", "ID_HIP", "ID_GAIA_DR3"]
        for col in id_columns:
            if col in self.table_of_observations.colnames:
                col_values = np.char.lower(np.char.strip(self.table_of_observations[col].astype(str)))
                col_values = np.char.replace(col_values, " ", "")
                col_values = np.char.replace(col_values, "_", "")
                for idx, val in enumerate(col_values):
                    if val not in lookup:
                        lookup[val] = []
                    lookup[val].append(idx)
        return lookup

    def _find_observations_by_id_or_simbad(self, target_name: str) -> Table:
        """
        Find observations by target name using local IDs or SIMBAD fallback.

        Attempts to match the target name against local ID columns. If not found,
        queries SIMBAD to resolve the name and match against MAIN_ID.

        Parameters
        ----------
        target_name : str
            Name of the target to search for (case-insensitive, normalized).

        Returns
        -------
        astropy.table.Table
            Table of matching observations.

        Raises
        ------
        ValueError
            If the target cannot be found locally or via SIMBAD.
        """
        name_norm = _normalize_name(target_name)
        # Fast local lookup using precomputed dictionary
        if name_norm in self._normalized_id_lookup:
            indices = self._normalized_id_lookup[name_norm]
            unique_indices = np.unique(indices)
            return self.table_of_observations[unique_indices]
        warnings.warn(f"Target '{target_name}' not found in local ID columns. Trying SIMBAD...")

        # 2. SIMBAD query fallback
        try:
            result = Simbad.query_object(target_name)
            if result is not None and len(result) > 0:
                simbad_main_id = _normalize_name(result['main_id'][0])
                main_id_col = self.table_of_observations['MAIN_ID'].astype(str)
                main_id_norm = np.char.lower(np.char.strip(main_id_col))
                main_id_norm = np.char.replace(main_id_norm, " ", "")
                main_id_norm = np.char.replace(main_id_norm, "_", "")
                mask = main_id_norm == simbad_main_id
                if np.any(mask):
                    return self.table_of_observations[mask]
                else:
                    warnings.warn(f"SIMBAD resolved '{target_name}' to '{result['MAIN_ID'][0]}', but this MAIN_ID is not in the observation table.")
            else:
                warnings.warn(f"SIMBAD could not resolve '{target_name}'.")
        except Exception as e:
            warnings.warn(f"SIMBAD query failed for '{target_name}': {e}")

        # 3. Failure
        raise ValueError(
            f"Target '{target_name}' could not be found in local ID columns or via SIMBAD."
        )

    def observations_from_name_SIMBAD(
        self,
        target_name: str,
        summary: Optional[str] = None,
        usable_only: bool = False,
    ) -> Table:
        """
        Retrieve observations for a target by name, with SIMBAD fallback.

        Matches the target name against local ID columns, falling back to SIMBAD
        if not found. Optionally returns a summary view.

        Parameters
        ----------
        target_name : str
            Name of the target to search for.
        summary : {'NORMAL', 'SHORT', 'MEDIUM', 'OBSLOG'}, optional
            Selects the summary view to return. If None, returns all columns.
        usable_only : bool, optional
            If True, only include usable observations.

        Returns
        -------
        astropy.table.Table
            Table of matching observations (possibly summarized).
        """
        if usable_only:
            table_of_observations = self.table_of_observations[
                ~self._not_usable_observations_mask
            ].copy()
        else:
            table_of_observations = self.table_of_observations.copy()

        # Use robust ID-based matching with SIMBAD fallback
        matching_observations = self._find_observations_by_id_or_simbad(target_name)

        # Only keep those in the current filtered table_of_observations
        # (in case usable_only is True)
        mask = np.isin(matching_observations["MAIN_ID"], table_of_observations["MAIN_ID"])
        matching_observations = matching_observations[mask]

        if summary == "NORMAL":
            return matching_observations[self._keys_for_summary]
        elif summary == "SHORT":
            return matching_observations[self._keys_for_short_summary]
        elif summary == "MEDIUM":
            return matching_observations[self._keys_for_medium_summary]
        elif summary == "OBSLOG":
            return matching_observations[self._keys_for_obslog_summary]
        else:
            return matching_observations

    def get_observation_SIMBAD(
        self,
        target_name: str,
        obs_band: Optional[str] = None,
        date: Optional[Union[str, int]] = None,
        summary: Optional[str] = None,
        usable_only: bool = False,
    ) -> Table:
        """
        Retrieve a specific observation for a target, optionally filtered by band and date.

        Parameters
        ----------
        target_name : str
            Name of the target to search for.
        obs_band : str, optional
            Observation band or filter (e.g., 'OBS_YJ', 'OBS_H').
        date : str or int, optional
            Observation date (format as in the table, e.g., MJD or ISO string).
        summary : {'NORMAL', 'SHORT', 'MEDIUM', 'OBSLOG'}, optional
            Selects the summary view to return. If None, returns all columns.
        usable_only : bool, optional
            If True, only include usable observations.

        Returns
        -------
        astropy.table.Table
            Table of matching observations (possibly empty if no match).
        """
        observations = self.observations_from_name_SIMBAD(
            target_name,
            summary=summary,
            usable_only=usable_only,
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

    def target_list_to_observation_table(
        self,
        target_list: List[str],
        obs_band: Optional[str] = None,
        date: Optional[Union[str, int]] = None,
        summary: Optional[str] = None,
        usable_only: bool = False,
    ) -> Table:
        """
        Convert a list of target names into a combined observation table.

        This method provides a convenient way to retrieve observations for multiple
        targets simultaneously. For each target in the input list, it queries the
        database using SIMBAD fallback if necessary, then combines all results into
        a single Astropy Table. This is particularly useful for batch processing
        of multiple targets in reduction pipelines.

        The method internally calls :meth:`get_observation_SIMBAD` for each target,
        ensuring consistent target name resolution and observation filtering across
        all targets.

        Parameters
        ----------
        target_list : list of str
            List of target names to search for. Each name will be resolved using
            local database IDs first, with SIMBAD fallback if not found locally.
            Names are normalized (case-insensitive, whitespace/underscore stripped).
        obs_band : str, optional
            Observation band or filter to restrict results (e.g., 'OBS_YJ', 'OBS_H').
            If None, observations in all bands are returned. Default is None.
        date : str or int, optional
            Observation date to restrict results (format as in database, e.g., MJD
            or ISO string). If None, observations from all dates are returned.
            Default is None.
        summary : {'NORMAL', 'SHORT', 'MEDIUM', 'OBSLOG'}, optional
            Selects the summary view to return. If None, returns all columns.
            See :meth:`get_observation_SIMBAD` for details. Default is None.
        usable_only : bool, optional
            If True, only include observations flagged as usable for science
            analysis (sufficient exposure time, pupil-stabilized, HCI-ready).
            Default is False.

        Returns
        -------
        astropy.table.Table
            Combined table of observations for all targets. Rows are stacked
            vertically using :func:`astropy.table.vstack`. If a target has no
            matching observations, it contributes no rows to the result.
            The table preserves all metadata and column information from
            individual target queries.

        Raises
        ------
        ValueError
            If any target name cannot be resolved through local database or SIMBAD.
            The error will specify which target failed resolution.

        See Also
        --------
        get_observation_SIMBAD : Retrieve observations for a single target
        observations_from_name_SIMBAD : Get all observations for a target
        retrieve_observation_metadata : Convert observation table to objects

        Notes
        -----
        - The method handles empty results gracefully - targets with no matching
          observations simply contribute no rows to the final table
        - All coordinate references are in ICRS (J2000) frame in decimal degrees
        - Exposure times are in seconds, following standard SPHERE conventions
        - The combined table maintains the same column structure as individual
          target queries, enabling seamless downstream processing

        Examples
        --------
        Get observations for multiple targets:

        >>> target_list = ['HD 95086', 'Beta Pic', '51 Eri']
        >>> obs_table = database.target_list_to_observation_table(target_list)
        >>> print(f"Found {len(obs_table)} observations")

        Filter by band and use summary view:

        >>> obs_table = database.target_list_to_observation_table(
        ...     target_list,
        ...     obs_band='OBS_H',
        ...     summary='SHORT',
        ...     usable_only=True
        ... )

        Process results for pipeline reduction:

        >>> observations = database.retrieve_observation_metadata(obs_table)
        >>> # Pass observations to reduction pipeline
        """
        obs_tables = []
        for target_name in target_list:
            target_obs = self.get_observation_SIMBAD(
                target_name=target_name,
                obs_band=obs_band,
                date=date,
                summary=summary,
                usable_only=usable_only,
            )
            if len(target_obs) > 0:
                obs_tables.append(target_obs)
        
        if not obs_tables:
            # Return empty table with correct structure if no observations found
            return Table()
        
        return vstack(obs_tables)

    def retrieve_observation(
        self,
        target_name: str,
        obs_band: Optional[str] = None,
        date: Optional[Union[str, int]] = None
    ) -> Union[IRDISObservation, IFSObservation]:
        """
        Retrieve an observation object for a given target, band, and date.

        This method constructs an IRDISObservation or IFSObservation object for the
        specified target, using the file and calibration tables. It performs a
        coordinate-based match to select science files within 1.2 arcmin of the target.

        Parameters
        ----------
        target_name : str
            Name of the target to retrieve.
        obs_band : str, optional
            Observation band or filter.
        date : str or int, optional
            Observation date.

        Returns
        -------
        IRDISObservation or IFSObservation
            Observation object for the specified target.

        Raises
        ------
        ValueError
            If no matching target is found.
        """
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

    def retrieve_observation_metadata(
        self,
        table_of_reduction_targets: Table
    ) -> List[Union[IRDISObservation, IFSObservation]]:
        """
        Construct a list of observation objects for a set of reduction targets.

        Parameters
        ----------
        table_of_reduction_targets : astropy.table.Table
            Table of targets to reduce (must include 'MAIN_ID', filter, and date columns).

        Returns
        -------
        list
            List of IRDISObservation or IFSObservation objects.
        """
        observation_object_list = []
        for observation in tqdm(table_of_reduction_targets):
            observation_object = self.retrieve_observation(
            observation["MAIN_ID"],
            observation[self._filter_keyword],
            observation["NIGHT_START"],
            )
            observation_object_list.append(observation_object)
            # except:
            #     print(f"Observation {observation['MAIN_ID']} not found.")
            #     continue

        return observation_object_list

    def plot_observations(self) -> None:
        """
        Placeholder for future plotting functionality.

        Returns
        -------
        None
        """
        pass