import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.simbad import Simbad
from tqdm.auto import tqdm

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
        return  # already present
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


USABLE_MIN_EXPTIME_SCI: float = 5.0


def usable_mask(table: Table) -> np.ndarray:
    """Return a boolean mask of observations usable for high-contrast imaging.

    An observation is usable when it is HCI-ready, pupil-stabilized (ADI), and
    exceeds a minimum total science exposure time.

    Parameters
    ----------
    table : astropy.table.Table
        Observation table containing ``HCI_READY``, ``DEROTATOR_MODE`` and
        ``TOTAL_EXPTIME_SCI`` columns.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where the observation is usable.
    """
    ready = np.asarray(table["HCI_READY"], dtype=bool)
    pupil = np.asarray(table["DEROTATOR_MODE"] != "FIELD")
    long_enough = np.asarray(table["TOTAL_EXPTIME_SCI"] >= USABLE_MIN_EXPTIME_SCI)
    return ready & pupil & long_enough


_HEAD = ["MAIN_ID"]
_TAIL = ["NIGHT_START", "FILTER", "WAFFLE_MODE", "HCI_READY", "DEROTATOR_MODE", "PRIMARY_SCIENCE"]

SUMMARY_COLUMNS: dict = {
    "NORMAL": _HEAD
    + ["ID_GAIA_DR3", "ID_HIP", "RA", "DEC", "OTYPE", "SP_TYPE", "FLUX_H", "STARS_IN_CONE"]
    + _TAIL
    + ["TOTAL_EXPTIME_SCI", "ROTATION", "MEAN_FWHM", "OBS_PROG_ID", "TOTAL_FILE_SIZE_MB"],
    "OBSLOG": _HEAD
    + _TAIL
    + ["DIT", "NDIT", "NCUBES", "TOTAL_EXPTIME_SCI", "TOTAL_EXPTIME_FLUX",
       "ROTATION", "MEAN_FWHM", "MEAN_TAU", "OBS_PROG_ID"],
    "SHORT": _HEAD
    + _TAIL
    + ["GAIA_TEFF", "MOCA_AID", "MOCA_BANYAN_PROB", "MOCA_AGE_MYR",
       "TOTAL_EXPTIME_SCI", "ROTATION", "MEAN_FWHM", "OBS_PROG_ID"],
    "MEDIUM": _HEAD
    + _TAIL
    + ["GAIA_TEFF", "GAIA_LOGG", "GAIA_MH", "MOCA_AID", "MOCA_BANYAN_PROB",
       "MOCA_ASSOCIATION_NAME", "MOCA_ASSOCIATION_TYPE", "MOCA_AGE_MYR", "MOCA_AGE_MYR_UNC",
       "FLUX_H", "DIT", "NDIT", "TOTAL_EXPTIME_SCI", "ROTATION", "MEAN_FWHM",
       "STDDEV_FWHM", "OBS_PROG_ID"],
}


def view(table: Table, summary: str | None = None) -> Table:
    """Return a copy of ``table`` reduced to a named summary column set.

    Parameters
    ----------
    table : astropy.table.Table
        Any observation table (filtered or not).
    summary : {'NORMAL', 'SHORT', 'MEDIUM', 'OBSLOG'} or None
        Named column set. Columns not present in ``table`` are silently
        dropped. ``None`` returns all columns.

    Returns
    -------
    astropy.table.Table
        A copy with the selected columns.

    Raises
    ------
    KeyError
        If ``summary`` is not a known name.
    """
    if summary is None:
        return table.copy()
    columns = [c for c in SUMMARY_COLUMNS[summary] if c in table.colnames]
    return table[columns].copy()


def _isin(column, values) -> np.ndarray:
    """Membership test that is safe for bytes/unicode string columns.

    ``numpy.isin`` silently fails to match a ``|S`` (bytes) column against a
    list of ``str`` values, so string columns are stringified on both sides.
    """
    data = np.asarray(column)
    if data.dtype.kind in ("S", "U"):
        return np.isin(data.astype(str), np.asarray(list(values)).astype(str))
    return np.isin(data, list(values))


def _criterion_mask(column, cond) -> np.ndarray:
    """Boolean mask for one filter criterion on a single column.

    ``cond`` is a scalar (equality), a list/set/ndarray (membership), or a
    ``('not in', sequence)`` tuple (exclusion). Missing-value handling is done
    by the caller via the column mask, so masked positions may take an
    arbitrary value here.
    """
    if isinstance(cond, tuple):
        if len(cond) == 2 and cond[0] == "not in":
            values = cond[1]
            if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
                raise TypeError("('not in', ...) requires a sequence of values.")
            return ~_isin(column, values)
        raise ValueError("Tuple criteria must be of the form ('not in', sequence).")
    if isinstance(cond, (list, set, np.ndarray)):
        return _isin(column, cond)
    return np.asarray(column == cond)


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
        self, table_of_observations: Optional[Table] = None, table_of_files: Optional[Table] = None, instrument: str = "irdis"
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
        self.table_of_files = convert_table_to_little_endian(table_of_files)

        # ---------- 3) keep rows only for the chosen instrument ----------------
        if "INSTRUMENT" in self.table_of_observations.colnames:
            mask_instr = self.table_of_observations["INSTRUMENT"] == self.instrument
            self.table_of_observations = self.table_of_observations[mask_instr]

        # ---------- 4) SHUTTER column type fix  --------------------------------
        if isinstance(self.table_of_files["SHUTTER"][0], str):
            self.table_of_files["SHUTTER"] = [s.lower() in ("true", "t", "1") for s in self.table_of_files["SHUTTER"]]

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
        dark_and_background_selection = np.logical_or(files["DPR_TYPE"] == "DARK", files["DPR_TYPE"] == "DARK,BACKGROUND")
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
        """Return a copy of the table with only usable observations.

        Returns
        -------
        astropy.table.Table
            Observations flagged usable by :func:`usable_mask`.
        """
        return self.table_of_observations[usable_mask(self.table_of_observations)].copy()

    @property
    def columns(self) -> List[str]:
        """Column names available for filtering."""
        return self.table_of_observations.colnames

    def filter(self, *masks, usable_only: bool = False, target_list=None, **criteria) -> Table:
        """Return observations matching all given criteria and masks.

        Parameters
        ----------
        *masks : numpy.ndarray or callable
            Boolean arrays, or callables ``f(table) -> bool array``, combined
            with logical AND. Use these for comparisons and cross-column
            arithmetic (e.g. absolute magnitude, colours).
        usable_only : bool, optional
            If True, restrict to high-contrast-usable observations
            (see :func:`usable_mask`).
        target_list : list of str, optional
            Restrict to these targets first (resolved by name; see
            :meth:`observations_from_name_SIMBAD`). ``None`` uses all rows.
        **criteria : scalar, list, or ('not in', sequence)
            Per-column tests: a scalar means equality, a list means membership,
            and ``('not in', seq)`` means exclusion. A row whose value for a
            criterion's column is missing is always excluded.

        Returns
        -------
        astropy.table.Table
            A copy of the matching rows (all columns). Empty if nothing matches.

        Raises
        ------
        KeyError
            If a criterion names a column not in the table.
        ValueError
            If a boolean-array mask has the wrong length.
        TypeError
            If a ``('not in', <non-sequence>)`` criterion is malformed (the
            second element is not a sequence of values).
        """
        import difflib

        table = self.table_of_observations
        if target_list is not None:
            resolved = self.observations_from_name_SIMBAD(target_list)
            if resolved is None:
                return table[np.zeros(len(table), dtype=bool)].copy()
            resolved_ids = np.unique(np.asarray(resolved["MAIN_ID"]))
            table = table[np.isin(np.asarray(table["MAIN_ID"]), resolved_ids)]

        mask = np.ones(len(table), dtype=bool)

        for column_name, cond in criteria.items():
            if column_name not in table.colnames:
                suggestion = difflib.get_close_matches(column_name, table.colnames, n=1)
                hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
                raise KeyError(f"{column_name!r} is not a column.{hint} Available columns: {table.colnames}")
            column = table[column_name]
            present = ~np.ma.getmaskarray(column)
            mask &= present & _criterion_mask(column, cond)

        for spec in masks:
            m = spec(table) if callable(spec) else np.ma.asarray(spec)
            if len(m) != len(table):
                raise ValueError(f"Mask length {len(m)} does not match table length {len(table)}.")
            mask &= np.ma.filled(np.ma.asarray(m), False)

        if usable_only:
            mask &= usable_mask(table)

        return table[np.ma.filled(mask, False)].copy()

    def show_in_browser(self, summary: Optional[str] = None, usable_only: bool = False) -> None:
        """Display the observation table in a web browser using JSViewer.

        Parameters
        ----------
        summary : {'NORMAL', 'SHORT', 'MEDIUM', 'OBSLOG'}, optional
            Named column set; ``None`` shows all columns.
        usable_only : bool, optional
            If True, restrict to usable observations (see :func:`usable_mask`).
        """
        table = self.table_of_observations
        if usable_only:
            table = table[usable_mask(table)]
        view(table, summary).show_in_browser(jsviewer=True)

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

        Attempts to match the target name against local ID columns, including
        binary star naming variations (with/without 'A' suffix). If not found,
        queries SIMBAD to resolve the name and match against MAIN_ID.

        Parameters
        ----------
        target_name : str
            Name of the target to search for (case-insensitive, normalized).

        Returns
        -------
        astropy.table.Table
            Table of matching observations. Returns None if there are
            no observations of the ``target_name``.

        Raises
        ------
        ValueError
            If the target cannot be resolved by SIMBAD.
        """

        def _try_local_lookup(name: str) -> Optional[Table]:
            """Try to find target in local database by normalized name."""
            name_norm = _normalize_name(name)
            if name_norm in self._normalized_id_lookup:
                indices = self._normalized_id_lookup[name_norm]
                unique_indices = np.unique(indices)
                return self.table_of_observations[unique_indices]
            return None

        # 1. Try direct lookup
        result = _try_local_lookup(target_name)
        if result is not None:
            return result

        # 2. Try binary star naming variations
        name_norm = _normalize_name(target_name)

        # If name ends with 'a', try without it (e.g., "HD 95086 A" -> "HD 95086")
        if name_norm.endswith("a"):
            base_name = target_name.rstrip().rstrip("aA")  # Remove trailing A/a
            result = _try_local_lookup(base_name)
            if result is not None:
                return result

        # If name doesn't end with 'a', try adding it (e.g., "HD 95086" -> "HD 95086 A")
        else:
            result = _try_local_lookup(target_name + " A")
            if result is not None:
                return result

        # 3. SIMBAD query fallback
        warnings.warn(f"Target '{target_name}' not found in local ID columns. Trying SIMBAD...")

        result = Simbad.query_object(target_name)
        if result is not None and len(result) > 0:
            simbad_main_id = _normalize_name(result["main_id"][0])
            main_id_col = self.table_of_observations["MAIN_ID"].astype(str)
            main_id_norm = np.char.lower(np.char.strip(main_id_col))
            main_id_norm = np.char.replace(main_id_norm, " ", "")
            main_id_norm = np.char.replace(main_id_norm, "_", "")
            mask = main_id_norm == simbad_main_id
            if np.any(mask):
                return self.table_of_observations[mask]
            else:
                warnings.warn(
                    f"SIMBAD resolved '{target_name}' to '{result['main_id'][0]}', but this MAIN_ID is not in the observation table."
                )
        else:
            warnings.warn(f"SIMBAD could not resolve '{target_name}'.")

    def observations_from_name_SIMBAD(
        self,
        target_name: Union[str, Sequence[str]],
        summary: Optional[str] = None,
        usable_only: bool = False,
    ) -> Table:
        """
        Retrieve observations for one or more targets by name, with SIMBAD fallback.

        Each name is matched against local ID columns, falling back to SIMBAD if
        not found. Optionally returns a summary view.

        Parameters
        ----------
        target_name : str or sequence of str
            A single target name, or a list/tuple of target names. Observations
            for all resolved targets are combined; a target that resolves via
            multiple names is only included once.
        summary : {'NORMAL', 'SHORT', 'MEDIUM', 'OBSLOG'}, optional
            Selects the summary view to return. If None, returns all columns.
        usable_only : bool, optional
            If True, only include usable observations.

        Returns
        -------
        astropy.table.Table
            Table of matching observations (possibly summarized). Returns
            None if none of the ``target_name``\\ (s) resolve to any observations.
        """
        if usable_only:
            table_of_observations = self.table_of_observations[usable_mask(self.table_of_observations)].copy()
        else:
            table_of_observations = self.table_of_observations.copy()

        names = [target_name] if isinstance(target_name, str) else list(target_name)

        # Resolve each name (local ID lookup with SIMBAD fallback) and collect the
        # matched MAIN_IDs. Selecting rows once by this set naturally deduplicates
        # targets reached through more than one name.
        resolved_main_ids: set = set()
        found_any = False
        for name in names:
            matches = self._find_observations_by_id_or_simbad(name)
            if matches is not None:
                found_any = True
                resolved_main_ids.update(np.asarray(matches["MAIN_ID"]).tolist())

        if not found_any:
            return None

        mask = np.isin(np.asarray(table_of_observations["MAIN_ID"]), list(resolved_main_ids))
        matching_observations = table_of_observations[mask]

        return view(matching_observations, summary)

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
            Table of matching observations (possibly empty if no match). Returns
            None if there are no observations of the ``target_name``.
        """
        observations = self.observations_from_name_SIMBAD(
            target_name,
            summary=summary,
            usable_only=usable_only,
        )

        # print(observations)
        # print("-----------------")

        if observations is not None:
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
            if target_obs is not None and len(target_obs) > 0:
                obs_tables.append(target_obs)

        if not obs_tables:
            # Return empty table with correct structure if no observations found
            return Table()

        return vstack(obs_tables)

    def retrieve_observation(
        self, target_name: str, obs_band: Optional[str] = None, date: Optional[Union[str, int]] = None
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
        file_coordinates = SkyCoord(ra=science_files["RA"] * u.degree, dec=science_files["DEC"] * u.degree)
        target_coordinates = SkyCoord(ra=observation["RA_HEADER"] * u.degree, dec=observation["DEC_HEADER"] * u.degree)
        if len(target_coordinates) == 0:
            raise ValueError("No Targets found with the given information.")
        # 1.15 arcmin to include sky frames displaced from the sequence (e.g. 1.12 for HD135344)

        target_selection = target_coordinates.separation(file_coordinates) < 1.2 * u.arcmin
        file_table = science_files[target_selection]

        # Check which instrument was used for the observation
        instrument = observation["INSTRUMENT"][0]
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
            if observation["POLARIMETRY"][0]:
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
        self, table_of_reduction_targets: Table
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
