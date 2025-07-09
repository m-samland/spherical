#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = (
    "M. Samland @ MPIA (Heidelberg, Germany), J. Kemmer @ MPIA (Heidelberg, Germany)"
)

import glob
import itertools
import os
import re
import time

import numpy as np
import pandas as pd
from astropy.io.ascii.core import InconsistentTableError
from astropy.table import Column, Table, TableMergeError, hstack, join, unique, vstack
from astroquery.gaia import Gaia

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


def convert_table_to_little_endian(table):
    if table is None:
        return None

    for colname in table.colnames:
        col = table[colname]
        if np.issubdtype(col.dtype, np.number):
            table[colname] = col.astype(col.dtype.newbyteorder('='))
    return table


def normalize_shutter_column(table):
    """
    Normalize the 'SHUTTER' column to strict boolean values.

    Parameters
    ----------
    table : pandas.DataFrame or astropy.table.Table
        Table containing a 'SHUTTER' column to normalize.

    Returns
    -------
    table : same type as input
        Modified table with 'SHUTTER' column coerced to bool or None.
    """
    if 'SHUTTER' not in table.columns:
        return table  # nothing to do

    def convert_to_bool(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            val = val.strip().upper()
            if val in ("T", "TRUE", "1"):
                return True
            elif val in ("F", "FALSE", "0"):
                return False
        return None  # fallback for unexpected values

    if isinstance(table, pd.DataFrame):
        table['SHUTTER'] = table['SHUTTER'].apply(convert_to_bool)
    else:
        # Assume astropy Table
        table['SHUTTER'] = [convert_to_bool(v) for v in table['SHUTTER']]

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


def add_night_start_date(df: pd.DataFrame, key="DATE_OBS") -> pd.DataFrame:
    if "NIGHT_START" not in df.columns:
        times = pd.to_datetime(df[key], errors='coerce', utc=True)
        night_start = (times - pd.to_timedelta((times.dt.hour < 12).astype(int), unit='d')).dt.date.astype(str)
        # Replace invalid parsed dates with a warning string
        night_start = night_start.where(times.notna(), "INVALID_DATE")
        df["NIGHT_START"] = night_start
    return df


def make_selection_mask(table, condition_dictionary, logical_operation="and"):
    filters = []
    for key in condition_dictionary.keys():
        filters.append(table[key] == condition_dictionary[key])
    if logical_operation == "and":
        selection_mask = np.logical_and.reduce(filters)
    elif logical_operation == "or":
        selection_mask = np.logical_or.reduce(filters)
    return selection_mask


def filter_table(table, condition_dictionary):
    selection_mask = make_selection_mask(table, condition_dictionary)
    return table[selection_mask].copy()


def find_nearest(array, value):
    """Return index of array value closest to specified value."""
    idx = (np.abs(array - value)).argmin()
    return idx


def find_duplicate_values(table_of_targets, keyword="MJD_MEAN"):
    """Create table of targets with same proper motion in target list"""

    counter = 0
    for key in table_of_targets.group_by(keyword).groups.keys:
        main_rows = table_of_targets[table_of_targets[keyword] == key[0]]
        if len(main_rows) > 1:
            main_row = main_rows[0]
            dtypes = []
            for i in range(len(main_row.dtype)):
                dtypes.append(main_row.dtype[i])

            if counter == 0:  # Create table from one row for first iteration
                duplicate = Table(
                    rows=main_row,
                    names=main_row.colnames,
                    dtype=dtypes,
                    meta=main_row.meta,
                )
                for i in range(len(main_rows) - 1):
                    duplicate.add_row(main_rows[i + 1])
                counter += 1
            else:
                # simbad_table.add_row(query_result['MAIN_ID', 'RA', 'DEC', 'COO_ERR_MAJA', 'COO_ERR_MINA'][0])
                for i in range(len(main_rows)):
                    duplicate.add_row(main_rows[i])
    try:
        duplicate = duplicate[duplicate[keyword] != 0]
    except UnboundLocalError:
        duplicate = []
    return duplicate


def find_duplicate_files(table_of_files):
    counter = 0
    for key in table_of_files.group_by("MJD_OBS").groups.keys:
        main_rows = table_of_files[table_of_files["MJD_OBS"] == key[0]]
        if len(main_rows) >= 3:
            print(main_rows)
            print("WARNING!")

        if len(main_rows) > 1:
            main_row = main_rows[0]
            dtypes = []
            for i in range(len(main_row.dtype)):
                dtypes.append(main_row.dtype[i])

            if counter == 0:  # Create table from one row for first iteration
                duplicate = Table(
                    rows=main_row,
                    names=main_row.colnames,
                    dtype=dtypes,
                    meta=main_row.meta,
                )
                for i in range(len(main_rows) - 1):
                    duplicate.add_row(main_rows[i + 1])
                counter += 1
            else:
                # simbad_table.add_row(query_result['MAIN_ID', 'RA', 'DEC', 'COO_ERR_MAJA', 'COO_ERR_MINA'][0])
                for i in range(len(main_rows)):
                    duplicate.add_row(main_rows[i])

    return duplicate


def find_names_of_duplicates(
    duplicate_table,
    simbad_keys=[
        "FLUX_V",
        "FLUX_R",
        "FLUX_I",
        "FLUX_J",
        "FLUX_H",
        "FLUX_K",
        "PM_RA",
        "PM_DEC",
        "PM_RA_ERR",
        "PM_DEC_ERR",
    ],
):
    duplicates_by_mjd = duplicate_table.group_by("MJD_MEAN")
    indices = duplicates_by_mjd.groups.indices
    number_of_groups = len(duplicates_by_mjd.groups.keys)

    names_of_duplicates = []
    for group_idx in range(number_of_groups):
        start_index = indices[group_idx]
        print("group_idx: {} start_idx: {}".format(group_idx, start_index))
        available_values = []
        group = duplicates_by_mjd[indices[group_idx] : indices[group_idx + 1]]
        print(group)
        group_size = len(group)
        print(group_size)
        for row in group:
            fluxes = 0
            for key in simbad_keys:
                if row[key] < 500:
                    fluxes += 1
            available_values.append(fluxes)
        print(available_values)
        index_of_entry_to_retain = np.argmax(available_values)
        duplicate_indices = np.where(np.arange(group_size) != index_of_entry_to_retain)[
            0
        ]
        print("indices of duplicate: {}".format(duplicate_indices))
        print("argmax: {}".format(np.argmax(available_values)))
        names_of_duplicates.append(group["MAIN_ID"][duplicate_indices].data)

    list_of_duplicate_names = []
    for arr in names_of_duplicates:
        for name in arr:
            list_of_duplicate_names.append(name)
    list_of_duplicate_names = list(dict.fromkeys(list_of_duplicate_names))

    return list_of_duplicate_names


def remove_spaces(name):
    """Replace the spaces in the SIMBAD Main_ID by underscores."""
    splitname = " ".join(name.split())
    return splitname.replace(" ", "_")

def convert_bytestring_columns(df):
    """
    Convert all bytestring columns in the DataFrame to normal strings.
    
    :param df: DataFrame with bytestring columns
    :return: DataFrame with bytestring columns converted to normal strings
    """
    # Iterate over DataFrame columns
    for col in df.columns:
        # Check if the column dtype is 'object' (which is common for bytestrings)
        if df[col].dtype == 'object':
            # Convert bytestrings to normal strings
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    return df

def replace_substrings(df, column_name, old_substring, new_substring):
    """
    Replace all occurrences of a substring in a specified column with another string.
    
    :param df: DataFrame containing the column to be modified.
    :param column_name: Name of the column in which substrings will be replaced.
    :param old_substring: Substring to be replaced.
    :param new_substring: Replacement string.
    :return: DataFrame with substrings replaced.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Use str.replace to replace old_substring with new_substring in the specified column
    df[column_name] = df[column_name].astype(str).str.replace(old_substring, new_substring, regex=False)
    
    return df

def query_gspphot(dr3_ids):
    """
    Query Gaia GSPPhot catalog for a list of DR3 IDs.
    
    :param dr3_ids: List of Gaia DR3 IDs.
    :return: DataFrame with GSPPhot parameters.
    """
    # Initialize Gaia query
    Gaia.MAIN_GAIA_TABLE = 'gaiaedr3.gaia_source'  # Use Gaia DR3
    
    # Initialize empty list to store results
    results = []
    
    # Query each ID and collect results
    for dr3_id in tqdm(dr3_ids):
        dr3_id = dr3_id.replace('Gaia DR1 ', '')
        dr3_id = dr3_id.replace('Gaia DR2 ', '')
        dr3_id = dr3_id.replace('Gaia DR3 ', '')
        
        if dr3_id == '':  # Skip empty IDs
            continue
        
        query = f"""
        SELECT
            source_id,
            teff_gspphot,
            logg_gspphot,
            mh_gspphot,
            distance_gspphot,
            azero_gspphot,
            ruwe
        FROM
            gaiadr3.gaia_source
        WHERE
            source_id = {dr3_id}
        """
        try:
            job = Gaia.launch_job(query)
        except:
            print(f"Error querying Gaia for source ID {dr3_id}")
            continue
        result = job.get_results()
        results.append(result.to_pandas())

    
    # Combine all results into a single DataFrame
    df = pd.concat(results, ignore_index=True)
    return df

def collect_reduction_infos(database, reduction_folder, show=False):
    """Collect the reduction_info and qualitiy flags for a sphere_database."""
    reduction_infos = Table()
    print("Collecting reduction_infos")
    for observation in tqdm(database.return_usable_only()):
        observation_obj = database.retrieve_IRDIS_observation(
            remove_spaces(observation["MAIN_ID"]),
            observation["DB_FILTER"],
            observation["DATE_SHORT"],
        )
        reduction_info = observation_obj.get_reduction_info(reduction_folder)
        reduction_info = hstack([observation_obj.data_quality_flags, reduction_info])
        main_id = Column(
            np.repeat(observation_obj._target_name, len(reduction_info)), name="MAIN_ID"
        )
        reduction_info.add_column(main_id, index=0)
        date = Column(
            np.repeat(
                observation_obj.observation["DATE_SHORT"][0], len(reduction_info)
            ),
            name="DATE_SHORT",
        )
        reduction_info.add_column(date, index=1)
        date = Column(
            np.repeat(observation_obj.observation["DB_FILTER"][0], len(reduction_info)),
            name="DB_FILTER",
        )
        reduction_info.add_column(date, index=2)
        reduction_infos = vstack([reduction_infos, reduction_info], join_type="outer")
    if show:
        reduction_infos.show_in_browser(jsviewer=True)
    return reduction_infos


def collect_detected_sources(
    observation_list, reduction_folder, package, match_sources=False, show=False
):
    """Assemble a table of all detected sources from a list of
    observed targets for a specifiy post-processing package."""
    table_of_results = Table()
    print("Collecting detected sources from {}".format(package))
    for observation in tqdm(observation_list):
        main_id = remove_spaces(observation["MAIN_ID"])
        db_filter = observation["DB_FILTER"]
        date_short = observation["DATE_SHORT"]
        results_object = ADI_observation_results(
            os.path.join(
                reduction_folder,
                main_id,
                db_filter,
                date_short,
                "converted_mastercubes/",
            ),
            db_filter,
            package,
            match_sources,
        )
        results = results_object.results
        if len(results) > 0:
            rep_main_id = Column(np.repeat(main_id, len(results)), name="MAIN_ID")
            results.add_column(rep_main_id, index=0)
            date = Column(np.repeat(date_short, len(results)), name="DATE_SHORT")
            results.add_column(date, index=1)
            unique_id = []
            for idx, _ in enumerate(results):
                unique_id.append("{}_{}".format(main_id, idx))
            unique_id = Column(unique_id, name="UNIQUE_ID")
            results.add_column(unique_id, index=0)
        try:
            table_of_results = vstack([table_of_results, results], join_type="outer")
        except TableMergeError:
            results["flag_lobe_H2"].set_fill_value(-999)
            results["flag_lobe_H2"] = results["flag_lobe_H2"].astype(float)
            table_of_results = vstack([table_of_results, results], join_type="outer")

    if show:
        table_of_results.show_in_browser(jsviewer=True)
    return table_of_results


class ADI_observation_results(object):
    def __init__(self, reduction_directory, filter_string, package, match_sources):
        self._valid_packages = ["andromeda", "pyklip"]
        self.package = package
        if package not in self._valid_packages:
            raise KeyError(
                "Unknown package: '{0}'. "
                "Possible options are: "
                "{1}".format(package, self._valid_packages)
            )
        self.results_directory = os.path.join(
            reduction_directory, "results_{}*".format(self.package)
        )
        self._filter_string = filter_string
        self._filter_names = self._get_filter(filter_string)
        # check if DB or BB filter.
        if len(self._filter_names) > 1:
            self._results_channel1 = self._read_in_results_file("channel1")
            self._results_channel2 = self._read_in_results_file("channel2")
            self._forced_phot_channel1 = self._read_in_forced_phot("channel1")
            self._forced_phot_channel2 = self._read_in_forced_phot("channel2")
            self.unordered_results = self._merge_results()
            if match_sources:
                self.results = self._match_results()
            else:
                self.results = self.unordered_results
        else:
            if self.package == "andromeda":
                bb_keys = [
                    ("contrast-[1]", "contrast_{0}-[1]"),
                    ("err_contrast-[1]", "err_contrast_{0}-[1]"),
                    ("magnitude-[mag]", "magnitude_{0}-[mag]"),
                    ("err_mag", "err_mag_{0}"),
                    ("flag_pos", "flag_pos_{0}"),
                    ("flag_flux", "flag_flux_{0}"),
                    ("flag_lobe", "flag_lobe_{0}"),
                ]
                self.results = self._read_in_results_file("combined")
            elif self.package == "pyklip" or self.package == "vip":
                bb_keys = [
                    ("contrast-[1]", "contrast_{0}-[1]"),
                    ("err_contrast-[1]", "err_contrast_{0}-[1]"),
                    ("magnitude-[mag]", "magnitude_{0}-[mag]"),
                    ("err_mag", "err_mag_{0}"),
                ]
                self.results = self._read_in_results_file("channel1")
            if self.results is None:
                self.results = Table()
            else:
                self.results = self._rename_keys(
                    self.results, bb_keys, self._filter_names
                )

    def _get_filter(self, filter_string):
        if len(filter_string) > 5:
            name = [
                filter_string[-3:][:2],
                filter_string[-3:][0] + filter_string[-3:][2],
            ]
        else:
            name = [filter_string[-1]]
        return name

    def _read_in_results_file(self, filter=None, verbose=False):
        filepath = glob.glob(
            os.path.join(
                self.results_directory, "*_{}_Results_Files" ".dat".format(filter)
            )
        )
        if len(filepath) > 0:
            results_file = Table.read(filepath[0], format="ascii")
            results_file = results_file[:99]
            results_file.remove_column("Index-[1]")
        else:
            if verbose:
                print(
                    "No results file found for '{}' .\n"
                    "Returning 'None'.\n".format(filter)
                )
            results_file = None
        return results_file

    def _read_in_forced_phot(self, filter=None, verbose=False):
        filepath = glob.glob(
            os.path.join(
                self.results_directory,
                "*_{}_ForcedPhotometry" "_Files" ".dat".format(filter),
            )
        )
        if len(filepath) > 0:
            try:
                results_file = Table.read(filepath[0], format="ascii")
            except InconsistentTableError:
                if verbose:
                    print(
                        "Inconsistent number of Columns and values in "
                        "forced photometry file of '{}' .\n"
                        "Returning 'None'.\n".format(filter)
                    )
                results_file = None
        else:
            if verbose:
                print(
                    "No forced photometry found for '{}' .\n"
                    "Returning 'None'.\n".format(filter)
                )
            results_file = None
        return results_file

    def _rename_keys(self, table, keys_to_rename, insertion=None):
        for old, new in keys_to_rename:
            if insertion:
                table.rename_column(old, new.format(*insertion))
            else:
                table.rename_column(old, new)
        return table

    def _insert_forced_phot(self, table, forced_expressions):
        # add values from forced photometry to table
        try:
            table_coords = table["offset_x-[px]", "offset_y-[px]"]
            forced_coords = self._forced_phot_channel1["offset_x-[px]", "offset_y-[px]"]
            for index_channel2, source_channel2 in enumerate(forced_coords):
                for index_channel1, source_channel1 in enumerate(table_coords):
                    if np.allclose(
                        np.array([source_channel1[0], source_channel1[1]]),
                        np.array([source_channel2[0], source_channel2[1]]),
                        atol=2,
                    ):
                        for expression in forced_expressions:
                            table[index_channel1][
                                expression
                            ] = self._forced_phot_channel1[index_channel2][
                                forced_expressions[expression]
                            ]
        except:
            if self.package == "andromeda" and len(self._forced_phot_channel1) == 1:
                old_expressions = {
                    "contrast-[1]_2": "Forced_Photometry-[Contrast]",
                    "err_contrast-[1]_2": "3-sig_Upper_Limit-[Contrat]",
                    "magnitude-[mag]_2": "Forced_Photometry-[Mag]",
                    "err_mag_2": "3-sig_Upper_Limit-[Mag]",
                }
                index = self._forced_phot_channel1["Index"][0] - 1
                for expression in old_expressions:
                    table[index][expression] = self._forced_phot_channel1[index][
                        old_expressions[expression]
                    ]
        return table

    def _merge_results(self):
        if self.package == "andromeda":
            names_with_suffix = [
                ("contrast-[1]", "contrast-[1]_{}"),
                ("err_contrast-[1]", "err_contrast-[1]_{}"),
                ("magnitude-[mag]", "magnitude-[mag]_{}"),
                ("err_mag", "err_mag_{}"),
                ("flag_pos", "flag_pos_{}"),
                ("flag_flux", "flag_flux_{}"),
                ("flag_lobe", "flag_lobe_{}"),
            ]
            missing_keys = [
                "contrast-[1]_{}",
                "err_contrast-[1]_{}",
                "magnitude-[mag]_{}",
                "err_mag_{}",
                "flag_pos_{}",
                "flag_flux_{}",
                "flag_lobe_{}",
            ]
            forced_expressions = {
                "contrast-[1]_2": "Forced_Photometry-[Contrast]",
                "err_contrast-[1]_2": "3-sig_Upper_Limit-[Contrast]",
                "magnitude-[mag]_2": "Forced_Photometry-[Mag]",
                "err_mag_2": "3-sig_Upper_Limit-[Mag]",
            }
            name_replacements = [
                ("contrast-[1]_1", "contrast_{0}-[1]"),
                ("err_contrast-[1]_1", "err_contrast_{0}-[1]"),
                ("magnitude-[mag]_1", "magnitude_{0}-[mag]"),
                ("err_mag_1", "err_mag_{0}"),
                ("flag_pos_1", "flag_pos_{0}"),
                ("flag_flux_1", "flag_flux_{0}"),
                ("flag_lobe_1", "flag_lobe_{0}"),
                ("contrast-[1]_2", "contrast_{1}-[1]"),
                ("err_contrast-[1]_2", "err_contrast_{1}-[1]"),
                ("magnitude-[mag]_2", "magnitude_{1}-[mag]"),
                ("err_mag_2", "err_mag_{1}"),
                ("flag_pos_2", "flag_pos_{1}"),
                ("flag_flux_2", "flag_flux_{1}"),
                ("flag_lobe_2", "flag_lobe_{1}"),
            ]
        if self.package == "pyklip" or self.package == "vip":
            names_with_suffix = [
                ("contrast-[1]", "contrast-[1]_{}"),
                ("err_contrast-[1]", "err_contrast-[1]_{}"),
                ("magnitude-[mag]", "magnitude-[mag]_{}"),
                ("err_mag", "err_mag_{}"),
            ]
            missing_keys = [
                "contrast-[1]_{}",
                "err_contrast-[1]_{}",
                "magnitude-[mag]_{}",
                "err_mag_{}",
            ]
            forced_expressions = {
                "contrast-[1]_2": "contrast-[1]",
                "err_contrast-[1]_2": "err_contrast-[1]",
                "magnitude-[mag]_2": "magnitude-[mag]",
                "err_mag_2": "err_mag",
            }
            name_replacements = [
                ("contrast-[1]_1", "contrast_{0}-[1]"),
                ("err_contrast-[1]_1", "err_contrast_{0}-[1]"),
                ("magnitude-[mag]_1", "magnitude_{0}-[mag]"),
                ("err_mag_1", "err_mag_{0}"),
                ("contrast-[1]_2", "contrast_{1}-[1]"),
                ("err_contrast-[1]_2", "err_contrast_{1}-[1]"),
                ("magnitude-[mag]_2", "magnitude_{1}-[mag]"),
                ("err_mag_2", "err_mag_{1}"),
            ]
        if not self._results_channel1 and not self._results_channel2:
            return Table()
        if self._results_channel1 and self._results_channel2:
            merged_results = join(
                self._results_channel1,
                self._results_channel2,
                join_type="outer",
                keys=[
                    "coord_x-[px]",
                    "coord_y-[px]",
                    "offset_x-[px]",
                    "offset_y-[px]",
                    "sep-[mas]",
                    "err_sep-[mas]",
                    "PA-[deg]",
                    "err_PA-[deg]",
                    "SNR-[1]",
                    "nb_px>thresh-[1]",
                ],
            )

        elif self._results_channel1 and not self._results_channel2:
            merged_results = self._results_channel1
            merged_results = self._rename_keys(merged_results, names_with_suffix, [1])
            for key in missing_keys:
                masked_array = Table.MaskedColumn(
                    np.zeros(len(merged_results)),
                    name=key.format(2),
                    mask=np.ones(len(merged_results)) * True,
                )
                merged_results.add_column(masked_array)

        elif self._results_channel2 and not self._results_channel1:
            merged_results = self._results_channel2
            merged_results = self._rename_keys(merged_results, names_with_suffix, [2])
            for key in missing_keys:
                masked_array = Table.MaskedColumn(
                    np.zeros(len(merged_results)),
                    name=key.format(1),
                    mask=np.ones(len(merged_results)) * True,
                )
                merged_results.add_column(masked_array)

        if self._forced_phot_channel1:
            merged_results = self._insert_forced_phot(
                merged_results, forced_expressions
            )
            forced_phot = Column(
                data=np.invert(merged_results["contrast-[1]_2"].mask),
                name="forced_photometry_{}".format(self._filter_string),
            )
            if self.package == "andromeda":
                merged_results.add_column(col=forced_phot, index=17)
            else:
                merged_results.add_column(col=forced_phot, index=14)

        else:
            forced_phot = Column(
                data=np.zeros(len(merged_results)).astype("bool"),
                name="forced_photometry_{}".format(self._filter_string),
            )
            if self.package == "andromeda":
                merged_results.add_column(col=forced_phot, index=17)
            else:
                merged_results.add_column(col=forced_phot, index=14)

        return self._rename_keys(merged_results, name_replacements, self._filter_names)

    def _match_results(self):
        if len(self.unordered_results) == 0:
            return Table()
        complete = self.unordered_results.copy()
        mask_channel1 = np.invert(
            complete["contrast_{}-[1]".format(self._filter_names[0])].mask
        )
        mask_channel2 = np.invert(
            complete["contrast_{}-[1]".format(self._filter_names[1])].mask
        )
        coords_channel1 = complete[mask_channel1]["coord_x-[px]", "coord_y-[px]"]
        coords_channel2 = complete[mask_channel2]["coord_x-[px]", "coord_y-[px]"]
        data_channel1 = complete[mask_channel1]
        if len(data_channel1) == 0:
            return Table()
        data_channel2 = complete[mask_channel2]
        matched_table = Table()
        for coords1, coords2 in itertools.product(coords_channel1, coords_channel2):
            if np.allclose(
                np.array([coords1[0], coords1[1]]),
                np.array([coords2[0], coords2[1]]),
                atol=3,
            ):
                mask1 = np.logical_and(
                    data_channel1["coord_x-[px]"] == coords1[0],
                    data_channel1["coord_y-[px]"] == coords1[1],
                )
                entries1 = data_channel1[mask1]
                mask2 = np.logical_and(
                    data_channel2["coord_x-[px]"] == coords2[0],
                    data_channel2["coord_y-[px]"] == coords2[1],
                )
                entries2 = data_channel2[mask2]
                for key in entries1.colnames:
                    if entries1[key].mask:
                        entries1[key] = entries2[key]
                matched_table = vstack([matched_table, entries1])
        unique_table = vstack([matched_table, data_channel1], join_type="outer")
        unique_table = unique(
            unique_table, keys=["coord_x-[px]", "coord_y-[px]"], keep="first"
        )
        return unique_table


def retry_query(function, number_of_retries=3, verbose=False, **kwargs):
    for attempt in range(int(number_of_retries)):
        if verbose:
            print(attempt)
        try:
            result = function(**kwargs)
            if result is not None:
                return result
            time.sleep(0.3)
        except:
            pass
    return None


def compute_fits_header_data_size(header): 
    """Estimate the uncompressed size of a FITS HDU in megabytes (MB) using header metadata.

    This function calculates the size of a FITS file or HDU (Header + Data Unit)
    using only its header. It accounts for the standard FITS structure, where both
    headers and data are stored in blocks of 2880 bytes.

    The size is computed by:
    - Estimating the number of 80-character header cards, padded to 2880 bytes.
    - Using NAXIS, BITPIX, and NAXISn keywords to compute data block size,
      also padded to 2880 bytes.

    This works for image HDUs (e.g., BITPIX = 8, 16, 32, -32, -64) and does not
    currently handle BINTABLE, compressed image HDUs, or variable-length arrays.

    Args:
        header (astropy.io.fits.Header): The FITS header from a primary or extension HDU.

    Returns:
        float: The estimated size in **megabytes (MB)** of the HDU including header and data blocks.
               This reflects the **uncompressed** FITS storage size on disk.
               The estimate is rounded to 3 digits after the decimal point.

    Example:
        >>> from astropy.io import fits
        >>> with fits.open("example.fits") as hdul:
        ...     size = compute_fits_header_data_size(hdul[0].header)
        ...     print(f"{size:.2f} MB")

    Notes:
        - The result will differ from `os.path.getsize()` if the file is compressed.
        - For full file size of a multi-extension FITS, sum the result for each HDU.
    """
    # Compute header size (always a multiple of 2880 bytes)
    header_cards = len(header)
    header_size = ((header_cards * 80 + 2880 - 1) // 2880) * 2880

    # Compute data size
    if header.get('NAXIS', 0) == 0:
        data_size = 0
    else:
        naxis = header['NAXIS']
        bitpix = header['BITPIX']
        dims = [header[f'NAXIS{i}'] for i in range(1, naxis + 1)]
        num_elements = np.prod(dims)
        bytes_per_element = abs(bitpix) // 8
        data_size_raw = num_elements * bytes_per_element
        data_size = ((data_size_raw + 2880 - 1) // 2880) * 2880

    total_size_bytes = header_size + data_size
    return np.round(total_size_bytes / (1024 ** 2), 3)  # Convert to MB


def matches_pattern(s):
    """
    Check if the last two characters of a string match the pattern.

    Parameters
    ----------
    s : str
        The string to be checked.

    Returns
    -------
    bool
        True if the string matches the pattern, False otherwise.
    """

    pattern = r"( [b-h]|[1-9][b-h]|[1-9][B-D])$"
    return bool(re.search(pattern, s))


def parse_boolean_column(column):
    """
    Convert a column of strings like 'True', 'False', '1', '0' into a boolean mask.
    Accepts either a NumPy array or an Astropy Column.
    """
    lowered = np.char.lower(column.astype(str))
    return np.isin(lowered, ["true", "t", "1"])


def filter_for_science_frames(
    table_of_files,
    instrument: str,
    polarimetry: bool = False,
    remove_fillers: bool = True,
):
    """
    Filters science frames by instrument type, polarimetry status, and optional filler removal.

    Parameters
    ----------
    table_of_files : astropy.table.Table
        Input table containing header keyword metadata.
    instrument : str
        Either 'irdis' or 'ifs'. Filters rows based on DET_ID.
    polarimetry : bool
        If True, includes only frames with 'DPR_TECH' containing 'POLARIMETRY'.
        If False, excludes such frames.
    remove_fillers : bool
        If True, removes rows where 'OBJECT' contains 'filler'.

    Returns
    -------
    Tuple[Table or None, Table or None, Table or None, Table or None, Table or None]
        t_phot, t_center, t_coro, t_center_coro, t_science
    """
    instrument = instrument.lower()
    if instrument not in ("irdis", "ifs"):
        raise ValueError("Instrument must be 'irdis' or 'ifs'.")

    t_instrument = table_of_files[table_of_files["DET_ID"] == instrument.upper()]
    dpr_type_col = np.char.upper(t_instrument["DPR_TECH"].astype(str))
    shutter_mask = parse_boolean_column(t_instrument["SHUTTER"])

    # Base science mask
    base_mask = np.logical_and.reduce(
        (
            t_instrument["DEC"] != -10000,
            dpr_type_col != "DARK",
            dpr_type_col != "FLAT,LAMP",
            dpr_type_col != "OBJECT,ASTROMETRY",
            dpr_type_col != "STD",
            t_instrument["CORO"].astype(str) != "N/A",
            t_instrument["READOUT_MODE"].astype(str) == "Nondest",
            shutter_mask,  # Assumed to be clean boolean
        )
    )

    t_science = t_instrument[base_mask]

    # Apply polarimetry inclusion/exclusion filter for irdis
    if instrument == 'irdis':
        tech_col = np.char.upper(t_science["DPR_TECH"].astype(str))
        if polarimetry:
            t_science = t_science[np.char.find(tech_col, "POLARIMETRY") >= 0]
        else:
            t_science = t_science[np.char.find(tech_col, "POLARIMETRY") == -1]

    # Remove filler targets
    if remove_fillers:
        object_col = np.char.lower(t_science["OBJECT"].astype(str))
        filler_mask = np.char.find(object_col, "filler") == -1
        t_science = t_science[filler_mask]

    def safe_filter_by_type(dpr_types):
        if isinstance(dpr_types, str):
            dpr_types = [dpr_types]
        mask = np.isin(t_science["DPR_TYPE"], dpr_types)
        result = t_science[mask]
        try:
            n_keys = len(result.group_by("OBJECT").groups.keys)
            print(f"Number of Object keys for {', '.join(dpr_types)}: {n_keys}")
        except Exception:
            print(f"No frames for {', '.join(dpr_types)}.")
            return None
        return result if len(result) > 0 else None

    t_phot = safe_filter_by_type("OBJECT,FLUX")
    t_coro = safe_filter_by_type("OBJECT")
    t_center = safe_filter_by_type("OBJECT,CENTER")
    t_center_coro = safe_filter_by_type(["OBJECT", "OBJECT,CENTER"])
    t_science = safe_filter_by_type(["OBJECT", "OBJECT,CENTER", "OBJECT,FLUX", "SKY"])

    return t_phot, t_center, t_coro, t_center_coro, t_science