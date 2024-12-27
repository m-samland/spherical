#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = (
    "M. Samland @ MPIA (Heidelberg, Germany), J. Kemmer @ MPIA (Heidelberg, Germany)"
)

import glob
import itertools
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.ascii.core import InconsistentTableError
from astropy.table import Column, Table, TableMergeError, hstack, join, unique, vstack
from astropy.time import Time
from astroquery.gaia import Gaia

from tqdm import tqdm



def add_night_start_date(table, key="DATE_OBS"):
    if "NIGHT_START" not in table.keys():
        night_start = []
        times = Time(table[key]).to_datetime()
        for time in times:
            if time.hour < 12:
                new_time = time - timedelta(1)
            else:
                new_time = time
            night_start.append(str(new_time.date()))
        table["NIGHT_START"] = night_start
    return table


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


# #eri_list = retrieve_observations_from_name(table_of_obs, target_name='51 Eri')
#
# # SELECT TARGET FROM COORDINATES
# # SINGLE TARGET REDUCTION
#
#
# def collect_reduction_results(reduction_result_directory):
#     """ Crawls through directory tree given by reduction_result_directory
#     parameter and collects the content of reduction_info.fits for each reduction
#     as rows of a summary table.
#     """
#
#     files = [y for x in os.walk(reduction_result_directory)
#              for y in glob.glob(os.path.join(x[0], 'reduction_info.fits'))]
#     file_col = Column(name='FILE', data=files)
#     table_list = []
#     for f in files:
#         table_list.append(Table.read(f))
#     table_of_reductions = vstack(table_list)
#
#     return table_of_reductions
#
#
# #reduction_result_directory = '/mnt/fhgfs/sphere_shared/automatic_reduction/reductions/'
# #table_of_reductions = collect_reduction_results(reduction_result_directory)
# #table_of_reductions = Table.read('Table_of_reduction_andromed_ext_test.fits')
#
# def read_single_andromeda_result_file(path, filter_name):
#     andromeda_table = Table.read(path, format='ascii')
#     number_of_detections = len(andromeda_table)
#     filter_column = []
#     for i in range(number_of_detections):
#         filter_column.append(filter_name)
#     filter_column = Column(data=filter_column, name='Filter')
#     andromeda_table.add_column(filter_column, index=0)
#
#     return andromeda_table
#
#
# def andromeda_read_in_wrapper(list_of_paths, filename_keyword, filter_name, max_length=99):
#     try:
#         if any(filename_keyword in file for file in list_of_paths):
#             index = [i for i, s in enumerate(list_of_paths) if filename_keyword in s][0]
#             detections_table = read_single_andromeda_result_file(list_of_paths[index],
#                                                                  filter_name=filter_name)
#             if len(detections_table) < max_length:
#                 return detections_table
#             else:
#                 return None
#     except:
#         return None
#
#
# def create_detection_table_andromeda(table_of_reductions):
#
#     all_tables = []
#     for idx, reduction in enumerate(table_of_reductions):
#         print('Reading in files: {} of {}'.format(idx + 1, len(table_of_reductions)))
#         print(reduction['MAIN_ID'], reduction['DATE_SHORT'], reduction['DB_FILTER'])
#         wdir = os.path.dirname(reduction['FILE'])
#         date = reduction['DATE_SHORT']
#         db_filter = reduction['DB_FILTER']
#
#         andromeda_file_paths = [y for x in os.walk(wdir) for y in glob.glob(os.path.join(x[0], '*Results_Files.dat'))]
#         #print(list(map(os.path.basename, andromeda_file_paths)))
#         # print(andromeda_file_paths)
#         if len(andromeda_file_paths) is 0:
#             continue
#
#         names = list(map(os.path.basename, andromeda_file_paths))
#         print(names)
#         #filenames = list(map(os.path.basename, andromeda_file_paths))
#         #['DB_H23', 'DB_K12', 'DB_Y23', 'BB_J', 'BB_H', 'BB_Ks', 'DB_NDH23']
#         #any('channel1' in file for file in names)
#
#         broadband = False
#         if db_filter == 'DB_H23':
#             filter_name_channel1 = 'H2'
#             filter_name_channel2 = 'H3'
#             sdi_name = 'H2-H3'
#             forced_name = 'H3forced'
#             combined_name = 'H2+H3'
#         elif db_filter == 'DB_K12':
#             filter_name_channel1 = 'K1'
#             filter_name_channel2 = 'K2'
#             sdi_name = 'K1-K2'
#             forced_name = 'K2forced'
#             combined_name = 'K1+K2'
#         elif db_filter == 'DB_Y23':
#             filter_name_channel1 = 'Y2'
#             filter_name_channel2 = 'Y3'
#             forced_name = 'Y3forced'
#             sdi_name = 'Y2-Y3'
#             combined_name = 'Y2+Y3'
#         elif db_filter == 'DB_NDH23':
#             filter_name_channel1 = 'NDH2'
#             filter_name_channel2 = 'NDH3'
#             sdi_name = 'NDH2-NDH3'
#             forced_name = 'NDH3forced'
#             combined_name = 'NDH2+NDH3'
#         elif db_filter == 'BB_J':
#             broadband = True
#             combined_name = 'BB_J'
#         elif db_filter == 'BB_H':
#             broadband = True
#             combined_name = 'BB_H'
#         elif db_filter == 'BB_Ks':
#             broadband = True
#             combined_name = 'BB_Ks'
#         else:
#             raise ValueError('Non supported filter {}'.format(db_filter))
#
#         # left and right image combined table should always exist
#         max_length = 99
#         stacked_andromeda_table = []
#         if broadband is True:
#             detections_table = andromeda_read_in_wrapper(
#                 list_of_paths=andromeda_file_paths, filename_keyword='combined',
#                 filter_name=combined_name, max_length=max_length)
#             if detection_table is not None:
#                 stacked_andromeda_table.append(detection_table)
#         else:
#             if any('channel1' in file for file in names):
#                 try:
#                     index_channel_1 = [i for i, s in enumerate(andromeda_file_paths) if 'channel1' in s][0]
#                     channel1_table = read_single_andromeda_result_file(
#                         andromeda_file_paths[index_channel_1], filter_name=filter_name_channel1)
#                     if len(channel1_table) < 99:
#                         stacked_andromeda_table.append(channel1_table)
#                 except:
#                     pass
#             if any('channel2' in file for file in names):
#                 try:
#                     index_channel_2 = [i for i, s in enumerate(andromeda_file_paths) if 'channel2' in s][0]
#                     channel2_table = read_single_andromeda_result_file(
#                         andromeda_file_paths[index_channel_2], filter_name=filter_name_channel2)
#                     if len(channel2_table) < 99:
#                         stacked_andromeda_table.append(channel2_table)
#                 except:
#                     pass
#             if any('SADI' in file for file in names):
#                 try:
#                     index_sdi = [i for i, s in enumerate(andromeda_file_paths) if 'SADI' in s][0]
#                     sdi_table = read_single_andromeda_result_file(andromeda_file_paths[index_sdi], filter_name=sdi_name)
#                     if len(sdi_table) < 99:
#                         stacked_andromeda_table.append(sdi_table)
#                 except:
#                     pass
#
#         if len(stacked_andromeda_table) is 0:
#             continue
#
#         stacked_andromeda_table = vstack(stacked_andromeda_table)
#
#         base_table = []
#         for i in range(len(stacked_andromeda_table)):
#             base_table.append(reduction)
#         base_table = vstack(base_table)
#
#         companion_table = hstack([base_table, stacked_andromeda_table])
#         companion_table.write(os.path.join(wdir, 'andromeda_detections.fits'), overwrite=True)
#
#         all_tables.append(companion_table)
#
#     all_tables = vstack(all_tables)
#
#     return all_tables
