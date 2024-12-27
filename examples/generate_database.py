import os
import warnings
from pathlib import Path

from astropy.table import Table

from spherical.sphere_database import master_file_table, observation_table, target_table

__author__ = "M. Samland @ MPIA (Heidelberg, Germany)"

warnings.filterwarnings("ignore")

# Define path to store the database
table_path = os.path.join(str(Path.home()), "sphere/database")
if not os.path.exists(table_path):
    os.makedirs(table_path)

# Name of file that contains the table of sphere files. None if building tables from scratch.
# If existing file is provided, and build_file_table is set to True, the existing file will be updated if 
# the date range includes new files, this can save a lot of time.
existing_master_file_table = "table_of_files_all_2024_07_29.csv"

# Set file name ending for the database files
file_ending = "all_2024_07_29"

# Set which tables to build, file tale is required for target table, and target table is required for observation table
build_file_table = False
build_target_table = True
build_observation_table = True

# Set date range for file table on which to build target and observation sequence table
start_date = None #e.g. "2023-12-01"
end_date = None #e.g. "2024-04-01"

if build_file_table:
    table_of_files = master_file_table.make_master_file_table(
        table_path,
        start_date=start_date,
        end_date=end_date,
        file_ending=file_ending,
        save=True, savefull=False,
        existing_master_file_table_path=existing_master_file_table)
else:
    table_of_files = Table.read(os.path.join(
        table_path, existing_master_file_table))

if build_target_table:
    table_of_IFS_targets, not_found_IFS = target_table.make_target_list_with_SIMBAD(
        table_of_files=table_of_files,
        instrument="IFS",
        remove_fillers=False,
        J_mag_limit=10.0,
        search_radius=3.0,
    )
    table_of_IFS_targets.write(
        os.path.join(table_path, f"table_of_IFS_targets_{file_ending}.fits"),
        format="fits",
        overwrite=True,
    )
else:
    try:
        table_of_IFS_targets = Table.read(
            os.path.join(table_path, f"table_of_IFS_targets_{file_ending}.fits")
        )
    except FileNotFoundError:
        table_of_IFS_targets = None
        print("Target table not found.")


if build_observation_table:
    (
        table_of_IFS_observations,
        table_of_IFS_targets,
    ) = observation_table.create_observation_table(
        table_of_files,
        table_of_IFS_targets,
        instrument="IFS",
        cone_size_science=15.0,
        cone_size_sky=73.0,
        remove_fillers=True,
    )

    table_of_IFS_observations.write(
        os.path.join(table_path, f"table_of_IFS_observations_{file_ending}.fits"),
        format='fits',
        overwrite=True,
    )
else:
    try:
        table_of_IFS_observations = Table.read(
            os.path.join(table_path, f"table_of_IFS_observations_{file_ending}.fits")
        )
    except FileNotFoundError:
        table_of_IFS_observations = None
        print("Observation table not found.")

print("Done.")