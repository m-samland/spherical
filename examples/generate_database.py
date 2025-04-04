import warnings
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack

from spherical.sphere_database import file_table, observation_table, target_table
from spherical.sphere_database.sphere_database import Sphere_database

__author__ = "M. Samland @ MPIA (Heidelberg, Germany)"
warnings.filterwarnings("ignore")

# Define path to store the database
table_path = Path.home() / "data/sphere/database"
table_path.mkdir(parents=True, exist_ok=True)

# Name of file that contains the table of sphere files. None if building tables from scratch.
# If existing file is provided, and build_file_table is set to True, the existing file will be updated if
# the date range includes new files, this will save a lot of time.
existing_file_table = table_path / "table_of_IFS_files.csv"

# Set file name ending for the database files
file_ending = ''

# Set which tables to build, file tale is required for target table, and target table is required for observation table
# If not building the tables, the existing tables will be read from the path
build_file_table = True
build_target_table = True
build_observation_table = True

# Set date range for updating or generating file table (ESO archive file headers)
start_date = '2025-01-01'  # None or e.g. "2016-09-15"
end_date = '2026-01-01'    # None or e.g. "2016-09-16"

if build_file_table:
    table_of_files = file_table.make_file_table(
        table_path,
        start_date=start_date,
        end_date=end_date,
        file_ending=file_ending,
        save=True, 
        existing_file_table_path=existing_file_table,
        batch_size=100,
    )
else:
    table_of_files = Table.read(table_path / existing_file_table)

if build_target_table:
    table_of_IFS_targets, not_found_IFS = target_table.make_target_list_with_SIMBAD(
        table_of_files=table_of_files,
        instrument="IFS",
        remove_fillers=False,
        parallax_limit=1e-3,
        J_mag_limit=15.0,
        search_radius=3.0,
        batch_size=250,
        min_delay=1.0,
    )
    
    table_of_IFS_targets.write(
        table_path / f"table_of_IFS_targets{file_ending}.fits",
        format="fits",
        overwrite=True,
    )
else:
    try:
        table_of_IFS_targets = Table.read(
            table_path / f"table_of_IFS_targets{file_ending}.fits"
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
        table_path / f"table_of_IFS_observations{file_ending}.fits",
        format='fits',
        overwrite=True,
    )
else:
    try:
        table_of_IFS_observations = Table.read(
            table_path / f"table_of_IFS_observations{file_ending}.fits"
        )
    except FileNotFoundError:
        table_of_IFS_observations = None
        print("Observation table not found.")

print("Done.")

# # Example of initializing database object and retrieving observation objects
database = Sphere_database(
    table_of_IFS_observations, table_of_files, instrument='IFS')

# Let's get all observations of Beta Pic
target_list = ['Beta Pic']
obs_table = []
for target_name in target_list:
    obs_table.append(database.get_observation_SIMBAD(target_name))

obs_table = vstack(obs_table)

# Filter which observations we want to see
obs_table_mask = np.logical_and.reduce([
    obs_table['TOTAL_EXPTIME'] > 15,
    obs_table['DEROTATOR_MODE'] == 'PUPIL',
    obs_table['FAILED_SEQ'] == False])

obs_table = obs_table[obs_table_mask]
print(obs_table)

# Each observation object contains information on the associated files
observation_object_list = database.retrieve_observation_object_list(obs_table)
print(observation_object_list[0].frames)