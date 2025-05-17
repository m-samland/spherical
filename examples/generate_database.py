import warnings
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack

from spherical.database import file_table, observation_table, target_table
from spherical.database.sphere_database import Sphere_database

warnings.filterwarnings("ignore")

# Define path to store the database
table_path = Path.home() / "data/sphere/database"
table_path.mkdir(parents=True, exist_ok=True)

# Choose instrument ('ifs', 'irdis')
instrument = "ifs"
polarimetry = False

# Set file name ending for the database files, e.g. "_test"
output_suffix = ''

# If existing file is provided, and build_file_table is set to True, the existing file will be updated if
# the date range includes new files, this will save a lot of time.
existing_file_table = table_path / f"table_of_files_{instrument}.csv"

# Set which tables to build, file tale is required for target table, and target table is required for observation table
# If not building the tables, the existing tables will be read from the path
build_file_table = True
start_date = '2016-09-15'  # None or e.g. "2016-09-15"
end_date = '2016-09-16'    # None or e.g. "2016-09-16"

# Build target table
build_target_table = True
J_mag_limit = 15.
parallax_limit = 1e-3
search_radius = 3.0

# Build observation table
build_observation_table = True

if instrument == 'ifs':
    mode_name = f"{instrument}"
elif instrument == 'irdis':
    mode_name = f"{instrument}_pol_{polarimetry}"


if build_file_table:
    table_of_files = file_table.make_file_table(
        output_dir=table_path,
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        output_suffix=output_suffix,
        existing_table_path=existing_file_table,
        batch_size=150,
        cache=False,
    )
else:
    table_of_files = Table.read(table_path / existing_file_table)

if build_target_table:
    table_of_targets, not_found = target_table.make_target_list_with_SIMBAD(
        table_of_files=table_of_files,
        instrument=instrument,
        polarimetry=polarimetry,
        remove_fillers=False,
        parallax_limit=parallax_limit,
        J_mag_limit=J_mag_limit,
        search_radius=search_radius,
        batch_size=250,
        min_delay=1.0,
        group_by_healpix=True,
    )    
    table_of_targets.write(
        table_path / f"table_of_targets_{mode_name}{output_suffix}.fits",
        format="fits",
        overwrite=True,
    )
else:
    try:
        table_of_targets = Table.read(
            table_path / f"table_of_targets_{mode_name}{output_suffix}.fits"
        )
    except FileNotFoundError:
        table_of_targets = None
        print("Target table not found.")

if build_observation_table:
    (
        table_of_observations,
        table_of_targets,
    ) = observation_table.create_observation_table(
        table_of_files=table_of_files,
        table_of_targets=table_of_targets,
        instrument=instrument,
        polarimetry=polarimetry,
        cone_size_science=15.0,
        remove_fillers=False,
        reorder_columns=True,
    )
    # output_suffix = 'refactor'
    table_of_observations.write(
        table_path / f"table_of_observations_{mode_name}{output_suffix}.fits",
        format='fits',
        overwrite=True,
    )
else:
    try:
        table_of_observations = Table.read(
            table_path / f"table_of_observations_{mode_name}{output_suffix}.fits"
        )
    except FileNotFoundError:
        table_of_observations = None
        print("Observation table not found.")

print("Done.")

# Example of initializing database object and retrieving observation objects
database = Sphere_database(
    table_of_observations, table_of_files, instrument=instrument)

# Let's get all observations of Beta Pic
target_list = ['Beta Pic']
obs_table = []
for target_name in target_list:
    obs_table.append(database.get_observation_SIMBAD(target_name))

obs_table = vstack(obs_table)

# Filter which observations we want to see
obs_table_mask = np.logical_and.reduce([
    obs_table['TOTAL_EXPTIME_SCI'] > 15,
    obs_table['DEROTATOR_MODE'] == 'PUPIL',
    obs_table['HCI_READY'] == True])

obs_table = obs_table[obs_table_mask]
print(obs_table)

# Each observation object contains information on the associated files
observation_object_list = database.retrieve_observation_object_list(obs_table)
print(observation_object_list[0].frames)
