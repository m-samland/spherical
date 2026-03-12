"""Generate SPHERE database tables (file table, target table, observation table).

Usage Modes
-----------
1. **Build from scratch**:
   Set ``build_file_table = True`` and choose a start/end date range.
   The file table is created by querying the ESO archive for all SPHERE files
   in that range, retrieving their FITS headers, and saving the result as a CSV.
   This can take hours for large date ranges (e.g. the full archive 2014–today).

2. **Extend an existing table** (e.g. add new observations):
   Keep ``build_file_table = True`` and widen the date range (e.g. update
   ``end_date`` to today). The function automatically detects the existing output
   file, loads all DP.IDs that were already downloaded, and only retrieves
   headers for new files. This is the recommended way to keep a database
   up to date.

3. **Resume an interrupted build**:
   Simply re-run the script with the same parameters. If a previous run was
   interrupted, a ``*_partial.csv`` file will exist alongside the output.
   On the next run, already-downloaded entries from both the output file and the
   partial file are skipped. Once all batches complete, the partial file is
   merged into the final output and deleted. No manual intervention needed.

4. **Skip file table generation** (use existing):
   Set ``build_file_table = False`` to load the file table from disk and only
   rebuild the target and/or observation tables.

Notes
-----
- The ``existing_table_path`` parameter is kept for backward compatibility. When
  set, its entries are also used to skip known files. In most cases you can leave
  it pointed at the output file or remove it entirely — the auto-detection
  handles everything.
- Set ``resume=False`` in ``make_file_table()`` to force a clean download,
  discarding any partial progress from a previous interrupted run.
"""

import warnings
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack

from spherical.database import file_table, observation_table, target_table
from spherical.database.gaia_astrophysical_params import query_gaia_astrophysical_params
from spherical.database.mocadb_matching import query_mocadb_for_targets
from spherical.database.sphere_database import SphereDatabase

warnings.filterwarnings("ignore")

# Define path to store the database
table_path = Path.home() / "data/sphere/database"
table_path.mkdir(parents=True, exist_ok=True)

# Choose instrument ('ifs', 'irdis')
instrument = "ifs"
polarimetry = False

# Set file name ending for the database files, e.g. "_test"
output_suffix = ''

# If True, ignore output_suffix and overwrite the canonical tables in place
# (table_of_files_<instrument>.csv, table_of_targets_<mode>.fits, etc.).
# If False, write new tables with the output_suffix appended to their names.
update_in_place = True

# Optional: point to an external file table to merge with.
# In most cases this is not needed — the function auto-detects the output file
# and any partial file from a previous interrupted run.
existing_file_table = table_path / f"table_of_files_{instrument}.csv"

# Set which tables to build. File table is required for target table,
# and target table is required for observation table.
# If not building the tables, the existing tables will be read from disk.
build_file_table = True
start_date = '2016-09-15'  # None or e.g. "2016-09-15"
end_date = '2016-09-16'    # None or e.g. "2016-09-16"

# Build target table
build_target_table = True
J_mag_limit = 14.
parallax_limit = 1e-3
search_radius = 3.0

# Build observation table
build_observation_table = True

if instrument == 'ifs':
    mode_name = f"{instrument}"
elif instrument == 'irdis':
    mode_name = f"{instrument}_pol_{polarimetry}"

# Resolve the suffix actually used for output file names
file_suffix = '' if update_in_place else output_suffix


if build_file_table:
    table_of_files = file_table.make_file_table(
        output_dir=table_path,
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        output_suffix=file_suffix,
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
        table_path / f"table_of_targets_{mode_name}{file_suffix}.fits",
        format="fits",
        overwrite=True,
    )
else:
    try:
        table_of_targets = Table.read(
            table_path / f"table_of_targets_{mode_name}{file_suffix}.fits"
        )
    except FileNotFoundError:
        table_of_targets = None
        print("Target table not found.")

# --- Optional: enrich target table with MOCAdb age / association data ---
enrich_with_mocadb = True
if enrich_with_mocadb and table_of_targets is not None:
    try:
        table_of_targets = query_mocadb_for_targets(
            table_of_targets, include_tier2=True
        )
        table_of_targets.write(
            table_path / f"table_of_targets_{mode_name}{file_suffix}.fits",
            format="fits",
            overwrite=True,
        )
    except Exception as e:
        print(f"MOCAdb enrichment failed: {e}")

# --- Optional: enrich target table with Gaia DR3 astrophysical parameters ---
enrich_with_gaia = True
if enrich_with_gaia and table_of_targets is not None:
    try:
        table_of_targets = query_gaia_astrophysical_params(table_of_targets)
        table_of_targets.write(
            table_path / f"table_of_targets_{mode_name}{file_suffix}.fits",
            format="fits",
            overwrite=True,
        )
    except Exception as e:
        print(f"Gaia astrophysical parameter enrichment failed: {e}")

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
    table_of_observations.write(
        table_path / f"table_of_observations_{mode_name}{file_suffix}.fits",
        format='fits',
        overwrite=True,
    )
else:
    try:
        table_of_observations = Table.read(
            table_path / f"table_of_observations_{mode_name}{file_suffix}.fits"
        )
    except FileNotFoundError:
        table_of_observations = None
        print("Observation table not found.")

print("Done.")

# Example of initializing database object and retrieving observation objects
database = SphereDatabase(
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
observation_object_list = database.retrieve_observation_metadata(obs_table)
print(observation_object_list[0].frames)
