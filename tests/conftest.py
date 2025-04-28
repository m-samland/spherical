import pytest
from astropy.table import Table
from unittest.mock import patch

from spherical.database.file_table import make_file_table
from spherical.database.target_table import make_target_list_with_SIMBAD
from spherical.database.observation_table import create_observation_table
from spherical.database.sphere_database import Sphere_database

FILE_ENDING = "test"
INSTRUMENT = "ifs"
START_DATE = '2016-09-15'

@pytest.fixture(scope="session")
def persistent_table_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("persistent_sphere_database")
    return path

@pytest.fixture(scope="session")
def initial_file_table(persistent_table_path):
    end_date = '2016-09-16'

    table_of_files = make_file_table(
        output_dir=persistent_table_path,
        instrument=INSTRUMENT,
        start_date=START_DATE,
        end_date=end_date,
        output_suffix=FILE_ENDING,
        cache=False,
        existing_table_path=None,
        batch_size=100,
        date_batch_months=1,
    )
    return table_of_files

# Extend initial table by one day to test merging and extension of table
@pytest.fixture(scope="session")
def persistent_file_table(persistent_table_path, initial_file_table):
    # Path to the existing file table
    existing_table_path = persistent_table_path / f"table_of_files_{INSTRUMENT}{FILE_ENDING}.csv"

    # Now expand the date range
    end_date = '2016-09-17'  # one day larger

    updated_table_of_files = make_file_table(
        output_dir=persistent_table_path,
        instrument=INSTRUMENT,
        start_date=START_DATE,
        end_date=end_date,
        output_suffix=FILE_ENDING,
        cache=False,
        existing_table_path=existing_table_path,
        batch_size=100,
        date_batch_months=1,
    )

    return updated_table_of_files

@pytest.fixture(scope="session")
def persistent_target_table(persistent_file_table, persistent_table_path):
    table_of_IFS_targets, _ = make_target_list_with_SIMBAD(
        table_of_files=persistent_file_table,
        instrument=INSTRUMENT,
        remove_fillers=False,
        J_mag_limit=15.0,
        search_radius=3.0,
    )

    target_table_path = persistent_table_path / f"persistent_targets_{INSTRUMENT}{FILE_ENDING}.fits"
    table_of_IFS_targets.write(target_table_path, format="fits", overwrite=True)

    return table_of_IFS_targets

@pytest.fixture(scope="session")
def persistent_observation_table(persistent_file_table, persistent_target_table, persistent_table_path):
    observation_table, _ = create_observation_table(
        persistent_file_table,
        persistent_target_table,
        instrument=INSTRUMENT,
        cone_size_science=15.0,
        cone_size_sky=73.0,
        remove_fillers=True,
    )

    observation_table_path = persistent_table_path / f"persistent_observations_{INSTRUMENT}{FILE_ENDING}.fits"
    observation_table.write(observation_table_path, format='fits', overwrite=True)

    return observation_table

@pytest.fixture(scope="session")
def sphere_db(persistent_observation_table, persistent_file_table):
    return Sphere_database(
        table_of_observations=persistent_observation_table,
        table_of_files=persistent_file_table,
        instrument=INSTRUMENT,
    )

@pytest.fixture(scope="session")
def persistent_observation_SIMBAD_table(sphere_db):
    with patch("spherical.database.sphere_database.Simbad.query_object") as mock_query:
        mock_query.return_value = Table({
            'ra': ['05 47 17.0876901'],
            'dec': ['-51 03 59.441135'],
            'main_id': ['* bet Pic']
        })

        observations = sphere_db.get_observation_SIMBAD(
            target_name="* bet Pic",
            obs_band=None,
            date=None,
            summary="SHORT",
            usable_only=True,
            query_radius=10.0
        )

    return observations
