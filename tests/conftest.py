from unittest.mock import patch

import pytest
from astropy.table import Table

from spherical.sphere_database.master_file_table import make_master_file_table
from spherical.sphere_database.observation_table import create_observation_table
from spherical.sphere_database.sphere_database import Sphere_database
from spherical.sphere_database.target_table import make_target_list_with_SIMBAD

FILE_ENDING = "test"


@pytest.fixture(scope="session")
def persistent_table_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("persistent_sphere_database")
    return path


@pytest.fixture(scope="session")
def persistent_master_table(persistent_table_path):
    start_date = '2016-09-15'
    end_date = '2016-09-16'

    table_of_files = make_master_file_table(
        persistent_table_path,
        start_date=start_date,
        end_date=end_date,
        file_ending=FILE_ENDING,
        save=True,
        savefull=False,
        cache=False,
        existing_master_file_table_path=None,
        batch_size=50,
    )
    return table_of_files


@pytest.fixture(scope="session")
def persistent_target_table(persistent_master_table, persistent_table_path):
    table_of_IFS_targets, not_found_IFS = make_target_list_with_SIMBAD(
        table_of_files=persistent_master_table,
        instrument="IFS",
        remove_fillers=False,
        J_mag_limit=10.0,
        search_radius=3.0,
    )

    target_table_path = persistent_table_path / f"persistent_IFS_targets_{FILE_ENDING}.fits"
    table_of_IFS_targets.write(target_table_path, format="fits", overwrite=True)

    return table_of_IFS_targets


@pytest.fixture(scope="session")
def persistent_observation_table(persistent_master_table, persistent_target_table, persistent_table_path):
    observation_table, updated_target_table = create_observation_table(
        persistent_master_table,
        persistent_target_table,
        instrument="IFS",
        cone_size_science=15.0,
        cone_size_sky=73.0,
        remove_fillers=True,
    )

    observation_table_path = persistent_table_path / f"persistent_IFS_observations_{FILE_ENDING}.fits"
    observation_table.write(observation_table_path, format='fits', overwrite=True)

    return observation_table


@pytest.fixture(scope="session")
def sphere_db(persistent_observation_table, persistent_master_table):
    return Sphere_database(
        table_of_observations=persistent_observation_table,
        table_of_files=persistent_master_table,
        instrument="IFS"
    )

@pytest.fixture(scope="session")
def persistent_observation_SIMBAD_table(sphere_db):
    with patch("spherical.sphere_database.sphere_database.Simbad.query_object") as mock_query:
        mock_query.return_value = Table({
            'ra': ['05 47 17.0876901'],
            'dec': ['-51 03 59.441135'],
            'main_id': ['* bet Pic']
        })

        # Call method to get observations matching the mock query
        observations = sphere_db.get_observation_SIMBAD(
            target_name="* bet Pic",
            obs_band=None,  # Test without specifying a filter
            date=None,      # Test without specifying a date
            summary="SHORT",
            usable_only=True,
            query_radius=10.0
        )

    return observations