from unittest.mock import patch

import pytest
from astropy.table import Table

from spherical.database.file_table import make_file_table
from spherical.database.observation_table import create_observation_table
from spherical.database.sphere_database import SphereDatabase
from spherical.database.target_table import make_target_list_with_SIMBAD

FILE_ENDING = "test"
INSTRUMENT = "ifs"
POLARIMETRY = False
START_DATE = '2016-09-15'


@pytest.fixture(autouse=True)
def _isolate_astroquery_cache(tmp_path, monkeypatch):
    """Point astroquery's on-disk response cache at a throwaway directory.

    `query_eso_data` calls `Eso.query_instrument` without passing `cache=`, so
    astroquery's default `cache=True` applies and `make_file_table(cache=False)` does not
    actually reach it. A warm developer cache then serves the query with no HTTP request
    at all — which silently recorded cassettes containing the per-DP.ID header fetches but
    *none* of the queries that produce those DP.IDs. Those replay only on the machine that
    recorded them; anywhere else the unmatched query is an error.

    Isolating the cache here keeps recording reproducible and stops a developer's cache
    from masking a request the cassette needs to contain.

    Patched on the *class*, not on the imported `Eso`: astroquery exports a module-level
    instance, `cache_location` is a property on `BaseQuery`, and `file_table` builds its
    own object with `Eso()`. Setting the attribute on the exported instance leaves that
    fresh object still pointing at the developer's real cache.
    """
    from astroquery.eso import Eso

    cache_dir = tmp_path / "astroquery_eso"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(type(Eso), "cache_location", cache_dir, raising=False)


@pytest.fixture(scope="module")
def vcr_config():
    """VCR settings for the recorded ESO archive traffic.

    `record_mode` is deliberately not set here: pytest-recording already defaults to
    "none" (an unmatched request is an error, not a silent live call), and setting it
    in this dict would override `--record-mode=rewrite` and make re-recording impossible.

    `body` is deliberately NOT matched on. astroquery 0.4.10 submits the WDB query as a
    multipart POST whose boundary is regenerated per request
    (``--ab8319b16558684f024ba6ce6993d06a``), so the body never compares equal between
    recording and replay. Including it silently fails every query match, and because
    `query_eso_data` swallows the resulting error the tests then run against empty tables.

    The cost of leaving it out: the two POSTs per batch (calibration, then science) go to
    the same URL, so vcr distinguishes them only by recorded order. That is deterministic
    here — `make_file_table` always issues them in that order — but it does mean a change
    to *what* is asked for would not be caught by a match failure. The header GETs carry
    their DP.ID in the query string and are matched exactly.
    """
    return {
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "filter_headers": ["authorization", "cookie", "set-cookie"],
        "decode_compressed_response": True,
    }

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
        polarimetry=POLARIMETRY,
        remove_fillers=False,
        J_mag_limit=15.0,
        search_radius=3.0,
        group_by_healpix=True,
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
        polarimetry=POLARIMETRY,
        cone_size_science=15.0,
        remove_fillers=False,
        group_by_time_gaps=False,
        reorder_columns=True,
    )

    observation_table_path = persistent_table_path / f"persistent_observations_{INSTRUMENT}{FILE_ENDING}.fits"
    observation_table.write(observation_table_path, format='fits', overwrite=True)

    return observation_table

@pytest.fixture(scope="session")
def sphere_db(persistent_observation_table, persistent_file_table):
    return SphereDatabase(
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
            usable_only=True
        )

    return observations
