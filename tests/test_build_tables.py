from unittest.mock import patch

import numpy as np
from astropy.table import Table

from spherical.database import build
from spherical.database import provenance as prov


def _fake_file_table():
    return Table({
        "DP.ID": ["a", "b"],
        "NIGHT_START": ["2016-09-15", "2016-09-16"],
        "RA": [10.0, 10.0],
        "DEC": [-5.0, -5.0],
    })


def _fake_target_table():
    return Table({"MAIN_ID": ["Beta Pic"], "RA_HEADER": [10.0], "DEC_HEADER": [-5.0]})


def _fake_obs_table():
    return Table({"MAIN_ID": ["Beta Pic"], "NIGHT_START": ["2016-09-15"]})


def test_build_tables_writes_files_and_returns_provenance(tmp_path):
    with patch.object(build.target_table, "make_target_list_with_SIMBAD",
                      return_value=(_fake_target_table(), [])), \
         patch.object(build, "query_mocadb_for_targets", side_effect=lambda t, **k: t), \
         patch.object(build, "query_gaia_astrophysical_params", side_effect=lambda t, **k: t), \
         patch.object(build.observation_table, "create_observation_table",
                      return_value=(_fake_obs_table(), _fake_target_table())):
        p = build.build_tables(tmp_path, "ifs", _fake_file_table(),
                               J_mag_limit=14.0, cone_size_science=15.0)

    assert (tmp_path / "table_of_targets_ifs.fits").exists()
    assert (tmp_path / "table_of_observations_ifs.fits").exists()
    assert p.mode == "ifs"
    assert p.eso_coverage_end == "2016-09-16"
    assert p.enrichment == {"gaia": "ok", "moca": "ok"}
    assert p.build_parameters["J_mag_limit"] == 14.0
    # Provenance is embedded in the written observation table.
    got = prov.extract_from_meta(Table.read(tmp_path / "table_of_observations_ifs.fits"))
    assert got["mode"] == "ifs"


def test_build_tables_records_failed_enrichment(tmp_path):
    def boom(t, **k):
        raise RuntimeError("MOCA down")

    with patch.object(build.target_table, "make_target_list_with_SIMBAD",
                      return_value=(_fake_target_table(), [])), \
         patch.object(build, "query_mocadb_for_targets", side_effect=boom), \
         patch.object(build, "query_gaia_astrophysical_params", side_effect=lambda t, **k: t), \
         patch.object(build.observation_table, "create_observation_table",
                      return_value=(_fake_obs_table(), _fake_target_table())):
        p = build.build_tables(tmp_path, "ifs", _fake_file_table())

    assert p.enrichment["moca"] == "failed"
    assert p.enrichment["gaia"] == "ok"
    assert (tmp_path / "table_of_observations_ifs.fits").exists()


def test_sam_naming(tmp_path):
    with patch.object(build.target_table, "make_target_list_with_SIMBAD",
                      return_value=(_fake_target_table(), [])), \
         patch.object(build, "query_mocadb_for_targets", side_effect=lambda t, **k: t), \
         patch.object(build, "query_gaia_astrophysical_params", side_effect=lambda t, **k: t), \
         patch.object(build.observation_table, "create_observation_table",
                      return_value=(_fake_obs_table(), _fake_target_table())):
        build.build_tables(tmp_path, "ifs", _fake_file_table(), sparse_aperture_masking=True)
    assert (tmp_path / "table_of_observations_ifs_sam.fits").exists()
