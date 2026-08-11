from unittest.mock import patch

from astropy.table import Table

from spherical.database import build
from spherical.database import provenance as prov


def _seed(tmp_path):
    # A pre-existing target table and a local file table CSV.
    Table({"MAIN_ID": ["Beta Pic"], "RA_HEADER": [10.0], "DEC_HEADER": [-5.0]}).write(
        tmp_path / "table_of_targets_ifs.fits", format="fits", overwrite=True)
    Table({"DP.ID": ["a"], "NIGHT_START": ["2016-09-15"], "RA": [10.0], "DEC": [-5.0]}).write(
        tmp_path / "table_of_files_ifs.csv", format="csv", overwrite=True)


def test_enrich_only_updates_status_without_eso(tmp_path):
    _seed(tmp_path)
    with patch.object(build, "query_mocadb_for_targets", side_effect=lambda t, **k: t), \
         patch.object(build, "query_gaia_astrophysical_params", side_effect=lambda t, **k: t), \
         patch.object(build.observation_table, "create_observation_table",
                      return_value=(Table({"MAIN_ID": ["Beta Pic"]}),
                                    Table({"MAIN_ID": ["Beta Pic"]}))), \
         patch.object(build.file_table, "make_file_table") as mocked_eso:
        record = build.enrich_tables(tmp_path, "ifs")

    mocked_eso.assert_not_called()
    assert record.enrichment["gaia"]["status"] == "ok"
    assert record.enrichment["moca"]["status"] == "ok"
    assert record.gaia_query_utc is not None
    stored = prov.read_provenance(tmp_path)["ifs"]
    assert stored.enrichment["gaia"]["status"] == "ok"
