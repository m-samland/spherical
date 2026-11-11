import warnings

import pytest
from astropy.table import Table

from spherical.database import provenance as prov


def _sample():
    return prov.TableProvenance(
        instrument="ifs", mode="ifs", source="eso-extend",
        spherical_version="1.2.3", generated_utc="2026-07-02T00:00:00Z",
        eso_coverage_start="2014-05-01", eso_coverage_end="2026-06-30",
    )


def test_embed_then_extract():
    t = Table({"a": [1, 2]})
    prov.embed_in_meta(t, _sample())
    got = prov.extract_from_meta(t)
    assert got["mode"] == "ifs"
    assert got["eso_coverage_end"] == "2026-06-30"
    assert got["gaia_data_release"] == "GaiaDR3"


def test_survives_fits_roundtrip(tmp_path):
    t = Table({"a": [1, 2]})
    prov.embed_in_meta(t, _sample())
    path = tmp_path / "t.fits"
    t.write(path, format="fits", overwrite=True)
    t2 = Table.read(path)
    got = prov.extract_from_meta(t2)
    assert got["spherical_version"] == "1.2.3"
    assert got["eso_coverage_start"] == "2014-05-01"
    assert got["mode"] == "ifs"


def test_tables_and_provenance_file_carry_the_table_format(tmp_path):
    t = Table({"a": [1, 2]})
    prov.embed_in_meta(t, _sample())
    t.write(tmp_path / "t.fits")
    assert prov.extract_from_meta(Table.read(tmp_path / "t.fits"))["table_format"] == prov.TABLE_FORMAT

    prov.write_provenance(tmp_path, {"ifs": _sample()})
    assert prov.read_table_format(tmp_path) == prov.TABLE_FORMAT


def test_only_a_newer_table_format_warns():
    assert "upgrade spherical" in prov.newer_table_format_message(prov.TABLE_FORMAT + 1)
    for table_format in (prov.TABLE_FORMAT, 1, None, "", "unknown"):
        assert prov.newer_table_format_message(table_format) is None


def test_warn_if_newer_table_format_reads_the_header():
    t = Table({"a": [1]})
    t.meta["SPHTFMT"] = prov.TABLE_FORMAT + 1
    with pytest.warns(UserWarning, match="upgrade spherical"):
        prov.warn_if_newer_table_format(t)
    t.meta["SPHTFMT"] = prov.TABLE_FORMAT
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prov.warn_if_newer_table_format(t)
        prov.warn_if_newer_table_format(Table({"a": [1]}))  # tables before the stamp


def test_sphere_database_checks_the_table_format(monkeypatch):
    from spherical.database import sphere_database

    class Checked(Exception):
        pass

    def check(table):
        raise Checked

    monkeypatch.setattr(sphere_database, "warn_if_newer_table_format", check)
    with pytest.raises(Checked):
        sphere_database.SphereDatabase(Table({"MAIN_ID": ["HD 1"]}), Table({"SHUTTER": [True]}))
