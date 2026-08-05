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
