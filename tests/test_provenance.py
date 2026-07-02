from pathlib import Path

from spherical.database import provenance as prov


def _sample(mode="ifs"):
    return prov.TableProvenance(
        instrument="ifs",
        mode=mode,
        source="eso-extend",
        spherical_version="1.2.3",
        generated_utc="2026-07-02T00:00:00Z",
        eso_coverage_start="2014-05-01",
        eso_coverage_end="2026-06-30",
        enrichment={"gaia": "ok", "moca": "ok"},
        build_parameters={"J_mag_limit": 14.0, "cone_size_science": 15.0},
    )


def test_to_from_dict_roundtrip():
    p = _sample()
    assert prov.TableProvenance.from_dict(p.to_dict()) == p


def test_write_then_read(tmp_path):
    prov.write_provenance(tmp_path, {"ifs": _sample("ifs")})
    got = prov.read_provenance(tmp_path)
    assert got["ifs"] == _sample("ifs")


def test_read_missing_returns_empty(tmp_path):
    assert prov.read_provenance(tmp_path) == {}


def test_write_merges_and_preserves_other_modes(tmp_path):
    prov.write_provenance(tmp_path, {"ifs": _sample("ifs")})
    prov.write_provenance(tmp_path, {"ifs_sam": _sample("ifs_sam")})
    got = prov.read_provenance(tmp_path)
    assert set(got) == {"ifs", "ifs_sam"}


def test_provenance_file_is_visible_json(tmp_path):
    prov.write_provenance(tmp_path, {"ifs": _sample("ifs")})
    assert (tmp_path / "database_provenance.json").exists()


def test_gaia_release_default():
    assert prov.TableProvenance(instrument="ifs", mode="ifs").gaia_data_release == "GaiaDR3"


def test_read_corrupt_record_returns_empty(tmp_path):
    import json
    path = tmp_path / "database_provenance.json"
    # Valid JSON, but a record missing the required 'instrument'/'mode' fields.
    path.write_text(json.dumps({"tables": {"ifs": {"source": "zenodo"}}}))
    assert prov.read_provenance(tmp_path) == {}
