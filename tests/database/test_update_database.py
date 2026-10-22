from unittest.mock import patch

from astropy.table import Table

from spherical.database import build
from spherical.database import provenance as prov


def test_modes_for_ifs():
    modes = build.modes_for_instrument("ifs")
    assert {"polarimetry": False, "sparse_aperture_masking": False} in modes
    assert {"polarimetry": False, "sparse_aperture_masking": True} in modes
    assert len(modes) == 2


def test_modes_for_irdis_includes_polarimetry_and_sam():
    modes = build.modes_for_instrument("irdis")
    assert {"polarimetry": True, "sparse_aperture_masking": False} in modes
    assert {"polarimetry": False, "sparse_aperture_masking": True} in modes
    assert len(modes) == 3


def test_modes_skip_sam():
    assert all(not m["sparse_aperture_masking"] for m in build.modes_for_instrument("irdis", skip_sam=True))


def test_derive_start_date_from_provenance(tmp_path):
    prov.write_provenance(tmp_path, {"ifs": prov.TableProvenance(
        instrument="ifs", mode="ifs", eso_coverage_end="2026-06-30")})
    got = build.derive_start_date(tmp_path, "ifs", overlap_days=7)
    assert got == "2026-06-23"


def test_derive_start_date_fallback_to_csv(tmp_path):
    Table({"DP.ID": ["a"], "NIGHT_START": ["2020-01-10"]}).write(
        tmp_path / "table_of_files_ifs.csv", format="csv", overwrite=True)
    got = build.derive_start_date(tmp_path, "ifs", overlap_days=5)
    assert got == "2020-01-05"


def test_derive_start_date_none_when_no_baseline(tmp_path):
    assert build.derive_start_date(tmp_path, "ifs", overlap_days=7) is None


def test_update_database_extends_then_builds_all_modes(tmp_path):
    file_tbl = Table({"DP.ID": ["a"], "NIGHT_START": ["2016-09-16"], "RA": [1.0], "DEC": [1.0]})
    built = []

    def fake_build_tables(dest, instrument, table_of_files, **kw):
        mode = build.resolve_mode_name(instrument, kw.get("polarimetry", False),
                                       kw.get("sparse_aperture_masking", False))
        built.append(mode)
        return prov.TableProvenance(instrument=instrument, mode=mode)

    with patch.object(build.file_table, "make_file_table", return_value=file_tbl) as mocked_eso, \
         patch.object(build, "build_tables", side_effect=fake_build_tables):
        result = build.update_database(tmp_path, "irdis", start_date="2016-09-15",
                                       end_date="2016-09-16")

    mocked_eso.assert_called_once()
    assert set(built) == {"irdis", "irdis_polarimetry", "irdis_sam"}
    assert set(result) == {"irdis", "irdis_polarimetry", "irdis_sam"}


def test_update_database_errors_without_baseline_or_start_date(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        build.update_database(tmp_path, "ifs")


def test_suffixed_build_keeps_the_real_tables_provenance(tmp_path):
    """A --suffix build records its own entries and leaves the real ones alone (#216)."""
    real = prov.TableProvenance(instrument="irdis", mode="irdis", eso_coverage_end="2026-06-30")
    prov.write_provenance(tmp_path, {"irdis": real})
    file_tbl = Table({"DP.ID": ["a"], "NIGHT_START": ["2016-09-16"], "RA": [1.0], "DEC": [1.0]})

    def fake_build_tables(dest, instrument, table_of_files, **kw):
        mode = build.resolve_mode_name(instrument, kw.get("polarimetry", False),
                                       kw.get("sparse_aperture_masking", False))
        return prov.TableProvenance(instrument=instrument, mode=mode, eso_coverage_end="2016-09-16")

    with patch.object(build.file_table, "make_file_table", return_value=file_tbl), \
         patch.object(build, "build_tables", side_effect=fake_build_tables):
        build.update_database(tmp_path, "irdis", start_date="2016-09-15", end_date="2016-09-16",
                              suffix="_test")

    records = prov.read_provenance(tmp_path)
    assert records["irdis"].eso_coverage_end == "2026-06-30"
    assert records["irdis_test"].eso_coverage_end == "2016-09-16"


def test_suffixed_update_does_not_resume_from_the_real_tables(tmp_path):
    prov.write_provenance(tmp_path, {"ifs": prov.TableProvenance(
        instrument="ifs", mode="ifs", eso_coverage_end="2026-06-30")})
    assert build.derive_start_date(tmp_path, "ifs", overlap_days=7, suffix="_test") is None
