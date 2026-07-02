import pytest

from spherical.database.database_utils import resolve_mode_name


def test_ifs_standard():
    assert resolve_mode_name("ifs") == "ifs"


def test_ifs_sam():
    assert resolve_mode_name("ifs", sparse_aperture_masking=True) == "ifs_sam"


def test_irdis_standard():
    assert resolve_mode_name("irdis") == "irdis"


def test_irdis_polarimetry():
    assert resolve_mode_name("irdis", polarimetry=True) == "irdis_polarimetry"


def test_irdis_sam():
    assert resolve_mode_name("irdis", sparse_aperture_masking=True) == "irdis_sam"


def test_sam_takes_precedence_over_polarimetry():
    assert resolve_mode_name("irdis", polarimetry=True, sparse_aperture_masking=True) == "irdis_sam"


def test_case_insensitive():
    assert resolve_mode_name("IFS") == "ifs"


def test_invalid_instrument():
    with pytest.raises(ValueError):
        resolve_mode_name("zimpol")


def test_no_bool_strings_in_any_mode():
    for inst, pol, sam in [
        ("ifs", False, False), ("ifs", False, True),
        ("irdis", False, False), ("irdis", True, False), ("irdis", False, True),
    ]:
        name = resolve_mode_name(inst, pol, sam)
        assert "True" not in name and "False" not in name
