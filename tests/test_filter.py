"""Unit tests for observation-table filtering, presentation, and usable mask.

These construct small astropy Tables directly, or a lightweight SphereDatabase
via ``__new__`` (bypassing the heavy, network-dependent ``__init__``), so no
real data or network access is needed.
"""
import numpy as np
import numpy.ma as ma
import pytest
from astropy.table import Table

from spherical.database.sphere_database import (
    SUMMARY_COLUMNS,
    USABLE_MIN_EXPTIME_SCI,
    SphereDatabase,
    usable_mask,
    view,
)


def _usable_table():
    return Table(
        {
            "HCI_READY": [True, True, True, False],
            "DEROTATOR_MODE": ["PUPIL", "FIELD", "PUPIL", "PUPIL"],
            "TOTAL_EXPTIME_SCI": [100.0, 100.0, 1.0, 100.0],
        }
    )


def test_usable_mask_requires_all_three_conditions():
    t = _usable_table()
    result = usable_mask(t)
    # row0 usable; row1 FIELD; row2 too short; row3 not HCI_READY
    assert list(result) == [True, False, False, False]


def test_usable_min_exptime_default():
    assert USABLE_MIN_EXPTIME_SCI == 5.0


def _obs_table():
    return Table(
        {
            "MAIN_ID": ["a", "b"],
            "NIGHT_START": ["2020-01-01", "2020-01-02"],
            "FILTER": ["OBS_YJ", "OBS_H"],
            "HCI_READY": [True, False],
            "OBS_PROG_ID": ["095.C-0298", "111.24QL.001"],
            "MEAN_FWHM": [0.8, 1.5],
        }
    )


def test_view_none_returns_all_columns():
    t = _obs_table()
    result = view(t, None)
    assert result.colnames == t.colnames
    assert result is not t  # a copy


def test_view_selects_named_subset():
    t = _obs_table()
    result = view(t, "OBSLOG")
    # every returned column is in the OBSLOG set and present in the table
    assert set(result.colnames).issubset(set(SUMMARY_COLUMNS["OBSLOG"]))
    assert "MAIN_ID" in result.colnames


def test_view_silently_drops_absent_columns():
    t = _obs_table()  # lacks most MEDIUM columns
    result = view(t, "MEDIUM")
    assert "MAIN_ID" in result.colnames
    assert "GAIA_TEFF" not in result.colnames  # absent in this table, dropped


def test_view_unknown_summary_raises():
    with pytest.raises(KeyError):
        view(_obs_table(), "NOPE")


def _filter_table():
    return Table(
        {
            "MAIN_ID": ["a", "b", "c", "d"],
            "MEAN_FWHM": [0.8, 1.5, 0.9, 0.7],
            "FLUX_J": [6.0, 7.0, 9.0, 5.0],
            "DISTANCE": [10.0, 50.0, 20.0, 30.0],
            "MOCA_ASSOCIATION_NAME": ma.MaskedArray(
                [b"bpmg", b"other", b"bpmg", b"zzz"], mask=[False, False, False, True]
            ),
            "OBS_PROG_ID": [b"095.C-0298", b"111.24QL.001", b"110.240D.001", b"095.C-0298"],
            "MOCA_AGE_MYR": ma.MaskedArray([20.0, 30.0, 99.0, 10.0], mask=[False, True, False, False]),
        }
    )


def _lightweight_db(table):
    db = SphereDatabase.__new__(SphereDatabase)  # bypass heavy __init__
    db.table_of_observations = table
    return db


def test_filter_equality_criterion():
    db = _lightweight_db(_filter_table())
    result = db.filter(MOCA_ASSOCIATION_NAME="bpmg")
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"a", "c"}


def test_filter_membership_list():
    db = _lightweight_db(_filter_table())
    result = db.filter(OBS_PROG_ID=["110.240D.001", "111.24QL.001"])
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"b", "c"}


def test_filter_not_in_exclusion():
    db = _lightweight_db(_filter_table())
    result = db.filter(OBS_PROG_ID=("not in", ["095.C-0298"]))
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"b", "c"}


def test_filter_mask_and_criteria_compose():
    db = _lightweight_db(_filter_table())
    result = db.filter(lambda t: t["MEAN_FWHM"] < 1.0, MOCA_ASSOCIATION_NAME="bpmg")
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"a", "c"}


def test_filter_cross_column_mask():
    db = _lightweight_db(_filter_table())
    # absolute J mag = FLUX_J - 5*log10(DISTANCE/10); keep intrinsically bright.
    # a: 6-0=6.0, b: 7-3.49=3.51, c: 9-1.51=7.49, d: 5-2.39=2.61  -> < 5 keeps {b, d}
    result = db.filter(lambda t: t["FLUX_J"] - 5 * np.log10(t["DISTANCE"] / 10) < 5)
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"b", "d"}


def test_filter_excludes_missing_for_criterion():
    db = _lightweight_db(_filter_table())
    # row 'd' has MOCA_ASSOCIATION_NAME masked -> excluded even from "not equal to bpmg"
    result = db.filter(MOCA_ASSOCIATION_NAME=("not in", ["bpmg"]))
    assert "d" not in set(np.asarray(result["MAIN_ID"]).astype(str))
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"b"}


def test_filter_excludes_missing_for_numeric_criterion():
    db = _lightweight_db(_filter_table())
    # row 'b' has MOCA_AGE_MYR masked -> excluded
    result = db.filter(MOCA_AGE_MYR=("not in", [999.0]))
    assert "b" not in set(np.asarray(result["MAIN_ID"]).astype(str))


def test_filter_unknown_column_suggests():
    db = _lightweight_db(_filter_table())
    with pytest.raises(KeyError, match="MEAN_FWHM"):
        db.filter(MEAN_FWH=0.8)


def test_filter_mask_wrong_length_raises():
    db = _lightweight_db(_filter_table())
    with pytest.raises(ValueError):
        db.filter(np.array([True, False]))  # len 2 != 4


def test_columns_property():
    db = _lightweight_db(_filter_table())
    assert db.columns == db.table_of_observations.colnames


def _named_db():
    table = Table(
        {
            "MAIN_ID": ["* bet Pic", "* bet Pic", "* alf Cen", "HD 1"],
            "MEAN_FWHM": [0.8, 1.5, 0.9, 0.7],
        }
    )
    db = SphereDatabase.__new__(SphereDatabase)
    db.table_of_observations = table
    db._normalized_id_lookup = {"betapic": [0, 1], "alfcen": [2], "hd1": [3]}
    return db


def test_filter_target_list_restricts():
    db = _named_db()
    result = db.filter(target_list=["beta pic"])
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"* bet Pic"}
    assert len(result) == 2


def test_filter_target_list_with_criteria():
    db = _named_db()
    result = db.filter(target_list=["beta pic"], MEAN_FWHM=0.8)
    assert len(result) == 1


def test_filter_target_list_unresolved_returns_empty():
    from unittest.mock import patch

    db = _named_db()
    with patch("spherical.database.sphere_database.Simbad.query_object", return_value=None):
        result = db.filter(target_list=["does-not-exist"])
    assert len(result) == 0


def test_rotation_missing_is_nan_and_excluded_by_filter():
    # A missing ROTATION (NaN) round-trips to a masked column; a ROTATION
    # criterion must exclude it rather than treat -10000 as a real value.
    t = Table(
        {
            "MAIN_ID": ["a", "b"],
            "ROTATION": ma.MaskedArray([25.0, np.nan], mask=[False, True]),
        }
    )
    db = _lightweight_db(t)
    result = db.filter(ROTATION=("not in", [-10000.0]))
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"a"}


def test_compute_rotation_returns_nan_for_polarimetry():
    from spherical.database.observation_table import compute_derotation_info

    result = compute_derotation_info(observation_files=Table(), polarimetry=True)
    assert np.isnan(result["ROTATION"])
    assert result["DEROTATOR_FLAG"] is True
