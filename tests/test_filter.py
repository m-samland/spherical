"""Unit tests for observation-table filtering, presentation, and usable mask.

These construct small astropy Tables directly, or a lightweight SphereDatabase
via ``__new__`` (bypassing the heavy, network-dependent ``__init__``), so no
real data or network access is needed.
"""
from astropy.table import Table

from spherical.database.sphere_database import (
    USABLE_MIN_EXPTIME_SCI,
    SUMMARY_COLUMNS,
    usable_mask,
    view,
)
import pytest


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
