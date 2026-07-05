"""Unit tests for observation-table filtering, presentation, and usable mask.

These construct small astropy Tables directly, or a lightweight SphereDatabase
via ``__new__`` (bypassing the heavy, network-dependent ``__init__``), so no
real data or network access is needed.
"""
from astropy.table import Table

from spherical.database.sphere_database import (
    USABLE_MIN_EXPTIME_SCI,
    usable_mask,
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
