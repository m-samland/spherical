"""SphereDatabase access to the other-observations table (#224)."""

import inspect

import pytest
from astropy.table import Table

from spherical.database.sphere_database import SphereDatabase


def _db(other, instrument="irdis"):
    db = SphereDatabase.__new__(SphereDatabase)  # bypass the file-table set-up in __init__
    db.instrument = instrument
    db.table_of_other_observations = other
    return db


OTHER = Table({"CATEGORY": ["solar_system", "unmatched", "solar_system"], "OBJECT": ["Ceres", "S CrA", "Io"],
               "INSTRUMENT": ["irdis", "irdis", "ifs"]})


def test_init_accepts_the_other_observations_table():
    assert "table_of_other_observations" in inspect.signature(SphereDatabase.__init__).parameters


def test_other_observations_of_this_instrument():
    assert _db(OTHER).other_observations()["OBJECT"].tolist() == ["Ceres", "S CrA"]


def test_other_observations_by_category():
    assert _db(OTHER).other_observations("unmatched")["OBJECT"].tolist() == ["S CrA"]
    assert _db(OTHER, "ifs").other_observations("solar_system")["OBJECT"].tolist() == ["Io"]


def test_unknown_category_raises():
    with pytest.raises(ValueError, match="category"):
        _db(OTHER).other_observations("galaxy")


def test_missing_table_raises():
    with pytest.raises(ValueError, match="table_of_other_observations"):
        _db(None).other_observations()
