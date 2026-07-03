"""Unit tests for SphereDatabase.observations_from_name_SIMBAD.

These build a minimal SphereDatabase via ``__new__`` to bypass the heavy
(network-dependent) ``__init__`` and exercise only the name-resolution and
row-selection logic, including the str-or-list handling and deduplication.
"""
from unittest.mock import patch

import numpy as np
from astropy.table import Table

from spherical.database.sphere_database import SphereDatabase


def _make_lightweight_db():
    tobs = Table({"MAIN_ID": ["* bet Pic", "* bet Pic", "* alf Cen", "HD 1", "HD 1"]})
    db = SphereDatabase.__new__(SphereDatabase)  # bypass heavy __init__
    db.table_of_observations = tobs
    db._not_usable_observations_mask = np.zeros(len(tobs), bool)
    # _normalize_name lowercases and strips spaces/underscores
    db._normalized_id_lookup = {"betapic": [0, 1], "alfcen": [2], "hd1": [3, 4]}
    return db


def test_single_string():
    db = _make_lightweight_db()
    assert len(db.observations_from_name_SIMBAD("HD 1")) == 2


def test_list_combines_targets():
    db = _make_lightweight_db()
    result = db.observations_from_name_SIMBAD(["beta pic", "HD 1"])
    assert len(result) == 4  # 2 bet Pic + 2 HD 1
    assert set(np.asarray(result["MAIN_ID"]).tolist()) == {"* bet Pic", "HD 1"}


def test_list_deduplicates_repeated_target():
    db = _make_lightweight_db()
    # Same target reached via a duplicated name is included only once.
    assert len(db.observations_from_name_SIMBAD(["HD 1", "HD 1"])) == 2


def test_partial_resolution_keeps_resolved():
    db = _make_lightweight_db()
    # One resolvable, one not: SIMBAD is only consulted for the unresolved name.
    with patch("spherical.database.sphere_database.Simbad.query_object", return_value=None):
        result = db.observations_from_name_SIMBAD(["beta pic", "does-not-exist"])
    assert len(result) == 2
    assert set(np.asarray(result["MAIN_ID"]).tolist()) == {"* bet Pic"}


def test_all_unresolved_returns_none():
    db = _make_lightweight_db()
    with patch("spherical.database.sphere_database.Simbad.query_object", return_value=None):
        assert db.observations_from_name_SIMBAD(["does-not-exist"]) is None
