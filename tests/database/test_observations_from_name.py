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


class TestBuildNormalizedIdLookup:
    """Unit tests for SphereDatabase._build_normalized_id_lookup.

    The lookup is normally injected by the tests above, so the builder itself
    had no coverage. It indexes ID cells that may hold several pipe-joined
    designations for one object (SIMBAD's ``all_ids`` separator, preserved by
    ``target_table.extract_ids``).
    """

    @staticmethod
    def _build(tobs):
        db = SphereDatabase.__new__(SphereDatabase)  # bypass heavy __init__
        db.table_of_observations = tobs
        return db._build_normalized_id_lookup()

    def test_pipe_joined_designations_are_indexed_separately(self):
        # "HD 135344|HD 135344A" must make both names resolve, and the joined
        # string must not itself become a key.
        lookup = self._build(Table({
            "MAIN_ID": ["HD 135344", "HD 135344B"],
            "ID_HD": ["HD 135344|HD 135344A", "HD 135344B"],
        }))
        assert lookup["hd135344"] == [0]
        assert lookup["hd135344a"] == [0]
        assert lookup["hd135344b"] == [1]
        assert "hd135344|hd135344a" not in lookup

    def test_designation_reachable_when_main_id_is_a_different_catalogue(self):
        # The case that sent real lookups to the network: MAIN_ID is a variable-star
        # name, and the HD number only exists inside a pipe-joined cell.
        lookup = self._build(Table({
            "MAIN_ID": ["V* DX Cha"],
            "ID_HD": ["HD 104237A|HD 104237"],
        }))
        assert lookup["hd104237"] == [0]
        assert lookup["hd104237a"] == [0]
        assert lookup["v*dxcha"] == [0]

    def test_empty_id_cells_are_skipped(self):
        # Masked IDs stringify to "". Indexing them made every row without an HD
        # number answer to the empty name.
        lookup = self._build(Table({
            "MAIN_ID": ["* bet Pic", "* alf Cen"],
            "ID_HD": ["HD 39060", ""],
        }))
        assert "" not in lookup
        assert lookup["hd39060"] == [0]
        assert lookup["*alfcen"] == [1]

    def test_row_is_not_repeated_for_duplicate_designations_in_one_cell(self):
        lookup = self._build(Table({
            "MAIN_ID": ["HD 167665"],
            "ID_HD": ["HD 167665B|HD 167665B|HD 167665"],
        }))
        assert lookup["hd167665b"] == [0]
        assert lookup["hd167665"] == [0]

    def test_same_designation_across_rows_collects_all_rows(self):
        lookup = self._build(Table({
            "MAIN_ID": ["HD 1", "HD 1", "HD 2"],
            "ID_HD": ["HD 1", "HD 1|HD 1A", "HD 2"],
        }))
        assert lookup["hd1"] == [0, 1]
        assert lookup["hd1a"] == [1]

    def test_missing_id_columns_are_tolerated(self):
        lookup = self._build(Table({"MAIN_ID": ["HD 1"]}))
        assert lookup == {"hd1": [0]}

    def test_normalization_matches_the_query_side(self):
        # Builder and _try_local_lookup must normalize identically or names silently
        # stop resolving; both go through _normalize_name.
        lookup = self._build(Table({"MAIN_ID": ["  bet_Pic  "]}))
        assert lookup == {"betpic": [0]}
