"""Every column of the published tables has a description, and every description a column.

The manifests in docs/_data/columns/ are written from the release tables by
docs/tools/write_column_manifests.py. A table release that adds a column fails here
until spherical.database.column_docs describes it.
"""

import re
from pathlib import Path

import pytest

from spherical.database.column_docs import COLUMNS, GROUPS, OVERRIDES, TABLE_TYPES, describe

MANIFESTS = Path(__file__).parents[2] / "docs" / "_data" / "columns"
MODES = ("ifs", "irdis")
# Same list as tests/docs/test_prose_rules.py; descriptions render on a docs page.
BANNED = [
    "seamless", "seamlessly", "powerful", "effortless", "effortlessly", "comprehensive", "cutting-edge",
    "state-of-the-art", "leverage", "leverages", "unlock", "unlocks", "delve", "simply", "just",
    "easily", "intuitive", "intuitively",
]
_WORD = re.compile(r"(?<![\w-])(" + "|".join(re.escape(word) for word in BANNED) + r")(?![\w-])", re.IGNORECASE)


def manifest(table_type: str, mode: str) -> list[str]:
    # One name per line; file-table names such as "TEL AZ" contain spaces.
    return (MANIFESTS / f"table_columns_{table_type}_{mode}.txt").read_text().splitlines()


MANIFEST_IDS = [(table_type, mode) for table_type in TABLE_TYPES for mode in MODES]


@pytest.mark.parametrize("table_type,mode", MANIFEST_IDS)
def test_manifests_exist(table_type, mode):
    assert manifest(table_type, mode)


@pytest.mark.parametrize("table_type,mode", MANIFEST_IDS)
def test_every_manifest_column_is_documented(table_type, mode):
    missing = [name for name in manifest(table_type, mode) if (table_type, name) not in OVERRIDES and name not in COLUMNS]
    assert missing == [], f"table_columns_{table_type}_{mode}.txt: no description in column_docs for {missing}"


def test_no_documented_column_is_absent_from_every_manifest():
    published = {name for table_type, mode in MANIFEST_IDS for name in manifest(table_type, mode)}
    assert sorted(set(COLUMNS) - published) == []
    stale = [key for key in OVERRIDES if key[1] not in {*manifest(key[0], "ifs"), *manifest(key[0], "irdis")}]
    assert stale == []


@pytest.mark.parametrize("name", sorted(COLUMNS) + [f"{t}:{n}" for t, n in OVERRIDES])
def test_columns_well_formed(name):
    column = OVERRIDES[tuple(name.split(":"))] if ":" in name else COLUMNS[name]
    assert column.group in GROUPS
    assert column.description and column.description.endswith(".")
    assert "—" not in column.description
    assert _WORD.findall(column.description) == []


def test_units_match_table_units():
    rows = [line.split("\t") for line in (MANIFESTS / "table_units.txt").read_text().splitlines()]
    assert {name: COLUMNS[name].unit for name, _ in rows} == dict(rows)


def test_override_wins_for_its_table():
    assert describe("files", "NDIT") is OVERRIDES[("files", "NDIT")]
    assert describe("observations", "NDIT") is COLUMNS["NDIT"]


def test_unknown_column_raises():
    with pytest.raises(KeyError):
        describe("observations", "NOT_A_COLUMN")
