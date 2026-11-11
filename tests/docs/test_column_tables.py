"""The Database columns page lists every published column once, grouped and in table order."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "_ext"))

import column_tables  # noqa: E402

from spherical.database.column_docs import GROUPS  # noqa: E402

MANIFESTS = Path(__file__).resolve().parents[2] / "docs" / "_data" / "columns"


def first_cells(rst: str) -> list[str]:
    return re.findall(r"^\s*\* - ``(.+)``$", rst, flags=re.MULTILINE)


def test_render_lists_every_manifest_column_once():
    manifests = column_tables.read_manifests(MANIFESTS, "observations")
    names = first_cells(column_tables.render_column_rst("observations", manifests))
    assert sorted(names) == sorted(manifests["ifs"])


def test_groups_follow_GROUPS_order():
    rst = column_tables.render_column_rst("observations", column_tables.read_manifests(MANIFESTS, "observations"))
    rubrics = re.findall(r"^\.\. rubric:: (.+)$", rst, flags=re.MULTILINE)
    assert rubrics == [group for group in GROUPS if group in rubrics]
    assert "Gaia enrichment" in rubrics and "File" in rubrics


def test_columns_keep_manifest_order_within_group():
    manifests = {"ifs": ["FLUX_FLAG", "HCI_READY", "CORO_FLAG"], "irdis": ["FLUX_FLAG", "HCI_READY", "CORO_FLAG"]}
    assert first_cells(column_tables.render_column_rst("observations", manifests)) == ["FLUX_FLAG", "HCI_READY", "CORO_FLAG"]


def test_render_marks_mode_only_columns():
    manifests = {"ifs": ["MAIN_ID", "HCI_READY"], "irdis": ["MAIN_ID", "ROTATION"]}
    rst = column_tables.render_column_rst("observations", manifests)
    assert first_cells(rst) == ["MAIN_ID", "HCI_READY", "ROTATION"]
    assert "IFS only. True when" in rst
    assert "IRDIS only. Field rotation" in rst
    assert "only. Main SIMBAD" not in rst


def test_inline_code_becomes_rst_literal():
    rst = column_tables.render_column_rst("targets", {"ifs": ["OBJ_HEADER"], "irdis": ["OBJ_HEADER"]})
    assert "``OBJECT`` keyword" in rst


def test_unknown_table_type_raises():
    with pytest.raises(ValueError, match="observations"):
        column_tables.render_column_rst("obs", {"ifs": [], "irdis": []})
