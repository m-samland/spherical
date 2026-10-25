"""Render the database column tables from ``column_docs`` and the committed manifests.

Stdlib only (``spherical.database.column_docs`` is too), so ``tests/docs`` imports it
without Sphinx. The manifests decide which columns a table shows and in which order;
``column_docs`` supplies the group, unit and description of each.
"""

from __future__ import annotations

import re
from pathlib import Path

from spherical.database.column_docs import GROUPS, TABLE_TYPES, describe

MODES = {"ifs": "IFS", "irdis": "IRDIS"}


def manifest_path(directory: Path, table_type: str, mode: str) -> Path:
    return directory / f"table_columns_{table_type}_{mode}.txt"


def read_manifests(directory: Path, table_type: str) -> dict[str, list[str]]:
    """Column names per mode, in table order; names may contain spaces, so one per line."""
    return {mode: manifest_path(directory, table_type, mode).read_text().splitlines() for mode in MODES}


def _rst(text: str) -> str:
    """Markdown inline code in a description becomes an RST literal."""
    return re.sub(r"`([^`]+)`", r"``\1``", text)


def render_column_rst(table_type: str, manifests: dict[str, list[str]]) -> str:
    """One rubric and list-table per group, in ``GROUPS`` order; empty groups are left out."""
    if table_type not in TABLE_TYPES:
        raise ValueError(f"unknown table type {table_type!r}, expected one of {', '.join(TABLE_TYPES)}")
    names = list(dict.fromkeys(manifests["ifs"] + manifests["irdis"]))
    by_group: dict[str, list[str]] = {group: [] for group in GROUPS}
    for name in names:
        by_group[describe(table_type, name).group].append(name)

    lines: list[str] = []
    for group, members in by_group.items():
        if not members:
            continue
        lines += [
            f".. rubric:: {group}",
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 25 10 65",
            "   :class: column-table",
            "",
            "   * - Column",
            "     - Unit",
            "     - Description",
        ]
        for name in members:
            column = describe(table_type, name)
            only = [label for mode, label in MODES.items() if name in manifests[mode]]
            prefix = f"{only[0]} only. " if len(only) == 1 else ""
            lines += [f"   * - ``{name}``", f"     - {column.unit}", f"     - {prefix}{_rst(column.description)}"]
        lines.append("")
    return "\n".join(lines)
