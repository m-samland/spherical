"""Shared helpers for the read-only monitoring scripts.

Deliberately free of `spherical` imports so `crash_reports` / `reduction_status`
keep working in a base install without the `pipeline` extra.
"""
from __future__ import annotations

from typing import Callable, Sequence

__all__ = ["instrument_from_band", "format_table"]


def instrument_from_band(band: str) -> str:
    """Return the instrument an observation band belongs to.

    IFS observations are the only ones whose band is named ``OBS_*`` (``OBS_YJ``
    or ``OBS_H``); every IRDIS mode carries a filter-derived prefix (``DB_``,
    ``BB_``, ``NB_``, ``DP_``).
    """
    return "IFS" if band.upper().startswith("OBS_") else "IRDIS"


def format_table(
    rows: Sequence[dict],
    columns: Sequence[tuple[str, Callable[[dict], str]]],
    trailing: tuple[str, Callable[[dict], str]] | None = None,
) -> str:
    """Render *rows* as a fixed-width table.

    Column widths follow the content rather than a fixed padding, so long
    target names and 6-character bands cannot push the remaining columns out of
    alignment with the header. *trailing*, if given, is a final free-width
    column that is never padded.
    """
    cells = [[fn(row) for _, fn in columns] for row in rows]
    widths = [
        max([len(header)] + [len(row[i]) for row in cells])
        for i, (header, _) in enumerate(columns)
    ]

    def _line(values: Sequence[str], last: str | None) -> str:
        line = "  ".join(v.ljust(w) for v, w in zip(values, widths))
        if last is not None:
            line = f"{line}  {last}"
        return line.rstrip()

    header = _line(
        [header for header, _ in columns],
        trailing[0] if trailing is not None else None,
    )
    lines = [header, "-" * len(header)]
    for row, source in zip(cells, rows):
        lines.append(_line(row, trailing[1](source) if trailing is not None else None))
    return "\n".join(lines)
