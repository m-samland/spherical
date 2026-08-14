"""Selection of targets whose epochs can discriminate companions from background stars.

A companion shares its host's proper motion; a background star does not. That
test only has power when the host moved far enough between the first and last
epoch for the two hypotheses to predict measurably different positions.
:func:`select_multi_epoch_targets` keeps the targets where they do.

Kept free of non-stdlib imports beyond numpy/astropy so the base install (no
``pipeline`` extra) can use it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np
from astropy.table import Table
from astropy.time import Time

logger = logging.getLogger(__name__)

__all__ = ["read_host_list", "select_multi_epoch_targets"]

PathLike = Union[str, os.PathLike]

SPAN_COLUMN = "_MULTI_EPOCH_SPAN_YR"
BG_MOTION_COLUMN = "_MULTI_EPOCH_BG_PX"
N_EPOCHS_COLUMN = "_MULTI_EPOCH_N"


def _decoded(column) -> np.ndarray:
    """Return ``column`` as unicode strings, decoding a byte column if needed."""
    values = np.asarray(column)
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    return values.astype(str)


def _invalid(column) -> np.ndarray:
    """Boolean mask, ``True`` where the value is masked, NaN, or infinite."""
    values = np.asarray(column, dtype=float)
    invalid = ~np.isfinite(values)
    mask = getattr(column, "mask", None)
    if mask is not None:
        invalid |= np.asarray(mask, dtype=bool)
    return invalid


def select_multi_epoch_targets(
    table: Table,
    *,
    min_bg_motion_px: float = 1.0,
    pixel_scale_mas: float = 12.25,
    min_epochs: int = 2,
) -> Table:
    """Keep observations of hosts whose epochs can separate companions from background stars.

    A target survives when it has at least ``min_epochs`` observations and its
    proper motion displaces a stationary background object by at least
    ``min_bg_motion_px`` pixels over the span between its earliest and latest
    epoch.

    Quality cuts are the caller's job: pass a table already filtered through
    :meth:`spherical.database.sphere_database.SphereDatabase.filter`, because the
    epoch span is measured across the observations that survive those cuts.

    Parameters
    ----------
    table : astropy.table.Table
        Observation table with ``MAIN_ID``, ``NIGHT_START``, ``PMRA`` and
        ``PMDEC`` columns. ``PMRA`` is expected to include the cos(Dec) factor,
        as :mod:`spherical.database.target_table` fills it from SIMBAD.
    min_bg_motion_px : float, optional
        Minimum predicted background-object motion, in pixels.
    pixel_scale_mas : float, optional
        Pixel scale used to convert the predicted motion to pixels [mas/pixel].
    min_epochs : int, optional
        Minimum number of observations a target must have.

    Returns
    -------
    astropy.table.Table
        The surviving rows, in input order, with three added columns constant
        per target: ``_MULTI_EPOCH_SPAN_YR``, ``_MULTI_EPOCH_BG_PX`` and
        ``_MULTI_EPOCH_N``.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If a ``NIGHT_START`` value cannot be parsed as a date.
    """
    required = ("MAIN_ID", "NIGHT_START", "PMRA", "PMDEC")
    missing = [name for name in required if name not in table.colnames]
    if missing:
        raise KeyError(f"Observation table is missing required column(s): {missing}")

    n_rows = len(table)
    keep = np.zeros(n_rows, dtype=bool)
    span_yr = np.zeros(n_rows, dtype=float)
    bg_motion_px = np.zeros(n_rows, dtype=float)
    n_epochs = np.zeros(n_rows, dtype=int)

    if n_rows:
        main_ids = _decoded(table["MAIN_ID"])
        night_start = _decoded(table["NIGHT_START"])
        pmra = np.asarray(table["PMRA"], dtype=float)
        pmdec = np.asarray(table["PMDEC"], dtype=float)
        pm_invalid = _invalid(table["PMRA"]) | _invalid(table["PMDEC"])

        for main_id in dict.fromkeys(main_ids):
            rows = np.flatnonzero(main_ids == main_id)
            if len(rows) < min_epochs:
                continue
            if pm_invalid[rows].any():
                logger.warning(
                    "Multi-epoch selection: dropping %s, proper motion is missing on "
                    "at least one of its %d observations.",
                    main_id,
                    len(rows),
                )
                continue

            times = Time(list(night_start[rows]), format="iso")
            span = float(times.max().jyear - times.min().jyear)
            if span <= 0.0:
                continue

            pm_total = float(np.hypot(pmra[rows[0]], pmdec[rows[0]]))
            motion_px = pm_total * span / pixel_scale_mas
            if motion_px < min_bg_motion_px:
                continue

            keep[rows] = True
            span_yr[rows] = span
            bg_motion_px[rows] = motion_px
            n_epochs[rows] = len(rows)

    selected = table[keep]
    selected[SPAN_COLUMN] = span_yr[keep]
    selected[BG_MOTION_COLUMN] = bg_motion_px[keep]
    selected[N_EPOCHS_COLUMN] = n_epochs[keep]

    logger.info(
        "Multi-epoch selection: %d/%d observations on %d targets pass "
        "(>= %d epochs, >= %.2f px background motion).",
        len(selected),
        n_rows,
        len(set(_decoded(selected["MAIN_ID"]))) if len(selected) else 0,
        min_epochs,
        min_bg_motion_px,
    )
    return selected


def read_host_list(path: PathLike) -> List[str]:
    """Read a newline-separated list of target names.

    Blank lines are skipped, and everything from a ``#`` to the end of a line is
    treated as a comment. Intended for feeding
    :meth:`spherical.database.sphere_database.SphereDatabase.filter`'s
    ``exclude_targets``, which resolves the names through SIMBAD and therefore
    needs network; reading the file does not.

    Parameters
    ----------
    path : str or os.PathLike
        File to read.

    Returns
    -------
    list of str
        Target names, in file order.
    """
    names = []
    for line in Path(path).read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.append(name)
    return names
