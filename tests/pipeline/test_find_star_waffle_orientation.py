"""star_centers_from_waffle_img_cube: orientations missing from the header.

Data from 2014 lacks ``HIERARCH ESO OCS WAFFLE ORIENT``, so the frames table holds
``'N/A'``. The PZ Tel CENTER frame of 2014-08-07 shows the spots on the diagonals,
so a missing orientation falls back to ``'x'`` (#247).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from spherical.pipeline.steps.find_star import (
    WAFFLE_SPOT_FREQUENCY as FREQ,
)
from spherical.pipeline.steps.find_star import (
    resolve_waffle_orientation,
    star_centers_from_waffle_img_cube,
)

PIXEL = 12.25          # IRDIS mas/px
WAVE = 2110.0          # nm
LOD = WAVE * 1e-9 / 7.99 * 180 / np.pi * 3600 * 1000 / PIXEL
SIGMA = LOD / 2.355
R_SPOT = FREQ * LOD
N = 240
TRUE_CENTER = (120.3, 119.7)


def _x_frame():
    """Four unit-peak waffle spots on the diagonals, as for orientation ``'x'``."""
    yy, xx = np.mgrid[:N, :N]
    img = np.zeros((N, N))
    for s in range(4):
        angle = np.pi / 4 + np.pi / 2 * s
        cx = TRUE_CENTER[0] + R_SPOT * np.cos(angle)
        cy = TRUE_CENTER[1] + R_SPOT * np.sin(angle)
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * SIGMA ** 2))
    return (img + np.random.default_rng(0).normal(0, 0.02, img.shape))[None, :, :]


def _fit(waffle_orientation, logger=None):
    _, _, img_centers, _ = star_centers_from_waffle_img_cube(
        _x_frame(), wave=np.array([WAVE]), waffle_orientation=waffle_orientation,
        center_guess=np.array([[120, 120]]), pixel=PIXEL, orientation_offset=0.,
        save_plot=False, save_path=None, logger=logger)
    return img_centers[0]


@pytest.mark.parametrize("missing", ["N/A", "n/a", None, float("nan"), np.nan, pd.NA, ""])
def test_missing_orientation_resolves_to_x(missing):
    assert resolve_waffle_orientation(missing) == ("x", True)


@pytest.mark.parametrize("value", ["+", "x", np.str_("x")])
def test_known_orientation_is_kept(value):
    assert resolve_waffle_orientation(value) == (str(value), False)


def test_unknown_orientation_raises_a_clear_error():
    with pytest.raises(ValueError, match="waffle orientation"):
        resolve_waffle_orientation("y")


def test_fit_with_missing_orientation_uses_x_and_warns(caplog):
    logger = logging.getLogger("test_waffle_orientation")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        center = _fit("N/A", logger=logger)
    assert np.hypot(*(center - np.array(TRUE_CENTER))) < 0.1
    assert "assuming 'x'" in caplog.text
