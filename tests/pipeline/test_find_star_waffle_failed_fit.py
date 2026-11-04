"""star_centers_from_waffle_img_cube: a spot fit that leaves its search box is a failed fit (#286).

The ``mask_deviating`` refit has no bounds. Where a spot is missing and a single hot pixel sits near the
edge of its 16 px search box, the fit runs onto or past that pixel, and its position and amplitude used
to be stored as a measurement.
"""
from __future__ import annotations

import numpy as np
import pytest

from spherical.pipeline.steps.find_star import (
    WAFFLE_SPOT_FREQUENCY as FREQ,
)
from spherical.pipeline.steps.find_star import (
    star_centers_from_waffle_img_cube,
    waffle_spot_box_centers,
)

PIXEL = 12.25          # IRDIS mas/px
WAVE = 2110.0          # nm
LOD = WAVE * 1e-9 / 7.99 * 180 / np.pi * 3600 * 1000 / PIXEL
SIGMA = LOD / 2.355
N = 240
TRUE_CENTER = (120.3, 119.7)
GUESS = np.array([[120, 120]])
BOX = 8


def _frame(missing_spot=None, hot_pixel=None, seed=0):
    """Four unit-peak waffle spots with noise, optionally one spot removed and one hot pixel (x, y, value)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:N, :N]
    img = rng.normal(0, 0.02, (N, N))
    for s in range(4):
        if s == missing_spot:
            continue
        cx = TRUE_CENTER[0] + FREQ * LOD * np.cos(np.pi / 2 * s)
        cy = TRUE_CENTER[1] + FREQ * LOD * np.sin(np.pi / 2 * s)
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * SIGMA ** 2))
    if hot_pixel is not None:
        x, y, value = hot_pixel
        img[y, x] = value
    return img[None, :, :]


def _fit(img):
    spot_centers, _, img_centers, amplitudes = star_centers_from_waffle_img_cube(
        img, wave=np.array([WAVE]), waffle_orientation='+', center_guess=GUESS, pixel=PIXEL,
        orientation_offset=0., fit_background=True, fit_symmetric_gaussian=False, mask_deviating=True,
        deviation_threshold=0.8, high_pass=False, save_plot=False, save_path=None)
    return spot_centers[0], img_centers[0], amplitudes[0]


@pytest.mark.parametrize(
    ("offset", "value", "seed"),
    [((BOX - 1, BOX - 1), value, seed) for value in (3.0, 30.0) for seed in (0, 1, 2)]
    # Far runaways, like the off-frame fits of #278: the refit ends near y = 74 and (-4, -8) in box coordinates.
    + [((0, BOX - 1), 1.0, 0), ((-BOX, -BOX), 0.2, 0)],
)
def test_spot_fit_leaving_its_box_is_stored_as_nan(offset, value, seed):
    box_center = waffle_spot_box_centers(GUESS[0], LOD, 0.)[0].astype(int)
    hot = (box_center[0] + offset[0], box_center[1] + offset[1], value)
    spot_centers, img_center, amplitudes = _fit(_frame(missing_spot=0, hot_pixel=hot, seed=seed))
    reference_centers, _, reference_amplitudes = _fit(_frame(seed=seed))

    assert np.all(np.isnan(spot_centers[0]))
    assert np.isnan(amplitudes[0])
    np.testing.assert_allclose(spot_centers[1:], reference_centers[1:], atol=1e-6)
    np.testing.assert_allclose(amplitudes[1:], reference_amplitudes[1:], rtol=1e-6)
    assert np.all(np.isnan(img_center))


def test_good_spots_are_kept():
    spot_centers, img_center, amplitudes = _fit(_frame())
    assert np.all(np.isfinite(spot_centers))
    assert np.all(np.isfinite(amplitudes))
    assert np.hypot(*(img_center - np.array(TRUE_CENTER))) < 0.1
