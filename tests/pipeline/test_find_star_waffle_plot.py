"""star_centers_from_waffle_img_cube: the diagnostic PDF must not fail the fit.

The spot insets were drawn with ``vmin=0, vmax=sub.max()``, so a cutout without
positive pixels (a faint spot after background removal) made matplotlib raise in
``pdf.savefig()`` and aborted the whole sequence (#246).
"""
from __future__ import annotations

import numpy as np

from spherical.pipeline.steps.find_star import (
    WAFFLE_SPOT_FREQUENCY as FREQ,
)
from spherical.pipeline.steps.find_star import (
    _spot_inset_limits,
    star_centers_from_waffle_img_cube,
)

PIXEL = 12.25          # IRDIS mas/px
WAVE = 1625.0          # nm
LOD = WAVE * 1e-9 / 7.99 * 180 / np.pi * 3600 * 1000 / PIXEL
SIGMA = LOD / 2.355
R_SPOT = FREQ * LOD
N = 240
TRUE_CENTER = (120.3, 119.7)


def _frame_below_zero(pedestal=-10.0):
    """Four unit-peak waffle spots on a negative pedestal, so every spot cutout is negative."""
    yy, xx = np.mgrid[:N, :N]
    img = np.full((N, N), pedestal)
    for s in range(4):
        cx = TRUE_CENTER[0] + R_SPOT * np.cos(np.pi / 2 * s)
        cy = TRUE_CENTER[1] + R_SPOT * np.sin(np.pi / 2 * s)
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * SIGMA ** 2))
    return (img + np.random.default_rng(0).normal(0, 0.02, img.shape))[None, :, :]


def test_negative_spot_cutouts_still_write_the_diagnostic_pdf(tmp_path):
    plot_path = tmp_path / "CENTER_img_000.pdf"
    _, _, img_centers, _ = star_centers_from_waffle_img_cube(
        _frame_below_zero(), wave=np.array([WAVE]), waffle_orientation='+',
        center_guess=np.array([[120, 120]]), pixel=PIXEL, orientation_offset=0.,
        save_plot=True, save_path=str(plot_path))
    assert plot_path.stat().st_size > 0
    assert np.hypot(*(img_centers[0] - np.array(TRUE_CENTER))) < 0.1


def test_inset_limits_keep_zero_to_peak_for_positive_cutouts():
    assert _spot_inset_limits(np.array([[-1.0, 0.5], [2.0, 0.0]])) == (0.0, 2.0)


def test_inset_limits_are_ordered_for_cutouts_without_positive_pixels():
    vmin, vmax = _spot_inset_limits(np.array([[-9.0, -8.0], [-10.0, -8.5]]))
    assert (vmin, vmax) == (-10.0, -8.0)
    vmin, vmax = _spot_inset_limits(np.full((2, 2), -3.0))
    assert vmin < vmax
