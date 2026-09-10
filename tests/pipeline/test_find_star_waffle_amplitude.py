"""star_centers_from_waffle_img_cube: the fitted amplitude must measure the spot.

The ``mask_deviating`` refit used to drop the ``Const2D`` component fitted in the
first pass, so on a coronagraphic halo the refit Gaussian absorbed the pedestal and
``spot_fit_amplitudes.fits`` reported the spot *plus* the halo (empirically about
``1 + 2 x halo``).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from spherical.pipeline.steps.find_star import star_centers_from_waffle_img_cube

PIXEL = 12.25          # IRDIS mas/px
WAVE = 2110.0          # nm
FREQ = 10 * np.sqrt(2) * 0.97
LOD = WAVE * 1e-9 / 7.99 * 180 / np.pi * 3600 * 1000 / PIXEL
SIGMA = LOD / 2.355
R_SPOT = FREQ * LOD
N = 240
TRUE_CENTER = (120.3, 119.7)   # deliberately non-integer


def _frame(halo_to_spot, halo_scale_px=60.0, noise=0.02, seed=0):
    """Four unit-peak waffle spots on an exponential stellar halo."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:N, :N]
    img = np.zeros((N, N))
    for s in range(4):
        cx = TRUE_CENTER[0] + R_SPOT * np.cos(np.pi / 2 * s)
        cy = TRUE_CENTER[1] + R_SPOT * np.sin(np.pi / 2 * s)
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * SIGMA ** 2))
    r = np.hypot(xx - TRUE_CENTER[0], yy - TRUE_CENTER[1])
    # normalised so the halo at the spot radius is halo_to_spot x the spot peak
    img += np.exp(-r / halo_scale_px) * halo_to_spot / np.exp(-R_SPOT / halo_scale_px)
    return (img + rng.normal(0, noise, img.shape))[None, :, :]


def _fit(img):
    spot_centers, _, img_centers, amplitudes = star_centers_from_waffle_img_cube(
        img, wave=np.array([WAVE]), waffle_orientation='+',
        center_guess=np.array([[120, 120]]), pixel=PIXEL, orientation_offset=0.,
        fit_background=True, fit_symmetric_gaussian=True, mask_deviating=True,
        deviation_threshold=0.8, high_pass=False, save_plot=False, save_path=None)
    return spot_centers[0], img_centers[0], amplitudes[0]


class TestWaffleSpotAmplitudeOnHalo:
    @pytest.mark.parametrize("halo_to_spot", [0.0, 0.3, 1.0, 3.0])
    def test_amplitude_measures_the_spot_not_the_halo(self, halo_to_spot):
        _, _, amplitudes = _fit(_frame(halo_to_spot))
        measured = float(np.nanmean(amplitudes))
        assert measured == pytest.approx(1.0, abs=0.05), (
            f"halo/spot={halo_to_spot}: amplitude {measured:.3f} should track the "
            f"unit spot peak, not the pedestal under it"
        )

    def test_star_center_is_recovered_on_a_bright_halo(self):
        _, img_center, _ = _fit(_frame(3.0))
        assert np.hypot(*(img_center - np.array(TRUE_CENTER))) < 0.1
