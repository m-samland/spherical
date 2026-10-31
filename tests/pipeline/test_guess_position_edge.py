"""guess_position_psf: footprint-based edge exclusion and mask wiring."""
from __future__ import annotations

import numpy as np
import pytest

from spherical.pipeline.steps.find_star import guess_position_psf


def _ifs_like(n=262, nwave=39, star_yx=(188, 170)):
    """charis v2.1.0 geometry: NaN border, field rows 21..240 / cols 8..255."""
    rng = np.random.default_rng(0)
    cube = np.full((nwave, n, n), np.nan)
    yy, xx = np.mgrid[:n, :n]
    infield = (yy >= 21) & (yy <= 240) & (xx >= 8) & (xx <= 255)
    cube[:, infield] = rng.normal(100.0, 1.0, size=infield.sum())
    sy, sx = star_yx
    cube += (5000.0 * np.exp(-((yy - sy) ** 2 + (xx - sx) ** 2) / 8.0))[None] * infield[None]
    cube[:, ~infield] = np.nan
    return cube, star_yx


class TestGuessPositionEdgeExclusion:
    def test_finds_the_star_on_a_clean_ifs_frame(self):
        cube, truth = _ifs_like()
        assert guess_position_psf(cube, exclude_edge_pixels=30) == truth

    def test_hot_pixel_in_the_fov_transition_is_excluded(self):
        # Field starts at col 8; a hot pixel at col 12 is in-field but sits in
        # the noisy transition the margin exists to avoid.
        cube, truth = _ifs_like()
        cube[:, 130, 12] = 1e6
        assert guess_position_psf(cube, exclude_edge_pixels=30) == truth

    def test_a_fully_finite_frame_keeps_every_pixel(self):
        # Cropped IRDIS: no real border pixels, so no margin is applied and the
        # brightest pixel wins wherever it is.
        rng = np.random.default_rng(1)
        cube = rng.normal(100.0, 1.0, size=(5, 200, 200))
        cube[:, 3, 150] = 1e6
        assert guess_position_psf(cube, exclude_edge_pixels=30) == (3, 150)

    def test_bad_pixel_mask_suppresses_a_hot_pixel_inside_the_field(self):
        cube, truth = _ifs_like()
        cube[:, 130, 130] = 1e6
        bpm = np.zeros(cube.shape, dtype=bool)
        bpm[:, 130, 130] = True
        assert guess_position_psf(
            cube, exclude_edge_pixels=30, bad_pixel_mask=bpm) == truth

    def test_bad_pixel_bad_in_a_minority_of_channels_is_not_masked(self):
        cube, truth = _ifs_like()
        bpm = np.zeros(cube.shape, dtype=bool)
        bpm[:3, truth[0], truth[1]] = True  # 3 of 39 channels
        assert guess_position_psf(
            cube, exclude_edge_pixels=30, bad_pixel_mask=bpm) == truth

    def test_coronagraph_mask_is_applied_at_the_given_centre(self):
        cube, truth = _ifs_like()
        cube[:, 100, 100] = 1e6
        assert guess_position_psf(
            cube, exclude_edge_pixels=30,
            coronagraph_center_xy=(100.0, 100.0), coronagraph_mask_radius=30,
        ) == truth

    def test_no_coronagraph_mask_without_a_centre(self):
        cube, _ = _ifs_like()
        cube[:, 100, 100] = 1e6
        assert guess_position_psf(
            cube, exclude_edge_pixels=30, coronagraph_center_xy=None) == (100, 100)

    def test_raises_when_every_pixel_is_masked_out(self):
        cube = np.full((5, 60, 60), np.nan)
        with pytest.raises(ValueError, match="no valid pixels"):
            guess_position_psf(cube, exclude_edge_pixels=5)
