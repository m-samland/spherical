"""valid_fov_mask: the shared usable-FOV footprint."""
from __future__ import annotations

import numpy as np

from spherical.pipeline.fov import valid_fov_mask


def _ifs_like(n=262, nwave=39, nframes=3):
    """Border NaN in every plane, field rows 21..240 / cols 8..255 (v2.1.0)."""
    cube = np.full((nwave, nframes, n, n), np.nan)
    yy, xx = np.mgrid[:n, :n]
    infield = (yy >= 21) & (yy <= 240) & (xx >= 8) & (xx <= 255)
    cube[:, :, infield] = 1.0
    return cube, infield


class TestValidFovMask:
    def test_matches_the_finite_data_footprint(self):
        cube, infield = _ifs_like()
        assert np.array_equal(valid_fov_mask(cube, exclude_edge_pixels=0), infield)

    def test_erosion_pulls_the_footprint_in_from_the_invalid_region(self):
        cube, _ = _ifs_like()
        mask = valid_fov_mask(cube, exclude_edge_pixels=30)
        assert mask[130, 130]
        assert not mask[25, 130], "5 px inside the FOV boundary must be excluded"
        assert mask[130, 60]

    def test_array_border_is_not_eroded(self):
        # The margin is about real detector edge effects, not the array bound.
        # A cropped frame has no real border pixels, so nothing is excluded.
        cube = np.ones((5, 3, 200, 200))
        assert valid_fov_mask(cube, exclude_edge_pixels=30).all()

    def test_pixel_finite_in_only_some_planes_stays_in_field(self):
        cube, _ = _ifs_like()
        cube[0, 0, 130, 130] = np.nan
        mask = valid_fov_mask(cube, exclude_edge_pixels=30)
        assert mask[130, 130]
        assert mask[131, 131]

    def test_a_dead_region_invalid_in_every_plane_is_eroded_around(self):
        cube = np.ones((5, 3, 200, 200))
        cube[:, :, 98:103, 98:103] = np.nan
        mask = valid_fov_mask(cube, exclude_edge_pixels=20)
        assert not mask[100, 100]
        assert not mask[110, 100], "10 px from a dead region must be excluded"
        assert mask[140, 140]

    def test_erosion_that_would_empty_the_footprint_is_not_applied(self):
        cube = np.full((5, 3, 40, 40), np.nan)
        cube[:, :, 18:22, 18:22] = 1.0
        assert valid_fov_mask(cube, exclude_edge_pixels=50).any()

    def test_accepts_a_plain_2d_image(self):
        img = np.full((50, 50), np.nan)
        img[10:40, 10:40] = 1.0
        mask = valid_fov_mask(img, exclude_edge_pixels=5)
        assert mask.shape == (50, 50)
        assert mask[25, 25] and not mask[12, 25]

    def test_rejects_data_with_fewer_than_two_dimensions(self):
        try:
            valid_fov_mask(np.ones(10))
        except ValueError as exc:
            assert "at least 2" in str(exc)
        else:
            raise AssertionError("expected ValueError")
