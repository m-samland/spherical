"""guess_position_psf: dead-region-overlap NaN should not raise a RuntimeWarning."""
from __future__ import annotations

import warnings

import numpy as np

from spherical.pipeline.steps.find_star import guess_position_psf


class TestGuessPositionPsfNoWarnOnDeadRegionOverlap:
    def test_two_channel_dbi_with_overlapping_nan_pixels_is_silent(self):
        # Simulate IRDIS DBI: 2 wavelengths, 100x100, a bright PSF at (50, 60),
        # and a dead-region column that is NaN in BOTH channels — exactly the
        # 51 Eri K12 pattern that used to trigger 'All-NaN slice encountered'.
        cube = np.zeros((2, 100, 100), dtype=float)
        cube[0, 50, 60] = 1000.0
        cube[1, 50, 60] = 1000.0
        cube[:, 15:20, 30:40] = np.nan  # dead in BOTH channels

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cy, cx = guess_position_psf(cube, exclude_edge_pixels=5)

        assert (cy, cx) == (50, 60)
        nan_warns = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "All-NaN slice" in str(w.message)
        ]
        assert nan_warns == [], (
            f"guess_position_psf should silence expected dead-region "
            f"All-NaN warning, got: {[str(w.message) for w in nan_warns]}"
        )

    def test_clean_two_channel_cube_still_finds_center(self):
        cube = np.zeros((2, 60, 60), dtype=float)
        cube[:, 30, 25] = 500.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning now becomes a test failure
            cy, cx = guess_position_psf(cube, exclude_edge_pixels=3)
        assert (cy, cx) == (30, 25)
