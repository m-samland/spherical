"""Tests for the IRDIS calibration step (Phase 3)."""
from __future__ import annotations

import numpy as np
import pytest

from spherical.pipeline.steps.irdis_calibration import (
    DEAD_COL_SLICE_LEFT_HALF_EDGE,
    DEAD_COL_SLICE_MID,
    DEAD_COL_SLICE_RIGHT_HALF_EDGE,
    DEAD_ROW_SLICE_BOTTOM,
    DEAD_ROW_SLICE_TOP,
    DETECTOR_COLS,
    DETECTOR_ROWS,
    HALF_COLS,
    dead_region_mask,
    split_detector_cube,
    split_detector_image,
)


class TestDetectorGeometry:
    def test_constants(self):
        assert DETECTOR_ROWS == 1024
        assert DETECTOR_COLS == 2048
        assert HALF_COLS == 1024
        assert DEAD_ROW_SLICE_BOTTOM == slice(0, 15)
        assert DEAD_ROW_SLICE_TOP == slice(1013, 1024)
        assert DEAD_COL_SLICE_LEFT_HALF_EDGE == slice(0, 50)
        assert DEAD_COL_SLICE_MID == slice(941, 1078)
        assert DEAD_COL_SLICE_RIGHT_HALF_EDGE == slice(1966, 2048)


class TestSplitDetector:
    def test_split_image_shape(self):
        img = np.arange(1024 * 2048).reshape(1024, 2048).astype(np.float32)
        split = split_detector_image(img)
        assert split.shape == (2, 1024, 1024)

    def test_split_image_preserves_values(self):
        img = np.arange(1024 * 2048).reshape(1024, 2048).astype(np.float32)
        split = split_detector_image(img)
        np.testing.assert_array_equal(split[0], img[:, 0:1024])
        np.testing.assert_array_equal(split[1], img[:, 1024:2048])

    def test_split_cube_shape(self):
        cube = np.zeros((5, 1024, 2048), dtype=np.float32)
        split = split_detector_cube(cube)
        assert split.shape == (5, 2, 1024, 1024)

    def test_split_cube_preserves_values(self):
        cube = np.random.default_rng(42).random((3, 1024, 2048)).astype(np.float32)
        split = split_detector_cube(cube)
        np.testing.assert_array_equal(split[:, 0], cube[:, :, 0:1024])
        np.testing.assert_array_equal(split[:, 1], cube[:, :, 1024:2048])

    def test_split_image_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            split_detector_image(np.zeros((100, 100)))


class TestDeadRegionMask:
    def test_shape_and_dtype(self):
        mask = dead_region_mask()
        assert mask.shape == (2, 1024, 1024)
        assert mask.dtype == np.bool_

    def test_top_and_bottom_row_bands_flagged_on_both_halves(self):
        mask = dead_region_mask()
        assert mask[:, DEAD_ROW_SLICE_BOTTOM, :].all()
        assert mask[:, DEAD_ROW_SLICE_TOP, :].all()

    def test_left_half_edge_flagged(self):
        mask = dead_region_mask()
        assert mask[0, :, DEAD_COL_SLICE_LEFT_HALF_EDGE].all()

    def test_right_half_edge_flagged(self):
        mask = dead_region_mask()
        # DEAD_COL_SLICE_RIGHT_HALF_EDGE is 1966..2048 in RAW coords,
        # which is 942..1024 in per-half coords.
        assert mask[1, :, 942:1024].all()

    def test_mid_band_flagged_across_seam(self):
        mask = dead_region_mask()
        # DEAD_COL_SLICE_MID = 941..1078 in RAW coords.
        # Channel 0: raw cols 941..1023 → per-half 941..1023.
        # Channel 1: raw cols 1024..1077 → per-half 0..53 (exclusive of 54).
        assert mask[0, :, 941:1024].all()
        assert mask[1, :, 0:54].all()

    def test_center_of_frame_not_flagged(self):
        mask = dead_region_mask()
        # A pixel in the middle of the left half, well away from any dead band.
        assert not mask[0, 512, 512]
        # A pixel in the middle of the right half.
        assert not mask[1, 512, 512]
