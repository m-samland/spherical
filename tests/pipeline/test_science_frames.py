"""Tests for the shared science-frame helpers."""
from __future__ import annotations

import numpy as np
import pytest

from spherical.pipeline.science_frames import (
    normalize_centers_to_frames,
    science_frame_type,
    verify_frame_axis,
)


class TestScienceFrameType:
    def test_waffle_uses_center(self):
        assert science_frame_type(True) == "center"

    def test_non_waffle_uses_coro(self):
        assert science_frame_type(False) == "coro"

    def test_accepts_numpy_bool(self):
        assert science_frame_type(np.bool_(True)) == "center"


class TestNormalizeCentersToFrames:
    def test_irdis_waffle_passes_through(self):
        centers = np.random.default_rng(0).normal(size=(2, 40, 2))
        out = normalize_centers_to_frames(centers, 40, "IRDIS", True)
        np.testing.assert_array_equal(out, centers)

    def test_irdis_non_waffle_passes_through(self):
        centers = np.random.default_rng(1).normal(size=(2, 137, 2))
        out = normalize_centers_to_frames(centers, 137, "IRDIS", False)
        np.testing.assert_array_equal(out, centers)

    def test_ifs_waffle_passes_through(self):
        centers = np.random.default_rng(2).normal(size=(39, 12, 2))
        out = normalize_centers_to_frames(centers, 12, "IFS", True)
        np.testing.assert_array_equal(out, centers)

    def test_ifs_non_waffle_collapses_and_broadcasts(self):
        centers = np.random.default_rng(3).normal(size=(39, 3, 2))
        out = normalize_centers_to_frames(centers, 200, "IFS", False)
        assert out.shape == (39, 200, 2)
        expected = np.nanmean(centers, axis=1)
        np.testing.assert_allclose(out[:, 0, :], expected)
        np.testing.assert_allclose(out[:, 199, :], expected)

    def test_ifs_non_waffle_ignores_nan_in_the_collapse(self):
        centers = np.full((2, 4, 2), np.nan)
        centers[:, 1, :] = 5.0
        out = normalize_centers_to_frames(centers, 3, "IFS", False)
        np.testing.assert_allclose(out, 5.0)

    def test_unknown_instrument_raises(self):
        centers = np.zeros((2, 4, 2))
        with pytest.raises(ValueError, match="Unknown instrument"):
            normalize_centers_to_frames(centers, 4, "SPHERE", False)


class TestVerifyFrameAxis:
    def test_matching_axis_is_silent(self):
        verify_frame_axis(np.zeros((2, 10, 2)), 10, "coro")

    def test_mismatch_raises_and_names_both_counts(self):
        with pytest.raises(ValueError) as excinfo:
            verify_frame_axis(np.zeros((2, 10, 2)), 7, "center")
        message = str(excinfo.value)
        assert "10" in message
        assert "7" in message
        assert "frames_info_center.csv" in message
