"""Tests for the IRDIS preprocess step (Phase 4)."""
from __future__ import annotations

import numpy as np
import pytest

from spherical.pipeline.steps.irdis_preprocess import (
    NOMINAL_STAR_POSITIONS_H_BAND,
    NOMINAL_STAR_POSITIONS_K_BAND,
    coarse_star_position,
    nominal_star_positions,
)


class TestNominalStarPositions:
    def test_k_band_lookup(self):
        pos = nominal_star_positions("DB_K12")
        assert pos.shape == (2, 2)
        np.testing.assert_allclose(pos, np.array(NOMINAL_STAR_POSITIONS_K_BAND))

    def test_h_band_lookup(self):
        pos = nominal_star_positions("DB_H23")
        np.testing.assert_allclose(pos, np.array(NOMINAL_STAR_POSITIONS_H_BAND))

    def test_broadband_h_is_h_band(self):
        pos = nominal_star_positions("BB_H")
        np.testing.assert_allclose(pos, np.array(NOMINAL_STAR_POSITIONS_H_BAND))

    def test_broadband_ks_is_k_band(self):
        pos = nominal_star_positions("BB_Ks")
        np.testing.assert_allclose(pos, np.array(NOMINAL_STAR_POSITIONS_K_BAND))

    def test_unknown_filter_raises(self):
        with pytest.raises(ValueError):
            nominal_star_positions("NOT_A_FILTER")


class TestCoarseStarPosition:
    def _make_image_with_star(self, cx=500.0, cy=500.0, amp=1000.0):
        img = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32) * 5
        y, x = np.mgrid[:1024, :1024]
        img += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 3.0**2))
        return img

    def test_recovers_star_near_nominal(self):
        img = self._make_image_with_star(cx=502.0, cy=498.0)
        cx, cy = coarse_star_position(img, nominal_xy=(500.0, 500.0), search_radius=20)
        assert abs(cx - 502.0) < 1.5
        assert abs(cy - 498.0) < 1.5

    def test_widens_search_when_star_outside_window(self):
        img = self._make_image_with_star(cx=800.0, cy=200.0, amp=2000.0)
        cx, cy = coarse_star_position(img, nominal_xy=(500.0, 500.0), search_radius=20)
        assert abs(cx - 800.0) < 3.0
        assert abs(cy - 200.0) < 3.0

    def test_falls_back_to_nominal_when_no_significant_peak(self):
        img = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32) * 5
        cx, cy = coarse_star_position(img, nominal_xy=(500.0, 500.0), search_radius=20)
        assert (cx, cy) == (500.0, 500.0)

    def test_ignores_nan_pixels(self):
        img = self._make_image_with_star(cx=505.0, cy=507.0, amp=1000.0)
        img[:15, :] = np.nan  # simulate dead row band
        cx, cy = coarse_star_position(img, nominal_xy=(500.0, 500.0), search_radius=30)
        assert abs(cx - 505.0) < 1.5
        assert abs(cy - 507.0) < 1.5
