"""Tests for the IRDIS preprocess step (Phase 4)."""
from __future__ import annotations

import numpy as np
import pytest

from spherical.pipeline.steps.irdis_calibration import dead_region_mask
from spherical.pipeline.steps.irdis_preprocess import (
    NOMINAL_STAR_POSITIONS_H_BAND,
    NOMINAL_STAR_POSITIONS_K_BAND,
    analytic_ivar,
    background_region_mask,
    coarse_star_position,
    fit_background_scale,
    nominal_star_positions,
    subtract_scaled_background,
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
        import warnings
        img = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32) * 5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cx, cy = coarse_star_position(img, nominal_xy=(500.0, 500.0), search_radius=20)
        assert (cx, cy) == (500.0, 500.0)

    def test_warns_on_fallback(self):
        import warnings
        img = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32) * 5
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            coarse_star_position(img, nominal_xy=(500.0, 500.0), search_radius=20)
        assert any(issubclass(wi.category, UserWarning) for wi in w), \
            "expected UserWarning on nominal fallback"

    def test_ignores_nan_pixels(self):
        img = self._make_image_with_star(cx=505.0, cy=507.0, amp=1000.0)
        img[:15, :] = np.nan  # simulate dead row band
        cx, cy = coarse_star_position(img, nominal_xy=(500.0, 500.0), search_radius=30)
        assert abs(cx - 505.0) < 1.5
        assert abs(cy - 507.0) < 1.5


class TestBackgroundRegionMask:
    def test_shape_and_dtype(self):
        dm = dead_region_mask()
        bpm = np.zeros((1024, 1024), dtype=bool)
        mask = background_region_mask(
            shape=(1024, 1024),
            star_xy=(500.0, 500.0),
            star_mask_radius=40,
            dead_mask_ch=dm[0],
            bpm_ch=bpm,
        )
        assert mask.shape == (1024, 1024)
        assert mask.dtype == np.bool_

    def test_star_region_excluded(self):
        dm = np.zeros((1024, 1024), dtype=bool)
        bpm = np.zeros((1024, 1024), dtype=bool)
        mask = background_region_mask(
            shape=(1024, 1024),
            star_xy=(500.0, 500.0),
            star_mask_radius=40,
            dead_mask_ch=dm,
            bpm_ch=bpm,
        )
        assert not mask[500, 500]
        assert not mask[500, 460]  # on the circle boundary
        assert mask[500, 600]  # outside

    def test_dead_and_bad_excluded(self):
        dm = dead_region_mask()
        bpm = np.zeros((1024, 1024), dtype=bool)
        bpm[300, 300] = True
        mask = background_region_mask(
            shape=(1024, 1024),
            star_xy=(500.0, 500.0),
            star_mask_radius=40,
            dead_mask_ch=dm[0],
            bpm_ch=bpm,
        )
        assert not mask[0, 500]  # dead row band
        assert not mask[300, 300]  # bad pixel


class TestFitBackgroundScale:
    def test_recovers_known_scale_without_noise(self):
        rng = np.random.default_rng(0)
        bg = rng.uniform(50.0, 200.0, size=(200, 200)).astype(np.float32)
        s_true = 1.35
        frame = s_true * bg
        mask = np.ones_like(bg, dtype=bool)
        s = fit_background_scale(frame, bg, mask)
        assert abs(s - s_true) < 1e-4

    def test_ignores_masked_pixels(self):
        bg = np.full((100, 100), 100.0, dtype=np.float32)
        frame = 2.0 * bg
        frame[50, 50] = 1e6  # would blow up an unweighted fit
        mask = np.ones_like(bg, dtype=bool)
        mask[50, 50] = False
        s = fit_background_scale(frame, bg, mask)
        assert abs(s - 2.0) < 1e-4

    def test_sigma_clips_outliers(self):
        rng = np.random.default_rng(1)
        bg = rng.uniform(50.0, 200.0, size=(200, 200)).astype(np.float32)
        s_true = 1.35
        frame = s_true * bg + rng.standard_normal(bg.shape).astype(np.float32) * 0.5
        # Inject a strong outlier that would bias a naive fit.
        frame[100, 100] = 1e5
        mask = np.ones_like(bg, dtype=bool)
        s = fit_background_scale(frame, bg, mask, n_sigma_clip=3, max_iter=3)
        assert abs(s - s_true) < 0.02

    def test_returns_zero_for_empty_mask(self):
        bg = np.ones((10, 10), dtype=np.float32)
        frame = np.ones((10, 10), dtype=np.float32)
        mask = np.zeros((10, 10), dtype=bool)
        assert fit_background_scale(frame, bg, mask) == 0.0


class TestSubtractScaledBackground:
    def test_subtracts_scaled_bg_and_returns_scale(self):
        rng = np.random.default_rng(2)
        bg = rng.uniform(50.0, 200.0, size=(1024, 1024)).astype(np.float32)
        s_true = 1.10
        frame = s_true * bg
        dm = dead_region_mask()[0]
        bpm = np.zeros((1024, 1024), dtype=bool)
        residual, s = subtract_scaled_background(
            frame_ch=frame,
            bg_ch=bg,
            star_xy=(500.0, 500.0),
            star_mask_radius=40,
            dead_mask_ch=dm,
            bpm_ch=bpm,
        )
        assert abs(s - s_true) < 1e-3
        # Off the dead bands the residual should be ~0.
        assert abs(residual[600, 600]) < 1e-2


class TestAnalyticIvar:
    def test_read_noise_only_at_zero_counts(self):
        counts = np.zeros((4, 4), dtype=np.float32)
        flat = np.ones_like(counts)
        ivar = analytic_ivar(counts, flat, gain=1.75, read_noise=4.4)
        # ivar = gain^2 / (0 + read_noise^2) = 1.75^2 / 4.4^2
        expected = 1.75 ** 2 / 4.4 ** 2
        np.testing.assert_allclose(ivar, expected, rtol=1e-5)

    def test_photon_dominated_at_high_counts(self):
        counts = np.full((4, 4), 10000.0, dtype=np.float32)
        flat = np.ones_like(counts)
        ivar = analytic_ivar(counts, flat, gain=1.75, read_noise=4.4)
        # At high counts, ivar ≈ gain / counts.
        expected = 1.75 / 10000.0
        np.testing.assert_allclose(ivar, expected, rtol=1.15e-3)

    def test_flat_propagation(self):
        counts = np.full((4, 4), 1000.0, dtype=np.float32)
        flat_unit = np.ones_like(counts)
        flat_half = np.full_like(counts, 0.5)
        iv_unit = analytic_ivar(counts, flat_unit, gain=1.75, read_noise=4.4)
        iv_half = analytic_ivar(counts, flat_half, gain=1.75, read_noise=4.4)
        # ivar scales as flat^2 → half-flat gives quarter ivar.
        np.testing.assert_allclose(iv_half, iv_unit * 0.25, rtol=1e-5)

    def test_negative_counts_treated_as_zero(self):
        counts = np.full((4, 4), -500.0, dtype=np.float32)
        flat = np.ones_like(counts)
        ivar = analytic_ivar(counts, flat, gain=1.75, read_noise=4.4)
        expected = 1.75 ** 2 / 4.4 ** 2
        np.testing.assert_allclose(ivar, expected, rtol=1e-5)

    def test_nan_gives_zero_ivar(self):
        counts = np.array([[100.0, np.nan], [np.nan, 100.0]], dtype=np.float32)
        flat = np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float32)
        ivar = analytic_ivar(counts, flat, gain=1.75, read_noise=4.4)
        assert ivar[0, 0] > 0
        assert ivar[0, 1] == 0.0
        assert ivar[1, 0] == 0.0
        assert ivar[1, 1] > 0

    def test_dtype_float32(self):
        counts = np.full((4, 4), 100.0, dtype=np.float32)
        flat = np.ones_like(counts)
        ivar = analytic_ivar(counts, flat, gain=1.75, read_noise=4.4)
        assert ivar.dtype == np.float32
