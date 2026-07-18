"""Tests for the IRDIS preprocess step (Phase 4)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from spherical.pipeline.pipeline_config import IRDISPreprocessConfig
from spherical.pipeline.steps.irdis_calibration import DEAD_ROW_SLICE_BOTTOM, dead_region_mask
from spherical.pipeline.steps.irdis_preprocess import (
    NOMINAL_STAR_POSITIONS_H_BAND,
    NOMINAL_STAR_POSITIONS_K_BAND,
    analytic_ivar,
    apply_anamorphism,
    apply_crop,
    background_region_mask,
    coarse_star_position,
    fit_background_scale,
    fix_badpix_nan_safe,
    nominal_star_positions,
    preprocess_frame_type,
    sigma_filter_ignore_dead,
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


class TestFixBadpixNanSafe:
    def test_interpolates_bad_pixel_in_bulk(self):
        frame = np.full((100, 100), 100.0, dtype=np.float32)
        frame[50, 50] = 1e5
        bpm = np.zeros_like(frame, dtype=bool)
        bpm[50, 50] = True
        dead = np.zeros_like(frame, dtype=bool)
        fixed = fix_badpix_nan_safe(frame, bpm, dead)
        assert abs(fixed[50, 50] - 100.0) < 1.0

    def test_preserves_dead_region_nan(self):
        dm = dead_region_mask()[0]
        frame = np.full(dm.shape, 100.0, dtype=np.float32)
        frame[dm] = np.nan
        bpm = np.zeros_like(dm)
        # Add a bad pixel far from any dead band.
        bpm[500, 500] = True
        frame[500, 500] = 1e5
        fixed = fix_badpix_nan_safe(frame, bpm, dm)
        assert np.all(np.isnan(fixed[dm]))
        assert abs(fixed[500, 500] - 100.0) < 1.0

    def test_dead_region_boundary_bad_pixel_no_nan_halo(self):
        """Regression test — Phase 3 downstream contract concrete case.

        Bad pixel at (row=20, col=100) — 5 rows above DEAD_ROW_SLICE_BOTTOM
        (which ends at row 15). After preprocess:
            fixed[20, 100] must be finite (interpolated cleanly);
            fixed[5, 100] must still be NaN (dead region preserved).
        """
        assert DEAD_ROW_SLICE_BOTTOM.stop == 15
        dm = dead_region_mask()[0]
        frame = np.full(dm.shape, 100.0, dtype=np.float32)
        frame[dm] = np.nan
        bpm = np.zeros_like(dm)
        bpm[20, 100] = True
        frame[20, 100] = 1e5
        fixed = fix_badpix_nan_safe(frame, bpm, dm)
        assert np.isfinite(fixed[20, 100]), "bad pixel near dead-band was not interpolated"
        assert abs(fixed[20, 100] - 100.0) < 5.0, "interpolated value contaminated by dead-region NaN"
        assert np.isnan(fixed[5, 100]), "dead-region pixel was overwritten"

    def test_no_op_when_no_bad_pixels(self):
        frame = np.full((50, 50), 100.0, dtype=np.float32)
        bpm = np.zeros_like(frame, dtype=bool)
        dead = np.zeros_like(frame, dtype=bool)
        fixed = fix_badpix_nan_safe(frame, bpm, dead)
        np.testing.assert_allclose(fixed, 100.0, rtol=1e-6)


class TestSigmaFilterIgnoreDead:
    def test_flags_transient_outlier(self):
        rng = np.random.default_rng(0)
        frame = 100.0 + rng.standard_normal((100, 100)).astype(np.float32) * 2.0
        frame[50, 50] = 1e4  # transient
        dead = np.zeros((100, 100), dtype=bool)
        cleaned, mask = sigma_filter_ignore_dead(frame, dead, box=7, nsigma=4)
        assert mask[50, 50]

    def test_dead_pixels_never_flagged(self):
        dm = dead_region_mask()[0]
        frame = np.full(dm.shape, 100.0, dtype=np.float32)
        frame[dm] = np.nan
        cleaned, mask = sigma_filter_ignore_dead(frame, dm, box=7, nsigma=4)
        # No dead-region pixel should be flagged as a transient outlier.
        assert not mask[dm].any()

    def test_dead_pixels_remain_nan(self):
        dm = dead_region_mask()[0]
        frame = np.full(dm.shape, 100.0, dtype=np.float32)
        frame[dm] = np.nan
        cleaned, _ = sigma_filter_ignore_dead(frame, dm, box=7, nsigma=4)
        assert np.all(np.isnan(cleaned[dm]))


class TestApplyAnamorphism:
    def test_identity_when_factor_one(self):
        cube = np.random.default_rng(0).standard_normal((3, 100, 100)).astype(np.float32)
        dm = np.zeros((100, 100), dtype=bool)
        out = apply_anamorphism(cube, factor=1.0, dead_mask_ch=dm)
        np.testing.assert_array_equal(out, cube)

    def test_preserves_shape(self):
        cube = np.ones((3, 100, 100), dtype=np.float32)
        dm = np.zeros((100, 100), dtype=bool)
        out = apply_anamorphism(cube, factor=1.0062, dead_mask_ch=dm)
        assert out.shape == cube.shape

    def test_dead_region_stays_nan(self):
        dm = dead_region_mask()[0]
        cube = np.full((2,) + dm.shape, 100.0, dtype=np.float32)
        cube[:, dm] = np.nan
        out = apply_anamorphism(cube, factor=1.0062, dead_mask_ch=dm)
        assert np.all(np.isnan(out[:, dm]))


class TestApplyCrop:
    def test_crops_square_around_star(self):
        cube = np.ones((3, 1024, 1024), dtype=np.float32)
        ivar = np.ones_like(cube)
        cube_c, ivar_c, (x0, y0) = apply_crop(cube, ivar, star_xy=(500.0, 500.0), crop_size=200)
        assert cube_c.shape == (3, 200, 200)
        assert ivar_c.shape == (3, 200, 200)
        assert x0 == 400
        assert y0 == 400

    def test_offset_clamped_to_frame(self):
        cube = np.ones((2, 100, 100), dtype=np.float32)
        ivar = np.ones_like(cube)
        cube_c, ivar_c, (x0, y0) = apply_crop(cube, ivar, star_xy=(10.0, 90.0), crop_size=50)
        assert x0 == 0
        assert y0 == 50

    def test_preserves_values(self):
        cube = np.arange(1024 * 1024, dtype=np.float32).reshape(1, 1024, 1024)
        ivar = cube.copy()
        cube_c, _, (x0, y0) = apply_crop(cube, ivar, star_xy=(500.0, 500.0), crop_size=100)
        np.testing.assert_array_equal(cube_c[0], cube[0, y0:y0 + 100, x0:x0 + 100])


def _write_raw_irdis_file(path, n_dit=1, level=500.0):
    from astropy.io import fits
    frame = np.full((1024, 2048), level, dtype=np.float32)
    if n_dit > 1:
        frame = np.broadcast_to(frame, (n_dit, 1024, 2048)).copy()
    else:
        frame = frame[np.newaxis, ...]
    fits.writeto(path, frame)
    return str(path)


class TestPreprocessFrameType:
    def _make_calibration(self):
        dm = dead_region_mask()
        # Flat is 1.0 everywhere except dead regions (NaN).
        flat = np.ones((2, 1024, 1024), dtype=np.float32)
        flat[dm] = np.nan
        # Background is 100 counts everywhere except dead regions.
        bg = np.full((2, 1024, 1024), 100.0, dtype=np.float32)
        bg[dm] = np.nan
        # Empty bpm (dead regions False by convention).
        bpm = np.zeros((2, 1024, 1024), dtype=bool)
        return flat, bg, bpm

    def test_returns_shapes_and_dtypes(self, tmp_path):
        flat, bg, bpm = self._make_calibration()
        # A single CORO file with 2 DITs, level = star=0 + bg=100.
        p = _write_raw_irdis_file(tmp_path / "coro.fits", n_dit=2, level=100.0)
        star_positions = np.array([[500.0, 500.0], [500.0, 500.0]])
        cfg = IRDISPreprocessConfig()
        cube, ivar, bpm_out, offsets = preprocess_frame_type(
            [p], flat, bg, bpm, star_positions, filter_comb="DB_K12",
            is_flux=False, preprocess_config=cfg, logger=MagicMock(),
        )
        assert cube.shape == (2, 2, 1024, 1024)
        assert ivar.shape == cube.shape
        assert cube.dtype == np.float32
        assert ivar.dtype == np.float32
        assert bpm_out.shape == (2, 1024, 1024)
        assert offsets is None  # crop disabled by default

    def test_dead_regions_nan_data_zero_ivar(self, tmp_path):
        flat, bg, bpm = self._make_calibration()
        p = _write_raw_irdis_file(tmp_path / "coro.fits", n_dit=1, level=200.0)
        star_positions = np.array([[500.0, 500.0], [500.0, 500.0]])
        cfg = IRDISPreprocessConfig()
        cube, ivar, _, _ = preprocess_frame_type(
            [p], flat, bg, bpm, star_positions, filter_comb="DB_K12",
            is_flux=False, preprocess_config=cfg, logger=MagicMock(),
        )
        dm = dead_region_mask()
        # Data NaN in dead regions on both channels, all times.
        assert np.all(np.isnan(cube[0, :, dm[0]]))
        assert np.all(np.isnan(cube[1, :, dm[1]]))
        assert np.all(ivar[0, :, dm[0]] == 0)
        assert np.all(ivar[1, :, dm[1]] == 0)

    def test_scaled_bg_subtraction_yields_zero_signal(self, tmp_path):
        flat, bg, bpm = self._make_calibration()
        # Frames all at bg-level → after subtraction, near zero in illuminated region.
        p = _write_raw_irdis_file(tmp_path / "coro.fits", n_dit=1, level=100.0)
        star_positions = np.array([[500.0, 500.0], [500.0, 500.0]])
        cfg = IRDISPreprocessConfig()
        cube, _, _, _ = preprocess_frame_type(
            [p], flat, bg, bpm, star_positions, filter_comb="DB_K12",
            is_flux=False, preprocess_config=cfg, logger=MagicMock(),
        )
        # Pixel well away from dead bands and the star mask.
        assert abs(cube[0, 0, 700, 700]) < 1.0

    def test_dead_region_boundary_regression(self, tmp_path):
        """Phase 3 downstream-contract concrete case (repeated at cube level).

        Bad pixel at (ch=0, row=20, col=100), 5 rows from
        DEAD_ROW_SLICE_BOTTOM.stop. After preprocess we expect the pixel
        interpolated cleanly and ivar=0 there, dead-region row 5 still NaN
        with ivar=0.
        """
        flat, bg, bpm = self._make_calibration()
        bpm[0, 20, 100] = True  # flag as bad
        # Write a raw frame at the bg level with one huge outlier at the target
        # pixel; the robust scale fit pulls ~1.0, bulk residual ≈ 0, and the
        # interpolated value should therefore also sit near 0 (not be the raw
        # outlier magnitude, and crucially not NaN from a dead-region neighbor).
        from astropy.io import fits
        raw = np.full((1024, 2048), 100.0, dtype=np.float32)
        raw[20, 100] = 1e6  # left-half pixel (column < 1024)
        path = tmp_path / "coro.fits"
        fits.writeto(path, raw[np.newaxis, ...])

        star_positions = np.array([[500.0, 500.0], [500.0, 500.0]])
        cfg = IRDISPreprocessConfig()
        cube, ivar, _, _ = preprocess_frame_type(
            [str(path)], flat, bg, bpm, star_positions,
            filter_comb="DB_K12", is_flux=False,
            preprocess_config=cfg, logger=MagicMock(),
        )
        # Interpolated cleanly (no NaN halo from dead-region neighbors); the
        # value should be small (bulk residual ~ 0), not NaN, not ~1e6.
        assert np.isfinite(cube[0, 0, 20, 100]), "bad pixel near dead-band was not interpolated"
        assert abs(cube[0, 0, 20, 100]) < 20.0, "interpolated value has unexpected magnitude"
        assert ivar[0, 0, 20, 100] == 0.0
        # Dead-region pixel preserved.
        assert np.isnan(cube[0, 0, 5, 100])
        assert ivar[0, 0, 5, 100] == 0.0

    def test_flux_frames_skip_transient_sigma_clip(self, tmp_path):
        flat, bg, bpm = self._make_calibration()
        # FLUX frame with the star peak — must NOT be flagged as a transient.
        from astropy.io import fits
        raw = np.full((1024, 2048), 100.0, dtype=np.float32)
        raw[500, 500] = 5e4  # star peak in left half (raw col 500)
        path = tmp_path / "flux.fits"
        fits.writeto(path, raw[np.newaxis, ...])

        star_positions = np.array([[500.0, 500.0], [500.0, 500.0]])
        cfg = IRDISPreprocessConfig()
        cube, _, _, _ = preprocess_frame_type(
            [str(path)], flat, bg, bpm, star_positions,
            filter_comb="DB_K12", is_flux=True,
            preprocess_config=cfg, logger=MagicMock(),
        )
        # Star peak must survive.
        assert cube[0, 0, 500, 500] > 4e4
