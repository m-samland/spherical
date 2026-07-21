"""Tests for the Phase-2 Moffat core repair helper (``psf_repair``)."""
from __future__ import annotations

import numpy as np
import pytest

from spherical.pipeline.psf_repair import repair_psf_core


def _moffat_window(shape=(21, 21), amp=1000.0, gamma=2.5, alpha=2.5,
                   center=None, bg=5.0, noise_rms=1.0, seed=0):
    """Build a clean Moffat-shaped window with Gaussian noise (for tests)."""
    ny, nx = shape
    yy, xx = np.indices(shape)
    if center is None:
        center = (nx / 2.0, ny / 2.0)
    cx, cy = center
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    prof = amp * (1.0 + r2 / gamma ** 2) ** (-alpha) + bg
    rng = np.random.default_rng(seed)
    return prof + rng.normal(0.0, noise_rms, size=shape)


class TestRepairPsfCore:
    def test_returns_unchanged_when_no_bpm_in_core(self):
        win = _moffat_window()
        ivar = np.ones_like(win)  # all good
        res = repair_psf_core(win, ivar, (10.0, 10.0), core_radius_px=3.5)
        assert res.status == "skipped_no_bpm"
        assert res.n_repaired == 0
        np.testing.assert_array_equal(res.window_out, win)

    def test_repairs_injected_bad_pixel_with_finite_positive_value(self):
        win = _moffat_window(amp=1000.0, gamma=2.5, alpha=2.5,
                             center=(10.0, 10.0), bg=5.0, noise_rms=0.5)
        peak_true = float(win[10, 10])
        # inject a bad pixel at the peak
        ivar = np.ones_like(win)
        ivar[10, 10] = 0.0
        # simulate what preprocess would do — poison the value
        win_poisoned = win.copy()
        win_poisoned[10, 10] = 0.0
        res = repair_psf_core(win_poisoned, ivar, (10.0, 10.0), core_radius_px=3.5)
        assert res.status == "repaired"
        assert res.n_repaired == 1
        repaired_value = float(res.window_out[10, 10])
        assert np.isfinite(repaired_value)
        # The Moffat fit should recover a value close to the true peak
        # (within ~10% for a clean Moffat source).
        assert abs(repaired_value - peak_true) / peak_true < 0.15
        # Ivar of the repaired pixel is downweighted, not zero.
        assert 0 < res.ivar_out[10, 10] < 1.0
        assert res.residual_rms_frac < 0.10

    def test_fallback_when_residual_rms_too_high(self):
        # Pure noise (no PSF at all) → Moffat fit should have high residual RMS
        rng = np.random.default_rng(1)
        win = rng.normal(0.0, 100.0, size=(21, 21))
        ivar = np.ones_like(win)
        ivar[10, 10] = 0.0
        win[10, 10] = 0.0
        res = repair_psf_core(
            win, ivar, (10.0, 10.0), core_radius_px=3.5,
            residual_rms_frac_threshold=0.05)
        assert res.status in ("skipped_bad_fit", "skipped_insufficient_good_pixels")
        assert res.n_repaired == 0
        # Original window returned unchanged (bad pixel not repaired)
        np.testing.assert_array_equal(res.window_out, win)

    def test_skipped_when_too_few_good_pixels(self):
        # Bad pixels almost fill the core
        win = _moffat_window()
        ivar = np.ones_like(win)
        yy, xx = np.indices(win.shape)
        r = np.sqrt((yy - 10) ** 2 + (xx - 10) ** 2)
        core = r <= 3.5
        # kill 90% of the core pixels
        core_indices = np.argwhere(core)
        rng = np.random.default_rng(2)
        rng.shuffle(core_indices)
        kill = core_indices[: int(0.9 * len(core_indices))]
        for iy, ix in kill:
            ivar[iy, ix] = 0.0
            win[iy, ix] = 0.0
        res = repair_psf_core(win, ivar, (10.0, 10.0), core_radius_px=3.5,
                              min_good_core_pixels=8)
        assert res.status == "skipped_insufficient_good_pixels"

    def test_edge_case_center_outside_window(self):
        win = _moffat_window()
        ivar = np.ones_like(win)
        res = repair_psf_core(win, ivar, (-5.0, 10.0), core_radius_px=3.5)
        assert res.status == "skipped_edge"

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same-shape"):
            repair_psf_core(np.zeros((5, 5)), np.zeros((6, 5)), (2.0, 2.0), 1.0)

    def test_leaves_pixels_outside_core_untouched(self):
        win = _moffat_window(amp=1000.0, center=(10.0, 10.0))
        ivar = np.ones_like(win)
        # a bad pixel *outside* the core should NOT be repaired
        ivar[0, 0] = 0.0
        win[0, 0] = -9999.0
        # ...and a bad pixel *inside* the core should be
        ivar[10, 10] = 0.0
        win_before_10_10 = win[10, 10]
        win_backup = win.copy()
        win_backup[10, 10] = 0.0
        res = repair_psf_core(win_backup, ivar, (10.0, 10.0), core_radius_px=3.5)
        assert res.status == "repaired"
        assert res.window_out[0, 0] == pytest.approx(-9999.0)
        assert res.window_out[10, 10] != pytest.approx(win_before_10_10)
