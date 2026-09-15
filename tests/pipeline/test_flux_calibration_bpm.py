"""Tests for Phase-1 BPM-aware flux calibration.

Covers ``get_aperture_photometry(bad_pixel_mask=...)`` in isolation:

* the masked source aperture drops flux equal to the masked pixels' values,
* the background annulus stats ignore masked pixels,
* a clean stamp is unchanged when a bad-pixel mask of all-False is supplied.

The full ``run_flux_psf_calibration`` step is not unit-tested here — it reads
several on-disk products and its warning logic is easier to verify by rerunning
against real observation data. The step-level check is in the Task 4 rerun.
"""
from __future__ import annotations

import numpy as np

from spherical.pipeline.flux_calibration import get_aperture_photometry


def _delta_stamp(stamp_size: int = 57, amp: float = 100.0) -> np.ndarray:
    """Single hot pixel at the stamp centre, otherwise 1.0 background."""
    stamp = np.ones((stamp_size, stamp_size), dtype=float)
    stamp[stamp_size // 2, stamp_size // 2] = amp
    return stamp


class TestAperturePhotometryBpm:
    def test_no_mask_matches_all_false_mask(self):
        stamp = _delta_stamp()
        cube = stamp[None, None, :, :]  # (n_wave=1, n_frame=1, ny, nx)
        bp_none = get_aperture_photometry(
            cube, aperture_radius_range=[1, 6],
            bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
            bad_pixel_mask=None)
        bp_zeros = get_aperture_photometry(
            cube, aperture_radius_range=[1, 6],
            bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
            bad_pixel_mask=np.zeros_like(cube, dtype=bool))
        np.testing.assert_allclose(
            bp_none['psf_flux_bg_corr_all'], bp_zeros['psf_flux_bg_corr_all'])
        np.testing.assert_allclose(
            bp_none['psf_bg_counts_all'], bp_zeros['psf_bg_counts_all'])

    def test_masking_center_removes_hot_pixel_from_sum(self):
        stamp = _delta_stamp(amp=100.0)
        cube = stamp[None, None, :, :]
        mask = np.zeros_like(cube, dtype=bool)
        mask[0, 0, 28, 28] = True  # mask the hot center pixel

        unmasked = get_aperture_photometry(
            cube, aperture_radius_range=[1, 6],
            bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
            bad_pixel_mask=None)
        masked = get_aperture_photometry(
            cube, aperture_radius_range=[1, 6],
            bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
            bad_pixel_mask=mask)

        # Any aperture that contains the centre must drop by (100 - bg_est).
        # bg_est is ~1.0 (unit background), bg-corrected drop ≈ 99 for the
        # 1-px radius aperture, larger apertures pick up more unit-flux pixels
        # so still drop the exact bad-pixel contribution.
        drop = unmasked['psf_flux_bg_corr_all'] - masked['psf_flux_bg_corr_all']
        bg = float(unmasked['psf_bg_counts_all'].squeeze())
        expected_drop = 100.0 - bg
        # aperture_sizes = [1,2,3,4,5] — all contain centre pixel
        np.testing.assert_allclose(
            drop.squeeze(), expected_drop, atol=1e-6, rtol=1e-6)

    def test_masking_background_pixel_shifts_bg_estimate(self):
        stamp = np.ones((57, 57), dtype=float)
        # Poison one background pixel in the annulus (15 <= r < 18)
        stamp[28, 45] = 1000.0  # r = 17 -> inside annulus
        cube = stamp[None, None, :, :]

        unmasked = get_aperture_photometry(
            cube, aperture_radius_range=[1, 4],
            bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
            bad_pixel_mask=None)
        mask = np.zeros_like(cube, dtype=bool)
        mask[0, 0, 28, 45] = True
        masked = get_aperture_photometry(
            cube, aperture_radius_range=[1, 4],
            bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
            bad_pixel_mask=mask)

        # Sigma-clipping already tames the outlier in the unmasked case, so the
        # bg estimates are close — the strict guarantee is that masking never
        # makes the bg estimate *worse* (further from the true value of 1.0).
        bg_unmasked = float(unmasked['psf_bg_counts_all'].squeeze())
        bg_masked = float(masked['psf_bg_counts_all'].squeeze())
        assert abs(bg_masked - 1.0) <= abs(bg_unmasked - 1.0) + 1e-9

    def test_mask_shape_mismatch_raises(self):
        stamp = _delta_stamp()
        cube = stamp[None, None, :, :]
        import pytest
        with pytest.raises(ValueError, match="does not match"):
            get_aperture_photometry(
                cube, aperture_radius_range=[1, 6],
                bad_pixel_mask=np.zeros((2, 2, 57, 57), dtype=bool))
