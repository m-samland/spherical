"""extract_satellite_spot_stamps: NaN-safe near-edge extraction (#163)."""
from __future__ import annotations

import logging

import numpy as np

from spherical.pipeline import toolbox

LOGGER = logging.getLogger("test")


def _centers(cy, cx, nwave=2, nframes=1, nspots=1):
    xy = np.full((nwave, nframes, nspots, 2), np.nan)
    xy[..., 0] = cx
    xy[..., 1] = cy
    return xy


class TestExtractSatelliteSpotStamps:
    def test_interior_center_extracts_a_full_finite_stamp(self):
        cube = np.random.default_rng(0).normal(size=(2, 1, 262, 262))
        out = toolbox.extract_satellite_spot_stamps(
            cube, _centers(130.0, 130.0), stamp_size=57, logger=LOGGER)
        assert out.shape == (2, 1, 57, 57)
        assert np.isfinite(out).all()

    def test_near_edge_center_does_not_raise(self):
        # #163: the cutout trims to (55, 57) and the assignment used to raise.
        cube = np.random.default_rng(0).normal(size=(2, 1, 262, 262))
        out = toolbox.extract_satellite_spot_stamps(
            cube, _centers(26.3, 130.0), stamp_size=57, logger=LOGGER)
        assert out.shape == (2, 1, 57, 57)

    def test_near_edge_stamp_keeps_most_of_its_data(self):
        cube = np.random.default_rng(0).normal(size=(2, 1, 262, 262))
        out = toolbox.extract_satellite_spot_stamps(
            cube, _centers(26.3, 130.0), stamp_size=57, logger=LOGGER)
        nan_frac = np.isnan(out[0, 0]).mean()
        assert 0.0 < nan_frac < 0.15, (
            f"only the off-frame padding should be NaN, got {nan_frac:.2%}")

    def test_interior_nan_stays_local(self):
        cube = np.random.default_rng(0).normal(size=(2, 1, 262, 262))
        cube[:, :, 130, 130] = np.nan
        out = toolbox.extract_satellite_spot_stamps(
            cube, _centers(130.4, 130.4), stamp_size=57, logger=LOGGER)
        assert np.isnan(out[0, 0]).sum() < 20, "one bad lenslet must not eat the stamp"

    def test_finite_stamp_is_bit_identical_to_the_previous_behaviour(self):
        from astropy.nddata import Cutout2D
        from scipy.ndimage import shift

        cube = np.random.default_rng(0).normal(size=(2, 1, 262, 262))
        out = toolbox.extract_satellite_spot_stamps(
            cube, _centers(130.4, 130.4), stamp_size=57, logger=LOGGER)
        c = Cutout2D(cube[0, 0], (130.4, 130.4), 57, mode='partial',
                     fill_value=np.nan, copy=True)
        sp = np.array(c.position_original) - np.array(c.input_position_original)
        expected = shift(c.data, (sp[-1], sp[-2]), order=3, mode='constant',
                         cval=0.0, prefilter=True)
        assert np.array_equal(out[0, 0], expected)

    def test_warns_about_partial_stamps(self, caplog):
        cube = np.random.default_rng(0).normal(size=(2, 1, 262, 262))
        with caplog.at_level(logging.WARNING):
            toolbox.extract_satellite_spot_stamps(
                cube, _centers(26.3, 130.0), stamp_size=57, logger=LOGGER)
        assert any("outside the frame" in r.message for r in caplog.records)

    def test_nan_position_is_skipped(self):
        cube = np.random.default_rng(0).normal(size=(2, 1, 262, 262))
        out = toolbox.extract_satellite_spot_stamps(
            cube, _centers(np.nan, np.nan), stamp_size=57, logger=LOGGER)
        assert np.isnan(out).all()


class TestDownstreamNanPolicy:
    def test_masked_nan_keeps_aperture_photometry_finite(self):
        # NaN inside the aperture poisons the sum unless it is masked.
        from spherical.pipeline import flux_calibration

        g = np.mgrid[:57, :57]
        psf = 1e4 * np.exp(-(((g[0] - 28) ** 2 + (g[1] - 28) ** 2) / 8.0))
        stamps = np.random.default_rng(0).normal(100.0, 1.0, size=(2, 3, 57, 57)) + psf
        stamps[:, :, 27:30, 27:30] = np.nan

        unmasked = flux_calibration.get_aperture_photometry(
            stamps.copy(), aperture_radius_range=[1, 15],
            bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
            bad_pixel_mask=None)
        masked = flux_calibration.get_aperture_photometry(
            stamps.copy(), aperture_radius_range=[1, 15],
            bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
            bad_pixel_mask=~np.isfinite(stamps))

        assert not np.isfinite(np.asarray(unmasked['psf_flux_bg_corr_all'])).all()
        assert np.isfinite(np.asarray(masked['psf_flux_bg_corr_all'])).all()

    def test_psf_cube_helper_replaces_nan_with_zero(self):
        from spherical.pipeline.steps.flux_psf_calibration import finalize_psf_cube

        cube = np.ones((2, 3, 57, 57))
        cube[:, :, :2, :] = np.nan
        out = finalize_psf_cube(cube, LOGGER)
        assert np.isfinite(out).all()
        assert (out[:, :, :2, :] == 0.0).all()
        assert (out[:, :, 2:, :] == 1.0).all()
