"""coronagraph_center_from_disk: prefer the measured centre, else no mask."""
from __future__ import annotations

import logging

import numpy as np
from astropy.io import fits

from spherical.pipeline.steps.flux_psf_calibration import coronagraph_center_from_disk

LOGGER = logging.getLogger("test")


class TestCoronagraphCenterFromDisk:
    def test_reads_the_robust_fit_when_present(self, tmp_path):
        c = np.zeros((3, 2, 2))
        c[..., 0], c[..., 1] = 127.38, 127.80
        fits.writeto(tmp_path / "image_centers_fitted_robust.fits", c)
        xy = coronagraph_center_from_disk(str(tmp_path), LOGGER)
        assert np.allclose(xy, (127.38, 127.80), atol=1e-2)

    def test_prefers_the_robust_fit_over_the_raw_centers(self, tmp_path):
        r = np.zeros((3, 2, 2))
        r[..., 0], r[..., 1] = 127.0, 128.0
        c = np.zeros((3, 2, 2))
        c[..., 0], c[..., 1] = 10.0, 20.0
        fits.writeto(tmp_path / "image_centers_fitted_robust.fits", r)
        fits.writeto(tmp_path / "image_centers.fits", c)
        assert np.allclose(coronagraph_center_from_disk(str(tmp_path), LOGGER),
                           (127.0, 128.0))

    def test_falls_back_to_the_raw_centers(self, tmp_path):
        c = np.zeros((3, 2, 2))
        c[..., 0], c[..., 1] = 120.0, 125.0
        fits.writeto(tmp_path / "image_centers.fits", c)
        assert np.allclose(coronagraph_center_from_disk(str(tmp_path), LOGGER),
                           (120.0, 125.0))

    def test_skips_an_all_nan_center_file(self, tmp_path):
        fits.writeto(tmp_path / "image_centers_fitted_robust.fits",
                     np.full((3, 2, 2), np.nan))
        c = np.zeros((3, 2, 2))
        c[..., 0], c[..., 1] = 120.0, 125.0
        fits.writeto(tmp_path / "image_centers.fits", c)
        assert np.allclose(coronagraph_center_from_disk(str(tmp_path), LOGGER),
                           (120.0, 125.0))

    def test_falls_back_to_the_irdis_nominal(self, tmp_path):
        """Nothing on disk, but the filter is known: use the CORO-frame nominal."""
        xy = coronagraph_center_from_disk(str(tmp_path), LOGGER,
                                          irdis_filter_comb="DB_H23")
        assert np.allclose(xy, (485.81, 523.54), atol=1e-2)

    def test_the_nominal_does_not_override_a_measured_centre(self, tmp_path):
        c = np.zeros((3, 2, 2))
        c[..., 0], c[..., 1] = 480.0, 520.0
        fits.writeto(tmp_path / "image_centers_fitted_robust.fits", c)
        assert np.allclose(
            coronagraph_center_from_disk(str(tmp_path), LOGGER,
                                         irdis_filter_comb="DB_H23"),
            (480.0, 520.0))

    def test_returns_none_when_nothing_is_on_disk(self, tmp_path):
        """IFS with no centre file: no nominal exists, so no mask."""
        assert coronagraph_center_from_disk(str(tmp_path), LOGGER) is None
