"""Tests for the IRDIS branch of process_extracted_centers (Phase 5)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from astropy.io import fits


def _make_image_centers(tmp_path, n_time=100, jitter_amplitude=0.05, outliers=(50,)):
    rng = np.random.default_rng(0)
    centers = np.zeros((2, n_time, 2), dtype=np.float32)
    centers[0, :, 0] = 25.0 + rng.normal(0, jitter_amplitude, n_time)
    centers[0, :, 1] = 30.0 + rng.normal(0, jitter_amplitude, n_time)
    centers[1, :, 0] = 28.0 + rng.normal(0, jitter_amplitude, n_time)
    centers[1, :, 1] = 27.0 + rng.normal(0, jitter_amplitude, n_time)
    for t in outliers:
        centers[0, t, 0] += 30.0  # large outlier in ch0 x
        centers[1, t, 1] += 40.0  # large outlier in ch1 y
    fits.writeto(tmp_path / "image_centers.fits", centers, overwrite=True)
    fits.writeto(tmp_path / "wavelengths.fits", np.array([2100.0, 2250.0]), overwrite=True)
    return centers


class TestIRDISWaffleCenterFit:
    def test_writes_robust_output(self, tmp_path):
        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        _make_image_centers(tmp_path)
        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IRDIS"]}
        observation.frames = {"CORO": None}

        run_polynomial_center_fit(
            converted_dir=str(tmp_path),
            observation=observation,
            extraction_parameters={"method": "optext", "linear_wavelength": True},
            non_least_square_methods=["optext"],
        )
        robust = fits.getdata(str(tmp_path / "image_centers_fitted_robust.fits"))
        assert robust.shape == (2, 100, 2)
        # No polynomial file should be written for IRDIS.
        assert not (tmp_path / "image_centers_fitted.fits").exists()

    def test_replaces_outliers_with_local_median(self, tmp_path):
        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        _make_image_centers(tmp_path, outliers=(50,))
        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IRDIS"]}
        observation.frames = {"CORO": None}

        run_polynomial_center_fit(
            converted_dir=str(tmp_path),
            observation=observation,
            extraction_parameters={"method": "optext", "linear_wavelength": True},
            non_least_square_methods=["optext"],
        )
        robust = fits.getdata(str(tmp_path / "image_centers_fitted_robust.fits"))
        # Outlier frame 50 in ch0 x should be pulled back to ~25.
        assert abs(robust[0, 50, 0] - 25.0) < 1.0
        assert abs(robust[1, 50, 1] - 27.0) < 1.0

    def test_records_outlier_frame_indices(self, tmp_path):
        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        _make_image_centers(tmp_path, outliers=(50, 75))
        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IRDIS"]}
        observation.frames = {"CORO": None}

        run_polynomial_center_fit(
            converted_dir=str(tmp_path),
            observation=observation,
            extraction_parameters={"method": "optext", "linear_wavelength": True},
            non_least_square_methods=["optext"],
        )
        outliers = fits.getdata(str(tmp_path / "additional_outputs" / "center_outlier_frames.fits"))
        assert outliers.shape[0] == 2  # per channel
        # Frame 50 flagged in ch0 (x-outlier); frame 75 in ch1 (y-outlier).
        assert 50 in outliers[0]
        assert 75 in outliers[1]

    def test_ifs_branch_unchanged(self, tmp_path):
        """Regression: IFS still writes both fitted files."""
        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        n_wave, n_time = 39, 5
        centers = np.zeros((n_wave, n_time, 2), dtype=np.float32) + 128.0
        fits.writeto(tmp_path / "image_centers.fits", centers, overwrite=True)
        fits.writeto(tmp_path / "wavelengths.fits", np.linspace(1000, 1600, n_wave), overwrite=True)

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IFS"]}
        observation.frames = {}

        run_polynomial_center_fit(
            converted_dir=str(tmp_path),
            observation=observation,
            extraction_parameters={"method": "optext", "linear_wavelength": True},
            non_least_square_methods=["optext"],
        )
        assert (tmp_path / "image_centers_fitted.fits").exists()
        assert (tmp_path / "image_centers_fitted_robust.fits").exists()
