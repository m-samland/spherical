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


def _waffle_observation(coro_frames=None):
    """A continuous-waffle observation: CENTER frames are the science frames.

    ``WAFFLE_MODE`` — not the presence of CORO frames — is what selects the
    branch, so ``coro_frames`` is free to be non-empty (waffle sequences do
    pick up stray CORO exposures; see the dispatch test below).
    """
    obs = MagicMock()
    obs.observation = {"INSTRUMENT": ["IRDIS"], "FILTER": ["DB_K12"], "WAFFLE_MODE": [True]}
    obs.frames = {"CORO": coro_frames}
    return obs


class TestIRDISWaffleCenterFit:
    def test_writes_robust_output(self, tmp_path):
        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        _make_image_centers(tmp_path)
        observation = _waffle_observation()

        run_polynomial_center_fit(
            converted_dir=str(tmp_path),
            observation=observation,
            extraction_parameters={"method": "optext", "linear_wavelength": True},
            non_least_square_methods=["optext"],
        )
        robust = fits.getdata(str(tmp_path / "image_centers_fitted_robust.fits"))
        assert robust.shape == (2, 100, 2)
        # IRDIS writes image_centers_fitted.fits as the pre-outlier empirical
        # centers so plot_image_center_evolution can render (needs 3 files).
        # It is a copy of image_centers.fits, NOT a polynomial fit.
        assert (tmp_path / "image_centers_fitted.fits").exists()
        raw = fits.getdata(str(tmp_path / "image_centers.fits"))
        pre_outlier = fits.getdata(str(tmp_path / "image_centers_fitted.fits"))
        np.testing.assert_array_equal(pre_outlier, raw)

    def test_keeps_outliers_in_the_robust_file(self, tmp_path):
        """Outliers are flagged, not replaced: the jitter they sit on is real (#145)."""
        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        raw = _make_image_centers(tmp_path, outliers=(50,))
        observation = _waffle_observation()

        run_polynomial_center_fit(
            converted_dir=str(tmp_path),
            observation=observation,
            extraction_parameters={"method": "optext", "linear_wavelength": True},
            non_least_square_methods=["optext"],
        )
        robust = fits.getdata(str(tmp_path / "image_centers_fitted_robust.fits"))
        np.testing.assert_array_equal(robust, raw)

    def test_records_outlier_frame_indices(self, tmp_path):
        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        _make_image_centers(tmp_path, outliers=(50, 75))
        observation = _waffle_observation()

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


class TestIRDISBranchDispatch:
    """``WAFFLE_MODE`` selects the branch, not the presence of CORO frames.

    DMS propagation exists to place the star in CORO frames from the header
    dither offsets. A waffle sequence measures the star directly off the
    satellite spots in every CENTER frame, so it must never be propagated —
    even when the sequence also picked up a few stray CORO exposures, which
    27 IRDIS observations in the archive do.
    """

    def test_waffle_with_stray_coro_frames_uses_center_branch(self, tmp_path):
        """A waffle sequence with CORO frames present stays on the CENTER branch.

        Dispatching on ``frames["CORO"]`` instead sent these to DMS propagation,
        which wrote a CORO-length centers array against a CENTER-length raw
        array (issue #129).
        """
        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        _make_image_centers(tmp_path, n_time=17, outliers=())
        # No frames_info_*.csv on disk: the DMS branch would fail to find them.
        run_polynomial_center_fit(
            converted_dir=str(tmp_path),
            observation=_waffle_observation(coro_frames=[1, 2]),
            extraction_parameters={"method": "optext", "linear_wavelength": True},
            non_least_square_methods=["optext"],
        )
        robust = fits.getdata(str(tmp_path / "image_centers_fitted_robust.fits"))
        assert robust.shape[1] == 17, "centers should stay on the CENTER frame grid"

    def test_non_waffle_uses_dms_branch(self, tmp_path):
        """A coronagraphic sequence propagates onto the CORO frame grid."""
        import pandas as pd

        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        _make_image_centers(tmp_path, n_time=3, outliers=())
        pd.DataFrame({
            "MJD": [0.0, 0.5, 1.0],
            "INS1 PAC X": [0.0, 0.0, 0.0],
            "INS1 PAC Y": [0.0, 0.0, 0.0],
        }).to_csv(tmp_path / "frames_info_center.csv", index=False)
        pd.DataFrame({
            "MJD": [0.1, 0.2, 0.9, 1.1],
            "INS1 PAC X": [0.0, 0.0, 0.0, 0.0],
            "INS1 PAC Y": [0.0, 0.0, 0.0, 0.0],
        }).to_csv(tmp_path / "frames_info_coro.csv", index=False)

        observation = MagicMock()
        observation.observation = {
            "INSTRUMENT": ["IRDIS"],
            "FILTER": ["DB_K12"],
            "WAFFLE_MODE": [False],
        }
        observation.frames = {}

        run_polynomial_center_fit(
            converted_dir=str(tmp_path),
            observation=observation,
            extraction_parameters={"method": "optext", "linear_wavelength": True},
            non_least_square_methods=["optext"],
        )
        robust = fits.getdata(str(tmp_path / "image_centers_fitted_robust.fits"))
        assert robust.shape[1] == 4, "centers should land on the CORO frame grid"


class TestIRDISDmsPropagation:
    def _observation(self, filter_name="DB_K12"):
        """A normal coronagraphic sequence: CORO frames are the science frames.

        The DMS offsets in the CORO headers are what place the star, so this is
        the only branch where the propagation applies. Waffle sequences measure
        the centers directly from the satellite spots and need no propagation.
        """
        obs = MagicMock()
        obs.observation = {
            "INSTRUMENT": ["IRDIS"],
            "FILTER": [filter_name],
            "WAFFLE_MODE": [False],
        }
        obs.frames = {"CORO": [1, 2]}
        return obs

    def test_global_anchor_at_dms_zero(self, tmp_path):
        """CENTER frames all at PAC=0 → S₀ is the median of the measured centers;
        each CORO frame's propagated position = S₀ + PAC_coro/18."""
        import pandas as pd

        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        center_centers = np.zeros((2, 3, 2), dtype=np.float32)
        center_centers[0] = [[25.0, 30.0]] * 3
        center_centers[1] = [[28.0, 27.0]] * 3
        fits.writeto(tmp_path / "image_centers.fits", center_centers, overwrite=True)
        fits.writeto(tmp_path / "wavelengths.fits", np.array([2100.0, 2250.0]), overwrite=True)

        pd.DataFrame({
            "MJD": [0.0, 0.5, 1.0],
            "INS1 PAC X": [0.0, 0.0, 0.0],
            "INS1 PAC Y": [0.0, 0.0, 0.0],
        }).to_csv(tmp_path / "frames_info_center.csv", index=False)
        pd.DataFrame({
            "MJD": [0.1, 0.9],
            "INS1 PAC X": [18.0, -36.0],   # +1 px, -2 px in x
            "INS1 PAC Y": [-18.0, 0.0],    # -1 px,  0 px in y
        }).to_csv(tmp_path / "frames_info_coro.csv", index=False)

        run_polynomial_center_fit(
            converted_dir=str(tmp_path),
            observation=self._observation(),
            extraction_parameters={"method": "optext", "linear_wavelength": True},
            non_least_square_methods=["optext"],
        )
        robust = fits.getdata(str(tmp_path / "image_centers_fitted_robust.fits"))
        assert robust.shape == (2, 2, 2)
        # ch0 S₀ = (25, 30); ch1 S₀ = (28, 27); add per-CORO DMS.
        assert abs(robust[0, 0, 0] - (25.0 + 1.0)) < 1e-3
        assert abs(robust[0, 0, 1] - (30.0 - 1.0)) < 1e-3
        assert abs(robust[1, 1, 0] - (28.0 - 2.0)) < 1e-3
        assert abs(robust[1, 1, 1] - (27.0 + 0.0)) < 1e-3

    def test_center_pac_subtracted_from_anchor(self, tmp_path):
        """CENTER frames at non-zero PAC — S₀ = median(center − PAC/18), so a
        constant offset in the CENTER's PAC should NOT bias the propagation."""
        import pandas as pd

        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        # All CENTER frames at PAC=(+18, +18) µm → +1 px in each axis. Measured
        # centers are the star positions AT that dither, i.e. S₀ shifted by +1.
        center_centers = np.zeros((2, 2, 2), dtype=np.float32)
        center_centers[0] = [[25.0 + 1.0, 30.0 + 1.0]] * 2   # true S₀ ch0 = (25, 30)
        center_centers[1] = [[28.0 + 1.0, 27.0 + 1.0]] * 2   # true S₀ ch1 = (28, 27)
        fits.writeto(tmp_path / "image_centers.fits", center_centers, overwrite=True)
        fits.writeto(tmp_path / "wavelengths.fits", np.array([2100.0, 2250.0]), overwrite=True)

        pd.DataFrame({
            "MJD": [0.0, 1.0],
            "INS1 PAC X": [18.0, 18.0],
            "INS1 PAC Y": [18.0, 18.0],
        }).to_csv(tmp_path / "frames_info_center.csv", index=False)
        pd.DataFrame({
            "MJD": [0.5],
            "INS1 PAC X": [-36.0],
            "INS1 PAC Y": [-36.0],
        }).to_csv(tmp_path / "frames_info_coro.csv", index=False)

        run_polynomial_center_fit(
            converted_dir=str(tmp_path),
            observation=self._observation(),
            extraction_parameters={"method": "optext", "linear_wavelength": True},
            non_least_square_methods=["optext"],
        )
        robust = fits.getdata(str(tmp_path / "image_centers_fitted_robust.fits"))
        # ch0: S₀ = (25, 30); CORO PAC = (-2, -2) px → (23, 28).
        assert abs(robust[0, 0, 0] - 23.0) < 1e-3
        assert abs(robust[0, 0, 1] - 28.0) < 1e-3
        # ch1: S₀ = (28, 27); CORO PAC = (-2, -2) px → (26, 25).
        assert abs(robust[1, 0, 0] - 26.0) < 1e-3
        assert abs(robust[1, 0, 1] - 25.0) < 1e-3

    def test_fallback_to_nominal_when_all_nan_channel(self, tmp_path):
        """If all CENTER fits failed for one channel, fall back to the
        filter's nominal star position (from find_star.nominal_star_positions)."""
        import pandas as pd

        from spherical.pipeline.steps.find_star import nominal_star_positions
        from spherical.pipeline.steps.process_centers import run_polynomial_center_fit

        center_centers = np.zeros((2, 2, 2), dtype=np.float32)
        center_centers[0] = [[25.0, 30.0]] * 2
        center_centers[1] = np.nan  # ch1 all-NaN
        fits.writeto(tmp_path / "image_centers.fits", center_centers, overwrite=True)
        fits.writeto(tmp_path / "wavelengths.fits", np.array([2100.0, 2250.0]), overwrite=True)

        pd.DataFrame({
            "MJD": [0.0, 1.0],
            "INS1 PAC X": [0.0, 0.0],
            "INS1 PAC Y": [0.0, 0.0],
        }).to_csv(tmp_path / "frames_info_center.csv", index=False)
        pd.DataFrame({
            "MJD": [0.5],
            "INS1 PAC X": [0.0],
            "INS1 PAC Y": [0.0],
        }).to_csv(tmp_path / "frames_info_coro.csv", index=False)

        run_polynomial_center_fit(
            converted_dir=str(tmp_path),
            observation=self._observation("DB_K12"),
            extraction_parameters={"method": "optext", "linear_wavelength": True},
            non_least_square_methods=["optext"],
        )
        robust = fits.getdata(str(tmp_path / "image_centers_fitted_robust.fits"))
        nominal = nominal_star_positions("DB_K12")
        # ch0 anchored from data:
        assert abs(robust[0, 0, 0] - 25.0) < 1e-3
        assert abs(robust[0, 0, 1] - 30.0) < 1e-3
        # ch1 falls back to nominal:
        assert abs(robust[1, 0, 0] - nominal[1, 0]) < 1e-3
        assert abs(robust[1, 0, 1] - nominal[1, 1]) < 1e-3
