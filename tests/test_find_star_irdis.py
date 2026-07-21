"""Tests for the IRDIS branch of the shared find_centers step (Phase 5)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits


def _make_center_cube_file(tmp_path, shape=(2, 4, 60, 60), apply_crop=False, crop_offset=(0, 0)):
    cube = np.zeros(shape, dtype=np.float32)
    header = fits.Header()
    header["HIERARCH SPHERICAL CROP APPLIED"] = apply_crop
    if apply_crop:
        header["HIERARCH SPHERICAL CROP SIZE"] = shape[-1]
        header["HIERARCH SPHERICAL CROP X0 CH0"] = int(crop_offset[0])
        header["HIERARCH SPHERICAL CROP Y0 CH0"] = int(crop_offset[1])
        header["HIERARCH SPHERICAL CROP X0 CH1"] = int(crop_offset[0])
        header["HIERARCH SPHERICAL CROP Y0 CH1"] = int(crop_offset[1])
    path = tmp_path / "center_cube.fits"
    fits.writeto(path, cube, header, overwrite=True)
    return path


class TestFitCentersInParallelInstrumentDispatch:
    def test_ifs_path_passes_ifs_and_none_guess(self, tmp_path):
        from spherical.pipeline.steps import find_star

        _make_center_cube_file(tmp_path, shape=(39, 4, 60, 60))
        fits.writeto(tmp_path / "wavelengths.fits", np.linspace(1000, 1600, 39), overwrite=True)
        pd.DataFrame({
            "OCS WAFFLE ORIENT": ["+"] * 4,
            "INS COMB IFLT": ["OBS_YJ"] * 4,
        }).to_csv(tmp_path / "frames_info_center.csv", index=False)

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IFS"], "FILTER": ["OBS_YJ"]}
        observation.frames = {"CORO": None}

        with patch.object(find_star, "parallel_map_ordered", return_value=[]) as pm:
            try:
                find_star.fit_centers_in_parallel(str(tmp_path), observation, ncpu=1)
            except Exception:
                pass
            args_list = pm.call_args.kwargs.get("args_list") or pm.call_args.args[1]
            assert all(a[-2] == "IFS" for a in args_list)
            assert all(a[-1] is None for a in args_list)

    def test_irdis_path_passes_nominal_center_guess(self, tmp_path):
        from spherical.pipeline.steps import find_star
        from spherical.pipeline.steps.irdis_preprocess import NOMINAL_STAR_POSITIONS_K_BAND

        _make_center_cube_file(tmp_path)
        fits.writeto(tmp_path / "wavelengths.fits", np.array([2110.0, 2251.0]), overwrite=True)
        pd.DataFrame({
            "OCS WAFFLE ORIENT": ["+"] * 4,
            "INS COMB IFLT": ["DB_K12"] * 4,
        }).to_csv(tmp_path / "frames_info_center.csv", index=False)

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IRDIS"], "FILTER": ["DB_K12"]}
        observation.frames = {"CORO": None}

        with patch.object(find_star, "parallel_map_ordered", return_value=[]) as pm:
            try:
                find_star.fit_centers_in_parallel(str(tmp_path), observation, ncpu=1)
            except Exception:
                pass
            args_list = pm.call_args.kwargs.get("args_list") or pm.call_args.args[1]
            assert all(a[-2] == "IRDIS" for a in args_list)
            expected = np.array(NOMINAL_STAR_POSITIONS_K_BAND, dtype=np.float64)
            for a in args_list:
                cg = a[-1]
                assert cg is not None
                assert cg.shape == (2, 2)
                np.testing.assert_allclose(cg, expected, atol=1e-6)


class TestCropAwareNominalShift:
    def test_crop_offset_subtracted_from_nominal(self, tmp_path):
        from spherical.pipeline.steps import find_star
        from spherical.pipeline.steps.irdis_preprocess import NOMINAL_STAR_POSITIONS_K_BAND

        crop_offset = (455, 494)
        _make_center_cube_file(tmp_path, apply_crop=True, crop_offset=crop_offset)
        fits.writeto(tmp_path / "wavelengths.fits", np.array([2110.0, 2251.0]), overwrite=True)
        pd.DataFrame({
            "OCS WAFFLE ORIENT": ["+"] * 4,
            "INS COMB IFLT": ["DB_K12"] * 4,
        }).to_csv(tmp_path / "frames_info_center.csv", index=False)

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IRDIS"], "FILTER": ["DB_K12"]}
        observation.frames = {"CORO": None}

        with patch.object(find_star, "parallel_map_ordered", return_value=[]) as pm:
            try:
                find_star.fit_centers_in_parallel(str(tmp_path), observation, ncpu=1)
            except Exception:
                pass
            args_list = pm.call_args.kwargs.get("args_list") or pm.call_args.args[1]
            cg = args_list[0][-1]
            expected = np.array(NOMINAL_STAR_POSITIONS_K_BAND, dtype=np.float64).copy()
            expected[:, 0] -= crop_offset[0]
            expected[:, 1] -= crop_offset[1]
            np.testing.assert_allclose(cg, expected, atol=1e-6)


class TestMeasureCenterWaffleDispatchBug:
    def test_irdis_no_longer_falls_through_to_else_raise(self):
        """The paired-if bug at find_star.py:472-478 must be fixed so IRDIS
        does not raise 'Only IRDIS and IFS instruments known.'"""
        from spherical.pipeline.steps.find_star import measure_center_waffle

        cube = np.zeros((1, 1, 60, 60), dtype=np.float32)
        wavelengths = np.array([2110.0])
        center_guess = np.array([[30.0, 30.0]])

        with pytest.raises(Exception) as exc_info:
            measure_center_waffle(
                cube=cube,
                wavelengths=wavelengths,
                waffle_orientation="+",
                frames_info=None,
                bpm_cube=None,
                outputdir="/tmp",
                instrument="IRDIS",
                center_guess=center_guess,
                crop=False,
                crop_center=None,
                fit_background=True,
                fit_symmetric_gaussian=True,
                high_pass=False,
                save_plot=False,
                save_path=None,
            )
        assert "Only IRDIS and IFS" not in str(exc_info.value)


class TestCrossChannelOffset:
    def test_written_from_fitted_centers(self, tmp_path):
        """Empirical offset = nanmedian_t(image_centers[1] - image_centers[0])."""
        from spherical.pipeline.steps import find_star

        _make_center_cube_file(tmp_path)
        fits.writeto(tmp_path / "wavelengths.fits", np.array([2110.0, 2251.0]), overwrite=True)
        pd.DataFrame({
            "OCS WAFFLE ORIENT": ["+"] * 4,
            "INS COMB IFLT": ["DB_K12"] * 4,
        }).to_csv(tmp_path / "frames_info_center.csv", index=False)

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IRDIS"], "FILTER": ["DB_K12"]}
        observation.frames = {"CORO": None}

        # ch0 ≈ (100, 200), ch1 ≈ (101, 187) with a few NaNs to exercise nanmedian.
        image_centers_full = np.array([
            [[100.0, 200.0], [100.5, 200.2], [np.nan, np.nan], [99.8, 199.9]],
            [[101.0, 187.0], [101.2, 187.1], [100.9, 187.4], [np.nan, np.nan]],
        ], dtype=np.float32)
        # parallel_map_ordered returns a list of per-frame 4-tuples;
        # fit_centers_in_parallel concatenates on axis=1.
        fake_results = [
            (
                np.zeros((4, 1, 2), dtype=np.float32),          # spot_centers  (nspots, 1, 2)
                np.zeros((6, 1), dtype=np.float32),             # spot_distances (6, 1)
                image_centers_full[:, i:i + 1, :],              # image_centers  (2, 1, 2)
                np.zeros((2, 1), dtype=np.float32),             # spot_amplitudes (2, 1)
            )
            for i in range(4)
        ]
        with patch.object(find_star, "parallel_map_ordered", return_value=fake_results):
            find_star.fit_centers_in_parallel(str(tmp_path), observation, ncpu=1)

        offset_path = tmp_path / "additional_outputs" / "cross_channel_offset.fits"
        assert offset_path.exists()
        offset = fits.getdata(str(offset_path))
        assert offset.shape == (2,)
        # ch1 - ch0 per component, per time:
        #   x diffs: 1.0, 0.7, nan, nan  → nanmedian = 0.85
        #   y diffs: -13.0, -13.1, nan, nan → nanmedian = -13.05
        np.testing.assert_allclose(offset[0], 0.85, atol=1e-3)
        np.testing.assert_allclose(offset[1], -13.05, atol=1e-3)

    def test_preserved_on_second_run(self, tmp_path):
        from spherical.pipeline.steps import find_star

        _make_center_cube_file(tmp_path)
        fits.writeto(tmp_path / "wavelengths.fits", np.array([2110.0, 2251.0]), overwrite=True)
        pd.DataFrame({
            "OCS WAFFLE ORIENT": ["+"] * 4,
            "INS COMB IFLT": ["DB_K12"] * 4,
        }).to_csv(tmp_path / "frames_info_center.csv", index=False)

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IRDIS"], "FILTER": ["DB_K12"]}
        observation.frames = {"CORO": None}

        (tmp_path / "additional_outputs").mkdir(exist_ok=True)
        sentinel = np.array([99.0, -99.0], dtype=np.float32)
        fits.writeto(
            tmp_path / "additional_outputs" / "cross_channel_offset.fits",
            sentinel,
            overwrite=True,
        )

        image_centers_full = np.array([[[100.0, 200.0]], [[101.0, 187.0]]], dtype=np.float32)
        fake_results = [(
            np.zeros((4, 1, 2), dtype=np.float32),
            np.zeros((6, 1), dtype=np.float32),
            image_centers_full,
            np.zeros((2, 1), dtype=np.float32),
        )]
        with patch.object(find_star, "parallel_map_ordered", return_value=fake_results):
            find_star.fit_centers_in_parallel(str(tmp_path), observation, ncpu=1)

        offset = fits.getdata(
            str(tmp_path / "additional_outputs" / "cross_channel_offset.fits")
        )
        np.testing.assert_array_equal(offset, sentinel)

    def test_not_written_for_ifs(self, tmp_path):
        from spherical.pipeline.steps import find_star

        _make_center_cube_file(tmp_path, shape=(39, 4, 60, 60))
        fits.writeto(tmp_path / "wavelengths.fits", np.linspace(1000, 1600, 39), overwrite=True)
        pd.DataFrame({
            "OCS WAFFLE ORIENT": ["+"] * 4,
            "INS COMB IFLT": ["OBS_YJ"] * 4,
        }).to_csv(tmp_path / "frames_info_center.csv", index=False)

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IFS"], "FILTER": ["OBS_YJ"]}
        observation.frames = {"CORO": None}

        with patch.object(find_star, "parallel_map_ordered", return_value=[]):
            try:
                find_star.fit_centers_in_parallel(str(tmp_path), observation, ncpu=1)
            except Exception:
                pass

        assert not (tmp_path / "additional_outputs" / "cross_channel_offset.fits").exists()
