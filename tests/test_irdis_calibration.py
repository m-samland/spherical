"""Tests for the IRDIS calibration step (Phase 3)."""
from __future__ import annotations

import numpy as np
import pytest

from spherical.pipeline.steps.irdis_calibration import (
    DEAD_COL_SLICE_LEFT_HALF_EDGE,
    DEAD_COL_SLICE_MID,
    DEAD_COL_SLICE_RIGHT_HALF_EDGE,
    DEAD_ROW_SLICE_BOTTOM,
    DEAD_ROW_SLICE_TOP,
    DETECTOR_COLS,
    DETECTOR_ROWS,
    HALF_COLS,
    dead_region_mask,
    split_detector_cube,
    split_detector_image,
)


class TestDetectorGeometry:
    def test_constants(self):
        assert DETECTOR_ROWS == 1024
        assert DETECTOR_COLS == 2048
        assert HALF_COLS == 1024
        assert DEAD_ROW_SLICE_BOTTOM == slice(0, 15)
        assert DEAD_ROW_SLICE_TOP == slice(1013, 1024)
        assert DEAD_COL_SLICE_LEFT_HALF_EDGE == slice(0, 50)
        assert DEAD_COL_SLICE_MID == slice(941, 1078)
        assert DEAD_COL_SLICE_RIGHT_HALF_EDGE == slice(1966, 2048)


class TestSplitDetector:
    def test_split_image_shape(self):
        img = np.arange(1024 * 2048).reshape(1024, 2048).astype(np.float32)
        split = split_detector_image(img)
        assert split.shape == (2, 1024, 1024)

    def test_split_image_preserves_values(self):
        img = np.arange(1024 * 2048).reshape(1024, 2048).astype(np.float32)
        split = split_detector_image(img)
        np.testing.assert_array_equal(split[0], img[:, 0:1024])
        np.testing.assert_array_equal(split[1], img[:, 1024:2048])

    def test_split_cube_shape(self):
        cube = np.zeros((5, 1024, 2048), dtype=np.float32)
        split = split_detector_cube(cube)
        assert split.shape == (5, 2, 1024, 1024)

    def test_split_cube_preserves_values(self):
        cube = np.random.default_rng(42).random((3, 1024, 2048)).astype(np.float32)
        split = split_detector_cube(cube)
        np.testing.assert_array_equal(split[:, 0], cube[:, :, 0:1024])
        np.testing.assert_array_equal(split[:, 1], cube[:, :, 1024:2048])

    def test_split_image_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            split_detector_image(np.zeros((100, 100)))


class TestDeadRegionMask:
    def test_shape_and_dtype(self):
        mask = dead_region_mask()
        assert mask.shape == (2, 1024, 1024)
        assert mask.dtype == np.bool_

    def test_top_and_bottom_row_bands_flagged_on_both_halves(self):
        mask = dead_region_mask()
        assert mask[:, DEAD_ROW_SLICE_BOTTOM, :].all()
        assert mask[:, DEAD_ROW_SLICE_TOP, :].all()

    def test_left_half_edge_flagged(self):
        mask = dead_region_mask()
        assert mask[0, :, DEAD_COL_SLICE_LEFT_HALF_EDGE].all()

    def test_right_half_edge_flagged(self):
        mask = dead_region_mask()
        # DEAD_COL_SLICE_RIGHT_HALF_EDGE is 1966..2048 in RAW coords,
        # which is 942..1024 in per-half coords.
        assert mask[1, :, 942:1024].all()

    def test_mid_band_flagged_across_seam(self):
        mask = dead_region_mask()
        # DEAD_COL_SLICE_MID = 941..1078 in RAW coords.
        # Channel 0: raw cols 941..1023 → per-half 941..1023.
        # Channel 1: raw cols 1024..1077 → per-half 0..53 (exclusive of 54).
        assert mask[0, :, 941:1024].all()
        assert mask[1, :, 0:54].all()

    def test_center_of_frame_not_flagged(self):
        mask = dead_region_mask()
        # A pixel in the middle of the left half, well away from any dead band.
        assert not mask[0, 512, 512]
        # A pixel in the middle of the right half.
        assert not mask[1, 512, 512]


class TestBuildMasterBackground:
    def _write_bg_fits(self, tmp_path, values, name="bg.fits"):
        from astropy.io import fits
        path = tmp_path / name
        fits.writeto(path, values.astype(np.float32))
        return str(path)

    def test_returns_split_shape_float32(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_master_background

        bg1 = np.full((1, 1024, 2048), 100.0)
        p1 = self._write_bg_fits(tmp_path, bg1, "bg1.fits")
        bg2 = np.full((1, 1024, 2048), 110.0)
        p2 = self._write_bg_fits(tmp_path, bg2, "bg2.fits")

        result = build_master_background([p1, p2], MagicMock(), logger=MagicMock())
        assert result.shape == (2, 1024, 1024)
        assert result.dtype == np.float32

    def test_sigma_clipped_mean_matches_mean_when_no_outliers(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_master_background

        arr1 = np.full((1, 1024, 2048), 100.0)
        arr2 = np.full((1, 1024, 2048), 110.0)
        p1 = self._write_bg_fits(tmp_path, arr1, "b1.fits")
        p2 = self._write_bg_fits(tmp_path, arr2, "b2.fits")

        result = build_master_background([p1, p2], MagicMock(), logger=MagicMock())
        # Non-dead pixel in centre of left half.
        assert result[0, 512, 512] == pytest.approx(105.0, rel=1e-4)

    def test_dead_regions_are_nan(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_master_background

        arr = np.full((1, 1024, 2048), 100.0)
        p = self._write_bg_fits(tmp_path, arr, "b.fits")

        result = build_master_background([p], MagicMock(), logger=MagicMock())
        # Bottom row band.
        assert np.isnan(result[0, 0, 512])
        assert np.isnan(result[1, 0, 512])
        # Top row band.
        assert np.isnan(result[0, 1023, 512])
        # Left-half edge.
        assert np.isnan(result[0, 512, 10])

    def test_handles_2d_frames_without_leading_axis(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_master_background

        arr_2d = np.full((1024, 2048), 100.0)
        p = self._write_bg_fits(tmp_path, arr_2d, "b.fits")

        result = build_master_background([p], MagicMock(), logger=MagicMock())
        assert result.shape == (2, 1024, 1024)
        assert result[0, 512, 512] == pytest.approx(100.0)

    def test_handles_multi_ndit_cubes(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_master_background

        arr = np.stack([np.full((1024, 2048), 100.0),
                        np.full((1024, 2048), 110.0),
                        np.full((1024, 2048), 105.0)])  # (3, 1024, 2048)
        p = self._write_bg_fits(tmp_path, arr, "b.fits")

        result = build_master_background([p], MagicMock(), logger=MagicMock())
        # Sigma-clipped mean of {100, 110, 105} ≈ 105.
        assert result[0, 512, 512] == pytest.approx(105.0, abs=1.0)

    def test_logs_start_and_completion(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_master_background

        arr = np.full((1, 1024, 2048), 100.0)
        p = self._write_bg_fits(tmp_path, arr, "b.fits")

        logger = MagicMock()
        build_master_background([p], MagicMock(), logger=logger)
        assert logger.info.called


class TestEstimateSmoothIllumination:
    def test_uniform_input_returns_uniform_output(self):
        from spherical.pipeline.steps.irdis_calibration import estimate_smooth_illumination

        img = np.full((1024, 1024), 100.0, dtype=np.float32)
        model = estimate_smooth_illumination(img, block_size=64)
        assert model.shape == img.shape
        # Interior of the image should be ≈100 everywhere.
        np.testing.assert_allclose(model[100:900, 100:900], 100.0, rtol=1e-3)

    def test_recovers_smooth_gradient(self):
        from spherical.pipeline.steps.irdis_calibration import estimate_smooth_illumination

        y, x = np.mgrid[:1024, :1024].astype(np.float32)
        r2 = (x - 512) ** 2 + (y - 512) ** 2
        img = 100.0 - r2 / 40000.0
        model = estimate_smooth_illumination(img, block_size=64)
        np.testing.assert_allclose(model[100:900, 100:900], img[100:900, 100:900], rtol=5e-3)

    def test_robust_to_bad_pixels(self):
        from spherical.pipeline.steps.irdis_calibration import estimate_smooth_illumination

        img = np.full((1024, 1024), 100.0, dtype=np.float32)
        rng = np.random.default_rng(0)
        img[rng.integers(0, 1024, 200), rng.integers(0, 1024, 200)] = np.nan
        img[rng.integers(0, 1024, 200), rng.integers(0, 1024, 200)] = 100000.0
        model = estimate_smooth_illumination(img, block_size=64)
        assert abs(np.nanmedian(model[100:900, 100:900]) - 100.0) < 5.0

    def test_fast_enough(self):
        import time

        from spherical.pipeline.steps.irdis_calibration import estimate_smooth_illumination

        img = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32) + 100
        t0 = time.perf_counter()
        _ = estimate_smooth_illumination(img, block_size=64)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Target < 100 ms; guard band at 500 ms.
        assert elapsed_ms < 500, f"took {elapsed_ms:.1f} ms"


class TestBuildMasterFlat:
    def _make_flat_files(self, tmp_path, dit_values, slope_per_pixel=100.0, bias=200.0):
        from astropy.io import fits
        paths = []
        for i, dit in enumerate(dit_values):
            frame = np.full((1024, 2048), slope_per_pixel * dit + bias, dtype=np.float32)
            path = tmp_path / f"flat_dit{dit}_{i}.fits"
            fits.writeto(path, frame[np.newaxis, ...])
            paths.append(str(path))
        return paths

    def test_linear_fit_recovers_slope_when_multiple_dits(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_master_flat

        dits = [1.0, 2.0, 3.0, 4.0, 5.0]
        paths = self._make_flat_files(tmp_path, dits, slope_per_pixel=100.0, bias=200.0)
        master_bg = np.zeros((2, 1024, 1024), dtype=np.float32)

        flat, response = build_master_flat(paths, dits, master_bg, MagicMock(), logger=MagicMock())
        # For a spatially uniform slope, after illumination detrending the
        # flat is 1.0 everywhere in the illuminated region.
        assert flat.shape == (2, 1024, 1024)
        assert response.shape == (2, 1024, 1024)
        assert flat[0, 512, 512] == pytest.approx(1.0, rel=1e-3)
        assert response[0, 512, 512] == pytest.approx(100.0, rel=1e-4)

    def test_falls_back_to_bg_subtracted_median_when_single_dit(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_master_flat

        dits = [1.0, 1.0, 1.0]
        paths = self._make_flat_files(tmp_path, dits, slope_per_pixel=100.0, bias=200.0)
        master_bg = np.full((2, 1024, 1024), 200.0, dtype=np.float32)

        flat, response = build_master_flat(paths, dits, master_bg, MagicMock(), logger=MagicMock())
        assert flat[0, 512, 512] == pytest.approx(1.0, rel=1e-3)
        assert response[0, 512, 512] == pytest.approx(100.0, rel=1e-4)

    def test_dead_regions_are_nan(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_master_flat

        paths = self._make_flat_files(tmp_path, [1.0, 2.0])
        master_bg = np.zeros((2, 1024, 1024), dtype=np.float32)
        flat, response = build_master_flat(paths, [1.0, 2.0], master_bg, MagicMock(), logger=MagicMock())
        assert np.isnan(flat[0, 0, 512])
        assert np.isnan(response[0, 0, 512])
        assert np.isnan(flat[1, 1023, 512])

    def test_dtype_float32(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_master_flat

        paths = self._make_flat_files(tmp_path, [1.0, 2.0])
        master_bg = np.zeros((2, 1024, 1024), dtype=np.float32)
        flat, response = build_master_flat(paths, [1.0, 2.0], master_bg, MagicMock(), logger=MagicMock())
        assert flat.dtype == np.float32
        assert response.dtype == np.float32

    def test_illumination_gradient_is_removed(self, tmp_path):
        """A synthetic FLAT with a radial vignetting pattern should produce a
        detrended flat that is closer to unity in the illuminated region than
        the raw response would be."""
        from unittest.mock import MagicMock

        from astropy.io import fits

        from spherical.pipeline.steps.irdis_calibration import build_master_flat

        y, x = np.mgrid[:1024, :2048].astype(np.float32)
        # Radial peak per half.
        r2_left = (x - 512) ** 2 + (y - 512) ** 2
        r2_right = (x - 1536) ** 2 + (y - 512) ** 2
        illum = np.where(x < 1024, 1.0 - r2_left / 800000.0, 1.0 - r2_right / 800000.0)
        illum = np.clip(illum, 0.3, 1.0).astype(np.float32)

        paths = []
        for i, dit in enumerate([1.0, 2.0, 3.0]):
            frame = 100.0 * dit * illum + 200.0
            path = tmp_path / f"flat_{i}.fits"
            fits.writeto(path, frame[np.newaxis, ...].astype(np.float32))
            paths.append(str(path))

        master_bg = np.zeros((2, 1024, 1024), dtype=np.float32)
        flat, _ = build_master_flat(paths, [1.0, 2.0, 3.0], master_bg, MagicMock(), logger=MagicMock())

        # In the illuminated centre of each half the detrended flat should be
        # tightly concentrated around 1.0 (the illumination has been removed).
        for ch in range(2):
            centre = flat[ch, 200:800, 200:800]
            centre_finite = centre[np.isfinite(centre)]
            assert abs(float(np.median(centre_finite)) - 1.0) < 0.01
            assert float(np.std(centre_finite)) < 0.02, (
                f"channel {ch}: std={float(np.std(centre_finite)):.4f} — illumination "
                "gradient not removed"
            )


class TestBuildBadPixelMap:
    """Tests for build_bad_pixel_map. Note: first argument is the NORMALIZED
    ``master_flat`` (median ≈ 1 per half after illumination detrending),
    not the raw response — so that vignetting doesn't cause legitimate
    edge pixels to be flagged as sigma-outliers."""

    def _uniform_flat(self, value=1.0):
        return np.full((2, 1024, 1024), value, dtype=np.float32)

    def _uniform_bg(self, value=50.0, n_frames=3):
        return np.full((n_frames, 2, 1024, 1024), value, dtype=np.float32)

    def _cfg(self):
        from unittest.mock import MagicMock
        return MagicMock(
            flat_badpix_sigma=5.0,
            background_badpix_sigma=5.0,
            flat_relative_response_min=0.5,
            flat_relative_response_max=1.5,
        )

    def test_uniform_inputs_no_bad_pixels(self):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_bad_pixel_map

        flat = self._uniform_flat()
        bg_stack = self._uniform_bg()
        master_bg = bg_stack.mean(axis=0)

        bpm = build_bad_pixel_map(flat, bg_stack, master_bg, self._cfg(), logger=MagicMock())
        assert not bpm[0, 512, 512]
        assert not bpm[1, 512, 512]

    def test_low_response_pixel_flagged(self):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_bad_pixel_map

        flat = self._uniform_flat(1.0)
        flat[0, 500, 500] = 0.1  # 10% of nominal → below hard floor.
        bg_stack = self._uniform_bg()
        master_bg = bg_stack.mean(axis=0)

        bpm = build_bad_pixel_map(flat, bg_stack, master_bg, self._cfg(), logger=MagicMock())
        assert bpm[0, 500, 500]

    def test_hot_pixel_in_background_flagged(self):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_bad_pixel_map

        flat = self._uniform_flat(1.0)
        master_bg = np.full((2, 1024, 1024), 50.0, dtype=np.float32)
        master_bg[1, 300, 300] = 50000.0
        bg_stack = np.stack([master_bg, master_bg, master_bg])

        bpm = build_bad_pixel_map(flat, bg_stack, master_bg, self._cfg(), logger=MagicMock())
        assert bpm[1, 300, 300]

    def test_dead_regions_are_false(self):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import (
            build_bad_pixel_map,
            dead_region_mask,
        )

        flat = np.full((2, 1024, 1024), np.nan, dtype=np.float32)
        flat[:, 20:1000, 100:900] = 1.0
        bg_stack = self._uniform_bg()
        master_bg = bg_stack.mean(axis=0)
        master_bg[dead_region_mask()] = np.nan

        bpm = build_bad_pixel_map(flat, bg_stack, master_bg, self._cfg(), logger=MagicMock())
        # Never flag pixels inside dead regions.
        assert not bpm[0, 0, 512]
        assert not bpm[0, 1023, 512]

    def test_shape_and_dtype(self):
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import build_bad_pixel_map

        flat = self._uniform_flat()
        bg_stack = self._uniform_bg()
        master_bg = bg_stack.mean(axis=0)
        bpm = build_bad_pixel_map(flat, bg_stack, master_bg, self._cfg(), logger=MagicMock())
        assert bpm.shape == (2, 1024, 1024)
        assert bpm.dtype == np.bool_

    def test_vignetted_background_does_not_flag_edge_pixels(self):
        """Regression: previously the hot-pixel test used the raw background
        directly, so the bright arc on a K-band detector half caused a huge
        fraction of edge pixels to be flagged as "hot". After the fix the
        test operates on the residual after subtracting the smooth
        illumination model, so a purely-smooth vignetted background
        produces no flags."""
        from unittest.mock import MagicMock

        from spherical.pipeline.steps.irdis_calibration import (
            build_bad_pixel_map,
            dead_region_mask,
        )

        # Smooth Gaussian-like vignetted background (radial peak at half-center)
        # + realistic photon noise. Without noise the mad_std of the residual
        # collapses to numerical dust and everything gets flagged as an
        # outlier; real data has non-zero noise that bounds the sigma test.
        rng = np.random.default_rng(0)
        y, x = np.mgrid[:1024, :1024].astype(np.float32)
        r2 = (x - 512) ** 2 + (y - 512) ** 2
        bg_signal = 20.0 + 80.0 * np.exp(-r2 / (2 * 300.0 ** 2))
        bg_half_ch0 = (bg_signal + rng.normal(0, 2.0, bg_signal.shape)).astype(np.float32)
        bg_half_ch1 = (bg_signal + rng.normal(0, 2.0, bg_signal.shape)).astype(np.float32)
        master_bg = np.stack([bg_half_ch0, bg_half_ch1])
        master_bg[dead_region_mask()] = np.nan
        bg_stack = np.stack([
            master_bg + rng.normal(0, 2.0, master_bg.shape).astype(np.float32),
            master_bg + rng.normal(0, 2.0, master_bg.shape).astype(np.float32),
            master_bg + rng.normal(0, 2.0, master_bg.shape).astype(np.float32),
        ])

        # Add faint pixel-scale variation to the flat too so the flat sigma
        # test operates on a non-degenerate distribution.
        flat = self._uniform_flat(1.0) + rng.normal(0, 0.01, (2, 1024, 1024)).astype(np.float32)
        flat[dead_region_mask()] = np.nan

        bpm = build_bad_pixel_map(flat, bg_stack, master_bg, self._cfg(), logger=MagicMock())
        dead = dead_region_mask()
        for ch in range(2):
            valid = ~dead[ch]
            frac = bpm[ch].sum() / valid.sum()
            assert frac < 0.02, f"channel {ch}: {100*frac:.2f}% flagged on smooth vignetted bg"


class TestRunIRDISCalibration:
    def _write_full_dataset(self, tmp_path):
        """Write synthetic FLAT (5 DITs) + BG_SCIENCE (2 frames) FITS files."""
        from astropy.io import fits

        flat_dir = tmp_path / "flats"
        flat_dir.mkdir()
        flat_paths = []
        for i, dit in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            frame = np.full((1024, 2048), 100.0 * dit + 200.0, dtype=np.float32)
            path = flat_dir / f"flat_{i}.fits"
            fits.writeto(path, frame[np.newaxis, ...], header=fits.Header({"EXPTIME": dit}))
            flat_paths.append(str(path))

        bg_dir = tmp_path / "bgs"
        bg_dir.mkdir()
        bg_paths = []
        for i in range(2):
            frame = np.full((1024, 2048), 50.0, dtype=np.float32)
            path = bg_dir / f"bg_{i}.fits"
            fits.writeto(path, frame[np.newaxis, ...], header=fits.Header({"EXPTIME": 4.0}))
            bg_paths.append(str(path))

        return flat_paths, bg_paths

    def _make_observation(self, flat_paths, bg_paths):
        from types import SimpleNamespace

        from astropy.table import Table
        return SimpleNamespace(
            frames={
                "FLAT": Table({"FILE": flat_paths}),
                "BG_SCIENCE": Table({"FILE": bg_paths}),
            }
        )

    def test_writes_three_outputs(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.pipeline_config import IRDISCalibrationConfig
        from spherical.pipeline.steps.irdis_calibration import run_irdis_calibration

        flat_paths, bg_paths = self._write_full_dataset(tmp_path)
        observation = self._make_observation(flat_paths, bg_paths)
        outdir = tmp_path / "calib"

        run_irdis_calibration(observation, IRDISCalibrationConfig(), outdir, MagicMock())
        assert (outdir / "master_flat.fits").exists()
        assert (outdir / "master_background.fits").exists()
        assert (outdir / "badpixel_map.fits").exists()

    def test_outputs_have_split_shape(self, tmp_path):
        from unittest.mock import MagicMock

        from astropy.io import fits

        from spherical.pipeline.pipeline_config import IRDISCalibrationConfig
        from spherical.pipeline.steps.irdis_calibration import run_irdis_calibration

        flat_paths, bg_paths = self._write_full_dataset(tmp_path)
        observation = self._make_observation(flat_paths, bg_paths)
        outdir = tmp_path / "calib"

        run_irdis_calibration(observation, IRDISCalibrationConfig(), outdir, MagicMock())
        assert fits.getdata(outdir / "master_flat.fits").shape == (2, 1024, 1024)
        assert fits.getdata(outdir / "master_background.fits").shape == (2, 1024, 1024)
        assert fits.getdata(outdir / "badpixel_map.fits").shape == (2, 1024, 1024)

    def test_skips_when_outputs_exist_and_not_forced(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from spherical.pipeline.pipeline_config import IRDISCalibrationConfig
        from spherical.pipeline.steps.irdis_calibration import run_irdis_calibration

        flat_paths, bg_paths = self._write_full_dataset(tmp_path)
        observation = self._make_observation(flat_paths, bg_paths)
        outdir = tmp_path / "calib"
        outdir.mkdir()
        for name in ("master_flat.fits", "master_background.fits", "badpixel_map.fits"):
            (outdir / name).write_text("stub")

        logger = MagicMock()
        with patch("spherical.pipeline.steps.irdis_calibration.build_master_background") as bg:
            run_irdis_calibration(observation, IRDISCalibrationConfig(), outdir, logger)
        bg.assert_not_called()
        skip_calls = [c for c in logger.info.call_args_list
                      if c.kwargs.get("extra", {}).get("status") == "skipped_complete"]
        assert skip_calls

    def test_force_true_recomputes_even_when_outputs_exist(self, tmp_path):
        from unittest.mock import MagicMock

        from astropy.io import fits

        from spherical.pipeline.pipeline_config import IRDISCalibrationConfig
        from spherical.pipeline.steps.irdis_calibration import run_irdis_calibration

        flat_paths, bg_paths = self._write_full_dataset(tmp_path)
        observation = self._make_observation(flat_paths, bg_paths)
        outdir = tmp_path / "calib"
        outdir.mkdir()
        for name in ("master_flat.fits", "master_background.fits", "badpixel_map.fits"):
            (outdir / name).write_text("stub")

        run_irdis_calibration(observation, IRDISCalibrationConfig(), outdir, MagicMock(), force=True)
        data = fits.getdata(outdir / "master_flat.fits")
        assert data.shape == (2, 1024, 1024)
