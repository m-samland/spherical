"""Tests for optional frame alignment (#161)."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from spherical.pipeline import imutils


class TestImutilsShiftRepair:
    def test_sequence_shift_value_works(self):
        frame = np.zeros((16, 16))
        frame[8, 8] = 1.0
        shifted = imutils.shift(frame, (2.0, 0.0), method="fft")
        assert shifted.shape == frame.shape
        np.testing.assert_allclose(shifted.sum(), 1.0, atol=1e-9)

    def test_scalar_shift_value_still_works(self):
        frame = np.zeros((16, 16))
        frame[8, 8] = 1.0
        shifted = imutils.shift(frame, 1.0, method="fft")
        np.testing.assert_allclose(shifted.sum(), 1.0, atol=1e-9)

    def test_integer_shift_moves_the_peak_in_x(self):
        frame = np.zeros((16, 16))
        frame[8, 8] = 1.0
        shifted = imutils.shift(frame, (3.0, 0.0), method="fft")
        peak = np.unravel_index(np.argmax(shifted), shifted.shape)
        assert peak == (8, 11)

    def test_integer_shift_moves_the_peak_in_y(self):
        frame = np.zeros((16, 16))
        frame[8, 8] = 1.0
        shifted = imutils.shift(frame, (0.0, -2.0), method="fft")
        peak = np.unravel_index(np.argmax(shifted), shifted.shape)
        assert peak == (6, 8)

    def test_non_square_fft_shift_raises(self):
        frame = np.zeros((16, 32))
        with pytest.raises(ValueError, match="square"):
            imutils._shift_fft(frame, np.array([1.0, 1.0]))

    def test_square_fft_shift_does_not_raise(self):
        frame = np.zeros((16, 16))
        imutils._shift_fft(frame, np.array([1.0, 1.0]))

    @pytest.mark.parametrize("n", [16, 17, 65])
    def test_integer_shift_is_exact_at_both_parities(self, n):
        """The ramp must be un-shifted with ifftshift, which differs for odd n."""
        frame = np.zeros((n, n))
        frame[n // 2, n // 2] = 1.0
        shifted = imutils.shift(frame, (3.0, -2.0), method="fft")
        peak = np.unravel_index(np.argmax(shifted), shifted.shape)
        assert peak == (n // 2 - 2, n // 2 + 3)
        assert shifted.max() == pytest.approx(1.0, abs=1e-9)

    def test_odd_square_frame_is_accepted(self):
        frame = np.zeros((17, 17))
        imutils.shift(frame, (1.5, 0.0), method="fft")

    def test_non_square_shift_still_rejected(self):
        with pytest.raises(ValueError, match="square"):
            imutils.shift(np.zeros((16, 32)), (1.5, 0.0), method="fft")


from spherical.pipeline.steps.align_frames import (  # noqa: E402
    DEFAULT_PAD_WIDTH,
    pad_to_odd,
    shift_frame,
    shift_to_target,
)


def _gaussian(n, cx, cy, sigma=2.0):
    yy, xx = np.mgrid[:n, :n]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def _centroid(frame):
    finite = np.where(np.isfinite(frame), frame, 0.0)
    yy, xx = np.mgrid[: frame.shape[0], : frame.shape[1]]
    total = finite.sum()
    return (float((xx * finite).sum() / total), float((yy * finite).sum() / total))


class TestPadToOdd:
    def test_even_axis_padded_at_the_high_edge(self):
        cube = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
        out = pad_to_odd(cube)
        assert out.shape == (2, 3, 5, 5)
        np.testing.assert_array_equal(out[..., :4, :4], cube)
        np.testing.assert_array_equal(out[..., 4, :4], cube[..., 3, :4])
        np.testing.assert_array_equal(out[..., :4, 4], cube[..., :4, 3])

    def test_odd_axis_untouched(self):
        cube = np.zeros((2, 3, 5, 5), dtype=np.float32)
        out = pad_to_odd(cube)
        assert out.shape == cube.shape
        assert out is cube

    def test_ifs_262_becomes_263_with_the_same_target_pixel(self):
        cube = np.zeros((39, 2, 262, 262), dtype=np.float32)
        out = pad_to_odd(cube)
        assert out.shape[-2:] == (263, 263)
        assert 262 // 2 == 263 // 2 == 131


class TestShiftFrame:
    @pytest.mark.parametrize("method", ["fft", "interp", "auto"])
    def test_subpixel_shift_moves_the_centroid(self, method):
        n = 65
        frame = _gaussian(n, 30.0, 30.0)
        shifted = shift_frame(frame, (2.4, -1.7), method=method)
        cx, cy = _centroid(shifted)
        assert cx == pytest.approx(32.4, abs=0.05)
        assert cy == pytest.approx(28.3, abs=0.05)

    def test_coarse_rounds_to_an_integer_shift(self):
        n = 65
        frame = _gaussian(n, 30.0, 30.0)
        shifted = shift_frame(frame, (2.4, -1.7), method="coarse")
        cx, cy = _centroid(shifted)
        assert cx == pytest.approx(32.0, abs=1e-4)
        assert cy == pytest.approx(28.0, abs=1e-4)

    def test_coarse_does_not_interpolate(self):
        frame = _gaussian(65, 30.0, 30.0)
        shifted = shift_frame(frame, (3.0, 0.0), method="coarse")
        np.testing.assert_allclose(np.sort(shifted.ravel())[-1], frame.max(), rtol=1e-6)

    def test_shape_is_preserved(self):
        frame = _gaussian(65, 30.0, 30.0)
        assert shift_frame(frame, (2.4, -1.7)).shape == (65, 65)

    def test_padding_prevents_fft_wraparound(self):
        """A bright source near the border must not reappear on the far edge."""
        n = 65
        frame = np.zeros((n, n))
        frame[32, 2] = 100.0
        shifted = shift_frame(frame, (-4.0, 0.0), method="fft", pad=DEFAULT_PAD_WIDTH)
        assert np.abs(shifted[:, -6:]).max() < 1.0

    def test_auto_uses_interp_when_nan_present(self):
        n = 33
        frame = _gaussian(n, 16.0, 16.0)
        frame[0, 0] = np.nan
        shifted = shift_frame(frame, (0.5, 0.5), method="auto")
        # An FFT shift on a filled NaN would spread ringing everywhere; a spline
        # keeps it local, so the far corner stays clean.
        assert np.isfinite(shifted[-1, -1])
        assert abs(shifted[-1, -1]) < 1e-6

    def test_nan_is_restored_and_does_not_spread(self):
        n = 33
        frame = _gaussian(n, 16.0, 16.0)
        frame[0:3, 0:3] = np.nan
        shifted = shift_frame(frame, (0.4, 0.4), method="interp")
        n_nan = int(np.isnan(shifted).sum())
        assert n_nan > 0
        assert n_nan <= 25  # 3x3 shifted by <1 px covers at most 4x4 + slack

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown shift method"):
            shift_frame(_gaussian(33, 16.0, 16.0), (0.5, 0.5), method="bilinear")

    def test_non_square_frame_raises(self):
        with pytest.raises(ValueError, match="square"):
            shift_frame(np.zeros((32, 64)), (0.5, 0.5), method="fft")


class TestShiftToTarget:
    @pytest.mark.parametrize("method", ["fft", "interp", "coarse", "auto"])
    def test_star_lands_on_the_centre_pixel(self, method):
        n = 65
        target = n // 2
        frame = _gaussian(n, 30.3, 35.6)
        out = shift_to_target(frame, (30.3, 35.6), method=method)
        cx, cy = _centroid(out)
        tol = 0.6 if method == "coarse" else 0.05
        assert cx == pytest.approx(target, abs=tol)
        assert cy == pytest.approx(target, abs=tol)

    def test_already_centred_frame_is_nearly_unchanged(self):
        n = 65
        frame = _gaussian(n, n // 2, n // 2)
        out = shift_to_target(frame, (n // 2, n // 2), method="fft")
        np.testing.assert_allclose(out, frame, atol=1e-6)


class _Fixture:
    """Builds a minimal converted/ directory for the alignment step."""

    def __init__(self, tmp_path, n_wave, n_frames, size, waffle, star=None):
        from astropy.io import fits

        self.dir = tmp_path / "converted"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n_wave, self.n_frames, self.size = n_wave, n_frames, size
        self.identifier = "center" if waffle else "coro"
        self.star = star if star is not None else (size / 2 + 3.4, size / 2 - 2.6)

        cube = np.zeros((n_wave, n_frames, size, size), dtype=np.float32)
        for w in range(n_wave):
            for f in range(n_frames):
                cube[w, f] = _gaussian(size, self.star[0], self.star[1])
        fits.writeto(self.dir / f"{self.identifier}_cube.fits", cube, overwrite=True)

        fits.writeto(
            self.dir / "wavelengths.fits",
            np.linspace(1000.0, 2000.0, n_wave).astype(np.float32),
            overwrite=True,
        )

        centers = np.zeros((n_wave, n_frames, 2), dtype=np.float32)
        centers[..., 0] = self.star[0]
        centers[..., 1] = self.star[1]
        fits.writeto(
            self.dir / "image_centers_fitted_robust.fits", centers, overwrite=True
        )

        import pandas as pd
        pd.DataFrame({"DEROT ANGLE": np.zeros(n_frames)}).to_csv(
            self.dir / f"frames_info_{self.identifier}.csv", index=False
        )

    def aligned_path(self):
        return self.dir / f"{self.identifier}_cube_aligned.fits"


def _stamp_waffle(fx, waffle):
    """Write the provenance keyword preprocess stamps onto the science cube."""
    from astropy.io import fits

    path = fx.dir / f"{fx.identifier}_cube.fits"
    with fits.open(path, mode="update") as hdul:
        hdul[0].header["HIERARCH SPHERICAL WAFFLE MODE"] = bool(waffle)


class TestRunFrameAlignment:
    def test_irdis_waffle_writes_center_aligned_with_star_on_centre(self, tmp_path):
        from unittest.mock import MagicMock

        from astropy.io import fits

        from spherical.pipeline.pipeline_config import AlignmentConfig
        from spherical.pipeline.steps.align_frames import run_frame_alignment

        fx = _Fixture(tmp_path, n_wave=2, n_frames=3, size=65, waffle=True)
        out = run_frame_alignment(
            str(fx.dir), AlignmentConfig(), MagicMock(), continuous_satellite_spots=True
        )
        assert out == fx.aligned_path()

        data = fits.getdata(out)
        assert data.shape == (2, 3, 65, 65)
        target = 65 // 2
        for w in range(2):
            for f in range(3):
                cx, cy = _centroid(data[w, f])
                assert cx == pytest.approx(target, abs=0.05)
                assert cy == pytest.approx(target, abs=0.05)

    def test_irdis_non_waffle_writes_coro_aligned(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.pipeline_config import AlignmentConfig
        from spherical.pipeline.steps.align_frames import run_frame_alignment

        fx = _Fixture(tmp_path, n_wave=2, n_frames=4, size=65, waffle=False)
        out = run_frame_alignment(
            str(fx.dir), AlignmentConfig(), MagicMock(), continuous_satellite_spots=False
        )
        assert out.name == "coro_cube_aligned.fits"

    def test_ifs_262_is_padded_to_263(self, tmp_path):
        from unittest.mock import MagicMock

        from astropy.io import fits

        from spherical.pipeline.pipeline_config import AlignmentConfig
        from spherical.pipeline.steps.align_frames import run_frame_alignment

        fx = _Fixture(tmp_path, n_wave=39, n_frames=1, size=262, waffle=True)
        out = run_frame_alignment(
            str(fx.dir), AlignmentConfig(), MagicMock(), continuous_satellite_spots=True
        )
        data = fits.getdata(out)
        assert data.shape == (39, 1, 263, 263)

    def test_no_ivar_is_written(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.pipeline_config import AlignmentConfig
        from spherical.pipeline.steps.align_frames import run_frame_alignment

        fx = _Fixture(tmp_path, n_wave=2, n_frames=2, size=65, waffle=True)
        run_frame_alignment(
            str(fx.dir), AlignmentConfig(), MagicMock(), continuous_satellite_spots=True
        )
        assert not (fx.dir / "center_ivar_cube_aligned.fits").exists()
        assert not (fx.dir / "coro_ivar_cube_aligned.fits").exists()

    def test_header_records_what_was_done(self, tmp_path):
        from unittest.mock import MagicMock

        from astropy.io import fits

        from spherical.pipeline.pipeline_config import AlignmentConfig
        from spherical.pipeline.steps.align_frames import run_frame_alignment

        fx = _Fixture(tmp_path, n_wave=2, n_frames=2, size=65, waffle=True)
        out = run_frame_alignment(
            str(fx.dir),
            AlignmentConfig(shift_method="fft", pad_width=6),
            MagicMock(),
            continuous_satellite_spots=True,
        )
        header = fits.getheader(out)
        assert header["HIERARCH SPHERICAL ALIGNED"] is True
        assert header["HIERARCH SPHERICAL ALIGN TARGET X"] == 32
        assert header["HIERARCH SPHERICAL ALIGN TARGET Y"] == 32
        assert header["HIERARCH SPHERICAL ALIGN METHOD"] == "fft"
        assert header["HIERARCH SPHERICAL ALIGN PAD"] == 6
        assert "HIERARCH SPHERICAL ALIGN REPAIRED" in header

    def test_ifs_repair_logs_a_placeholder_warning(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.pipeline_config import AlignmentConfig
        from spherical.pipeline.steps.align_frames import run_frame_alignment

        logger = MagicMock()
        fx = _Fixture(tmp_path, n_wave=39, n_frames=1, size=262, waffle=True)
        run_frame_alignment(
            str(fx.dir),
            AlignmentConfig(repair_bad_pixels=True),
            logger,
            continuous_satellite_spots=True,
        )
        messages = " ".join(str(c) for c in logger.warning.call_args_list)
        assert "IFS bad-pixel interpolation not yet implemented" in messages

    def test_irdis_repair_is_satisfied_by_preprocess(self, tmp_path):
        from unittest.mock import MagicMock

        from astropy.io import fits

        from spherical.pipeline.pipeline_config import AlignmentConfig
        from spherical.pipeline.steps.align_frames import run_frame_alignment

        logger = MagicMock()
        fx = _Fixture(tmp_path, n_wave=2, n_frames=1, size=65, waffle=True)
        out = run_frame_alignment(
            str(fx.dir),
            AlignmentConfig(repair_bad_pixels=True),
            logger,
            continuous_satellite_spots=True,
        )
        assert fits.getheader(out)["HIERARCH SPHERICAL ALIGN REPAIRED"] is True
        messages = " ".join(str(c) for c in logger.warning.call_args_list)
        assert "IFS bad-pixel interpolation" not in messages

    def test_frame_axis_mismatch_raises(self, tmp_path):
        from unittest.mock import MagicMock

        from astropy.io import fits

        from spherical.pipeline.pipeline_config import AlignmentConfig
        from spherical.pipeline.steps.align_frames import run_frame_alignment

        fx = _Fixture(tmp_path, n_wave=2, n_frames=3, size=65, waffle=True)
        fits.writeto(
            fx.dir / "image_centers_fitted_robust.fits",
            np.zeros((2, 5, 2), dtype=np.float32),
            overwrite=True,
        )
        with pytest.raises(ValueError, match="Frame-axis mismatch"):
            run_frame_alignment(
                str(fx.dir), AlignmentConfig(), MagicMock(),
                continuous_satellite_spots=True,
            )


class TestWaffleModeFromHeader:
    """The flag comes from WAFFLE_MODE, never from guessing at files on disk.

    Inside the pipeline the orchestrator always passes it. A standalone re-run
    reads the keyword preprocess stamped into the cube, and refuses to guess
    when an older reduction does not carry it.
    """

    def test_explicit_flag_wins_over_the_header(self, tmp_path):
        from astropy.io import fits

        from spherical.pipeline.steps.align_frames import resolve_waffle_mode

        fx = _Fixture(tmp_path, n_wave=2, n_frames=3, size=65, waffle=True)
        _stamp_waffle(fx, False)
        assert resolve_waffle_mode(str(fx.dir), True) is True
        assert fits.getheader(
            fx.dir / "center_cube.fits"
        )["HIERARCH SPHERICAL WAFFLE MODE"] is False

    @pytest.mark.parametrize("waffle", [True, False])
    def test_header_keyword_is_read_when_the_flag_is_omitted(self, tmp_path, waffle):
        from spherical.pipeline.steps.align_frames import resolve_waffle_mode

        fx = _Fixture(tmp_path, n_wave=2, n_frames=3, size=65, waffle=waffle)
        _stamp_waffle(fx, waffle)
        assert resolve_waffle_mode(str(fx.dir), None) is waffle

    def test_missing_keyword_raises_instead_of_guessing(self, tmp_path):
        from spherical.pipeline.steps.align_frames import resolve_waffle_mode

        fx = _Fixture(tmp_path, n_wave=2, n_frames=3, size=65, waffle=False)
        with pytest.raises(ValueError, match="WAFFLE MODE"):
            resolve_waffle_mode(str(fx.dir), None)

    def test_no_inference_helper_survives(self):
        from spherical.pipeline.steps import align_frames

        assert not hasattr(align_frames, "infer_continuous_satellite_spots")
        assert "symlink" not in pathlib.Path(align_frames.__file__).read_text()

    def test_standalone_run_uses_the_header(self, tmp_path):
        from unittest.mock import MagicMock

        from spherical.pipeline.pipeline_config import AlignmentConfig
        from spherical.pipeline.steps.align_frames import run_frame_alignment

        fx = _Fixture(tmp_path, n_wave=39, n_frames=3, size=65, waffle=False)
        _stamp_waffle(fx, False)
        out = run_frame_alignment(str(fx.dir), AlignmentConfig(), MagicMock())
        assert out.name == "coro_cube_aligned.fits"
