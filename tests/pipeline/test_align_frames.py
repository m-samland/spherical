"""Tests for optional frame alignment (#161)."""
from __future__ import annotations

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
