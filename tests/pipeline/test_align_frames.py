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
