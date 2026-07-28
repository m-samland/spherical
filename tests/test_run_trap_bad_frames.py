"""Test the CENTER-outlier → bad_frames loader in run_trap.

Verifies the union-across-channels behaviour of the small loader block added
to run_trap.py, without exercising the full TRAP call path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits


def _load_center_outliers_as_bad_frames(outliers_path: Path):
    """Standalone copy of the run_trap loader logic — kept in sync manually.

    Loads outlier_frames.fits (shape (n_wave, k_max), -1 = padding), unions
    the per-channel positive indices, returns a sorted list (or None if no
    outliers were flagged).
    """
    outliers_per_ch = fits.getdata(str(outliers_path))
    union: set[int] = set()
    for ch in range(outliers_per_ch.shape[0]):
        union.update(int(i) for i in outliers_per_ch[ch] if int(i) >= 0)
    return sorted(union) if union else None


class TestCenterOutliersUnion:
    def test_padded_outlier_file_unions_correctly(self, tmp_path: Path):
        # ch0 flags frames 3, 7, 12; ch1 flags frames 7, 20, 42.
        # Union: {3, 7, 12, 20, 42}. Padding is -1.
        outliers = np.array([
            [3, 7, 12, -1, -1],
            [7, 20, 42, -1, -1],
        ], dtype=np.int32)
        p = tmp_path / "center_outlier_frames.fits"
        fits.writeto(str(p), outliers)
        bad_frames = _load_center_outliers_as_bad_frames(p)
        assert bad_frames == [3, 7, 12, 20, 42]

    def test_all_padding_returns_none(self, tmp_path: Path):
        outliers = np.full((2, 4), -1, dtype=np.int32)
        p = tmp_path / "center_outlier_frames.fits"
        fits.writeto(str(p), outliers)
        assert _load_center_outliers_as_bad_frames(p) is None

    def test_single_channel_zero_indices_kept(self, tmp_path: Path):
        # Index 0 is a legitimate frame index and must not be treated as padding.
        outliers = np.array([[0, -1, -1]], dtype=np.int32)
        p = tmp_path / "center_outlier_frames.fits"
        fits.writeto(str(p), outliers)
        assert _load_center_outliers_as_bad_frames(p) == [0]
