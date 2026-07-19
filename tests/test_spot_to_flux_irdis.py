"""Tests for the IRDIS per-band normalization range in spot_to_flux."""
from __future__ import annotations

import numpy as np

from spherical.pipeline.steps.spot_to_flux import (
    _BAND_NORMALIZATION_RANGES_MICRON,
    _detect_normalization_range,
)


class TestDetectNormalizationRange:
    def test_ifs_yj_range(self):
        wl = np.linspace(0.95, 1.35, 39)  # YJ mode
        assert _detect_normalization_range(wl) == _BAND_NORMALIZATION_RANGES_MICRON["IFS"]

    def test_ifs_yjh_range(self):
        wl = np.linspace(0.95, 1.65, 39)  # YJH mode; max reaches into H
        assert _detect_normalization_range(wl) == _BAND_NORMALIZATION_RANGES_MICRON["IFS"]

    def test_irdis_h_range(self):
        wl = np.array([1.593, 1.667])  # H2, H3
        assert _detect_normalization_range(wl) == _BAND_NORMALIZATION_RANGES_MICRON["IRDIS_H"]

    def test_irdis_k_range(self):
        wl = np.array([2.103, 2.255])  # K1, K2
        assert _detect_normalization_range(wl) == _BAND_NORMALIZATION_RANGES_MICRON["IRDIS_K"]

    def test_contains_at_least_one_channel(self):
        for wl_um in ([1.593, 1.667], [2.103, 2.255], [1.0, 1.3, 1.5]):
            rng = _detect_normalization_range(np.array(wl_um))
            assert any(rng[0] <= w <= rng[1] for w in wl_um), \
                f"Normalization range {rng} contains none of {wl_um}"
