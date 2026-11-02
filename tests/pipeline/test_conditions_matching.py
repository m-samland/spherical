"""Tests for matching condition samples to frames and finding gaps."""

import numpy as np
import pandas as pd
import pytest

from spherical.conditions.matching import find_gaps, match_to_frames

DAY = 86400.0
T0 = 58000.0


def _samples(intervals, values, segments=None):
    frame = pd.DataFrame(
        {
            "mjd_start": [T0 + a / DAY for a, _ in intervals],
            "mjd_end": [T0 + b / DAY for _, b in intervals],
            "value": values,
        }
    )
    if segments is not None:
        frame["segment"] = segments
    return frame


def _frames(*intervals):
    return pd.DataFrame({"MJD START": [T0 + a / DAY for a, _ in intervals], "MJD END": [T0 + b / DAY for _, b in intervals]})


def test_overlap_weighted_mean():
    samples = _samples([(0, 10), (10, 20)], [1.0, 3.0], [0, 0])
    out = match_to_frames(samples, _frames((5, 15)), ["value"], max_interp_gap_s=60)
    assert out["value"].iloc[0] == pytest.approx(2.0)
    assert out["value_std"].iloc[0] == pytest.approx(1.0)
    assert out["n_samples"].iloc[0] == 2
    assert out["flag"].iloc[0] == "direct"
    assert out["segment"].iloc[0] == 0


def test_short_frame_between_samples_is_interpolated():
    samples = _samples([(0, 10), (30, 40)], [1.0, 3.0], [0, 0])
    out = match_to_frames(samples, _frames((19, 20)), ["value"], max_interp_gap_s=60)
    assert out["value"].iloc[0] == pytest.approx(1 + 2 * 14.5 / 30)
    assert out["flag"].iloc[0] == "interpolated"
    assert out["n_samples"].iloc[0] == 0


def test_long_hole_is_gap():
    samples = _samples([(0, 10), (30, 40)], [1.0, 3.0], [0, 0])
    out = match_to_frames(samples, _frames((19, 20)), ["value"], max_interp_gap_s=10)
    assert np.isnan(out["value"].iloc[0])
    assert out["flag"].iloc[0] == "gap"
    assert out["segment"].iloc[0] == -1


def test_no_interpolation_across_segments():
    samples = _samples([(0, 10), (30, 40)], [1.0, 3.0], [0, 1])
    out = match_to_frames(samples, _frames((19, 20)), ["value"], max_interp_gap_s=60)
    assert np.isnan(out["value"].iloc[0])
    assert out["flag"].iloc[0] == "gap"


def test_overlap_does_not_mix_segments():
    samples = _samples([(0, 10), (10, 20)], [1.0, 3.0], [0, 1])
    out = match_to_frames(samples, _frames((7, 20)), ["value"], max_interp_gap_s=60)
    assert out["value"].iloc[0] == pytest.approx(3.0)
    assert out["segment"].iloc[0] == 1
    assert out["n_samples"].iloc[0] == 1
    assert out["flag"].iloc[0] == "direct"


def test_frames_outside_samples_and_empty_samples_are_gaps():
    samples = _samples([(0, 10)], [1.0], [0])
    out = match_to_frames(samples, _frames((50, 51)), ["value"], max_interp_gap_s=60)
    assert out["flag"].iloc[0] == "gap"
    empty = match_to_frames(samples.iloc[:0], _frames((0, 1)), ["value"], max_interp_gap_s=60)
    assert empty["flag"].iloc[0] == "gap"


def test_find_gaps():
    samples = _samples([(s, s + 10) for s in [*range(0, 100, 10), *range(200, 300, 10)]], 1.0)
    window = (T0, T0 + 300 / DAY)
    gaps = find_gaps(samples, window)
    assert len(gaps) == 1
    assert gaps[0] == pytest.approx((T0 + 100 / DAY, T0 + 200 / DAY))

    short_hole = _samples([(0, 10), (30, 40)], [1.0, 1.0])
    assert find_gaps(short_hole, (T0, T0 + 40 / DAY), factor=3.0) == []


def test_find_gaps_reports_window_edges_and_empty():
    samples = _samples([(100, 110), (110, 120)], [1.0, 1.0])
    gaps = find_gaps(samples, (T0, T0 + 200 / DAY))
    assert len(gaps) == 2
    assert gaps[0] == pytest.approx((T0, T0 + 100 / DAY))
    assert gaps[1] == pytest.approx((T0 + 120 / DAY, T0 + 200 / DAY))
    assert find_gaps(samples.iloc[:0], (T0, T0 + 1.0)) == [(T0, T0 + 1.0)]
