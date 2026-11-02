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


def _two_column_samples(intervals, a, b, segments=None):
    frame = _samples(intervals, a, segments).rename(columns={"value": "a"})
    frame["b"] = b
    return frame


def test_all_nan_column_does_not_affect_other_columns():
    samples = _two_column_samples([(0, 10), (10, 20)], [np.nan, np.nan], [1.0, 3.0], [0, 0])
    out = match_to_frames(samples, _frames((5, 15)), ["a", "b"], max_interp_gap_s=60)
    assert np.isnan(out["a"].iloc[0])
    assert out["b"].iloc[0] == pytest.approx(2.0)
    assert out["flag"].iloc[0] == "direct"


def test_nan_samples_are_left_out_of_the_weighted_mean():
    samples = _two_column_samples([(0, 10), (10, 20)], [np.nan, 4.0], [1.0, 3.0], [0, 0])
    out = match_to_frames(samples, _frames((5, 15)), ["a", "b"], max_interp_gap_s=60)
    assert out["a"].iloc[0] == pytest.approx(4.0)
    assert out["b"].iloc[0] == pytest.approx(2.0)


def test_all_nan_overlap_is_gap():
    samples = _two_column_samples([(0, 10), (10, 20)], [np.nan, np.nan], [np.nan, np.nan], [0, 0])
    out = match_to_frames(samples, _frames((5, 15)), ["a", "b"], max_interp_gap_s=60)
    assert out["flag"].iloc[0] == "gap"
    assert np.isnan(out[["a", "b"]].iloc[0]).all()


def test_interpolation_skips_nan_neighbour():
    samples = _samples([(0, 10), (10, 20), (40, 50)], [1.0, np.nan, 5.0], [0, 0, 0])
    out = match_to_frames(samples, _frames((29.5, 30.5)), ["value"], max_interp_gap_s=60)
    assert out["flag"].iloc[0] == "interpolated"
    assert out["value"].iloc[0] == pytest.approx(1.0 + (30 - 5) / (45 - 5) * 4.0)


def test_frame_overlapping_only_nan_samples_is_interpolated():
    samples = _samples([(0, 10), (10, 20), (20, 30)], [1.0, np.nan, 3.0], [0, 0, 0])
    out = match_to_frames(samples, _frames((12, 18)), ["value"], max_interp_gap_s=60)
    assert out["flag"].iloc[0] == "interpolated"
    assert out["value"].iloc[0] == pytest.approx(2.0)
    assert out["n_samples"].iloc[0] == 0
