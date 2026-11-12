import numpy as np
import pandas as pd
import pytest

from spherical.pipeline.star_amplitude import (
    anchor,
    block_disagreement,
    block_table,
    block_wfs_level,
    classify,
    fill_in_time,
    reference_segment,
    spot_anchored_amplitude,
    usable_blocks,
    wfs_amplitude,
    wfs_samples,
)


def test_reference_segment_is_the_most_common_direct_segment():
    m = pd.DataFrame({"visloop_flux_subap": [1.0, 1.0, 1.0, 1.0], "flag_visloop": ["direct", "direct", "gap", "direct"],
                      "segment": [1, 1, 0, 0]})
    assert reference_segment(m) == 1


def test_reference_segment_is_none_without_segment_column():
    m = pd.DataFrame({"visloop_flux_subap": [1.0], "flag_visloop": ["direct"]})
    assert reference_segment(m) is None


def test_wfs_samples_keep_direct_positive_values_of_the_reference_segment():
    m = pd.DataFrame({"visloop_flux_subap": [1.0, 2.0, 3.0, -1.0, 5.0],
                      "flag_visloop": ["direct", "direct", "interpolated", "direct", "direct"], "segment": [0, 0, 0, 0, 1]})
    np.testing.assert_array_equal(wfs_samples(m, 0), [1.0, 2.0, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(wfs_samples(m.drop(columns="segment"), None), [1.0, 2.0, np.nan, np.nan, 5.0])


def test_block_table_leaves_a_block_emptied_in_one_channel_undefined():
    flux = np.array([[10.0, 10.0, 8.0, 8.0], [20.0, 20.0, 16.0, 16.0]])
    kept = np.array([[True, True, True, True], [True, True, False, False]])
    table = block_table(flux, kept, np.array([1.0, 1.0, 0.8, 0.8]), np.array([0, 0, 1, 1]), np.ones_like(flux))
    row = table[(table.channel == 1) & (table.block == 1)].iloc[0]
    assert row.n_kept == 0 and np.isnan(row.ratio)
    assert table[(table.channel == 0) & (table.block == 1)].ratio.iloc[0] == 10.0


def test_block_wfs_level_is_the_median_over_channels():
    flux = np.full((3, 2), 10.0)
    table = block_table(flux, np.ones((3, 2), bool), np.array([1.0, 2.0]), np.array([0, 1]), np.ones_like(flux))
    np.testing.assert_array_equal(block_wfs_level(table), [1.0, 2.0])


def test_flux_blocks_under_thin_cirrus_are_variable_although_nothing_drops():
    label, offset, _ = classify(np.array([0.92, 0.92]), np.ones(50))
    assert label == "variable" and offset == pytest.approx(0.08)


def test_flat_sequence_is_stable():
    assert classify(np.array([1.0, 1.01]), np.ones(50))[0] == "stable"


def test_drop_in_the_science_frames_is_variable():
    science = np.ones(50)
    science[30:] = 0.85
    label, _, depth = classify(np.array([1.0, 1.0]), science)
    assert label == "variable" and depth == pytest.approx(0.15)


def test_no_wfs_in_the_flux_blocks():
    assert classify(np.array([np.nan, np.nan]), np.ones(50))[0] == "no_wfs"


def test_a_flux_block_with_all_samples_in_another_segment_has_no_wfs_level():
    matched = pd.DataFrame({"visloop_flux_subap": [1.0, 1.0, 1.0, 1.0], "flag_visloop": ["direct"] * 4,
                            "segment": [0, 0, 1, 1]})
    flux = np.full((1, 4), 10.0)
    table = block_table(flux, np.ones((1, 4), bool), wfs_samples(matched, 0), np.array([0, 0, 1, 1]), np.ones_like(flux))
    level = block_wfs_level(table)
    assert np.isfinite(level[0]) and np.isnan(level[1])


def _table(ratios, levels=None, channels=1):
    levels = levels or [1.0] * len(ratios)
    return pd.DataFrame([dict(channel=c, block=b, n_kept=3, flux=r * w, wfs=w, ratio=r, sharpness=1.0)
                         for c in range(channels) for b, (r, w) in enumerate(zip(ratios, levels))])


def test_block_more_than_30_percent_low_is_not_usable():
    np.testing.assert_array_equal(usable_blocks(_table([1.0, 0.65])), [True, False])


def test_block_under_half_the_wfs_level_is_not_usable():
    np.testing.assert_array_equal(usable_blocks(_table([1.0, 1.0], levels=[1.0, 0.4])), [True, False])


def test_a_block_bad_in_one_channel_is_not_usable_in_any():
    table = _table([1.0, 1.0], channels=2)
    table.loc[(table.channel == 1) & (table.block == 1), "ratio"] = 0.6
    np.testing.assert_array_equal(usable_blocks(table), [True, False])


def test_a_channel_emptied_in_one_block_leaves_the_block_usable():
    table = _table([1.0, 0.8], channels=2)
    empty = (table.channel == 1) & (table.block == 1)
    table.loc[empty, ["flux", "wfs", "ratio"]] = np.nan
    table.loc[empty, "n_kept"] = 0
    np.testing.assert_array_equal(usable_blocks(table), [True, True])
    scale, level = anchor(table, usable_blocks(table))
    np.testing.assert_allclose(scale, [0.9, 1.0])
    np.testing.assert_allclose(level, [0.9, 1.0])


def test_block_excluded_by_hand_is_not_usable():
    np.testing.assert_array_equal(usable_blocks(_table([1.0, 1.0]), excluded=[0]), [False, True])


def test_disagreeing_blocks_both_stay_usable_and_the_anchor_is_the_median_of_block_ratios():
    table = _table([1.0, 0.8])
    usable = usable_blocks(table)
    assert usable.all() and block_disagreement(table, usable) == pytest.approx(0.25)
    scale, level = anchor(table, usable)
    assert scale[0] == pytest.approx(0.9) and level[0] == pytest.approx(0.9)


def test_single_block_has_no_disagreement():
    assert np.isnan(block_disagreement(_table([1.0]), np.array([True])))


def test_wfs_amplitude_follows_the_wfs_and_drops_frames_below_the_floor():
    amp = wfs_amplitude(np.array([2.0]), np.array([2.0]), np.array([1.0, 0.5, 0.05, 1.0]), floor=0.10)
    np.testing.assert_allclose(amp[0, [0, 1, 3]], [1.0, 0.5, 1.0])
    assert np.isnan(amp[0, 2])


def test_fill_in_time_interpolates_gaps_and_reports_them():
    filled, n, longest = fill_in_time(np.array([1.0, np.nan, np.nan, 4.0]), np.array([0.0, 1.0, 2.0, 3.0]) / 86400)
    np.testing.assert_allclose(filled, [1.0, 2.0, 3.0, 4.0])
    assert n == 2 and longest == pytest.approx(3.0)


def test_spot_shape_takes_its_level_from_the_anchored_wfs_next_to_the_blocks():
    spots = np.ones((1, 20))
    spots[0, 10:] = 0.5
    spots[0, 3] = np.nan
    wfs_amp = np.full((1, 20), 0.9)
    wfs_amp[0, 10:] = 0.45
    out, levels = spot_anchored_amplitude(spots, wfs_amp, np.arange(20.0), np.array([-1.0, 21.0]))
    np.testing.assert_allclose(out, 0.9 * spots)
    np.testing.assert_allclose(levels, [[0.9, 0.9]])
