import numpy as np
import pandas as pd
import pytest

from spherical.pipeline.star_amplitude import (
    block_table,
    block_wfs_level,
    classify,
    reference_segment,
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
