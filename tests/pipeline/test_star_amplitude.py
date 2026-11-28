import logging

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from spherical.pipeline.star_amplitude import (
    ALL_SEGMENTS,
    FALLBACK_STATUSES,
    NO_VETTING,
    NO_WFS,
    anchor,
    block_disagreement,
    block_table,
    block_wfs_level,
    classify,
    configuration_changes,
    configuration_summary,
    edge_value,
    fill_in_time,
    frame_readings,
    load_star_amplitude,
    psf_block_mean,
    reference_segment,
    sample_keys,
    segment_configurations,
    sparta_readings,
    spot_anchored_amplitude,
    usable_blocks,
    wfs_amplitude,
    wfs_samples,
)

M = ("Wide", "FILT_M", "OPEN")
S = ("Wide", "FILT_S", "OPEN")


def test_reference_segment_is_the_most_common_direct_segment():
    m = pd.DataFrame({"visloop_flux_subap": [1.0, 1.0, 1.0, 1.0], "flag_visloop": ["direct", "direct", "gap", "direct"],
                      "segment": [1, 1, 0, 0]})
    assert reference_segment(m) == 1


def test_reference_segment_is_none_without_segment_column():
    m = pd.DataFrame({"visloop_flux_subap": [1.0], "flag_visloop": ["direct"]})
    assert reference_segment(m) is None


def test_reference_segment_tie_goes_to_the_lowest_segment():
    science = pd.DataFrame({"visloop_flux_subap": [1.0] * 4, "flag_visloop": ["direct"] * 4, "segment": [2, 2, 0, 0]})
    assert reference_segment(science) == 0


def test_configuration_summary_counts_science_frames_and_flux_blocks_per_configuration():
    config = _config([("a", 0, "Wide", "FILT_M", "OPEN", "OPEN"), ("b", 1, "Wide", "FILT_M", "OPEN", "ND_2.0"),
                      ("c", 2, "Wide", "FILT_S", "OPEN", "OPEN")])
    science = pd.DataFrame({"visloop_flux_subap": [1.0, 1.0, 1.0, np.nan], "flag_visloop": ["direct"] * 4,
                            "segment": [0, 0, 2, 2]})
    flux = pd.DataFrame({"visloop_flux_subap": [1.0] * 4, "flag_visloop": ["direct"] * 4, "segment": [1, 1, 2, 2]})
    summary = configuration_summary(config, science, flux, [0, 0, 1, np.nan])
    assert summary["spatial_filter"].tolist() == ["FILT_M", "FILT_S"]
    assert summary["segments"].tolist() == [[0, 1], [2]]
    assert summary["n_science"].tolist() == [2, 1]
    assert summary["n_flux_blocks"].tolist() == [1, 1]
    assert configuration_summary(None, science, flux, [0, 0, 1, 1]) is None


def test_wfs_samples_keep_direct_positive_values():
    m = pd.DataFrame({"visloop_flux_subap": [1.0, 2.0, 3.0, -1.0, 5.0, 6.0],
                      "flag_visloop": ["direct", "direct", "interpolated", "direct", "direct", "direct"],
                      "segment": [0, 0, 0, 0, 1, -1]})
    np.testing.assert_array_equal(wfs_samples(m), [1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
    np.testing.assert_array_equal(wfs_samples(m.drop(columns="segment")), [1.0, 2.0, np.nan, np.nan, 5.0, 6.0])


def test_sparta_readings_are_the_positive_readings_in_time_order_with_their_configuration():
    configuration = {0: M, 1: M, 2: S}
    visloop = pd.DataFrame({"mjd_start": [2.0, 0.0, 1.0, 3.0, 4.0], "mjd_end": [2.5, 0.5, 1.5, 3.5, 4.5],
                            "flux_subap": [3.0, 1.0, 2.0, -1.0, 5.0], "segment": [2, 0, 1, 2, 7]})
    times, values, keys = sparta_readings(visloop, configuration)
    np.testing.assert_array_equal(times, [0.25, 1.25, 2.25, 4.25])
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 5.0])
    assert keys == [M, M, S, None]
    assert sparta_readings(visloop, None)[2] == [ALL_SEGMENTS] * 4
    assert sparta_readings(visloop.drop(columns="segment"), configuration) is None
    assert sparta_readings(None, configuration) is None


def test_frames_shorter_than_the_reading_cadence_count_each_reading_once():
    # Four frames of 8 s share each 30 s reading.
    times = np.arange(12) * 8 / 86400
    values = np.repeat([207.71, 208.58, 208.83], 4)
    values[5] = np.nan
    keys = [M] * 6 + [S] * 6
    t, v, k = frame_readings(times, values, keys)
    np.testing.assert_array_equal(v, [207.71, 208.58, 208.58, 208.83])
    assert k == [M, M, S, S]
    assert t[0] == times[0] and t[1] == times[4]


def test_an_empty_or_duplicated_readings_table_is_handled():
    columns = ["mjd_start", "mjd_end", "flux_subap", "segment"]
    assert sparta_readings(pd.DataFrame(columns=columns), None) is None
    assert sparta_readings(pd.DataFrame([[0.0, 1.0, -1.0, 0]], columns=columns), None) is None
    # SPARTA repeats the last reading of a file as the first of the next, with a shifted start and another segment.
    boundary = pd.DataFrame([[0.0, 1.0, 5.0, 0], [1.0, 2.0, 6.0, 0], [1.2, 2.2, 6.0, 1], [2.2, 3.2, 7.0, 1]],
                            columns=columns)
    times, values, keys = sparta_readings(boundary, {0: M, 1: S})
    np.testing.assert_array_equal(values, [5.0, 6.0, 7.0])
    np.testing.assert_array_equal(times, [0.5, 1.5, 2.7])
    assert keys == [M, M, S]


def _config(rows):
    return pd.DataFrame(rows, columns=["dp_id", "segment", "wfs_mode", "spatial_filter", "wfs_filter", "nd_filter"])


def test_flux_frames_behind_the_nd_share_the_wfs_configuration_of_the_science_frames():
    config = _config([("a", 0, *M, "ND_2.0"), ("b", 1, *M, "OPEN"), ("c", 1, *M, "OPEN"), ("d", 2, *M, "ND_2.0")])
    configuration = segment_configurations(config)
    flux = pd.DataFrame({"visloop_flux_subap": [1.0] * 4, "flag_visloop": ["direct"] * 4, "segment": [0, 0, 2, 2]})
    science = pd.DataFrame({"visloop_flux_subap": [1.0, 1.0], "flag_visloop": ["direct"] * 2, "segment": [1, 1]})
    assert set(sample_keys(flux, configuration)) == set(sample_keys(science, configuration)) == {M}


def test_sample_keys_mark_rows_without_a_direct_sample_or_a_known_segment():
    configuration = segment_configurations(_config([("a", 0, *S, "ND_2.0"), ("b", 1, *M, "OPEN")]))
    m = pd.DataFrame({"visloop_flux_subap": [1.0, 1.0, 1.0, 1.0], "flag_visloop": ["direct", "direct", "gap", "direct"],
                      "segment": [0, 1, 1, 3]})
    assert sample_keys(m, configuration) == [S, M, None, None]
    assert sample_keys(m, None) == [ALL_SEGMENTS, ALL_SEGMENTS, None, ALL_SEGMENTS]


def test_missing_configuration_values_compare_equal():
    configuration = segment_configurations(_config([("a", 0, None, "FILT_M", "OPEN", "ND_2.0"),
                                                    ("b", 1, None, "FILT_M", "OPEN", "OPEN")]))
    assert configuration[0] == configuration[1]


def test_a_config_table_without_the_configuration_keys_is_unusable():
    assert segment_configurations(None) is None
    assert segment_configurations(_config([]).drop(columns="spatial_filter")) is None


def test_the_step_at_a_change_is_measured_from_median_edge_values_and_not_applied():
    other = ("Wide", "FILT_L", "OPEN")
    times = np.arange(30) / 1440.0
    values = np.r_[np.full(10, 1.0), np.full(10, 0.5), np.full(10, 0.25)]
    values[10] = 5.0  # one outlier next to the change does not move the median
    changes = configuration_changes(times, values, [M] * 10 + [S] * 10 + [other] * 10, {M, S, other})
    assert [(c["before"], c["after"]) for c in changes] == [(M, S), (S, other)]
    assert [c["step"] for c in changes] == pytest.approx([0.5, 0.5])
    assert changes[0]["gap_s"] == pytest.approx(60.0)
    assert changes[0]["time"] == pytest.approx(times[10])


def test_only_changes_between_runs_of_the_science_configurations_are_reported():
    acquisition = ("Wide", "FILT_L", "OPEN")
    times = np.arange(40) / 1440.0
    keys = [acquisition] * 10 + [M] * 10 + [acquisition] * 10 + [S] * 5 + [acquisition] * 5
    changes = configuration_changes(times, np.ones(40), keys, {M, S})
    # Every change involves a configuration without science frames, so none compares two science configurations.
    assert changes == []
    direct = configuration_changes(times, np.ones(40), [acquisition] * 10 + [M] * 20 + [S] * 10, {M, S})
    assert [(c["before"], c["after"]) for c in direct] == [(M, S)]
    assert configuration_changes(times, np.ones(40), [M] * 40, {M}) == []
    assert configuration_changes(times, np.ones(40), [acquisition] * 40, {M}) == []
    assert configuration_changes(times, np.full(40, np.nan), [M] * 40, {M}) == []


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
    table = block_table(flux, np.ones((1, 4), bool), np.where(matched["segment"] == 0, wfs_samples(matched), np.nan), np.array([0, 0, 1, 1]), np.ones_like(flux))
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
    wfs = np.array([1.0, 0.5, 0.05, 1.0])
    amp = wfs_amplitude(np.array([2.0]), np.array([2.0]), wfs, wfs, floor=0.10)
    np.testing.assert_allclose(amp[0, [0, 1, 3]], [1.0, 0.5, 1.0])
    assert np.isnan(amp[0, 2])


def test_wfs_amplitude_never_floors_a_filled_frame_and_ignores_it_in_the_percentile():
    measured = np.array([1.0, 1.0, np.nan, 0.05, 1.0])
    filled = np.array([1.0, 1.0, 0.9, 0.05, 1.0])
    amp = wfs_amplitude(np.array([1.0]), np.array([1.0]), filled, measured, floor=0.10)
    assert np.isfinite(amp[0, 2]) and np.isnan(amp[0, 3])
    assert np.isfinite(amp[0, [0, 1, 4]]).all()
    filled_low = np.array([1.0, 1.0, 0.01, 1.0, 1.0])
    amp = wfs_amplitude(np.array([1.0]), np.array([1.0]), filled_low, np.array([1.0, 1.0, np.nan, 1.0, 1.0]), floor=0.10)
    assert np.isfinite(amp).all()


def test_fill_in_time_interpolates_between_robust_edge_values_and_reports_the_gaps():
    filled, n, longest, fewest = fill_in_time(np.array([1.0, np.nan, np.nan, 4.0]), np.array([0.0, 1.0, 2.0, 3.0]) / 86400)
    np.testing.assert_allclose(filled, [1.0, 2.0, 3.0, 4.0])
    assert n == 2 and longest == pytest.approx(3.0) and fewest == 1


def test_an_outlier_next_to_a_gap_does_not_set_the_fill():
    values = np.r_[np.full(6, 1.0), [3.0], [np.nan] * 3, np.full(6, 1.0)]
    filled, _, _, fewest = fill_in_time(values, np.arange(values.size) / 1440.0)
    np.testing.assert_allclose(filled[7:10], 1.0)
    assert fewest == 5


def test_fill_follows_a_trend_across_the_gap_and_is_constant_beyond_the_ends():
    times = np.arange(20) / 1440.0
    values = np.linspace(1.0, 0.6, 20)
    measured = values.copy()
    measured[[0, 1, 9, 10, 19]] = np.nan
    filled, n, longest, _ = fill_in_time(measured, times, n_samples=3)
    np.testing.assert_allclose(filled[[9, 10]], values[[9, 10]], rtol=1e-12)
    np.testing.assert_allclose(filled[[0, 1]], np.mean(values[2:5]))
    np.testing.assert_allclose(filled[19], np.mean(values[16:19]))
    assert n == 5 and longest == pytest.approx(180.0)


def test_edge_values_use_only_samples_within_the_window():
    times = np.array([0.0, 60.0, 400.0, 500.0]) / 86400
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert edge_value(times, values, "after", 5, 300.0)[0::2] == (1.5, 2)
    assert edge_value(times, values, "before", 5, 300.0)[0::2] == (3.5, 2)
    assert edge_value(times, values, "after", 1, 300.0)[0::2] == (1.0, 1)
    assert edge_value(times, values, "after", 5, 10.0)[0::2] == (1.0, 1)


def test_fill_takes_its_edge_values_from_the_readings_between_the_first_and_last_frame():
    frame_times = np.arange(10) / 1440.0
    measured = np.r_[np.full(4, 1.0), [np.nan] * 2, np.full(4, 1.0)]
    reading_times = np.r_[-5.0, np.arange(10)] / 1440.0
    reading_values = np.r_[100.0, [2.0] * 4, [9.0] * 2, [4.0] * 4]
    filled, n, _, fewest = fill_in_time(measured, frame_times, n_samples=2, readings=(reading_times, reading_values))
    # The readings inside the gap and the one before the first frame are not used.
    np.testing.assert_allclose(filled[[4, 5]], [2.75, 3.25])
    assert n == 2 and fewest == 2


def test_a_side_without_readings_falls_back_to_the_measured_frames():
    times = np.arange(6) / 1440.0
    measured = np.array([1.0, 1.0, np.nan, np.nan, 3.0, 3.0])
    filled = fill_in_time(measured, times, readings=(np.array([0.0, 1.0]) / 1440.0, np.array([1.0, 1.0])))[0]
    np.testing.assert_allclose(filled[[2, 3]], [1.75, 2.25])


def test_a_frame_without_a_time_is_not_filled_and_does_not_stop_the_fill():
    times = np.array([0.0, 1.0, np.nan, 3.0, 4.0]) / 1440.0
    measured = np.array([1.0, np.nan, np.nan, 3.0, 3.0])
    readings = (np.array([0.0, 3.0, 4.0]) / 1440.0, np.array([1.0, 3.0, 3.0]))
    filled, n, _, _ = fill_in_time(measured, times, readings=readings)
    assert np.isfinite(filled[1]) and np.isnan(filled[2]) and n == 1


def test_fill_in_time_leaves_complete_and_empty_series_alone():
    assert fill_in_time(np.ones(3), np.arange(3.0))[1:] == (0, 0.0, None)
    assert np.isnan(fill_in_time(np.full(3, np.nan), np.arange(3.0))[0]).all()


def test_spot_shape_takes_its_level_from_the_anchored_wfs_next_to_the_blocks():
    spots = np.ones((1, 20))
    spots[0, 10:] = 0.5
    spots[0, 3] = np.nan
    wfs_amp = np.full((1, 20), 0.9)
    wfs_amp[0, 10:] = 0.45
    out, levels = spot_anchored_amplitude(spots, wfs_amp, np.arange(20.0), np.array([-1.0, 21.0]))
    np.testing.assert_allclose(out, 0.9 * spots)
    np.testing.assert_allclose(levels, [[0.9, 0.9]])


def test_only_measured_frames_set_the_spot_level():
    spots = np.ones((1, 20))
    wfs = np.full((1, 20), 0.8)
    wfs[0, 8:12] = 0.5  # filled frames next to the block
    measured = np.ones(20, dtype=bool)
    measured[8:12] = False
    times = np.arange(20.0)
    amplitude, levels = spot_anchored_amplitude(spots, wfs, times, [9.5], measured=measured)
    np.testing.assert_allclose(levels, 0.8)
    np.testing.assert_allclose(amplitude, 0.8)


def test_block_excluded_by_hand_does_not_fail_the_good_blocks():
    np.testing.assert_array_equal(usable_blocks(_table([1.0, 1.0, 2.0]), excluded=[2]), [True, True, False])


def _write_star_amplitude(directory, amplitude, blocks, mode="wfs", reason="variable"):
    from spherical.pipeline.steps.star_amplitude_calibration import _write

    header = {"MODE": mode, "REASON": reason, "LEVOFF": None, "DROPDEP": None, "DISAGREE": None, "SCALEUNC": None,
              "NFLOOR": 0, "NFILLED": 0, "MAXGAP_S": 0.0, "SEGMENT": None, "WFSFLOOR": 0.1, "WAFMODE": "wfs"}
    n_wave = np.asarray(amplitude).shape[0]
    _write(directory, amplitude, header, np.asarray(blocks), np.ones(n_wave), np.ones(n_wave))


AMPLITUDE = np.array([[1.0, 0.9, np.nan, 1.1, 1.0], [1.0, 0.8, np.nan, 1.2, 1.0]])


def test_load_star_amplitude_round_trips(tmp_path, caplog):
    _write_star_amplitude(tmp_path, AMPLITUDE, [1, 0])
    with caplog.at_level(logging.INFO, logger="spherical.pipeline.star_amplitude"):
        amplitude, mask = load_star_amplitude(tmp_path, 5, 2)
    np.testing.assert_allclose(amplitude, AMPLITUDE.astype(np.float32))
    np.testing.assert_array_equal(mask, [True, False])
    assert "mode wfs, reason variable" in caplog.text


def test_load_star_amplitude_ignores_a_file_without_a_blocks_extension(tmp_path, caplog):
    fits.writeto(tmp_path / "star_amplitude.fits", AMPLITUDE.astype(np.float32), header=fits.Header({"MODE": "wfs"}))
    with caplog.at_level(logging.WARNING, logger="spherical.pipeline.star_amplitude"):
        assert load_star_amplitude(tmp_path, 5, 2) == (None, None)
    assert 'force={"calibrate_star_amplitude"}' in caplog.text


def test_load_star_amplitude_ignores_a_file_on_another_frame_axis(tmp_path, caplog):
    _write_star_amplitude(tmp_path, AMPLITUDE, [1, 0])
    with caplog.at_level(logging.WARNING, logger="spherical.pipeline.star_amplitude"):
        assert load_star_amplitude(tmp_path, 6, 2) == (None, None)
    assert "calibrate_star_amplitude" in caplog.text


def test_load_star_amplitude_ignores_a_mask_on_another_block_axis(tmp_path, caplog):
    _write_star_amplitude(tmp_path, AMPLITUDE, [1, 0])
    with caplog.at_level(logging.WARNING, logger="spherical.pipeline.star_amplitude"):
        assert load_star_amplitude(tmp_path, 5, 3) == (None, None)
    assert "calibrate_star_amplitude" in caplog.text


def test_load_star_amplitude_ignores_a_mask_without_any_block(tmp_path, caplog):
    _write_star_amplitude(tmp_path, AMPLITUDE, [0, 0])
    with caplog.at_level(logging.WARNING, logger="spherical.pipeline.star_amplitude"):
        assert load_star_amplitude(tmp_path, 5, 2) == (None, None)
    assert "calibrate_star_amplitude" in caplog.text


def test_load_star_amplitude_uses_every_block_with_an_empty_mask(tmp_path, caplog):
    _write_star_amplitude(tmp_path, np.ones((2, 5)), np.zeros(0), mode="constant", reason="stable")
    with caplog.at_level(logging.WARNING, logger="spherical.pipeline.star_amplitude"):
        amplitude, mask = load_star_amplitude(tmp_path, 5, 4)
    np.testing.assert_array_equal(amplitude, np.ones((2, 5)))
    assert mask is None
    assert not caplog.records


def test_load_star_amplitude_without_files(tmp_path):
    assert load_star_amplitude(tmp_path, 5, 2) == (None, None)


def test_psf_block_mean_uses_only_the_masked_blocks():
    cube = np.stack([np.ones((1, 3, 3)), 3 * np.ones((1, 3, 3))], axis=1)
    np.testing.assert_allclose(psf_block_mean(cube, np.array([1, 0], dtype=bool)), np.ones((1, 3, 3)))
    np.testing.assert_allclose(psf_block_mean(cube, None), 2 * np.ones((1, 3, 3)))


def test_load_star_amplitude_sets_a_channel_without_amplitude_to_one(tmp_path, caplog):
    amplitude = np.array([[np.nan] * 5, [1.0, 0.8, np.nan, 1.2, 1.0]])
    _write_star_amplitude(tmp_path, amplitude, [1, 0])
    with caplog.at_level(logging.WARNING, logger="spherical.pipeline.star_amplitude"):
        loaded, mask = load_star_amplitude(tmp_path, 5, 2)
    np.testing.assert_array_equal(loaded[0], np.ones(5))
    np.testing.assert_allclose(loaded[1], amplitude[1].astype(np.float32))
    np.testing.assert_array_equal(mask, [True, False])
    assert "Channels [0]" in caplog.text


def test_load_star_amplitude_ignores_a_file_without_any_finite_channel(tmp_path, caplog):
    _write_star_amplitude(tmp_path, np.full((2, 5), np.nan), [1, 0])
    with caplog.at_level(logging.WARNING, logger="spherical.pipeline.star_amplitude"):
        assert load_star_amplitude(tmp_path, 5, 2) == (None, None)
    assert "calibrate_star_amplitude" in caplog.text


@pytest.mark.parametrize("reason", FALLBACK_STATUSES)
def test_load_star_amplitude_of_a_fallback_has_no_amplitude_but_keeps_its_mask(tmp_path, reason):
    _write_star_amplitude(tmp_path, np.ones((2, 5)), [1, 0], mode="constant", reason=reason)
    amplitude, mask = load_star_amplitude(tmp_path, 5, 2)
    assert amplitude is None
    np.testing.assert_array_equal(mask, [True, False])


def test_load_star_amplitude_of_a_fallback_with_an_empty_mask_uses_every_block(tmp_path):
    _write_star_amplitude(tmp_path, np.ones((2, 5)), np.zeros(0), mode="constant", reason=NO_VETTING)
    assert load_star_amplitude(tmp_path, 5, 4) == (None, None)


def test_load_star_amplitude_of_a_fallback_still_checks_its_mask(tmp_path, caplog):
    _write_star_amplitude(tmp_path, np.ones((2, 5)), [0, 0], mode="constant", reason=NO_WFS)
    with caplog.at_level(logging.WARNING, logger="spherical.pipeline.star_amplitude"):
        assert load_star_amplitude(tmp_path, 5, 2) == (None, None)
    assert "calibrate_star_amplitude" in caplog.text
