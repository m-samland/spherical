import logging

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from spherical.pipeline.star_amplitude import (
    FALLBACK_STATUSES,
    NO_VETTING,
    NO_WFS,
    anchor,
    block_disagreement,
    block_table,
    block_wfs_level,
    classify,
    compatible_segments,
    fill_in_time,
    load_star_amplitude,
    psf_block_mean,
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


def test_wfs_samples_keep_direct_positive_values_of_the_accepted_segments():
    m = pd.DataFrame({"visloop_flux_subap": [1.0, 2.0, 3.0, -1.0, 5.0, 6.0],
                      "flag_visloop": ["direct", "direct", "interpolated", "direct", "direct", "direct"],
                      "segment": [0, 0, 0, 0, 1, -1]})
    np.testing.assert_array_equal(wfs_samples(m, {0}), [1.0, 2.0, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(wfs_samples(m, set()), [np.nan] * 6)
    np.testing.assert_array_equal(wfs_samples(m, None), [1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
    np.testing.assert_array_equal(wfs_samples(m.drop(columns="segment"), None), [1.0, 2.0, np.nan, np.nan, 5.0, 6.0])


def test_a_table_without_segment_column_keeps_its_direct_samples_whatever_the_accepted_segments():
    m = pd.DataFrame({"visloop_flux_subap": [1.0, 2.0, 3.0, -1.0],
                      "flag_visloop": ["direct", "direct", "interpolated", "direct"]})
    for segments in ({0}, set(), None):
        np.testing.assert_array_equal(wfs_samples(m, segments), [1.0, 2.0, np.nan, np.nan])


def _config(rows):
    return pd.DataFrame(rows, columns=["dp_id", "segment", "wfs_mode", "spatial_filter", "wfs_filter", "nd_filter"])


def test_flux_frames_behind_the_nd_share_the_wfs_configuration_of_the_science_frames():
    config = _config([("a", 0, "Wide", "FILT_M", "OPEN", "ND_2.0"), ("b", 1, "Wide", "FILT_M", "OPEN", "OPEN"),
                      ("c", 1, "Wide", "FILT_M", "OPEN", "OPEN"), ("d", 2, "Wide", "FILT_M", "OPEN", "ND_2.0")])
    segments = compatible_segments(config, 1)
    assert segments == {0, 1, 2}
    flux = pd.DataFrame({"visloop_flux_subap": [1.0, 1.0, 1.0, 1.0], "flag_visloop": ["direct"] * 4,
                         "segment": [0, 0, 2, 2]})
    science = pd.DataFrame({"visloop_flux_subap": [1.0, 1.0], "flag_visloop": ["direct"] * 2, "segment": [1, 1]})
    assert np.isfinite(wfs_samples(flux, segments)).all() and np.isfinite(wfs_samples(science, segments)).all()


def test_a_segment_with_another_spatial_filter_is_missing():
    config = _config([("a", 0, "Wide", "FILT_S", "OPEN", "ND_2.0"), ("b", 1, "Wide", "FILT_M", "OPEN", "OPEN"),
                      ("c", 2, "Wide", "FILT_M", "OPEN", "ND_2.0")])
    segments = compatible_segments(config, 1)
    assert segments == {1, 2}
    flux = pd.DataFrame({"visloop_flux_subap": [1.0, 1.0, 1.0], "flag_visloop": ["direct"] * 3, "segment": [0, 2, 3]})
    np.testing.assert_array_equal(wfs_samples(flux, segments), [np.nan, 1.0, np.nan])


def test_compatible_segments_without_a_usable_config_table_accept_every_segment():
    assert compatible_segments(None, 1) is None
    assert compatible_segments(_config([]).drop(columns="spatial_filter"), 1) is None


def test_compatible_segments_are_empty_for_a_reference_outside_the_config_table():
    config = _config([("a", 0, "Wide", "FILT_M", "OPEN", "OPEN")])
    assert compatible_segments(config, 3) == set()
    assert compatible_segments(config, None) == set()


def test_missing_configuration_values_compare_equal():
    config = _config([("a", 0, None, "FILT_M", "OPEN", "ND_2.0"), ("b", 1, None, "FILT_M", "OPEN", "OPEN")])
    assert compatible_segments(config, 1) == {0, 1}


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
    table = block_table(flux, np.ones((1, 4), bool), wfs_samples(matched, {0}), np.array([0, 0, 1, 1]), np.ones_like(flux))
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
