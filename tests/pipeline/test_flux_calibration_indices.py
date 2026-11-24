"""``get_flux_calibration_indices`` across LST 0h (#193) and ``flux_block_ranges`` (#270)."""

import numpy as np
import pandas as pd

from spherical.pipeline.flux_calibration import (
    flux_block_ranges,
    get_flux_calibration_indices,
    plot_flux_normalization_factors,
)

SIDEREAL_HOURS_PER_DAY = 24 * 1.0027379


def _frames(minutes, exptime, lst_at_zero):
    """Frames at the given minutes after MJD 58000, with LST wrapped into [0, 24)."""
    minutes = np.asarray(minutes, dtype=float)
    mjd = 58000.0 + minutes / 1440.0
    lst = (lst_at_zero + (mjd - 58000.0) * SIDEREAL_HOURS_PER_DAY) % 24
    return pd.DataFrame({"MJD": mjd, "LST": lst, "EXPTIME": exptime})


def _sequence(lst_at_zero):
    """FLUX, CENTER, (CORO), CENTER, FLUX: two flux frames at each end."""
    flux = _frames([0.0, 0.1, 80.0, 80.1], 4.0, lst_at_zero)
    center = _frames([2.0, 78.0], 4.0, lst_at_zero)
    return center, flux


def test_blocks_and_pairing_do_not_depend_on_lst():
    """The same sequence gives the same result whether or not it crosses LST 0h."""
    for lst_at_zero in (22.0, 23.5):
        center, flux = _sequence(lst_at_zero)
        indices, discontinuities = get_flux_calibration_indices(center, flux)
        assert discontinuities.tolist() == [1]
        assert indices["flux_idx"].tolist() == [1, 2]
        assert indices["science_idx"].tolist() == [0, 1]


def test_nearest_center_frame_is_found_across_the_wrap():
    """A flux frame at LST 23.99h belongs with the CENTER frame at 0.01h, not the one at 23.8h."""
    flux = _frames([0.0], 4.0, lst_at_zero=23.99)
    center = _frames([-11.0, 1.2], 4.0, lst_at_zero=23.99)
    indices, _ = get_flux_calibration_indices(center, flux)
    assert indices["science_idx"].tolist() == [1]


def _flux_group(start_minute, n_frames=10):
    return [start_minute + 0.1 * i for i in range(n_frames)]


def _blocks(center_minutes, flux_minutes):
    center = _frames(center_minutes, 4.0, lst_at_zero=3.0)
    flux = _frames(flux_minutes, 4.0, lst_at_zero=3.0)
    _, discontinuities = get_flux_calibration_indices(center, flux)
    return flux_block_ranges(discontinuities, len(flux))


def test_opening_and_closing_groups_give_two_blocks():
    assert _blocks([5.0, 75.0], _flux_group(0.0) + _flux_group(80.0)) == [(0, 10), (10, 20)]


def test_middle_flux_group_is_one_block():
    """Its two ends pair with different CENTER frames, which must not split it."""
    flux = _flux_group(0.0) + _flux_group(60.0) + _flux_group(120.0)
    assert _blocks([5.0, 55.0, 66.0, 115.0], flux) == [(0, 10), (10, 20), (20, 30)]


def test_cubes_on_one_side_of_the_sequence_stay_separate_blocks():
    """Both pair with the same CENTER frame, which must not merge them."""
    flux = _flux_group(0.0) + _flux_group(10.0)
    assert _blocks([20.0, 80.0], flux) == [(0, 10), (10, 20)]


def test_groups_sharing_a_single_center_frame_stay_separate_blocks():
    assert _blocks([40.0], _flux_group(0.0) + _flux_group(80.0)) == [(0, 10), (10, 20)]


def test_contiguous_flux_frames_are_one_block():
    assert _blocks([20.0, 80.0], _flux_group(0.0)) == [(0, 10)]


def test_single_flux_frame_is_one_block():
    assert _blocks([20.0, 80.0], [0.0]) == [(0, 1)]


def test_pairing_table_keeps_one_row_per_center_frame():
    """``spot_to_flux`` pairs on this table, so the block fix must leave it alone."""
    flux = _frames(_flux_group(0.0) + _flux_group(60.0) + _flux_group(120.0), 4.0, lst_at_zero=3.0)
    center = _frames([5.0, 55.0, 66.0, 115.0], 4.0, lst_at_zero=3.0)
    indices, _ = get_flux_calibration_indices(center, flux)
    assert indices["flux_idx"].tolist() == [9, 10, 19, 20]
    assert indices["science_idx"].tolist() == [0, 1, 2, 3]


def test_normalization_plot_reads_indices_written_before_the_fix(tmp_path):
    """spot_to_flux may plot a flux_calibration_indices.csv from an older reduction."""
    old = pd.DataFrame({
        "flux_idx": [1, 2], "flux_lst": [23.9, 0.1],
        "science_idx": [0, 1], "science_lst": [23.92, 0.08], "lst_diff": [0.02, 0.02],
    })
    plot_flux_normalization_factors(old, np.ones((2, 3)), savefig=True, savedir=tmp_path)
    assert (tmp_path / "normalization_factors.png").exists()
