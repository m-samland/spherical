"""``get_flux_calibration_indices`` across LST 0h (#193)."""

import numpy as np
import pandas as pd

from spherical.pipeline.flux_calibration import (
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


def test_normalization_plot_reads_indices_written_before_the_fix(tmp_path):
    """spot_to_flux may plot a flux_calibration_indices.csv from an older reduction."""
    old = pd.DataFrame({
        "flux_idx": [1, 2], "flux_lst": [23.9, 0.1],
        "science_idx": [0, 1], "science_lst": [23.92, 0.08], "lst_diff": [0.02, 0.02],
    })
    plot_flux_normalization_factors(old, np.ones((2, 3)), savefig=True, savedir=tmp_path)
    assert (tmp_path / "normalization_factors.png").exists()
