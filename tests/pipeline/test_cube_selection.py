"""Flux cube selection (#172)."""
import numpy as np
import pandas as pd
import pytest

from spherical.pipeline.cube_selection import (
    core_peaks,
    resolve_cube_selection,
    select_flux_cubes,
    summarise_cubes,
)


def _frames(origfiles, dits, nds, mjds):
    return pd.DataFrame({
        "ORIGFILE": origfiles, "DET SEQ1 DIT": dits,
        "INS4 FILT2 NAME": nds, "MJD": mjds,
    })


def _cubes(frames, peak_per_frame, exposure_level=None):
    """Summarise with a (n_wave=2, n_frames) peak array built from one value per frame."""
    peaks = np.tile(np.asarray(peak_per_frame, dtype=float), (2, 1))
    if exposure_level is None:
        exposure_level = np.asarray(frames["DET SEQ1 DIT"], dtype=float)
    return summarise_cubes(frames, np.asarray(exposure_level, dtype=float), peaks)


def _pi_men():
    """2 cubes, 90 + 8 DITs, as expand_frames_info produces them."""
    frames = _frames(
        origfiles=["a.fits"] * 90 + ["b.fits"] * 8,
        dits=[0.837464] * 90 + [8.0] * 8,
        nds=["ND_2.0"] * 90 + ["ND_1.0"] * 8,
        mjds=list(np.linspace(57376.10, 57376.11, 90)) + list(np.linspace(57376.18, 57376.19, 8)),
    )
    return _cubes(frames, [8598.0] * 90 + [47029.0] * 8)


def _saturated(cubes, threshold=40000.0):
    return cubes["peak_adu"].to_numpy() >= threshold


# --- core peaks -------------------------------------------------------------

def test_core_peaks_ignores_a_hot_pixel_outside_the_box():
    cube = np.zeros((2, 1, 200, 200))
    cube[:, 0, 100, 100] = 9000.0
    cube[:, 0, 10, 10] = 65000.0
    peaks = core_peaks(cube, guesses=[[(100, 100), (100, 100)]], box=15)
    assert peaks.shape == (2, 1)
    assert np.allclose(peaks[:, 0], 9000.0)


def test_core_peaks_uses_the_per_channel_guess():
    cube = np.zeros((2, 1, 200, 200))
    cube[0, 0, 100, 100] = 9000.0
    cube[1, 0, 130, 100] = 47000.0
    peaks = core_peaks(cube, guesses=[[(100, 100), (130, 100)]], box=15)
    assert peaks[0, 0] == pytest.approx(9000.0)
    assert peaks[1, 0] == pytest.approx(47000.0)


def test_core_peaks_accepts_one_guess_for_all_channels():
    """IFS passes one (cy, cx) per frame, not one per channel."""
    cube = np.zeros((3, 1, 100, 100))
    cube[:, 0, 50, 50] = [100.0, 200.0, 300.0]
    peaks = core_peaks(cube, guesses=[(50, 50)], box=5)
    assert peaks[:, 0].tolist() == [100.0, 200.0, 300.0]


# --- cube summary -----------------------------------------------------------

def test_summarise_cubes_collapses_dits_into_cubes():
    cubes = _pi_men()
    assert cubes["origfile"].tolist() == ["a.fits", "b.fits"]
    assert cubes["n_dits"].tolist() == [90, 8]
    assert cubes["dit"].tolist() == [0.837464, 8.0]
    assert cubes["peak_adu"].tolist() == [8598.0, 47029.0]


def test_one_outlier_frame_does_not_condemn_a_cube():
    frames = _frames(["a.fits"] * 5, [1.0] * 5, ["ND_2.0"] * 5, np.arange(5.0))
    cubes = _cubes(frames, [9000.0, 9000.0, 65000.0, 9000.0, 9000.0])
    assert not _saturated(cubes)[0]


def test_summarise_cubes_without_peaks_reports_nan():
    frames = _frames(["a.fits"] * 2, [1.0] * 2, ["ND_2.0"] * 2, [0.0, 1.0])
    cubes = summarise_cubes(frames, np.ones(2), peaks=None)
    assert np.isnan(cubes["peak_adu"].iloc[0])


def test_summarise_cubes_rejects_a_length_mismatch():
    frames = _frames(["a.fits"] * 2, [1.0] * 2, ["ND_2.0"] * 2, [0.0, 1.0])
    with pytest.raises(ValueError):
        summarise_cubes(frames, np.ones(3), peaks=None)


# --- automatic selection ----------------------------------------------------

def test_pi_men_drops_the_saturated_cube():
    """The saturated 8 s cube is the one adjacent to the science sequence."""
    cubes = _pi_men()
    keep, status = select_flux_cubes(cubes, _saturated(cubes), science_mid_mjd=57376.15)
    assert keep.tolist() == [True, False]
    assert status == "dropped_saturated"


def test_unsaturated_cubes_are_all_kept_whatever_their_setup():
    frames = _frames(["a.fits", "b.fits", "c.fits"], [0.84, 4.0, 4.0],
                     ["ND_2.0", "ND_1.0", "ND_1.0"], [1.0, 2.0, 5.0])
    cubes = _cubes(frames, [3000.0, 20000.0, 11000.0])
    keep, status = select_flux_cubes(cubes, _saturated(cubes), science_mid_mjd=3.0)
    assert keep.tolist() == [True, True, True]
    assert status == "all_kept"


def test_saturation_is_judged_per_cube_not_per_setup():
    """Same setup at both ends; better Strehl saturates only the last cube."""
    frames = _frames(["a.fits", "b.fits"], [4.0, 4.0], ["ND_1.0"] * 2, [1.0, 5.0])
    cubes = _cubes(frames, [30000.0, 45000.0])
    keep, _ = select_flux_cubes(cubes, _saturated(cubes), science_mid_mjd=3.0)
    assert keep.tolist() == [True, False]


def test_all_saturated_keeps_the_least_exposed_cube_by_level_not_dit():
    """8 s at ND_2.0 is fainter than 4 s at ND_1.0 despite the longer DIT."""
    frames = _frames(["a.fits", "b.fits"], [4.0, 8.0], ["ND_1.0", "ND_2.0"], [1.0, 2.0])
    cubes = _cubes(frames, [50000.0, 45000.0], exposure_level=[4.0 * 0.1, 8.0 * 0.01])
    keep, status = select_flux_cubes(cubes, _saturated(cubes), science_mid_mjd=1.5)
    assert keep.tolist() == [False, True]
    assert status == "all_saturated"


# --- user choices -----------------------------------------------------------

def _bracketing():
    """Saturated first attempt, corrected second, both before; one cube after."""
    frames = _frames(["a.fits", "b.fits", "c.fits"], [4.0, 1.0, 1.0],
                     ["ND_1.0", "ND_2.0", "ND_2.0"], [1.0, 1.1, 9.0])
    return _cubes(frames, [50000.0, 9000.0, 4500.0])


@pytest.mark.parametrize(
    "selection,expected,status",
    [("auto", [False, True, True], "dropped_saturated"),
     ("before", [False, True, False], "dropped_saturated"),
     ("after", [False, False, True], "all_kept"),
     ("all", [True, True, True], "forced"),
     (0, [True, False, False], "forced"),
     (-1, [False, False, True], "forced"),
     ("b.fits", [False, True, False], "forced")],
)
def test_user_selection(selection, expected, status):
    cubes = _bracketing()
    keep, got = select_flux_cubes(cubes, _saturated(cubes), science_mid_mjd=5.0, selection=selection)
    assert keep.tolist() == expected
    assert got == status


@pytest.mark.parametrize("selection", ["nope.fits", 3])
def test_selection_matching_no_cube_raises(selection):
    with pytest.raises(ValueError, match="a.fits"):
        resolve_cube_selection(_bracketing(), selection, science_mid_mjd=5.0)


def test_before_with_no_cube_before_raises():
    frames = _frames(["a.fits"], [1.0], ["ND_2.0"], [9.0])
    with pytest.raises(ValueError):
        resolve_cube_selection(_cubes(frames, [100.0]), "before", science_mid_mjd=5.0)
