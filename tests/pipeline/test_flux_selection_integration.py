"""Flux cube selection wired into the calibration step (#172)."""
import numpy as np
import pandas as pd

from spherical.pipeline.flux_calibration import get_flux_calibration_indices
from spherical.pipeline.steps.flux_psf_calibration import apply_flux_cube_selection


def _saturated_first(n_wave=2, per_channel_guesses=True, clean_peak=9000.0):
    """2 saturated DITs in a first cube, then 3 clean ones: the common pattern."""
    cube = np.zeros((n_wave, 5, 41, 41))
    cube[:, :2, 20, 20] = 50000.0
    cube[:, 2:, 20, 20] = clean_peak
    ivar = np.arange(5.0)[None, :, None, None] * np.ones_like(cube)
    bpm = ivar == 0
    frames = pd.DataFrame({
        "ORIGFILE": ["a.fits"] * 2 + ["b.fits"] * 3,
        "DET SEQ1 DIT": [8.0] * 2 + [0.837464] * 3,
        "INS4 FILT2 NAME": ["ND_1.0"] * 2 + ["ND_2.0"] * 3,
        "MJD": [1.00, 1.01, 1.05, 1.06, 1.07],
        "DIT INDEX": [0, 1, 0, 1, 2],
    })
    guess = [(20, 20)] * n_wave if per_channel_guesses else (20, 20)
    guesses = [guess for _ in range(5)]
    return cube, ivar, bpm, frames, guesses


def _select(selection="auto", **kwargs):
    cube, ivar, bpm, frames, guesses = _saturated_first(**kwargs)
    return apply_flux_cube_selection(
        cube, ivar, bpm, frames, guesses,
        exposure_level=np.asarray(frames["DET SEQ1 DIT"], dtype=float),
        science_mid_mjd=1.5, threshold_adu=40000.0, nonlinearity_adu=30000.0,
        selection=selection,
    )


def test_every_co_indexed_array_is_subset_together():
    cube, ivar, bpm, frames, guesses, report = _select()
    assert cube.shape[1] == ivar.shape[1] == bpm.shape[1] == len(frames) == len(guesses) == 3
    assert set(frames["ORIGFILE"]) == {"b.fits"}
    assert ivar[0, :, 0, 0].tolist() == [2.0, 3.0, 4.0]


def test_frames_are_reindexed_and_start_on_a_first_readout():
    _, _, _, frames, _, _ = _select()
    assert frames.index.tolist() == [0, 1, 2]
    assert frames["DIT INDEX"].iloc[0] == 0


def test_report_records_every_input_cube():
    *_, report = _select()
    assert report["origfile"].tolist() == ["a.fits", "b.fits"]
    assert report["saturated"].tolist() == [True, False]
    assert report["nonlinear"].tolist() == [False, False]
    assert report["kept"].tolist() == [False, True]
    assert (report["status"] == "dropped_saturated").all()


def test_kept_cube_above_the_nonlinearity_level_is_flagged():
    *_, report = _select(clean_peak=32000.0)
    assert report["kept"].tolist() == [False, True]
    assert report["nonlinear"].tolist() == [False, True]


def test_all_keeps_everything_without_measuring():
    cube, *_, report = _select(selection="all")
    assert cube.shape[1] == 5
    assert report["peak_adu"].isna().all()
    assert not report["nonlinear"].any()


def test_ifs_guesses_work_in_every_mode():
    """IFS carries one guess per frame, not per channel."""
    for selection in ("all", "auto"):
        cube, *_ = _select(selection=selection, n_wave=3, per_channel_guesses=False)
        assert cube.shape[0] == 3


def test_mixed_dits_do_not_split_a_long_dit_cube():
    """0.84 s cube, then a contiguous 32 s cube: one block, no discontinuity."""
    seconds = np.array([0, 1, 2, 60, 93, 126], dtype=float)
    exptime = [0.837464] * 3 + [32.0] * 3
    flux = pd.DataFrame({"MJD": 58000.0 + seconds / 86400.0, "LST": 3.0 + seconds / 3600.0,
                         "EXPTIME": exptime})
    center = pd.DataFrame({"MJD": [58000.0 + 360.0 / 86400.0], "LST": [3.1]})
    _, discontinuities = get_flux_calibration_indices(center, flux)
    assert discontinuities.tolist() == []


def test_block_normalization_is_robust_to_one_bad_frame():
    """One frame at 1.5x (a failed background, bad core pixels) must not rescale the others (#159)."""
    from spherical.pipeline.steps.flux_psf_calibration import block_normalization

    flux = np.array([[100.0, 100.0, 150.0, 100.0, 100.0]])
    assert block_normalization(flux, first_combined=0)[0].tolist() == [1.0, 1.0, 1.5, 1.0, 1.0]


def test_block_normalization_reference_skips_excluded_first_frame():
    from spherical.pipeline.steps.flux_psf_calibration import block_normalization

    flux = np.array([[500.0, 100.0, 100.0, 100.0]])
    assert block_normalization(flux, first_combined=1)[0].tolist() == [5.0, 1.0, 1.0, 1.0]
