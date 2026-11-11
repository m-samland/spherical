"""FLUX frame vetting (#81).

The rules run on stamps measured by the step's own aperture photometry, with a
Moffat PSF broader than the diffraction limit, as a real PSF is. The end-to-end
tests run ``run_flux_psf_calibration`` on a synthetic three-channel sequence.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.modeling import models

from spherical.pipeline.flux_calibration import get_aperture_photometry
from spherical.pipeline.flux_vetting import (
    FluxVettingThresholds,
    NoUsableFluxFrameError,
    airy_sharpness,
    subset_photometry,
    vet_flux_frames,
)
from spherical.pipeline.steps.flux_psf_calibration import (
    STEP_PRODUCTS,
    finalize_psf_cube,
    first_combined_frame,
    run_flux_psf_calibration,
)

LOGGER = logging.getLogger("test")
STAMP = 57
IRDIS_H23 = np.array([1593.0, 1667.0])
IRDIS_PIXEL_MAS = 12.25
IFS_YJ = np.array([1000.0, 1100.0, 1200.0, 1300.0])
IFS_PIXEL_MAS = 7.46


def _moffat(amplitude=1000.0, gamma=3.0):
    yy, xx = np.indices((STAMP, STAMP))
    return models.Moffat2D(amplitude=amplitude, x_0=STAMP // 2, y_0=STAMP // 2, gamma=gamma, alpha=2.5)(xx, yy)


def _stamps(n_wave, n_frames, seed=0):
    noise = np.random.default_rng(seed).normal(0.0, 1.0, (n_wave, n_frames, STAMP, STAMP))
    return _moffat()[None, None] + noise


def _vet(stamps, wavelengths, pixel_scale_mas, centers=None, origfile=None, block=None):
    n_wave, n_frames = stamps.shape[:2]
    if centers is None:
        centers = np.full((n_wave, n_frames, 2), 300.0)
    origfile = np.array(["a.fits"] * n_frames) if origfile is None else np.asarray(origfile)
    block = np.zeros(n_frames, dtype=int) if block is None else np.asarray(block)
    photometry = get_aperture_photometry(stamps, aperture_radius_range=[1, 15],
                                         bg_aperture_inner_radius=15, bg_aperture_outer_radius=18)
    thresholds = FluxVettingThresholds.from_parameters({}, "IFS")
    return vet_flux_frames(photometry, centers, wavelengths, pixel_scale_mas, origfile, block, thresholds), photometry


def test_good_frames_are_all_kept():
    vetting, _ = _vet(_stamps(2, 6), IRDIS_H23, IRDIS_PIXEL_MAS)
    assert vetting.frame_kept.all()
    assert vetting.channel_kept.all()
    assert (vetting.table["decision"] == "kept").all()
    assert vetting.table["failed_tests"].eq("").all()


def test_frame_without_star_is_rejected_in_every_channel():
    stamps = _stamps(2, 6)
    stamps[:, 3] -= _moffat()
    vetting, _ = _vet(stamps, IRDIS_H23, IRDIS_PIXEL_MAS)
    assert vetting.frame_kept.tolist() == [True, True, True, False, True, True]
    rejected = vetting.table[vetting.table["frame"] == 3]
    assert (rejected["decision"] == "frame_rejected").all()
    assert not rejected["pass_frame_detection"].any()


def test_hot_pixel_fit_masks_only_its_channel():
    """A fit locked on a hot pixel has S/N in the hundreds and a stable position; only the Airy bound catches it."""
    stamps = _stamps(2, 6)
    stamps[1] = np.random.default_rng(1).normal(0.0, 1.0, (6, STAMP, STAMP))
    stamps[1, :, STAMP // 2, STAMP // 2] = 3000.0
    vetting, _ = _vet(stamps, IRDIS_H23, IRDIS_PIXEL_MAS)
    assert vetting.frame_kept.all()
    assert vetting.channel_kept[0].all()
    assert not vetting.channel_kept[1].any()
    hot = vetting.table[vetting.table["channel"] == 1]
    assert (hot["failed_tests"] == "sharpness_airy").all()
    assert (hot["snr"] > 100).all()


def test_blurred_frame_fails_the_lower_sharpness_bound():
    stamps = _stamps(2, 6)
    stamps[:, 2] += _moffat(gamma=6.0) * 0.5 - _moffat()
    vetting, _ = _vet(stamps, IRDIS_H23, IRDIS_PIXEL_MAS)
    assert not vetting.frame_kept[2]
    assert (vetting.table.loc[vetting.table["frame"] == 2, "failed_tests"].str.contains("sharpness_low")).all()


def test_faint_frame_fails_the_flux_test():
    stamps = _stamps(2, 6)
    stamps[:, 4] -= 0.7 * _moffat()
    vetting, _ = _vet(stamps, IRDIS_H23, IRDIS_PIXEL_MAS)
    assert vetting.frame_kept.tolist() == [True, True, True, True, False, True]
    assert vetting.table.loc[vetting.table["frame"] == 4, "relative_flux"].between(0.25, 0.35).all()


def test_position_reference_comes_from_detected_frames_of_the_same_file():
    """The star shows in the first frames only; the fits on noise wander, and do not move the reference."""
    stamps = _stamps(2, 8)
    stamps[:, 3:] -= _moffat()
    centers = np.full((2, 8, 2), 300.0)
    centers[:, 3:] = np.random.default_rng(2).uniform(100.0, 900.0, (2, 5, 2))
    vetting, _ = _vet(stamps, IRDIS_H23, IRDIS_PIXEL_MAS, centers=centers)
    assert vetting.frame_kept.tolist() == [True] * 3 + [False] * 5
    assert (vetting.table.loc[vetting.table["frame"] < 3, "offset_px"] == 0.0).all()


def test_displaced_fit_fails_position():
    centers = np.full((2, 6, 2), 300.0)
    centers[:, 5] += 20.0
    vetting, _ = _vet(_stamps(2, 6), IRDIS_H23, IRDIS_PIXEL_MAS, centers=centers)
    assert not vetting.frame_kept[5]
    assert vetting.frame_kept[:5].all()


def test_each_raw_file_has_its_own_position_reference():
    centers = np.full((2, 6, 2), 300.0)
    centers[:, 3:] = 450.0
    vetting, _ = _vet(_stamps(2, 6), IRDIS_H23, IRDIS_PIXEL_MAS, centers=centers,
                      origfile=["a.fits"] * 3 + ["b.fits"] * 3)
    assert vetting.frame_kept.all()


def test_ifs_frame_with_scattered_channel_centres_is_rejected():
    centers = np.full((4, 6, 2), 300.0)
    centers[:, 1] += np.random.default_rng(3).normal(0.0, 8.0, (4, 2))
    vetting, _ = _vet(_stamps(4, 6), IFS_YJ, IFS_PIXEL_MAS, centers=centers)
    assert not vetting.frame_kept[1]
    assert not vetting.table.loc[vetting.table["frame"] == 1, "pass_frame_detection"].any()


def test_one_weak_ifs_channel_is_masked_without_rejecting_the_frame():
    stamps = _stamps(4, 6)
    stamps[0, 2] -= _moffat()
    vetting, _ = _vet(stamps, IFS_YJ, IFS_PIXEL_MAS)
    assert vetting.frame_kept.all()
    assert not vetting.channel_kept[0, 2]
    assert vetting.channel_kept.sum() == 4 * 6 - 1


def test_relative_flux_is_per_block():
    stamps = _stamps(2, 6)
    stamps[:, 3:] += 2.0 * _moffat()
    vetting, _ = _vet(stamps, IRDIS_H23, IRDIS_PIXEL_MAS, block=[0, 0, 0, 1, 1, 1])
    assert vetting.frame_kept.all()
    assert np.allclose(vetting.table["relative_flux"], 1.0, atol=0.05)


def test_sharpness_is_per_block():
    stamps = _stamps(2, 6)
    stamps[:, 3:] += _moffat(gamma=6.0) * 0.5 - _moffat()
    vetting, _ = _vet(stamps, IRDIS_H23, IRDIS_PIXEL_MAS, block=[0, 0, 0, 1, 1, 1])
    assert vetting.frame_kept.all()
    assert np.allclose(vetting.table["relative_sharpness"], 1.0, atol=0.05)


def test_thresholds_come_from_the_reduction_parameters():
    thresholds = FluxVettingThresholds.from_parameters({"flux_vetting_max_offset_px": 7, "unrelated": 1}, "IFS")
    assert thresholds.max_offset_px == 7.0
    assert thresholds.frame_snr == 10.0


def test_frame_snr_threshold_is_per_instrument():
    from spherical.pipeline.pipeline_config import PreprocConfig

    parameters = {"flux_vetting_frame_snr_irdis": 50.0, "flux_vetting_frame_snr_ifs": 12.0}
    assert FluxVettingThresholds.from_parameters(parameters, "IRDIS").frame_snr == 50.0
    assert FluxVettingThresholds.from_parameters(parameters, "IFS").frame_snr == 12.0
    defaults = PreprocConfig()
    for instrument in ("irdis", "ifs"):
        thresholds = FluxVettingThresholds.from_parameters({}, instrument.upper())
        assert thresholds.frame_snr == getattr(defaults, f"flux_vetting_frame_snr_{instrument}")
        assert thresholds.channel_snr == getattr(defaults, f"flux_vetting_channel_snr_{instrument}")


def test_irdis_hot_pixel_channel_does_not_ride_on_a_bright_other_channel():
    """A frame without the star in K2 but a bright source in K1 keeps only K1, if anything."""
    stamps = _stamps(2, 6)
    stamps[1] = np.random.default_rng(6).normal(0.0, 1.0, (6, STAMP, STAMP))
    stamps[1, :, STAMP // 2 - 1:STAMP // 2 + 2, STAMP // 2 - 1:STAMP // 2 + 2] += 12.0
    photometry = get_aperture_photometry(stamps, aperture_radius_range=[1, 15],
                                         bg_aperture_inner_radius=15, bg_aperture_outer_radius=18)
    vetting = vet_flux_frames(photometry, np.full((2, 6, 2), 300.0), IRDIS_H23, IRDIS_PIXEL_MAS,
                              np.array(["a.fits"] * 6), np.zeros(6),
                              thresholds=FluxVettingThresholds.from_parameters({}, "IRDIS"))
    assert vetting.table.loc[vetting.table["channel"] == 1, "snr"].between(10, 50).all()
    assert not vetting.channel_kept[1].any()


def test_marginal_noise_fit_fails_irdis_detection():
    """A cube without the star where a noise peak reaches S/N 20 passes the IFS threshold but not the IRDIS one."""
    stamps = np.random.default_rng(5).normal(0.0, 1.0, (2, 6, STAMP, STAMP))
    stamps[:, :, STAMP // 2 - 1:STAMP // 2 + 2, STAMP // 2 - 1:STAMP // 2 + 2] += 12.0
    photometry = get_aperture_photometry(stamps, aperture_radius_range=[1, 15],
                                         bg_aperture_inner_radius=15, bg_aperture_outer_radius=18)
    args = (photometry, np.full((2, 6, 2), 300.0), IRDIS_H23, IRDIS_PIXEL_MAS, np.array(["a.fits"] * 6), np.zeros(6))
    snr = vet_flux_frames(*args, thresholds=FluxVettingThresholds.from_parameters({}, "IFS")).table["snr"]
    assert snr.between(10, 50).all()
    irdis = vet_flux_frames(*args, thresholds=FluxVettingThresholds.from_parameters({}, "IRDIS"))
    assert not irdis.frame_kept.any()


def test_airy_sharpness_matches_the_integrated_airy_pattern():
    from scipy.special import j1

    wavelength_nm = np.array([1000.0, 1600.0, 2200.0])
    lambda_over_d_px = wavelength_nm * 1e-9 / 7.99 * 206264.806e3 / IRDIS_PIXEL_MAS
    radius = np.linspace(1e-6, 5.0, 200001)
    expected = []
    for scale in lambda_over_d_px:
        x = np.pi * radius / scale
        integrand = (2 * j1(x) / x) ** 2 * radius
        cumulative = np.cumsum(integrand)
        expected.append(cumulative[np.searchsorted(radius, 3.0)] / cumulative[-1])
    assert np.allclose(airy_sharpness(wavelength_nm, IRDIS_PIXEL_MAS), expected, rtol=1e-3)


def test_subset_photometry_drops_rejected_frames_and_blanks_masked_channels():
    vetting, photometry = _vet(_stamps(2, 4), IRDIS_H23, IRDIS_PIXEL_MAS)
    frame_kept = np.array([True, False, True, True])
    channel_kept = np.ones((2, 4), dtype=bool)
    channel_kept[1, 2] = False
    subset = subset_photometry(photometry, frame_kept, channel_kept & frame_kept)
    assert subset["psf_flux_bg_corr_all"].shape == (14, 2, 3)
    assert np.isnan(subset["psf_flux_bg_corr_all"][:, 1, 1]).all()
    assert np.isfinite(subset["psf_flux_bg_corr_all"][:, 0]).all()
    assert subset["aperture_sizes"].tolist() == photometry["aperture_sizes"].tolist()


def test_finalize_keeps_empty_blocks_and_channels_nan_and_zeroes_partial_nan():
    cube = np.ones((3, 2, STAMP, STAMP))
    cube[0, 1] = np.nan
    cube[1] = np.nan
    cube[2, :, :2] = np.nan
    out = finalize_psf_cube(cube, LOGGER)
    assert np.isnan(out[0, 1]).all() and (out[0, 0] == 1.0).all()
    assert np.isnan(out[1]).all()
    assert (out[2, :, :2] == 0.0).all()
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        combined = np.nanmean(out, axis=1)
    assert np.isfinite(combined[[0, 2]]).all() and np.isnan(combined[1]).all()


def test_finalize_turns_a_block_of_infinities_into_nan():
    """A frame with zero aperture flux divides its stamp by zero."""
    cube = np.ones((1, 2, STAMP, STAMP))
    cube[0, 1] = np.inf
    out = finalize_psf_cube(cube, LOGGER)
    assert np.isnan(out[0, 1]).all()
    assert (np.nanmean(out, axis=1) == 1.0).all()


# --- run_flux_psf_calibration end to end -----------------------------------

WAVELENGTHS = np.array([1100.0, 1200.0, 1300.0])
N_FRAMES, SIZE, DIT = 6, 160, 2.0
CENTER_XY = (100.3, 99.6)
REDUCTION_PARAMETERS = {
    "flux_combination_method": "median",
    "exclude_first_flux_frame": False,
    "exclude_first_flux_frame_all": False,
    "flux_cube_selection_irdis": "auto",
    "flux_cube_selection_ifs": "all",
    "flux_saturation_adu": 40000.0,
    "flux_nonlinearity_adu": 30000.0,
}


def _flux_cube(frames_without_star=(), n_frames=N_FRAMES):
    yy, xx = np.indices((SIZE, SIZE))
    psf = models.Moffat2D(amplitude=1000.0, x_0=CENTER_XY[0], y_0=CENTER_XY[1], gamma=3.0, alpha=2.5)(xx, yy)
    cube = np.random.default_rng(0).normal(5.0, 0.5, (len(WAVELENGTHS), n_frames, SIZE, SIZE))
    with_star = np.setdiff1d(np.arange(n_frames), frames_without_star)
    cube[:, with_star] += psf
    return cube


def _frames_info(mjd0, n):
    mjd = mjd0 + np.arange(n) * DIT / 86400
    return pd.DataFrame({
        "MJD": mjd, "LST": mjd, "EXPTIME": DIT, "DET SEQ1 DIT": DIT, "INS4 FILT2 NAME": "OPEN",
        "ORIGFILE": "cube.fits", "DIT INDEX": np.arange(n),
    })


def _run(tmp_path, cube, **parameters):
    fits.writeto(tmp_path / "wavelengths.fits", WAVELENGTHS)
    fits.writeto(tmp_path / "flux_cube.fits", cube)
    fits.writeto(tmp_path / "flux_ivar_cube.fits", np.ones_like(cube))
    _frames_info(60000.0, cube.shape[1]).to_csv(tmp_path / "frames_info_flux.csv", index=False)
    _frames_info(60000.01, 2).to_csv(tmp_path / "frames_info_center.csv", index=False)
    run_flux_psf_calibration(str(tmp_path), {**REDUCTION_PARAMETERS, **parameters}, logger=LOGGER)


def test_step_removes_a_frame_without_star_from_every_product(tmp_path):
    _run(tmp_path, _flux_cube(frames_without_star=(3,)))

    vetting = pd.read_csv(tmp_path / "additional_outputs" / "flux_frame_vetting.csv")
    assert len(vetting) == len(WAVELENGTHS) * N_FRAMES
    assert set(vetting.loc[vetting["decision"] == "frame_rejected", "frame"]) == {3}
    assert set(vetting.columns) >= {"DIT INDEX", "MJD", "origfile", "decision", "failed_tests"}

    selected = pd.read_csv(tmp_path / "frames_info_flux_selected.csv")
    assert selected["DIT INDEX"].tolist() == [0, 1, 2, 4, 5]
    for name in ("flux_amplitude_calibrated.fits",):
        assert fits.getdata(tmp_path / name).shape[-1] == 5
    for name in ("flux_centers.fits", "flux_stamps_uncalibrated.fits", "flux_stamps_dit_nd_calibrated.fits",
                 "nd_attenuation.fits", "flux_gauss_amplitudes.fits"):
        assert fits.getdata(tmp_path / "additional_outputs" / name).shape[1] == 5
    psf = fits.getdata(tmp_path / "psf_cube_for_postprocessing.fits")
    assert np.isfinite(psf).all() and psf.max() > 500


def test_step_keeps_every_frame_when_vetting_is_off(tmp_path):
    _run(tmp_path, _flux_cube(frames_without_star=(3,)), flux_frame_vetting=False)
    assert fits.getdata(tmp_path / "flux_amplitude_calibrated.fits").shape[-1] == N_FRAMES
    assert not (tmp_path / "additional_outputs" / "flux_frame_vetting.csv").exists()


def test_step_stops_when_no_frame_shows_the_star(tmp_path):
    for name in STEP_PRODUCTS:
        (tmp_path / name).write_text("from an earlier run")
    with pytest.raises(NoUsableFluxFrameError, match="No FLUX frame passes the vetting"):
        _run(tmp_path, _flux_cube(frames_without_star=range(N_FRAMES)))
    vetting = pd.read_csv(tmp_path / "additional_outputs" / "flux_frame_vetting.csv")
    assert (vetting["decision"] == "frame_rejected").all()
    assert not any((tmp_path / name).exists() for name in STEP_PRODUCTS)


def test_rejected_run_of_frames_does_not_split_its_block(tmp_path):
    """Twenty rejected frames leave a 42 s gap, more than 15 DITs, inside one block."""
    _run(tmp_path, _flux_cube(frames_without_star=range(4, 24), n_frames=28))
    assert len(pd.read_csv(tmp_path / "frames_info_flux_selected.csv")) == 8
    assert fits.getdata(tmp_path / "psf_cube_for_postprocessing.fits").shape[1] == 1
    assert len(pd.read_csv(tmp_path / "flux_calibration_indices.csv")) == 1


def _first_frame_calls(monkeypatch):
    from spherical.pipeline.steps import flux_psf_calibration

    calls = []
    original = flux_psf_calibration.first_combined_frame

    def spy(block_index, n_frames, parameters, first_frame_present=True):
        calls.append((int(block_index), n_frames, bool(first_frame_present)))
        return original(block_index, n_frames, parameters, first_frame_present)

    monkeypatch.setattr(flux_psf_calibration, "first_combined_frame", spy)
    return calls


def test_rejected_first_frame_is_not_followed_by_a_second_exclusion(tmp_path, monkeypatch):
    calls = _first_frame_calls(monkeypatch)
    _run(tmp_path, _flux_cube(frames_without_star=(0,)), exclude_first_flux_frame=True)
    assert calls == [(0, N_FRAMES - 1, False)]


def test_first_frame_flag_follows_the_block_position_before_the_vetting():
    params = {"exclude_first_flux_frame": True, "exclude_first_flux_frame_all": False}
    assert first_combined_frame(1, 3, params) == 0
    assert first_combined_frame(0, 3, params, first_frame_present=False) == 0
    assert first_combined_frame(0, 3, params) == 1


def test_step_masks_one_failed_channel_and_keeps_the_frame(tmp_path):
    cube = _flux_cube()
    cube[2, 3] = np.random.default_rng(1).normal(5.0, 0.5, (SIZE, SIZE))
    _run(tmp_path, cube)

    assert len(pd.read_csv(tmp_path / "frames_info_flux_selected.csv")) == N_FRAMES
    amplitude = fits.getdata(tmp_path / "flux_amplitude_calibrated.fits")
    masked = np.zeros(amplitude.shape[1:], dtype=bool)
    masked[2, 3] = True
    assert np.isnan(amplitude[:, masked]).all() and np.isfinite(amplitude[:, ~masked]).all()
    psf = fits.getdata(tmp_path / "psf_cube_for_postprocessing.fits")
    assert np.isfinite(psf).all() and psf[:, 0].max() > 500


def test_step_writes_nan_psf_for_a_channel_without_any_good_frame(tmp_path):
    cube = _flux_cube()
    cube[2] = np.random.default_rng(1).normal(5.0, 0.5, (N_FRAMES, SIZE, SIZE))
    _run(tmp_path, cube)

    psf = fits.getdata(tmp_path / "psf_cube_for_postprocessing.fits")
    assert np.isnan(psf[2]).all()
    assert np.isfinite(psf[:2]).all()
