"""Core bad pixels must not lower the FLUX photometry (#269).

Runs ``run_flux_psf_calibration`` on a synthetic three-channel FLUX sequence
with a Moffat PSF, once clean and once with bad pixels in the core, and compares
the radius-3 aperture flux in ``flux_amplitude_calibrated.fits``.
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.modeling import models

from spherical.pipeline.psf_repair import repair_psf_core
from spherical.pipeline.steps import flux_psf_calibration
from spherical.pipeline.steps.flux_psf_calibration import run_flux_psf_calibration

LOGGER = logging.getLogger("test")
WAVELENGTHS = np.array([1100.0, 1200.0, 1300.0])
N_FRAMES, SIZE, DIT = 4, 128, 2.0
CENTER_XY = (64.3, 63.6)
APERTURE_3PX = 2
REDUCTION_PARAMETERS = {
    "flux_combination_method": "median",
    "exclude_first_flux_frame": False,
    "exclude_first_flux_frame_all": False,
    "flux_cube_selection_irdis": "auto",
    "flux_cube_selection_ifs": "all",
    "flux_saturation_adu": 40000.0,
    "flux_nonlinearity_adu": 30000.0,
}


def _flux_cube():
    yy, xx = np.indices((SIZE, SIZE))
    psf = models.Moffat2D(amplitude=1000.0, x_0=CENTER_XY[0], y_0=CENTER_XY[1], gamma=2.0, alpha=2.5)(xx, yy)
    noise = np.random.default_rng(0).normal(0.0, 0.5, (len(WAVELENGTHS), N_FRAMES, SIZE, SIZE))
    return psf[None, None] + 5.0 + noise


def _frames_info(mjd0, n):
    mjd = mjd0 + np.arange(n) * DIT / 86400
    return pd.DataFrame({
        "MJD": mjd, "LST": mjd, "EXPTIME": DIT, "DET SEQ1 DIT": DIT, "INS4 FILT2 NAME": "OPEN",
        "ORIGFILE": "cube.fits", "DIT INDEX": np.arange(n),
    })


def _run(tmp_path, cube, ivar):
    fits.writeto(tmp_path / "wavelengths.fits", WAVELENGTHS)
    fits.writeto(tmp_path / "flux_cube.fits", cube)
    fits.writeto(tmp_path / "flux_ivar_cube.fits", ivar)
    _frames_info(60000.0, N_FRAMES).to_csv(tmp_path / "frames_info_flux.csv", index=False)
    _frames_info(60000.01, 2).to_csv(tmp_path / "frames_info_center.csv", index=False)
    run_flux_psf_calibration(str(tmp_path), REDUCTION_PARAMETERS, logger=LOGGER)
    return fits.getdata(tmp_path / "flux_amplitude_calibrated.fits")[APERTURE_3PX]


@pytest.fixture(scope="module")
def clean_flux(tmp_path_factory):
    cube = _flux_cube()
    return _run(tmp_path_factory.mktemp("clean"), cube, np.ones_like(cube))


def test_repaired_core_pixel_gives_the_clean_flux(tmp_path, clean_flux):
    """The flagged pixel holds garbage, so only the Moffat value can restore the flux."""
    cube = _flux_cube()
    ivar = np.ones_like(cube)
    peak_y, peak_x = round(CENTER_XY[1]), round(CENTER_XY[0])
    cube[:, :, peak_y, peak_x] = 0.0
    ivar[:, :, peak_y, peak_x] = 0.0

    flux = _run(tmp_path, cube, ivar)

    assert np.allclose(flux, clean_flux, rtol=5e-3)
    assert (tmp_path / "psf_cube_for_postprocessing_unrepaired.fits").exists()


def test_skipped_repair_counts_the_interpolated_values(tmp_path, clean_flux, monkeypatch):
    """A skipped repair keeps the upstream value, here exact, in the sum instead of masking it."""
    def skip(window, ivar_window, **kwargs):
        result = repair_psf_core(window, ivar_window, **kwargs)
        return dataclasses.replace(result, status="skipped_bad_fit", n_repaired=0,
                                   window_out=np.array(window, dtype=float), ivar_out=np.array(ivar_window))

    monkeypatch.setattr(flux_psf_calibration, "repair_psf_core", skip)
    cube = _flux_cube()
    ivar = np.ones_like(cube)
    ivar[:, :, round(CENTER_XY[1]), round(CENTER_XY[0])] = 0.0

    flux = _run(tmp_path, cube, ivar)

    log = pd.read_csv(tmp_path / "additional_outputs" / "flux_psf_repair_log.csv")
    assert (log["status"] == "skipped_bad_fit").all()
    assert np.allclose(flux, clean_flux, rtol=1e-3)
