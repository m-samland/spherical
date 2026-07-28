"""Moffat-based bad-pixel repair in the PSF core.

Used by :mod:`spherical.pipeline.steps.flux_psf_calibration` (Phase 2) to fill
in bad pixels that fall inside the Airy core of the flux PSF. Bad pixels near
the peak are a systematic problem for detectors with permanent defects at the
flux-PSF landing position (IRDIS K1 on SPHERE, for example): the upstream
neighbour interpolation used in ``irdis_preprocess`` under-estimates the peak
because the interpolants lie on the wing.

This module fits a 2D Moffat to the good pixels inside a small core window and
replaces *only* the bad pixels there with the model value. Everything outside
the core is left untouched, and if the Moffat residuals at good pixels exceed
a threshold the repair is skipped (fall back to the interpolated value with a
warning). The repaired-pixel inverse variance is downweighted so downstream
consumers that eventually read it can tell "this is a model value".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from astropy.modeling import fitting, models

Status = Literal[
    "repaired",
    "skipped_no_bpm",
    "skipped_edge",
    "skipped_insufficient_good_pixels",
    "skipped_bad_fit",
]


@dataclass
class RepairResult:
    """Outcome of one Moffat-repair pass on a single PSF window."""

    status: Status
    n_repaired: int
    residual_rms_frac: float  # RMS(residual)/amplitude at good core pixels
    amplitude: float
    fit_center_xy: Optional[Tuple[float, float]]  # in window coords
    window_out: np.ndarray  # possibly modified copy of the input window
    ivar_out: np.ndarray


def repair_psf_core(
    window: np.ndarray,
    ivar_window: np.ndarray,
    center_xy_in_window: Tuple[float, float],
    core_radius_px: float,
    residual_rms_frac_threshold: float = 0.10,
    downweight_ivar_factor: float = 0.1,
    min_good_core_pixels: int = 8,
) -> RepairResult:
    """Fit a 2D Moffat to the good pixels inside the PSF core and replace bad ones.

    Parameters
    ----------
    window
        2D window around the fitted PSF center. Values at bad pixels are the
        upstream-interpolated ones (from ``irdis_preprocess``); we ignore those
        for the fit and overwrite them if the fit succeeds.
    ivar_window
        Inverse variance for ``window``. Zero flags a bad pixel.
    center_xy_in_window
        (cx, cy) in window pixel coordinates. Typically comes from the
        pipeline's Gaussian centroid fit.
    core_radius_px
        Radius (px) of the core to fit and repair. A physically-motivated
        default is the Airy first-null radius, ``1.22 * lambda / D`` in pixels.
    residual_rms_frac_threshold
        If the Moffat residual RMS at good pixels exceeds this fraction of the
        fitted amplitude, the fit is deemed unreliable and the repair is
        skipped. Caller should log a WARNING and fall back to the upstream
        interpolation.
    downweight_ivar_factor
        Fraction of the median good-pixel ivar to assign to repaired pixels.
        ``0.1`` means repaired pixels get 10% of the typical good-pixel ivar,
        signaling "these are model values" to any downstream noise-weighted
        step.
    min_good_core_pixels
        Minimum number of good pixels inside the core disc required to attempt
        the 5-parameter Moffat fit. Fewer than this returns
        ``skipped_insufficient_good_pixels``.

    Returns
    -------
    RepairResult
        Includes the (possibly-modified) window and ivar; the caller writes
        these back into the frame it extracted them from.
    """
    win = np.asarray(window, dtype=float)
    if win.ndim != 2 or win.shape != np.asarray(ivar_window).shape:
        raise ValueError(
            f"window and ivar_window must be same-shape 2D arrays, got "
            f"{win.shape} and {np.asarray(ivar_window).shape}"
        )

    ny, nx = win.shape
    cx, cy = float(center_xy_in_window[0]), float(center_xy_in_window[1])
    if not (0 <= cx < nx and 0 <= cy < ny):
        return RepairResult(
            status="skipped_edge",
            n_repaired=0,
            residual_rms_frac=float("nan"),
            amplitude=float("nan"),
            fit_center_xy=None,
            window_out=win.copy(),
            ivar_out=np.asarray(ivar_window).copy(),
        )

    yy, xx = np.indices(win.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    core_disc = r <= core_radius_px
    bad = (np.asarray(ivar_window) == 0) | ~np.isfinite(win)
    bad_in_core = core_disc & bad
    n_bad_in_core = int(bad_in_core.sum())

    if n_bad_in_core == 0:
        return RepairResult(
            status="skipped_no_bpm",
            n_repaired=0,
            residual_rms_frac=float("nan"),
            amplitude=float("nan"),
            fit_center_xy=None,
            window_out=win.copy(),
            ivar_out=np.asarray(ivar_window).copy(),
        )

    good_in_core = core_disc & ~bad
    if good_in_core.sum() < min_good_core_pixels:
        return RepairResult(
            status="skipped_insufficient_good_pixels",
            n_repaired=0,
            residual_rms_frac=float("nan"),
            amplitude=float("nan"),
            fit_center_xy=None,
            window_out=win.copy(),
            ivar_out=np.asarray(ivar_window).copy(),
        )

    peak_guess = float(np.nanmax(win[good_in_core]))
    if not np.isfinite(peak_guess) or peak_guess <= 0:
        peak_guess = float(np.nanmax(np.abs(win[good_in_core])))
    m_init = models.Moffat2D(
        amplitude=peak_guess,
        x_0=cx,
        y_0=cy,
        gamma=max(core_radius_px / 2.0, 0.75),
        alpha=2.5,
        bounds={
            "amplitude": (0.0, None),
            "x_0": (cx - 1.5, cx + 1.5),
            "y_0": (cy - 1.5, cy + 1.5),
            "gamma": (0.3, 10.0),
            "alpha": (0.3, 10.0),
        },
    )
    fitter = fitting.LevMarLSQFitter()
    try:
        model = fitter(
            m_init,
            xx[good_in_core],
            yy[good_in_core],
            win[good_in_core],
        )
    except Exception:
        return RepairResult(
            status="skipped_bad_fit",
            n_repaired=0,
            residual_rms_frac=float("nan"),
            amplitude=float("nan"),
            fit_center_xy=None,
            window_out=win.copy(),
            ivar_out=np.asarray(ivar_window).copy(),
        )

    amp = float(model.amplitude.value)
    fit_cx = float(model.x_0.value)
    fit_cy = float(model.y_0.value)

    model_good = model(xx[good_in_core], yy[good_in_core])
    residuals = win[good_in_core] - model_good
    rms = float(np.sqrt(np.nanmean(residuals ** 2)))
    rms_frac = rms / max(amp, 1e-30)

    if (not np.isfinite(rms_frac)) or rms_frac > residual_rms_frac_threshold:
        return RepairResult(
            status="skipped_bad_fit",
            n_repaired=0,
            residual_rms_frac=rms_frac,
            amplitude=amp,
            fit_center_xy=(fit_cx, fit_cy),
            window_out=win.copy(),
            ivar_out=np.asarray(ivar_window).copy(),
        )

    win_out = win.copy()
    win_out[bad_in_core] = model(xx[bad_in_core], yy[bad_in_core])

    ivar_out = np.asarray(ivar_window, dtype=float).copy()
    median_good_ivar = float(np.nanmedian(ivar_out[good_in_core]))
    if not np.isfinite(median_good_ivar) or median_good_ivar <= 0:
        median_good_ivar = 0.0
    ivar_out[bad_in_core] = downweight_ivar_factor * median_good_ivar

    return RepairResult(
        status="repaired",
        n_repaired=n_bad_in_core,
        residual_rms_frac=rms_frac,
        amplitude=amp,
        fit_center_xy=(fit_cx, fit_cy),
        window_out=win_out,
        ivar_out=ivar_out,
    )
