"""Process extracted centers: instrument-dispatched center fitting.

For IFS: polynomial-across-wavelength two-pass fit with sigma-clipping. For
IRDIS (waffle-CENTER path): temporal moving-median outlier flagging + local
median replacement (2 wavelength points make a polynomial across wavelength
meaningless). For IRDIS (non-waffle, with CORO): DMS-header offset
propagation from CENTER waffle measurements (Task 4).
"""
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip
from scipy.ndimage import median_filter

from spherical.pipeline.logging_utils import optional_logger


@optional_logger
def run_polynomial_center_fit(
    converted_dir: str,
    observation,
    extraction_parameters: Dict[str, str | bool],
    non_least_square_methods: List[str],
    logger,
) -> None:
    """Fit / smooth star center positions for IFS or IRDIS.

    Dispatches on ``observation.observation['INSTRUMENT'][0]``. IFS behavior
    is byte-identical to the previous implementation; IRDIS gets a per-channel
    temporal moving-median outlier flag with local-median replacement.

    Parameters
    ----------
    converted_dir : str
        Directory containing ``image_centers.fits`` and ``wavelengths.fits``.
    observation : IFSObservation | IRDISObservation
        Observation carrying ``observation`` (metadata Table) and ``frames`` (dict).
    extraction_parameters : dict
        IFS-only; keys ``method`` (str) and ``linear_wavelength`` (bool).
    non_least_square_methods : list of str
        IFS-only.
    """
    logger.info("Starting center fit", extra={"step": "polynomial_center_fit", "status": "started"})
    instrument = str(observation.observation["INSTRUMENT"][0]).upper()

    if instrument == "IRDIS":
        _run_irdis_temporal_center_fit(converted_dir, observation, logger)
        return

    _run_ifs_polynomial_center_fit(converted_dir, extraction_parameters, non_least_square_methods, logger)


def _run_irdis_temporal_center_fit(converted_dir: str, observation, logger) -> None:
    coro_frames = observation.frames.get("CORO")
    if coro_frames is not None and len(coro_frames) > 0:
        _run_irdis_dms_propagation(converted_dir, observation, logger)
        return

    image_centers = np.asarray(
        fits.getdata(os.path.join(converted_dir, "image_centers.fits")),
        dtype=np.float32,
    )
    n_wave, n_time, _ = image_centers.shape
    robust = image_centers.copy()

    additional_outputs = Path(converted_dir) / "additional_outputs"
    additional_outputs.mkdir(exist_ok=True)

    outliers_per_ch: list[np.ndarray] = []
    box = 21
    for ch in range(n_wave):
        x = image_centers[ch, :, 0].astype(np.float32)
        y = image_centers[ch, :, 1].astype(np.float32)
        x_med = median_filter(x, size=box, mode="nearest")
        y_med = median_filter(y, size=box, mode="nearest")
        rx = x - x_med
        ry = y - y_med
        sx = float(mad_std(rx, ignore_nan=True) or 0.0)
        sy = float(mad_std(ry, ignore_nan=True) or 0.0)
        outlier_x = np.abs(rx) > 5.0 * sx if sx > 0 else np.zeros_like(rx, dtype=bool)
        outlier_y = np.abs(ry) > 5.0 * sy if sy > 0 else np.zeros_like(ry, dtype=bool)
        nan_mask = ~(np.isfinite(x) & np.isfinite(y))
        replace = outlier_x | outlier_y | nan_mask

        robust[ch, replace, 0] = x_med[replace]
        robust[ch, replace, 1] = y_med[replace]
        idx = np.where(replace)[0].astype(np.int32)
        outliers_per_ch.append(idx)
        logger.info(
            f"IRDIS ch{ch}: {int(replace.sum())} outlier frames (5σ moving-median)",
            extra={"step": "polynomial_center_fit", "status": "info"},
        )

    fits.writeto(
        os.path.join(converted_dir, "image_centers_fitted_robust.fits"),
        robust,
        overwrite=True,
    )

    k_max = max((arr.size for arr in outliers_per_ch), default=0)
    packed = np.full((n_wave, max(k_max, 1)), -1, dtype=np.int32)
    for ch, arr in enumerate(outliers_per_ch):
        packed[ch, : arr.size] = arr
    fits.writeto(
        str(additional_outputs / "center_outlier_frames.fits"),
        packed,
        overwrite=True,
    )

    logger.info("Step finished", extra={"step": "polynomial_center_fit", "status": "success"})


def _run_irdis_dms_propagation(converted_dir: str, observation, logger) -> None:
    """Propagate CENTER waffle centers to CORO frames via DMS-header offsets.

    Reads ``INS1 PAC X/Y`` (µm) from ``frames_info_{center,coro}.csv``,
    converts to pixels using the IRDIS DMS pixel scale 18 µm/px, and shifts
    the nearest-in-time CENTER-frame center by that delta for each CORO
    frame. The reference dataset is continuous-waffle, so this branch is
    validated only against synthetic inputs in Phase 5; real end-to-end
    validation is deferred.
    """
    import pandas as pd

    PIXEL_SCALE_UM = 18.0

    center_centers = np.asarray(
        fits.getdata(os.path.join(converted_dir, "image_centers.fits")),
        dtype=np.float32,
    )
    frames_center = pd.read_csv(os.path.join(converted_dir, "frames_info_center.csv"))
    frames_coro = pd.read_csv(os.path.join(converted_dir, "frames_info_coro.csv"))

    mjd_center = frames_center["MJD_OBS"].to_numpy()
    pac_x_center = frames_center["INS1 PAC X"].to_numpy() / PIXEL_SCALE_UM
    pac_y_center = frames_center["INS1 PAC Y"].to_numpy() / PIXEL_SCALE_UM

    mjd_coro = frames_coro["MJD_OBS"].to_numpy()
    pac_x_coro = frames_coro["INS1 PAC X"].to_numpy() / PIXEL_SCALE_UM
    pac_y_coro = frames_coro["INS1 PAC Y"].to_numpy() / PIXEL_SCALE_UM

    n_wave = center_centers.shape[0]
    n_coro = len(frames_coro)
    propagated = np.zeros((n_wave, n_coro, 2), dtype=np.float32)

    nearest_idx = np.argmin(np.abs(mjd_coro[:, None] - mjd_center[None, :]), axis=1)
    for ch in range(n_wave):
        for i, ci in enumerate(nearest_idx):
            dx = pac_x_coro[i] - pac_x_center[ci]
            dy = pac_y_coro[i] - pac_y_center[ci]
            propagated[ch, i, 0] = center_centers[ch, ci, 0] - dx
            propagated[ch, i, 1] = center_centers[ch, ci, 1] - dy

    fits.writeto(
        os.path.join(converted_dir, "image_centers_fitted_robust.fits"),
        propagated,
        overwrite=True,
    )
    logger.info(
        f"IRDIS DMS propagation: {n_coro} CORO frames from {len(frames_center)} CENTER frames.",
        extra={"step": "polynomial_center_fit", "status": "info"},
    )


def _run_ifs_polynomial_center_fit(
    converted_dir: str,
    extraction_parameters: Dict[str, str | bool],
    non_least_square_methods: List[str],
    logger,
) -> None:
    plot_dir = os.path.join(converted_dir, "center_plots/")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    image_centers = fits.getdata(os.path.join(converted_dir, "image_centers.fits"))
    wavelengths = fits.getdata(os.path.join(converted_dir, "wavelengths.fits"))
    wavelengths = np.array(wavelengths).flatten()
    image_centers = np.array(image_centers)
    if (
        extraction_parameters["method"] in non_least_square_methods
        and extraction_parameters["linear_wavelength"] is True
    ):
        remove_indices = [0, 1, 21, 38]
    else:
        remove_indices = [0, 13, 19, 20]
    anomalous_centers_mask = np.zeros(wavelengths.shape, dtype=bool)
    anomalous_centers_mask[remove_indices] = True
    image_centers_fitted = np.zeros_like(image_centers)
    image_centers_fitted2 = np.zeros_like(image_centers)
    for frame_idx in range(image_centers.shape[1]):
        good_wavelengths = ~anomalous_centers_mask & np.all(
            np.isfinite(image_centers[:, frame_idx]), axis=1
        )
        try:
            cx = np.polyfit(
                wavelengths[good_wavelengths],
                image_centers[good_wavelengths, frame_idx, 0],
                deg=2,
            )
            cy = np.polyfit(
                wavelengths[good_wavelengths],
                image_centers[good_wavelengths, frame_idx, 1],
                deg=3,
            )
            image_centers_fitted[:, frame_idx, 0] = np.poly1d(cx)(wavelengths)
            image_centers_fitted[:, frame_idx, 1] = np.poly1d(cy)(wavelengths)
        except Exception:
            logger.exception(
                f"Frame {frame_idx}: Polyfit failed (first pass)",
                extra={"step": "polynomial_center_fit", "status": "failed"},
            )
            image_centers_fitted[:, frame_idx, 0] = np.nan
            image_centers_fitted[:, frame_idx, 1] = np.nan
    for frame_idx in range(image_centers.shape[1]):
        if np.all(~np.isfinite(image_centers_fitted[:, frame_idx])):
            image_centers_fitted2[:, frame_idx, 0] = np.nan
            image_centers_fitted2[:, frame_idx, 1] = np.nan
        else:
            deviation = image_centers - image_centers_fitted
            filtered = sigma_clip(
                deviation[:, frame_idx, :],
                sigma=3,
                cenfunc="median",
                stdfunc="std",
                masked=True,
                copy=True,
            )
            mask = getattr(filtered, "mask", None)
            mask_any = (
                np.any(mask, axis=1)
                if mask is not None
                else np.zeros(wavelengths.shape, dtype=bool)
            )
            mask_any = np.array(mask_any, dtype=bool)
            mask_any[remove_indices] = True
            try:
                good_idx = ~mask_any
                cx = np.polyfit(
                    wavelengths[good_idx],
                    image_centers[good_idx, frame_idx, 0],
                    deg=2,
                )
                cy = np.polyfit(
                    wavelengths[good_idx],
                    image_centers[good_idx, frame_idx, 1],
                    deg=3,
                )
                image_centers_fitted2[:, frame_idx, 0] = np.poly1d(cx)(wavelengths)
                image_centers_fitted2[:, frame_idx, 1] = np.poly1d(cy)(wavelengths)
            except Exception:
                logger.exception(
                    f"Frame {frame_idx}: Polyfit failed (robust)",
                    extra={"step": "polynomial_center_fit", "status": "failed"},
                )
                image_centers_fitted2[:, frame_idx, 0] = np.nan
                image_centers_fitted2[:, frame_idx, 1] = np.nan
    fits.writeto(
        os.path.join(converted_dir, "image_centers_fitted.fits"),
        image_centers_fitted,
        overwrite=True,
    )
    fits.writeto(
        os.path.join(converted_dir, "image_centers_fitted_robust.fits"),
        image_centers_fitted2,
        overwrite=True,
    )
    logger.info("Step finished", extra={"step": "polynomial_center_fit", "status": "success"})
