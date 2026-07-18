"""VLT/SPHERE IRDIS master calibration step (Phase 3).

Builds per-observation master background, master flat, and bad-pixel map from
archive ``FLAT`` and ``BG_SCIENCE`` frames. All products are stored in the
canonical split layout ``(2, 1024, 1024)`` so downstream science-frame
application is a straight indexed operation.

Dead detector regions are named constants — the row/column bands that were
hardcoded in ``simplified_IRDIS_reduction.py`` are declared once here and
consumed everywhere.

Pixel-validity convention (charis-compatible; see design spec):

- Dead detector regions → NaN in master data, ``False`` in the bad-pixel map
  (so they are never interpolated). Downstream steps write ``ivar = 0`` at
  these pixels.
- Bad pixels flagged by this step → ``True`` in the bad-pixel map.
  Downstream steps replace the data by interpolation and set ``ivar = 0``.

This step is an internal-guard step (spec §6): if all outputs already exist
on disk it logs ``skipped_complete`` and returns immediately.
"""
from __future__ import annotations

import numpy as np

# --- detector geometry -----------------------------------------------------
# The IRDIS detector is 1024×2048; the two dual-band halves live at
# columns [0:1024] (channel 0) and [1024:2048] (channel 1).

DETECTOR_ROWS: int = 1024
DETECTOR_COLS: int = 2048
HALF_COLS: int = 1024

# Dead regions in RAW detector coordinates (rows apply to both halves; column
# slices are RAW-column indices — see dead_region_mask() for the per-half
# projection).
DEAD_ROW_SLICE_BOTTOM: slice = slice(0, 15)
DEAD_ROW_SLICE_TOP: slice = slice(1013, 1024)
DEAD_COL_SLICE_LEFT_HALF_EDGE: slice = slice(0, 50)
DEAD_COL_SLICE_MID: slice = slice(941, 1078)
DEAD_COL_SLICE_RIGHT_HALF_EDGE: slice = slice(1966, 2048)


def split_detector_image(img: np.ndarray) -> np.ndarray:
    """Split a raw ``(1024, 2048)`` IRDIS frame into ``(2, 1024, 1024)``.

    Channel 0 is the left half (raw columns 0:1024); channel 1 is the right
    half (raw columns 1024:2048).
    """
    if img.shape != (DETECTOR_ROWS, DETECTOR_COLS):
        raise ValueError(
            f"Expected shape ({DETECTOR_ROWS}, {DETECTOR_COLS}), got {img.shape}"
        )
    out = np.empty((2, DETECTOR_ROWS, HALF_COLS), dtype=img.dtype)
    out[0] = img[:, 0:HALF_COLS]
    out[1] = img[:, HALF_COLS:DETECTOR_COLS]
    return out


def split_detector_cube(cube: np.ndarray) -> np.ndarray:
    """Split a raw ``(N, 1024, 2048)`` IRDIS cube into ``(N, 2, 1024, 1024)``."""
    if cube.ndim != 3 or cube.shape[1:] != (DETECTOR_ROWS, DETECTOR_COLS):
        raise ValueError(
            f"Expected shape (N, {DETECTOR_ROWS}, {DETECTOR_COLS}), got {cube.shape}"
        )
    n = cube.shape[0]
    out = np.empty((n, 2, DETECTOR_ROWS, HALF_COLS), dtype=cube.dtype)
    out[:, 0] = cube[:, :, 0:HALF_COLS]
    out[:, 1] = cube[:, :, HALF_COLS:DETECTOR_COLS]
    return out


def dead_region_mask() -> np.ndarray:
    """Return a ``(2, 1024, 1024)`` bool mask, ``True`` at dead-region pixels.

    Projects the raw-coordinate dead bands into per-half coordinates. Row
    bands (``DEAD_ROW_SLICE_BOTTOM``, ``DEAD_ROW_SLICE_TOP``) apply to both
    halves identically. Column bands split across the seam.
    """
    mask = np.zeros((2, DETECTOR_ROWS, HALF_COLS), dtype=bool)

    mask[:, DEAD_ROW_SLICE_BOTTOM, :] = True
    mask[:, DEAD_ROW_SLICE_TOP, :] = True

    # Left-half-edge band (raw cols 0..50 → channel 0 cols 0..50).
    mask[0, :, DEAD_COL_SLICE_LEFT_HALF_EDGE] = True

    # Mid band (raw cols 941..1078) — spans both halves.
    # Channel 0: raw 941..1023 → per-half 941..1023.
    mask[0, :, DEAD_COL_SLICE_MID.start:HALF_COLS] = True
    # Channel 1: raw 1024..1077 → per-half 0..54.
    mask[1, :, 0:DEAD_COL_SLICE_MID.stop - HALF_COLS] = True

    # Right-half-edge band (raw cols 1966..2048 → channel 1 cols 942..1024).
    mask[1, :, DEAD_COL_SLICE_RIGHT_HALF_EDGE.start - HALF_COLS:HALF_COLS] = True

    return mask


def _load_frames_as_split_stack(paths: list[str]) -> np.ndarray:
    """Read a list of raw IRDIS FITS files and return a split cube.

    Squeezes each file's primary HDU to 3D (adding a leading axis if the file
    stores a single 2-D frame), splits into halves, and concatenates along the
    leading (frame) axis.

    Returns
    -------
    np.ndarray
        Shape ``(N_total_frames, 2, 1024, 1024)``, dtype ``float32``.
    """
    from astropy.io import fits

    chunks = []
    for path in paths:
        data = np.asarray(fits.getdata(path), dtype=np.float32)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim != 3:
            raise ValueError(
                f"Unexpected data shape {data.shape} in {path}; expected 2-D or 3-D."
            )
        chunks.append(split_detector_cube(data))
    return np.concatenate(chunks, axis=0)


def build_master_background(bg_frames_paths, config, logger) -> np.ndarray:
    """Build the master background from BG_SCIENCE frames.

    Sigma-clipped mean stack (across all frames of all files) per pixel;
    dead-region pixels are NaN in the output. Returns ``(2, 1024, 1024)``
    float32.

    Parameters
    ----------
    bg_frames_paths : list of str
        Paths to raw BG_SCIENCE FITS files.
    config
        ``IRDISCalibrationConfig`` — provides sigma thresholds used later by
        ``build_bad_pixel_map``. The sigma-clip below uses a fixed 3σ for
        combination only.
    logger
        Structured logger.
    """
    from astropy.stats import sigma_clip

    logger.info(
        f"Building master background from {len(bg_frames_paths)} file(s)",
        extra={"step": "irdis_calibration", "status": "background_started"},
    )

    stack = _load_frames_as_split_stack(list(bg_frames_paths))
    clipped = sigma_clip(
        stack, sigma=3.0, axis=0, cenfunc="median", stdfunc="mad_std", masked=True
    )
    master = np.asarray(clipped.mean(axis=0), dtype=np.float32)

    master[dead_region_mask()] = np.nan

    logger.info(
        "Master background built",
        extra={"step": "irdis_calibration", "status": "background_complete"},
    )
    return master


def estimate_smooth_illumination(
    image: np.ndarray,
    block_size: int = 64,
) -> np.ndarray:
    """Estimate the large-scale illumination pattern of a detector-half image.

    Downsamples ``image`` by taking a ``nanmedian`` in ``block_size``-sized
    blocks, then upsamples the low-resolution grid back to the input shape
    via cubic-spline interpolation. Fast (~30 ms per 1024×1024 half) and
    intrinsically robust to bad pixels (a handful of outliers per block are
    invisible to the median).

    Used by :func:`build_master_flat` to remove the integrating-sphere
    illumination gradient before normalizing the flat. Can be reused for
    per-frame illumination correction in later phases.

    Parameters
    ----------
    image : np.ndarray
        Shape ``(H, W)``. Bad pixels should be NaN so ``nanmedian`` ignores
        them.
    block_size : int, optional
        Side length of the median-downsampling blocks. Default 64 → 16×16
        grid for a 1024×1024 half.

    Returns
    -------
    np.ndarray
        Same shape as ``image``, float32, containing the smooth illumination
        model. NaN-only blocks are replaced with the global nanmedian so the
        interpolation stays finite.
    """
    import warnings

    from scipy.interpolate import RegularGridInterpolator

    h, w = image.shape
    if h % block_size or w % block_size:
        raise ValueError(
            f"image shape {image.shape} not divisible by block_size {block_size}"
        )
    nb_y = h // block_size
    nb_x = w // block_size

    blocks = image.reshape(nb_y, block_size, nb_x, block_size).swapaxes(1, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low = np.nanmedian(blocks, axis=(2, 3)).astype(np.float32)

    if not np.all(np.isfinite(low)):
        fill = float(np.nanmedian(image))
        low = np.where(np.isfinite(low), low, fill)

    y_low = (np.arange(nb_y) + 0.5) * block_size
    x_low = (np.arange(nb_x) + 0.5) * block_size
    interpolator = RegularGridInterpolator(
        (y_low, x_low), low, method="cubic", bounds_error=False, fill_value=None,
    )
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    pts = np.stack([yy.ravel(), xx.ravel()], axis=-1)
    return interpolator(pts).reshape(h, w).astype(np.float32)


def _pixelwise_linear_slope(stack: np.ndarray, dits: np.ndarray) -> np.ndarray:
    """Per-pixel least-squares slope of ``counts`` vs. ``dit`` along axis 0.

    Fully vectorized closed-form OLS: ``slope = sum((x-x̄)(y-ȳ)) / sum((x-x̄)^2)``.

    Parameters
    ----------
    stack : np.ndarray
        Shape ``(N_frames, 2, 1024, 1024)``, float32.
    dits : np.ndarray
        Shape ``(N_frames,)``, DIT of each frame.

    Returns
    -------
    np.ndarray
        Slope per pixel, shape ``(2, 1024, 1024)``, float32.
    """
    dits = np.asarray(dits, dtype=np.float64)
    x = dits - dits.mean()
    denom = float((x * x).sum())
    if denom == 0.0:
        raise ValueError("All DITs are identical; cannot fit a slope.")
    y = stack.astype(np.float64)
    y_mean = y.mean(axis=0)
    num = (x[:, None, None, None] * (y - y_mean[None, ...])).sum(axis=0)
    return (num / denom).astype(np.float32)


def _detrend_illumination_and_normalize(
    response: np.ndarray, dead_mask: np.ndarray, block_size: int = 64
) -> np.ndarray:
    """Divide out the smooth illumination pattern per half and normalize by
    the median of the illumination-detrended, illuminated region.

    The result is a flat-field whose pixel-scale structure encodes the true
    detector response variation, not the integrating-sphere illumination.
    """
    normalized = np.empty_like(response)
    for ch in range(2):
        ch_data = response[ch].copy()
        ch_data[dead_mask[ch]] = np.nan
        illum = estimate_smooth_illumination(ch_data, block_size=block_size)
        with np.errstate(divide="ignore", invalid="ignore"):
            detrended = ch_data / illum
        valid = ~dead_mask[ch] & np.isfinite(detrended)
        med = float(np.median(detrended[valid])) if valid.any() else 1.0
        normalized[ch] = detrended / med if med != 0.0 else detrended
    return normalized


def build_master_flat(
    flat_frames_paths,
    flat_dits,
    master_background: np.ndarray,
    config,
    logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the master flat and per-pixel response map from FLAT frames.

    When ``flat_dits`` contains ≥ 2 distinct values, uses a per-pixel linear
    fit of counts vs. DIT (slope = response; removes bias/dark analytically).
    Otherwise falls back to a background-subtracted, sigma-clipped median
    across frames.

    The per-pixel response is then divided by an estimate of the smooth
    illumination pattern (integrating-sphere vignetting) so the flat encodes
    the true pixel-scale detector response, not the illumination.

    Returns
    -------
    master_flat : np.ndarray
        Shape ``(2, 1024, 1024)`` float32, normalized by the median response
        of the illumination-detrended, illuminated region per half. Dead
        regions are NaN.
    response_map : np.ndarray
        Un-normalized slope (or bg-subtracted median), same shape. Used by
        ``build_bad_pixel_map`` for the abnormal-response detection.
    """
    from astropy.stats import sigma_clip

    stack = _load_frames_as_split_stack(list(flat_frames_paths))
    dead_mask = dead_region_mask()
    dits_array = np.asarray(flat_dits, dtype=np.float64)

    if np.unique(dits_array).size >= 2:
        logger.info(
            f"Building master flat from {len(flat_frames_paths)} frames via DIT-linear fit "
            f"({np.unique(dits_array).size} distinct DITs).",
            extra={"step": "irdis_calibration", "status": "flat_started"},
        )
        response = _pixelwise_linear_slope(stack, dits_array)
    else:
        logger.info(
            f"Building master flat from {len(flat_frames_paths)} frames via "
            "background-subtracted median (single DIT).",
            extra={"step": "irdis_calibration", "status": "flat_started"},
        )
        stack_bg_sub = stack - master_background[None, ...]
        clipped = sigma_clip(
            stack_bg_sub, sigma=3.0, axis=0, cenfunc="median", stdfunc="mad_std", masked=True
        )
        response = np.asarray(clipped.mean(axis=0), dtype=np.float32)

    response[dead_mask] = np.nan
    flat = _detrend_illumination_and_normalize(response, dead_mask)
    flat[dead_mask] = np.nan

    logger.info(
        "Master flat built",
        extra={"step": "irdis_calibration", "status": "flat_complete"},
    )
    return flat, response
