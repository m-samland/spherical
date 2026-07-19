"""VLT/SPHERE IRDIS science preprocess step (Phase 4).

Replaces the IFS pipeline's ``extract_cubes`` + ``bundle_output`` combo for
IRDIS. Consumes raw CORO/CENTER/FLUX frames plus the Phase 3 master
calibration products (background, flat, bpm) and writes the ``converted/``
folder in a layout byte-compatible with the IFS pipeline so downstream
shared steps and TRAP work without further generalization.

Pixel-validity convention (charis-compatible; see design spec):

- Dead detector regions → NaN in data, ``ivar = 0``, ``False`` in the
  propagated bad-pixel map (never interpolated).
- Bad pixels flagged by calibration → interpolated in data, ``ivar = 0``.
- Real measurements → finite data, ``ivar > 0``.

The step is gated by :func:`spherical.pipeline.step_registry.should_run`
against the eight output files listed in ``IRDIS_STEP_REGISTRY``; it is not
internal-guard.
"""
from __future__ import annotations

import warnings

import numpy as np

from spherical.pipeline.transmission import wavelength_bandwidth_filter

# --- per-filter nominal star positions -------------------------------------
# Per-half (x, y) coordinates. Values verbatim from
# ``simplified_IRDIS_reduction.py``'s K/H band guesses (measured empirically
# across archive datasets). A robust matched-filter refinement is deferred to
# Phase 5 (find_centers generalization).

NOMINAL_STAR_POSITIONS_H_BAND: tuple[tuple[float, float], tuple[float, float]] = (
    (485.81, 523.54),
    (487.95, 514.36),
)
NOMINAL_STAR_POSITIONS_K_BAND: tuple[tuple[float, float], tuple[float, float]] = (
    (480.0, 524.7),
    (482.5, 511.4),
)
# Vigan (arthur-vigan/sphere) IRDIS default_center. Used as the fallback for
# IRDIS filters we have not yet calibrated per-band values for. See the
# 2026-07-19 decision entry: DB_Y23, DB_J23, DB_H34, DB_NDH23, and any NB_*
# filter routes here until we measure the star position on a reference
# dataset in that filter.
NOMINAL_STAR_POSITIONS_DEFAULT_VIGAN: tuple[tuple[float, float], tuple[float, float]] = (
    (485.0, 520.0),
    (486.0, 508.0),
)

_K_BAND_FILTERS: frozenset[str] = frozenset({"DB_K12", "BB_K", "BB_Ks"})
_H_BAND_FILTERS: frozenset[str] = frozenset({"DB_H23", "BB_H"})


def nominal_star_positions(filter_comb: str) -> np.ndarray:
    """Return per-channel nominal ``(x, y)`` star positions for a filter.

    Filter-name whitelist rather than a wavelength-based K-vs-H dispatch:
    only the filters we have measured on a reference dataset get their
    calibrated per-band nominal; every other IRDIS filter falls back to
    Vigan's static IRDIS default ``((485, 520), (486, 508))``. This keeps
    ``DB_H34``, ``DB_Y23``, ``DB_J23``, ``NB_*``, etc. from silently
    receiving the K-band or H-band nominal just because their max
    wavelength happens to cross a threshold.

    Filter → nominal:

    - ``DB_K12``, ``BB_K``, ``BB_Ks`` → K-band nominal
      (``NOMINAL_STAR_POSITIONS_K_BAND``)
    - ``DB_H23``, ``BB_H`` → H-band nominal
      (``NOMINAL_STAR_POSITIONS_H_BAND``)
    - anything else recognised by
      :func:`spherical.pipeline.transmission.wavelength_bandwidth_filter`
      → Vigan default (``NOMINAL_STAR_POSITIONS_DEFAULT_VIGAN``)

    Parameters
    ----------
    filter_comb : str
        IRDIS filter combination string (e.g. ``"DB_K12"``, ``"DB_H23"``,
        ``"BB_H"``, ``"DB_Y23"``).

    Returns
    -------
    np.ndarray
        Shape ``(2, 2)`` — rows are channels, columns are ``(x, y)`` in
        per-half detector coordinates.
    """
    # Validate the filter is known — raises ValueError on unknown names,
    # matching the previous behavior. Wavelengths not used for dispatch.
    wavelength_bandwidth_filter(filter_comb)
    if filter_comb in _K_BAND_FILTERS:
        return np.array(NOMINAL_STAR_POSITIONS_K_BAND, dtype=np.float64)
    if filter_comb in _H_BAND_FILTERS:
        return np.array(NOMINAL_STAR_POSITIONS_H_BAND, dtype=np.float64)
    return np.array(NOMINAL_STAR_POSITIONS_DEFAULT_VIGAN, dtype=np.float64)


def coarse_star_position(
    image: np.ndarray,
    nominal_xy: tuple[float, float],
    search_radius: int = 100,
) -> tuple[float, float]:
    """Locate the brightest peak in a window around ``nominal_xy``.

    Consumed by Phase 5's `find_centers` shared-step generalization; not
    called from the Phase 4 orchestrator (`run_irdis_preprocess` uses
    `nominal_star_positions` directly).

    Searches inside a square window of half-side ``search_radius`` around
    ``nominal_xy``; if no peak reaches 5σ above the local ``mad_std`` the
    window is widened to the full frame; if still no significant peak is
    found the nominal position is returned unchanged.

    Parameters
    ----------
    image : np.ndarray
        Single detector-half image ``(H, W)``. NaN pixels (dead regions)
        are ignored.
    nominal_xy : (float, float)
        Center of the initial search window and fallback position.
    search_radius : int, optional
        Half-side of the initial search window in pixels. Default 100.

    Returns
    -------
    (float, float)
        ``(x, y)`` peak position, or ``nominal_xy`` on fallback.
    """
    from astropy.stats import mad_std

    h, w = image.shape
    nom_x, nom_y = float(nominal_xy[0]), float(nominal_xy[1])

    def _peak_in_window(x0: int, x1: int, y0: int, y1: int) -> tuple[float, float, float]:
        sub = image[y0:y1, x0:x1]
        finite = np.isfinite(sub)
        if not finite.any():
            return (nom_x, nom_y, 0.0)
        sub_masked = np.where(finite, sub, -np.inf)
        idx = np.unravel_index(np.argmax(sub_masked), sub.shape)
        peak_val = float(sub[idx])
        sigma = float(mad_std(sub[finite], ignore_nan=True))
        med = float(np.median(sub[finite]))
        snr = (peak_val - med) / sigma if sigma > 0.0 else 0.0
        return (float(idx[1] + x0), float(idx[0] + y0), snr)

    x0 = max(int(nom_x - search_radius), 0)
    x1 = min(int(nom_x + search_radius) + 1, w)
    y0 = max(int(nom_y - search_radius), 0)
    y1 = min(int(nom_y + search_radius) + 1, h)
    cx, cy, snr = _peak_in_window(x0, x1, y0, y1)
    if snr >= 5.0:
        return (cx, cy)

    cx, cy, snr = _peak_in_window(0, w, 0, h)
    if snr >= 5.0:
        return (cx, cy)

    warnings.warn(
        f"coarse_star_position: no significant peak found; "
        f"falling back to nominal ({nom_x}, {nom_y})",
        UserWarning,
        stacklevel=2,
    )
    return (nom_x, nom_y)


def background_region_mask(
    shape: tuple[int, int],
    star_xy: tuple[float, float],
    star_mask_radius: int,
    dead_mask_ch: np.ndarray,
    bpm_ch: np.ndarray,
) -> np.ndarray:
    """Build the mask of pixels that participate in a background fit.

    Excludes: (a) a circular region of radius ``star_mask_radius`` around
    ``star_xy``, (b) dead-region pixels, (c) known bad pixels.

    Factored as its own helper so a future PCA background model can reuse
    the same exclusion logic without depending on the scalar-fit code path.
    """
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    r2 = (xx - star_xy[0]) ** 2 + (yy - star_xy[1]) ** 2
    star_mask = r2 <= float(star_mask_radius) ** 2
    return (~star_mask) & (~dead_mask_ch) & (~bpm_ch)


def fit_background_scale(
    frame_ch: np.ndarray,
    bg_ch: np.ndarray,
    region_mask: np.ndarray,
    n_sigma_clip: int = 3,
    max_iter: int = 3,
) -> float:
    """Robust scalar least-squares fit of ``frame ≈ s · bg`` on masked pixels.

    Iterates the closed-form solution
    ``s = Σ(m·frame·bg) / Σ(m·bg²)``, tightening the mask each pass with a
    ``n_sigma_clip · mad_std`` clip on residuals.

    Parameters
    ----------
    frame_ch, bg_ch : np.ndarray
        Same shape ``(H, W)``. NaN in either is treated as masked-out.
    region_mask : np.ndarray of bool
        Initial validity mask (see :func:`background_region_mask`).
    n_sigma_clip : int, optional
        Residual-clipping threshold in units of ``mad_std``. Default 3.
    max_iter : int, optional
        Maximum number of clipping iterations. Default 3.

    Returns
    -------
    float
        Best-fit scalar; ``0.0`` if the mask empties out or the fit
        degenerates.
    """
    from astropy.stats import mad_std

    finite = np.isfinite(frame_ch) & np.isfinite(bg_ch)
    mask = region_mask & finite
    s = 0.0
    for _ in range(max_iter):
        if not mask.any():
            return 0.0
        # `np.where` selects 0.0 at excluded pixels before the sum; a plain
        # `mask * frame_ch * bg_ch` would multiply first, and `0 * NaN = NaN`
        # leaks any NaN in frame_ch/bg_ch outside the mask (e.g. dead-region
        # background pixels) into the sum despite being masked out.
        frame_masked = np.where(mask, frame_ch, 0.0)
        bg_masked = np.where(mask, bg_ch, 0.0)
        num = float((frame_masked * bg_masked).sum())
        den = float((bg_masked * bg_masked).sum())
        if den <= 0.0:
            return 0.0
        s_new = num / den
        residual = frame_ch - s_new * bg_ch
        sigma = float(mad_std(residual[mask], ignore_nan=True))
        if not np.isfinite(sigma) or sigma <= 0.0 or abs(s_new - s) < 1e-6:
            return s_new
        s = s_new
        mask = mask & (np.abs(residual) < n_sigma_clip * sigma)
    return s


def subtract_scaled_background(
    frame_ch: np.ndarray,
    bg_ch: np.ndarray,
    star_xy: tuple[float, float],
    star_mask_radius: int,
    dead_mask_ch: np.ndarray,
    bpm_ch: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit and subtract a scaled master background from one channel of one frame.

    Returns
    -------
    (np.ndarray, float)
        The residual ``frame - s · bg`` and the fitted scale ``s``.
    """
    region_mask = background_region_mask(
        shape=frame_ch.shape,
        star_xy=star_xy,
        star_mask_radius=star_mask_radius,
        dead_mask_ch=dead_mask_ch,
        bpm_ch=bpm_ch,
    )
    s = fit_background_scale(frame_ch, bg_ch, region_mask)
    return frame_ch - s * bg_ch, s


def analytic_ivar(
    counts_after_bg: np.ndarray,
    flat: np.ndarray,
    gain: float,
    read_noise: float,
) -> np.ndarray:
    """Analytic per-pixel inverse variance in counts² units.

    Base model (before the flat is applied to the data):
    ``ivar_counts = gain² / (max(counts, 0) · gain + read_noise²)``, i.e.
    photon shot noise plus read noise, converted from electrons back into
    counts. Dividing the data by the flat scales counts by ``1/flat``; ivar
    therefore scales by ``flat²``.

    Parameters
    ----------
    counts_after_bg : np.ndarray
        Background-subtracted counts (before flat division). Any NaN → 0 ivar.
    flat : np.ndarray
        Master flat, same shape. Any NaN → 0 ivar.
    gain : float
        Detector gain in e⁻/count (IRDIS default 1.75).
    read_noise : float
        Read noise in electrons (IRDIS default 4.4).

    Returns
    -------
    np.ndarray
        float32 ivar map, same shape.
    """
    counts_pos = np.where(np.isfinite(counts_after_bg), np.maximum(counts_after_bg, 0.0), 0.0)
    base = gain * gain / (counts_pos * gain + read_noise * read_noise)
    flat_sq = np.where(np.isfinite(flat), flat, 0.0) ** 2
    ivar = base * flat_sq
    invalid = ~np.isfinite(counts_after_bg) | ~np.isfinite(flat)
    ivar = np.where(invalid, 0.0, ivar)
    return ivar.astype(np.float32)


def _fix_badpix_vec_one_pass(
    frame: np.ndarray,
    target_bpm: np.ndarray,
    exclusion_mask: np.ndarray,
    dmax: int,
    npix: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One vectorized pass of the K-nearest-good-neighbor fixer.

    Returns ``(fixed, updated)`` where ``updated`` marks the target pixels
    that received a new value. Targets with fewer than ``npix`` good
    neighbors inside a ``(2·dmax+1)²`` window are left at their original
    value and reported ``False`` in ``updated`` — the caller can drive a
    second pass over them with a wider window.
    """
    h, w = frame.shape
    win = 2 * dmax + 1

    dy, dx = np.mgrid[-dmax:dmax + 1, -dmax:dmax + 1]
    dist_win = np.sqrt(dx.astype(np.float32) ** 2 + dy.astype(np.float32) ** 2).ravel()
    dist_win[dist_win == 0] = 1e-6  # avoid /0 at the target (excluded anyway)

    bp_y, bp_x = np.where(target_bpm)
    n_bad = bp_y.size
    updated = np.zeros_like(target_bpm)
    if n_bad == 0:
        return frame.copy(), updated

    padded_frame = np.pad(frame, dmax, mode="constant", constant_values=0.0)
    padded_exc = np.pad(exclusion_mask, dmax, mode="constant", constant_values=True)

    yi = bp_y[:, None, None] + np.arange(win)[None, :, None]
    xi = bp_x[:, None, None] + np.arange(win)[None, None, :]
    windows_vals = padded_frame[yi, xi].reshape(n_bad, -1)
    windows_exc = padded_exc[yi, xi].reshape(n_bad, -1)

    LARGE = np.float32(1e10)
    dists_masked = np.where(windows_exc, LARGE, dist_win)

    part = np.argpartition(dists_masked, npix, axis=1)[:, :npix]
    top_vals = np.take_along_axis(windows_vals, part, axis=1)
    top_dists = np.take_along_axis(dists_masked, part, axis=1)
    enough = (top_dists < LARGE).all(axis=1)

    val_order = np.argsort(top_vals, axis=1)
    sorted_vals = np.take_along_axis(top_vals, val_order, axis=1)
    sorted_dists = np.take_along_axis(top_dists, val_order, axis=1)

    mid_vals = sorted_vals[:, 1:-1]
    mid_dists = sorted_dists[:, 1:-1]
    weights = np.float32(1.0) / mid_dists
    new_vals = (mid_vals * weights).sum(axis=1) / weights.sum(axis=1)

    result = frame.copy()
    result[bp_y[enough], bp_x[enough]] = new_vals[enough].astype(frame.dtype)
    updated[bp_y[enough], bp_x[enough]] = True
    return result, updated


def fix_badpix_nan_safe(
    frame_ch: np.ndarray,
    bpm_ch: np.ndarray,
    dead_mask_ch: np.ndarray,
    npix: int = 12,
) -> np.ndarray:
    """Interpolate bad pixels without pulling NaN from dead-region neighbors.

    Vectorized K-nearest-good-neighbor fill with the same weight=1/distance,
    toss-max-min semantics as ``imutils.fix_badpix(weight=True)``. Runs a
    two-pass search: an initial 11×11 window (``dmax=5``) catches the vast
    majority of bad pixels cheaply; a fallback 21×21 window (``dmax=10``)
    handles any target that lacked ``npix`` good neighbors in the small
    window. Both passes exclude dead-region pixels and other bad pixels
    from the good-neighbor pool — that is the fix relative to the previous
    ``imutils.fix_badpix`` wrapper, which allowed dead pixels to enter the
    neighbor pool as zero-substituted "good" values and consequently biased
    bad pixels at the dead-band boundary toward zero.

    Dead and illuminated-NaN pixels stay NaN in the working array
    throughout; because they are also in the exclusion mask they never
    enter the argpartition selection and therefore never participate in
    the weighted mean. Keeping the NaN explicit rather than substituting a
    sentinel value avoids hiding invalid pixels behind a numeric 0.

    A target pixel that still lacks ``npix`` good neighbors after the
    second pass is left at its input value; downstream code observes
    ``ivar = 0`` at every bad pixel regardless of interpolation success,
    so no-info semantics are preserved.

    Parameters
    ----------
    frame_ch : np.ndarray
        Single-channel 2-D image, may contain NaN in dead-region pixels.
    bpm_ch : np.ndarray of bool
        Bad-pixel targets to interpolate (Phase 3 bpm has ``False`` in dead
        regions).
    dead_mask_ch : np.ndarray of bool
        Dead-region mask for the same channel; never overwritten with
        interpolated values.
    npix : int, optional
        Number of nearest good neighbors to average. Default 12.

    Returns
    -------
    np.ndarray
        Interpolated frame; dead-region pixels are NaN.
    """
    finite = np.isfinite(frame_ch)
    frame_work = frame_ch.astype(np.float32, copy=True)

    target_bpm = bpm_ch | (~finite & ~dead_mask_ch)
    exclusion = target_bpm | dead_mask_ch  # dead never enters the good pool

    fixed, updated = _fix_badpix_vec_one_pass(
        frame_work, target_bpm, exclusion, dmax=5, npix=npix,
    )
    remaining = target_bpm & ~updated
    if remaining.any():
        # Pixels updated in pass 1 are now valid neighbors for pass 2.
        exc_pass2 = exclusion & ~updated
        fixed, _ = _fix_badpix_vec_one_pass(
            fixed, remaining, exc_pass2, dmax=10, npix=npix,
        )

    fixed = fixed.astype(np.float32)
    fixed[dead_mask_ch] = np.nan
    return fixed


def sigma_filter_ignore_dead(
    frame_ch: np.ndarray,
    dead_mask_ch: np.ndarray,
    box: int = 7,
    nsigma: float = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Transient sigma-clip on a single-channel frame, ignoring dead regions.

    Wraps :func:`spherical.pipeline.imutils.sigma_filter`. NaN dead pixels
    are substituted with the frame's median before the box-filter test so
    they don't appear as extreme outliers to their neighbors. The returned
    transient mask is forced to ``False`` in dead regions; the cleaned
    frame has NaN restored in dead regions.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Cleaned frame and boolean transient mask.
    """
    from spherical.pipeline import imutils

    fill = float(np.nanmedian(frame_ch)) if np.isfinite(frame_ch).any() else 0.0
    frame_work = np.where(np.isfinite(frame_ch), frame_ch, fill).astype(np.float32)

    cleaned, mask = imutils.sigma_filter(
        frame_work, box=box, nsigma=nsigma, return_mask=True, iterate=False,
    )
    cleaned = np.asarray(cleaned, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    mask[dead_mask_ch] = False
    cleaned[dead_mask_ch] = np.nan
    return cleaned, mask


def apply_anamorphism(
    cube_ch: np.ndarray,
    factor: float,
    dead_mask_ch: np.ndarray,
) -> np.ndarray:
    """Rescale the y-axis of a single-channel cube by ``factor``.

    Uses ``scipy.ndimage.zoom`` (cubic spline). Output shape matches input:
    the zoomed array is center-cropped or edge-padded back to original H.
    NaN dead-region pixels are re-set to NaN on the output.

    Parameters
    ----------
    cube_ch : np.ndarray
        Either ``(H, W)`` or ``(n_time, H, W)``.
    factor : float
        Anamorphism factor (default 1.0062 for IRDIS). ``1.0`` is a no-op.
    dead_mask_ch : np.ndarray of bool
        Dead-region mask for this channel, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        Rescaled cube, same shape as input, float32.
    """
    if factor == 1.0:
        return cube_ch
    from scipy.ndimage import zoom

    is_3d = cube_ch.ndim == 3
    frames = cube_ch if is_3d else cube_ch[np.newaxis, ...]
    n, h, w = frames.shape
    frames_filled = np.where(np.isfinite(frames), frames, 0.0).astype(np.float32)

    zoomed = zoom(frames_filled, (1.0, factor, 1.0), order=3, mode="nearest")
    new_h = zoomed.shape[1]
    if new_h > h:
        start = (new_h - h) // 2
        zoomed = zoomed[:, start:start + h, :]
    elif new_h < h:
        pad = h - new_h
        pad_lo = pad // 2
        pad_hi = pad - pad_lo
        zoomed = np.pad(zoomed, ((0, 0), (pad_lo, pad_hi), (0, 0)), mode="edge")

    zoomed = zoomed.astype(np.float32)
    zoomed[:, dead_mask_ch] = np.nan
    return zoomed if is_3d else zoomed[0]


def apply_crop(
    cube_ch: np.ndarray,
    ivar_ch: np.ndarray,
    star_xy: tuple[float, float],
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Crop a ``crop_size`` × ``crop_size`` square around ``star_xy``.

    The crop is clamped to the frame; when the star is near an edge the
    crop shifts inward so the full box fits. Returns the crop offsets
    ``(x0, y0)`` of the lower-left corner in original per-half coordinates.

    Parameters
    ----------
    cube_ch, ivar_ch : np.ndarray
        Either ``(H, W)`` or ``(n_time, H, W)``; last two axes are cropped.
    star_xy : (float, float)
        Nominal star ``(x, y)`` position around which to crop.
    crop_size : int
        Side length of the output square.

    Returns
    -------
    (np.ndarray, np.ndarray, (int, int))
        Cropped cube, cropped ivar, ``(x_offset, y_offset)``.
    """
    is_3d = cube_ch.ndim == 3
    h, w = cube_ch.shape[-2], cube_ch.shape[-1]

    x0 = int(round(star_xy[0] - crop_size / 2))
    y0 = int(round(star_xy[1] - crop_size / 2))
    x0 = max(0, min(x0, w - crop_size))
    y0 = max(0, min(y0, h - crop_size))

    if is_3d:
        cube_c = cube_ch[:, y0:y0 + crop_size, x0:x0 + crop_size]
        ivar_c = ivar_ch[:, y0:y0 + crop_size, x0:x0 + crop_size]
    else:
        cube_c = cube_ch[y0:y0 + crop_size, x0:x0 + crop_size]
        ivar_c = ivar_ch[y0:y0 + crop_size, x0:x0 + crop_size]
    return cube_c, ivar_c, (x0, y0)


def _load_raw_frames_expanded(paths: list[str]) -> np.ndarray:
    """Read raw IRDIS files and return a ``(N_total, 1024, 2048)`` cube.

    Each file may store one 2-D frame or an NDIT × 2-D cube; NDIT frames
    are unrolled into the leading axis, preserving intra-file order.
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
        chunks.append(data)
    return np.concatenate(chunks, axis=0)


# Module-level worker state for the ProcessPoolExecutor path. Populated by
# ``_worker_init``; consumed by ``_process_chunk``. Each worker process gets
# its own module-level dict on init.
_WORKER_STATE: dict = {}


def _worker_init(
    master_flat: np.ndarray,
    master_bg: np.ndarray,
    bpm_bool: np.ndarray,
    star_positions_xy: np.ndarray,
    mask_radius: int,
    is_flux: bool,
    gain: float,
    read_noise: float,
    fix_badpix: bool,
    warn_threshold: int,
    transient_nsigma: float,
) -> None:
    """Initializer for both ProcessPoolExecutor workers and the serial path.

    Every call fully overwrites ``_WORKER_STATE`` so back-to-back invocations
    from ``run_irdis_preprocess`` (one per frame type) don't leak state.
    """
    from spherical.pipeline.steps.irdis_calibration import dead_region_mask

    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        master_flat=master_flat,
        master_bg=master_bg,
        bpm_bool=bpm_bool,
        dead=dead_region_mask(),
        star_positions=star_positions_xy,
        mask_radius=mask_radius,
        is_flux=is_flux,
        gain=gain,
        read_noise=read_noise,
        fix_badpix=fix_badpix,
        warn_threshold=warn_threshold,
        transient_nsigma=transient_nsigma,
    )


def _process_chunk(
    chunk_data: tuple[list[int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Process one chunk of frames. Runs in worker or main process.

    Parameters
    ----------
    chunk_data : (frame_indices, split_slice)
        ``frame_indices`` is a list of global frame indices for logging;
        ``split_slice`` is ``(n_chunk, 2, 1024, 1024)``.

    Returns
    -------
    cube_slice : np.ndarray
        ``(2, n_chunk, 1024, 1024)`` float32.
    ivar_slice : np.ndarray
        Same shape.
    transient_counts : list of (frame_idx, ch, n_transient)
        Per (frame, channel) transient sigma-clip hit counts. Empty when
        ``is_flux`` (no transient clip runs) or when ``transient_nsigma <= 0``.
        Consumed by the main process to log a summary; not persisted.
    warnings : list of (frame_idx, ch, n_transient)
        Subset of ``transient_counts`` where the per-frame count exceeded
        ``warn_threshold``; separately surfaced so the main process can log
        one WARNING per excess.
    """
    frame_indices, chunk_split = chunk_data
    n = len(frame_indices)

    mf = _WORKER_STATE["master_flat"]
    mb = _WORKER_STATE["master_bg"]
    bpm_bool = _WORKER_STATE["bpm_bool"]
    dead = _WORKER_STATE["dead"]
    star_positions = _WORKER_STATE["star_positions"]
    mask_radius = _WORKER_STATE["mask_radius"]
    is_flux = _WORKER_STATE["is_flux"]
    gain = _WORKER_STATE["gain"]
    read_noise = _WORKER_STATE["read_noise"]
    do_fix_badpix = _WORKER_STATE["fix_badpix"]
    warn_threshold = _WORKER_STATE["warn_threshold"]
    transient_nsigma = _WORKER_STATE["transient_nsigma"]

    cube_slice = np.empty((2, n, 1024, 1024), dtype=np.float32)
    ivar_slice = np.empty((2, n, 1024, 1024), dtype=np.float32)
    transient_counts: list[tuple[int, int, int]] = []
    warnings: list[tuple[int, int, int]] = []
    do_transient = (not is_flux) and (transient_nsigma > 0)

    for i, frame_idx in enumerate(frame_indices):
        for ch in range(2):
            frame = chunk_split[i, ch]
            star_xy = (float(star_positions[ch, 0]), float(star_positions[ch, 1]))

            bg_sub, _s = subtract_scaled_background(
                frame_ch=frame,
                bg_ch=mb[ch],
                star_xy=star_xy,
                star_mask_radius=mask_radius,
                dead_mask_ch=dead[ch],
                bpm_ch=bpm_bool[ch],
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                divided = bg_sub / mf[ch]

            ivar = analytic_ivar(
                counts_after_bg=bg_sub, flat=mf[ch], gain=gain, read_noise=read_noise,
            )

            divided = np.where(dead[ch], np.nan, divided).astype(np.float32)
            ivar[dead[ch]] = 0.0

            if do_fix_badpix:
                divided = fix_badpix_nan_safe(divided, bpm_bool[ch], dead[ch])
                ivar[bpm_bool[ch]] = 0.0

            if do_transient:
                cleaned, transient_mask = sigma_filter_ignore_dead(
                    divided, dead[ch], nsigma=transient_nsigma,
                )
                n_transient = int(transient_mask.sum())
                transient_counts.append((frame_idx, ch, n_transient))
                ivar[transient_mask] = 0.0
                if n_transient > warn_threshold:
                    warnings.append((frame_idx, ch, n_transient))
                else:
                    divided = cleaned

            cube_slice[ch, i] = divided
            ivar_slice[ch, i] = ivar

    return cube_slice, ivar_slice, transient_counts, warnings


def preprocess_frame_type(
    frame_paths: list[str],
    master_flat: np.ndarray,
    master_background: np.ndarray,
    bpm: np.ndarray,
    star_positions_xy: np.ndarray,
    is_flux: bool,
    preprocess_config,
    logger,
    ncpu: int = 1,
    frame_type_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Full spec §4 preprocess pipeline for one frame type of one observation.

    Runs, per frame per channel:
        1. Split into halves (already done in-bulk).
        2. Scaled background subtraction (per-frame scalar fit on star-masked region).
        3. Flat division.
        4. Analytic inverse variance (with NaN → 0).
        5. Dead-region NaN in data + 0 in ivar.
        6. NaN-safe bad-pixel replacement + ivar=0 at replaced pixels.
        7. Non-FLUX only: transient sigma-clip, ivar=0 at flagged pixels.
        8. Optional anamorphism correction (default off).
        9. Optional crop around the star (default off).

    When ``ncpu > 1`` and ``n_time > 2`` the per-frame loop runs in a
    ``ProcessPoolExecutor`` with roughly ``4·ncpu`` chunks for load balancing.
    Serial path used otherwise.

    Parameters
    ----------
    ncpu : int, optional
        Maximum worker processes for the per-frame loop. Default 1 (serial).
    frame_type_name : str, optional
        Label used in log and progress-bar messages (``"CORO"``,
        ``"CENTER"``, ``"FLUX"``). Falls back to ``"FLUX"`` / ``"science"``.

    Returns
    -------
    cube : np.ndarray
        ``(2, n_time, ny, nx)`` float32 — n_wave-first ordering matching
        the IFS bundle-output convention.
    ivar : np.ndarray
        Same shape as ``cube``, float32.
    bpm_out : np.ndarray
        ``(2, 1024, 1024)`` uint8 — Phase 3 detector bpm, dead-region clamped.
        Per-frame transient sigma-clip hits are recorded as ``ivar = 0`` in
        ``ivar`` but are deliberately NOT unioned into this map (that would
        just accumulate statistical false positives from N independent
        per-frame tests). Stays at native detector resolution.
    crop_offsets : np.ndarray or None
        ``(2, 2)`` int array of ``[[x0_ch0, y0_ch0], [x0_ch1, y0_ch1]]``
        when cropping is enabled; ``None`` otherwise.
    """
    from concurrent.futures import ProcessPoolExecutor

    from tqdm import tqdm

    from spherical.pipeline.steps.irdis_calibration import (
        dead_region_mask,
        split_detector_cube,
    )

    label = frame_type_name or ("FLUX" if is_flux else "science")

    raw = _load_raw_frames_expanded(list(frame_paths))
    split = split_detector_cube(raw)  # (n_time, 2, 1024, 1024)
    n_time = split.shape[0]

    logger.info(
        f"Preprocessing {label}: {len(frame_paths)} file(s), {n_time} total frames "
        f"(is_flux={is_flux}, fix_badpix={preprocess_config.fix_badpix}, ncpu={ncpu})",
        extra={"step": "preprocess_irdis", "status": "frame_type_started"},
    )

    dead = dead_region_mask()  # (2, 1024, 1024)
    bpm_bool = bpm.astype(bool)
    warn_threshold = int(0.01 * 1024 * 1024)
    mask_radius = (
        preprocess_config.flux_star_mask_radius
        if is_flux
        else preprocess_config.star_mask_radius
    )

    # Chunk the frames. For ncpu>1 we target ~4× more chunks than workers so
    # a fast worker can steal work from a slower one; capped so a chunk is at
    # least one frame.
    if ncpu <= 1 or n_time <= 2:
        chunks: list[tuple[list[int], np.ndarray]] = [
            (list(range(n_time)), split)
        ]
    else:
        chunk_size = max(1, n_time // (4 * ncpu))
        chunks = [
            (list(range(i, min(i + chunk_size, n_time))), split[i:i + chunk_size])
            for i in range(0, n_time, chunk_size)
        ]

    init_args = (
        master_flat, master_background, bpm_bool, star_positions_xy,
        mask_radius, is_flux, preprocess_config.gain, preprocess_config.read_noise,
        preprocess_config.fix_badpix, warn_threshold,
        float(preprocess_config.transient_nsigma),
    )

    if ncpu <= 1 or n_time <= 2:
        _worker_init(*init_args)
        results = [
            _process_chunk(chunk)
            for chunk in tqdm(chunks, desc=f"IRDIS {label}", unit="chunk")
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=ncpu,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            results = list(
                tqdm(
                    pool.map(_process_chunk, chunks),
                    total=len(chunks), desc=f"IRDIS {label}", unit="chunk",
                )
            )

    cube_out = np.concatenate([r[0] for r in results], axis=1)
    ivar_out = np.concatenate([r[1] for r in results], axis=1)

    all_counts = [c for r in results for c in r[2]]
    for _, _, _, wlist in results:
        for frame_idx, ch, n_transient in wlist:
            logger.warning(
                f"Frame {frame_idx} ch{ch}: {n_transient} transient bad pixels "
                f"(>{warn_threshold}) — no interpolation performed",
                extra={"step": "preprocess_irdis", "status": "transient_warning"},
            )
    if all_counts:
        counts_arr = np.asarray([c[2] for c in all_counts], dtype=np.int64)
        logger.info(
            f"{label}: transient sigma-clip summary ({counts_arr.size} frame-channels, "
            f"nsigma={preprocess_config.transient_nsigma}) — total={int(counts_arr.sum())}, "
            f"mean={counts_arr.mean():.1f}/frame-channel, median={int(np.median(counts_arr))}, "
            f"min={int(counts_arr.min())}, max={int(counts_arr.max())}",
            extra={"step": "preprocess_irdis", "status": "transient_summary"},
        )

    if preprocess_config.correct_anamorphism:
        for ch in range(2):
            cube_out[ch] = apply_anamorphism(
                cube_out[ch], preprocess_config.anamorphism_factor, dead[ch],
            )
            ivar_out[ch] = apply_anamorphism(
                ivar_out[ch], preprocess_config.anamorphism_factor, dead[ch],
            )
            ivar_out[ch, :, dead[ch]] = 0.0

    crop_offsets: np.ndarray | None = None
    if preprocess_config.crop:
        crop_offsets = np.zeros((2, 2), dtype=np.int32)
        cropped_data = []
        cropped_ivar = []
        for ch in range(2):
            if preprocess_config.crop_center is not None:
                star_xy = (
                    float(preprocess_config.crop_center[0]),
                    float(preprocess_config.crop_center[1]),
                )
            else:
                star_xy = tuple(float(v) for v in star_positions_xy[ch])
            cube_c, ivar_c, (x0, y0) = apply_crop(
                cube_out[ch], ivar_out[ch], star_xy, preprocess_config.crop_size,
            )
            cropped_data.append(cube_c)
            cropped_ivar.append(ivar_c)
            crop_offsets[ch] = [x0, y0]
        cube_out = np.stack(cropped_data, axis=0)
        ivar_out = np.stack(cropped_ivar, axis=0)

    # bpm_out echoes the Phase 3 calibration bpm dead-region-clamped for safety.
    # Per-frame transient hits are already reflected as ivar=0 in `ivar_out`;
    # they are deliberately NOT unioned in here (a union of per-frame 5σ tests
    # across N frames is an accumulator of statistical false positives, not a
    # detector-defect map).
    bpm_out = (bpm_bool & ~dead).astype(np.uint8)
    logger.info(
        f"Preprocessing complete: cube shape {cube_out.shape}",
        extra={"step": "preprocess_irdis", "status": "frame_type_complete"},
    )
    return cube_out, ivar_out, bpm_out, crop_offsets


def run_irdis_preprocess(
    observation,
    config,
    calib_outputdir,
    converted_outputdir,
    logger,
) -> None:
    """Build and write all eight ``converted/`` products for one IRDIS observation.

    Not internal-guard — the caller (``execute_irdis_target``) gates via
    ``should_run("preprocess_irdis", ..., registry=IRDIS_STEP_REGISTRY)`` on
    the presence of the eight output files. When invoked, always recomputes.

    Outputs (all under ``converted_outputdir``):

    - ``coro_cube.fits``, ``center_cube.fits``, ``flux_cube.fits`` —
      ``(2, n_time, ny, nx)`` float32 with anamorphism + crop metadata in
      the primary header.
    - ``coro_ivar_cube.fits``, ``center_ivar_cube.fits``,
      ``flux_ivar_cube.fits`` — same shape.
    - ``wavelengths.fits`` — 2 float32 entries in nm.
    - ``badpixel_map.fits`` — ``(2, 1024, 1024)`` uint8, the Phase 3 detector
      bpm dead-region clamped. Per-frame transient sigma-clip hits are
      recorded in the ivar cubes (``ivar = 0``) and are NOT unioned into
      this file.

    Silently skips any frame type whose ``observation.frames[key]`` is
    missing or empty.
    """
    from pathlib import Path

    from astropy.io import fits

    converted_outputdir = Path(converted_outputdir)
    converted_outputdir.mkdir(parents=True, exist_ok=True)
    calib_outputdir = Path(calib_outputdir)

    logger.info(
        "Starting IRDIS preprocess",
        extra={"step": "preprocess_irdis", "status": "started"},
    )

    master_flat = np.asarray(fits.getdata(calib_outputdir / "master_flat.fits"), dtype=np.float32)
    master_bg = np.asarray(fits.getdata(calib_outputdir / "master_background.fits"), dtype=np.float32)
    bpm = np.asarray(fits.getdata(calib_outputdir / "badpixel_map.fits"), dtype=np.uint8)

    filter_comb = str(observation.observation["FILTER"][0])
    star_positions = nominal_star_positions(filter_comb)

    preprocess_cfg = config.irdis_preprocessing

    for key in ("CORO", "CENTER", "FLUX"):
        table = observation.frames.get(key)
        if table is None or len(table) == 0:
            logger.info(
                f"Skipping frame type {key}: no frames on observation",
                extra={"step": "preprocess_irdis", "status": f"{key.lower()}_skipped"},
            )
            continue

        paths = [str(p) for p in table["FILE"]]
        cube, ivar, _bpm_out, offsets = preprocess_frame_type(
            frame_paths=paths,
            master_flat=master_flat,
            master_background=master_bg,
            bpm=bpm,
            star_positions_xy=star_positions,
            is_flux=(key == "FLUX"),
            preprocess_config=preprocess_cfg,
            logger=logger,
            ncpu=int(config.resources.ncpu_preprocess),
            frame_type_name=key,
        )

        header = fits.Header()
        header["HIERARCH SPHERICAL ANAMORPHISM FACTOR"] = float(preprocess_cfg.anamorphism_factor)
        header["HIERARCH SPHERICAL ANAMORPHISM APPLIED"] = bool(preprocess_cfg.correct_anamorphism)
        header["HIERARCH SPHERICAL CROP APPLIED"] = bool(preprocess_cfg.crop)
        if preprocess_cfg.crop and offsets is not None:
            header["HIERARCH SPHERICAL CROP SIZE"] = int(preprocess_cfg.crop_size)
            header["HIERARCH SPHERICAL CROP X0 CH0"] = int(offsets[0, 0])
            header["HIERARCH SPHERICAL CROP Y0 CH0"] = int(offsets[0, 1])
            header["HIERARCH SPHERICAL CROP X0 CH1"] = int(offsets[1, 0])
            header["HIERARCH SPHERICAL CROP Y0 CH1"] = int(offsets[1, 1])
        header["HIERARCH SPHERICAL FILTER"] = filter_comb

        fits.writeto(
            converted_outputdir / f"{key.lower()}_cube.fits",
            cube, header=header, overwrite=True,
        )
        fits.writeto(
            converted_outputdir / f"{key.lower()}_ivar_cube.fits",
            ivar,
            header=header,
            overwrite=True,
        )

    wave, _bandwidth = wavelength_bandwidth_filter(filter_comb)
    fits.writeto(
        converted_outputdir / "wavelengths.fits",
        np.asarray(wave, dtype=np.float32),
        overwrite=True,
    )
    # Phase 3 detector bpm dead-region-clamped for safety (already the case in
    # Phase 3, but defensive). Per-frame transient sigma-clip hits are already
    # reflected as ivar=0 in the ivar cubes and are NOT unioned in here — see
    # `preprocess_frame_type` for the design note.
    from spherical.pipeline.steps.irdis_calibration import dead_region_mask
    dead = dead_region_mask()
    bpm_out = (bpm.astype(bool) & ~dead).astype(np.uint8)
    fits.writeto(
        converted_outputdir / "badpixel_map.fits",
        bpm_out,
        overwrite=True,
    )

    logger.info(
        f"IRDIS preprocess written to {converted_outputdir}",
        extra={"step": "preprocess_irdis", "status": "success"},
    )
