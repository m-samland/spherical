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


def nominal_star_positions(filter_comb: str) -> np.ndarray:
    """Return per-channel nominal ``(x, y)`` star positions for a filter.

    Dispatches K-band vs. H-band by peak wavelength (``> 2000 nm`` → K),
    matching ``simplified_IRDIS_reduction.py``'s convention. Fallback used
    when a coarse-localization search fails; also used as the search-window
    center in :func:`coarse_star_position`.

    Parameters
    ----------
    filter_comb : str
        IRDIS filter combination string (e.g. ``"DB_K12"``, ``"DB_H23"``,
        ``"BB_H"``). Passed to
        :func:`spherical.pipeline.transmission.wavelength_bandwidth_filter`.

    Returns
    -------
    np.ndarray
        Shape ``(2, 2)`` — rows are channels, columns are ``(x, y)`` in
        per-half detector coordinates.
    """
    wavelengths, _ = wavelength_bandwidth_filter(filter_comb)
    if np.max(np.asarray(wavelengths, dtype=float)) > 2000.0:
        return np.array(NOMINAL_STAR_POSITIONS_K_BAND, dtype=np.float64)
    return np.array(NOMINAL_STAR_POSITIONS_H_BAND, dtype=np.float64)


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


def fix_badpix_nan_safe(
    frame_ch: np.ndarray,
    bpm_ch: np.ndarray,
    dead_mask_ch: np.ndarray,
    npix: int = 12,
) -> np.ndarray:
    """Interpolate bad pixels without pulling NaN from dead-region neighbors.

    Wraps :func:`spherical.pipeline.imutils.fix_badpix`. Preprocess substitutes
    NaN pixels with ``0.0`` so the vendored routine's arithmetic mean stays
    finite; the ``bpm`` argument passed to it is restricted to real bad
    pixels (plus any residual NaN outside the dead region) so the target
    loop does not waste time reprocessing the entire dead region on every
    call. After the call, dead-region pixels are re-set to NaN
    unconditionally (Phase 3 downstream-contract invariant), regardless of
    what imutils computed for them.

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
    from spherical.pipeline import imutils

    finite = np.isfinite(frame_ch)
    frame_work = np.where(finite, frame_ch, 0.0).astype(np.float32)

    # imutils.fix_badpix uses `bpm` both to enumerate targets and to
    # exclude those same pixels from the good-neighbor pool. Passing
    # only real bad pixels (plus any residual NaN in the illuminated
    # region) keeps the target loop small while still preventing the
    # zero-substituted NaN values from being averaged into their own
    # neighborhoods. Dead-region pixels stay `False` in the argument
    # here — but that means imutils treats them as good neighbors with
    # value 0. That is acceptable in practice because (a) Phase 3's bpm
    # already excludes dead regions, so real bad-pixel targets are
    # never inside the dead band, and (b) imutils picks the 12 nearest
    # good pixels for a target, which for any bad pixel more than a
    # few rows from a dead band will all be illuminated pixels, never
    # zero-substituted dead ones. The dead-region invariant is enforced
    # unconditionally by the final `fixed[dead_mask_ch] = np.nan` line,
    # not by anything about the bpm argument's construction.
    target_bpm = bpm_ch | (~finite & ~dead_mask_ch)

    fixed = imutils.fix_badpix(frame_work, target_bpm.astype(np.uint8), npix=npix, weight=True)

    fixed = np.asarray(fixed, dtype=np.float32)
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


def preprocess_frame_type(
    frame_paths: list[str],
    master_flat: np.ndarray,
    master_background: np.ndarray,
    bpm: np.ndarray,
    star_positions_xy: np.ndarray,
    is_flux: bool,
    preprocess_config,
    logger,
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

    Returns
    -------
    cube : np.ndarray
        ``(2, n_time, ny, nx)`` float32 — n_wave-first ordering matching
        the IFS bundle-output convention.
    ivar : np.ndarray
        Same shape as ``cube``, float32.
    bpm_out : np.ndarray
        ``(2, 1024, 1024)`` uint8 — Phase 3 bpm unioned with per-observation
        transient bad pixels; never ``True`` in dead regions. Stays at
        native detector resolution.
    crop_offsets : np.ndarray or None
        ``(2, 2)`` int array of ``[[x0_ch0, y0_ch0], [x0_ch1, y0_ch1]]``
        when cropping is enabled; ``None`` otherwise.
    """
    from spherical.pipeline.steps.irdis_calibration import (
        dead_region_mask,
        split_detector_cube,
    )

    logger.info(
        f"Preprocessing {len(frame_paths)} file(s) "
        f"(is_flux={is_flux}, fix_badpix={preprocess_config.fix_badpix})",
        extra={"step": "preprocess_irdis", "status": "frame_type_started"},
    )

    raw = _load_raw_frames_expanded(list(frame_paths))
    split = split_detector_cube(raw)  # (n_time, 2, 1024, 1024)
    n_time = split.shape[0]

    dead = dead_region_mask()  # (2, 1024, 1024)
    bpm_bool = bpm.astype(bool)

    # Working cubes in (n_wave, n_time, H, W) ordering to match IFS convention.
    cube_out = np.empty((2, n_time, 1024, 1024), dtype=np.float32)
    ivar_out = np.empty((2, n_time, 1024, 1024), dtype=np.float32)
    transient_union = np.zeros((2, 1024, 1024), dtype=bool)

    warn_threshold = int(0.01 * 1024 * 1024)

    for t in range(n_time):
        for ch in range(2):
            frame = split[t, ch]
            star_xy = tuple(float(v) for v in star_positions_xy[ch])

            bg_sub, _s = subtract_scaled_background(
                frame_ch=frame,
                bg_ch=master_background[ch],
                star_xy=star_xy,
                star_mask_radius=preprocess_config.star_mask_radius,
                dead_mask_ch=dead[ch],
                bpm_ch=bpm_bool[ch],
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                divided = bg_sub / master_flat[ch]

            ivar = analytic_ivar(
                counts_after_bg=bg_sub,
                flat=master_flat[ch],
                gain=preprocess_config.gain,
                read_noise=preprocess_config.read_noise,
            )

            divided = np.where(dead[ch], np.nan, divided).astype(np.float32)
            ivar[dead[ch]] = 0.0

            if preprocess_config.fix_badpix:
                divided = fix_badpix_nan_safe(divided, bpm_bool[ch], dead[ch])
                ivar[bpm_bool[ch]] = 0.0

            if not is_flux:
                cleaned, transient_mask = sigma_filter_ignore_dead(
                    divided, dead[ch],
                )
                n_transient = int(transient_mask.sum())
                # Always mark transient outliers in ivar / bpm — they carry no
                # information whether or not we interpolate them.
                ivar[transient_mask] = 0.0
                transient_union[ch] |= transient_mask
                if n_transient > warn_threshold:
                    logger.warning(
                        f"Frame {t} ch{ch}: {n_transient} transient bad pixels "
                        f"(>{warn_threshold}) — no interpolation performed",
                        extra={"step": "preprocess_irdis", "status": "transient_warning"},
                    )
                else:
                    divided = cleaned

            cube_out[ch, t] = divided
            ivar_out[ch, t] = ivar

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

    bpm_out = (bpm_bool | transient_union) & ~dead
    logger.info(
        f"Preprocessing complete: cube shape {cube_out.shape}",
        extra={"step": "preprocess_irdis", "status": "frame_type_complete"},
    )
    return cube_out, ivar_out, bpm_out.astype(np.uint8), crop_offsets


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
    - ``badpixel_map.fits`` — ``(2, 1024, 1024)`` uint8, Phase 3 bpm
      unioned with per-observation transient bad pixels, ``False`` in dead
      regions.

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
    bpm_union = bpm.astype(bool).copy()

    for key in ("CORO", "CENTER", "FLUX"):
        table = observation.frames.get(key)
        if table is None or len(table) == 0:
            logger.info(
                f"Skipping frame type {key}: no frames on observation",
                extra={"step": "preprocess_irdis", "status": f"{key.lower()}_skipped"},
            )
            continue

        paths = [str(p) for p in table["FILE"]]
        cube, ivar, bpm_out, offsets = preprocess_frame_type(
            frame_paths=paths,
            master_flat=master_flat,
            master_background=master_bg,
            bpm=bpm,
            star_positions_xy=star_positions,
            is_flux=(key == "FLUX"),
            preprocess_config=preprocess_cfg,
            logger=logger,
        )
        bpm_union |= bpm_out.astype(bool)

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
    # bpm_union already excludes dead regions — preprocess_frame_type ends
    # with `bpm_out = (bpm_bool | transient_union) & ~dead`, so the union
    # of per-frame-type bpms preserves that invariant.
    fits.writeto(
        converted_outputdir / "badpixel_map.fits",
        bpm_union.astype(np.uint8),
        overwrite=True,
    )

    logger.info(
        f"IRDIS preprocess written to {converted_outputdir}",
        extra={"step": "preprocess_irdis", "status": "success"},
    )
