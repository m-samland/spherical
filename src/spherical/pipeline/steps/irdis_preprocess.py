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
        num = float((mask * frame_ch * bg_ch).sum())
        den = float((mask * bg_ch * bg_ch).sum())
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
    finite; the neighbor-exclusion mask passed as ``bpm`` unions in dead
    pixels + any residual NaNs so those pixels are excluded from the neighbor
    pool. After the call, dead-region pixels are re-set to NaN
    unconditionally (Phase 3 downstream-contract invariant).

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

    # Neighbor-exclusion mask: bad pixels themselves, dead-region pixels,
    # and any residual NaN in the frame. imutils.fix_badpix reads bpm as
    # "target OR excluded neighbor" — since targets and excluded neighbors
    # are exactly the same set of pixels to skip when averaging, passing
    # the union works. Dead pixels are still False in bpm_ch so they are
    # not themselves interpolated by the vendored routine's target loop.
    neighbor_bpm = bpm_ch | dead_mask_ch | (~finite)

    fixed = imutils.fix_badpix(frame_work, neighbor_bpm.astype(np.uint8), npix=npix, weight=True)

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
