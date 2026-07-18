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

    return (nom_x, nom_y)
