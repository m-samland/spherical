"""Bad-pixel masks derived from an inverse-variance cube.

The IRDIS path gets a genuine boolean bad-pixel map from calibration, and its
``ivar`` cube carries hard zeros at those pixels. On IFS, ``ivar == 0`` finds
nothing at all — measured on 51 Eri OBS_H there is not one exact zero anywhere,
hex or resampled — which silently turns every such test into a no-op. Two
independent reasons, both worth knowing:

1. charis zeroes ``ivar`` on the *raw detector frame* before the optimal
   extraction (``extractcube.py``). The extraction is a weighted sum over
   detector pixels, so a dead pixel only removes weight: the extracted spaxel
   emerges with reduced but non-zero ivar.
2. charis *does* flag bad spaxels after extraction —
   ``fit_psflets._smoothandmask_hexgeometry`` rejects lenslets deviating from
   their hexagonal neighbours in ivar or flux, separately at each wavelength —
   but it writes a **sentinel value rather than a zero**. That sentinel was
   ``1e-15`` until charis ``feature/ivar`` changed it to ``0`` (merged into
   ``devel`` 2026-07-28); cubes extracted before that carry the old value.

Even with the sentinel at ``0``, an exact-value test does not survive the
hexagonal-to-square resampling that produces the cubes this package consumes.
The resample is an area-weighted average, so a flagged lenslet only reaches a
square pixel undiluted when that pixel lies entirely inside it: on 51 Eri, 633
flagged lenslets per channel each span ~2.6 square pixels, yet only 12 of 47,014
illuminated square pixels retain the pure sentinel level (``1e-15 x 0.3849``, the
square:hex area ratio). Everything else is blended with good neighbours.

:func:`bad_pixel_mask_from_ivar` therefore thresholds against a *local* baseline,
which recovers both the sentinel spaxels and the diluted skirt around them. A
global threshold cannot work: ivar legitimately drops one to two orders of
magnitude inside the stellar halo (on 51 Eri, 76% of the pixels within 6 px of
the star sit below half the field median at 1.56 um), so a global cut would flag
the entire inner region. Comparing each pixel to the median of its own
neighbourhood removes that smooth radial component and leaves the isolated
deficits that mark damaged spaxels.

That this reproduces charis's own judgement is checkable, and it does: on the
51 Eri hex cube the local-baseline test flags ~667 lenslets per channel against
the 633 charis itself sentinelled.

The resulting mask is inherently three-dimensional, and again charis agrees. A
lenslet's spectrum is dispersed across the detector, so a bad detector pixel
kills one ``(lenslet, wavelength)`` pair rather than a whole spaxel: 11,476
distinct lenslets are flagged in at least one of 39 channels but only 7 in twenty
or more, and none in all of them (charis's own sentinel: 11,088 / 3 / 0).
Collapsing the mask over wavelength would therefore either discard almost
everything or mask two thirds of the field.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

__all__ = ["bad_pixel_mask_from_ivar"]


def bad_pixel_mask_from_ivar(
    ivar: np.ndarray,
    ratio_threshold: float = 0.2,
    filter_size: int = 7,
    illuminated_threshold: float = 1e-6,
) -> np.ndarray:
    """Flag pixels whose inverse variance is anomalously low for their surroundings.

    Parameters
    ----------
    ivar
        Inverse-variance array, ``(..., ny, nx)``. The last two axes are treated
        as the image; every leading axis (wavelength, frame) is processed
        independently, so the returned mask keeps the input's per-channel and
        per-frame structure.
    ratio_threshold
        A pixel is flagged when ``ivar`` is below this fraction of the median of
        its ``filter_size`` x ``filter_size`` neighbourhood. ``0.2`` flags roughly
        2% of the illuminated IFS field per channel; loosening it towards ``0.5``
        recovers more of the area a bad lenslet spreads over during resampling at
        the cost of more false positives.
    filter_size
        Side length of the median-filter window defining the local baseline. Must
        be large enough to span a bad-pixel cluster and small enough to track the
        halo gradient; 7 px is a good compromise at both IFS and IRDIS sampling.
    illuminated_threshold
        Pixels whose local baseline falls below this fraction of the image's own
        median baseline are outside the illuminated field (the resampled IFS cube
        is square but the lenslet array is not, leaving ~32% of the frame at a
        floor value). They are never flagged — they carry no data to repair, and
        flagging them would swamp the mask.

    Returns
    -------
    numpy.ndarray
        Boolean array of the same shape as ``ivar``. ``True`` marks a bad pixel,
        matching TRAP's ``bad_pixel_mask_full`` convention.

    Notes
    -----
    Non-finite and non-positive ivar values are always flagged: they carry no
    weight regardless of their surroundings.
    """
    ivar = np.asarray(ivar, dtype=float)
    if ivar.ndim < 2:
        raise ValueError(f"ivar must have at least 2 dimensions, got {ivar.ndim}")
    if filter_size < 3 or filter_size % 2 == 0:
        raise ValueError(f"filter_size must be an odd integer >= 3, got {filter_size}")

    flat = ivar.reshape(-1, *ivar.shape[-2:])
    mask = np.zeros(flat.shape, dtype=bool)

    for i, plane in enumerate(flat):
        unusable = ~np.isfinite(plane) | (plane <= 0)
        clean = np.where(unusable, 0.0, plane)
        baseline = median_filter(clean, size=filter_size)

        positive = baseline[baseline > 0]
        if positive.size == 0:
            mask[i] = unusable
            continue
        illuminated = baseline > illuminated_threshold * np.median(positive)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(baseline > 0, clean / baseline, np.inf)
        mask[i] = illuminated & (unusable | (ratio < ratio_threshold))

    return mask.reshape(ivar.shape)
