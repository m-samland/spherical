"""Bad-pixel masks derived from an inverse-variance cube.

**Since charis's variance-propagating resample (charis issue 013),
``ivar == 0`` is the primary bad-spaxel test on IFS**, and this local-baseline
detector is a *fallback* (and the tool the IRDIS path uses when its calibration
``badpixel_map.fits`` is missing). The IFS default ``ratio_threshold`` is
therefore ``0.0`` — the exact zeros are caught by the ``ivar <= 0`` branch, and
the soft test is off (see below for why). The history is worth keeping because it
explains both the fallback role and why a global threshold cannot be used.

Historically, ``ivar == 0`` found nothing on IFS — on the pre-fix 51 Eri OBS_H
cubes there was not one exact zero, hex or resampled. Two reasons:

1. charis zeroes ``ivar`` on the *raw detector frame* before the optimal
   extraction (``extractcube.py``). The extraction is a weighted sum over
   detector pixels, so a dead pixel only removes weight: the extracted spaxel
   emerges with reduced but non-zero ivar.
2. charis *does* flag bad spaxels after extraction —
   ``fit_psflets._smoothandmask_hexgeometry`` rejects lenslets deviating from
   their hexagonal neighbours in ivar or flux, separately at each wavelength —
   but it wrote a sentinel ``1e-15`` rather than a zero until charis
   ``feature/ivar`` (merged 2026-07-28) changed it to ``0``.

Even with the sentinel at ``0``, an exact-value test did not survive the
hexagonal-to-square resampling *while that resample applied the flux operator to
ivar*: a flagged lenslet's zero was averaged against good neighbours. charis
issue 013 fixed the resample to propagate variance, so a square that draws from
any flagged lenslet is now an exact ``ivar == 0`` (measured ~6-11% of the
illuminated field per channel on 51 Eri OBS_H, matching charis's own flags).
That is why the exact-zero test now works and the soft skirt-recovery this module
was built for is obsolete on IFS: the corrected resample also imprints a real
3-5x moiré on the ivar, so a positive ``ratio_threshold`` only flags good moiré
troughs (~0.1-0.3% of the field at 0.2) rather than real defects.

When it *is* enabled (IRDIS fallback, or a deliberately non-zero threshold),
:func:`bad_pixel_mask_from_ivar` thresholds against a *local* baseline rather
than a global one: ivar legitimately drops one to two orders of magnitude inside
the stellar halo (on 51 Eri, 76% of the pixels within 6 px of the star sit below
half the field median at 1.56 um), so a global cut would flag the entire inner
region. Comparing each pixel to the median of its own neighbourhood removes that
smooth radial component and leaves the isolated deficits that mark damaged
spaxels.

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
    in_field: np.ndarray | None = None,
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
        its ``filter_size`` x ``filter_size`` neighbourhood, *in addition to* the
        always-flagged exact zeros / non-finite values. ``0.0`` disables this soft
        test, leaving the mask equal to ``ivar <= 0`` within the illuminated
        field — the correct choice on IFS since charis issue 013 (see module
        docstring). A positive value re-enables the soft test (IRDIS fallback);
        on the moiré-carrying IFS ivar it mostly flags good troughs.
    filter_size
        Side length of the median-filter window defining the local baseline. Must
        be large enough to span a bad-pixel cluster and small enough to track the
        halo gradient; 7 px is a good compromise at both IFS and IRDIS sampling.
    illuminated_threshold
        Only used when ``in_field`` is ``None``. Pixels whose local baseline falls
        below this fraction of the image's own median baseline are treated as
        outside the illuminated field (the resampled IFS cube is square but the
        lenslet array is not, leaving ~32% of the frame at a floor value) and are
        never flagged. This local-baseline proxy for "in field" fails inside a
        *cluster* of exact zeros, where the baseline itself collapses to ~0 — see
        ``in_field``.
    in_field
        Optional boolean mask, broadcastable to ``ivar.shape``, that states
        directly which pixels are in the illuminated field (``True``). When given,
        it replaces the ``illuminated_threshold`` baseline proxy: every in-field
        ``ivar <= 0`` pixel is flagged (so exact-zero *clusters* no longer escape),
        and out-of-field pixels are never flagged (the reduction footprint handles
        those). The natural source is ``np.isfinite(data)`` — the unilluminated
        border is NaN in the data but merely 0 in ivar, so ivar alone cannot tell
        an in-field bad-lenslet cluster from the border.

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

    flat_field = None
    if in_field is not None:
        field_full = np.broadcast_to(np.asarray(in_field, dtype=bool), ivar.shape)
        flat_field = field_full.reshape(-1, *ivar.shape[-2:])

    for i, plane in enumerate(flat):
        unusable = ~np.isfinite(plane) | (plane <= 0)
        clean = np.where(unusable, 0.0, plane)
        baseline = median_filter(clean, size=filter_size)

        if flat_field is not None:
            field = flat_field[i]
        else:
            positive = baseline[baseline > 0]
            if positive.size == 0:
                mask[i] = unusable
                continue
            field = baseline > illuminated_threshold * np.median(positive)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(baseline > 0, clean / baseline, np.inf)
        mask[i] = field & (unusable | (ratio < ratio_threshold))

    return mask.reshape(ivar.shape)
