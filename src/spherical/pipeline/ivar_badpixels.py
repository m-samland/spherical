"""Bad-pixel masks derived from an inverse-variance cube.

The IRDIS path gets a genuine boolean bad-pixel map from calibration, and its
``ivar`` cube carries hard zeros at those pixels. The IFS path does not: charis
zeroes ``ivar`` on the *raw detector frame* before the optimal extraction
(``extractcube.py``), so a dead detector pixel only removes weight from the
lenslet it feeds — the extracted spaxel emerges with reduced but non-zero ivar,
and the hexagonal-to-square resampling then averages it against its neighbours.
Measured on 51 Eri OBS_H there is not a single exact zero anywhere in the IFS
cubes, which silently turns every ``ivar == 0`` test into a no-op.

:func:`bad_pixel_mask_from_ivar` replaces that test with a threshold against a
*local* baseline. A global threshold cannot work: ivar legitimately drops one to
two orders of magnitude inside the stellar halo (on 51 Eri, 76% of the pixels
within 6 px of the star sit below half the field median at 1.56 um), so a global
cut would flag the entire inner region. Comparing each pixel to the median of its
own neighbourhood removes that smooth radial component and leaves the isolated
deficits that mark damaged spaxels.

The resulting mask is inherently three-dimensional. A lenslet's spectrum is
dispersed across the detector, so a dead detector pixel kills one
``(lenslet, wavelength)`` pair rather than a whole spaxel: on 51 Eri, 11,476
distinct lenslets are flagged in at least one of 39 channels but only 7 in twenty
or more, and none in all of them. Collapsing the mask over wavelength would
therefore either discard almost everything or mask two thirds of the field.
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
