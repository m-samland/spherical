"""The usable field-of-view footprint of a reduced cube.

A pixel is in-field wherever it *ever* carries finite data. This is the same
test :mod:`spherical.pipeline.run_trap` applies when it gates the TRAP
bad-pixel mask, and it is deliberately identical: two notions of "in field" in
one pipeline is how the two drift apart.

The test is ``np.isfinite(data)``, not ``ivar > 0``. On a charis v2.1.0 IFS
reduction the unilluminated border is NaN in the data (30.6% of the frame on
the 51 Eri OBS_H reference run) while the inverse variance carries no zeros at
all -- 685M pixels scanned across the flux, center and coro cubes, uniform
floor 3.849e-16. An ivar-based footprint selects nothing on IFS.

``exclude_edge_pixels`` erodes inward from the invalid region, to keep a caller
away from real detector edge effects -- dead bands, unilluminated lenslets and
the noisy transition into them. It is not a margin against the array boundary,
which is why the erosion uses ``border_value=1``: a cropped frame has no real
border pixels and nothing there needs excluding.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["valid_fov_mask"]


def valid_fov_mask(data: np.ndarray, exclude_edge_pixels: int = 0) -> np.ndarray:
    """Return the usable field-of-view footprint of an image or cube.

    Parameters
    ----------
    data : np.ndarray
        Image or cube, ``(..., ny, nx)``. Leading axes (wavelength, frame) are
        reduced with ``.any()``: a pixel is in-field wherever it is finite in at
        least one plane.
    exclude_edge_pixels : int, optional
        Erode the footprint by this many pixels inward from the invalid region.
        The array boundary is not eroded. Default 0.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``data.shape[-2:]``, ``True`` inside the usable
        field. Erosion alone never empties the footprint: if it would remove
        every pixel, the un-eroded footprint is returned. Data that is
        non-finite everywhere still gives an all-``False`` mask, which callers
        are expected to detect and report.

    Raises
    ------
    ValueError
        If ``data`` has fewer than two dimensions.
    """
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError(
            f"data must have at least 2 dimensions, got shape {data.shape}"
        )

    finite = np.isfinite(data)
    leading = tuple(range(data.ndim - 2))
    footprint = finite.any(axis=leading) if leading else finite

    if exclude_edge_pixels > 0:
        # border_value=1 treats everything beyond the array as in-field, so the
        # array boundary is left alone and only real invalid regions erode.
        eroded = ndimage.binary_erosion(
            footprint, iterations=exclude_edge_pixels, border_value=1
        )
        if eroded.any():
            footprint = eroded

    return footprint
