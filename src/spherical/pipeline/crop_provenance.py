"""The crop provenance cards, and the one place that knows their names.

With ``irdis_preprocessing.crop`` on, every cropped product is delivered in crop
coordinates rather than detector coordinates. Which frame a file is in is not
recoverable from the data, only from these cards, so a product that is cropped,
or that carries coordinates measured on cropped data, has to say so. A consumer
that assumes detector coordinates is wrong by the crop origin, which is hundreds
of pixels, with nothing to raise on.

Written by the preprocess step, read by the centre fit
(:mod:`spherical.pipeline.steps.find_star`) and the centre propagation
(:mod:`spherical.pipeline.steps.process_centers`).

Imports numpy and ``astropy.io.fits`` only.
"""
from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

#: Set on every product the preprocess step writes, cropped or not, so its
#: absence means "written before crop provenance existed" rather than "not
#: cropped". The origin cards are present only when this is True.
CROP_APPLIED = "HIERARCH SPHERICAL CROP APPLIED"
CROP_SIZE = "HIERARCH SPHERICAL CROP SIZE"
#: Per channel, because the two IRDIS channels sit at different detector
#: positions and therefore have different origins for one shared crop size.
CROP_ORIGIN = (
    ("HIERARCH SPHERICAL CROP X0 CH0", "HIERARCH SPHERICAL CROP Y0 CH0"),
    ("HIERARCH SPHERICAL CROP X0 CH1", "HIERARCH SPHERICAL CROP Y0 CH1"),
)

#: Every card this module writes, for copying between products.
CROP_KEYWORDS = (
    (CROP_APPLIED, CROP_SIZE)
    + tuple(k for pair in CROP_ORIGIN for k in pair)
)


def stamp_crop_cards(header, offsets, crop_size) -> fits.Header:
    """Record the crop geometry on ``header`` and return it.

    Args:
        header: The header to stamp. Modified in place.
        offsets: Shape ``(2, 2)`` per-channel ``(x0, y0)`` origins, or ``None``
            when this product was not cropped. ``None`` still writes
            ``CROP APPLIED = False``, which is what lets a reader tell an
            uncropped product from one that predates these cards.
        crop_size: Side length of the delivered crop. Ignored when ``offsets``
            is ``None``.

    Returns:
        The same header, for chaining.
    """
    header[CROP_APPLIED] = bool(offsets is not None)
    if offsets is None:
        return header
    offsets = np.asarray(offsets)
    header[CROP_SIZE] = int(crop_size)
    for ch, (kx, ky) in enumerate(CROP_ORIGIN):
        header[kx] = int(offsets[ch, 0])
        header[ky] = int(offsets[ch, 1])
    return header


def copy_crop_cards(dst, src) -> fits.Header:
    """Copy whichever crop cards ``src`` carries onto ``dst``, and return it.

    For products that are not themselves images but whose values are in the
    cube's coordinate frame, such as the measured centres. Cards absent from
    ``src`` are not invented: an IFS reduction has none, and a cube written
    before this provenance existed has none either.

    Args:
        dst: The header to stamp. Modified in place.
        src: The header to read, typically a science cube's.

    Returns:
        ``dst``, for chaining.
    """
    for key in CROP_KEYWORDS:
        if key in src:
            dst[key] = src[key]
    return dst


def read_crop_origins(header):
    """Return the per-channel crop origins, or ``None`` if the product is uncropped.

    Args:
        header: A header carrying the cards :func:`stamp_crop_cards` writes.

    Returns:
        ``(x0, y0)``, each an integer array of length 2 indexed by channel, or
        ``None`` when ``CROP APPLIED`` is absent or False. ``None`` means the
        coordinates are already in the detector frame, so a caller subtracting
        the origin should subtract nothing rather than zero-by-accident.
    """
    if not bool(header.get(CROP_APPLIED, False)):
        return None
    x0 = np.array([int(header.get(kx, 0)) for kx, _ in CROP_ORIGIN])
    y0 = np.array([int(header.get(ky, 0)) for _, ky in CROP_ORIGIN])
    return x0, y0


def crop_cards_from_cube(converted_dir, cube_names=("center_cube.fits", "coro_cube.fits")):
    """Build a header carrying the crop cards of the first cube that exists.

    CORO and CENTER share one crop origin by construction, so either answers for
    both. Returns an empty header when no cube is found or none is stamped,
    which leaves the caller writing a product with no crop cards rather than
    failing: that is the correct outcome for IFS and for older reductions.

    Args:
        converted_dir: The observation's ``converted/`` directory.
        cube_names: Cubes to try, in order.

    Returns:
        A :class:`astropy.io.fits.Header` holding only crop cards, possibly none.
    """
    out = fits.Header()
    for name in cube_names:
        path = os.path.join(str(converted_dir), name)
        if os.path.exists(path):
            return copy_crop_cards(out, fits.getheader(path))
    return out
