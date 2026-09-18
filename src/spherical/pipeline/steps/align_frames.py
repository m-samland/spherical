"""Optional frame alignment: a science cube with the star on a fixed pixel.

A leaf product. Nothing in the pipeline consumes it — TRAP forward-models the
centre instead, spot photometry uses spot positions on the unshifted cube, and
the flux PSF is an independently extracted stamp. It exists for external
consumers: classical ADI/PCA, SDI, and visual inspection.

No inverse variance is written alongside it. Interpolation correlates
neighbouring pixels, so a shifted ivar is no longer a valid per-pixel weight;
that is the same reason TRAP works on unshifted data.
"""
from __future__ import annotations

import numpy as np

# Padding carried around the frame during the shift and removed afterwards.
# An FFT shift is periodic and would wrap flux from one edge to the other; a
# cubic spline's support reaches 2 px past the border. Shifts are sub-pixel once
# the crop origin puts the star within half a pixel of the centre, so 8 is
# generous for both.
DEFAULT_PAD_WIDTH = 8

_VALID_METHODS = ("auto", "fft", "interp", "coarse")


def pad_to_odd(cube: np.ndarray) -> np.ndarray:
    """Pad the last two axes to odd length at the high edge, by edge replication.

    IFS cubes arrive at 262x262 from the charis extraction. With an even axis
    TRAP's ``N // 2`` centre sits half a pixel off the geometric centre, and the
    axis carries a lone Nyquist bin with no conjugate partner that leaks ringing
    into a real-valued FFT shift. Padding 262 to 263 fixes both without losing
    data or resampling: the target pixel is 131 either way, and in a 263 array
    131 is both ``N // 2`` and the exact geometric centre.

    Edge replication rather than NaN, so the added row and column do not create a
    new invalid stripe that the NaN-restore path then has to carry.

    Args:
        cube: Any array whose last two axes are spatial.

    Returns:
        The input unchanged when both axes are already odd, otherwise a padded
        copy.
    """
    h, w = cube.shape[-2], cube.shape[-1]
    pad_y, pad_x = (h % 2 == 0), (w % 2 == 0)
    if not (pad_y or pad_x):
        return cube
    pad_width = [(0, 0)] * (cube.ndim - 2) + [(0, int(pad_y)), (0, int(pad_x))]
    return np.pad(cube, pad_width, mode="edge")


def _resolve_method(frame: np.ndarray, method: str) -> str:
    """Resolve ``"auto"`` to ``"fft"`` on clean frames, ``"interp"`` where NaN is."""
    if method != "auto":
        return method
    return "interp" if np.isnan(frame).any() else "fft"


def shift_frame(
    frame: np.ndarray,
    shift_xy: tuple[float, float],
    method: str = "auto",
    pad: int = DEFAULT_PAD_WIDTH,
) -> np.ndarray:
    """Shift one square frame by ``(dx, dy)`` pixels, preserving its shape.

    ``"auto"`` picks FFT when the frame has no NaN and a cubic spline when it
    does. FFT avoids the spline's photometric smoothing, but it is global, so
    ringing from a filled NaN edge would spread across the whole frame; the
    spline keeps it local. ``"coarse"`` rounds the shift to an integer and does
    no interpolation at all.

    NaN is filled before the shift and restored afterwards from a
    nearest-neighbour-shifted validity mask, so invalid pixels move with the data
    instead of growing.

    Args:
        frame: Square 2-D image.
        shift_xy: ``(dx, dy)`` in pixels; ``dx`` moves along axis 1, ``dy`` along
            axis 0.
        method: ``"auto"``, ``"fft"``, ``"interp"`` or ``"coarse"``.
        pad: Edge-replicated padding added before the shift and removed after.

    Returns:
        Shifted frame, same shape as the input, as float32.

    Raises:
        ValueError: If ``method`` is unknown, or the frame is not square.
    """
    from scipy import ndimage

    from spherical.pipeline import imutils

    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown shift method {method!r}; expected one of {_VALID_METHODS}."
        )
    if frame.ndim != 2 or frame.shape[0] != frame.shape[1]:
        raise ValueError(f"shift_frame requires a square 2-D frame, got {frame.shape}.")

    dx, dy = float(shift_xy[0]), float(shift_xy[1])
    resolved = _resolve_method(frame, method)

    invalid = np.isnan(frame)
    has_nan = bool(invalid.any())
    fill = float(np.nanmedian(frame)) if has_nan and np.isfinite(frame).any() else 0.0
    work = np.where(invalid, fill, frame).astype(np.float64)

    padded = np.pad(work, pad, mode="edge")

    if resolved == "coarse":
        idx, idy = int(round(dx)), int(round(dy))
        shifted = np.roll(np.roll(padded, idx, axis=1), idy, axis=0)
    elif resolved == "fft":
        shifted = imutils.shift(padded, (dx, dy), method="fft")
    else:
        shifted = ndimage.shift(padded, (dy, dx), order=3, mode="nearest")

    out = shifted[pad:pad + frame.shape[0], pad:pad + frame.shape[1]]

    if has_nan:
        mask_padded = np.pad(invalid.astype(np.float64), pad, mode="edge")
        # Nearest-neighbour so the mask stays boolean: a spline or FFT would
        # smear the invalid region outward every time the cube is aligned.
        mask_shifted = ndimage.shift(mask_padded, (dy, dx), order=0, mode="nearest")
        mask_out = mask_shifted[pad:pad + frame.shape[0], pad:pad + frame.shape[1]]
        out = np.where(mask_out > 0.5, np.nan, out)

    return out.astype(np.float32)


def shift_to_target(
    frame: np.ndarray,
    center_xy: tuple[float, float],
    method: str = "auto",
    pad: int = DEFAULT_PAD_WIDTH,
) -> np.ndarray:
    """Shift ``frame`` so the star at ``center_xy`` lands on ``N // 2``.

    For odd ``N`` that pixel is both TRAP's image centre and the array's exact
    geometric centre.

    Args:
        frame: Square 2-D image with odd side length.
        center_xy: Measured star position ``(x, y)`` in this frame's own
            coordinates.
        method: See :func:`shift_frame`.
        pad: See :func:`shift_frame`.

    Returns:
        Shifted frame, same shape as the input.
    """
    target = frame.shape[-1] // 2
    return shift_frame(
        frame,
        (target - float(center_xy[0]), target - float(center_xy[1])),
        method=method,
        pad=pad,
    )
