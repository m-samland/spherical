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

from spherical.pipeline.science_frames import WAFFLE_KEYWORD

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


def resolve_waffle_mode(converted_dir, continuous_satellite_spots: bool | None) -> bool:
    """Return the WAFFLE_MODE flag, from the caller or from the cube header.

    Inside the pipeline the orchestrator holds the observation and passes the
    flag, so this only falls back for a standalone re-run on an already-reduced
    dataset. It reads the keyword the ``cube_header_update`` step stamps into
    the cubes rather than guessing from file names or frame counts: the centres
    carry the CENTER
    frame axis in three of the four instrument x waffle cases, so frame counts
    cannot tell the cases apart, and a wrong answer would silently align the
    calibration frames instead of the science ones.

    Args:
        converted_dir: The observation's ``converted/`` directory.
        continuous_satellite_spots: The flag, or ``None`` to read the header.

    Returns:
        The WAFFLE_MODE flag.

    Raises:
        ValueError: When the flag is omitted and no cube carries the keyword,
            which is the case for reductions made before it was introduced.
    """
    from pathlib import Path

    from astropy.io import fits

    if continuous_satellite_spots is not None:
        return bool(continuous_satellite_spots)

    converted_dir = Path(converted_dir)
    for name in ("center_cube.fits", "coro_cube.fits"):
        path = converted_dir / name
        if not path.exists():
            continue
        value = fits.getheader(path).get(WAFFLE_KEYWORD)
        if value is not None:
            return bool(value)

    raise ValueError(
        f"No {WAFFLE_KEYWORD} keyword in the cubes under {converted_dir}, so the "
        "science frame type is unknown. Reductions made before the keyword was "
        "introduced do not carry it. Pass continuous_satellite_spots explicitly "
        "(the observation's WAFFLE_MODE), or re-run the cube_header_update step."
    )


def _run_bad_pixel_repair(
    instrument: str,
    alignment_config,
    logger,
    fix_badpix: bool | None = None,
) -> bool | None:
    """Bad-pixel repair gate. Returns whether repaired data reaches the shift.

    The scaffolding is deliberately identical for both instruments so the gap is
    explicit in the log rather than implicit in the structure. On IRDIS the
    requirement is met by ``fix_badpix`` in preprocess, which is a separate
    config flag, so this cannot answer for IRDIS without being told what it was.
    On IFS the interpolation is not implemented: charis marks bad lenslets as
    ``ivar == 0`` and the repair has to work on the extracted spaxel grid rather
    than the detector, which is its own design.

    Returns:
        True when repaired data reaches the shift, False when it does not, and
        ``None`` when it cannot be known, which is the standalone re-run with no
        preprocess config to hand. ``None`` is recorded as ``UNKNOWN`` rather
        than being collapsed into either answer.
    """
    if not alignment_config.repair_bad_pixels:
        return False
    if instrument == "IRDIS":
        return fix_badpix
    logger.warning(
        "IFS bad-pixel interpolation not yet implemented. Skipped.",
        extra={"step": "frame_alignment", "status": "repair_not_implemented"},
    )
    return False


def run_frame_alignment(
    converted_dir,
    alignment_config,
    logger,
    continuous_satellite_spots: bool | None = None,
    fix_badpix: bool | None = None,
):
    """Write a copy of the science cube with the star on the centre pixel.

    The instrument is derived from the cube's wavelength axis and the science
    frame type from ``WAFFLE_MODE``, so this runs standalone on an
    already-reduced dataset without an observation object. Centres are already
    in the science cube's own coordinates, so no crop offset is applied here.

    Args:
        converted_dir: The observation's ``converted/`` directory.
        alignment_config: Shift method, pad width and the bad-pixel repair gate.
        logger: Pipeline logger adapter.
        continuous_satellite_spots: The observation's ``WAFFLE_MODE`` flag.
            Read from the cube header when omitted, which is the standalone
            re-run path.
        fix_badpix: The IRDIS preprocess ``fix_badpix`` flag, which is what
            actually satisfies the repair requirement on that instrument. The
            orchestrator passes it; omitting it records the repair state as
            ``UNKNOWN`` rather than claiming one.

    Returns:
        The :class:`pathlib.Path` of the aligned cube that was written.
    """
    from pathlib import Path

    import pandas as pd
    from astropy.io import fits

    from spherical.pipeline.science_frames import (
        normalize_centers_to_frames,
        science_frame_type,
        verify_frame_axis,
    )

    converted_dir = Path(converted_dir)
    continuous_satellite_spots = resolve_waffle_mode(
        converted_dir, continuous_satellite_spots
    )
    identifier = science_frame_type(continuous_satellite_spots)

    cube_path = converted_dir / f"{identifier}_cube.fits"
    with fits.open(cube_path) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float32)
        source_header = hdul[0].header.copy()

    n_wave = cube.shape[0]
    instrument = "IRDIS" if n_wave == 2 else "IFS"

    n_frames = len(pd.read_csv(converted_dir / f"frames_info_{identifier}.csv"))
    centers = np.asarray(
        fits.getdata(converted_dir / "image_centers_fitted_robust.fits"),
        dtype=np.float64,
    )
    centers = normalize_centers_to_frames(
        centers, n_frames, instrument, continuous_satellite_spots
    )
    verify_frame_axis(centers, n_frames, identifier)
    if cube.shape[1] != n_frames:
        raise ValueError(
            f"{cube_path.name} has {cube.shape[1]} frames but "
            f"frames_info_{identifier}.csv has {n_frames} rows."
        )

    repaired = _run_bad_pixel_repair(
        instrument, alignment_config, logger, fix_badpix=fix_badpix
    )

    original_size = cube.shape[-1]
    cube = pad_to_odd(cube)
    if cube.shape[-1] != original_size:
        logger.info(
            f"Padded {original_size} -> {cube.shape[-1]} at the high edge so the "
            "target pixel is both N//2 and the geometric centre.",
            extra={"step": "frame_alignment", "status": "padded"},
        )

    target = cube.shape[-1] // 2
    aligned = np.empty_like(cube)
    for w in range(n_wave):
        for f in range(n_frames):
            aligned[w, f] = shift_to_target(
                cube[w, f],
                (centers[w, f, 0], centers[w, f, 1]),
                method=alignment_config.shift_method,
                pad=alignment_config.pad_width,
            )

    header = source_header
    header["HIERARCH SPHERICAL ALIGNED"] = True
    header["HIERARCH SPHERICAL ALIGN TARGET X"] = int(target)
    header["HIERARCH SPHERICAL ALIGN TARGET Y"] = int(target)
    header["HIERARCH SPHERICAL ALIGN METHOD"] = str(alignment_config.shift_method)
    header["HIERARCH SPHERICAL ALIGN PAD"] = int(alignment_config.pad_width)
    # Never claim a repair that cannot be confirmed: a reader checking this
    # keyword is asking exactly the question UNKNOWN answers honestly.
    header["HIERARCH SPHERICAL ALIGN REPAIRED"] = (
        "UNKNOWN" if repaired is None else bool(repaired)
    )
    header.add_comment(
        "Aligned cube: for classical post-processing and display, not weighted "
        "inference. No shifted inverse variance is provided because "
        "interpolation correlates neighbouring pixels."
    )

    out_path = converted_dir / f"{identifier}_cube_aligned.fits"
    fits.writeto(out_path, aligned, header=header, overwrite=True)
    logger.info(
        f"Wrote {out_path.name}: {aligned.shape}, star on pixel "
        f"({target}, {target}), method={alignment_config.shift_method}.",
        extra={"step": "frame_alignment", "status": "success"},
    )
    return out_path
