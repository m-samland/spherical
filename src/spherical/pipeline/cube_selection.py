"""Choose which cubes of a frame type within a sequence to use.

``frames_info`` tables are row-per-DIT (see
:func:`spherical.database.metadata.expand_frames_info`), but the unit of "which
exposure should we use" is the raw file: one ``ORIGFILE``, one instrumental
setup, one observer decision.  FLUX uses this today; issue #143 (which CENTER
exposure calibrates the CORO sequence) is the same problem.

Orthogonal to ``exclude_first_flux_frame``, which drops the first DIT *inside*
a cube.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SATURATION_CHECKED = ("auto", "before", "after")


def core_peaks(cube, guesses, box=15):
    """Peak value in a box around the PSF, per (wavelength, frame).

    Not a whole-frame maximum: an IRDIS flux frame is an uncropped detector half
    and FLUX skips the transient sigma-clip in pre-processing, so one hot pixel
    would condemn an otherwise good cube.

    Args:
        cube: Flux cube, shape ``(n_wave, n_frames, ny, nx)``.
        guesses: One entry per frame, either one ``(cy, cx)`` for all channels
            (IFS) or one ``(cy, cx)`` per channel (IRDIS, #170).
        box: Half-width of the box in pixels.

    Returns:
        Array of shape ``(n_wave, n_frames)``, NaN where the box has no finite pixel.
    """
    n_wave, n_frames = cube.shape[:2]
    peaks = np.full((n_wave, n_frames), np.nan)
    for f in range(n_frames):
        per_channel = np.broadcast_to(np.asarray(guesses[f], dtype=int).reshape(-1, 2), (n_wave, 2))
        for ch, (cy, cx) in enumerate(per_channel):
            window = cube[ch, f, max(0, cy - box):cy + box + 1, max(0, cx - box):cx + box + 1]
            if np.isfinite(window).any():
                peaks[ch, f] = np.nanmax(window)
    return peaks


def summarise_cubes(frames_info, exposure_level, peaks=None):
    """Collapse a row-per-DIT ``frames_info`` into one row per cube, in time order.

    ``peak_adu`` is the median over the cube's frames of the brightest channel's
    core peak, so a single cosmic-ray frame cannot condemn a cube.

    Args:
        frames_info: Row-per-DIT table with ``ORIGFILE``, ``DET SEQ1 DIT``,
            ``INS4 FILT2 NAME`` and ``MJD``.
        exposure_level: ``DIT x ND transmission`` per frame.
        peaks: Output of :func:`core_peaks`, or None when nothing was measured.
    """
    n_frames = len(frames_info)
    per_frame_peak = np.full(n_frames, np.nan) if peaks is None else np.fmax.reduce(peaks, axis=0)
    if len(exposure_level) != n_frames or len(per_frame_peak) != n_frames:
        raise ValueError(
            f"frames_info has {n_frames} rows, but exposure_level has {len(exposure_level)} "
            f"and peaks has {len(per_frame_peak)} frames."
        )
    frames = pd.DataFrame({
        "origfile": frames_info["ORIGFILE"].to_numpy(),
        "dit": frames_info["DET SEQ1 DIT"].to_numpy(dtype=float),
        "nd_filter": frames_info["INS4 FILT2 NAME"].to_numpy(),
        "mjd": frames_info["MJD"].to_numpy(dtype=float),
        "exposure_level": np.asarray(exposure_level, dtype=float),
        "peak_adu": per_frame_peak,
    })
    return frames.groupby("origfile", sort=False).agg(
        dit=("dit", "first"),
        nd_filter=("nd_filter", "first"),
        n_dits=("dit", "size"),
        mjd=("mjd", "mean"),
        exposure_level=("exposure_level", "first"),
        peak_adu=("peak_adu", "median"),
    ).reset_index()


def resolve_cube_selection(cubes, selection, science_mid_mjd):
    """The cubes a selection addresses, before any saturation check.

    Args:
        cubes: Output of :func:`summarise_cubes`.
        selection: ``"auto"``, ``"all"``, ``"before"``, ``"after"``, a
            time-ordered cube index, or an ``ORIGFILE``.
        science_mid_mjd: Midpoint of the science sequence; ``"before"`` and
            ``"after"`` split the cubes on it.
    """
    n = len(cubes)
    mjd = cubes["mjd"].to_numpy(dtype=float)
    if selection in ("auto", "all"):
        mask = np.ones(n, dtype=bool)
    elif selection == "before":
        mask = mjd < science_mid_mjd
    elif selection == "after":
        mask = mjd > science_mid_mjd
    elif isinstance(selection, (int, np.integer)) and not isinstance(selection, bool):
        mask = np.zeros(n, dtype=bool)
        if -n <= selection < n:
            mask[selection] = True
    else:
        mask = cubes["origfile"].to_numpy(dtype=str) == str(selection)
    if not mask.any():
        raise ValueError(
            f"Flux cube selection {selection!r} matches no cube; available, in time order: "
            f"{list(cubes['origfile'])}"
        )
    return mask


def select_flux_cubes(cubes, saturated, science_mid_mjd, selection="auto"):
    """Which cubes calibrate the PSF.

    ``"auto"``, ``"before"`` and ``"after"`` drop saturated cubes and keep every
    other cube they address; DIT and ND are scaled per frame downstream, so
    mixed setups combine correctly.  If every addressed cube is saturated, the
    one(s) with the lowest exposure level are kept, being the least clipped.
    ``"all"``, an index or an ``ORIGFILE`` are taken literally.

    Returns:
        ``(keep, status)``: a bool per cube, and one of ``"all_kept"``,
        ``"dropped_saturated"``, ``"all_saturated"`` or ``"forced"``.
    """
    candidates = resolve_cube_selection(cubes, selection, science_mid_mjd)
    if selection not in _SATURATION_CHECKED:
        return candidates, "forced"

    clean = candidates & ~np.asarray(saturated, dtype=bool)
    if clean.any():
        return clean, "all_kept" if clean.sum() == candidates.sum() else "dropped_saturated"

    level = cubes["exposure_level"].to_numpy(dtype=float)
    return candidates & np.isclose(level, level[candidates].min()), "all_saturated"
