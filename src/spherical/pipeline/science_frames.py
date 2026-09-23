"""Which frames are the science frames, and how their centres line up with them.

Shared by the TRAP wrapper (:mod:`spherical.pipeline.run_trap`) and the frame
alignment step (:mod:`spherical.pipeline.steps.align_frames`). Both need the same
three answers, and both instruments' conventions are encoded here once.

Imports numpy only — no charis, no trap.
"""
from __future__ import annotations

import numpy as np

#: Header card recording the observation's WAFFLE_MODE on every converted cube.
#: The science frame type follows from it (see :func:`science_frame_type`), and
#: the cubes are the only place a consumer outside the pipeline can read it.
#: Written by the cube_header_update step, read by the frame alignment step.
WAFFLE_KEYWORD = "HIERARCH SPHERICAL WAFFLE MODE"


def science_frame_type(continuous_satellite_spots: bool) -> str:
    """Return the frame type that carries the science, ``"center"`` or ``"coro"``.

    In a continuous-waffle sequence the satellite spots are present throughout,
    so the CENTER frames *are* the science frames. Otherwise the CORO frames are,
    and the CENTER frames exist only to measure the star position.

    Args:
        continuous_satellite_spots: The observation's ``WAFFLE_MODE`` flag.

    Returns:
        ``"center"`` or ``"coro"``. Used as the ``file_identifier`` that names
        ``{identifier}_cube.fits`` and ``frames_info_{identifier}.csv``.
    """
    return "center" if bool(continuous_satellite_spots) else "coro"


def frame_types_present(
    observation, candidates: tuple[str, ...] = ("CORO", "CENTER", "FLUX")
) -> tuple[str, ...]:
    """Return the frame types in ``candidates`` the observation actually has.

    Separate from :func:`science_frame_type`, and not derivable from it.
    ``WAFFLE_MODE`` is a majority-exposure-time test, so a waffle sequence can
    carry CORO frames alongside the CENTER frames that hold its science. Steps
    that write one product per frame type write one for each type here.

    Args:
        observation: An observation object with a ``frames`` mapping.
        candidates: Frame types to consider, in the order returned.

    Returns:
        The subset of ``candidates`` with at least one frame, in that order.
    """
    frames = observation.frames
    return tuple(
        ft for ft in candidates
        if frames.get(ft) is not None and len(frames[ft]) > 0
    )


def normalize_centers_to_frames(
    centers: np.ndarray,
    n_frames: int,
    instrument: str,
    continuous_satellite_spots: bool,
) -> np.ndarray:
    """Put ``image_centers_fitted_robust`` on the science cube's frame axis.

    Only one of the four instrument x waffle cases needs work:

    ============ ============ ===========================================
    instrument   waffle       shape of ``image_centers_fitted_robust``
    ============ ============ ===========================================
    IRDIS        yes          ``(2, n_center, 2)`` — already the science axis
    IRDIS        no           ``(2, n_coro, 2)`` — DMS-propagated per CORO frame
    IFS          yes          ``(39, n_center, 2)`` — already the science axis
    IFS          no           ``(39, n_center, 2)`` — collapse over time, broadcast
    ============ ============ ===========================================

    Args:
        centers: Shape ``(n_wave, n, 2)``.
        n_frames: Number of frames in the science cube.
        instrument: ``"IRDIS"`` or ``"IFS"`` (case-insensitive).
        continuous_satellite_spots: The observation's ``WAFFLE_MODE`` flag.

    Returns:
        Array of shape ``(n_wave, n_frames, 2)``.

    Raises:
        ValueError: If ``instrument`` is neither IRDIS nor IFS.
    """
    instrument = str(instrument).upper()
    if instrument not in ("IRDIS", "IFS"):
        raise ValueError(f"Unknown instrument {instrument!r}; expected IRDIS or IFS.")
    centers = np.asarray(centers)
    if instrument == "IFS" and not continuous_satellite_spots:
        collapsed = np.nanmean(centers, axis=1)
        return collapsed[:, None, :].repeat(n_frames, axis=1)
    return centers


def verify_frame_axis(centers: np.ndarray, n_frames: int, file_identifier: str) -> None:
    """Raise if the centre array and the frame table disagree on frame count.

    A silent mismatch would mis-associate parallactic angles with frames, which
    produces a plausible-looking but wrong reduction rather than an error.

    Args:
        centers: Shape ``(n_wave, n, 2)``.
        n_frames: Row count of ``frames_info_{file_identifier}.csv``.
        file_identifier: ``"center"`` or ``"coro"``, used in the message.

    Raises:
        ValueError: If ``centers.shape[1] != n_frames``.
    """
    if centers.shape[1] != n_frames:
        raise ValueError(
            f"Frame-axis mismatch: image_centers_fitted_robust has "
            f"{centers.shape[1]} frames but frames_info_{file_identifier}.csv "
            f"has {n_frames} rows. Parallactic angles would be silently "
            "mis-associated with frames."
        )
