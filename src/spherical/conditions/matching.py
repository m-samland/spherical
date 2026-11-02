"""Match condition samples to frame exposure windows and find holes in a sample series.

Only numpy and pandas are used, so the module imports in a base install.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400.0


def _segments(samples: pd.DataFrame) -> np.ndarray:
    if "segment" in samples:
        return samples["segment"].to_numpy(dtype=int)
    return np.zeros(len(samples), dtype=int)


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    valid = ~np.isnan(values) & (weights > 0)
    if not valid.any():
        return np.nan, np.nan
    values, weights = values[valid], weights[valid]
    mean = np.average(values, weights=weights)
    return float(mean), float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def match_to_frames(
    samples: pd.DataFrame,
    frames: pd.DataFrame,
    value_columns: Sequence[str],
    *,
    max_interp_gap_s: float,
) -> pd.DataFrame:
    """Assign sample values to each frame, by overlap or by interpolation across a short hole.

    A frame that overlaps samples gets the overlap-weighted mean of the samples in the segment with
    the largest overlap (flag ``direct``). A frame without overlap, for example one shorter than the
    sample cadence, is interpolated linearly at its midpoint between the nearest sample midpoints
    before and after it (flag ``interpolated``). That needs both neighbours in the same segment and
    within ``max_interp_gap_s`` of the frame midpoint. Otherwise the values are NaN (flag ``gap``).
    Each column ignores its NaN samples, so a column without a valid value is NaN on its own while the others
    keep theirs. A frame whose overlapping samples are NaN in every column is treated like a frame without
    overlap and interpolated if it can be, otherwise flagged ``gap``. A segment is never mixed with another, because a WFS configuration change makes values on either
    side incomparable.

    Args:
        samples: Table with ``mjd_start``, ``mjd_end``, the value columns and optionally an integer
            ``segment`` column (all samples belong to segment 0 when absent).
        frames: Table with ``MJD START`` and ``MJD END``, as in ``frames_info_*.csv``.
        value_columns: Columns of ``samples`` to transfer.
        max_interp_gap_s: Largest distance in seconds between the frame midpoint and each of the two
            neighbouring sample midpoints for interpolation.

    Returns:
        One row per frame in the order of ``frames``, with the value columns, ``<col>_std``
        (overlap-weighted standard deviation, NaN unless the frame is ``direct``), ``n_samples``
        (overlapping samples used), ``segment`` (-1 when none) and ``flag``.
    """
    value_columns = list(value_columns)
    n_frames = len(frames)
    out = {col: np.full(n_frames, np.nan) for col in value_columns}
    out.update({f"{col}_std": np.full(n_frames, np.nan) for col in value_columns})
    n_used = np.zeros(n_frames, dtype=int)
    segment_out = np.full(n_frames, -1, dtype=int)
    flag = np.full(n_frames, "gap", dtype=object)

    order = np.argsort(samples["mjd_start"].to_numpy(dtype=float), kind="stable") if len(samples) else np.empty(0, int)
    start = samples["mjd_start"].to_numpy(dtype=float)[order]
    end = samples["mjd_end"].to_numpy(dtype=float)[order]
    segment = _segments(samples)[order]
    values = {col: samples[col].to_numpy(dtype=float)[order] for col in value_columns}
    mid = 0.5 * (start + end)
    max_gap = max_interp_gap_s / SECONDS_PER_DAY

    frame_start = frames["MJD START"].to_numpy(dtype=float)
    frame_end = frames["MJD END"].to_numpy(dtype=float)

    for i in range(n_frames):
        overlap = np.clip(np.minimum(end, frame_end[i]) - np.maximum(start, frame_start[i]), 0.0, None)
        hit = overlap > 0
        if hit.any():
            per_segment = {seg: overlap[hit & (segment == seg)].sum() for seg in np.unique(segment[hit])}
            best = max(per_segment, key=per_segment.get)
            chosen = hit & (segment == best)
            for col in value_columns:
                out[col][i], out[f"{col}_std"][i] = _weighted_mean_std(values[col][chosen], overlap[chosen])
            if not all(np.isnan(out[col][i]) for col in value_columns):
                n_used[i] = int(chosen.sum())
                segment_out[i] = best
                flag[i] = "direct"
                continue

        if len(start) == 0:
            continue
        frame_mid = 0.5 * (frame_start[i] + frame_end[i])
        for col in value_columns:
            valid = np.flatnonzero(~np.isnan(values[col]))
            position = int(np.searchsorted(mid[valid], frame_mid, side="right"))
            if position == 0 or position >= len(valid):
                continue
            before, after = valid[position - 1], valid[position]
            if segment[before] != segment[after]:
                continue
            if frame_mid - mid[before] > max_gap or mid[after] - frame_mid > max_gap:
                continue
            weight = (frame_mid - mid[before]) / (mid[after] - mid[before])
            out[col][i] = values[col][before] + weight * (values[col][after] - values[col][before])
            if flag[i] != "interpolated":
                segment_out[i] = segment[before]
                flag[i] = "interpolated"

    result = pd.DataFrame(out, index=frames.index)
    result["n_samples"] = n_used
    result["segment"] = segment_out
    result["flag"] = flag
    return result


def find_gaps(samples: pd.DataFrame, window: tuple[float, float], *, factor: float = 3.0) -> list[tuple[float, float]]:
    """Return the MJD intervals inside ``window`` that no sample interval covers.

    Holes shorter than ``factor`` times the median sample interval are ignored, so ordinary jitter
    between consecutive samples is not reported. Without samples the whole window is one gap.

    Args:
        samples: Table with ``mjd_start`` and ``mjd_end``.
        window: ``(start, end)`` in MJD.
        factor: Minimum hole length in units of the median sample interval.

    Returns:
        Sorted list of ``(start, end)`` MJD intervals.
    """
    lo, hi = float(window[0]), float(window[1])
    if len(samples) == 0:
        return [(lo, hi)] if hi > lo else []

    start = samples["mjd_start"].to_numpy(dtype=float)
    end = samples["mjd_end"].to_numpy(dtype=float)
    min_hole = factor * float(np.median(end - start))

    gaps: list[tuple[float, float]] = []
    covered_until = lo
    for s, e in sorted(zip(start, end, strict=True)):
        if e <= covered_until:
            continue
        if s > covered_until:
            hole_end = min(s, hi)
            if hole_end - covered_until >= min_hole:
                gaps.append((covered_until, hole_end))
        covered_until = e
        if covered_until >= hi:
            break
    if hi - covered_until >= min_hole and covered_until < hi:
        gaps.append((covered_until, hi))
    return gaps
