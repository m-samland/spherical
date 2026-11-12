"""Star amplitude of the science frames on the scale of the FLUX frames, anchored to the SPARTA WFS flux."""

from __future__ import annotations

import numpy as np
import pandas as pd

WFS_COLUMN = "visloop_flux_subap"
TOP_PERCENTILE = 90
LEVEL_TOLERANCE = 0.03
DROP_THRESHOLD = 0.10


def _direct_positive(matched: pd.DataFrame) -> np.ndarray:
    if WFS_COLUMN not in matched or "flag_visloop" not in matched:
        return np.zeros(len(matched), dtype=bool)
    values = pd.to_numeric(matched[WFS_COLUMN], errors="coerce").to_numpy(dtype=float)
    return (matched["flag_visloop"].to_numpy() == "direct") & np.isfinite(values) & (values > 0)


def reference_segment(matched: pd.DataFrame) -> int | None:
    """Return the most common SPARTA segment among the direct positive WFS samples.

    Args:
        matched: Matched condition table of the science frames.

    Returns:
        The most common ``segment >= 0``, or None without such samples or without a ``segment`` column.
    """
    if "segment" not in matched:
        return None
    segment = pd.to_numeric(matched["segment"], errors="coerce").to_numpy(dtype=float)
    selected = segment[_direct_positive(matched) & (segment >= 0)]
    if selected.size == 0:
        return None
    values, counts = np.unique(selected, return_counts=True)
    return int(values[np.argmax(counts)])


def wfs_samples(matched: pd.DataFrame, segment: int | None) -> np.ndarray:
    """Return the WFS flux per row, NaN unless direct, positive and in the reference segment.

    Args:
        matched: Matched condition table.
        segment: Reference segment, or None to accept every segment.

    Returns:
        Array of length ``len(matched)``.
    """
    good = _direct_positive(matched)
    if segment is not None and "segment" in matched:
        good &= pd.to_numeric(matched["segment"], errors="coerce").to_numpy(dtype=float) == segment
    elif segment is not None:
        good &= False
    out = np.full(len(matched), np.nan)
    if WFS_COLUMN in matched:
        values = pd.to_numeric(matched[WFS_COLUMN], errors="coerce").to_numpy(dtype=float)
        out[good] = values[good]
    return out


def block_table(flux, kept, wfs, block, sharpness) -> pd.DataFrame:
    """Summarise the FLUX frames per channel and block.

    Args:
        flux: Aperture flux, shape ``(n_wave, n_flux_frames)``.
        kept: Frames kept by the vetting, same shape.
        wfs: WFS flux per FLUX frame, shape ``(n_flux_frames,)``, NaN where missing.
        block: Block label per FLUX frame, shape ``(n_flux_frames,)``.
        sharpness: Sharpness of the FLUX frames, shape ``(n_wave, n_flux_frames)``.

    Returns:
        One row per channel and block (sorted by channel, then block) with columns
        ``channel, block, n_kept, flux, wfs, ratio, sharpness``. Medians run over the kept frames, ``wfs``
        over those that have a WFS sample. Undefined values are NaN.
    """
    flux = np.asarray(flux, dtype=float)
    kept = np.asarray(kept, dtype=bool)
    sharpness = np.asarray(sharpness, dtype=float)
    wfs = np.asarray(wfs, dtype=float)
    block = np.asarray(block)
    rows = []
    for channel in range(flux.shape[0]):
        for label in np.unique(block):
            use = kept[channel] & (block == label)
            n_kept = int(use.sum())
            with_wfs = use & np.isfinite(wfs)
            flux_med = float(np.median(flux[channel, use])) if n_kept else np.nan
            wfs_med = float(np.median(wfs[with_wfs])) if with_wfs.any() else np.nan
            sharp_med = float(np.median(sharpness[channel, use])) if n_kept else np.nan
            rows.append((channel, label, n_kept, flux_med, wfs_med, flux_med / wfs_med, sharp_med))
    return pd.DataFrame(rows, columns=["channel", "block", "n_kept", "flux", "wfs", "ratio", "sharpness"])


def block_wfs_level(table: pd.DataFrame) -> np.ndarray:
    """Return the WFS level per block in label order, the median over channels of ``wfs``."""
    return np.array([np.nan if group.isna().all() else np.nanmedian(group) for _, group in table.groupby("block")["wfs"]])


def classify(block_wfs, science_wfs) -> tuple[str, float, float]:
    """Classify a sequence as stable or variable from the WFS flux.

    Args:
        block_wfs: WFS level per FLUX block.
        science_wfs: WFS samples of the science frames.

    Returns:
        ``(label, level_offset, drop_depth)``. The label is ``"stable"``, ``"variable"`` or ``"no_wfs"``.
        ``level_offset`` is the largest relative deviation of a block from the median science WFS and
        ``drop_depth`` the relative depth of the lowest sample below the 90th percentile of all samples.
        Both are NaN for ``"no_wfs"``.
    """
    blocks = np.asarray(block_wfs, dtype=float)
    blocks = blocks[np.isfinite(blocks)]
    science = np.asarray(science_wfs, dtype=float)
    science = science[np.isfinite(science)]
    if blocks.size == 0 or science.size == 0:
        return "no_wfs", np.nan, np.nan
    level_offset = float(np.max(np.abs(blocks / np.median(science) - 1)))
    pooled = np.concatenate([science, blocks])
    drop_depth = float(1 - pooled.min() / np.percentile(pooled, TOP_PERCENTILE))
    stable = level_offset <= LEVEL_TOLERANCE and drop_depth <= DROP_THRESHOLD
    return ("stable" if stable else "variable"), level_offset, drop_depth
