"""Star amplitude of the science frames on the scale of the FLUX frames, anchored to the SPARTA WFS flux."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from spherical.conditions.run import WFS_CONFIG_KEYS

WFS_COLUMN = "visloop_flux_subap"
TOP_PERCENTILE = 90
LEVEL_TOLERANCE = 0.03
DROP_THRESHOLD = 0.10
TERRIBLE_WFS_LEVEL = 0.5
TERRIBLE_RATIO = 0.7
DISAGREEMENT = {"IRDIS": 0.15, "IFS": 0.25}
NEAREST_FRAMES = 5
FILL_WARNING = 0.25
FILL_CUTOFF = 0.5
SECONDS_PER_DAY = 86400.0
STAR_AMPLITUDE_FILE = "star_amplitude.fits"
NO_WFS = "star_amplitude_no_wfs"
NO_VETTING = "star_amplitude_no_vetting"
FRAME_MISMATCH = "star_amplitude_frame_mismatch"
NO_USABLE_BLOCK = "star_amplitude_no_usable_block"
MOSTLY_FILLED = "star_amplitude_mostly_filled"
FALLBACK_STATUSES = (NO_WFS, NO_VETTING, FRAME_MISMATCH, NO_USABLE_BLOCK, MOSTLY_FILLED)
LONG_FILL = "star_amplitude_long_fill"
WFS_CONFIG_CHANGE = "star_amplitude_wfs_config_change"

_logger = logging.getLogger(__name__)


def _direct_positive(matched: pd.DataFrame) -> np.ndarray:
    if WFS_COLUMN not in matched or "flag_visloop" not in matched:
        return np.zeros(len(matched), dtype=bool)
    values = pd.to_numeric(matched[WFS_COLUMN], errors="coerce").to_numpy(dtype=float)
    return (matched["flag_visloop"].to_numpy() == "direct") & np.isfinite(values) & (values > 0)


def segment_configurations(config: pd.DataFrame | None) -> dict[int, tuple] | None:
    """Return the WFS configuration of every SPARTA segment ``>= 0`` in ``conditions/sparta_config.csv``.

    Args:
        config: ``conditions/sparta_config.csv`` of the observation, None when absent.

    Returns:
        A mapping from segment to its values of ``WFS_CONFIG_KEYS``, missing values as
        None, so that they compare equal. None when ``config`` is None or lacks the column ``segment`` or one of the
        configuration keys.
    """
    if config is None or not {"segment", *WFS_CONFIG_KEYS} <= set(config.columns):
        return None
    segment = pd.to_numeric(config["segment"], errors="coerce")
    keys = config[list(WFS_CONFIG_KEYS)].astype(object)
    keys = keys.where(keys.notna(), None)
    configuration = {}
    for value, row in zip(segment, keys.itertuples(index=False, name=None)):
        if np.isfinite(value) and value >= 0:
            configuration.setdefault(int(value), row)
    return configuration


def sample_configurations(matched: pd.DataFrame, configuration: dict[int, tuple]) -> list[tuple | None]:
    """Return the WFS configuration of each row's direct positive WFS sample.

    Args:
        matched: Matched condition table.
        configuration: Output of :func:`segment_configurations`.

    Returns:
        One entry per row, None for a row without a direct positive sample or whose segment is not in
        ``configuration``.
    """
    if "segment" not in matched:
        return [None] * len(matched)
    segment = pd.to_numeric(matched["segment"], errors="coerce").to_numpy(dtype=float)
    return [configuration.get(int(value)) if good and np.isfinite(value) else None
            for value, good in zip(segment, _direct_positive(matched))]


def reference_segment(
    matched: pd.DataFrame, config: pd.DataFrame | None = None, flux: pd.DataFrame | None = None
) -> int | None:
    """Return the most common SPARTA segment among the direct positive WFS samples.

    Segments that tie go to the one whose WFS configuration has the most direct positive samples in ``flux``,
    because the FLUX samples set the anchor, and then to the lowest segment.

    Args:
        matched: Matched condition table of the science frames.
        config: ``conditions/sparta_config.csv`` of the observation for the tie-break, None to skip it.
        flux: Matched condition table of the FLUX frames for the tie-break, None to skip it.

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
    tied = values[counts == counts.max()].astype(int)
    configuration = segment_configurations(config)
    if tied.size > 1 and configuration is not None and flux is not None:
        flux_samples = Counter(sample_configurations(flux, configuration))
        n_flux = [flux_samples[configuration[value]] if value in configuration else 0 for value in tied]
        return int(tied[int(np.argmax(n_flux))])
    return int(tied[0])


def configuration_summary(
    config: pd.DataFrame | None, science: pd.DataFrame, flux: pd.DataFrame, flux_block
) -> pd.DataFrame | None:
    """Count the science frames and FLUX blocks with direct positive WFS samples in each WFS configuration.

    Args:
        config: ``conditions/sparta_config.csv`` of the observation, None when absent.
        science: Matched condition table of the science frames.
        flux: Matched condition table of the FLUX frames.
        flux_block: FLUX block label per row of ``flux``, NaN for a row that is not a FLUX frame kept by the
            vetting, so that only blocks entering the PSF are counted.

    Returns:
        One row per configuration that has a sample, in the order of its lowest segment, with the columns of
        ``WFS_CONFIG_KEYS``, ``segments`` (sorted list), ``n_science`` and
        ``n_flux_blocks``. None when :func:`segment_configurations` gives None.
    """
    configuration = segment_configurations(config)
    if configuration is None:
        return None
    science_config = sample_configurations(science, configuration)
    flux_config = sample_configurations(flux, configuration)
    flux_block = np.asarray(flux_block, dtype=float)
    seen = {row for row in science_config + flux_config if row is not None}
    rows = []
    for row in sorted(seen, key=lambda r: min(s for s, c in configuration.items() if c == r)):
        blocks = {flux_block[i] for i, c in enumerate(flux_config) if c == row and np.isfinite(flux_block[i])}
        rows.append({**dict(zip(WFS_CONFIG_KEYS, row)),
                     "segments": sorted(s for s, c in configuration.items() if c == row),
                     "n_science": sum(c == row for c in science_config), "n_flux_blocks": len(blocks)})
    return pd.DataFrame(rows, columns=[*WFS_CONFIG_KEYS, "segments", "n_science", "n_flux_blocks"])


def compatible_segments(config: pd.DataFrame | None, reference: int | None) -> set[int] | None:
    """Return the SPARTA segments with the WFS configuration of the reference segment.

    Segments are compared on ``WFS_CONFIG_KEYS``, which leave out the
    ``nd_filter`` that also cuts segments.

    Args:
        config: ``conditions/sparta_config.csv`` of the observation, None when absent.
        reference: Output of :func:`reference_segment`.

    Returns:
        The set of segments ``>= 0`` whose configuration equals that of ``reference``, empty when
        ``reference`` is None or not in the table. None when ``config`` is None or lacks the column
        ``segment`` or one of the configuration keys, which means every segment.
    """
    configuration = segment_configurations(config)
    if configuration is None:
        return None
    if reference is None or reference not in configuration:
        return set()
    return {value for value, row in configuration.items() if row == configuration[reference]}


def wfs_samples(matched: pd.DataFrame, segments: set[int] | None) -> np.ndarray:
    """Return the WFS flux per row, NaN unless direct, positive and in one of the accepted segments.

    Args:
        matched: Matched condition table.
        segments: Accepted segments from :func:`compatible_segments`, or None to accept every segment.
            A table without a ``segment`` column keeps all its direct positive samples either way.

    Returns:
        Array of length ``len(matched)``.
    """
    good = _direct_positive(matched)
    if segments is not None and "segment" in matched:
        segment = pd.to_numeric(matched["segment"], errors="coerce").to_numpy(dtype=float)
        good &= np.isin(segment, np.array(sorted(segments), dtype=float))
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


def _per_channel(table: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(labels, values)`` with values of shape ``(n_channel, n_block)`` in sorted label order."""
    labels = np.sort(table["block"].unique())
    channels = np.sort(table["channel"].unique())
    pivot = table.pivot(index="channel", columns="block", values=column).reindex(index=channels, columns=labels)
    return labels, pivot.to_numpy(dtype=float)


def usable_blocks(table: pd.DataFrame, excluded: Sequence[int] = ()) -> np.ndarray:
    """Return which FLUX blocks may anchor the amplitude, leaving out only gross failures.

    In each channel, over the blocks with data in it, a block fails when its WFS level is under
    ``TERRIBLE_WFS_LEVEL`` of the highest, or when its FLUX over WFS is under ``TERRIBLE_RATIO`` of the highest
    among the blocks that pass the level test. A block is usable when it has data in at least one channel,
    fails in none of the channels where it has data, and is not in ``excluded``. Excluded blocks do not
    enter the highest values, so a block left out by hand because it is anomalous cannot fail good ones.

    Args:
        table: Output of :func:`block_table`.
        excluded: Block labels that are not usable.

    Returns:
        Boolean array per block label in sorted order.
    """
    labels, ratio = _per_channel(table, "ratio")
    _, wfs = _per_channel(table, "wfs")
    has_data = np.isfinite(ratio)
    included = ~np.isin(labels, np.asarray(excluded))
    passes = np.ones(labels.shape, dtype=bool)
    for channel in range(ratio.shape[0]):
        data = has_data[channel] & included
        if not data.any():
            continue
        level_ok = data & (wfs[channel] >= TERRIBLE_WFS_LEVEL * np.max(wfs[channel][data]))
        ratio_ok = level_ok.copy()
        if level_ok.any():
            ratio_ok = level_ok & (ratio[channel] >= TERRIBLE_RATIO * np.max(ratio[channel][level_ok]))
        passes &= ~data | ratio_ok
    return has_data.any(axis=0) & passes & included


def block_disagreement(table: pd.DataFrame, usable: np.ndarray, column: str = "ratio") -> float:
    """Return the median over channels of ``max/min - 1`` of a column over the usable blocks with data.

    Args:
        table: Output of :func:`block_table`.
        usable: Output of :func:`usable_blocks`.
        column: Column to compare, ``ratio`` or ``flux``.

    Returns:
        The disagreement, NaN when no channel has two usable blocks with data.
    """
    _, values = _per_channel(table, column)
    spreads = []
    for row in values:
        row = row[np.asarray(usable, dtype=bool) & np.isfinite(row)]
        if row.size >= 2:
            spreads.append(row.max() / row.min() - 1)
    return float(np.median(spreads)) if spreads else np.nan


def anchor(table: pd.DataFrame, usable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-channel anchor ``(scale, level)`` over the usable blocks with data in the channel.

    Args:
        table: Output of :func:`block_table`.
        usable: Output of :func:`usable_blocks`.

    Returns:
        ``scale`` as the median of ``ratio`` and ``level`` as the mean of ``flux``, NaN for a channel
        without a usable block that has data.
    """
    _, ratio = _per_channel(table, "ratio")
    _, flux = _per_channel(table, "flux")
    use = np.asarray(usable, dtype=bool)[None, :] & np.isfinite(ratio) & np.isfinite(flux)
    scale = np.full(ratio.shape[0], np.nan)
    level = np.full(ratio.shape[0], np.nan)
    for channel in range(ratio.shape[0]):
        if use[channel].any():
            scale[channel] = np.median(ratio[channel, use[channel]])
            level[channel] = np.mean(flux[channel, use[channel]])
    return scale, level


def fill_in_time(values, times) -> tuple[np.ndarray, int, float]:
    """Fill non-finite values by linear interpolation in time, with the nearest value beyond the ends.

    Args:
        values: Samples per frame.
        times: Time of each frame in MJD, not necessarily sorted.

    Returns:
        ``(filled, n_filled, longest_gap_s)``. The gap is measured between the valid samples on either side
        of it, or from the outermost valid sample to the end frame at the edges. An all-NaN input stays NaN.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = np.isfinite(values)
    if valid.all() or not valid.any():
        return values.copy(), 0, 0.0
    order = np.argsort(times, kind="stable")
    t, v, ok = times[order], values[order], valid[order]
    filled_sorted = np.interp(t, t[ok], v[ok])
    gap_free = np.diff(np.flatnonzero(ok)) == 1
    spans = np.diff(t[ok])
    longest_inner = float(np.max(spans[~gap_free])) if (~gap_free).any() else 0.0
    edge_spans = []
    if not ok[0]:
        edge_spans.append(t[ok][0] - t[0])
    if not ok[-1]:
        edge_spans.append(t[-1] - t[ok][-1])
    longest = max([longest_inner] + edge_spans) * SECONDS_PER_DAY
    filled = np.empty_like(filled_sorted)
    filled[order] = filled_sorted
    return filled, int((~valid).sum()), longest


def below_floor(measured_wfs, floor: float) -> np.ndarray:
    """Return which frames have a measured WFS sample under ``floor`` times the 90th percentile of the measured samples.

    Args:
        measured_wfs: WFS flux of the science frames, NaN where there is no measurement.
        floor: Fraction of the 90th percentile.

    Returns:
        Boolean array, False for frames without a measurement.
    """
    measured = np.asarray(measured_wfs, dtype=float)
    finite = measured[np.isfinite(measured)]
    if not finite.size:
        return np.zeros(measured.shape, dtype=bool)
    return np.nan_to_num(measured, nan=np.inf) < floor * np.percentile(finite, TOP_PERCENTILE)


def wfs_amplitude(scale, level, filled_wfs, measured_wfs, floor: float) -> np.ndarray:
    """Return the amplitude ``scale * wfs / level`` per channel and science frame.

    Args:
        scale: Anchor scale per channel.
        level: Anchor level per channel.
        filled_wfs: WFS flux of the science frames with the gaps filled by :func:`fill_in_time`.
        measured_wfs: The same samples before the gaps were filled, NaN where there is no measurement.
        floor: Measured frames below this fraction of the 90th percentile of ``measured_wfs`` are NaN.
            Filled frames are never floored.

    Returns:
        Array of shape ``(n_wave, n_science)``, NaN rows for channels with a NaN anchor.
    """
    filled_wfs = np.asarray(filled_wfs, dtype=float)
    amplitude = np.asarray(scale, dtype=float)[:, None] * filled_wfs[None, :] / np.asarray(level, dtype=float)[:, None]
    amplitude[:, below_floor(measured_wfs, floor)] = np.nan
    return amplitude


def spot_anchored_amplitude(spot_variation, wfs_amp, science_times, usable_block_times) -> tuple[np.ndarray, np.ndarray]:
    """Scale the spot trace to the WFS amplitude next to the usable blocks.

    Args:
        spot_variation: Spot trace, shape ``(n_wave, n_science)``.
        wfs_amp: Output of :func:`wfs_amplitude`, same shape.
        science_times: Time of each science frame, on the axis of ``usable_block_times``.
        usable_block_times: Time of each usable block.

    Returns:
        ``(amplitude, levels)``. ``levels`` has shape ``(n_wave, n_blocks)`` and holds the per-block median of
        ``wfs_amp / spot_variation`` over the ``NEAREST_FRAMES`` nearest frames with both finite. The amplitude
        is the spot trace times the median level over blocks, NaN where ``wfs_amp`` is NaN.
    """
    spot_variation = np.asarray(spot_variation, dtype=float)
    wfs_amp = np.asarray(wfs_amp, dtype=float)
    science_times = np.asarray(science_times, dtype=float)
    block_times = np.asarray(usable_block_times, dtype=float)
    levels = np.full((spot_variation.shape[0], block_times.size), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = wfs_amp / spot_variation
    for channel in range(spot_variation.shape[0]):
        valid = np.isfinite(ratio[channel])
        if not valid.any():
            continue
        for index, time in enumerate(block_times):
            nearest = np.argsort(np.abs(science_times[valid] - time), kind="stable")[:NEAREST_FRAMES]
            levels[channel, index] = np.median(ratio[channel, valid][nearest])
    finite = np.isfinite(levels)
    level = np.array([np.median(row[ok]) if ok.any() else np.nan for row, ok in zip(levels, finite)])
    amplitude = spot_variation * level[:, None]
    amplitude[~np.isfinite(wfs_amp)] = np.nan
    return amplitude, levels


def load_star_amplitude(
    data_directory, n_frames: int, n_psf_blocks: int, logger=None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read the star amplitude and the PSF block mask written by ``calibrate_star_amplitude``.

    Args:
        data_directory: The ``converted/`` folder holding ``star_amplitude.fits``.
        n_frames: Number of science frames, the rows of ``frames_info_<science>.csv``.
        n_psf_blocks: Number of blocks on axis 1 of the PSF cube TRAP receives.
        logger: Logger for the warnings and the mode line, the module logger when None.

    Returns:
        ``(amplitude, block_mask)``: the amplitude of shape ``(n_wave, n_frames)`` (NaN marks a bad frame) and a
        boolean mask over the PSF blocks, None when the file holds no mask, which means every block. A channel that
        is NaN in every frame gets amplitude 1, with a warning. The amplitude is None when the step fell back (its
        ``REASON`` is one of ``FALLBACK_STATUSES``), because the amplitude 1 of a fallback is no calibration;
        the mask still applies. ``(None, None)`` when the file is absent, or, with a warning, when its frame axis
        or block axis does not match, its mask selects no block or no channel has a finite amplitude.
    """
    logger = _logger if logger is None else logger
    path = Path(data_directory) / STAR_AMPLITUDE_FILE
    if not path.exists():
        return None, None
    advice = 'Re-run the step with force={"calibrate_star_amplitude"}.'
    with fits.open(path) as hdul:
        amplitude = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
        mode, reason = header.get("MODE"), header.get("REASON")
        if "BLOCKS" not in hdul:
            logger.warning(
                f"{STAR_AMPLITUDE_FILE} has no BLOCKS extension; the file is ignored. {advice}",
                extra={"step": "trap_reduction", "status": "warning"},
            )
            return None, None
        # An empty image HDU reads back as None, not as an array of length 0.
        blocks = hdul["BLOCKS"].data
        blocks = np.zeros(0, dtype=bool) if blocks is None else np.array(blocks, dtype=bool).ravel()
    if amplitude.ndim != 2 or amplitude.shape[1] != n_frames:
        logger.warning(
            f"{STAR_AMPLITUDE_FILE} has shape {amplitude.shape}, not {n_frames} science frames; it is ignored. {advice}",
            extra={"step": "trap_reduction", "status": "warning"},
        )
        return None, None
    if blocks.size == 0:
        block_mask = None
    elif blocks.size != n_psf_blocks or not blocks.any():
        problem = (
            f"selects none of the {n_psf_blocks} PSF blocks"
            if blocks.size == n_psf_blocks
            else f"has {blocks.size} entries for {n_psf_blocks} PSF blocks"
        )
        logger.warning(
            f"The block mask of {STAR_AMPLITUDE_FILE} {problem}; the file is ignored. {advice}",
            extra={"step": "trap_reduction", "status": "warning"},
        )
        return None, None
    else:
        block_mask = blocks
    if reason in FALLBACK_STATUSES:
        logger.info(
            f"{STAR_AMPLITUDE_FILE} is a fallback ({reason}) and holds no star amplitude; only its block mask is used.",
            extra={"step": "trap_reduction", "status": "star_amplitude_fallback"},
        )
        return None, block_mask
    empty = np.flatnonzero(~np.isfinite(amplitude).any(axis=1))
    if empty.size == amplitude.shape[0]:
        logger.warning(
            f"{STAR_AMPLITUDE_FILE} has no finite star amplitude in any channel; the file is ignored. {advice}",
            extra={"step": "trap_reduction", "status": "warning"},
        )
        return None, None
    if empty.size:
        # The PSF of such a channel is NaN as well, so TRAP already skips it; left NaN, the channel would make
        # drop_failed_trace_frames reject every frame.
        amplitude[empty] = 1.0
        logger.warning(
            f"Channels {empty.tolist()} have no usable FLUX block and no star amplitude; they get amplitude 1 and "
            "TRAP skips them through their NaN PSF.",
            extra={"step": "trap_reduction", "status": "warning"},
        )
    logger.info(
        f"Star amplitude from {STAR_AMPLITUDE_FILE}: mode {mode}, reason {reason}.",
        extra={"step": "trap_reduction", "status": "star_amplitude"},
    )
    return amplitude, block_mask


def psf_block_mean(psf_cube: np.ndarray, block_mask: np.ndarray | None) -> np.ndarray:
    """Average the PSF cube over its blocks, ignoring NaN.

    Args:
        psf_cube: PSF cube of shape ``(n_wave, n_blocks, ny, nx)``.
        block_mask: Boolean mask over the blocks, or None to use every block.

    Returns:
        The mean PSF, shape ``(n_wave, ny, nx)``.
    """
    psf_cube = np.asarray(psf_cube)
    if block_mask is not None:
        psf_cube = psf_cube[:, np.asarray(block_mask, dtype=bool)]
    return np.nanmean(psf_cube, axis=1)
