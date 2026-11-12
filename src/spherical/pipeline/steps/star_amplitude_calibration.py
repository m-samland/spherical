"""Pipeline step that puts the star amplitude of every science frame on the flux scale of the FLUX frames.

The step reads ``additional_outputs/flux_frame_vetting.csv``, the frame tables and the matched condition tables
of the ``observing_conditions`` step, and in the spot mode of continuous-waffle sequences
``spot_amplitude_variation.fits``. It writes ``star_amplitude.fits`` and
``additional_outputs/star_amplitude_blocks.csv``. The calculations are in :mod:`spherical.pipeline.star_amplitude`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from spherical.pipeline import star_amplitude as sa
from spherical.pipeline.logging_utils import optional_logger
from spherical.pipeline.pipeline_config import PreprocConfig
from spherical.pipeline.science_frames import science_frame_type

STEP_NAME = "star_amplitude_calibration"
OUTPUT_FILE = "star_amplitude.fits"
BLOCK_TABLE_FILE = "star_amplitude_blocks.csv"
VETTING_FILE = "flux_frame_vetting.csv"
CONDITIONS_DIR = "conditions"


def _read_frames(converted_dir: Path, name: str) -> pd.DataFrame | None:
    path = converted_dir / f"frames_info_{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def _read_matched(converted_dir: Path, name: str) -> pd.DataFrame | None:
    path = converted_dir / CONDITIONS_DIR / f"matched_{name}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _card(value):
    """FITS headers cannot hold NaN, so an undefined number becomes an undefined card."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _n_wave(converted_dir: Path, vetting: pd.DataFrame | None) -> int | None:
    if vetting is not None:
        return int(vetting["channel"].nunique())
    path = converted_dir / "wavelengths.fits"
    if path.exists():
        return int(np.asarray(fits.getdata(path)).size)
    return None


def _vetting_frames(vetting: pd.DataFrame, frames_flux: pd.DataFrame | None) -> pd.DataFrame:
    """One row per vetted FLUX frame with its block, whether it is kept and its row in ``frames_info_flux.csv``.

    The join goes through ``(origfile, DIT INDEX)``, because the vetting ``frame`` column counts the frames
    after cube selection, not the rows of ``frames_info_flux.csv``.
    """
    frames = (
        vetting.groupby("frame", sort=True)
        .agg(origfile=("origfile", "first"), dit_index=("DIT INDEX", "first"), block=("block", "first"),
             kept=("decision", lambda d: bool((d != "frame_rejected").any())))
        .reset_index()
    )
    row = np.full(len(frames), -1)
    mjd = np.full(len(frames), np.nan)
    if frames_flux is not None and {"ORIGFILE", "DIT INDEX"} <= set(frames_flux.columns):
        lookup = {(str(o), int(d)): i for i, (o, d) in enumerate(zip(frames_flux["ORIGFILE"], frames_flux["DIT INDEX"]))}
        row = np.array([lookup.get((str(o), int(d)), -1) for o, d in zip(frames["origfile"], frames["dit_index"])])
        if "MJD START" in frames_flux:
            starts = frames_flux["MJD START"].to_numpy(dtype=float)
            mjd = np.where(row >= 0, starts[np.clip(row, 0, None)], np.nan)
    frames["flux_row"] = row
    frames["mjd_start"] = mjd
    return frames


def _channel_frame(vetting: pd.DataFrame, column: str) -> np.ndarray:
    return vetting.pivot(index="channel", columns="frame", values=column).sort_index().sort_index(axis=1).to_numpy()


def _write(converted_dir: Path, amplitude, header: dict, blocks, scale, level, spot_level_scatter=None) -> None:
    primary = fits.PrimaryHDU(np.asarray(amplitude, dtype=np.float32))
    for key, value in header.items():
        primary.header[key] = _card(value)
    columns = [fits.Column(name="scale", format="D", array=np.asarray(scale, dtype=float)),
               fits.Column(name="level", format="D", array=np.asarray(level, dtype=float))]
    if spot_level_scatter is not None:
        columns.append(fits.Column(name="spot_level_scatter", format="D", array=np.asarray(spot_level_scatter, dtype=float)))
    hdus = fits.HDUList([
        primary,
        fits.ImageHDU(np.asarray(blocks, dtype=np.uint8), name="BLOCKS"),
        fits.BinTableHDU.from_columns(columns, name="CHANNELS"),
    ])
    hdus.writeto(converted_dir / OUTPUT_FILE, overwrite=True)


def _write_block_table(converted_dir: Path, table: pd.DataFrame, psf_labels, usable_labels, excluded) -> None:
    table = table.copy()
    position = {label: index for index, label in enumerate(psf_labels)}
    table["psf_block"] = [position.get(label, -1) for label in table["block"]]
    table["usable"] = table["block"].isin(list(usable_labels))
    table["excluded_by_hand"] = table["block"].isin(list(excluded))
    additional = converted_dir / "additional_outputs"
    additional.mkdir(parents=True, exist_ok=True)
    table.to_csv(additional / BLOCK_TABLE_FILE, index=False)


@optional_logger
def run_star_amplitude_calibration(
    converted_dir,
    reduction_parameters: dict,
    continuous_satellite_spots: bool,
    instrument: str,
    observation_key: str,
    logger=None,
) -> str:
    """Give every science frame a star amplitude on the flux scale of the vetted FLUX blocks.

    A sequence whose WFS flux is stable gets amplitude 1 in every frame. A variable one follows the SPARTA
    WFS flux, anchored per channel to the FLUX blocks that are not gross failures, so the amplitude is
    relative to the mean PSF of those blocks. Continuous-waffle sequences with
    ``star_amplitude_waffle_mode="spots"`` take the shape of the spot trace and its level from the WFS
    amplitude next to the usable blocks. Missing or inconsistent inputs give amplitude 1 and a warning.

    The step reads the matched condition tables, so after refreshing the conditions force
    ``calibrate_star_amplitude``, and re-run TRAP after it.

    Args:
        converted_dir: The observation's ``converted/`` directory.
        reduction_parameters: ``asdict(config.preprocessing)``.
        continuous_satellite_spots: The observation's ``WAFFLE_MODE`` flag.
        instrument: ``"IFS"`` or ``"IRDIS"``, for the disagreement threshold.
        observation_key: ``target_folder_string`` of the observation, the key of
            ``star_amplitude_exclude_blocks``.
        logger: Pipeline logger.

    Returns:
        The mode written, ``"constant"``, ``"wfs"`` or ``"spots"``.

    Outputs:
        ``star_amplitude.fits`` in ``converted_dir``: the amplitude, shape ``(n_wave, n_science_frames)`` on
        the axis of ``frames_info_<science>.csv`` (NaN is a bad frame), with the diagnostics in the header;
        the extension ``BLOCKS``, a mask on the block axis of ``psf_cube_for_postprocessing.fits`` (empty
        without a vetting table); and ``CHANNELS``, the per-channel anchor. Also
        ``additional_outputs/star_amplitude_blocks.csv``, the per-block table.
    """
    converted_dir = Path(converted_dir)
    defaults = PreprocConfig()
    floor = float(reduction_parameters.get("star_amplitude_wfs_floor", defaults.star_amplitude_wfs_floor))
    waffle_mode = reduction_parameters.get("star_amplitude_waffle_mode", defaults.star_amplitude_waffle_mode)
    excluded = list((reduction_parameters.get("star_amplitude_exclude_blocks") or {}).get(observation_key, []))
    vetting_on = reduction_parameters.get("flux_frame_vetting", defaults.flux_frame_vetting)
    step = {"step": STEP_NAME}

    logger.info("Starting star amplitude calibration.", extra={**step, "status": "started"})
    # A stale amplitude from an earlier run must not reach TRAP when this run stops early.
    (converted_dir / OUTPUT_FILE).unlink(missing_ok=True)
    try:
        mode = _calibrate(converted_dir, floor, waffle_mode, excluded, vetting_on, continuous_satellite_spots,
                          instrument, logger)
    except Exception as exc:
        logger.error(f"Star amplitude calibration failed: {exc}", exc_info=True, extra={**step, "status": "failed"})
        return "constant"
    if mode is None:
        return "constant"
    logger.info("Finished star amplitude calibration.", extra={**step, "status": "success"})
    return mode


def _calibrate(converted_dir, floor, waffle_mode, excluded, vetting_on, continuous_satellite_spots, instrument, logger):
    science = science_frame_type(continuous_satellite_spots)
    frames_science = _read_frames(converted_dir, science)
    if frames_science is None:
        logger.error(f"No frames_info_{science}.csv; run compute_frames_info first.",
                     extra={"step": STEP_NAME, "status": "failed"})
        return None
    n_science = len(frames_science)
    header = {
        "MODE": "constant", "REASON": None, "LEVOFF": None, "DROPDEP": None, "DISAGREE": None, "SCALEUNC": None,
        "NFLOOR": 0, "NFILLED": 0, "MAXGAP_S": 0.0, "SEGMENT": None, "WFSFLOOR": floor, "WAFMODE": waffle_mode,
    }

    vetting_path = converted_dir / "additional_outputs" / VETTING_FILE
    vetting = pd.read_csv(vetting_path) if vetting_on and vetting_path.exists() else None
    n_wave = _n_wave(converted_dir, vetting)
    if n_wave is None:
        logger.error("Neither flux_frame_vetting.csv nor wavelengths.fits gives the number of channels.",
                     extra={"step": STEP_NAME, "status": "failed"})
        return None
    nan_channels = np.full(n_wave, np.nan)

    if vetting is None:
        why = "flux_frame_vetting is off" if not vetting_on else f"{VETTING_FILE} is missing"
        logger.warning(f"Star amplitude 1 for every frame, because {why}; TRAP uses every PSF block.",
                       extra={"step": STEP_NAME, "status": sa.NO_VETTING})
        header["REASON"] = sa.NO_VETTING
        _write(converted_dir, np.ones((n_wave, n_science)), header, np.zeros(0), nan_channels, nan_channels)
        return "constant"

    frames_flux = _read_frames(converted_dir, "flux")
    frames = _vetting_frames(vetting, frames_flux)
    labels = np.sort(frames["block"].unique())
    psf_labels = np.sort(frames.loc[frames["kept"], "block"].unique())
    kept_not_excluded = psf_labels[~np.isin(psf_labels, excluded)]
    flux = _channel_frame(vetting, "flux_r3")
    kept = _channel_frame(vetting, "decision") == "kept"
    sharpness = _channel_frame(vetting, "sharpness")
    block = frames["block"].to_numpy()

    def fallback(status, message, wfs=None):
        table = sa.block_table(flux, kept, np.full(len(frames), np.nan) if wfs is None else wfs, block, sharpness)
        usable = np.isin(labels, kept_not_excluded)
        disagreement = sa.block_disagreement(table, usable, "flux")
        _warn_disagreement(disagreement, instrument, "flux", logger)
        header.update(REASON=status, DISAGREE=disagreement, SCALEUNC=disagreement / 2)
        logger.warning(f"Star amplitude 1 for every frame: {message}", extra={"step": STEP_NAME, "status": status})
        _write(converted_dir, np.ones((n_wave, n_science)), header, np.isin(psf_labels, kept_not_excluded),
               nan_channels, nan_channels)
        _write_block_table(converted_dir, table, psf_labels, kept_not_excluded, excluded)
        return "constant"

    matched_science = _read_matched(converted_dir, science)
    matched_flux = _read_matched(converted_dir, "flux")
    if matched_science is None or matched_flux is None:
        return fallback(sa.NO_WFS, "no matched condition tables; run observing_conditions first.")
    if len(matched_science) != n_science or frames_flux is None or len(matched_flux) != len(frames_flux):
        return fallback(sa.FRAME_MISMATCH,
                        "the matched condition tables do not have the rows of the frame tables; "
                        "force observing_conditions.")

    segment = sa.reference_segment(matched_science)
    header["SEGMENT"] = segment
    science_wfs = sa.wfs_samples(matched_science, segment)
    flux_wfs_rows = sa.wfs_samples(matched_flux, segment)
    row = frames["flux_row"].to_numpy()
    flux_wfs = np.where(row >= 0, flux_wfs_rows[np.clip(row, 0, None)], np.nan)
    table = sa.block_table(flux, kept, flux_wfs, block, sharpness)
    label, level_offset, drop_depth = sa.classify(sa.block_wfs_level(table), science_wfs)
    if label == "no_wfs":
        return fallback(sa.NO_WFS, "no direct WFS flux in the reference segment during the FLUX "
                        "or science frames.", wfs=flux_wfs)
    header.update(LEVOFF=level_offset, DROPDEP=drop_depth)

    usable = sa.usable_blocks(table, excluded=excluded)
    left_out = [int(b) for b, ok in zip(labels, usable) if not ok and b in psf_labels and b not in excluded]
    if left_out:
        logger.warning(f"FLUX blocks {left_out} fail the terrible-block test and are left out of the PSF.",
                       extra={"step": STEP_NAME, "status": "star_amplitude_block_left_out"})
    if not usable.any():
        return fallback(sa.NO_USABLE_BLOCK, "no FLUX block is usable.", wfs=flux_wfs)
    usable_labels = labels[usable]
    disagreement = sa.block_disagreement(table, usable, "ratio")
    _warn_disagreement(disagreement, instrument, "ratio", logger)
    scale, level = sa.anchor(table, usable)
    header.update(REASON=label, DISAGREE=disagreement, SCALEUNC=disagreement / 2)
    spot_level_scatter = None

    if label == "stable":
        mode = "constant"
        amplitude = np.ones((n_wave, n_science))
    else:
        times = frames_science["MJD START"].to_numpy(dtype=float)
        filled, n_filled, longest_gap = sa.fill_in_time(science_wfs, times)
        amplitude = sa.wfs_amplitude(scale, level, filled, science_wfs, floor)
        n_floor = int(sa.below_floor(science_wfs, floor).sum())
        header.update(NFILLED=n_filled, MAXGAP_S=longest_gap, NFLOOR=n_floor)
        if n_floor:
            logger.warning(f"{n_floor} science frames have WFS flux below {floor:g} of the sequence's 90th "
                           "percentile and become bad frames.",
                           extra={"step": STEP_NAME, "status": "star_amplitude_below_floor"})
        mode = "wfs"
        if continuous_satellite_spots and waffle_mode == "spots":
            spots = _read_spot_trace(converted_dir, amplitude.shape, logger)
            if spots is not None:
                use = frames["kept"].to_numpy() & np.isin(block, usable_labels)
                block_times = [np.nanmean(frames.loc[use & (block == b), "mjd_start"]) for b in usable_labels]
                amplitude, levels = sa.spot_anchored_amplitude(spots, amplitude, times, block_times)
                spot_level_scatter = np.array([
                    np.nanmax(row) / np.nanmin(row) - 1 if np.isfinite(row).any() else np.nan for row in levels
                ])
                mode = "spots"
    header["MODE"] = mode
    logger.info(f"Star amplitude mode {mode}: sequence {label}, block disagreement {disagreement:.3f}.",
                extra={"step": STEP_NAME, "status": "star_amplitude_mode", "mode": mode, "class": label,
                       "disagreement": disagreement})
    _write(converted_dir, amplitude, header, np.isin(psf_labels, usable_labels), scale, level, spot_level_scatter)
    _write_block_table(converted_dir, table, psf_labels, usable_labels, excluded)
    return mode


def _warn_disagreement(disagreement: float, instrument: str, column: str, logger) -> None:
    limit = sa.DISAGREEMENT.get(str(instrument).upper(), sa.DISAGREEMENT["IFS"])
    if np.isfinite(disagreement) and disagreement > limit:
        logger.warning(f"The usable FLUX blocks disagree by {disagreement:.3f} in {column} (limit {limit:g}).",
                       extra={"step": STEP_NAME, "status": "star_amplitude_blocks_disagree"})


def _read_spot_trace(converted_dir: Path, shape: tuple[int, int], logger) -> np.ndarray | None:
    path = converted_dir / "spot_amplitude_variation.fits"
    try:
        spots = np.asarray(fits.getdata(path), dtype=float)
    except Exception as exc:
        spots, reason = None, f"it cannot be read ({exc})"
    else:
        reason = f"its shape {spots.shape} is not {shape}"
    if spots is not None and spots.shape == shape:
        return spots
    logger.warning(f"Spot mode needs spot_amplitude_variation.fits, but {reason}; the amplitude follows the WFS.",
                   extra={"step": STEP_NAME, "status": "star_amplitude_no_spot_trace"})
    return None
