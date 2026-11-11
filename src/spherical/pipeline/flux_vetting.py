"""Vetting of FLUX frames before they build the PSF and the flux scale (#81).

Each FLUX frame and channel is measured on its PSF stamp, and the frames and
channels that do not show the star as it should look are removed: fits on noise,
on another source or on a hot pixel, and images blurred by AO that has not
settled. The thresholds come from the FLUX frames of 146 IRDIS and 99 IFS
sequences, where good frames and failures separate by orders of magnitude.

A test fails one channel of one frame. A frame is rejected in all channels when
it fails detection or when more than half of its channels fail; otherwise only
the failed channels are masked, so one weak channel does not remove a time step.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd
from scipy.special import j0, j1

from spherical.pipeline.pipeline_config import PreprocConfig
from spherical.pipeline.steps.find_star import TELESCOPE_DIAMETER_M

#: Aperture radii (px) of the detection S/N, the flux and the sharpness.
CORE_RADIUS = 3
WING_RADIUS = 5

TESTS = (
    "frame_detection",
    "channel_detection",
    "position",
    "flux",
    "sharpness_low",
    "sharpness_airy",
)

#: Thresholds with an ``_irdis`` and an ``_ifs`` entry in ``PreprocConfig``.
PER_INSTRUMENT = ("frame_snr", "channel_snr")


class NoUsableFluxFrameError(ValueError):
    """Raised when no FLUX frame passes the vetting, so no PSF can be built."""


@dataclass(frozen=True, slots=True)
class FluxVettingThresholds:
    """Thresholds of the FLUX frame vetting.

    The values and their meaning are the ``flux_vetting_*`` fields of
    ``PreprocConfig``, which hold the defaults.
    """

    frame_snr: float
    channel_snr: float
    max_channel_scatter_px: float
    max_offset_px: float
    min_relative_flux: float
    min_relative_sharpness: float
    max_airy_sharpness: float

    @classmethod
    def from_parameters(cls, reduction_parameters, instrument) -> FluxVettingThresholds:
        """Thresholds from the ``flux_vetting_*`` entries of the reduction parameters.

        The S/N thresholds come from the ``_irdis`` or ``_ifs`` entry, by
        ``instrument``. Real IRDIS frames reach S/N of hundreds in each channel,
        while faint IFS frames and band-edge channels legitimately sit near 10.
        """
        parameters = {**asdict(PreprocConfig()), **reduction_parameters}
        suffix = f"_{instrument.lower()}"
        return cls(**{
            f.name: float(parameters[f"flux_vetting_{f.name}{suffix if f.name in PER_INSTRUMENT else ''}"])
            for f in fields(cls)
        })


@dataclass(frozen=True)
class FluxVetting:
    """Outcome of :func:`vet_flux_frames`.

    Attributes:
        table: One row per frame and channel with every measurement, each test
            result (True means passed) and the decision.
        frame_kept: Shape ``(n_frames,)``, frames that stay on the frame axis.
        channel_kept: Shape ``(n_wave, n_frames)``, channels that enter the PSF
            and the flux scale. False wherever the frame is rejected.
    """

    table: pd.DataFrame
    frame_kept: np.ndarray
    channel_kept: np.ndarray


def airy_sharpness(wavelengths_nm, pixel_scale_mas, telescope_diameter_m=TELESCOPE_DIAMETER_M):
    """Flux within ``CORE_RADIUS`` over flux within ``WING_RADIUS`` of an unobstructed Airy pattern.

    The central obscuration moves light from the core into the rings and lowers
    the ratio, so the unobstructed value is the upper limit for a real PSF.
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    lambda_over_d_px = wavelengths_nm * 1e-9 / telescope_diameter_m * 206264.806e3 / pixel_scale_mas

    def encircled_energy(radius_px):
        x = np.pi * radius_px / lambda_over_d_px
        return 1.0 - j0(x) ** 2 - j1(x) ** 2

    return encircled_energy(CORE_RADIUS) / encircled_energy(WING_RADIUS)


def block_labels(block_ranges, n_frames):
    """Block index of every frame, from the ``(start, stop)`` ranges of ``flux_block_ranges``."""
    labels = np.full(n_frames, -1, dtype=int)
    for index, (start, stop) in enumerate(block_ranges):
        labels[start:stop] = index
    return labels


def _nanmedian(values, axis):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def vet_flux_frames(photometry, centers_xy, wavelengths_nm, pixel_scale_mas, origfile, block, thresholds):
    """Measure and vet every FLUX frame and channel.

    Args:
        photometry: Output of ``flux_calibration.get_aperture_photometry`` on the
            calibrated stamps, with radii 3 and 5 among ``aperture_sizes``.
        centers_xy: Fitted ``(x, y)`` centres, shape ``(n_wave, n_frames, 2)``.
        wavelengths_nm: Channel wavelengths in nm.
        pixel_scale_mas: Detector pixel scale.
        origfile: Raw file of every frame, shape ``(n_frames,)``. The position
            reference is taken per file, because the star may sit elsewhere in
            another cube of the sequence.
        block: Block index of every frame, shape ``(n_frames,)``.
        thresholds: The rules' thresholds.

    Returns:
        FluxVetting.
    """
    radii = [int(r) for r in photometry["aperture_sizes"]]
    flux = np.asarray(photometry["psf_flux_bg_corr_all"], dtype=float)
    core = flux[radii.index(CORE_RADIUS)]
    wing = flux[radii.index(WING_RADIUS)]
    core_area = float(np.asarray(photometry["psf_area"])[radii.index(CORE_RADIUS)])
    bg_std = np.asarray(photometry["psf_bg_std_all"], dtype=float)
    centers_xy = np.asarray(centers_xy, dtype=float)
    origfile = np.asarray(origfile)
    block = np.asarray(block)
    n_wave, n_frames = core.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        # The existing ``snr_all`` divides by the aperture area, not its square root.
        snr = np.where(bg_std > 0, core / (bg_std * np.sqrt(core_area)), np.nan)
        frame_snr = _nanmedian(snr, axis=0)
        common_xy = _nanmedian(centers_xy, axis=0)
        channel_scatter = np.hypot(*(centers_xy - common_xy[None]).transpose(2, 0, 1))
        frame_scatter = _nanmedian(channel_scatter, axis=0)

        frame_detected = frame_snr >= thresholds.frame_snr
        if n_wave > 2:
            frame_detected &= frame_scatter <= thresholds.max_channel_scatter_px
        channel_detected = snr >= thresholds.channel_snr
        detected = frame_detected[None] & channel_detected

        offset = np.full((n_wave, n_frames), np.nan)
        for name in np.unique(origfile):
            members = origfile == name
            reference = _nanmedian(np.where(detected[:, members, None], centers_xy[:, members], np.nan), axis=1)
            offset[:, members] = np.hypot(*(centers_xy[:, members] - reference[:, None]).transpose(2, 0, 1))
        position_ok = offset <= thresholds.max_offset_px
        reference_frames = detected & position_ok

        # Both are relative to the frame's own block, so a block that is fainter or
        # blurrier as a whole, from a cloud or a change of the AO correction, is kept.
        sharpness = core / wing
        relative_flux = np.full((n_wave, n_frames), np.nan)
        relative_sharpness = np.full((n_wave, n_frames), np.nan)
        for label in np.unique(block):
            members = block == label
            for relative, value in ((relative_flux, core), (relative_sharpness, sharpness)):
                reference = _nanmedian(np.where(reference_frames[:, members], value[:, members], np.nan), axis=1)
                relative[:, members] = value[:, members] / reference[:, None]

        airy_ratio = sharpness / airy_sharpness(wavelengths_nm, pixel_scale_mas)[:, None]

        passed = {
            "frame_detection": np.broadcast_to(frame_detected[None], (n_wave, n_frames)),
            "channel_detection": channel_detected,
            "position": position_ok,
            "flux": relative_flux >= thresholds.min_relative_flux,
            "sharpness_low": relative_sharpness >= thresholds.min_relative_sharpness,
            "sharpness_airy": airy_ratio <= thresholds.max_airy_sharpness,
        }
    channel_ok = np.logical_and.reduce(list(passed.values()))
    frame_kept = frame_detected & ((~channel_ok).sum(axis=0) <= n_wave / 2)
    channel_kept = channel_ok & frame_kept[None]

    decision = np.where(~frame_kept[None], "frame_rejected", np.where(channel_kept, "kept", "channel_masked"))
    failed = np.array([";".join(name for name in TESTS if not passed[name][ch, f])
                       for ch in range(n_wave) for f in range(n_frames)]).reshape(n_wave, n_frames)
    channel, frame = np.meshgrid(np.arange(n_wave), np.arange(n_frames), indexing="ij")
    columns = {
        "frame": frame, "channel": channel,
        "wavelength_nm": np.broadcast_to(np.asarray(wavelengths_nm, dtype=float)[:, None], (n_wave, n_frames)),
        "origfile": np.broadcast_to(origfile[None], (n_wave, n_frames)),
        "block": np.broadcast_to(block[None], (n_wave, n_frames)),
        "x": centers_xy[..., 0], "y": centers_xy[..., 1],
        "snr": snr, "frame_snr": np.broadcast_to(frame_snr[None], (n_wave, n_frames)),
        "channel_scatter_px": np.broadcast_to(frame_scatter[None], (n_wave, n_frames)),
        "offset_px": offset, "flux_r3": core, "relative_flux": relative_flux,
        "sharpness": sharpness, "relative_sharpness": relative_sharpness, "airy_ratio": airy_ratio,
        **{f"pass_{name}": passed[name] for name in TESTS},
        "failed_tests": failed, "decision": decision,
    }
    table = pd.DataFrame({name: np.asarray(values).ravel() for name, values in columns.items()})
    return FluxVetting(table=table, frame_kept=frame_kept, channel_kept=channel_kept)


def failed_test_counts(table):
    """How often each test failed, over the rows of ``table``, for the log."""
    return {name: int((~table[f"pass_{name}"]).sum()) for name in TESTS if (~table[f"pass_{name}"]).any()}


def subset_photometry(photometry, frame_kept, channel_kept):
    """Photometry restricted to the kept frames, NaN in masked channels.

    Every array with a frame axis has it last. Masked channels must read NaN,
    not the zero ``get_aperture_photometry`` returns for a fully masked stamp,
    so the block normalisation and ``spot_to_flux`` skip them.
    """
    n_frames = len(frame_kept)
    masked = ~channel_kept[:, frame_kept]
    subset = {}
    for key, value in photometry.items():
        value = np.asarray(value)
        if value.ndim >= 2 and value.shape[-1] == n_frames:
            value = value[..., frame_kept].astype(float)
            value[..., masked] = np.nan
        subset[key] = value
    return subset


def report_flux_vetting(vetting, warn_kept_fraction, logger):
    """Log the outcome of the vetting, and stop when no frame survives.

    Raises:
        NoUsableFluxFrameError: When no frame is kept, because the PSF would be
            noise and every contrast of the sequence meaningless.
    """
    table = vetting.table
    n_frames, n_kept = len(vetting.frame_kept), int(vetting.frame_kept.sum())
    step = {"step": "flux_psf_calibration"}
    if n_kept == 0:
        message = (f"No FLUX frame passes the vetting ({n_frames} frames; failed tests over channels and frames: "
                   f"{failed_test_counts(table)}). The sequence has no usable PSF; see "
                   f"additional_outputs/flux_frame_vetting.csv.")
        logger.error(message, extra={**step, "status": "flux_vetting_no_frame"})
        raise NoUsableFluxFrameError(message)

    rejected = table[table["decision"] == "frame_rejected"]
    masked = table[table["decision"] == "channel_masked"]
    logger.info(
        f"FLUX frame vetting kept {n_kept} of {n_frames} frames and masked {len(masked)} channels of kept frames.",
        extra={**step, "status": "flux_vetting_summary", "n_frames": n_frames, "n_kept": n_kept,
               "n_channels_masked": len(masked)},
    )
    if len(rejected):
        frames = rejected.drop_duplicates("frame")
        logger.warning(
            f"FLUX frame vetting rejected {len(frames)} of {n_frames} frames "
            f"({frames.groupby('origfile').size().to_dict()}); failed tests: {failed_test_counts(rejected)}.",
            extra={**step, "status": "flux_frames_rejected", "n_rejected": len(frames)},
        )
    if n_kept < warn_kept_fraction * n_frames:
        logger.warning(
            f"Only {n_kept} of {n_frames} FLUX frames pass the vetting. The PSF and the flux scale rest on few "
            "frames; check additional_outputs/flux_frame_vetting.csv before using the photometry.",
            extra={**step, "status": "flux_vetting_few_frames", "kept_fraction": n_kept / n_frames},
        )
    for channel in np.flatnonzero(~vetting.channel_kept.any(axis=1)):
        rows = table[table["channel"] == channel]
        logger.warning(
            f"Channel {channel} ({rows['wavelength_nm'].iloc[0]:.0f} nm) keeps no FLUX frame "
            f"(failed tests: {failed_test_counts(rows)}). Its PSF is NaN and TRAP skips it.",
            extra={**step, "status": "flux_channel_unusable", "channel": int(channel)},
        )
    kept_blocks = set(table.loc[table["decision"] != "frame_rejected", "block"])
    emptied = sorted(set(table["block"]) - kept_blocks)
    if emptied:
        logger.warning(
            f"FLUX blocks {emptied} lose every frame to the vetting and do not enter the PSF.",
            extra={**step, "status": "flux_block_emptied", "blocks": [int(b) for b in emptied]},
        )
