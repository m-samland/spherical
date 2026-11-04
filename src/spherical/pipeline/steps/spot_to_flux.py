"""
Spot-to-Flux Normalization Step

Parameters
----------
converted_dir : str
    Directory where the output files are stored and written.
reduction_parameters : dict
    Reduction parameters dict (not used in this step, but included for API consistency).
"""
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits

from spherical.pipeline import flux_calibration
from spherical.pipeline.logging_utils import optional_logger

_BAND_NORMALIZATION_RANGES_MICRON: dict[str, tuple[float, float]] = {
    "IFS": (1.0, 1.3),
    "IRDIS_H": (1.5, 1.7),
    "IRDIS_K": (2.0, 2.3),
}


def _detect_normalization_range(wavelengths_um: np.ndarray) -> tuple[float, float]:
    """Return the spot-to-flux normalization wavelength range for the band.

    Auto-detects on the shape and value of ``wavelengths_um``:

    - **≥ 3 channels** ⇒ IFS (YJ or YJH). Range = 1.0–1.3 µm regardless of
      whether the mode reaches 1.6 µm — matches the historical IFS default.
    - **2 channels** ⇒ IRDIS DBI. Dispatch by max wavelength: > 2.0 µm → K,
      else → H.
    - **1 channel or fewer** ⇒ fall back to the IFS default (defensive).

    The returned range always contains at least one of the input wavelengths.
    """
    wl = np.asarray(wavelengths_um, dtype=float)
    if wl.size >= 3:
        return _BAND_NORMALIZATION_RANGES_MICRON["IFS"]
    if wl.size == 2:
        max_wl = float(np.nanmax(wl))
        if max_wl > 2.0:
            return _BAND_NORMALIZATION_RANGES_MICRON["IRDIS_K"]
        return _BAND_NORMALIZATION_RANGES_MICRON["IRDIS_H"]
    return _BAND_NORMALIZATION_RANGES_MICRON["IFS"]


def resolve_flux_frame_table(converted_dir) -> Path:
    """The flux frame table describing the *calibrated* products.

    ``flux_psf_calibration`` writes ``frames_info_flux_selected.csv`` for the
    frame axis of ``flux_amplitude_calibrated.fits``. ``frames_info_flux.csv``
    describes the full extraction and is the fallback for reductions made
    before selection existed.
    """
    converted_dir = Path(converted_dir)
    selected = converted_dir / 'frames_info_flux_selected.csv'
    return selected if selected.exists() else converted_dir / 'frames_info_flux.csv'


# Normal scatter between the four spots of a frame is below 7% in IRDIS, while a frame whose spots failed under
# deep cloud reads up to a factor of 28 apart. Faint channels (IFS band edges, water bands) scatter more as a rule,
# so a channel is also judged against its own typical scatter.
SPOT_SCATTER_LIMIT = 0.3
SPOT_SCATTER_NOISE_FACTOR = 3.0


@dataclass(frozen=True)
class SpotTrace:
    """Spot flux and its relative variation per channel (axis 0) and CENTER frame (axis 1).

    Attributes:
        spot_flux: Mean spot flux, on the most common waffle amplitude and DIT, NaN where not measured.
        variation: Spot flux over its time mean per channel. NaN in failed frames; filled channels carry the
            median variation of the frame's measured channels.
        failed_frames: Frames whose spots disagree in more than half of the channels or have no measured channel.
        filled: Channel values taken from the other channels of the frame.
    """

    spot_flux: np.ndarray
    variation: np.ndarray
    failed_frames: np.ndarray
    filled: np.ndarray


def spot_trace(spot_amplitudes, waffle_amplitude, dit) -> SpotTrace:
    """Relative spot flux per frame, freed of waffle amplitude and DIT changes and of failed spot measurements (#271).

    Spot flux goes as the square of the waffle amplitude and linearly with the DIT, so each frame is put on the
    most common of both; a frame without a positive value of either is not measured. A channel of a frame is not
    measured when fewer than two spots are finite, their mean is not positive, or their scatter exceeds both
    ``SPOT_SCATTER_LIMIT`` and ``SPOT_SCATTER_NOISE_FACTOR`` times the median scatter of that channel. A frame not
    measured in more than half of its channels is failed and stays NaN, since its transmission is unknown.
    Otherwise the missing channels take the median variation of the measured channels of the same frame, because a
    change in transmission dims all wavelengths nearly alike.

    Args:
        spot_amplitudes: Spot fluxes, shape ``(n_wave, n_frames, n_spots)``.
        waffle_amplitude: ``OCS WAFFLE AMPL`` per CENTER frame.
        dit: ``DET SEQ1 DIT`` per CENTER frame.

    Returns:
        The spot flux, its variation, the failed frames and the filled channel values.
    """
    spots = np.asarray(spot_amplitudes, dtype=float)
    waffle_amplitude = np.asarray(waffle_amplitude, dtype=float)
    dit = np.asarray(dit, dtype=float)
    scale = _relative_to_most_common(waffle_amplitude) ** 2 * _relative_to_most_common(dit)

    # All-NaN channels and frames are expected here and become NaN, so numpy's empty-slice warnings are noise.
    with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        n_finite = np.isfinite(spots).sum(axis=2)
        mean = np.nanmean(spots, axis=2)
        scatter = np.nanstd(spots, axis=2) / np.abs(mean)
        # Frames whose spots vanished scatter far beyond any noisy channel; if they set the typical scatter, a
        # sequence where most spots vanished would accept them.
        typical = np.nanmedian(np.where((n_finite >= 2) & (scatter < 1), scatter, np.nan), axis=1, keepdims=True)
        disagree = scatter > np.maximum(SPOT_SCATTER_LIMIT, SPOT_SCATTER_NOISE_FACTOR * typical)
        measured = (n_finite >= 2) & ~disagree & (mean > 0) & np.isfinite(scale)[None, :]

    spot_flux = np.where(measured, mean * scale[None, :], np.nan)
    failed = measured.mean(axis=0) < 0.5
    spot_flux[:, failed] = np.nan

    filled = ~measured & ~failed[None, :]
    with warnings.catch_warnings(), np.errstate(invalid='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        variation = spot_flux / np.nanmean(spot_flux, axis=1, keepdims=True)
        frame_median = np.nanmedian(np.where(measured, variation, np.nan), axis=0)
    variation = np.where(filled, frame_median[None, :], variation)
    return SpotTrace(spot_flux=spot_flux, variation=variation, failed_frames=failed, filled=filled)


def _relative_to_most_common(values):
    """Most common positive value over each value; NaN where a value is not positive, 1 when none is."""
    valid = np.isfinite(values) & (values > 0)
    if not valid.any():
        return np.ones(values.shape)
    unique, counts = np.unique(values[valid], return_counts=True)
    return np.where(valid, unique[np.argmax(counts)] / np.where(valid, values, 1.0), np.nan)


@optional_logger
def run_spot_to_flux_normalization(
    converted_dir: str,
    reduction_parameters: Dict[str, str | bool],
    logger
) -> None:
    """
    Normalize satellite spot fluxes to absolute flux scale.

    This is the eleventh step in the SPHERE/IFS data reduction pipeline. It
    computes normalization factors to convert satellite spot fluxes to absolute
    flux units by comparing them with calibrated PSF flux measurements. This
    step is essential for absolute photometry and flux calibration.

    Required Input Files
    --------------------
    From previous steps:

    ``converted_dir/wavelengths.fits``
        Wavelength array for the data cube
    ``converted_dir/flux_amplitude_calibrated.fits``
        Calibrated flux amplitudes
    ``converted_dir/additional_outputs/spot_amplitudes.fits``
        Satellite spot amplitudes
    ``converted_dir/frames_info_flux_selected.csv``
        Frame information for the selected flux cubes; frames_info_flux.csv
        for reductions made before flux cube selection existed
    ``converted_dir/frames_info_center.csv``
        Frame information for center data
    ``converted_dir/flux_calibration_indices.csv``
        Frame indices for flux calibration

    Generated Output Files
    ----------------------
    In converted_dir:

    ``spot_amplitude_variation.fits``
        Spot flux over its time mean, per channel and CENTER frame, on one
        waffle amplitude and DIT (see ``spot_trace``). NaN in frames whose
        spots failed.
    
    In converted_dir/additional_outputs/:

    ``spot_normalization_factors.fits``
        Wavelength-dependent normalization factors
    ``spot_normalization_factors_average.fits``
        Averaged normalization factors
    ``spot_normalization_factors_stddev.fits``
        Standard deviation of normalization factors

    In converted_dir/flux_plots/:

    ``psf_flux.png``
        Plot of PSF flux spectrum
    ``spot_flux_rescaled.png``
        Plot of rescaled spot flux spectrum
    ``flux_normalization_factors.png``
        Plot of wavelength-dependent normalization factors
    ``flux_timeseries.png``
        Plot of flux variation with hour angle

    Parameters
    ----------
    converted_dir : str
        Directory containing the input files and where outputs will be written.
    reduction_parameters : dict
        Reduction parameters dict (not used in this step, but included for API
        consistency).
    logger : logging.Logger
        Logger instance injected by @optional_logger for structured logging.

    Returns
    -------
    None
        This function writes normalization factors and plots to disk and does not
        return a value.

    Notes
    -----
    - Converts wavelengths to microns for internal calculations
    - Normalizes spectra in wavelength range 1.0-1.3 microns
    - Computes normalization factors by comparing:
        * Calibrated PSF flux measurements
        * Mean satellite spot amplitudes
    - Handles temporal variations by:
        * Computing per-frame normalization factors
        * Averaging across frames
        * Calculating standard deviations
    - Creates multiple visualizations:
        * Flux spectra for PSF and spots
        * Wavelength-dependent normalization factors
        * Temporal variation with hour angle
    - All output arrays are saved as float32 for efficiency

    Examples
    --------
    >>> run_spot_to_flux_normalization(
    ...     converted_dir="/path/to/converted",
    ...     reduction_parameters={},
    ...     logger=logger
    ... )
    """
    logger.info("Starting spot-to-flux normalization", extra={"step": "spot_to_flux_normalization", "status": "started"})
    plot_dir = os.path.join(converted_dir, 'flux_plots/')
    additional_outputs_dir = Path(converted_dir) / 'additional_outputs'
    additional_outputs_dir.mkdir(exist_ok=True)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        logger.debug(f"Created plot directory: {plot_dir}")
    # Load required files and log their shapes
    wavelengths_path = os.path.join(converted_dir, 'wavelengths.fits')
    flux_amplitude_path = os.path.join(converted_dir, 'flux_amplitude_calibrated.fits')
    additional_outputs_dir = Path(converted_dir) / 'additional_outputs'
    spot_amplitude_path = additional_outputs_dir / 'spot_amplitudes.fits'
    frames_info_flux_path = resolve_flux_frame_table(converted_dir)
    frames_info_center_path = os.path.join(converted_dir, 'frames_info_center.csv')
    flux_calibration_indices_path = os.path.join(converted_dir, 'flux_calibration_indices.csv')
    for fpath in [wavelengths_path, flux_amplitude_path, spot_amplitude_path, frames_info_flux_path, frames_info_center_path, flux_calibration_indices_path]:
        if not os.path.exists(fpath):
            logger.warning(f"Missing required file: {fpath}", extra={"step": "spot_to_flux_normalization", "status": "failed"})
    wavelengths = fits.getdata(wavelengths_path) * u.nm
    wavelengths = wavelengths.to(u.micron)
    norm_range = _detect_normalization_range(wavelengths.value)
    logger.info(
        f"spot_to_flux normalization range: {norm_range[0]:.2f}–{norm_range[1]:.2f} µm",
        extra={"step": "spot_to_flux_normalization", "status": "info"},
    )
    flux_amplitude = fits.getdata(flux_amplitude_path)[2]
    spot_amplitude = fits.getdata(spot_amplitude_path)
    logger.debug(f"Loaded wavelengths shape: {wavelengths.shape}, flux_amplitude shape: {flux_amplitude.shape}, spot_amplitude shape: {spot_amplitude.shape}")
    frames_info = {}
    frames_info['FLUX'] = pd.read_csv(frames_info_flux_path)
    frames_info['CENTER'] = pd.read_csv(frames_info_center_path)
    if 'OCS WAFFLE AMPL' in frames_info['CENTER']:
        waffle_amplitude = frames_info['CENTER']['OCS WAFFLE AMPL'].to_numpy(dtype=float)
    else:
        logger.warning("frames_info_center.csv has no OCS WAFFLE AMPL; spot flux is not corrected for its changes.",
                       extra={"step": "spot_to_flux_normalization", "status": "warning"})
        waffle_amplitude = np.ones(len(frames_info['CENTER']))
    trace = spot_trace(spot_amplitude, waffle_amplitude, frames_info['CENTER']['DET SEQ1 DIT'].to_numpy(dtype=float))
    n_filled = int(trace.filled.sum())
    if trace.failed_frames.any() or n_filled:
        logger.warning(
            f"Spot trace: {int(trace.failed_frames.sum())} of {trace.failed_frames.size} CENTER frames failed and are NaN "
            f"({np.flatnonzero(trace.failed_frames).tolist()}); {n_filled} single-channel values filled from the "
            "other channels of their frame.",
            extra={"step": "spot_to_flux_normalization", "status": "warning"},
        )
    master_spot_amplitude = trace.spot_flux
    if len(frames_info['FLUX']) != flux_amplitude.shape[-1]:
        raise ValueError(
            f"Flux frame table has {len(frames_info['FLUX'])} rows but "
            f"flux_amplitude_calibrated.fits has {flux_amplitude.shape[-1]} frames. "
            f"Table read from {frames_info_flux_path}."
        )
    logger.debug(f"Loaded frames_info: FLUX shape {frames_info['FLUX'].shape}, CENTER shape {frames_info['CENTER'].shape}")
    psf_flux = flux_calibration.SimpleSpectrum(
        wavelength=wavelengths,
        flux=flux_amplitude,
        norm_wavelength_range=list(norm_range) * u.micron,
        metadata=frames_info['FLUX'],
        rescale=False,
        normalize=False)
    spot_flux = flux_calibration.SimpleSpectrum(
        wavelength=wavelengths,
        flux=master_spot_amplitude,
        norm_wavelength_range=list(norm_range) * u.micron,
        metadata=frames_info['CENTER'],
        rescale=True,
        normalize=False)
    logger.debug("Constructed SimpleSpectrum objects for PSF and spot flux.")
    psf_flux.plot_flux(plot_original=False, autocolor=True, cmap=plt.cm.cool,
                       savefig=True, savedir=plot_dir, filename='psf_flux.png')
    spot_flux.plot_flux(plot_original=False, autocolor=True, cmap=plt.cm.cool,
                        savefig=True, savedir=plot_dir, filename='spot_flux_rescaled.png')
    flux_calibration_indices = pd.read_csv(flux_calibration_indices_path)
    normalization_factors, averaged_normalization, std_dev_normalization = flux_calibration.compute_flux_normalization_factors(
        flux_calibration_indices, psf_flux, spot_flux)
    logger.debug(f"Computed normalization factors: shape {normalization_factors.shape}")
    if len(wavelengths) > 2:
        wavelength_indices = np.arange(len(wavelengths))[1:-1]
    else:
        wavelength_indices = np.arange(len(wavelengths))
    flux_calibration.plot_flux_normalization_factors(
        flux_calibration_indices, normalization_factors[:, wavelength_indices],
        wavelengths=wavelengths[wavelength_indices], cmap=plt.cm.cool,
        savefig=True, savedir=plot_dir)
    fits.writeto(additional_outputs_dir / 'spot_normalization_factors.fits', normalization_factors, overwrite=True)
    fits.writeto(additional_outputs_dir / 'spot_normalization_factors_average.fits', averaged_normalization, overwrite=True)
    fits.writeto(additional_outputs_dir / 'spot_normalization_factors_stddev.fits', std_dev_normalization, overwrite=True)
    flux_calibration.plot_timeseries(
        frames_info['FLUX'], frames_info['CENTER'], psf_flux, spot_flux, averaged_normalization,
        x_axis_quantity='HOUR ANGLE', wavelength_channels=wavelength_indices,
        savefig=True, savedir=plot_dir)
    fits.writeto(os.path.join(converted_dir, 'spot_amplitude_variation.fits'), trace.variation, overwrite=True)
    logger.info("Step finished", extra={"step": "spot_to_flux_normalization", "status": "success"})
