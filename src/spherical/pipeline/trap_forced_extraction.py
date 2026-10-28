"""Contrast spectra at known companion positions from a finished TRAP reduction.

The extraction runs on a fresh ``DetectionAnalysis`` read from the result folder.
The object left over from the detection must not be reused: template matching
replaces its known positions with the detected ones, and once it holds fitted
candidates TRAP interpolates the noise normalization at the separation of a
detected candidate instead of at the requested position.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from trap.detection import DetectionAnalysis

from spherical.pipeline.step_registry import trap_detection_run


def forced_extraction_run(temporal_components_fraction: float, yx_positions: np.ndarray) -> str:
    """Name a forced extraction by its TRAP run and positions, for its completion marker."""
    positions = ";".join(f"{y:.2f},{x:.2f}" for y, x in np.atleast_2d(yx_positions))
    return f"{trap_detection_run(temporal_components_fraction)}|{positions}"


def forced_spectra_path(result_folder: Path, temporal_components_fraction: float) -> Path:
    """Path of the forced-extraction table in a TRAP result folder."""
    return Path(result_folder) / f"forced_spectra_{trap_detection_run(temporal_components_fraction)}.csv"


def extract_forced_spectra(
    result_folder: Path,
    temporal_components_fraction: float,
    yx_positions: np.ndarray,
    *,
    data_full,
    flux_psf_full,
    pa,
    wavelength_indices,
    inverse_variance_full=None,
    bad_frames=None,
    bad_pixel_mask_full=None,
    xy_image_centers=None,
    amplitude_modulation_full=None,
    analysis_factory=DetectionAnalysis,
) -> pd.DataFrame:
    """Re-reduce at fixed positions and write their contrast spectra.

    The position is not refitted. Uncertainties are normalized with the noise
    profile of the reduction's detection maps, with the configured known
    companion positions masked, as in a normal detection.

    Args:
        result_folder: TRAP result folder of a finished reduction.
        temporal_components_fraction: Fraction of the reduction to read.
        yx_positions: (N, 2) array of (y, x) offsets from the star in pixels.
        data_full: Science cube as passed to the TRAP reduction.
        flux_psf_full: PSF cube as passed to the TRAP reduction.
        pa: Derotation angle per frame in degrees.
        wavelength_indices: Channels to extract.
        inverse_variance_full: Inverse-variance cube, if the reduction used one.
        bad_frames: Frames excluded from the reduction.
        bad_pixel_mask_full: Bad-pixel mask of the reduction.
        xy_image_centers: Star position per frame and channel.
        amplitude_modulation_full: Stellar amplitude modulation per frame.
        analysis_factory: Builds the analysis object; replaced in tests.

    Returns:
        TRAP's spectrum table (``candidate_id`` ... ``norm_factor``) with the
        added columns ``position_index``, ``y_relative``, ``x_relative`` and
        ``separation_px``. The same table is written to
        :func:`forced_spectra_path`.
    """
    yx_positions = np.atleast_2d(np.asarray(yx_positions, dtype=float))
    analysis = analysis_factory(reduction_parameters=None, instrument=None)
    analysis.read_output(
        temporal_components_fraction,
        result_folder=str(result_folder),
        reduction_type="temporal",
        correlated_residuals=False,
        read_parameters=True,
        read_instrument=True,
    )
    analysis.reduction_parameters = analysis.reduction_parameters.merge(result_folder=str(result_folder))

    spectra = analysis.extract_candidate_spectra(
        temporal_components_fraction,
        data_full,
        flux_psf_full,
        pa,
        yx_candidate_positions=yx_positions,
        wavelength_indices=wavelength_indices,
        inverse_variance_full=inverse_variance_full,
        bad_frames=bad_frames,
        bad_pixel_mask_full=bad_pixel_mask_full,
        xy_image_centers=xy_image_centers,
        amplitude_modulation_full=amplitude_modulation_full,
        return_spectra=True,
    )
    spectra = spectra.reset_index(drop=True)
    index = spectra["candidate_id"].to_numpy(dtype=int)
    spectra["position_index"] = index
    spectra["y_relative"] = yx_positions[index, 0]
    spectra["x_relative"] = yx_positions[index, 1]
    spectra["separation_px"] = np.hypot(yx_positions[index, 0], yx_positions[index, 1])
    spectra.to_csv(forced_spectra_path(result_folder, temporal_components_fraction), index=False)
    return spectra
