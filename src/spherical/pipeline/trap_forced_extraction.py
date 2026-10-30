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


def _check_positions_reduced(contrast_tables, yx_positions: np.ndarray, result_folder) -> None:
    """Raise when a position lies outside the separations the reduction's noise profile covers.

    TRAP interpolates the normalization at the position's separation and clamps beyond
    the reduced range, so a position outside it, for example after the known positions
    changed without rerunning the reduction, would get silently wrong uncertainties.
    """
    finite = [table["sep (pix)"][np.isfinite(table["snr_normalization"])] for table in contrast_tables]
    low = min(float(np.min(sep)) for sep in finite if len(sep))
    high = max(float(np.max(sep)) for sep in finite if len(sep))
    for y, x in yx_positions:
        separation = float(np.hypot(y, x))
        if not low <= separation <= high:
            raise ValueError(
                f"Position ({y:.2f}, {x:.2f}) at {separation:.1f} px is outside the separations "
                f"{low:.1f}-{high:.1f} px of the TRAP reduction in {result_folder}. Rerun the reduction "
                f'with force={{"run_trap_reduction"}} after changing the search region or positions.'
            )


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

    Returns:
        TRAP's spectrum table (``candidate_id`` ... ``norm_factor``), where
        ``candidate_id`` indexes *yx_positions*, with the added columns
        ``y_relative`` and ``x_relative``. The same table is written to
        :func:`forced_spectra_path`.
    """
    yx_positions = np.atleast_2d(np.asarray(yx_positions, dtype=float))
    analysis = DetectionAnalysis(reduction_parameters=None, instrument=None)
    analysis.read_output(
        temporal_components_fraction,
        result_folder=str(result_folder),
        reduction_type="temporal",
        correlated_residuals=False,
        read_parameters=True,
        read_instrument=True,
    )
    analysis.reduction_parameters = analysis.reduction_parameters.merge(result_folder=str(result_folder))
    # The detection maps' noise profile, with the configured known positions masked,
    # as a normal detection computes it before any re-extraction.
    analysis.contrast_table_and_normalization(save=False, inplace=True)
    _check_positions_reduced(analysis.detection_products["contrast_tables"], yx_positions, result_folder)

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
    spectra["y_relative"] = yx_positions[index, 0]
    spectra["x_relative"] = yx_positions[index, 1]
    spectra.to_csv(forced_spectra_path(result_folder, temporal_components_fraction), index=False)
    return spectra
