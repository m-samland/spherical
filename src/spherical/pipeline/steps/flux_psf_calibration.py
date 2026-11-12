"""
Flux PSF Calibration Step

Parameters
----------
converted_dir : str
    Directory where the output files are stored and written.
reduction_parameters : dict
    Reduction parameters dict, must contain 'flux_combination_method', 'exclude_first_flux_frame', and 'exclude_first_flux_frame_all'.
"""
import os
import warnings
from pathlib import Path
from typing import Dict

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from spherical.pipeline import flux_calibration, toolbox, transmission
from spherical.pipeline.cube_selection import core_peaks, select_flux_cubes, summarise_cubes
from spherical.pipeline.flux_vetting import (
    FluxVettingThresholds,
    NoUsableFluxFrameError,
    block_labels,
    report_flux_vetting,
    subset_photometry,
    vet_flux_frames,
)
from spherical.pipeline.ivar_badpixels import bad_pixel_mask_from_ivar
from spherical.pipeline.logging_utils import optional_logger
from spherical.pipeline.pipeline_config import PreprocConfig
from spherical.pipeline.psf_repair import repair_psf_core
from spherical.pipeline.steps.find_star import (
    guess_position_psf,
    guess_positions_joint,
    guess_positions_per_channel,
    lambda_over_d_pixels,
    star_centers_from_PSF_img_cube,
)
from spherical.pipeline.steps.irdis_preprocess import nominal_star_positions

#: Distance (px) between the joint FLUX guess and a channel's own brightest
#: pixel above which the step reports that channel as reseeded.
GUESS_DISAGREEMENT_PX = 3.0


def coronagraph_center_from_disk(converted_dir, logger, irdis_filter_comb=None):
    """``(x, y)`` of the coronagraph centre, or ``None`` if unavailable.

    The centre marks where the star sat behind the coronagraph, which is where
    detector persistence from the CORO sequence appears in a trailing FLUX
    block (#83). The flux step runs after ``find_centers`` and
    ``process_extracted_centers``, so the measured centre is already on disk;
    prefer it to any literal. On the 51 Eri OBS_H reference run it is
    (127.38, 127.80), about 3 px from the (126, 131) this used to hard-code --
    and that literal was being applied to IRDIS half-frames too, where it
    means nothing.

    ``irdis_filter_comb`` supplies the IRDIS fallback: the nominal star
    position is the same CORO-frame quantity, measured per filter. Pass it
    only for IRDIS; IFS has no nominal and falls through to ``None``.
    """
    for name in ("image_centers_fitted_robust.fits", "image_centers.fits"):
        path = os.path.join(converted_dir, name)
        if not os.path.exists(path):
            continue
        centers = np.asarray(fits.getdata(path), dtype=float).reshape(-1, 2)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                    category=RuntimeWarning)
            xy = np.nanmedian(centers, axis=0)
        if np.all(np.isfinite(xy)):
            logger.info(
                f"Coronagraph centre from {name}: "
                f"(x, y) = ({xy[0]:.2f}, {xy[1]:.2f})",
                extra={"step": "flux_psf_calibration", "status": "center_from_disk"},
            )
            return float(xy[0]), float(xy[1])
        logger.warning(f"{name} holds no finite centre; trying the next source.")

    if irdis_filter_comb is not None:
        x, y = nominal_star_positions(irdis_filter_comb)[0]
        logger.warning(
            f"No measured coronagraph centre on disk; falling back to the "
            f"{irdis_filter_comb} nominal star position "
            f"(x, y) = ({x:.2f}, {y:.2f}).",
            extra={"step": "flux_psf_calibration", "status": "center_from_nominal"},
        )
        return float(x), float(y)

    logger.warning(
        "No measured coronagraph centre on disk; the centre guess will run "
        "without a coronagraph mask, so residual persistence from the CORO "
        "sequence is not rejected (see #83).",
        extra={"step": "flux_psf_calibration", "status": "no_coronagraph_mask"},
    )
    return None


#: Products in ``converted/`` that later steps read from this step.
STEP_PRODUCTS = ("psf_cube_for_postprocessing.fits", "psf_cube_for_postprocessing_unrepaired.fits",
                 "flux_amplitude_calibrated.fits", "flux_calibration_indices.csv", "frames_info_flux_selected.csv")


def finalize_psf_cube(cube, logger):
    """Return ``cube`` ready for the TRAP PSF template, shape ``(n_wave, n_blocks, ny, nx)``.

    ``run_trap`` collapses ``psf_cube_for_postprocessing.fits`` with
    ``np.nanmean`` over the blocks and has no NaN guard, and a partial stamp's
    padding sits in the same corner of every frame, so the block combine cannot
    remove it. Non-finite pixels of a stamp that holds data become zero, which is
    harmless to TRAP's forward model; a NaN is not.

    A stamp without any finite pixel is a block left without frames in that
    channel by the FLUX vetting. It stays NaN, so the mean over blocks takes the
    other blocks instead of averaging in zeros. A channel with no block left
    stays NaN as well, and TRAP skips a wavelength whose PSF is not finite.
    """
    cube = np.asarray(cube, dtype=float)
    finite = np.isfinite(cube)
    empty = ~finite.any(axis=(-2, -1))
    n_nan = int(np.count_nonzero(~finite & ~empty[..., None, None]))
    if n_nan:
        logger.warning(
            f"PSF cube carries {n_nan} non-finite pixels "
            f"({100 * n_nan / cube.size:.3f}%); writing them as zero so the "
            f"TRAP PSF template stays finite.",
            extra={"step": "flux_psf_calibration", "status": "psf_cube_nan_zeroed"},
        )
    return np.where(finite, cube, np.where(empty[..., None, None], np.nan, 0.0))


def build_nd_attenuation(frames_info_flux, wavelengths) -> np.ndarray:
    """ND attenuation for every flux frame, shape ``(n_wave, n_frames)``.

    Flux cubes within one sequence may legitimately carry different ND filters:
    two OBs on the same pointing, or an observer correcting the setup.  Scaling
    every frame by one scalar silently mis-calibrates those, so this returns the
    attenuation frame by frame instead.
    """
    nd_names = [str(x) for x in frames_info_flux['INS4 FILT2 NAME']]
    columns = []
    for nd in nd_names:
        _, attenuation = transmission.transmission_nd(nd, wave=wavelengths)
        columns.append(np.asarray(attenuation, dtype=float))
    return np.stack(columns, axis=-1)


def apply_flux_cube_selection(
    flux_cube, flux_ivar_cube, flux_bpm_cube, frames_info_flux, guess_positions_yx,
    exposure_level, science_mid_mjd, threshold_adu, nonlinearity_adu, selection, frame_indices=None,
):
    """Subset every flux-frame-indexed array to the selected cubes and frames.

    ``frames_info_flux`` is reindexed from 0 because everything downstream
    addresses flux frames positionally. ``flux_ivar_cube`` and ``flux_bpm_cube``
    may be None. ``"all"`` skips the core-peak measurement. ``frame_indices`` keeps
    only those time-ordered frames of the input, within the selected cubes.

    Returns:
        The five arrays subset, then a one-row-per-input-cube report.
    """
    peaks = None if selection == "all" else core_peaks(flux_cube, guess_positions_yx)
    report = summarise_cubes(frames_info_flux, exposure_level, peaks)
    peak_adu = report["peak_adu"].to_numpy()
    saturated = peak_adu >= threshold_adu
    keep, status = select_flux_cubes(report, saturated, science_mid_mjd, selection)
    report = report.assign(saturated=saturated, nonlinear=(peak_adu >= nonlinearity_adu) & ~saturated,
                           kept=keep, status=status)

    frame_mask = frames_info_flux["ORIGFILE"].isin(report.loc[keep, "origfile"]).to_numpy()
    if frame_indices is not None:
        chosen = np.zeros(len(frame_mask), dtype=bool)
        try:
            chosen[np.asarray(frame_indices, dtype=int)] = True
        except IndexError as error:
            raise ValueError(f"Flux frame indices {frame_indices} exceed the {len(frame_mask)} flux frames.") from error
        frame_mask = frame_mask & chosen
        if not frame_mask.any():
            raise ValueError(f"Flux frame indices {frame_indices} keep no frame of the selected cubes.")
        report["kept"] = report["origfile"].isin(frames_info_flux.loc[frame_mask, "ORIGFILE"])
    return (
        flux_cube[:, frame_mask],
        None if flux_ivar_cube is None else flux_ivar_cube[:, frame_mask],
        None if flux_bpm_cube is None else flux_bpm_cube[:, frame_mask],
        frames_info_flux.loc[frame_mask].reset_index(drop=True),
        [g for g, k in zip(guess_positions_yx, frame_mask) if k],
        report,
    )


def irdis_guess_smoothing_px(wavelengths_nm, pixel_scale_mas):
    """Gaussian sigma in pixels whose FWHM is lambda/D at the median wavelength."""
    return float(np.median(lambda_over_d_pixels(wavelengths_nm, pixel_scale_mas))) / (2 * np.sqrt(2 * np.log(2)))


def read_channel_offset(additional_outputs_dir, logger):
    """Star offset ``(dx, dy)`` of IRDIS channel 1 against channel 0 from the CENTER frames, or None."""
    path = Path(additional_outputs_dir) / "cross_channel_offset.fits"
    offset = fits.getdata(path) if path.exists() else None
    if offset is None or not np.all(np.isfinite(offset)):
        # The offset drifts by several pixels over the years, so no nominal value
        # can stand in for the measured one.
        logger.warning(
            "No measured cross-channel offset (cross_channel_offset.fits from the centring step); "
            "each IRDIS channel guesses the FLUX star position on its own.",
            extra={"step": "flux_psf_calibration", "status": "flux_guess_independent"},
        )
        return None
    return np.asarray(offset, dtype=float)


def bad_pixel_stamps(bpm_cube, flux_centers, stamp_size=57):
    """Bad-pixel stamps aligned to the flux stamps, shape ``(n_wave, n_frames, stamp_size, stamp_size)``.

    The stamp extractor shifts by sub-pixel amounts, but a boolean mask cannot
    be interpolated, so each stamp is cut at the integer-rounded center (a bad
    pixel stays bad, no partial credit). A frame without a finite center, or
    whose stamp leaves the detector, is masked entirely.
    """
    half = stamp_size // 2
    n_wave, n_frames, ny, nx = bpm_cube.shape
    stamps = np.ones((n_wave, n_frames, stamp_size, stamp_size), dtype=bool)
    for ch in range(n_wave):
        for f in range(n_frames):
            cx, cy = flux_centers[ch, f, 0]
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            y0, x0 = int(round(float(cy))) - half, int(round(float(cx))) - half
            y1, x1 = y0 + stamp_size, x0 + stamp_size
            if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
                continue
            stamps[ch, f] = bpm_cube[ch, f, y0:y1, x0:x1]
    return stamps


def block_normalization(aperture_flux, first_combined):
    """Each frame's aperture flux relative to its block's median (#159).

    The median, not the mean: one frame with bad photometry (a passing cloud,
    bad core pixels, a failed background) would otherwise rescale the whole
    block's combined PSF.

    Args:
        aperture_flux: Aperture flux per frame of one block, shape ``(n_wave, n_frames)``.
        first_combined: First frame of the block that enters the combination, so
            an excluded first frame does not set the reference.
    """
    return aperture_flux / np.nanmedian(aperture_flux[:, first_combined:], axis=1)[:, None]


def first_combined_frame(block_index, n_frames, reduction_parameters, first_frame_present=True):
    """Index of the first frame of a flux block that enters the combination.

    ``exclude_first_flux_frame_all`` drops the first frame of every block,
    ``exclude_first_flux_frame`` only that of the first block (#212). A
    single-frame block is always kept. ``block_index`` is the block's position
    before the FLUX vetting, and ``first_frame_present`` is False when the
    vetting already removed the block's first frame, so no second frame is dropped.
    """
    exclude = reduction_parameters['exclude_first_flux_frame_all'] or (
        block_index == 0 and reduction_parameters['exclude_first_flux_frame'])
    return int(exclude and first_frame_present and n_frames > 1)


@optional_logger
def run_flux_psf_calibration(
    converted_dir: str,
    reduction_parameters: Dict[str, str | bool],
    logger,
) -> None:
    """
    Calibrate flux PSF measurements in SPHERE/IFS data.

    This is the tenth step in the SPHERE/IFS data reduction pipeline. It performs
    flux calibration of PSF measurements, including DIT normalization, ND filter
    correction, and frame combination. This step is crucial for absolute flux
    calibration and photometric accuracy.

    Required Input Files
    --------------------
    From previous steps:

    ``converted_dir/wavelengths.fits``
        Wavelength array for the data cube.
    ``converted_dir/flux_cube.fits``
        Master cube of flux data.
    ``converted_dir/frames_info_center.csv``
        Frame information for center data.
    ``converted_dir/frames_info_flux.csv``
        Frame information for flux data.

    Generated Output Files
    ----------------------
    In ``converted_dir``:

    ``flux_amplitude_calibrated.fits``
        Calibrated flux amplitudes. Core bad pixels the Moffat repair treated
        count with their repaired (or, where the fit was skipped, upstream
        interpolated) values; other bad pixels are masked. A channel masked by
        the vetting is NaN for that frame.
    ``flux_calibration_indices.csv``
        Frame indices for flux calibration.
    ``frames_info_flux_selected.csv``
        Frame information for the frames of the selected flux cubes that pass
        the vetting, the frame axis of the calibrated products
        (``frames_info_flux.csv`` keeps the full extraction).
    ``psf_cube_for_postprocessing.fits``
        Combined calibrated flux PSF frames. A block or channel without a
        frame that passes the vetting is NaN.

    In ``converted_dir/additional_outputs/``:

    ``flux_centers.fits``
        Fitted centers of flux PSFs.
    ``flux_gauss_amplitudes.fits``
        Gaussian fit amplitudes for flux PSFs.
    ``flux_stamps_uncalibrated.fits``
        Extracted flux PSF stamps, cut from the Moffat-repaired cube whenever a
        repair ran, so these and the calibrated stamps below are the FLUX data
        to use downstream.
    ``flux_cube_repaired.fits``, ``flux_ivar_repaired.fits``, ``flux_psf_repair_log.csv``
        Moffat-repaired flux cube, its inverse variance, and one row per
        attempted repair. Written only when a PSF core holds a bad pixel, before
        the vetting, so their frame axis includes the rejected frames.
    ``nd_attenuation.fits``
        ND filter transmission correction per flux frame, shape (n_wave, n_frames).
    ``flux_cube_selection.csv``
        One row per flux cube: setup, core peak, saturated / nonlinear, kept.
    ``flux_frame_vetting.csv``
        One row per flux frame and channel of the selected cubes: the
        measurements of the vetting, each test's result and the decision
        (``kept``, ``channel_masked``, ``frame_rejected``).
    ``flux_stamps_dit_nd_calibrated.fits``
        DIT and ND-corrected flux stamps.
    ``flux_photometry.obj``
        Pickled photometry results.
    ``flux_snr.fits``
        Signal-to-noise ratios.
    ``flux_stamps_calibrated_bg_corrected.fits``
        Background-subtracted calibrated stamps.
    ``indices_of_discontinuity.csv``
        Indices where flux calibration changes.
    ``Flux_PSF_aperture_SNR.png``
        Plot of SNR vs aperture size.

    Parameters
    ----------
    converted_dir : str
        Directory containing the input files and where outputs will be written.
    reduction_parameters : dict
        Reduction parameters dict. Must contain:

        ``flux_combination_method`` (str)
            Method to combine flux frames ('mean' or 'median').
        ``exclude_first_flux_frame`` (bool)
            Whether to exclude the first frame of the first flux block.
        ``exclude_first_flux_frame_all`` (bool)
            Whether to exclude the first frame of every flux block.
        ``flux_cube_selection_irdis``, ``flux_cube_selection_ifs`` (str or int)
            Which flux cubes calibrate the PSF (see ``PreprocConfig``).
        ``flux_frame_indices`` (sequence of int or None)
            Which flux frames calibrate the PSF (see ``PreprocConfig``).
        ``flux_saturation_adu``, ``flux_nonlinearity_adu`` (float)
            Core-peak levels at which a flux cube is dropped or warned about.
    logger : logging.Logger
        Logger instance injected by @optional_logger for structured logging.

    Returns
    -------
    None
        This function writes calibrated flux data to disk and does not return
        a value.

    Notes
    -----
    - Extracts 57x57 pixel stamps centered on each flux PSF
    - Performs multiple calibration steps:

      * DIT normalization using most common DIT from center frames
      * ND filter transmission correction, frame by frame
      * Background subtraction using annulus photometry

    - Uses aperture photometry with:

      * Aperture radius range: 1-15 pixels
      * Background annulus: 15-18 pixels

    - Handles frame combination with:

      * Configurable combination method (mean/median)
      * Optional first frame exclusion
      * Frame sequence detection

    - Creates visualization of SNR vs aperture size
    - All output arrays are saved as float32 for efficiency

    The ``psf_cube_for_postprocessing.fits`` file (the master flux calibrated PSF
    frames) is built in six stages:

    1. **Cube Selection**: :func:`apply_flux_cube_selection` keeps the flux cubes
       chosen by ``flux_cube_selection_irdis`` or ``flux_cube_selection_ifs``
       (#172). The IRDIS default ``"auto"`` drops cubes whose PSF core reaches
       ``flux_saturation_adu`` and keeps all others, whatever their DIT and ND;
       IFS defaults to ``"all"``. Every later stage sees only the kept frames.

    2. **Frame Vetting** (#81): :func:`~spherical.pipeline.flux_vetting.vet_flux_frames`
       measures every frame and channel on its stamp and rejects frames without
       a detected star, with a displaced fit, far too little flux or a PSF too
       broad or sharper than the diffraction limit. A frame failing in more
       than half of its channels leaves the frame axis of every product of this
       step; a frame failing in fewer keeps its other channels, and the failed
       ones are NaN. Flux and sharpness are compared within the frame's block,
       so a block that is fainter or blurrier as a whole is kept. A block or a
       channel left without frames is NaN in ``psf_cube_for_postprocessing.fits``,
       and TRAP skips a channel without a finite PSF. No frame left at all stops
       the step with :class:`~spherical.pipeline.flux_vetting.NoUsableFluxFrameError`
       and removes the step's earlier products, so a resumed run does not take an
       old PSF for a finished step. ``flux_frame_vetting`` switches the vetting off.

    3. **Block Detection**: flux_calibration.flux_block_ranges() splits the frames
       into blocks at time gaps longer than 15 times the longer neighbouring DIT,
       ordered by MJD (#193). A block is usually the flux sequence before or after
       the coronagraphic sequence, and every group of frames is one block (#270).
       The blocks are found before the vetting, so rejected frames never split one.

    4. **Frame Normalization**: Each frame is divided by its 3-pixel aperture flux
       relative to the median over its own block (#159). Frame-to-frame
       transparency changes drop out, and the block keeps its median flux.

    5. **Frame Combination**: Frames within each block are combined using the
       specified method (mean or median). Optional first-frame exclusion handles
       potential settling effects after instrument changes.

    6. **Final Assembly**: Results from all blocks are assembled into a single
       array with dimensions (wavelengths, blocks, 57, 57).
    
    Output Dimensions
    -----------------
    ``psf_cube_for_postprocessing.fits``: (wavelengths, blocks, y_pixels, x_pixels)

    - wavelengths: Number of wavelength channels (IFS channels, or the two IRDIS
      filters)
    - blocks: Number of flux blocks from stage 3
    - y_pixels: 57 (PSF stamp height)
    - x_pixels: 57 (PSF stamp width)

    Examples
    --------
    >>> run_flux_psf_calibration(
    ...     converted_dir="/path/to/converted",
    ...     reduction_parameters={
    ...         "flux_combination_method": "mean",
    ...         "exclude_first_flux_frame": True,
    ...         "exclude_first_flux_frame_all": False,
    ...         "logger": logger
    ...     }
    ... )
    """
    logger.info("Starting flux PSF calibration", extra={"step": "flux_psf_calibration", "status": "started"})
    
    # Create directories for outputs
    plot_dir = os.path.join(converted_dir, 'flux_plots/')
    additional_outputs_dir = Path(converted_dir) / 'additional_outputs'
    additional_outputs_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        logger.debug(f"Created plot directory: {plot_dir}")
    logger.debug(f"Created additional outputs directory: {additional_outputs_dir}")
    # Load required files and log their shapes
    wavelengths_path = os.path.join(converted_dir, 'wavelengths.fits')
    flux_cube_path = os.path.join(converted_dir, 'flux_cube.fits')
    frames_info_center_path = os.path.join(converted_dir, 'frames_info_center.csv')
    frames_info_flux_path = os.path.join(converted_dir, 'frames_info_flux.csv')
    for fpath in [wavelengths_path, flux_cube_path, frames_info_center_path, frames_info_flux_path]:
        if not os.path.exists(fpath):
            logger.warning(f"Missing required file: {fpath}", extra={"step": "flux_psf_calibration", "status": "failed"})
    wavelengths = fits.getdata(wavelengths_path)
    flux_cube = fits.getdata(flux_cube_path).astype('float64')
    logger.debug(f"Loaded wavelengths shape: {wavelengths.shape}, flux_cube shape: {flux_cube.shape}")

    # Per-frame inverse variance cube written by preprocess. Used below to (a)
    # keep the centering fit and aperture photometry from being pulled by
    # interpolated values in the PSF core, and (b) drive the Phase 2 Moffat
    # repair of core bad pixels (IRDIS K1 in particular has 3-4 permanent bad
    # pixels at the flux-PSF landing spot).
    #
    # Since charis's variance-propagating resample (charis issue 013), a lenslet
    # charis flags survives as an exact ``ivar == 0`` in every square it touches,
    # so ``ivar == 0`` is now the usable test on IFS. Use ``ratio_threshold=0``:
    # the exact zeros are caught directly, and the soft local-baseline test would
    # only flag good moiré troughs the resample introduces into the ivar.
    flux_ivar_path = os.path.join(converted_dir, 'flux_ivar_cube.fits')
    if os.path.exists(flux_ivar_path):
        flux_ivar_cube = fits.getdata(flux_ivar_path).astype('float64')
        flux_bpm_cube = bad_pixel_mask_from_ivar(flux_ivar_cube, ratio_threshold=0.0)
        logger.debug(
            f"Loaded flux_ivar_cube for BPM: {int(flux_bpm_cube.sum())} bad "
            f"pixels across the full cube "
            f"({int((flux_ivar_cube == 0).sum())} of them exact zeros)"
        )
    else:
        flux_ivar_cube = None
        flux_bpm_cube = None
        logger.warning(
            "flux_ivar_cube.fits not found — centering and aperture photometry "
            "will not mask bad pixels for this observation, and Phase 2 Moffat "
            "core repair will be skipped.",
            extra={"step": "flux_psf_calibration", "status": "bpm_missing"},
        )

    # Instrument-aware pixel scale for loD-based Gaussian init. A 2-channel
    # cube marks IRDIS DBI (12.25 mas/px); anything larger is IFS (7.46 mas/px).
    is_irdis = flux_cube.shape[0] == 2
    pixel_scale_mas = 12.25 if is_irdis else 7.46
    frames_info = {}
    frames_info['CENTER'] = pd.read_csv(frames_info_center_path)
    frames_info['FLUX'] = pd.read_csv(frames_info_flux_path)
    logger.debug(f"Loaded frames_info: CENTER shape {frames_info['CENTER'].shape}, FLUX shape {frames_info['FLUX'].shape}")
    
    # First, compute guess positions for all frames
    logger.debug("Computing initial guess positions for all flux frames")
    guess_positions_yx = []
    # Persistence from the CORO sequence is an on-sky feature, so the mask
    # radius is angular. 224 mas reproduces the historical 30 px on IFS
    # (7.46 mas/px) exactly and gives IRDIS 18 px instead of the 30 px it was
    # inheriting from the IFS literal -- 368 mas, over-masking by 1.6x.
    CORONAGRAPH_MASK_RADIUS_MAS = 224.0
    coronagraph_mask_radius = int(round(CORONAGRAPH_MASK_RADIUS_MAS / pixel_scale_mas))
    coronagraph_center_xy = coronagraph_center_from_disk(
        converted_dir, logger,
        irdis_filter_comb=(frames_info['CENTER']['INS COMB IFLT'].iloc[0]
                           if is_irdis else None),
    )
    # IRDIS FLUX is never cropped, so its two channels keep the cross-channel
    # offset and each needs its own guess (#170). IFS channels share one PSF
    # position, and the wavelength median gives the guess more S/N.
    guess_kwargs = {"exclude_edge_pixels": 30, "coronagraph_center_xy": coronagraph_center_xy,
                    "coronagraph_mask_radius": coronagraph_mask_radius}
    channel_offset_xy = None
    if is_irdis:
        guess_kwargs["smooth_sigma_px"] = irdis_guess_smoothing_px(wavelengths, pixel_scale_mas)
        channel_offset_xy = read_channel_offset(additional_outputs_dir, logger)
    disagreeing = np.zeros(2, dtype=int)
    for frame_number in range(flux_cube.shape[1]):
        data = flux_cube[:, frame_number]
        bpm = None if flux_bpm_cube is None else flux_bpm_cube[:, frame_number]
        if not is_irdis:
            guess_yx = guess_position_psf(cube=data, bad_pixel_mask=bpm, **guess_kwargs)
        elif channel_offset_xy is None:
            guess_yx = guess_positions_per_channel(cube=data, bad_pixel_mask=bpm, **guess_kwargs)
        else:
            guess_yx, independent = guess_positions_joint(data, channel_offset_xy, bad_pixel_mask=bpm, **guess_kwargs)
            disagreeing += np.hypot(*(np.array(guess_yx) - np.array(independent)).T) > GUESS_DISAGREEMENT_PX
        guess_positions_yx.append(guess_yx)
    for channel in np.flatnonzero(disagreeing):
        logger.warning(
            f"The brightest pixel of IRDIS channel {channel} lies more than {GUESS_DISAGREEMENT_PX:.0f} px from the "
            f"star position given by the other channel and the cross-channel offset in {disagreeing[channel]} of "
            f"{flux_cube.shape[1]} FLUX frames; the step uses the joint position there.",
            extra={"step": "flux_psf_calibration", "status": "flux_guess_reseeded"},
        )
    
    # Replace unreliable first frame guess with second frame guess
    if len(guess_positions_yx) >= 2:
        logger.debug(f"Replacing unreliable first frame guess position {guess_positions_yx[0]} with second frame position {guess_positions_yx[1]}")
        guess_positions_yx[0] = guess_positions_yx[1]
    else:
        logger.warning("Less than 2 flux frames available, cannot replace first frame guess position")
    
    logger.debug(f"Computed guess positions for {len(guess_positions_yx)} frames")

    selection = reduction_parameters['flux_cube_selection_irdis' if is_irdis else 'flux_cube_selection_ifs']
    exposure_level = (np.asarray(frames_info['FLUX']['DET SEQ1 DIT'], dtype=float)
                      * build_nd_attenuation(frames_info['FLUX'], wavelengths).mean(axis=0))
    # CENTER frames bracket the CORO sequence, and in waffle mode they are the
    # science frames, so their MJD span locates the science sequence either way.
    center_mjd = frames_info['CENTER']['MJD']
    (
        flux_cube, flux_ivar_cube, flux_bpm_cube,
        frames_info['FLUX'], guess_positions_yx, selection_report,
    ) = apply_flux_cube_selection(
        flux_cube, flux_ivar_cube, flux_bpm_cube, frames_info['FLUX'], guess_positions_yx,
        exposure_level=exposure_level,
        science_mid_mjd=(center_mjd.min() + center_mjd.max()) / 2,
        threshold_adu=reduction_parameters['flux_saturation_adu'],
        nonlinearity_adu=reduction_parameters['flux_nonlinearity_adu'],
        selection=selection,
        frame_indices=reduction_parameters.get('flux_frame_indices'),
    )
    selection_report.to_csv(additional_outputs_dir / 'flux_cube_selection.csv', index=False)

    kept = selection_report['kept']
    kept_files = selection_report.loc[kept, 'origfile']
    if (kept & selection_report['saturated']).any():
        logger.warning(
            "The flux PSF is built from a cube whose core reaches flux_saturation_adu; "
            "the PSF reference and any photometry calibrated on it are unreliable.",
            extra={"step": "flux_psf_calibration", "status": "flux_saturated"},
        )
    nonlinear = selection_report.loc[kept & selection_report['nonlinear'], 'origfile'].tolist()
    if nonlinear:
        logger.warning(
            f"Flux PSF may be affected by non-linearity in {len(nonlinear)} of {len(kept_files)} "
            f"kept cubes (core peak >= flux_nonlinearity_adu): {nonlinear}",
            extra={"step": "flux_psf_calibration", "status": "flux_nonlinear"},
        )
    if not kept.all():
        logger.info(
            f"Flux cube selection {selection!r} kept {len(kept_files)}/{len(kept)} cubes: {kept_files.tolist()}",
            extra={"step": "flux_psf_calibration", "status": "cubes_selected"},
        )
    
    # Now compute flux centers using pre-computed guess positions.
    #
    # We call ``star_centers_from_PSF_img_cube`` twice: once with the per-frame
    # bad-pixel mask applied (this becomes the canonical result), and once
    # without any mask so we can log the flux-amplitude delta and raise a
    # warning if the interpolated bad-pixel values were biasing the fit by
    # more than 10% on any channel. Even for well-behaved data this is nearly
    # free (a single 2D Gaussian fit per frame per channel), and it surfaces
    # the "permanent bad pixel in PSF core" case (see IRDIS K1 for 51 Eri) as
    # an explicit log line instead of a silent bias.
    flux_centers = []
    flux_amplitudes = []
    flux_amplitudes_unmasked = []
    for frame_number in range(flux_cube.shape[1]):
        data = flux_cube[:, frame_number]
        per_frame_mask = flux_bpm_cube[:, frame_number] if flux_bpm_cube is not None else None
        flux_center, flux_amplitude = star_centers_from_PSF_img_cube(
            cube=data.copy(),
            wave=wavelengths,
            pixel=pixel_scale_mas,
            guess_center_yx=guess_positions_yx[frame_number],
            fit_background=True,
            fit_symmetric_gaussian=True,
            mask_deviating=False,
            deviation_threshold=0.8,
            exclude_edge_pixels=30,
            coronagraph_center_xy=coronagraph_center_xy,
            coronagraph_mask_radius=coronagraph_mask_radius,
            mask=per_frame_mask,
            save_path=None,
            verbose=False,
            logger=logger,
            frame_number=frame_number,
        )
        flux_centers.append(flux_center)
        flux_amplitudes.append(flux_amplitude)

        if per_frame_mask is not None:
            _, flux_amplitude_unmasked = star_centers_from_PSF_img_cube(
                cube=data.copy(),
                wave=wavelengths,
                pixel=pixel_scale_mas,
                guess_center_yx=guess_positions_yx[frame_number],
                fit_background=True,
                fit_symmetric_gaussian=True,
                mask_deviating=False,
                deviation_threshold=0.8,
                exclude_edge_pixels=30,
                coronagraph_center_xy=coronagraph_center_xy,
                coronagraph_mask_radius=coronagraph_mask_radius,
                mask=None,
                save_path=None,
                verbose=False,
                logger=logger,
                frame_number=frame_number,
            )
            flux_amplitudes_unmasked.append(flux_amplitude_unmasked)

    # Replace unreliable first frame centers with second frame centers
    if len(flux_centers) >= 2:
        logger.debug("Replacing unreliable first flux frame centers with second frame centers")
        # flux_centers[0] contains centers for all wavelengths in first frame
        # flux_centers[1] contains centers for all wavelengths in second frame
        flux_centers[0] = flux_centers[1].copy()
        logger.debug("Successfully replaced first frame centers")
    else:
        logger.warning("Less than 2 flux frames available, cannot replace first frame centers")

    # Continue with existing array transformations
    flux_centers = np.expand_dims(np.swapaxes(np.array(flux_centers), 0, 1), axis=2)
    flux_amplitudes = np.swapaxes(np.array(flux_amplitudes), 0, 1)
    logger.debug(f"Extracted flux_centers shape: {flux_centers.shape}, flux_amplitudes shape: {flux_amplitudes.shape}")

    # Diagnostic: BPM masking effect on the fitted amplitude, per channel.
    # We compare the masked (canonical) amplitudes to the unmasked ones and
    # flag any channel where any frame changed by more than 10% — that would
    # indicate a bad pixel is materially contaminating the star-flux estimate,
    # which propagates as a same-magnitude bias in the companion contrast.
    _WARN_THRESHOLD = 0.10
    if flux_bpm_cube is not None and flux_amplitudes_unmasked:
        amps_unmasked = np.swapaxes(np.array(flux_amplitudes_unmasked), 0, 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = (flux_amplitudes - amps_unmasked) / amps_unmasked
        n_wave = flux_amplitudes.shape[0]
        for ch in range(n_wave):
            ch_delta = delta[ch]
            finite = np.isfinite(ch_delta)
            if not finite.any():
                continue
            mean_delta = float(np.nanmean(ch_delta[finite]))
            max_abs_delta = float(np.nanmax(np.abs(ch_delta[finite])))
            offending = int(np.sum(np.abs(ch_delta[finite]) > _WARN_THRESHOLD))
            msg = (
                f"Flux-PSF amplitude BPM-masking delta on channel {ch}: "
                f"mean={100*mean_delta:+.2f}%, max|Δ|={100*max_abs_delta:.2f}%, "
                f"frames_over_{int(_WARN_THRESHOLD*100)}pct={offending}/{ch_delta.size}"
            )
            if offending > 0:
                logger.warning(
                    msg,
                    extra={
                        "step": "flux_psf_calibration",
                        "status": "bpm_bias_warning",
                        "channel": ch,
                        "mean_delta": mean_delta,
                        "max_abs_delta": max_abs_delta,
                        "n_frames_over_threshold": offending,
                    },
                )
            else:
                logger.info(
                    msg,
                    extra={
                        "step": "flux_psf_calibration",
                        "status": "bpm_bias_ok",
                        "channel": ch,
                        "mean_delta": mean_delta,
                        "max_abs_delta": max_abs_delta,
                    },
                )

    # Per-frame BPM count inside a 3-px core around each fitted center. Even
    # sub-threshold biases matter cumulatively; a permanent bad pixel in the
    # core should be visible in the log.
    if flux_bpm_cube is not None:
        core_radius = 3
        n_wave, n_frames = flux_amplitudes.shape
        _, _, ny, nx = flux_bpm_cube.shape
        yy, xx = np.indices((2 * core_radius + 1, 2 * core_radius + 1))
        core_disk = ((yy - core_radius) ** 2 + (xx - core_radius) ** 2) <= core_radius ** 2
        n_frames_with_core_bp = np.zeros(n_wave, dtype=int)
        for ch in range(n_wave):
            for f in range(n_frames):
                cx, cy = flux_centers[ch, f, 0]
                if not (np.isfinite(cx) and np.isfinite(cy)):
                    continue
                cxi, cyi = int(round(float(cx))), int(round(float(cy)))
                y0, x0 = cyi - core_radius, cxi - core_radius
                y1, x1 = y0 + 2 * core_radius + 1, x0 + 2 * core_radius + 1
                if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
                    continue
                bp_stamp = flux_bpm_cube[ch, f, y0:y1, x0:x1]
                bp_in_core = int(np.sum(bp_stamp & core_disk))
                if bp_in_core > 0:
                    n_frames_with_core_bp[ch] += 1
        for ch in range(n_wave):
            if n_frames_with_core_bp[ch] > 0:
                logger.warning(
                    f"Channel {ch}: {n_frames_with_core_bp[ch]}/{n_frames} flux "
                    f"frames have >=1 bad pixel inside a {core_radius}-px core "
                    f"around the fitted PSF center. Interpolated values in this "
                    f"region bias the star-flux estimate; Phase 2 Moffat repair "
                    f"is recommended.",
                    extra={
                        "step": "flux_psf_calibration",
                        "status": "core_bpm_warning",
                        "channel": ch,
                        "n_frames_with_core_bp": int(n_frames_with_core_bp[ch]),
                        "core_radius_px": core_radius,
                    },
                )

    # ------------------------------------------------------------------
    # Phase 2: Moffat-based bad-pixel repair in the PSF core.
    #
    # Runs on the raw detector frame at the nearest-pixel PSF center (i.e.
    # before the subpixel shift that would smear a bad pixel across a blob),
    # per-channel-skipped when no core BPM is present, saves both the
    # repaired flux_cube and (later) an unrepaired sibling of the combined
    # PSF cube so the effect is auditable.
    #
    # The photometry keeps the core pixels the repair treated (#269): the
    # Moffat value where the fit succeeded, the upstream interpolation where
    # it was skipped. Masking them drops their flux, which reads up to ~20%
    # low, while the interpolation alone stays within a few percent.
    # ------------------------------------------------------------------
    _TELESCOPE_D_M = 7.99
    _REPAIR_RESIDUAL_RMS_THRESHOLD = 0.10

    if (
        flux_bpm_cube is not None
        and flux_ivar_cube is not None
        and 'n_frames_with_core_bp' in dir()
    ):
        n_wave = flux_cube.shape[0]
        n_frames = flux_cube.shape[1]

        # Per-channel Airy first-null radius in px (physical core scale).
        # Also log the current 3-px normalization aperture in lambda/D units
        # so the wavelength-dependent geometry is visible in the log.
        per_channel_core_radius = []
        for ch in range(n_wave):
            lam_m = float(wavelengths[ch]) * 1e-9
            lam_over_D_px = (lam_m / _TELESCOPE_D_M) * 206265.0 * 1000.0 / pixel_scale_mas
            core_r = 1.22 * lam_over_D_px
            per_channel_core_radius.append(core_r)
            logger.info(
                f"Channel {ch}: lambda={float(wavelengths[ch]):.1f} nm, "
                f"lambda/D={lam_over_D_px:.2f} px, Airy null "
                f"(repair-core radius)={core_r:.2f} px, "
                f"3-px normalization aperture at {3.0 / lam_over_D_px:.2f} lambda/D",
                extra={
                    "step": "flux_psf_calibration",
                    "status": "core_geometry",
                    "channel": ch,
                    "lambda_nm": float(wavelengths[ch]),
                    "lambda_over_D_px": lam_over_D_px,
                    "core_radius_px": core_r,
                },
            )

        channels_to_repair = [
            ch for ch in range(n_wave) if int(n_frames_with_core_bp[ch]) > 0
        ]

        if not channels_to_repair:
            logger.info(
                "No PSF-core bad pixels detected on any channel; Phase 2 Moffat "
                "repair skipped for this observation.",
                extra={"step": "flux_psf_calibration", "status": "phase2_skipped_clean"},
            )
            flux_cube_for_stamps = flux_cube
            flux_cube_repaired = None
            flux_ivar_repaired = None
            photometry_bpm_cube = flux_bpm_cube
            repair_records: list = []
        else:
            logger.info(
                f"Running Phase 2 Moffat repair on channels {channels_to_repair}",
                extra={"step": "flux_psf_calibration", "status": "phase2_start"},
            )
            flux_cube_repaired = flux_cube.copy()
            flux_ivar_repaired = flux_ivar_cube.copy()
            photometry_bpm_cube = flux_bpm_cube.copy()
            repair_records = []
            _, _, ny_full, nx_full = flux_cube.shape
            for ch in channels_to_repair:
                core_r = per_channel_core_radius[ch]
                # Fit window slightly larger than the core so Moffat's gamma/alpha
                # (shape parameters) are constrained by pixels beyond the peak.
                wr = max(int(np.ceil(2 * core_r)), 6)
                for f in range(n_frames):
                    cx, cy = flux_centers[ch, f, 0]
                    if not (np.isfinite(cx) and np.isfinite(cy)):
                        continue
                    cxi = int(round(float(cx)))
                    cyi = int(round(float(cy)))
                    y0 = cyi - wr
                    y1 = cyi + wr + 1
                    x0 = cxi - wr
                    x1 = cxi + wr + 1
                    if y0 < 0 or x0 < 0 or y1 > ny_full or x1 > nx_full:
                        continue
                    win = flux_cube_repaired[ch, f, y0:y1, x0:x1]
                    ivar_win = flux_ivar_repaired[ch, f, y0:y1, x0:x1]
                    res = repair_psf_core(
                        window=win,
                        ivar_window=ivar_win,
                        center_xy_in_window=(float(cx) - x0, float(cy) - y0),
                        core_radius_px=core_r,
                        residual_rms_frac_threshold=_REPAIR_RESIDUAL_RMS_THRESHOLD,
                        bad_mask=flux_bpm_cube[ch, f, y0:y1, x0:x1],
                    )
                    repair_records.append({
                        "channel": ch,
                        "frame": f,
                        "status": res.status,
                        "n_repaired": res.n_repaired,
                        "residual_rms_frac": res.residual_rms_frac,
                        "amplitude": res.amplitude,
                    })
                    photometry_bpm_cube[ch, f, y0:y1, x0:x1] &= ~res.bad_in_core
                    if res.status == "repaired":
                        flux_cube_repaired[ch, f, y0:y1, x0:x1] = res.window_out
                        flux_ivar_repaired[ch, f, y0:y1, x0:x1] = res.ivar_out
                    elif res.bad_in_core.any():
                        fit_note = (
                            f" (residual RMS = {100*res.residual_rms_frac:.2f}% of amplitude, "
                            f"limit {int(100*_REPAIR_RESIDUAL_RMS_THRESHOLD)}%)"
                            if np.isfinite(res.residual_rms_frac) else ""
                        )
                        logger.warning(
                            f"Channel {ch}, frame {f}: Moffat repair {res.status}{fit_note}. "
                            f"Falling back to the upstream interpolated value at "
                            f"the {int(res.bad_in_core.sum())} core bad pixels, "
                            f"which also enter the flux photometry.",
                            extra={
                                "step": "flux_psf_calibration",
                                "status": "phase2_fallback",
                                "channel": ch,
                                "frame": f,
                                "repair_status": res.status,
                                "residual_rms_frac": float(res.residual_rms_frac),
                            },
                        )

            for ch in channels_to_repair:
                ch_records = [r for r in repair_records if r["channel"] == ch]
                n_rep = sum(1 for r in ch_records if r["status"] == "repaired")
                n_skip = sum(1 for r in ch_records if r["status"] != "repaired")
                n_pix = sum(int(r["n_repaired"]) for r in ch_records)
                rms_ok = [r["residual_rms_frac"] for r in ch_records
                          if r["status"] == "repaired" and np.isfinite(r["residual_rms_frac"])]
                median_rms = float(np.median(rms_ok)) if rms_ok else float("nan")
                logger.info(
                    f"Channel {ch}: Moffat repair applied to {n_rep}/{len(ch_records)} "
                    f"frames ({n_pix} pixels total); {n_skip} frames skipped; "
                    f"median residual RMS = {100*median_rms:.2f}% of amplitude",
                    extra={
                        "step": "flux_psf_calibration",
                        "status": "phase2_summary",
                        "channel": ch,
                        "n_frames_repaired": n_rep,
                        "n_frames_skipped": n_skip,
                        "n_pixels_repaired": n_pix,
                        "median_residual_rms_frac": median_rms,
                    },
                )

            fits.writeto(additional_outputs_dir / 'flux_cube_repaired.fits',
                         flux_cube_repaired.astype('float32'), overwrite=True)
            fits.writeto(additional_outputs_dir / 'flux_ivar_repaired.fits',
                         flux_ivar_repaired.astype('float32'), overwrite=True)
            pd.DataFrame(repair_records).to_csv(
                additional_outputs_dir / 'flux_psf_repair_log.csv', index=False)
            flux_cube_for_stamps = flux_cube_repaired
    else:
        flux_cube_for_stamps = flux_cube
        flux_cube_repaired = None
        flux_ivar_repaired = None
        photometry_bpm_cube = flux_bpm_cube
        repair_records = []

    flux_stamps = toolbox.extract_satellite_spot_stamps(
        flux_cube_for_stamps, flux_centers, stamp_size=57, shift_order=3, plot=False, logger=logger)
    logger.debug(f"Extracted flux_stamps shape: {flux_stamps.shape}")
    # Sibling stamps from the raw (unrepaired) flux cube, so we can build a
    # matched unrepaired psf_cube for direct comparison.
    if flux_cube_repaired is not None:
        flux_stamps_unrepaired = toolbox.extract_satellite_spot_stamps(
            flux_cube, flux_centers, stamp_size=57, shift_order=3, plot=False, logger=logger)
    else:
        flux_stamps_unrepaired = None

    flux_bpm_stamps = None if photometry_bpm_cube is None else bad_pixel_stamps(photometry_bpm_cube, flux_centers)
    unique_nds = frames_info['FLUX']['INS4 FILT2 NAME'].unique()
    if len(unique_nds) > 1:
        logger.warning(
            f"Non-unique ND filters in flux sequence: {list(unique_nds)}. "
            "Applying per-frame attenuation.",
            extra={"step": "flux_psf_calibration", "status": "mixed_nd"},
        )
    attenuation = build_nd_attenuation(frames_info['FLUX'], wavelengths)
    dits_flux = np.array(frames_info['FLUX']['DET SEQ1 DIT'])
    dits_center = np.array(frames_info['CENTER']['DET SEQ1 DIT'])
    unique_dits_center, unique_dits_center_counts = np.unique(dits_center, return_counts=True)
    if len(unique_dits_center) == 1:
        dits_factor = unique_dits_center[0] / dits_flux
        most_common_dit_center = unique_dits_center[0]
    else:
        most_common_dit_center = unique_dits_center[np.argmax(unique_dits_center_counts)]
        dits_factor = most_common_dit_center / dits_flux
    flux_stamps_calibrated = flux_stamps * dits_factor[None, :, None, None]
    flux_stamps_calibrated = flux_stamps_calibrated / attenuation[:, :, np.newaxis, np.newaxis]
    # Off-frame padding and interior bad lenslets are NaN in the stamps. Fold
    # them into the mask the photometry already honours, so they are excluded
    # from the aperture sum and the background statistics instead of poisoning
    # them (NaN inside the aperture otherwise returns NaN flux).
    stamp_nan = ~np.isfinite(flux_stamps_calibrated)
    if stamp_nan.any():
        flux_bpm_stamps = (stamp_nan if flux_bpm_stamps is None
                           else np.logical_or(flux_bpm_stamps, stamp_nan))
    flux_photometry = flux_calibration.get_aperture_photometry(
        flux_stamps_calibrated, aperture_radius_range=[1, 15],
        bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
        bad_pixel_mask=flux_bpm_stamps)

    _, indices_of_discontinuity = flux_calibration.get_flux_calibration_indices(
        frames_info['CENTER'], frames_info['FLUX'])
    original_blocks = flux_calibration.flux_block_ranges(indices_of_discontinuity, flux_stamps.shape[1])
    block = block_labels(original_blocks, flux_stamps.shape[1])
    opens_block = np.zeros(flux_stamps.shape[1], dtype=bool)
    opens_block[[start for start, _ in original_blocks]] = True
    if reduction_parameters.get('flux_frame_vetting', PreprocConfig().flux_frame_vetting):
        vetting = vet_flux_frames(
            flux_photometry, flux_centers[:, :, 0], wavelengths, pixel_scale_mas,
            origfile=frames_info['FLUX']['ORIGFILE'].to_numpy(), block=block,
            thresholds=FluxVettingThresholds.from_parameters(reduction_parameters, 'IRDIS' if is_irdis else 'IFS'))
        vetting.table.merge(
            frames_info['FLUX'][['DIT INDEX', 'MJD']].rename_axis('frame').reset_index(), on='frame', how='left',
        ).to_csv(additional_outputs_dir / 'flux_frame_vetting.csv', index=False)
        try:
            report_flux_vetting(vetting, reduction_parameters.get(
                'flux_vetting_warn_kept_fraction', PreprocConfig().flux_vetting_warn_kept_fraction), logger)
        except NoUsableFluxFrameError:
            # psf_cube_for_postprocessing.fits marks the step as done, so an earlier
            # run's products would otherwise pass for the result of this one.
            for name in STEP_PRODUCTS:
                (Path(converted_dir) / name).unlink(missing_ok=True)
            raise

        keep, masked = vetting.frame_kept, ~vetting.channel_kept[:, vetting.frame_kept]
        frames_info['FLUX'] = frames_info['FLUX'].loc[keep].reset_index(drop=True)
        flux_centers, flux_amplitudes = flux_centers[:, keep], flux_amplitudes[:, keep]
        attenuation, dits_factor = attenuation[:, keep], dits_factor[keep]
        flux_stamps = flux_stamps[:, keep]
        flux_stamps_calibrated = flux_stamps_calibrated[:, keep]
        flux_stamps_calibrated[masked] = np.nan
        if flux_stamps_unrepaired is not None:
            flux_stamps_unrepaired = flux_stamps_unrepaired[:, keep]
        if flux_bpm_cube is not None:
            flux_bpm_cube = flux_bpm_cube[:, keep]
        flux_photometry = subset_photometry(flux_photometry, keep, vetting.channel_kept)
        # Blocks keep the time grouping of the frames before the vetting. Recomputing
        # it would split a block at the gap left by a run of rejected frames, and the
        # survivors would form a block of their own with the same weight in run_trap.
        block, opens_block = block[keep], opens_block[keep]
        indices_of_discontinuity = np.flatnonzero(np.diff(block) != 0)
    else:
        masked = None
        logger.info("FLUX frame vetting is switched off.",
                    extra={"step": "flux_psf_calibration", "status": "flux_vetting_off"})

    # Every frame-axis product below shares the axis of frames_info_flux_selected.csv.
    # converted/flux_cube.fits and frames_info_flux.csv stay the complete
    # pre-processing record, so a re-run with another selection still works.
    frames_info['FLUX'].to_csv(Path(converted_dir) / 'frames_info_flux_selected.csv', index=False)
    fits.writeto(additional_outputs_dir / 'flux_centers.fits', flux_centers, overwrite=True)
    fits.writeto(additional_outputs_dir / 'flux_gauss_amplitudes.fits', flux_amplitudes, overwrite=True)
    fits.writeto(additional_outputs_dir / 'flux_stamps_uncalibrated.fits',
                 flux_stamps.astype('float32'), overwrite=True)
    fits.writeto(additional_outputs_dir / 'nd_attenuation.fits', attenuation, overwrite=True)
    fits.writeto(additional_outputs_dir / 'flux_stamps_dit_nd_calibrated.fits',
                 flux_stamps_calibrated, overwrite=True)
    filehandler = open(additional_outputs_dir / 'flux_photometry.obj', 'wb')
    pickle.dump(flux_photometry, filehandler)
    filehandler.close()
    fits.writeto(os.path.join(converted_dir, 'flux_amplitude_calibrated.fits'),
                 flux_photometry['psf_flux_bg_corr_all'], overwrite=True)
    fits.writeto(additional_outputs_dir / 'flux_snr.fits',
                 flux_photometry['snr_all'], overwrite=True)
    
    plt.close()
    plt.plot(flux_photometry['aperture_sizes'], flux_photometry['snr_all'][:, :, 0])
    plt.xlabel('Aperture Size (pix)')
    plt.ylabel('BG limited SNR')
    plt.savefig(additional_outputs_dir / 'Flux_PSF_aperture_SNR.png')
    plt.close()
    bg_sub_flux_stamps_calibrated = flux_stamps_calibrated - flux_photometry['psf_bg_counts_all'][:, :, None, None]
    fits.writeto(additional_outputs_dir / 'flux_stamps_calibrated_bg_corrected.fits',
                 bg_sub_flux_stamps_calibrated.astype('float32'), overwrite=True)
    flux_calibration_indices, _ = flux_calibration.get_flux_calibration_indices(
        frames_info['CENTER'], frames_info['FLUX'], indices_of_discontinuity=indices_of_discontinuity)
    flux_calibration_indices.to_csv(os.path.join(converted_dir, 'flux_calibration_indices.csv'))
    indices_of_discontinuity.tofile(additional_outputs_dir / 'indices_of_discontinuity.csv', sep=',')
    flux_blocks = flux_calibration.flux_block_ranges(indices_of_discontinuity, flux_stamps.shape[1])
    flux_calibration_frames = []
    if reduction_parameters['flux_combination_method'] == 'mean':
        comb_func = np.nanmean
    elif reduction_parameters['flux_combination_method'] == 'median':
        comb_func = np.nanmedian
    else:
        logger.warning('Unknown flux combination method.', extra={"step": "flux_psf_calibration", "status": "failed"})
        raise ValueError('Unknown flux combination method.')
    for lower_index, upper_range in flux_blocks:
        lower_index_frame_combine = first_combined_frame(
            block[lower_index], upper_range - lower_index, reduction_parameters, opens_block[lower_index])
        normalization_values = block_normalization(
            flux_photometry['psf_flux_bg_corr_all'][2][:, lower_index:upper_range],
            lower_index_frame_combine)
        # A combined frame far from its block median is a bad frame or a real
        # transparency change; either way the photometry plots deserve a look.
        deviation = np.nanmedian(np.abs(normalization_values[:, lower_index_frame_combine:] - 1), axis=0)
        deviant = np.nonzero(deviation > 0.2)[0] + lower_index + lower_index_frame_combine
        if deviant.size:
            logger.warning(
                f"Flux frames deviate by more than 20% from their block's median aperture flux: "
                f"{frames_info['FLUX'].loc[deviant, ['ORIGFILE', 'DIT INDEX']].values.tolist()}",
                extra={"step": "flux_psf_calibration", "status": "flux_frame_outlier"},
            )
        flux_calibration_frame = bg_sub_flux_stamps_calibrated[:, lower_index:upper_range] / normalization_values[:, :, None, None]
        flux_calibration_frame = comb_func(flux_calibration_frame[:, lower_index_frame_combine:], axis=1)
        flux_calibration_frames.append(flux_calibration_frame)
    flux_calibration_frames = np.array(flux_calibration_frames)
    flux_calibration_frames = np.swapaxes(flux_calibration_frames, 0, 1)
    fits.writeto(os.path.join(converted_dir, 'psf_cube_for_postprocessing.fits'),
                 finalize_psf_cube(flux_calibration_frames, logger).astype('float32'),
                 overwrite=True)

    # Diagnostic sibling: replay the same DIT/ND + BG-sub + normalize + combine
    # pipeline on the *raw* (unrepaired) stamps, so the only difference vs the
    # canonical psf_cube_for_postprocessing.fits is Moffat repair vs the
    # upstream neighbour-interpolated bad-pixel values. Only produced when
    # Phase 2 actually ran.
    if flux_stamps_unrepaired is not None:
        flux_stamps_calibrated_unrepaired = (
            flux_stamps_unrepaired * dits_factor[None, :, None, None]
            / attenuation[:, :, np.newaxis, np.newaxis]
        )
        flux_photometry_unrepaired = flux_calibration.get_aperture_photometry(
            flux_stamps_calibrated_unrepaired, aperture_radius_range=[1, 15],
            bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
            bad_pixel_mask=(bad_pixel_stamps(flux_bpm_cube, flux_centers)
                            | ~np.isfinite(flux_stamps_calibrated_unrepaired)))
        if masked is not None:
            flux_stamps_calibrated_unrepaired[masked] = np.nan
            flux_photometry_unrepaired['psf_flux_bg_corr_all'][..., masked] = np.nan
        bg_sub_stamps_unrepaired = (
            flux_stamps_calibrated_unrepaired
            - flux_photometry_unrepaired['psf_bg_counts_all'][:, :, None, None]
        )
        unrepaired_frames = []
        for lower_index, upper_range in flux_blocks:
            lower_index_frame_combine = first_combined_frame(
                block[lower_index], upper_range - lower_index, reduction_parameters, opens_block[lower_index])
            norm_u = block_normalization(
                flux_photometry_unrepaired['psf_flux_bg_corr_all'][2][:, lower_index:upper_range],
                lower_index_frame_combine)
            frame_u = bg_sub_stamps_unrepaired[:, lower_index:upper_range] / norm_u[:, :, None, None]
            frame_u = comb_func(frame_u[:, lower_index_frame_combine:], axis=1)
            unrepaired_frames.append(frame_u)
        unrepaired_cube = np.swapaxes(np.array(unrepaired_frames), 0, 1)
        fits.writeto(os.path.join(converted_dir, 'psf_cube_for_postprocessing_unrepaired.fits'),
                     finalize_psf_cube(unrepaired_cube, logger).astype('float32'),
                     overwrite=True)
        # Log peak-amplitude delta per channel so the effect is immediately
        # visible without having to diff the FITS files.
        for ch in range(flux_calibration_frames.shape[0]):
            peak_rep = float(np.nanmax(flux_calibration_frames[ch]))
            peak_unr = float(np.nanmax(unrepaired_cube[ch]))
            if peak_unr > 0:
                delta = 100.0 * (peak_rep - peak_unr) / peak_unr
            else:
                delta = float("nan")
            logger.info(
                f"Channel {ch}: PSF-cube peak (repaired vs unrepaired) = "
                f"{peak_rep:.3e} vs {peak_unr:.3e}  (Δ={delta:+.2f}%)",
                extra={"step": "flux_psf_calibration", "status": "phase2_peak_delta",
                       "channel": ch, "peak_repaired": peak_rep,
                       "peak_unrepaired": peak_unr, "peak_delta_pct": delta},
            )

    logger.info("Step finished", extra={"step": "flux_psf_calibration", "status": "success"})
