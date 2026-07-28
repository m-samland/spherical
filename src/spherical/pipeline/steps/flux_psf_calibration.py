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
from pathlib import Path
from typing import Dict

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from spherical.pipeline import flux_calibration, toolbox, transmission
from spherical.pipeline.logging_utils import optional_logger
from spherical.pipeline.psf_repair import repair_psf_core
from spherical.pipeline.steps.find_star import guess_position_psf, star_centers_from_PSF_img_cube


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
    -------------------
    From previous steps:
    - converted_dir/wavelengths.fits
        Wavelength array for the data cube
    - converted_dir/flux_cube.fits
        Master cube of flux data
    - converted_dir/frames_info_center.csv
        Frame information for center data
    - converted_dir/frames_info_flux.csv
        Frame information for flux data

    Generated Output Files
    ---------------------
    In converted_dir:
    - flux_amplitude_calibrated.fits
        Calibrated flux amplitudes
    - flux_calibration_indices.csv
        Frame indices for flux calibration
    - psf_cube_for_postprocessing.fits
        Combined calibrated flux PSF frames

    In converted_dir/additional_outputs/:
    - flux_centers.fits
        Fitted centers of flux PSFs
    - flux_gauss_amplitudes.fits
        Gaussian fit amplitudes for flux PSFs
    - flux_stamps_uncalibrated.fits
        Raw extracted flux PSF stamps
    - nd_attenuation.fits
        ND filter transmission correction
    - center_frame_dit_adjustment_factors.fits
        DIT normalization factors for center frames
    - flux_stamps_dit_nd_calibrated.fits
        DIT and ND-corrected flux stamps
    - flux_photometry.obj
        Pickled photometry results
    - flux_snr.fits
        Signal-to-noise ratios
    - flux_stamps_calibrated_bg_corrected.fits
        Background-subtracted calibrated stamps
    - indices_of_discontinuity.csv
        Indices where flux calibration changes
    - Flux_PSF_aperture_SNR.png
        Plot of SNR vs aperture size

    Parameters
    ----------
    converted_dir : str
        Directory containing the input files and where outputs will be written.
    reduction_parameters : dict
        Reduction parameters dict, must contain:
        - flux_combination_method: str
            Method to combine flux frames ('mean' or 'median')
        - exclude_first_flux_frame: bool
            Whether to exclude first frame in first sequence
        - exclude_first_flux_frame_all: bool
            Whether to exclude first frame in all sequences
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
        * ND filter transmission correction
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
    
    Master Flux Calibrated PSF Frames Generation
    -------------------------------------------
    The psf_cube_for_postprocessing.fits file is generated through a sophisticated 
    flux calibration and frame combination process:
    
    1. **Sequence Detection**: Uses flux_calibration.get_flux_calibration_indices() to 
       identify temporal segments where observing conditions are stable (e.g., same ND 
       filter, continuous observing).
    
    2. **Frame Normalization**: Within each sequence, frames are normalized using 
       3-pixel aperture photometry results. Each frame is divided by the sequence 
       mean to correct for temporal variations in flux.
    
    3. **Frame Combination**: Frames within each sequence are combined using the 
       specified method (mean or median). Optional first-frame exclusion handles 
       potential settling effects after instrument changes.
    
    4. **Final Assembly**: Results from all sequences are assembled into a single 
       array with dimensions (wavelengths, sequences, 57, 57).
    
    Output Dimensions
    ----------------
    psf_cube_for_postprocessing.fits: (wavelengths, sequences, y_pixels, x_pixels)
    - wavelengths: Number of IFS wavelength channels
    - sequences: Number of flux calibration sequences (temporal segments)
    - y_pixels: 57 (PSF stamp height)
    - x_pixels: 57 (PSF stamp width)
    
    Each sequence represents a temporally stable observing period with improved 
    SNR through frame combination and normalized flux calibration.

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

    # Per-frame inverse variance cube written by preprocess: ``ivar == 0``
    # marks pixels that are either dead detector regions, calibration-flagged
    # bad pixels, or per-frame sigma-clipped transients. Used below to (a)
    # keep the centering fit and aperture photometry from being pulled by
    # interpolated values in the PSF core, and (b) drive the Phase 2 Moffat
    # repair of core bad pixels (IRDIS K1 in particular has 3-4 permanent bad
    # pixels at the flux-PSF landing spot).
    flux_ivar_path = os.path.join(converted_dir, 'flux_ivar_cube.fits')
    if os.path.exists(flux_ivar_path):
        flux_ivar_cube = fits.getdata(flux_ivar_path).astype('float64')
        flux_bpm_cube = flux_ivar_cube == 0
        logger.debug(
            f"Loaded flux_ivar_cube for BPM: {int(flux_bpm_cube.sum())} bad "
            f"pixels across the full cube"
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
    for frame_number in range(flux_cube.shape[1]):
        data = flux_cube[:, frame_number]
        cy, cx = guess_position_psf(
            cube=data,
            exclude_edge_pixels=30,
            mask_coronagraph_center=True,
            coronagraph_mask_x=126,
            coronagraph_mask_y=131,
            coronagraph_mask_radius=30
        )
        guess_positions_yx.append((cy, cx))
    
    # Replace unreliable first frame guess with second frame guess
    if len(guess_positions_yx) >= 2:
        logger.debug(f"Replacing unreliable first frame guess position {guess_positions_yx[0]} with second frame position {guess_positions_yx[1]}")
        guess_positions_yx[0] = guess_positions_yx[1]
    else:
        logger.warning("Less than 2 flux frames available, cannot replace first frame guess position")
    
    logger.debug(f"Computed guess positions for {len(guess_positions_yx)} frames")
    
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
            mask_coronagraph_center=True,
            coronagraph_mask_x=126,
            coronagraph_mask_y=131,
            coronagraph_mask_radius=30,
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
                mask_coronagraph_center=True,
                coronagraph_mask_x=126,
                coronagraph_mask_y=131,
                coronagraph_mask_radius=30,
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
            repair_records: list = []
        else:
            logger.info(
                f"Running Phase 2 Moffat repair on channels {channels_to_repair}",
                extra={"step": "flux_psf_calibration", "status": "phase2_start"},
            )
            flux_cube_repaired = flux_cube.copy()
            flux_ivar_repaired = flux_ivar_cube.copy()
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
                    )
                    repair_records.append({
                        "channel": ch,
                        "frame": f,
                        "status": res.status,
                        "n_repaired": res.n_repaired,
                        "residual_rms_frac": res.residual_rms_frac,
                        "amplitude": res.amplitude,
                    })
                    if res.status == "repaired":
                        flux_cube_repaired[ch, f, y0:y1, x0:x1] = res.window_out
                        flux_ivar_repaired[ch, f, y0:y1, x0:x1] = res.ivar_out
                    elif res.status == "skipped_bad_fit" and np.isfinite(res.residual_rms_frac):
                        logger.warning(
                            f"Channel {ch}, frame {f}: Moffat fit residual RMS = "
                            f"{100*res.residual_rms_frac:.2f}% of amplitude "
                            f"(> {int(100*_REPAIR_RESIDUAL_RMS_THRESHOLD)}%). "
                            f"Falling back to the upstream interpolated value at "
                            f"the bad-pixel locations.",
                            extra={
                                "step": "flux_psf_calibration",
                                "status": "phase2_fallback",
                                "channel": ch,
                                "frame": f,
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
        repair_records = []

    fits.writeto(additional_outputs_dir / 'flux_centers.fits', flux_centers, overwrite=True)
    fits.writeto(additional_outputs_dir / 'flux_gauss_amplitudes.fits', flux_amplitudes, overwrite=True)
    flux_stamps = toolbox.extract_satellite_spot_stamps(
        flux_cube_for_stamps, flux_centers, stamp_size=57, shift_order=3, plot=False)
    logger.debug(f"Extracted flux_stamps shape: {flux_stamps.shape}")
    fits.writeto(additional_outputs_dir / 'flux_stamps_uncalibrated.fits',
                 flux_stamps.astype('float32'), overwrite=True)
    # Sibling stamps from the raw (unrepaired) flux cube, so we can build a
    # matched unrepaired psf_cube for direct comparison.
    if flux_cube_repaired is not None:
        flux_stamps_unrepaired = toolbox.extract_satellite_spot_stamps(
            flux_cube, flux_centers, stamp_size=57, shift_order=3, plot=False)
    else:
        flux_stamps_unrepaired = None

    # Nearest-pixel BPM stamps aligned to the extracted flux_stamps. The stamp
    # extractor above uses sub-pixel shifts, but interpolating a boolean mask
    # is meaningless — we use the integer-rounded center to grab the underlying
    # detector BPM stamp, which is the right thing for aperture masking (bad
    # pixel = bad pixel, no partial credit).
    if flux_bpm_cube is not None:
        stamp_size = 57
        half = stamp_size // 2
        n_wave, n_frames, ny, nx = flux_bpm_cube.shape
        flux_bpm_stamps = np.zeros(
            (n_wave, n_frames, stamp_size, stamp_size), dtype=bool)
        for ch in range(n_wave):
            for f in range(n_frames):
                cx, cy = flux_centers[ch, f, 0]
                if not (np.isfinite(cx) and np.isfinite(cy)):
                    flux_bpm_stamps[ch, f] = True
                    continue
                cxi, cyi = int(round(float(cx))), int(round(float(cy)))
                y0, x0 = cyi - half, cxi - half
                y1, x1 = y0 + stamp_size, x0 + stamp_size
                if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
                    flux_bpm_stamps[ch, f] = True
                    continue
                flux_bpm_stamps[ch, f] = flux_bpm_cube[ch, f, y0:y1, x0:x1]
    else:
        flux_bpm_stamps = None
    if len(frames_info['FLUX']['INS4 FILT2 NAME'].unique()) > 1:
        logger.warning('Non-unique ND filters in sequence.', extra={"step": "flux_psf_calibration", "status": "failed"})
        raise ValueError('Non-unique ND filters in sequence.')
    else:
        ND = frames_info['FLUX']['INS4 FILT2 NAME'].unique()[0]
    _, attenuation = transmission.transmission_nd(ND, wave=wavelengths)
    fits.writeto(additional_outputs_dir / 'nd_attenuation.fits', attenuation, overwrite=True)
    dits_flux = np.array(frames_info['FLUX']['DET SEQ1 DIT'])
    dits_center = np.array(frames_info['CENTER']['DET SEQ1 DIT'])
    unique_dits_center, unique_dits_center_counts = np.unique(dits_center, return_counts=True)
    if len(unique_dits_center) == 1:
        dits_factor = unique_dits_center[0] / dits_flux
        most_common_dit_center = unique_dits_center[0]
    else:
        most_common_dit_center = unique_dits_center[np.argmax(unique_dits_center_counts)]
        dits_factor = most_common_dit_center / dits_flux
    dit_factor_center = most_common_dit_center / dits_center
    fits.writeto(additional_outputs_dir / 'center_frame_dit_adjustment_factors.fits',
                 dit_factor_center, overwrite=True)
    flux_stamps_calibrated = flux_stamps * dits_factor[None, :, None, None]
    flux_stamps_calibrated = flux_stamps_calibrated / attenuation[:, np.newaxis, np.newaxis, np.newaxis]
    fits.writeto(additional_outputs_dir / 'flux_stamps_dit_nd_calibrated.fits',
                 flux_stamps_calibrated, overwrite=True)
    flux_photometry = flux_calibration.get_aperture_photometry(
        flux_stamps_calibrated, aperture_radius_range=[1, 15],
        bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
        bad_pixel_mask=flux_bpm_stamps)
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
    flux_calibration_indices, indices_of_discontinuity = flux_calibration.get_flux_calibration_indices(
        frames_info['CENTER'], frames_info['FLUX'])
    flux_calibration_indices.to_csv(os.path.join(converted_dir, 'flux_calibration_indices.csv'))
    indices_of_discontinuity.tofile(additional_outputs_dir / 'indices_of_discontinuity.csv', sep=',')
    number_of_flux_frames = flux_stamps.shape[1]
    flux_calibration_frames = []
    if reduction_parameters['flux_combination_method'] == 'mean':
        comb_func = np.nanmean
    elif reduction_parameters['flux_combination_method'] == 'median':
        comb_func = np.nanmedian
    else:
        logger.warning('Unknown flux combination method.', extra={"step": "flux_psf_calibration", "status": "failed"})
        raise ValueError('Unknown flux combination method.')
    for idx in range(len(flux_calibration_indices)):
        try:
            upper_range = flux_calibration_indices['flux_idx'].iloc[idx+1]
        except IndexError:
            upper_range = number_of_flux_frames
        if idx == 0:
            lower_index = 0
            lower_index_frame_combine = 0
            number_of_frames_to_combine = upper_range - lower_index
            if reduction_parameters['exclude_first_flux_frame'] and number_of_frames_to_combine > 1:
                lower_index_frame_combine = 1
        else:
            lower_index = flux_calibration_indices['flux_idx'].iloc[idx]
            lower_index_frame_combine = 0
            number_of_frames_to_combine = upper_range - lower_index
            if reduction_parameters['exclude_first_flux_frame_all'] and number_of_frames_to_combine > 1:
                lower_index_frame_combine = 1
        phot_values = flux_photometry['psf_flux_bg_corr_all'][2][:, lower_index: upper_range]
        reference_value = np.nanmean(
            flux_photometry['psf_flux_bg_corr_all'][2][:, lower_index_frame_combine:upper_range], axis=1)
        normalization_values = phot_values / reference_value[:, None]
        flux_calibration_frame = bg_sub_flux_stamps_calibrated[:, lower_index:upper_range] / normalization_values[:, :, None, None]
        flux_calibration_frame = comb_func(flux_calibration_frame[:, lower_index_frame_combine:], axis=1)
        flux_calibration_frames.append(flux_calibration_frame)
    flux_calibration_frames = np.array(flux_calibration_frames)
    flux_calibration_frames = np.swapaxes(flux_calibration_frames, 0, 1)
    fits.writeto(os.path.join(converted_dir, 'psf_cube_for_postprocessing.fits'),
                 flux_calibration_frames.astype('float32'), overwrite=True)

    # Diagnostic sibling: replay the same DIT/ND + BG-sub + normalize + combine
    # pipeline on the *raw* (unrepaired) stamps, so the only difference vs the
    # canonical psf_cube_for_postprocessing.fits is Moffat repair vs the
    # upstream neighbour-interpolated bad-pixel values. Only produced when
    # Phase 2 actually ran.
    if flux_stamps_unrepaired is not None:
        flux_stamps_calibrated_unrepaired = (
            flux_stamps_unrepaired * dits_factor[None, :, None, None]
            / attenuation[:, np.newaxis, np.newaxis, np.newaxis]
        )
        flux_photometry_unrepaired = flux_calibration.get_aperture_photometry(
            flux_stamps_calibrated_unrepaired, aperture_radius_range=[1, 15],
            bg_aperture_inner_radius=15, bg_aperture_outer_radius=18,
            bad_pixel_mask=flux_bpm_stamps)
        bg_sub_stamps_unrepaired = (
            flux_stamps_calibrated_unrepaired
            - flux_photometry_unrepaired['psf_bg_counts_all'][:, :, None, None]
        )
        unrepaired_frames = []
        for idx in range(len(flux_calibration_indices)):
            try:
                upper_range = flux_calibration_indices['flux_idx'].iloc[idx + 1]
            except IndexError:
                upper_range = number_of_flux_frames
            if idx == 0:
                lower_index = 0
                lower_index_frame_combine = 0
                n_combine = upper_range - lower_index
                if reduction_parameters['exclude_first_flux_frame'] and n_combine > 1:
                    lower_index_frame_combine = 1
            else:
                lower_index = flux_calibration_indices['flux_idx'].iloc[idx]
                lower_index_frame_combine = 0
                n_combine = upper_range - lower_index
                if reduction_parameters['exclude_first_flux_frame_all'] and n_combine > 1:
                    lower_index_frame_combine = 1
            phot_u = flux_photometry_unrepaired['psf_flux_bg_corr_all'][2][:, lower_index:upper_range]
            ref_u = np.nanmean(
                flux_photometry_unrepaired['psf_flux_bg_corr_all'][2][:, lower_index_frame_combine:upper_range],
                axis=1)
            norm_u = phot_u / ref_u[:, None]
            frame_u = bg_sub_stamps_unrepaired[:, lower_index:upper_range] / norm_u[:, :, None, None]
            frame_u = comb_func(frame_u[:, lower_index_frame_combine:], axis=1)
            unrepaired_frames.append(frame_u)
        unrepaired_cube = np.swapaxes(np.array(unrepaired_frames), 0, 1)
        fits.writeto(os.path.join(converted_dir, 'psf_cube_for_postprocessing_unrepaired.fits'),
                     unrepaired_cube.astype('float32'), overwrite=True)
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
